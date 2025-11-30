import pandas as pd
import numpy as np
import tensorflow as tf
import os
import glob
import ccxt 
from sklearn.preprocessing import MinMaxScaler
import time 

# --- Configuration Constants (Must match train.py) ---
LOOKBACK_TIMESTEPS = 60    # Past 60 minutes (1-hour sequence)
PREDICTION_HORIZON = 5     # Next 5 minutes to predict
NUM_FEATURES = 2           # Features used: ['Close', 'RSI']
RSI_PERIOD = 14            # Lookback period for RSI calculation
SYMBOL = 'BTC/USDT'        # Trading pair
TIMEFRAME = '1m'           # 1-minute candles
EXCHANGE_ID = 'binance'

# Storage Path for training data (to re-fit scaler)
TRAINING_DATA_PATH = 'data/raw_btc_data.csv'
MODELS_DIR = 'models/'

# --- Helper Functions (Ensuring Feature and Scaling Consistency) ---

def calculate_optimized_rsi(df, period):
    """Calculates the Relative Strength Index (RSI) using vectorized Pandas."""
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = delta.where(delta < 0, 0).abs()
    
    # Use EWM for Wilder smoothing equivalent
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    df['RSI'] = rsi
    return df

def get_fitted_scaler(file_path):
    """
    Loads the original training data and fits a new scaler instance.
    This is necessary to scale new data consistently with the model's training range.
    """
    if not os.path.exists(file_path):
        print(f"❌ Error: Training data not found at {file_path}. Cannot fit scaler for consistent normalization.")
        return None

    df_train = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
    df_train = calculate_optimized_rsi(df_train, RSI_PERIOD)
    
    # Use only the features used for training: Close and RSI
    features = ['Close', 'RSI']
    data_train = df_train[features].dropna().values
        
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data_train)
    print(f"✅ Scaler fitted using {len(data_train)} training samples.")
    return scaler

def get_live_data_sequence(scaler):
    """
    Fetches the latest live data, calculates RSI, extracts the sequence, and scales it.
    """
    exchange_class = getattr(ccxt, EXCHANGE_ID)
    exchange = exchange_class({'enableRateLimit': True})
    
    # Fetch enough data to cover the lookback period plus RSI calculation buffer
    fetch_limit = LOOKBACK_TIMESTEPS + RSI_PERIOD + 10 
    
    print(f"\n📥 Fetching {fetch_limit} recent {TIMEFRAME} candles for {SYMBOL}...")
    try:
        ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=fetch_limit)
    except Exception as e:
        print(f"❌ Error fetching live data: {e}")
        return None, None, None

    df_live = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df_live.set_index(pd.to_datetime(df_live['timestamp'], unit='ms'), inplace=True)
    
    # 1. Feature Engineering (RSI)
    df_live = calculate_optimized_rsi(df_live, RSI_PERIOD)
    
    # 2. Extract the required sequence (Close and RSI)
    features = ['Close', 'RSI']
    live_data_full = df_live[features].dropna().values
    
    if len(live_data_full) < LOOKBACK_TIMESTEPS:
        print(f"❌ Could only process {len(live_data_full)} minutes after cleanup. Need {LOOKBACK_TIMESTEPS}.")
        return None, None, None
        
    # Get the latest lookback sequence
    live_sequence = live_data_full[-LOOKBACK_TIMESTEPS:]
    
    # Store the actual last price (unscaled) for reference
    last_actual_price = live_sequence[-1, 0] # Close price is the 0th feature
    
    # 3. Normalize the live data using the fitted scaler
    input_normalized = scaler.transform(live_sequence)
    
    # Prepare input shape for the model: (1, LOOKBACK_TIMESTEPS, NUM_FEATURES)
    input_for_model = np.expand_dims(input_normalized, axis=0)
    
    return input_for_model, last_actual_price, live_sequence

def find_all_models(directory):
    """Dynamically finds all .h5 model files in the specified directory."""
    model_paths = glob.glob(os.path.join(directory, '*.h5'))
    
    if not model_paths:
        print(f"❌ No .h5 models found in the '{directory}' directory.")
    else:
        print(f"✅ Found {len(model_paths)} model(s) to compare.")
    return model_paths

def make_prediction(model_path, input_for_model, scaler, last_price):
    """Loads a single model, makes a prediction, and denormalizes the result."""
    model_name = os.path.basename(model_path)
    print(f"  -> Loading and predicting with: {model_name}...")
    
    try:
        # Load the model (Suppressing the Keras/HDF5 warning)
        # Removed problematic tf.get_logger().disable_tf_serving() call
        model = tf.keras.models.load_model(model_path, compile=False) 
    except Exception as e:
        print(f"  ❌ Failed to load model {model_name}. Error: {e}")
        return None

    # Make the Prediction
    prediction_normalized = model.predict(input_for_model, verbose=0)

    # Prediction is a sequence (1, 5, 2). We extract the inner sequence (5, 2)
    predicted_sequence_normalized = prediction_normalized[0] 
        
    # Inverse Transform (De-normalize) the prediction
    predicted_sequence_unscaled = scaler.inverse_transform(predicted_sequence_normalized)
    
    # The first predicted Close price (t+1) is at index [0, 0] (Close price is the first feature)
    predicted_price_t_plus_1 = predicted_sequence_unscaled[0, 0]
    
    # Calculate difference
    diff = predicted_price_t_plus_1 - last_price
    direction = "UP" if diff > 0 else "DOWN"
    
    return {
        'model': model_name,
        'predicted_price': predicted_price_t_plus_1,
        'last_price': last_price,
        'diff': diff,
        'direction': direction,
        'forecast_sequence': predicted_sequence_unscaled # Full 5-step forecast
    }

def predict_multiple_models():
    """Main pipeline to load all models, predict, and compare results."""
    
    # 1. Find all models
    model_paths = find_all_models(MODELS_DIR)
    if not model_paths:
        return
        
    # 2. Get the fitted scaler (crucial for consistent normalization)
    scaler = get_fitted_scaler(TRAINING_DATA_PATH)
    if scaler is None:
        return
        
    # 3. Get the live data sequence (normalized input) and the last known price
    input_for_model, last_actual_price, live_sequence = get_live_data_sequence(scaler)
    if input_for_model is None:
        return

    print("\n--- Starting Model Comparison Predictions ---")
    all_results = []
    
    # 4. Iterate and predict with each model
    for path in model_paths:
        result = make_prediction(path, input_for_model, scaler, last_actual_price)
        if result:
            all_results.append(result)

    # 5. Display Comparative Results
    print("\n==============================================")
    print("        COMPREHENSIVE BTC PRICE FORECAST      ")
    print("==============================================")
    print(f"Current Time: {pd.to_datetime(time.time(), unit='s')}")
    print(f"Last Actual BTC Close Price (t-0): ${last_actual_price:,.2f}")
    print("----------------------------------------------")
    
    if not all_results:
        print("No successful predictions to display.")
        return
        
    # Sort results by predicted price for easy comparison
    all_results.sort(key=lambda x: x['predicted_price'])
    
    for result in all_results:
        # Extract only the Close price predictions for the 5 steps
        forecast_close = [f"${p[0]:,.2f}" for p in result['forecast_sequence']]
        
        print(f"MODEL: {result['model']:<25}")
        print(f"  Predicted Price (t+1): ${result['predicted_price']:,.2f}")
        print(f"  Forecasted Move: {result['direction']:<4} by ${abs(result['diff']):.2f}")
        print(f"  Full 5-Min Close Price Forecast: {forecast_close}")
        print("----------------------------------------------")
        
    # Determine the consensus/average prediction
    avg_price = np.mean([r['predicted_price'] for r in all_results])
    overall_diff = avg_price - last_actual_price
    overall_direction = "UP" if overall_diff > 0 else "DOWN"
    
    print("--- CONSENSUS / ENSEMBLE VIEW ---")
    print(f"Average Predicted Price (t+1): ${avg_price:,.2f}")
    print(f"Overall Direction: {overall_direction} by ${abs(overall_diff):.2f}")
    print("==============================================")

if __name__ == "__main__":
    # Set to ERROR to suppress standard TensorFlow warnings on startup
    tf.get_logger().setLevel('ERROR') 
    predict_multiple_models()