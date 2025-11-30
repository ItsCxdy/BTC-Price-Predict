import pandas as pd
import numpy as np
import time
import tensorflow as tf
import os 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout, BatchNormalization, Reshape
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import ccxt 

# --- 1. CONFIGURATION CONSTANTS ---
# Model Input/Output Configuration (Matching CAT 2 Specification)
LOOKBACK_TIMESTEPS = 60    # Past 60 minutes (1-hour sequence) for input (X)
PREDICTION_HORIZON = 5     # Next 5 minutes to predict (Y)
NUM_FEATURES = 2           # Features used: ['Close', 'RSI']
RSI_PERIOD = 14            # Lookback period for RSI calculation

# Training Configuration
DATA_POINTS = 500000       # INCREASED: Targeting 500,000 1-minute candles (~350 days)
EXCHANGE_ID = 'binance'    # Target exchange for data
SYMBOL = 'BTC/USDT'        # Trading pair
TIMEFRAME = '1m'           # 1-minute candles (CAT 2 specification)
TEST_SIZE = 0.2            # 20% of data for validation
EPOCHS = 20                 # Increased epochs for fine-tuning
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# Define storage paths
DATA_PATH = 'data/raw_btc_data.csv'
MODEL_PATH = 'models/cat2_trained_model.h5'

# --- 2. DATA ACQUISITION AND FEATURE ENGINEERING ---

def load_or_download_data(exchange_id, symbol, timeframe, limit, file_path):
    """
    Checks if data exists locally. If yes, loads it. If no, downloads it via ccxt 
    and saves it to the specified file path.
    """
    # 1. Check for local file
    if os.path.exists(file_path):
        print(f"Loading data from local file: {file_path}")
        try:
            df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
            # Ensure the required columns are present after loading
            if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
                print(f"Successfully loaded {len(df)} data points from disk.")
                return df
            else:
                print("Local file is corrupted or missing necessary columns. Attempting download.")
        except Exception as e:
            print(f"Error loading local file ({e}). Attempting download.")
    
    # 2. Download if local file doesn't exist or is corrupted
    print(f"Local file not found or invalid. Downloading {limit} of {symbol} {timeframe} data from {exchange_id}...")
    
    try:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({'enableRateLimit': True})

        # Fetch the historical data (OHLCV format: [timestamp, open, high, low, close, volume])
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        if not ohlcv:
            print("ERROR: Download failed or returned empty data.")
            return pd.DataFrame()

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        print(f"Successfully downloaded {len(df)} data points.")
        
        # Save the raw data to the data folder
        df.to_csv(file_path)
        print(f"Raw data saved to: {file_path}")
        
        return df
        
    except AttributeError:
        print(f"ERROR: Exchange ID '{exchange_id}' not found in ccxt.")
        return pd.DataFrame()
    except Exception as e:
        print(f"ERROR: Failed to fetch data using ccxt. Exception: {e}")
        print("Please ensure ccxt is installed and you have an active internet connection.")
        return pd.DataFrame()

def calculate_optimized_rsi(df, period):
    """Calculates the Relative Strength Index (RSI) using vectorized Pandas."""
    start_time = time.time()
    
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = delta.where(delta < 0, 0).abs()

    # Use EWM (Exponential Weighted Moving Average) for Wilder smoothing equivalent
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    df['RSI'] = rsi
    
    end_time = time.time()
    print(f"RSI calculation complete. Time taken: {end_time - start_time:.4f} seconds.")
    return df

# --- 3. SEQUENTIAL DATA PREPARATION ---

def create_sequential_datasets(df, lookback_timesteps, prediction_horizon, features):
    """
    Transforms the DataFrame into sequences suitable for Keras sequential models.
    X (Input): (samples, lookback_timesteps, features)
    Y (Target): (samples, prediction_horizon, target_features)
    """
    
    # Drop NaNs created by RSI calculation
    df_clean = df[features].dropna()
    
    if len(df_clean) < lookback_timesteps + prediction_horizon:
        print("ERROR: Not enough data points remaining after cleanup for sequence creation.")
        return None, None, None
        
    data = df_clean.values
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, Y = [], []
    total_data_points = len(scaled_data)
    
    # Define the index where the last possible sequence can start
    end_index = total_data_points - lookback_timesteps - prediction_horizon + 1
    
    for i in range(end_index):
        # X: Past sequence (Lookback)
        x_sequence = scaled_data[i : i + lookback_timesteps]
        X.append(x_sequence)
        
        # Y: Future sequence (Prediction Horizon)
        y_sequence = scaled_data[i + lookback_timesteps : i + lookback_timesteps + prediction_horizon]
        Y.append(y_sequence)
        
    X = np.array(X)
    Y = np.array(Y)
    
    print(f"X shape (Input Sequences): {X.shape}") 
    print(f"Y shape (Target Sequences): {Y.shape}")
    
    return X, Y, scaler

# --- 4. CAT 2 MODEL ARCHITECTURE ---

def create_cat2_model(timesteps, features, prediction_horizon, learning_rate):
    """Creates the CAT 2 GPT-style sequential prediction model (Stacked Bi-LSTM)."""
    
    model = Sequential()
    
    # 1. First Bidirectional LSTM Layer
    model.add(Bidirectional(
        LSTM(units=128, return_sequences=True), 
        input_shape=(timesteps, features)
    ))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # 2. Second Bidirectional LSTM Layer
    model.add(Bidirectional(
        LSTM(units=128, return_sequences=True)
    ))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # 3. Third LSTM Layer (Context Vector Generation)
    model.add(LSTM(units=64, return_sequences=False))
    model.add(Dropout(0.2))

    # 4. Dense Layers (Prediction Head)
    model.add(Dense(units=512, activation='relu')) 
    model.add(Dense(units=256, activation='relu')) 

    # 5. Output Layer (Flattened Sequence Prediction)
    output_neurons = prediction_horizon * features
    model.add(Dense(units=output_neurons, activation='linear'))
    
    # 6. Reshape Layer (Crucial: Flattens output back into sequence (5, 2))
    model.add(Reshape((prediction_horizon, features)))

    # Compile the Model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    
    return model

# --- 5. MAIN TRAINING FUNCTION ---

def run_training_pipeline():
    """Runs the complete data processing, model building, and training pipeline."""
    
    # Ensure necessary directories exist
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    # Step 1: Data Acquisition (Load or Download)
    df = load_or_download_data(EXCHANGE_ID, SYMBOL, TIMEFRAME, DATA_POINTS, DATA_PATH)
    
    if df.empty:
        print("Training cannot proceed due to data failure.")
        return

    # Step 2: Feature Engineering
    df = calculate_optimized_rsi(df, RSI_PERIOD)
    
    # Step 3: Sequential Data Preparation
    X, Y, scaler = create_sequential_datasets(
        df, 
        LOOKBACK_TIMESTEPS, 
        PREDICTION_HORIZON, 
        features=['Close', 'RSI']
    )
    
    if X is None:
        return
        
    # Step 4: Train/Test Split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=TEST_SIZE, shuffle=False
    )
    
    print(f"\nTraining set size: {len(X_train)} samples")
    print(f"Validation set size: {len(X_test)} samples")
    
    # Step 5: Model Creation
    cat2_model = create_cat2_model(
        LOOKBACK_TIMESTEPS, 
        NUM_FEATURES, 
        PREDICTION_HORIZON, 
        LEARNING_RATE
    )
    
    print("\n--- CAT 2 Model Summary ---")
    cat2_model.summary()
    
    total_params = cat2_model.count_params()
    print(f"Total Trainable Parameters (Neurons): {total_params}") 
    if total_params < 200000:
        print("⚠️ WARNING: Capacity is less than 200,000. Consider increasing layer sizes.")

    # Step 6: Training the Model
    print("\n--- Starting Model Training ---")
    
    try:
        start_time = time.time()
        history = cat2_model.fit(
            X_train, 
            Y_train, 
            epochs=EPOCHS, 
            batch_size=BATCH_SIZE, 
            validation_data=(X_test, Y_test),
            verbose=1,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True)
            ]
        )
        end_time = time.time()
        
        print(f"\nTraining complete in {end_time - start_time:.2f} seconds.")
        print(f"Final Validation Loss (MSE): {history.history['val_loss'][-1]:.6f}")
        
        # Save the trained model to the models folder
        cat2_model.save(MODEL_PATH)
        print(f"Model saved to: {MODEL_PATH}")
        
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")

if __name__ == '__main__':
    # Increase TensorFlow logging verbosity slightly for better output
    tf.get_logger().setLevel('INFO') 
    run_training_pipeline()