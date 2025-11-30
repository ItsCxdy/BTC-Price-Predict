import numpy as np
import pandas as pd
# UPDATE: Import RandomForestRegressor instead of LinearRegression
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import yfinance as yf 
from datetime import datetime, timedelta 
# FIX: Import the necessary model for the updated algorithm
from sklearn.linear_model import LinearRegression # Keep for plotting title clarity if needed, but we'll use RF

# --- Configuration for Real Data ---
BTC_TICKER = 'BTC-USD'

# Dynamically set date range to be a recent 1-year window for 1h interval.
END_DATE = datetime.now().strftime('%Y-%m-%d') # Sets the end date to today
START_DATE = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d') # Sets the start date to 1 year ago

# The interval is changed to '1h' (1 hour) for a higher time frame. 
INTERVAL = '1h' 
# ---

# --- 1. Real Data Loading and Feature Engineering ---

# Helper function to calculate RSI
def calculate_rsi(series, window=14):
    """Calculates the Relative Strength Index (RSI)."""
    # Calculate difference between current and previous close
    delta = series.diff()
    # Calculate gain (positive changes) and loss (negative changes)
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    # Exponential Moving Average (EMA) for gain and loss
    avg_gain = gain.ewm(com=window - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=window - 1, adjust=False).mean()

    # Calculate Relative Strength (RS)
    rs = avg_gain / avg_loss
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    return rsi

def load_real_btc_data(ticker, start, end, interval, lookahead_steps=24):
    """
    Downloads real Bitcoin data and engineers features for prediction.
    UPDATE: Added RSI feature engineering.
    """
    print(f"Downloading {ticker} data from {start} to {end} with {interval} interval...")
    try:
        data = yf.download(ticker, start=start, end=end, interval=interval)
    except Exception as e:
        print(f"Error downloading data: {e}")
        return pd.DataFrame() 

    if data.empty:
        print("Data is empty. Check dates and ticker.")
        return pd.DataFrame()
    
    # Use the Close price as the primary variable
    df = data[['Close']].copy()
    df.columns = ['y_actual']
    
    # 1. Target Variable (y): The price 'lookahead_steps' into the future
    df['y_target'] = df['y_actual'].shift(-lookahead_steps)
    
    # 2. Feature 1: Short Moving Average (SMA_5) - 5 hours
    df['SMA_5'] = df['y_actual'].rolling(window=5).mean().shift(1)
    
    # 3. Feature 2: Medium Moving Average (SMA_10) - 10 hours
    df['SMA_10'] = df['y_actual'].rolling(window=10).mean().shift(1)
    
    # 4. Feature 3: Relative Strength Index (RSI_14)
    # The RSI at time t is based on prices up to t. We use .shift(1) to ensure the feature is from the previous period.
    df['RSI_14'] = calculate_rsi(df['y_actual'], window=14).shift(1)

    # 5. Feature 4: Recent Price
    df['Recent_Price'] = df['y_actual'].shift(1)

    # Final cleanup: Drop all rows where the target or features are NaN
    # RSI requires 14+ periods, SMA_10 requires 10 periods, and lookahead_steps removes the end rows.
    df.dropna(inplace=True)
    
    # Rename for simplicity in the modeling functions
    df.rename(columns={'y_actual': 'y_current'}, inplace=True) 

    return df.reset_index()

# --- 2. Walk-Forward Backtesting Function (UPDATED MODEL) ---
def walk_forward_backtest(df, train_window_size=2160, time_unit='hour'):
    """
    Performs a Walk-Forward backtest, predicting one step ahead where that step
    is pre-defined by the lookahead in the data loading function.
    UPDATE: Switched from LinearRegression to RandomForestRegressor.
    """
    
    # Since the target 'y_target' is pre-shifted, we are always predicting the next available row (step=1)
    prediction_step = 1 
    total_predictions = len(df) - train_window_size - prediction_step + 1
    
    results = []
    
    # --- Define features and target based on new DataFrame structure ---
    # UPDATE: Added RSI_14 to the features
    FEATURE_COLUMNS = ['Recent_Price', 'SMA_5', 'SMA_10', 'RSI_14']
    TARGET_COLUMN = 'y_target'

    # The loop runs for every single step-ahead prediction
    for i in range(total_predictions):
        # 1. Define Training Data (Sliding Window)
        train_start_index = i
        train_end_index = i + train_window_size
        
        # 2. Define Test Data (The Single Next Step to be predicted)
        test_start_index = train_end_index
        test_end_index = train_end_index + prediction_step
        
        if test_end_index > len(df):
            break

        # --- Data Preparation for the current step ---
        # Training Set
        train_df = df.iloc[train_start_index:train_end_index]
        X_train = train_df[FEATURE_COLUMNS].values
        y_train = train_df[TARGET_COLUMN].values

        # Test Set (The single next point)
        test_df = df.iloc[test_start_index:test_end_index]
        X_test = test_df[FEATURE_COLUMNS].values
        y_test = test_df[TARGET_COLUMN].values
        
        # --- Model Training and Prediction ---
        # UPDATE: Using Random Forest Regressor
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        # --- Evaluation ---
        mse = mean_squared_error(y_test, y_pred)
        
        # Record the result for this single step
        results.append({
            'Prediction_Time': test_df['Datetime'].iloc[0],
            'Actual_Y_Target': y_test[0], # The price 24 hours in the future
            'Predicted_Y_Target': y_pred[0], # The prediction for 24 hours in the future
            'MSE_Step': mse
        })

    # --- Summarize Results ---
    results_df = pd.DataFrame(results)
    
    return results_df

# --- 3. Visualization ---
def plot_results(results_df, lookback_steps=500, lookahead_steps=24):
    """Plots the actual vs. predicted prices for a specified number of recent steps."""
    
    # Select the last 'lookback_steps' for a clear visualization
    plot_df = results_df.tail(lookback_steps).copy()
    
    plt.figure(figsize=(14, 7))
    
    # Plot Actual Prices
    plt.plot(plot_df['Prediction_Time'], plot_df['Actual_Y_Target'], 
             label=f'Actual BTC Price (t+{lookahead_steps}h)', color='blue', linewidth=2)
    
    # Plot Predicted Prices
    plt.plot(plot_df['Prediction_Time'], plot_df['Predicted_Y_Target'], 
             label=f'Predicted BTC Price (t+{lookahead_steps}h)', color='red', linestyle='--', linewidth=1.5)
    
    # UPDATE title to reflect the new model
    plt.title(f'Walk-Forward BTC Price Prediction (Random Forest | Last {lookback_steps} Steps | {lookahead_steps} Hour Lookahead)')
    plt.xlabel('Prediction Time (t)')
    plt.ylabel(f'{BTC_TICKER} Price (USD)') 
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Improve x-axis labels readability
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # Calculate RMSE for the plotted period for better context
    mse_period = mean_squared_error(plot_df['Actual_Y_Target'], plot_df['Predicted_Y_Target'])
    rmse_period = np.sqrt(mse_period)
    print(f"\n--- Visualization Summary (Last {lookback_steps} Steps) ---")
    print(f"MSE for plotted period: {mse_period:.2f}")
    print(f"RMSE for plotted period: {rmse_period:.2f}")

# --- 4. Execution ---
if __name__ == "__main__":
    
    # Configuration for Backtest (Adjusted for 1h interval)
    TRAIN_WINDOW_SIZE = 2160 # Approx 90 days of hourly data for training
    LOOKAHEAD_STEPS = 24     # Predict 24 steps (1 day) ahead
    
    # Load Real BTC Data
    data = load_real_btc_data(
        BTC_TICKER, 
        START_DATE, 
        END_DATE, 
        INTERVAL,
        lookahead_steps=LOOKAHEAD_STEPS
    )

    if data.empty:
        print("\nCannot run backtest due to empty data.")
    else:
        print("\n--- Data Loaded (First 5 Rows) ---")
        print(data.head())
        
        # Run the walk-forward backtest
        walk_forward_results = walk_forward_backtest(
            data, 
            train_window_size=TRAIN_WINDOW_SIZE
        )
        
        # Print Summary
        total_predictions = len(walk_forward_results)
        avg_mse = walk_forward_results['MSE_Step'].mean()

        print(f"\n--- Walk-Forward Summary ({LOOKAHEAD_STEPS}-Hour BTC Prediction) ---")
        print(f"Model: Random Forest Regressor") # Added model name
        print(f"Total Steps Predicted: {total_predictions}")
        print(f"Average MSE across all steps: {avg_mse:.2f}")
        print(f"Average RMSE: {np.sqrt(avg_mse):.2f} USD")
        
        print("\n--- Detailed Walk-Forward Results (First 10 Steps) ---")
        print(walk_forward_results[[
            'Prediction_Time', 'Actual_Y_Target', 'Predicted_Y_Target', 'MSE_Step'
        ]].head(10))

        # Identify the steps with the largest errors
        worst_steps = walk_forward_results.sort_values(by='MSE_Step', ascending=False).head(5)
        print("\n--- 5 Steps with Highest Error (Worst Predictions) ---")
        print(worst_steps[['Prediction_Time', 'MSE_Step']])
        
        # Add the visualization step
        plot_results(walk_forward_results, lookback_steps=500, lookahead_steps=LOOKAHEAD_STEPS)

        print("\n--- Suggested Next Steps ---")
        print("We have upgraded to a Random Forest Regressor and added the Relative Strength Index (RSI) as a new feature.")
        print("Run the script again to see if the RMSE has improved.")
        print("Further improvements could include adding more technical indicators (e.g., MACD, Bollinger Bands) or hyperparameter tuning the Random Forest model.")