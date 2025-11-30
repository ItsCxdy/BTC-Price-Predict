import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# CONFIGURATION
# 7 days is the max limit for 1m data on yfinance free tier
DATA_PERIOD = "7d"  
INTERVAL = "1m"
TICKER = "BTC-USD"
SEQ_LENGTH = 60  # Look back at the last 60 minutes to predict the next 1

def fetch_and_process_data():
    """
    Fetches data, normalizes it, and creates sequences for the AI.
    """
    print(f"📥 Fetching {DATA_PERIOD} of {INTERVAL} data for {TICKER}...")
    
    # Fetch data
    df = yf.download(TICKER, period=DATA_PERIOD, interval=INTERVAL)
    
    if len(df) < SEQ_LENGTH:
        raise ValueError("Not enough data fetched to create sequences.")
    
    print(f"✅ Data fetched: {len(df)} candles.")

    # We use 'Close' price. 
    # TIP: To make the model 'reason' about volatility, we could add 'Volume' here later.
    data = df['Close'].values.reshape(-1, 1)

    # NORMALIZE: Neural networks work best with numbers between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # CREATE SEQUENCES (The "Textbook")
    # X = Input (e.g., Minutes 1-60)
    # y = Output (e.g., Minute 61)
    X, y = [], []
    for i in range(len(scaled_data) - SEQ_LENGTH):
        X.append(scaled_data[i:i+SEQ_LENGTH])
        y.append(scaled_data[i+SEQ_LENGTH])

    X = np.array(X)
    y = np.array(y)

    # Split into Training (80%) and Testing (20%)
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    print(f"📊 Training Data Shape: {X_train.shape}")
    print(f"📊 Testing Data Shape: {X_test.shape}")
    
    return X_train, y_train, X_test, y_test, scaler

if __name__ == "__main__":
    fetch_and_process_data()