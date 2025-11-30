import tensorflow as tf
import yfinance as yf
import pandas as pd

print("✅ TensorFlow Version:", tf.__version__)
print("✅ Pandas Version:", pd.__version__)

# Test Data Fetch
print("\nFetching live BTC Data...")
btc = yf.Ticker("BTC-USD")
history = btc.history(period="1d", interval="1m")

if not history.empty:
    print(f"✅ Success! Downloaded {len(history)} minutes of Bitcoin data.")
    print(f"   Current Price: ${history.iloc[-1]['Close']:.2f}")
else:
    print("❌ Error fetching data.")