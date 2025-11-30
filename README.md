BTC-Price-Predict: Time Series Forecasting using Bi-Directional LSTMs
This repository contains a sophisticated time series forecasting model designed to predict the price of Bitcoin (BTC/USDT) using a stacked Bidirectional Long Short-Term Memory (BiLSTM) neural network architecture.
The system features dynamic data loading from Binance, real-time feature engineering (RSI), and an ensemble prediction script that compares forecasts from multiple trained models.
🚀 Key Features
Bidirectional LSTM (BiLSTM): Uses two stacked BiLSTM layers to process sequences forward and backward, capturing comprehensive dependencies in the time series data.
Feature Engineering: Incorporates Relative Strength Index (RSI) alongside the Close price to enhance predictive power.
Dynamic Data Loading: Fetches live and historical 1-minute (1m) candlestick data using the ccxt library.
Ensemble Prediction: The predict.py script automatically loads and compares predictions from multiple .h5 models found in the models/ directory, providing a robust, comparative forecast.
Time Series Sequence Generation: Uses a 60-minute lookback window (LOOKBACK_TIMESTEPS = 60) to predict the next 5 minutes (PREDICTION_HORIZON = 5).
Reproducibility: Dependencies are frozen in requirements.txt.
📂 Project Structure
File/Folder
Description
train.py
Main script to load data, build the BiLSTM model, and train it. Saves the final model to models/.
predict.py
Loads multiple trained models, fetches the latest live data, makes comparative predictions, and reports an ensemble view.
data_loader.py
Contains helper functions for data fetching, caching, preprocessing, and sequence generation (X, Y matrices).
model_builder.py
Defines the build_bilstm_model function, setting up the stacked BiLSTM architecture, including Batch Normalization and Dropout.
rolling_backtest_strategy.py
Script for performing walk-forward validation (likely used to generate graphs).
sequential_prediction_viz.py
Script used for visualizing the prediction results against actual data.
requirements.txt
Frozen list of Python dependencies required to run the project.
data/
Directory where the raw cached data (raw_btc_data.csv) is stored. Ignored by Git.
models/
Directory where trained .h5 model files (e.g., cat2_trained_model.h5) are saved. Ignored by Git.
Graphs/
Directory for storing visualization outputs (e.g., Figure_1.png).

🛠️ Setup and Installation
1. Prerequisites
Python (3.9+)
Git
2. Clone the Repository
git clone [https://github.com/ItsCxdy/BTC-Price-Predict.git](https://github.com/ItsCxdy/BTC-Price-Predict.git)
cd BTC-Price-Predict


3. Setup Virtual Environment (Recommended)
python -m venv venv
# Activate the environment (Windows)
.\venv\Scripts\activate
# Activate the environment (Linux/macOS)
source venv/bin/activate


4. Install Dependencies
Install all necessary libraries using the frozen list in requirements.txt:
pip install -r requirements.txt


🏃 Usage
1. Training the Model (train.py)
This script handles data fetching, preprocessing, model building, and training over 20 epochs.
python train.py


Output Example:
The script will download data, process it, show the model summary, and begin training, saving the resulting model to models/cat2_trained_model.h5.
2. Making Ensemble Predictions (predict.py)
This script is designed to load all trained models in the models/ folder, fetch the latest live data, and provide a comparative forecast.
Prerequisite: You must have at least one trained model file (.h5) in the models/ directory.
python predict.py


Output Example:
The output will show the last actual BTC price, followed by individual predictions from each loaded model, and a final "Consensus / Ensemble View."
==============================================
        COMPREHENSIVE BTC PRICE FORECAST      
==============================================
Last Actual BTC Close Price (t-0): $91,379.15
----------------------------------------------
MODEL: 1cat2_trained_model.h5        
  Predicted Price (t+1): $91,405.50
  Forecasted Move: UP   by $26.35
  Full 5-Min Close Price Forecast: ['$91,405.50', '$91,410.12', ...]
----------------------------------------------
MODEL: cat2_trained_model.h5          
  Predicted Price (t+1): $91,398.22
  Forecasted Move: UP   by $19.07
  Full 5-Min Close Price Forecast: ['$91,398.22', '$91,401.40', ...]
----------------------------------------------
--- CONSENSUS / ENSEMBLE VIEW ---
Average Predicted Price (t+1): $91,401.86
Overall Direction: UP by $22.71
==============================================


⚙️ Model Architecture Details
The model is defined in model_builder.py and uses the following structure:
Input Layer: Expects a shape of (60, 2) (60 time steps, 2 features: Close Price and RSI).
Stacked BiLSTMs: Two layers of Bidirectional(LSTM(...)) are used to capture complex sequence relationships.
Regularization: BatchNormalization and Dropout layers are strategically placed between BiLSTMs to prevent overfitting and stabilize training.
Dense Layers: A final series of Dense layers collapses the sequential output into a vector.
Output Layer: A Dense layer with 10 neurons, followed by a Reshape layer to output the final prediction sequence of shape (5, 2) (5 predicted time steps, 2 features: Close Price and RSI).
