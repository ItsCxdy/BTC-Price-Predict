import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Configuration for CAT 2 Simulation ---
TOTAL_STEPS = 500              # Total minutes of market activity to simulate
PREDICTION_HORIZON = 60        # M: How many steps (minutes) ahead the model predicts in one sequence (1 hour)
PREDICTION_INTERVAL = 30       # N: How often the model re-predicts (e.g., every 30 minutes)
BASE_PRICE = 10000             # Starting price for the simulation (e.g., $10,000)

# --- 1. Data Simulation ---

def simulate_data(steps):
    """Generates the actual, continuous price path."""
    np.random.seed(42)
    # Simulate a path with slight upward drift and noise
    price_changes = np.random.normal(0, 0.5, steps) + np.linspace(0.0, 0.1, steps)
    actual_prices = BASE_PRICE + np.cumsum(price_changes)
    
    # Create time index
    time_index = pd.to_datetime('2024-01-01 09:00:00') + pd.to_timedelta(np.arange(steps), unit='min')
    
    return pd.Series(actual_prices, index=time_index, name='Actual_Price')

def simulate_predictions(actual_prices, horizon, interval):
    """
    Simulates the CAT 2 model's sequential output.
    At time T, it generates a sequence of T+1 to T+M predicted prices.
    """
    predictions = {}
    actual_data = actual_prices.values
    
    # Iterate through the time steps, starting a new prediction every 'interval' steps
    for i in range(0, TOTAL_STEPS - horizon, interval):
        time_T = actual_prices.index[i]
        
        # 1. Start the prediction from the actual price at time T
        start_price = actual_data[i]
        
        # 2. Simulate the prediction sequence for M steps
        predicted_sequence = []
        for j in range(1, horizon + 1):
            # The model is "not far off": prediction follows the actual path but with some drift/error.
            # Base prediction on the actual path (i + j) but add slight noise based on the step (j)
            
            # Use the actual movement from T to T+j as the base
            actual_movement = actual_data[i + j] - start_price
            
            # Add small prediction error that increases with horizon (more uncertainty further out)
            error = np.random.normal(0, 0.1 * (j/horizon)) * actual_movement
            
            # The predicted price at T+j is the start price + base movement + error
            predicted_price = actual_data[i] + actual_movement * (1 + 0.1 * np.sin(j/10))
            
            predicted_sequence.append(predicted_price)
        
        # Store prediction with its start time and the sequence
        predictions[time_T] = pd.Series(
            predicted_sequence, 
            index=actual_prices.index[i+1 : i + horizon + 1],
            name=f'Pred_Start_{i}'
        )
        
    return predictions

# --- 2. Visualization ---

def plot_sequential_predictions(actual_prices, predictions, horizon, interval):
    """Plots the actual price and overlays the sequential prediction paths."""
    
    plt.figure(figsize=(16, 8))
    
    # A. Plot Actual Price Data (Ground Truth)
    plt.plot(actual_prices.index, actual_prices.values, 
             label='Actual Price Path', color='#333333', linewidth=3) # Dark grey/black line
    
    # B. Plot Prediction Overlays
    for start_time, predicted_series in predictions.items():
        # Get the actual price where the prediction started (T)
        start_price_point = actual_prices[start_time]
        
        # The prediction starts at the next minute (T+1), so we need to plot the T point too
        plot_index = pd.to_datetime([start_time])
        plot_index = plot_index.append(predicted_series.index)
        
        plot_values = np.array([start_price_point])
        plot_values = np.concatenate((plot_values, predicted_series.values))
        
        # Plot the predicted path (dashed blue line)
        plt.plot(plot_index, plot_values,
                 color='#1f77b4', linestyle='--', alpha=0.6, linewidth=1.5)
        
        # C. Mark the Prediction Initiation Point (Time T)
        plt.plot(start_time, start_price_point, 'o', 
                 color='red', markersize=6, zorder=5, 
                 label=f'Prediction Start (T={start_time.strftime("%H:%M")})' if start_time == predictions.keys()[0] else "")
        
    plt.title(f'CAT 2 Sequential Price Prediction Model Simulation (M={horizon} Min Lookahead)')
    plt.xlabel('Time')
    plt.ylabel('Price Index Value')
    plt.legend(['Actual Price Path', 'Predicted Sequence'])
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# --- 3. Execution ---
if __name__ == "__main__":
    
    # Generate the ground truth data
    actual_prices = simulate_data(TOTAL_STEPS)
    
    # Generate the sequential predictions
    sequential_predictions = simulate_predictions(
        actual_prices, 
        PREDICTION_HORIZON, 
        PREDICTION_INTERVAL
    )
    
    # Visualize the results
    plot_sequential_predictions(
        actual_prices, 
        sequential_predictions, 
        PREDICTION_HORIZON, 
        PREDICTION_INTERVAL
    )
    
    print(f"\nSimulation Complete:")
    print(f"- Total Simulated Time: {TOTAL_STEPS} minutes.")
    print(f"- Model predicts a {PREDICTION_HORIZON}-minute sequence.")
    print(f"- A new prediction sequence is generated every {PREDICTION_INTERVAL} minutes.")