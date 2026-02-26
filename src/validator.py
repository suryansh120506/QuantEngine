import sqlite3
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# 1. Load data and Model
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
db_path = os.path.join(project_root, "data", "stock_vault.db")
model_path = os.path.join(project_root, "models", "reliance_model.keras")

conn = sqlite3.connect(db_path)
df = pd.read_sql('SELECT * FROM raw_market_data', conn)
conn.close()

# 2. Schema Cleaning (Flatten multi-index/tuples)
df.columns = [col if isinstance(col, str) else col[0] for col in df.columns]

# Identify 'Close' and 'Date' dynamically
target_col = 'Close' if 'Close' in df.columns else df.columns[1]
date_col = 'Date' if 'Date' in df.columns else df.columns[0]

# Ensure sorting
df = df.sort_values(by=date_col)

# 3. Prepare features (MA50)
df['MA50'] = df[target_col].rolling(window=50).mean()
# We take the last 150 available days to test the final 50 days
df_test = df.dropna().tail(150)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df_test[[target_col, 'MA50']].values)

# 4. Load Model
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    print(f"❌ Error loading model: {e}. Make sure you ran trainer.py!")
    exit()

# 5. Iterative Backtesting (Predicting the last 50 days)
print(f"🔄 Running Backtest for {target_col}...")
actual, predicted = [], []

for i in range(100, len(scaled_data)):
    # Create the 100-day window for this specific point in history
    X_input = np.reshape(scaled_data[i-100:i, :], (1, 100, 2))
    
    # Get the prediction
    pred_scaled = model.predict(X_input, verbose=0)
    
    # Inverse Scale prediction
    dummy_pred = np.zeros((1, 2))
    dummy_pred[0, 0] = pred_scaled[0, 0]
    predicted.append(scaler.inverse_transform(dummy_pred)[0, 0])
    
    # Inverse Scale actual value
    dummy_actual = np.zeros((1, 2))
    dummy_actual[0, 0] = scaled_data[i, 0]
    actual.append(scaler.inverse_transform(dummy_actual)[0, 0])

# 6. Final Visualization of Accuracy
plt.figure(figsize=(12,6))
plt.plot(actual, label='Actual Reliance Price', color='blue', linewidth=2)
plt.plot(predicted, label='Quant Engine Prediction', color='red', linestyle='--', linewidth=2)
plt.title('Stacked LSTM Backtest: Historical Accuracy (Reliance.NS)')
plt.xlabel('Days (Last 50 Trading Days)')
plt.ylabel('Price in ₹')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("✅ Backtest Complete. Check the graph to see how close the red line is to the blue!")