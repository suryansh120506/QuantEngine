import sqlite3
import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 1. Load Data from SQL Vault
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
db_path = os.path.join(project_root, "data", "stock_vault.db")
model_path = os.path.join(project_root, "models", "reliance_model.keras")

conn = sqlite3.connect(db_path)
df = pd.read_sql('SELECT * FROM raw_market_data', conn)

# DEBUG: Clean column names (Sometimes SQL adds extra levels)
# This flattens the names if they are tuples
df.columns = [col if isinstance(col, str) else col[0] for col in df.columns]
print(f"Columns found: {df.columns.tolist()}")

# 2. Advanced Feature Engineering
# We use .get() or check for 'Close' to avoid the KeyError
target_col = 'Close' if 'Close' in df.columns else df.columns[1] 

df['MA50'] = df[target_col].rolling(window=50).mean()
df = df.dropna()

# 3. Scaling for the Stacked LSTM
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df[[target_col, 'MA50']].values)

# 4. Create Sliding Window (100-day memory)
time_step = 100
X, y = [], []
for i in range(time_step, len(scaled_data)):
    X.append(scaled_data[i-time_step:i, :])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 2))

# 5. Build and Train the Engine
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(100, 2)),
    Dropout(0.2),
    LSTM(50, return_sequences=True),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

print(f"Engine Training Started for {target_col}...")
model.fit(X, y, epochs=10, batch_size=32)
# Save the 'Brain' so we can use it later
model.save(model_path)
print(f"Model saved to disk as {model_path}")
print("Success: Quantitative Engine Trained.")