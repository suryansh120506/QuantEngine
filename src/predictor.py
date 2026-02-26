import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

def get_lstm_prediction(df, close_col):
    """Modular function to handle model loading and inference."""
    base_path = os.path.dirname(os.path.abspath(__file__))
    # Adjusting path to find model from the src folder
    model_path = os.path.join(base_path, '..', 'models', 'reliance_model.keras')
    
    if not os.path.exists(model_path):
        return None, "Model file not found"

    model = tf.keras.models.load_model(model_path)
    df['MA50'] = df[close_col].rolling(window=50).mean()
    df_prep = df.dropna().tail(100)
    
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_input = scaler.fit_transform(df_prep[[close_col, 'MA50']].values)
    X_input = np.reshape(scaled_input, (1, 100, 2))
    
    raw_pred = model.predict(X_input)[0, 0]
    dummy = np.zeros((1, 2))
    dummy[0, 0] = raw_pred
    prediction = scaler.inverse_transform(dummy)[0, 0]
    
    return prediction, None