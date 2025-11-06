from keras.models import Sequential 
# FIX: Changed import to the top-level keras path
from keras.layers import LSTM, Dense

# --- LSTM Model Definition (Now Multivariate) ---
def build_lstm_model(time_steps: int, n_features: int) -> Sequential:
    """
    Creates the LSTM model architecture, updated for multivariate input.
    Input shape is (time_steps, n_features).
    """
    model = Sequential([
        LSTM(256, activation='relu', input_shape=(time_steps, n_features)), 
        Dense(128, activation='relu'),
        Dense(1, activation='linear')
    ])
    return model
