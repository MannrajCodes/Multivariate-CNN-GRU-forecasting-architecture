from keras.models import Sequential 
# FIX: Changed import to the top-level keras path
from keras.layers import GRU, Dense

# --- GRU Model Definition (Now Multivariate) ---
def build_gru_model(time_steps: int, n_features: int) -> Sequential:
    """
    Creates the GRU model architecture, updated for multivariate input.
    Input shape is (time_steps, n_features).
    """
    model = Sequential([
        GRU(256, activation='relu', input_shape=(time_steps, n_features)), 
        Dense(128, activation='relu'),
        Dense(1, activation='linear')
    ])
    return model
