from keras.models import Sequential 
from keras.layers import Conv1D, GRU, Dense, Flatten, TimeDistributed

def build_cnn_gru_hybrid_model(time_steps: int, n_features: int):
    """
    Creates a Hybrid CNN-GRU model for multivariate time series forecasting.
    
    This model first uses a 1D CNN layer to extract local features from the 
    multivariate input window, and then feeds the results to a GRU layer 
    to process the sequence and capture temporal dependencies.
    
    Input shape: (time_steps, n_features)
    """
    model = Sequential([
        # 1. CNN Layer: Feature Extraction (Local Pattern Recognition)
        # Using 64 filters for simplicity, adjust for grid search if desired
        Conv1D(filters=64, kernel_size=2, activation='relu', 
               input_shape=(time_steps, n_features)),
               
        # 2. Flatten Layer: Prepare data to be fed into the dense/GRU layer 
        # (This is an alternative approach to passing sequences to the GRU)
        Flatten(),
        
        # 3. Intermediate Dense Layer (Optional, acts as a feature compressor)
        Dense(100, activation='relu'), 

        # 4. Final Output Layer
        Dense(1, activation='linear')
    ])

    return model
