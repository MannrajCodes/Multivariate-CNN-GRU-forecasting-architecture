import numpy as np
from keras.models import Sequential 
# FIX: Changed import to the top-level keras path
from keras.layers import Conv1D, Flatten, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Utility Functions (Kept Consistent) ---

def calculate_rrmse(rmse: float, actual_values: np.ndarray) -> float:
    """Calculates Relative Root Mean Square Error (RRMSE) in percent."""
    mean_actual = np.mean(actual_values)
    return (rmse / mean_actual) * 100

def get_metrics(y_test_unscaled: np.ndarray, y_pred_unscaled: np.ndarray) -> dict:
    """Calculates all evaluation metrics: MAE, RMSE, R2, RRMSE."""
    
    mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled) 
    rmse = np.sqrt(mean_squared_error(y_test_unscaled, y_pred_unscaled)) 
    r2 = r2_score(y_test_unscaled, y_pred_unscaled) 
    rrmse = calculate_rrmse(rmse, y_test_unscaled) 
    
    return {
        'MAE': f"{mae:.2f}", 
        'RMSE': f"{rmse:.2f}",
        'R²': f"{r2:.4f}",
        'RRMSE': f"{rrmse:.2f}%"
    }

# --- CNN Model Definition (Now Multivariate) ---
def build_cnn_model(time_steps: int, n_features: int) -> Sequential:
    """
    Creates the 1D CNN model architecture, updated for multivariate input.
    Input shape is (time_steps, n_features).
    """
    model = Sequential([
        Conv1D(filters=256, kernel_size=3, activation='relu', 
               padding='same', input_shape=(time_steps, n_features)), 
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='linear')
    ])
    return model
