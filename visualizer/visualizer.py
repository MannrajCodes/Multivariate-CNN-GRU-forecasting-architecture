import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# Set a consistent style for all plots
sns.set_theme(style="whitegrid")

# Helper function to save and show the plot
def save_and_show_plot(save_path: str):
    """Saves the current matplotlib figure and then displays it."""
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close() # Close figure to free memory

# --- STANDARD PLOTS (UNMODIFIED) ---

def plot_time_series_comparison(data: pd.DataFrame, title: str, save_path: str):
    """Replicates Figure 2: Daily traffic time series for different language versions."""
    plt.figure(figsize=(12, 6))
    label = f"{data.index.name} - {title.split(' - ')[0]} Visits"
    data['views'].plot(label=label)
    plt.title(title)
    plt.xlabel("Day")
    plt.ylabel("Daily Visit Count")
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.legend(loc='upper right')
    plt.tight_layout()
    save_and_show_plot(save_path)

def plot_anomaly_detection_and_rolling_mean(data: pd.DataFrame, rolling_mean_col: str, title: str, save_path: str):
    """Replicates Figures 4-6: Effect of anomaly detection and rolling mean."""
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['views'], label='Original Visits (Cleaned)', linewidth=1.5, alpha=0.6, color='darkblue')
    plt.plot(data.index, data[rolling_mean_col], label='Rolling Mean (Smoothed)', 
             linewidth=2.5, color='coral')
    plt.title(f"{title} - Anomaly Handling and Smoothing")
    plt.xlabel("Day")
    plt.ylabel("Daily Visit Count")
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.legend()
    plt.tight_layout()
    save_and_show_plot(save_path)

def plot_forecast_vs_actual(y_test_unscaled: np.ndarray, y_pred_unscaled: np.ndarray, model_name: str, article: str, save_path: str):
    """Replicates Figures 15-17: Forecasted vs True values on the test set."""
    time_points = range(len(y_test_unscaled))
    plt.figure(figsize=(10, 5))
    plt.plot(time_points, y_test_unscaled.flatten(), label='True', color='darkblue', linewidth=2, alpha=0.8)
    plt.plot(time_points, y_pred_unscaled.flatten(), label='Forecasted', color='gold', linewidth=1.5, alpha=1.0)
    plt.title(f"{article} - {model_name} Forecast vs. Actual")
    plt.xlabel("Time (Test Set Index)")
    plt.ylabel("Daily Visit Count (Unscaled)")
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.legend()
    plt.tight_layout()
    save_and_show_plot(save_path)

def plot_training_validation_loss(history, model_name: str, article: str, save_path: str):
    """Replicates Figures 18-20: Training and Validation Loss Curves."""
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Train Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='darkorange')
    plt.title(f"{article} - {model_name} Loss Curve Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.ylim(0, max(max(history.history['loss']), max(history.history['val_loss'])) * 1.1)
    plt.legend()
    plt.tight_layout()
    save_and_show_plot(save_path)
    
def plot_model_comparison_metrics(results: dict, metric: str, article: str, save_path: str):
    """Plot for direct comparison of ALL models (CNN, GRU, LSTM, HYBRID)."""
    model_names = list(results.keys())
    values = [float(results[model][metric].replace('%', '')) for model in model_names]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(model_names, values, color=['skyblue', 'salmon', 'lightgreen', 'purple'])
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(values)*0.01), 
                 f'{yval:.3f}' if metric=='R²' else f'{yval:.2f}', 
                 ha='center', va='bottom', fontsize=10)

    plt.title(f"{article} - Model Comparison: {metric}")
    plt.ylabel(f"{metric} Value")
    
    if metric == 'R²':
        plt.ylim(min(values) * 0.95, 1.0) 
        
    plt.tight_layout()
    save_and_show_plot(save_path)


# --- NEW DEDICATED HYBRID COMPARISON PLOT ---

def plot_hybrid_performance_comparison(results: dict, metric: str, article: str, save_path: str):
    """
    New dedicated plot to compare Hybrid model performance against baselines, 
    designed for research paper emphasis.
    """
    model_names = list(results.keys())
    values = [float(results[model][metric].replace('%', '')) for model in model_names]
    
    # Define colors: Highlight the Hybrid model (last one) in a distinct color
    colors = ['lightblue', 'lightsalmon', 'lightgreen', 'darkred'] 

    plt.figure(figsize=(9, 6))
    bars = plt.bar(model_names, values, color=colors)

    # Add value labels on top of the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(values) * 0.01), 
                 f'{yval:.4f}' if metric == 'R²' else f'{yval:.2f}', 
                 ha='center', va='bottom', fontsize=10, weight='bold')

    plt.title(f"Hybrid Model Advantage: {metric} Comparison for {article}", fontsize=14)
    plt.ylabel(f"{metric} Value (Higher is better for R²; Lower for Error Metrics)", fontsize=10)
    plt.xlabel("Model Architecture")
    
    # Adjust Y-axis for R² to emphasize the difference near 1.0
    if metric == 'R²':
        plt.ylim(0.5, 1.0) 
        
    plt.tight_layout()
    save_and_show_plot(save_path)

# --- NEW ALL-FORECASTS-ON-ONE-PLOT FUNCTION ---

def plot_all_forecasts_comparison(y_test_unscaled: np.ndarray, y_pred_dict: dict, article: str, save_path: str):
    """
    Plots the true values against the forecasts of ALL models on a single chart.
    This provides the best visual comparison of model fit over time.
    """
    time_points = range(len(y_test_unscaled))
    
    # Set up a large figure for clarity
    plt.figure(figsize=(14, 7))
    
    # Plot the True Data (Baseline)
    plt.plot(time_points, y_test_unscaled.flatten(), 
             label='True Value', color='black', linewidth=2.5, alpha=0.9)
    
    # Define colors for the prediction lines
    pred_colors = {
        'CNN': 'skyblue', 
        'GRU': 'salmon', 
        'LSTM': 'lightgreen', 
        'CNN-GRU Hybrid': 'darkred' # Highlight the novel model
    }
    
    # Plot all predicted values
    for model_name, y_pred_unscaled in y_pred_dict.items():
        plt.plot(time_points, y_pred_unscaled.flatten(), 
                 label=f'{model_name} Forecast', 
                 color=pred_colors.get(model_name, 'gray'), # Use defined color, fall back to gray
                 linestyle='--', 
                 linewidth=1.2)

    plt.title(f"{article} - Multi-Model Forecast Comparison on Test Data", fontsize=16)
    plt.xlabel("Time (Test Set Index)")
    plt.ylabel("Daily Visit Count (Unscaled)")
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.legend(loc='upper right', ncol=2)
    plt.grid(True)
    plt.tight_layout()
    save_and_show_plot(save_path)
