import pandas as pd
import numpy as np
import tensorflow as tf
import os
import datetime 

# ----------------------------------------------------
# SUPPRESS TENSORFLOW WARNINGS AND ENABLE LONG PATH SUPPORT (IF NEEDED)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppresses most TensorFlow logs
# ----------------------------------------------------

from data_fetcher.wikipedia_api import fetch_page_views
from preprocessing.data_pipeline import DataPipeline
from models.cnn_model import build_cnn_model, get_metrics
from models.gru_model import build_gru_model
from models.lstm_model import build_lstm_model
from models.cnn_gru_hybrid_model import build_cnn_gru_hybrid_model 
from visualizer.visualizer import (
    plot_time_series_comparison,
    plot_anomaly_detection_and_rolling_mean,
    plot_forecast_vs_actual,
    plot_training_validation_loss,
    plot_model_comparison_metrics,
    plot_hybrid_performance_comparison,
    plot_all_forecasts_comparison
)

# --- Configuration (Project Settings) ---
LANG = 'en'
ARTICLE = 'Deep_learning' 
START_DATE = '20230101'
# FIX APPLIED HERE: Use yesterday's date dynamically to pull the latest available data.
YESTERDAY = datetime.datetime.now() - datetime.timedelta(days=1)
END_DATE = YESTERDAY.strftime('%Y%m%d') # Format as YYYYMMDD

TIME_STEPS = 3 
BATCH_SIZE = 32
EPOCHS = 100 
CONTAMINATION_RATE = 0.02 
ROLLING_WINDOW_SIZE = 30 
TEST_SPLIT_RATIO = 0.2

def setup_plot_directory(article_name):
    """Creates a unique, timestamped directory for saving all plots."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"plots/{article_name}_{timestamp}"
    os.makedirs(dir_name, exist_ok=True)
    print(f"🖼️ Plots will be saved in: {dir_name}")
    return dir_name

def main_forecasting_agent(lang, article, start_date, end_date, time_steps):
    print(f"🚀 Starting forecast for: {lang.upper()} Wikipedia - '{article}' (Multivariate Hybrid Model Comparison)")
    
    plot_dir = setup_plot_directory(article)
    
    # 1. Data Collection Agent
    print("\n1. Collecting live Wikipedia Page Views data...")
    raw_data = fetch_page_views(lang, article, start_date, end_date)
    
    if raw_data is None:
        print("🛑 Failed to fetch data. Exiting.")
        return
        
    plot_time_series_comparison(raw_data, title=f"{article} - Original Daily Visits", save_path=f"{plot_dir}/01_Original_Time_Series.png")
    
    # 2. Data Preparation Agent
    print("2. Preprocessing data and generating sequences (with Exogenous Features)...")
    pipeline = DataPipeline(time_steps=time_steps, 
                            test_size=TEST_SPLIT_RATIO,
                            contamination=CONTAMINATION_RATE, 
                            rolling_window=ROLLING_WINDOW_SIZE)
                            
    X_train, X_test, y_train, y_test, target_scaler, data_smoothed, rolling_col, n_features = pipeline.run_pipeline(raw_data, 'views')
    
    print(f"   -> Input Shape: [Samples, {time_steps}, {n_features} features] (Target + 4 Exogenous)")
    
    plot_anomaly_detection_and_rolling_mean(data_smoothed, rolling_col, title=article, save_path=f"{plot_dir}/02_Anomaly_and_Rolling_Mean.png")

    models_to_test = {
        "CNN": build_cnn_model(time_steps, n_features),
        "GRU": build_gru_model(time_steps, n_features),
        "LSTM": build_lstm_model(time_steps, n_features),
        "CNN-GRU Hybrid": build_cnn_gru_hybrid_model(time_steps, n_features),
    }
    
    all_results = {}
    all_predictions = {}
    y_test_unscaled = target_scaler.inverse_transform(y_test)
    metric_counter = 3

    # 3. Deep Learning Models Training and Evaluation Agent
    print("\n3. Training and evaluating Deep Learning Models...")
    
    for model_name, model in models_to_test.items():
        print(f"--- Training {model_name} Model (Features: {n_features}) ---")
        
        model.compile(optimizer='adam', loss='mse') 
        
        history = model.fit(X_train, y_train, 
                            epochs=EPOCHS, 
                            batch_size=BATCH_SIZE, 
                            verbose=0,
                            validation_split=0.1) 
                            
        plot_training_validation_loss(history, model_name, article, save_path=f"{plot_dir}/{metric_counter:02d}_Loss_Curve_{model_name.replace(' ', '_')}.png")
        
        y_pred_scaled = model.predict(X_test, verbose=0)
        y_pred_unscaled = target_scaler.inverse_transform(y_pred_scaled)
        
        # Store predictions
        all_predictions[model_name] = y_pred_unscaled 
        
        metrics = get_metrics(y_test_unscaled, y_pred_unscaled)
        all_results[model_name] = metrics
        
        plot_forecast_vs_actual(y_test_unscaled, y_pred_unscaled, model_name, article, save_path=f"{plot_dir}/{metric_counter + 1:02d}_Forecast_vs_Actual_{model_name.replace(' ', '_')}.png")
        
        metric_counter += 2 
    
    # 4. Results Reporting and Comparison Plots
    print("\n✅ Final Results Summary (Multivariate Hybrid Model Comparison):")
    results_df = pd.DataFrame(all_results).T
    print(results_df)

    # --- Generate ALL Model Comparison Plots (10-13) ---
    print("\n4. Generating ALL Model Comparison Plots...")
    comparison_counter = 10
    for metric in ['MAE', 'RMSE', 'R²', 'RRMSE']:
        plot_model_comparison_metrics(all_results, metric, article, save_path=f"{plot_dir}/{comparison_counter:02d}_Comparison_{metric}.png")
        comparison_counter += 1

    # --- Generate DEDICATED Hybrid Comparison Plots (14-15) ---
    print("   Generating DEDICATED HYBRID comparison plots (R2 and RMSE)...")
    plot_hybrid_performance_comparison(all_results, 'R²', article, save_path=f"{plot_dir}/14_DEDICATED_HYBRID_R2.png")
    plot_hybrid_performance_comparison(all_results, 'RMSE', article, save_path=f"{plot_dir}/15_DEDICATED_HYBRID_RMSE.png")

    # --- Generate NEW ALL-IN-ONE TIME SERIES PLOT (16) ---
    print("   Generating NEW ALL-IN-ONE Time Series Forecast plot...")
    plot_all_forecasts_comparison(y_test_unscaled, all_predictions, article, save_path=f"{plot_dir}/16_ALL_FORECASTS_TIME_SERIES_COMP.png")


if __name__ == "__main__":
    main_forecasting_agent(LANG, ARTICLE, START_DATE, END_DATE, TIME_STEPS)
