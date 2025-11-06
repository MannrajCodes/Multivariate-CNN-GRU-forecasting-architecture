import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest

# FIX: Suppressing the Pylance warning for this specific line. 
# The interpreter can resolve this path, but the linter gets confused by the structure.
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator # type: ignore


class DataPipeline:
    def __init__(self, time_steps=3, test_size=0.2, contamination=0.02, rolling_window=30):
        self.time_steps = time_steps
        self.test_size = test_size
        self.contamination = contamination
        self.rolling_window = rolling_window
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))

    def _add_exogenous_features(self, data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Extracts and normalizes time-based exogenous features."""
        
        data_exog = data.copy()
        
        # 1. Day of Week (0=Monday, 6=Sunday)
        data_exog['dayofweek'] = data_exog.index.dayofweek
        # 2. Day of Year (1 to 366)
        data_exog['dayofyear'] = data_exog.index.dayofyear
        # 3. Is Weekend? (Binary: 1 if Saturday or Sunday)
        data_exog['is_weekend'] = data_exog.index.dayofweek.isin([5, 6]).astype(int)
        
        # 4. Is Holiday? (Simple proxy)
        data_exog['is_holiday'] = ((data_exog.index.month == 1) & (data_exog.index.day == 1)).astype(int)

        exog_cols = ['dayofweek', 'dayofyear', 'is_weekend', 'is_holiday']
        
        data_exog[exog_cols] = self.feature_scaler.fit_transform(data_exog[exog_cols])
        
        return data_exog, exog_cols

    def _anomaly_detection_and_impute(self, data: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Applies Isolation Forest to detect and impute outliers/NaNs."""
        data_clean = data.copy()
        # Using .ffill() directly to avoid FutureWarning
        data_clean[target_col] = data_clean[target_col].ffill()
        
        iso_forest = IsolationForest(
            contamination=self.contamination, 
            random_state=42, 
            n_jobs=-1
        )
        
        outlier_preds = iso_forest.fit_predict(data_clean[[target_col]])
        
        data_clean.loc[outlier_preds == -1, target_col] = np.nan
        data_clean[target_col] = data_clean[target_col].ffill()

        return data_clean
    
    def _apply_rolling_mean(self, data: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, str]:
        """Applies rolling mean smoothing."""
        data_with_roll = data.copy()
        new_col = f'{target_col}_rolling'
        
        data_with_roll[new_col] = data_with_roll[target_col].rolling(
            window=self.rolling_window, 
            min_periods=1
        ).mean()
        
        data_with_roll[new_col] = data_with_roll[new_col].ffill()
        
        return data_with_roll, new_col

    def _scale_and_sequence(self, data: pd.DataFrame, target_col: str, exog_cols: list[str]):
        """Scales target, merges with features, and creates sequences."""
        
        data[target_col] = self.target_scaler.fit_transform(data[[target_col]])
        
        all_cols = exog_cols + [target_col]
        data_for_sequences = data[all_cols].values 
        
        generator = TimeseriesGenerator(
            data_for_sequences, 
            data_for_sequences[:, -1].reshape(-1, 1), 
            length=self.time_steps,
            batch_size=1
        )
        
        X, y = [], []
        for i in range(len(generator)):
            x_i, y_i = generator[i]
            X.append(x_i[0]) 
            y.append(y_i[0])
        
        return np.array(X), np.array(y)

    def run_pipeline(self, raw_data: pd.DataFrame, target_col: str):
        """Runs the complete multivariate data preparation pipeline."""
        
        data_exog, exog_cols = self._add_exogenous_features(raw_data)
        
        data_anom_fixed = self._anomaly_detection_and_impute(data_exog, target_col)
        data_smoothed, rolling_col = self._apply_rolling_mean(data_anom_fixed, target_col)
        
        X, y = self._scale_and_sequence(data_smoothed, rolling_col, exog_cols)
        
        data_length = len(X)
        split_index = int(data_length * (1 - self.test_size))
        
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        n_features = X_train.shape[2] 

        return X_train, X_test, y_train, y_test, self.target_scaler, data_smoothed, rolling_col, n_features
