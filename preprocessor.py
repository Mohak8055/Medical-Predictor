"""
Data preprocessing and anomaly detection module
"""

import pandas as pd
import numpy as np
from scipy import stats
from config import (
    ANOMALY_THRESHOLD_STD,
    MIN_VALID_GLUCOSE,
    MAX_VALID_GLUCOSE,
    PREDICTION_INTERVAL
)


class GlucosePreprocessor:
    """Handles data cleaning, anomaly detection, and feature engineering"""

    def __init__(self):
        self.anomalies = None
        self.cleaned_data = None
        self.stats = {}

    def detect_anomalies(self, df):
        """
        Detect anomalies in glucose readings:
        1. Sensor errors (values outside valid range)
        2. Statistical outliers (beyond threshold standard deviations)
        3. Sudden spikes/drops (rate of change)
        """
        anomalies = pd.DataFrame()

        # Type 1: Sensor errors (outside valid range)
        sensor_errors = (df['value'] < MIN_VALID_GLUCOSE) | (df['value'] > MAX_VALID_GLUCOSE)

        # Type 2: Statistical outliers (z-score method)
        z_scores = np.abs(stats.zscore(df['value'].dropna()))
        statistical_outliers = z_scores > ANOMALY_THRESHOLD_STD

        # Align the boolean array with the original dataframe
        statistical_outliers_full = pd.Series(False, index=df.index)
        statistical_outliers_full.iloc[df['value'].dropna().index] = statistical_outliers

        # Type 3: Sudden changes (rate of change)
        df['glucose_diff'] = df['value'].diff()
        df['time_diff'] = df['timestamp'].diff().dt.total_seconds() / 60  # minutes

        # Rate of change (mg/dL per minute)
        df['rate_of_change'] = df['glucose_diff'] / df['time_diff']

        # Sudden spike: change > 5 mg/dL per minute
        sudden_changes = df['rate_of_change'].abs() > 5

        # Type 4: Data gaps (more than 15 minutes between readings)
        data_gaps = df['time_diff'] > 15

        # Combine all anomaly types
        anomalies = df[sensor_errors | statistical_outliers_full | sudden_changes].copy()
        anomalies['anomaly_type'] = ''
        anomalies.loc[sensor_errors[sensor_errors].index, 'anomaly_type'] += 'sensor_error,'
        anomalies.loc[statistical_outliers_full[statistical_outliers_full].index, 'anomaly_type'] += 'statistical_outlier,'
        anomalies.loc[sudden_changes[sudden_changes].index, 'anomaly_type'] += 'sudden_change,'

        # Store gap information separately
        gaps = df[data_gaps].copy()
        gaps['gap_duration'] = gaps['time_diff']

        print(f"\n[OK] Anomaly Detection Results:")
        print(f"  Sensor errors: {sensor_errors.sum()}")
        print(f"  Statistical outliers: {statistical_outliers_full.sum()}")
        print(f"  Sudden changes: {sudden_changes.sum()}")
        print(f"  Data gaps (>15 min): {data_gaps.sum()}")
        print(f"  Total anomalies: {len(anomalies)}")

        self.anomalies = {
            'all': anomalies,
            'sensor_errors': df[sensor_errors],
            'statistical_outliers': df[statistical_outliers_full],
            'sudden_changes': df[sudden_changes],
            'data_gaps': gaps
        }

        return self.anomalies

    def clean_data(self, df):
        """
        Clean the data by:
        1. Removing sensor errors
        2. Handling outliers (cap instead of remove)
        3. Interpolating missing values
        4. Resampling to consistent intervals
        """
        print(f"\n[OK] Cleaning data...")
        original_count = len(df)

        # Create a copy
        cleaned = df.copy()

        # Remove sensor errors (clearly invalid readings)
        cleaned = cleaned[
            (cleaned['value'] >= MIN_VALID_GLUCOSE) &
            (cleaned['value'] <= MAX_VALID_GLUCOSE)
        ]
        print(f"  Removed {original_count - len(cleaned)} sensor errors")

        # Cap extreme outliers instead of removing them
        # (preserve the pattern but limit extreme values)
        q99 = cleaned['value'].quantile(0.99)
        q01 = cleaned['value'].quantile(0.01)
        cleaned['value'] = cleaned['value'].clip(lower=q01, upper=q99)

        # Set timestamp as index
        cleaned = cleaned.set_index('timestamp')

        # Resample to hourly intervals (aggregate minute-level data)
        cleaned_hourly = cleaned['value'].resample(PREDICTION_INTERVAL).agg({
            'mean': 'mean',
            'std': 'std',
            'min': 'min',
            'max': 'max',
            'count': 'count'
        })

        # Keep only the mean for primary predictions, but store stats
        cleaned_resampled = pd.DataFrame({
            'glucose': cleaned_hourly['mean'].values,
            'glucose_std': cleaned_hourly['std'].values,
            'glucose_min': cleaned_hourly['min'].values,
            'glucose_max': cleaned_hourly['max'].values,
            'reading_count': cleaned_hourly['count'].values
        }, index=cleaned_hourly.index)

        # Interpolate missing values (gaps)
        cleaned_resampled['glucose'] = cleaned_resampled['glucose'].interpolate(
            method='time',
            limit=24  # Don't interpolate gaps longer than 24 hours
        )

        print(f"  Resampled to {len(cleaned_resampled)} hourly intervals")

        self.cleaned_data = cleaned_resampled

        return cleaned_resampled

    def add_features(self, df):
        """
        Add time-based and statistical features for better predictions:
        - Time features (hour, day of week, month)
        - Lagged features (previous values)
        - Rolling statistics (trends)
        """
        features = df.copy()

        # Time-based features
        features['hour'] = features.index.hour
        features['day_of_week'] = features.index.dayofweek
        features['day_of_month'] = features.index.day
        features['month'] = features.index.month

        # Lagged features (previous values)
        for lag in [1, 2, 3, 6, 12, 24]:  # 1h, 2h, 3h, 6h, 12h, 24h ago
            features[f'glucose_lag_{lag}h'] = features['glucose'].shift(lag)

        # Rolling statistics (trends)
        for window in [6, 12, 24, 72]:  # 6h, 12h, 24h, 72h windows
            features[f'glucose_rolling_mean_{window}h'] = features['glucose'].rolling(window=window).mean()
            features[f'glucose_rolling_std_{window}h'] = features['glucose'].rolling(window=window).std()

        # Rate of change
        features['glucose_diff_1h'] = features['glucose'].diff(1)
        features['glucose_diff_3h'] = features['glucose'].diff(3)

        # Drop rows with NaN values from feature engineering
        features = features.dropna()

        print(f"\n[OK] Added features:")
        print(f"  Total features: {len(features.columns)}")
        print(f"  Samples after feature engineering: {len(features)}")

        return features

    def analyze_patterns(self, df):
        """Analyze glucose patterns and trends"""
        print(f"\n[OK] Pattern Analysis:")

        # Daily patterns
        hourly_avg = df.groupby(df.index.hour)['glucose'].mean()
        print(f"  Peak glucose hour: {hourly_avg.idxmax()}:00 (avg: {hourly_avg.max():.1f} mg/dL)")
        print(f"  Lowest glucose hour: {hourly_avg.idxmin()}:00 (avg: {hourly_avg.min():.1f} mg/dL)")

        # Weekly patterns
        daily_avg = df.groupby(df.index.dayofweek)['glucose'].mean()
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        print(f"  Highest avg day: {days[daily_avg.idxmax()]} ({daily_avg.max():.1f} mg/dL)")
        print(f"  Lowest avg day: {days[daily_avg.idxmin()]} ({daily_avg.min():.1f} mg/dL)")

        # Trend analysis
        from scipy.stats import linregress
        x = np.arange(len(df))
        slope, intercept, r_value, p_value, std_err = linregress(x, df['glucose'])

        trend = "increasing" if slope > 0 else "decreasing"
        print(f"  Overall trend: {trend} ({slope:.4f} mg/dL per hour)")
        print(f"  Trend strength (R²): {r_value**2:.4f}")

        # Variability analysis
        print(f"  Overall std dev: {df['glucose'].std():.2f} mg/dL")
        print(f"  Coefficient of variation: {(df['glucose'].std() / df['glucose'].mean() * 100):.2f}%")

        self.stats = {
            'hourly_pattern': hourly_avg,
            'daily_pattern': daily_avg,
            'trend_slope': slope,
            'trend_r2': r_value**2,
            'std_dev': df['glucose'].std()
        }

        return self.stats
