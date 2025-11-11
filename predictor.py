"""
Glucose prediction module using Facebook Prophet and ML models
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class GlucosePredictor:
    """Handles glucose level predictions using time series models"""

    def __init__(self):
        self.prophet_model = None
        self.ml_model = None
        self.scaler = StandardScaler()
        self.training_data = None
        self.feature_cols = None

    def prepare_prophet_data(self, df):
        """
        Prepare data for Prophet model
        Prophet requires columns: 'ds' (datetime) and 'y' (value)
        """
        prophet_df = pd.DataFrame({
            'ds': df.index,
            'y': df['glucose']
        })

        return prophet_df

    def train_prophet_model(self, df):
        """
        Train Facebook Prophet model
        Prophet is excellent for:
        - Capturing daily/weekly seasonality
        - Handling missing data and outliers
        - Providing uncertainty intervals
        - Making future predictions
        """
        print(f"\n[OK] Training Prophet model...")

        prophet_df = self.prepare_prophet_data(df)

        # Initialize Prophet with custom parameters
        self.prophet_model = Prophet(
            changepoint_prior_scale=0.05,  # Flexibility of trend changes
            seasonality_prior_scale=10.0,   # Strength of seasonality
            holidays_prior_scale=10.0,      # Holiday effects
            seasonality_mode='multiplicative',  # Seasonality type
            interval_width=0.95,            # 95% confidence interval
            daily_seasonality=True,         # Daily patterns
            weekly_seasonality=True,        # Weekly patterns
            yearly_seasonality=False        # Not enough data for yearly
        )

        # Add custom hourly seasonality
        self.prophet_model.add_seasonality(
            name='hourly',
            period=1,
            fourier_order=8
        )

        # Fit the model
        self.prophet_model.fit(prophet_df)

        print(f"  Prophet model trained on {len(prophet_df)} samples")

        return self.prophet_model

    def train_ml_model(self, df):
        """
        Train Random Forest model as a complementary predictor
        Uses engineered features for predictions
        """
        print(f"\n[OK] Training Random Forest model...")

        # Prepare features (all columns except glucose)
        self.feature_cols = [col for col in df.columns if col not in ['glucose', 'glucose_std', 'glucose_min', 'glucose_max', 'reading_count']]

        if len(self.feature_cols) == 0:
            print("  Warning: No features available for ML model")
            return None

        X = df[self.feature_cols].values
        y = df['glucose'].values

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train Random Forest
        self.ml_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        self.ml_model.fit(X_scaled, y)

        # Calculate training accuracy
        train_score = self.ml_model.score(X_scaled, y)
        print(f"  Random Forest trained (R² score: {train_score:.4f})")

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.ml_model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"  Top 5 important features:")
        for idx, row in feature_importance.head(5).iterrows():
            print(f"    {row['feature']}: {row['importance']:.4f}")

        self.training_data = df

        return self.ml_model

    def predict_future(self, periods, freq='H'):
        """
        Predict future glucose levels using Prophet

        Args:
            periods: Number of periods to predict
            freq: Frequency ('H' for hours, 'D' for days)

        Returns:
            DataFrame with predictions and confidence intervals
        """
        if self.prophet_model is None:
            raise ValueError("Prophet model not trained yet!")

        print(f"\n[OK] Generating predictions for {periods} {freq}...")

        # Create future dataframe
        future = self.prophet_model.make_future_dataframe(
            periods=periods,
            freq=freq,
            include_history=True
        )

        # Make predictions
        forecast = self.prophet_model.predict(future)

        # Extract relevant columns
        predictions = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']].copy()
        predictions.columns = ['timestamp', 'predicted_glucose', 'lower_bound', 'upper_bound', 'trend']

        # Ensure predictions are within valid range
        predictions['predicted_glucose'] = predictions['predicted_glucose'].clip(lower=40, upper=400)
        predictions['lower_bound'] = predictions['lower_bound'].clip(lower=40, upper=400)
        predictions['upper_bound'] = predictions['upper_bound'].clip(lower=40, upper=400)

        print(f"  Generated {len(predictions)} predictions")
        print(f"  Prediction range: {predictions['predicted_glucose'].iloc[-periods:].min():.1f} - {predictions['predicted_glucose'].iloc[-periods:].max():.1f} mg/dL")

        return predictions

    def predict_date_range(self, start_date, end_date):
        """
        Predict glucose levels for a specific date range

        Args:
            start_date: Start date (string or datetime)
            end_date: End date (string or datetime)

        Returns:
            DataFrame with predictions for the specified range
        """
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        # Calculate periods needed
        periods = int((end_date - start_date).total_seconds() / 3600) + 1

        # Get predictions
        predictions = self.predict_future(periods=periods, freq='H')

        # Filter to requested date range
        mask = (predictions['timestamp'] >= start_date) & (predictions['timestamp'] <= end_date)
        filtered_predictions = predictions[mask].copy()

        print(f"  Filtered to {len(filtered_predictions)} predictions for date range")

        return filtered_predictions

    def evaluate_model(self, df, test_size=0.2):
        """
        Evaluate model performance using train-test split
        """
        print(f"\n[OK] Evaluating model performance...")

        # Split data
        split_idx = int(len(df) * (1 - test_size))
        train_data = df.iloc[:split_idx]
        test_data = df.iloc[split_idx:]

        # Train on training data only
        prophet_df = self.prepare_prophet_data(train_data)
        model = Prophet(
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
            seasonality_mode='multiplicative',
            interval_width=0.95,
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False
        )
        model.fit(prophet_df)

        # Predict on test data
        future = model.make_future_dataframe(periods=len(test_data), freq='H')
        forecast = model.predict(future)

        # Calculate metrics on test set
        test_predictions = forecast.iloc[-len(test_data):]['yhat'].values
        test_actual = test_data['glucose'].values

        mae = np.mean(np.abs(test_predictions - test_actual))
        rmse = np.sqrt(np.mean((test_predictions - test_actual) ** 2))
        mape = np.mean(np.abs((test_actual - test_predictions) / test_actual)) * 100

        print(f"  Test Set Performance:")
        print(f"    MAE (Mean Absolute Error): {mae:.2f} mg/dL")
        print(f"    RMSE (Root Mean Squared Error): {rmse:.2f} mg/dL")
        print(f"    MAPE (Mean Absolute Percentage Error): {mape:.2f}%")

        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'test_predictions': test_predictions,
            'test_actual': test_actual
        }
