"""
Main application for glucose prediction system
"""

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from database import GlucoseDataFetcher
from preprocessor import GlucosePreprocessor
from predictor import GlucosePredictor
from visualizer import GlucoseVisualizer
from config import PATIENT_ID


class GlucosePredictionSystem:
    """Main system that orchestrates all components"""

    def __init__(self, patient_id=PATIENT_ID):
        self.patient_id = patient_id
        self.fetcher = GlucoseDataFetcher(patient_id)
        self.preprocessor = GlucosePreprocessor()
        self.predictor = GlucosePredictor()
        self.visualizer = GlucoseVisualizer()

        self.raw_data = None
        self.cleaned_data = None
        self.featured_data = None
        self.anomalies = None
        self.trained = False

    def load_data(self):
        """Load data from database"""
        print(f"\n{'='*80}")
        print(f"LOADING DATA FOR PATIENT {self.patient_id}")
        print(f"{'='*80}")

        self.raw_data = self.fetcher.fetch_patient_data()

        if self.raw_data is None or len(self.raw_data) == 0:
            print("[X] No data loaded. Exiting.")
            return False

        return True

    def preprocess_data(self):
        """Preprocess and analyze data"""
        print(f"\n{'='*80}")
        print(f"PREPROCESSING DATA")
        print(f"{'='*80}")

        if self.raw_data is None:
            print("[X] No raw data available. Load data first.")
            return False

        # Detect anomalies
        self.anomalies = self.preprocessor.detect_anomalies(self.raw_data)

        # Clean data
        self.cleaned_data = self.preprocessor.clean_data(self.raw_data)

        # Add features
        self.featured_data = self.preprocessor.add_features(self.cleaned_data)

        # Analyze patterns
        self.preprocessor.analyze_patterns(self.cleaned_data)

        return True

    def train_model(self):
        """Train prediction models"""
        print(f"\n{'='*80}")
        print(f"TRAINING PREDICTION MODELS")
        print(f"{'='*80}")

        if self.cleaned_data is None:
            print("[X] No cleaned data available. Preprocess data first.")
            return False

        # Train Prophet model (primary)
        self.predictor.train_prophet_model(self.cleaned_data)

        # Train ML model (secondary)
        if len(self.featured_data.columns) > 1:
            self.predictor.train_ml_model(self.featured_data)

        # Evaluate model
        self.predictor.evaluate_model(self.cleaned_data)

        self.trained = True
        return True

    def predict(self, periods=24, freq='H'):
        """
        Make predictions for future periods

        Args:
            periods: Number of periods to predict
            freq: Frequency ('H' for hours, 'D' for days, 'W' for weeks)

        Returns:
            DataFrame with predictions
        """
        if not self.trained:
            print("[X] Model not trained. Train model first.")
            return None

        print(f"\n{'='*80}")
        print(f"MAKING PREDICTIONS")
        print(f"{'='*80}")

        predictions = self.predictor.predict_future(periods=periods, freq=freq)

        # Get only future predictions
        last_date = self.cleaned_data.index[-1]
        future_predictions = predictions[predictions['timestamp'] > last_date]

        print(f"\n[*] Prediction Summary:")
        print(f"  Period: {future_predictions['timestamp'].min()} to {future_predictions['timestamp'].max()}")
        print(f"  Average predicted glucose: {future_predictions['predicted_glucose'].mean():.1f} mg/dL")
        print(f"  Min predicted glucose: {future_predictions['predicted_glucose'].min():.1f} mg/dL")
        print(f"  Max predicted glucose: {future_predictions['predicted_glucose'].max():.1f} mg/dL")

        return predictions

    def predict_date_range(self, start_date, end_date):
        """
        Predict glucose levels for a specific date range

        Args:
            start_date: Start date (string format: 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS')
            end_date: End date (string format: 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS')

        Returns:
            DataFrame with predictions for the date range
        """
        if not self.trained:
            print("[X] Model not trained. Train model first.")
            return None

        print(f"\n{'='*80}")
        print(f"PREDICTING FOR DATE RANGE")
        print(f"{'='*80}")

        predictions = self.predictor.predict_date_range(start_date, end_date)

        print(f"\n[*] Prediction Summary:")
        print(f"  Period: {predictions['timestamp'].min()} to {predictions['timestamp'].max()}")
        print(f"  Total predictions: {len(predictions)}")
        print(f"  Average predicted glucose: {predictions['predicted_glucose'].mean():.1f} mg/dL")
        print(f"  Min predicted glucose: {predictions['predicted_glucose'].min():.1f} mg/dL")
        print(f"  Max predicted glucose: {predictions['predicted_glucose'].max():.1f} mg/dL")

        return predictions

    def visualize_all(self, predictions=None, show=True, save_dir='./output'):
        """
        Create all visualizations

        Args:
            predictions: Predictions DataFrame (optional)
            show: Whether to display plots
            save_dir: Directory to save plots
        """
        import os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print(f"\n{'='*80}")
        print(f"GENERATING VISUALIZATIONS")
        print(f"{'='*80}")

        # Historical data
        self.visualizer.plot_historical_data(
            self.cleaned_data,
            title=f"Historical Glucose Data - Patient {self.patient_id}",
            save_path=f"{save_dir}/01_historical_data.png"
        )

        # Predictions
        if predictions is not None:
            self.visualizer.plot_predictions(
                self.cleaned_data,
                predictions,
                title=f"Glucose Predictions - Patient {self.patient_id}",
                save_path=f"{save_dir}/02_predictions.png"
            )

        # Anomalies
        if self.anomalies is not None:
            # Need to pass raw data for anomaly visualization
            self.visualizer.plot_anomalies(
                self.cleaned_data,
                self.anomalies,
                title=f"Anomaly Detection - Patient {self.patient_id}",
                save_path=f"{save_dir}/03_anomalies.png"
            )

        # Daily patterns
        self.visualizer.plot_daily_patterns(
            self.cleaned_data,
            title=f"Daily Glucose Patterns - Patient {self.patient_id}",
            save_path=f"{save_dir}/04_daily_patterns.png"
        )

        # Weekly patterns
        self.visualizer.plot_weekly_patterns(
            self.cleaned_data,
            title=f"Weekly Glucose Patterns - Patient {self.patient_id}",
            save_path=f"{save_dir}/05_weekly_patterns.png"
        )

        # Distribution
        self.visualizer.plot_distribution(
            self.cleaned_data,
            title=f"Glucose Distribution - Patient {self.patient_id}",
            save_path=f"{save_dir}/06_distribution.png"
        )

        # Dashboard
        if predictions is not None and self.anomalies is not None:
            self.visualizer.create_dashboard(
                self.cleaned_data,
                predictions,
                self.anomalies,
                save_path=f"{save_dir}/00_dashboard.png"
            )

        print(f"\n[OK] All visualizations saved to {save_dir}/")

        if show:
            self.visualizer.show_all()

    def generate_report(self, predictions=None):
        """Generate a text report of the analysis"""
        print(f"\n{'='*80}")
        print(f"GLUCOSE ANALYSIS REPORT - PATIENT {self.patient_id}")
        print(f"{'='*80}")

        # Data summary
        print(f"\n[*] DATA SUMMARY:")
        print(f"  Total raw readings: {len(self.raw_data):,}")
        print(f"  Cleaned readings (hourly): {len(self.cleaned_data):,}")
        print(f"  Date range: {self.cleaned_data.index.min().strftime('%Y-%m-%d %H:%M')} to {self.cleaned_data.index.max().strftime('%Y-%m-%d %H:%M')}")
        print(f"  Duration: {(self.cleaned_data.index.max() - self.cleaned_data.index.min()).days} days")

        # Glucose statistics
        print(f"\n[*] GLUCOSE STATISTICS:")
        print(f"  Average: {self.cleaned_data['glucose'].mean():.1f} mg/dL")
        print(f"  Median: {self.cleaned_data['glucose'].median():.1f} mg/dL")
        print(f"  Std Dev: {self.cleaned_data['glucose'].std():.1f} mg/dL")
        print(f"  Min: {self.cleaned_data['glucose'].min():.1f} mg/dL")
        print(f"  Max: {self.cleaned_data['glucose'].max():.1f} mg/dL")

        # Time in range
        in_range = ((self.cleaned_data['glucose'] >= 70) & (self.cleaned_data['glucose'] <= 180)).sum()
        below_range = (self.cleaned_data['glucose'] < 70).sum()
        above_range = (self.cleaned_data['glucose'] > 180).sum()
        total = len(self.cleaned_data)

        print(f"\n[*] TIME IN RANGE:")
        print(f"  In range (70-180 mg/dL): {in_range/total*100:.1f}% ({in_range:,} hours)")
        print(f"  Below range (<70 mg/dL): {below_range/total*100:.1f}% ({below_range:,} hours)")
        print(f"  Above range (>180 mg/dL): {above_range/total*100:.1f}% ({above_range:,} hours)")

        # Anomalies
        if self.anomalies:
            print(f"\n[!] ANOMALIES DETECTED:")
            print(f"  Sensor errors: {len(self.anomalies['sensor_errors']):,}")
            print(f"  Statistical outliers: {len(self.anomalies['statistical_outliers']):,}")
            print(f"  Sudden changes: {len(self.anomalies['sudden_changes']):,}")
            print(f"  Data gaps: {len(self.anomalies['data_gaps']):,}")

        # Patterns
        if self.preprocessor.stats:
            print(f"\n[*] PATTERNS:")
            hourly = self.preprocessor.stats['hourly_pattern']
            print(f"  Peak hour: {hourly.idxmax()}:00 (avg: {hourly.max():.1f} mg/dL)")
            print(f"  Lowest hour: {hourly.idxmin()}:00 (avg: {hourly.min():.1f} mg/dL)")

            trend = "increasing" if self.preprocessor.stats['trend_slope'] > 0 else "decreasing"
            print(f"  Overall trend: {trend} ({self.preprocessor.stats['trend_slope']:.4f} mg/dL per hour)")

        # Predictions
        if predictions is not None:
            last_date = self.cleaned_data.index[-1]
            future = predictions[predictions['timestamp'] > last_date]

            print(f"\n[*] PREDICTIONS:")
            print(f"  Prediction period: {future['timestamp'].min().strftime('%Y-%m-%d %H:%M')} to {future['timestamp'].max().strftime('%Y-%m-%d %H:%M')}")
            print(f"  Predicted average: {future['predicted_glucose'].mean():.1f} mg/dL")
            print(f"  Predicted range: {future['predicted_glucose'].min():.1f} - {future['predicted_glucose'].max():.1f} mg/dL")

        print(f"\n{'='*80}")

    def run_full_analysis(self, predict_hours=168, show_plots=True):
        """
        Run complete analysis pipeline

        Args:
            predict_hours: Number of hours to predict (default: 168 = 7 days)
            show_plots: Whether to display plots
        """
        # Load data
        if not self.load_data():
            return

        # Preprocess
        if not self.preprocess_data():
            return

        # Train model
        if not self.train_model():
            return

        # Make predictions
        predictions = self.predict(periods=predict_hours, freq='H')

        # Generate visualizations
        self.visualize_all(predictions=predictions, show=show_plots)

        # Generate report
        self.generate_report(predictions=predictions)

        return predictions

    def cleanup(self):
        """Cleanup resources"""
        self.fetcher.disconnect()
        self.visualizer.close_all()


def main():
    """Main entry point with examples"""
    print("""
    ================================================================

              GLUCOSE PREDICTION SYSTEM v1.0
              Patient ID: 132

    ================================================================
    """)

    # Initialize system
    system = GlucosePredictionSystem(patient_id=132)

    try:
        # Example 1: Run full analysis with 7-day prediction
        print("\n>> Running full analysis with 7-day prediction...\n")
        predictions_7days = system.run_full_analysis(predict_hours=168, show_plots=False)

        # Example 2: Predict for a specific date range
        print("\n\n>> Predicting for specific date range (1 month from now)...\n")
        from datetime import datetime, timedelta

        start_date = datetime.now() + timedelta(days=1)
        end_date = start_date + timedelta(days=30)

        predictions_custom = system.predict_date_range(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )

        # Save predictions to CSV
        if predictions_custom is not None:
            predictions_custom.to_csv('./output/predictions_custom_range.csv', index=False)
            print(f"\n[OK] Custom predictions saved to ./output/predictions_custom_range.csv")

        # Example 3: Query predictions for next year
        print("\n\n>> Predicting for next year...\n")
        predictions_year = system.predict(periods=365*24, freq='H')  # 1 year in hours

        if predictions_year is not None:
            # Save to CSV
            predictions_year.to_csv('./output/predictions_one_year.csv', index=False)
            print(f"\n[OK] Year predictions saved to ./output/predictions_one_year.csv")

            # Get future predictions only
            last_date = system.cleaned_data.index[-1]
            future_only = predictions_year[predictions_year['timestamp'] > last_date]

            # Analyze yearly predictions
            print(f"\n[*] YEARLY PREDICTION ANALYSIS:")
            print(f"  Total predictions: {len(future_only):,}")
            print(f"  Average glucose: {future_only['predicted_glucose'].mean():.1f} mg/dL")
            print(f"  Min glucose: {future_only['predicted_glucose'].min():.1f} mg/dL")
            print(f"  Max glucose: {future_only['predicted_glucose'].max():.1f} mg/dL")

            # Monthly averages
            future_only_copy = future_only.copy()
            future_only_copy['month'] = pd.to_datetime(future_only_copy['timestamp']).dt.to_period('M')
            monthly_avg = future_only_copy.groupby('month')['predicted_glucose'].mean()

            print(f"\n  Monthly averages for next year:")
            for month, avg in monthly_avg.head(12).items():
                print(f"    {month}: {avg:.1f} mg/dL")

        print(f"\n\n{'='*80}")
        print("[OK] ANALYSIS COMPLETE!")
        print(f"{'='*80}")
        print(f"\nAll results saved to './output/' directory")
        print(f"  - Visualizations (PNG files)")
        print(f"  - Predictions (CSV files)")

    except Exception as e:
        print(f"\n[X] Error during analysis: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        system.cleanup()


if __name__ == "__main__":
    main()
