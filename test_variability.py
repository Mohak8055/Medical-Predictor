"""
Test script to demonstrate realistic variability in predictions
"""

import pandas as pd
from database import GlucoseDataFetcher
from preprocessor import GlucosePreprocessor
from predictor import GlucosePredictor

# Configuration
PATIENT_ID = 132
PREDICT_DAYS = 30  # Predict 1 month

print("="*80)
print("Testing Prediction Variability")
print("="*80)

# Load and preprocess data
print("\n1. Loading data...")
fetcher = GlucoseDataFetcher(PATIENT_ID)
raw_data = fetcher.fetch_patient_data()
fetcher.disconnect()

print("2. Preprocessing...")
preprocessor = GlucosePreprocessor()
cleaned_data = preprocessor.clean_data(raw_data)
featured_data = preprocessor.add_features(cleaned_data)

# Test with Random Forest algorithm
print("\n3. Training Random Forest model...")
predictor = GlucosePredictor(algorithm='random_forest')
predictor.train_ml_model(featured_data)

# Generate predictions WITH noise (realistic)
print("\n4. Generating predictions WITH realistic variability...")
predictions_with_noise = predictor.predict_future(
    periods=PREDICT_DAYS * 24,
    freq='H',
    add_noise=True,
    noise_scale=0.15  # 15% variability
)

# Re-train to reset state
predictor2 = GlucosePredictor(algorithm='random_forest')
predictor2.train_ml_model(featured_data)

# Generate predictions WITHOUT noise (old behavior)
print("\n5. Generating predictions WITHOUT variability (old method)...")
predictions_without_noise = predictor2.predict_future(
    periods=PREDICT_DAYS * 24,
    freq='H',
    add_noise=False
)

# Get future predictions only
last_date = cleaned_data.index[-1]
future_with = predictions_with_noise[predictions_with_noise['timestamp'] > last_date]
future_without = predictions_without_noise[predictions_without_noise['timestamp'] > last_date]

# Analysis
print("\n" + "="*80)
print("COMPARISON RESULTS")
print("="*80)

print("\nWITH Realistic Variability:")
print(f"  Mean: {future_with['predicted_glucose'].mean():.1f} mg/dL")
print(f"  Std Dev: {future_with['predicted_glucose'].std():.1f} mg/dL")
print(f"  Range: {future_with['predicted_glucose'].min():.1f} - {future_with['predicted_glucose'].max():.1f} mg/dL")

print("\nWITHOUT Variability (Fixed Patterns):")
print(f"  Mean: {future_without['predicted_glucose'].mean():.1f} mg/dL")
print(f"  Std Dev: {future_without['predicted_glucose'].std():.1f} mg/dL")
print(f"  Range: {future_without['predicted_glucose'].min():.1f} - {future_without['predicted_glucose'].max():.1f} mg/dL")

print("\nHistorical Data (for reference):")
print(f"  Mean: {cleaned_data['glucose'].mean():.1f} mg/dL")
print(f"  Std Dev: {cleaned_data['glucose'].std():.1f} mg/dL")
print(f"  Range: {cleaned_data['glucose'].min():.1f} - {cleaned_data['glucose'].max():.1f} mg/dL")

# Check for fixed patterns
print("\n" + "="*80)
print("PATTERN ANALYSIS")
print("="*80)

# Sample Tuesdays at 2 PM
tuesday_2pm_with = future_with[
    (future_with['timestamp'].dt.dayofweek == 1) &
    (future_with['timestamp'].dt.hour == 14)
]['predicted_glucose'].values

tuesday_2pm_without = future_without[
    (future_without['timestamp'].dt.dayofweek == 1) &
    (future_without['timestamp'].dt.hour == 14)
]['predicted_glucose'].values

print(f"\nTuesday 2 PM predictions WITH variability:")
print(f"  Values: {tuesday_2pm_with[:5]}")
print(f"  Std Dev: {tuesday_2pm_with.std():.2f} mg/dL")

print(f"\nTuesday 2 PM predictions WITHOUT variability:")
print(f"  Values: {tuesday_2pm_without[:5]}")
print(f"  Std Dev: {tuesday_2pm_without.std():.2f} mg/dL")

if tuesday_2pm_without.std() < 0.01:
    print("  ⚠️  FIXED PATTERN DETECTED - All Tuesdays at 2 PM have same value!")
else:
    print("  ✓  Values vary naturally")

if tuesday_2pm_with.std() > 1.0:
    print("\n  ✓  WITH variability: Realistic variation across weeks!")

# Save data for visualization
print("\n6. Saving prediction data...")
future_with.to_csv('predictions_with_variability.csv', index=False)
future_without.to_csv('predictions_without_variability.csv', index=False)
print(f"  ✓ Saved 'predictions_with_variability.csv'")
print(f"  ✓ Saved 'predictions_without_variability.csv'")

print("\n" + "="*80)
print("TEST COMPLETE!")
print("="*80)
print("\nThe new method adds realistic variability to avoid fixed patterns.")
print("This makes predictions more like real-world glucose readings.")
