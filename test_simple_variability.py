"""
Simple test to demonstrate realistic variability in predictions (no DB required)
"""

import pandas as pd
import numpy as np
from predictor import GlucosePredictor

print("="*80)
print("Testing Prediction Variability (Simple Demo)")
print("="*80)

# Create synthetic glucose data
print("\n1. Creating synthetic glucose data...")
dates = pd.date_range(start='2024-01-01', periods=30*24, freq='H')

# Generate realistic glucose pattern with daily cycles
hours = np.array([d.hour for d in dates])
days = np.array([d.dayofweek for d in dates])

# Base pattern with morning spike and night dip
base_glucose = 120 + 30 * np.sin((hours - 6) * np.pi / 12)  # Daily pattern
weekend_boost = (days >= 5) * 10  # Slightly higher on weekends
noise = np.random.normal(0, 10, len(dates))  # Natural variability

glucose_values = base_glucose + weekend_boost + noise
glucose_values = np.clip(glucose_values, 70, 200)  # Keep in realistic range

# Create dataframe
cleaned_data = pd.DataFrame({
    'glucose': glucose_values
}, index=dates)

print(f"  Created {len(cleaned_data)} hours of synthetic data")
print(f"  Mean: {cleaned_data['glucose'].mean():.1f} mg/dL")
print(f"  Std: {cleaned_data['glucose'].std():.1f} mg/dL")

# Add features
print("\n2. Adding features...")
featured_data = cleaned_data.copy()
featured_data['hour'] = featured_data.index.hour
featured_data['day_of_week'] = featured_data.index.dayofweek
featured_data['day_of_month'] = featured_data.index.day
featured_data['month'] = featured_data.index.month
featured_data['is_weekend'] = (featured_data.index.dayofweek >= 5).astype(int)

# Calculate rolling features
featured_data['glucose_rolling_mean'] = featured_data['glucose'].rolling(window=24, min_periods=1).mean()
featured_data['glucose_rolling_std'] = featured_data['glucose'].rolling(window=24, min_periods=1).std()

print(f"  Added {len(featured_data.columns)} features")

# Train model
print("\n3. Training Random Forest model...")
predictor = GlucosePredictor(algorithm='random_forest')
predictor.train_ml_model(featured_data)

# Generate predictions WITH noise
print("\n4. Generating predictions WITH realistic variability...")
predictions_with = predictor.predict_future(
    periods=30 * 24,  # 30 days
    freq='H',
    add_noise=True,
    noise_scale=0.15
)

# Re-train for clean comparison
print("\n5. Re-training for comparison...")
predictor2 = GlucosePredictor(algorithm='random_forest')
predictor2.train_ml_model(featured_data)

# Generate predictions WITHOUT noise
print("\n6. Generating predictions WITHOUT variability...")
predictions_without = predictor2.predict_future(
    periods=30 * 24,
    freq='H',
    add_noise=False
)

# Get future predictions only
last_date = cleaned_data.index[-1]
future_with = predictions_with[predictions_with['timestamp'] > last_date]
future_without = predictions_without[predictions_without['timestamp'] > last_date]

# Analysis
print("\n" + "="*80)
print("COMPARISON RESULTS")
print("="*80)

print("\n📊 WITH Realistic Variability (NEW METHOD):")
print(f"  Mean: {future_with['predicted_glucose'].mean():.1f} mg/dL")
print(f"  Std Dev: {future_with['predicted_glucose'].std():.1f} mg/dL")
print(f"  Range: {future_with['predicted_glucose'].min():.1f} - {future_with['predicted_glucose'].max():.1f} mg/dL")

print("\n📉 WITHOUT Variability (OLD METHOD):")
print(f"  Mean: {future_without['predicted_glucose'].mean():.1f} mg/dL")
print(f"  Std Dev: {future_without['predicted_glucose'].std():.1f} mg/dL")
print(f"  Range: {future_without['predicted_glucose'].min():.1f} - {future_without['predicted_glucose'].max():.1f} mg/dL")

print("\n📈 Historical Data (for reference):")
print(f"  Mean: {cleaned_data['glucose'].mean():.1f} mg/dL")
print(f"  Std Dev: {cleaned_data['glucose'].std():.1f} mg/dL")
print(f"  Range: {cleaned_data['glucose'].min():.1f} - {cleaned_data['glucose'].max():.1f} mg/dL")

# Check for fixed patterns
print("\n" + "="*80)
print("PATTERN ANALYSIS - Testing for Fixed Patterns")
print("="*80)

# Sample Mondays at 9 AM
monday_9am_with = future_with[
    (future_with['timestamp'].dt.dayofweek == 0) &
    (future_with['timestamp'].dt.hour == 9)
]['predicted_glucose'].values

monday_9am_without = future_without[
    (future_without['timestamp'].dt.dayofweek == 0) &
    (future_without['timestamp'].dt.hour == 9)
]['predicted_glucose'].values

print(f"\n🔍 Checking Mondays at 9 AM (should vary week to week):")
print(f"\nWITH variability (first 4 weeks):")
for i, val in enumerate(monday_9am_with[:4], 1):
    print(f"  Week {i}: {val:.1f} mg/dL")
print(f"  Std Dev: {monday_9am_with.std():.2f} mg/dL")

print(f"\nWITHOUT variability (first 4 weeks):")
for i, val in enumerate(monday_9am_without[:4], 1):
    print(f"  Week {i}: {val:.1f} mg/dL")
print(f"  Std Dev: {monday_9am_without.std():.2f} mg/dL")

# Verdict
print("\n" + "="*80)
print("VERDICT")
print("="*80)

if monday_9am_without.std() < 0.5:
    print("\n❌ WITHOUT variability: FIXED PATTERN DETECTED!")
    print("   All Mondays at 9 AM have nearly identical values.")
    print("   This is unrealistic - real glucose varies even at same times.")
else:
    print("\n✓ WITHOUT variability: Some variation present")

if monday_9am_with.std() > 2.0:
    print("\n✅ WITH variability: REALISTIC VARIATION!")
    print("   Each week shows natural variation as expected in real life.")
    print("   Predictions are more realistic and life-like.")
else:
    print("\n⚠️  WITH variability: Limited variation")

# Hourly pattern comparison
print("\n" + "="*80)
print("HOURLY PATTERN DIVERSITY")
print("="*80)

# Count unique prediction values per hour (rounded)
with_hourly_diversity = []
without_hourly_diversity = []

for hour in range(24):
    hour_with = future_with[future_with['timestamp'].dt.hour == hour]['predicted_glucose'].round(1).nunique()
    hour_without = future_without[future_without['timestamp'].dt.hour == hour]['predicted_glucose'].round(1).nunique()
    with_hourly_diversity.append(hour_with)
    without_hourly_diversity.append(hour_without)

avg_with = np.mean(with_hourly_diversity)
avg_without = np.mean(without_hourly_diversity)

print(f"\nAverage unique values per hour:")
print(f"  WITH variability: {avg_with:.1f} different values")
print(f"  WITHOUT variability: {avg_without:.1f} different values")

if avg_with > avg_without * 2:
    print(f"\n✅ WITH variability shows {avg_with/avg_without:.1f}x more diversity!")
    print("   This means predictions vary more realistically across time.")

# Save results
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

future_with.to_csv('predictions_with_variability.csv', index=False)
future_without.to_csv('predictions_without_variability.csv', index=False)

print("\n✓ Saved:")
print("  - predictions_with_variability.csv")
print("  - predictions_without_variability.csv")

print("\n" + "="*80)
print("TEST COMPLETE!")
print("="*80)
print("\n💡 SUMMARY:")
print("   The NEW method adds realistic variability to avoid fixed patterns.")
print("   This makes predictions more like real-world glucose readings.")
print("   Each time period shows natural variation instead of repeating.")
print("\n   You can control the amount of variability with 'noise_scale':")
print("     - 0.10 = Low variability (10%)")
print("     - 0.15 = Medium variability (15%) - DEFAULT")
print("     - 0.20 = High variability (20%)")
print("="*80)
