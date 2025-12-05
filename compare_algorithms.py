"""
Compare all prediction algorithms side-by-side
Run this to see which algorithm works best for your data
"""

import pandas as pd
import numpy as np
from database import GlucoseDataFetcher
from preprocessor import GlucosePreprocessor
from predictor import GlucosePredictor
from datetime import datetime

# Configuration
PATIENT_ID = 132

# All available algorithms
ALGORITHMS = [
    'prophet',
    'linear_regression',
    'ridge',
    'lasso',
    'random_forest',
    'gradient_boosting',
    'xgboost',
    'svr',
    'knn',
    'decision_tree'
]

print("="*80)
print("ALGORITHM COMPARISON TOOL")
print("="*80)
print(f"\nPatient ID: {PATIENT_ID}")
print(f"Testing {len(ALGORITHMS)} algorithms")
print("\nThis will:")
print("  1. Load and preprocess data")
print("  2. Train each algorithm")
print("  3. Evaluate performance metrics")
print("  4. Compare results")
print("\n" + "="*80)

# Load and preprocess data
print("\n[1/4] Loading data...")
fetcher = GlucoseDataFetcher(PATIENT_ID)
raw_data = fetcher.fetch_patient_data()
fetcher.disconnect()

if raw_data is None or len(raw_data) == 0:
    print("Error: No data found!")
    exit(1)

print(f"  Loaded {len(raw_data):,} records")

print("\n[2/4] Preprocessing data...")
preprocessor = GlucosePreprocessor()
cleaned_data = preprocessor.clean_data(raw_data)
featured_data = preprocessor.add_features(cleaned_data)

print(f"  Cleaned data: {len(cleaned_data):,} hourly samples")
print(f"  Features: {len(featured_data.columns)}")

# Test each algorithm
print("\n[3/4] Testing algorithms...")
print("="*80)

results = []

for idx, algorithm in enumerate(ALGORITHMS, 1):
    print(f"\n[{idx}/{len(ALGORITHMS)}] Testing: {algorithm.upper()}")
    print("-" * 80)

    try:
        # Create predictor
        predictor = GlucosePredictor(algorithm=algorithm)

        # Train model
        if algorithm == 'prophet':
            predictor.train_prophet_model(cleaned_data)
            metrics = predictor.evaluate_model(cleaned_data, test_size=0.2)
        else:
            predictor.train_ml_model(featured_data)
            metrics = predictor.evaluate_model(featured_data, test_size=0.2)

        if metrics:
            results.append({
                'Algorithm': algorithm,
                'MAE': metrics['mae'],
                'RMSE': metrics['rmse'],
                'MAPE': metrics['mape'],
                'R²': metrics['r2'],
                'Correlation': metrics['correlation'],
                'Within ±20': metrics['within_20'],
                'Max Error': metrics['max_error'],
                'Median Error': metrics['median_error']
            })
            print(f"  ✓ {algorithm} completed successfully")
        else:
            print(f"  ✗ {algorithm} evaluation failed")

    except Exception as e:
        print(f"  ✗ {algorithm} failed: {str(e)}")
        continue

# Compare results
print("\n\n[4/4] Comparison Results")
print("="*80)

if len(results) == 0:
    print("No algorithms completed successfully!")
    exit(1)

# Create comparison dataframe
comparison_df = pd.DataFrame(results)

# Sort by MAE (lower is better)
comparison_df = comparison_df.sort_values('MAE')

print("\n📊 PERFORMANCE COMPARISON (sorted by MAE - lower is better)")
print("="*80)

# Display results with formatting
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:.2f}'.format)

print(comparison_df.to_string(index=False))

# Best algorithm
print("\n\n🏆 BEST ALGORITHM")
print("="*80)

best_algo = comparison_df.iloc[0]
print(f"Algorithm:       {best_algo['Algorithm'].upper()}")
print(f"MAE:             {best_algo['MAE']:.2f} mg/dL")
print(f"RMSE:            {best_algo['RMSE']:.2f} mg/dL")
print(f"R² Score:        {best_algo['R²']:.4f}")
print(f"Within ±20:      {best_algo['Within ±20']:.1f}%")

# Recommendations
print("\n\n💡 RECOMMENDATIONS")
print("="*80)

# Best for accuracy
best_mae = comparison_df.iloc[0]
print(f"\n✅ Best Overall Accuracy:")
print(f"   {best_mae['Algorithm'].upper()} (MAE: {best_mae['MAE']:.2f} mg/dL)")

# Best R² score
best_r2 = comparison_df.sort_values('R²', ascending=False).iloc[0]
print(f"\n✅ Best Model Fit:")
print(f"   {best_r2['Algorithm'].upper()} (R²: {best_r2['R²']:.4f})")

# Best within range
best_within = comparison_df.sort_values('Within ±20', ascending=False).iloc[0]
print(f"\n✅ Most Reliable (% within ±20 mg/dL):")
print(f"   {best_within['Algorithm'].upper()} ({best_within['Within ±20']:.1f}%)")

# Performance categories
print("\n\n📈 PERFORMANCE CATEGORIES")
print("="*80)

excellent = comparison_df[comparison_df['MAE'] < 15]
good = comparison_df[(comparison_df['MAE'] >= 15) & (comparison_df['MAE'] < 25)]
acceptable = comparison_df[(comparison_df['MAE'] >= 25) & (comparison_df['MAE'] < 35)]
poor = comparison_df[comparison_df['MAE'] >= 35]

if len(excellent) > 0:
    print(f"\n✅ Excellent (MAE < 15): {len(excellent)} algorithms")
    for _, row in excellent.iterrows():
        print(f"   - {row['Algorithm']}: {row['MAE']:.2f} mg/dL")

if len(good) > 0:
    print(f"\n✓  Good (MAE 15-25): {len(good)} algorithms")
    for _, row in good.iterrows():
        print(f"   - {row['Algorithm']}: {row['MAE']:.2f} mg/dL")

if len(acceptable) > 0:
    print(f"\n⚠️  Acceptable (MAE 25-35): {len(acceptable)} algorithms")
    for _, row in acceptable.iterrows():
        print(f"   - {row['Algorithm']}: {row['MAE']:.2f} mg/dL")

if len(poor) > 0:
    print(f"\n❌ Poor (MAE > 35): {len(poor)} algorithms")
    for _, row in poor.iterrows():
        print(f"   - {row['Algorithm']}: {row['MAE']:.2f} mg/dL")

# Save results
output_file = f"algorithm_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
comparison_df.to_csv(output_file, index=False)

print(f"\n\n💾 RESULTS SAVED")
print("="*80)
print(f"Saved to: {output_file}")

print("\n" + "="*80)
print("COMPARISON COMPLETE!")
print("="*80)
print(f"\nUse the {best_algo['Algorithm'].upper()} algorithm for best results!")
