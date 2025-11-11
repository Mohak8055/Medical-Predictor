"""
Custom query examples for glucose predictions

This script demonstrates how to query predictions for different scenarios
"""

from main import GlucosePredictionSystem
from datetime import datetime, timedelta
import pandas as pd


def example_1_predict_next_week():
    """Example 1: Predict glucose for the next 7 days"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Predict Next 7 Days")
    print("="*80)

    system = GlucosePredictionSystem(patient_id=132)

    # Load, preprocess, and train
    system.load_data()
    system.preprocess_data()
    system.train_model()

    # Predict next 7 days (168 hours)
    predictions = system.predict(periods=168, freq='H')

    # Get only future predictions
    last_date = system.cleaned_data.index[-1]
    future = predictions[predictions['timestamp'] > last_date]

    print(f"\n📊 Next 7 Days Predictions:")
    print(future.head(24))  # Show first 24 hours

    # Save to CSV
    future.to_csv('./output/next_7_days.csv', index=False)
    print(f"\n✓ Saved to ./output/next_7_days.csv")

    system.cleanup()
    return future


def example_2_predict_specific_dates():
    """Example 2: Predict for specific date range"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Predict for Specific Date Range")
    print("="*80)

    system = GlucosePredictionSystem(patient_id=132)

    # Load, preprocess, and train
    system.load_data()
    system.preprocess_data()
    system.train_model()

    # Predict for December 2025
    predictions = system.predict_date_range(
        start_date='2025-12-01',
        end_date='2025-12-31'
    )

    print(f"\n📊 December 2025 Predictions:")
    print(f"Total predictions: {len(predictions)}")
    print(f"\nSample predictions:")
    print(predictions.head(10))

    # Daily averages for December
    predictions_copy = predictions.copy()
    predictions_copy['date'] = pd.to_datetime(predictions_copy['timestamp']).dt.date
    daily_avg = predictions_copy.groupby('date')['predicted_glucose'].mean()

    print(f"\n📅 Daily Averages:")
    print(daily_avg)

    # Save to CSV
    predictions.to_csv('./output/december_2025.csv', index=False)
    print(f"\n✓ Saved to ./output/december_2025.csv")

    system.cleanup()
    return predictions


def example_3_predict_next_year():
    """Example 3: Predict for entire next year"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Predict Next Year")
    print("="*80)

    system = GlucosePredictionSystem(patient_id=132)

    # Load, preprocess, and train
    system.load_data()
    system.preprocess_data()
    system.train_model()

    # Predict next year (365 days)
    predictions = system.predict(periods=365*24, freq='H')

    # Get only future predictions
    last_date = system.cleaned_data.index[-1]
    future = predictions[predictions['timestamp'] > last_date]

    print(f"\n📊 Next Year Predictions:")
    print(f"Total predictions: {len(future):,}")
    print(f"Date range: {future['timestamp'].min()} to {future['timestamp'].max()}")

    # Monthly statistics
    future_copy = future.copy()
    future_copy['month'] = pd.to_datetime(future_copy['timestamp']).dt.to_period('M')
    monthly_stats = future_copy.groupby('month')['predicted_glucose'].agg([
        ('avg', 'mean'),
        ('min', 'min'),
        ('max', 'max'),
        ('std', 'std')
    ])

    print(f"\n📅 Monthly Statistics:")
    print(monthly_stats)

    # Save to CSV
    future.to_csv('./output/next_year_predictions.csv', index=False)
    monthly_stats.to_csv('./output/monthly_summary.csv')
    print(f"\n✓ Saved to ./output/next_year_predictions.csv")
    print(f"✓ Saved monthly summary to ./output/monthly_summary.csv")

    system.cleanup()
    return future


def example_4_predict_with_analysis():
    """Example 4: Predict with detailed analysis"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Predict with Detailed Analysis")
    print("="*80)

    system = GlucosePredictionSystem(patient_id=132)

    # Load, preprocess, and train
    system.load_data()
    system.preprocess_data()
    system.train_model()

    # Predict next 30 days
    predictions = system.predict(periods=30*24, freq='H')

    # Get future predictions
    last_date = system.cleaned_data.index[-1]
    future = predictions[predictions['timestamp'] > last_date]

    # Analyze predictions
    print(f"\n📊 Detailed Analysis:")

    # Time in range
    in_range = ((future['predicted_glucose'] >= 70) & (future['predicted_glucose'] <= 180)).sum()
    below_range = (future['predicted_glucose'] < 70).sum()
    above_range = (future['predicted_glucose'] > 180).sum()
    total = len(future)

    print(f"\n🎯 Predicted Time in Range:")
    print(f"  In range (70-180): {in_range/total*100:.1f}% ({in_range:,} hours)")
    print(f"  Below range (<70): {below_range/total*100:.1f}% ({below_range:,} hours)")
    print(f"  Above range (>180): {above_range/total*100:.1f}% ({above_range:,} hours)")

    # Weekly patterns in predictions
    future_copy = future.copy()
    future_copy['day_of_week'] = pd.to_datetime(future_copy['timestamp']).dt.dayofweek
    future_copy['hour'] = pd.to_datetime(future_copy['timestamp']).dt.hour

    weekly_avg = future_copy.groupby('day_of_week')['predicted_glucose'].mean()
    hourly_avg = future_copy.groupby('hour')['predicted_glucose'].mean()

    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    print(f"\n📅 Predicted Weekly Pattern:")
    for i, day in enumerate(days):
        print(f"  {day}: {weekly_avg.iloc[i]:.1f} mg/dL")

    print(f"\n🕐 Predicted Daily Pattern (average by hour):")
    print(f"  Peak hour: {hourly_avg.idxmax()}:00 ({hourly_avg.max():.1f} mg/dL)")
    print(f"  Lowest hour: {hourly_avg.idxmin()}:00 ({hourly_avg.min():.1f} mg/dL)")

    # Generate visualizations
    system.visualize_all(predictions=predictions, show=False)

    system.cleanup()
    return future


def example_5_compare_periods():
    """Example 5: Compare predictions for different time periods"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Compare Different Time Periods")
    print("="*80)

    system = GlucosePredictionSystem(patient_id=132)

    # Load, preprocess, and train
    system.load_data()
    system.preprocess_data()
    system.train_model()

    # Predict different periods
    periods = {
        '1 Week': 7*24,
        '1 Month': 30*24,
        '3 Months': 90*24,
        '6 Months': 180*24,
        '1 Year': 365*24
    }

    results = {}

    print(f"\n📊 Comparison of Different Prediction Periods:")
    print(f"\n{'Period':<15} {'Avg (mg/dL)':<12} {'Min (mg/dL)':<12} {'Max (mg/dL)':<12} {'Std Dev':<10}")
    print("-" * 65)

    for period_name, hours in periods.items():
        predictions = system.predict(periods=hours, freq='H')
        last_date = system.cleaned_data.index[-1]
        future = predictions[predictions['timestamp'] > last_date]

        avg = future['predicted_glucose'].mean()
        min_val = future['predicted_glucose'].min()
        max_val = future['predicted_glucose'].max()
        std = future['predicted_glucose'].std()

        results[period_name] = {
            'predictions': future,
            'avg': avg,
            'min': min_val,
            'max': max_val,
            'std': std
        }

        print(f"{period_name:<15} {avg:<12.1f} {min_val:<12.1f} {max_val:<12.1f} {std:<10.2f}")

    system.cleanup()
    return results


if __name__ == "__main__":
    print("""
    ================================================================

              CUSTOM QUERY EXAMPLES
              Glucose Prediction System

    ================================================================
    """)

    # Run examples
    print("\nSelect which example to run:")
    print("1. Predict next 7 days")
    print("2. Predict specific date range (December 2025)")
    print("3. Predict next year with monthly summary")
    print("4. Predict with detailed analysis")
    print("5. Compare different time periods")
    print("6. Run all examples")

    choice = input("\nEnter choice (1-6): ").strip()

    if choice == '1':
        example_1_predict_next_week()
    elif choice == '2':
        example_2_predict_specific_dates()
    elif choice == '3':
        example_3_predict_next_year()
    elif choice == '4':
        example_4_predict_with_analysis()
    elif choice == '5':
        example_5_compare_periods()
    elif choice == '6':
        example_1_predict_next_week()
        example_2_predict_specific_dates()
        example_3_predict_next_year()
        example_4_predict_with_analysis()
        example_5_compare_periods()
    else:
        print("Invalid choice. Running Example 1 by default.")
        example_1_predict_next_week()

    print("\n" + "="*80)
    print("✓ EXAMPLES COMPLETE!")
    print("="*80)
