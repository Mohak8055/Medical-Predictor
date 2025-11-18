"""
Test script for Medical Report Generator
"""

import pandas as pd
from datetime import datetime, timedelta
from medical_report import GlucoseReportGenerator

# Create sample prediction data
print("Creating sample prediction data...")

# Generate sample timestamps and glucose values
start_time = datetime.now()
timestamps = [start_time + timedelta(hours=i) for i in range(72)]  # 3 days of hourly predictions

# Sample glucose values with some variation
import numpy as np
np.random.seed(42)
glucose_values = 120 + 30 * np.sin(np.linspace(0, 6*np.pi, 72)) + np.random.normal(0, 10, 72)

# Create predictions DataFrame
predictions_df = pd.DataFrame({
    'ds': timestamps,
    'yhat': glucose_values,
    'yhat_lower': glucose_values - 15,
    'yhat_upper': glucose_values + 15
})

# Calculate statistics
statistics = {
    'avg_glucose': glucose_values.mean(),
    'min_glucose': glucose_values.min(),
    'max_glucose': glucose_values.max(),
    'std_glucose': glucose_values.std()
}

print(f"Sample data created: {len(predictions_df)} predictions")
print(f"Average glucose: {statistics['avg_glucose']:.1f} mg/dL")
print(f"Min: {statistics['min_glucose']:.1f} mg/dL")
print(f"Max: {statistics['max_glucose']:.1f} mg/dL")
print()

# Test PDF report generation
print("Generating PDF report...")
try:
    report_gen = GlucoseReportGenerator(
        patient_id=132,
        predictions_df=predictions_df,
        statistics=statistics
    )

    pdf_filename = "test_glucose_report.pdf"
    report_gen.generate_pdf_report(pdf_filename)
    print(f"[SUCCESS] PDF report generated successfully: {pdf_filename}")
except Exception as e:
    print(f"[ERROR] Error generating PDF report: {str(e)}")
    import traceback
    traceback.print_exc()

print()

# Test HTML report generation
print("Generating HTML report...")
try:
    report_gen = GlucoseReportGenerator(
        patient_id=132,
        predictions_df=predictions_df,
        statistics=statistics
    )

    html_filename = "test_glucose_report.html"
    report_gen.generate_html_report(html_filename)
    print(f"[SUCCESS] HTML report generated successfully: {html_filename}")
except Exception as e:
    print(f"[ERROR] Error generating HTML report: {str(e)}")
    import traceback
    traceback.print_exc()

print()
print("Test completed! Check the generated files:")
print(f"  - {pdf_filename}")
print(f"  - {html_filename}")
