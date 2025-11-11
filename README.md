# Glucose Prediction System

A comprehensive machine learning system with **interactive web interface** for analyzing and predicting glucose levels from continuous glucose monitoring (CGM) data.

## 🌟 New! Web Interface

Access the full system through an intuitive web interface:
- **Interactive patient selection** - Switch between patients easily
- **Real-time predictions** - Select any time period and get instant predictions
- **Visual dashboards** - See all metrics and charts in one place
- **Download data** - Export predictions and analysis as CSV files

## Features

- **Database Integration**: Direct connection to MySQL database for real-time data fetching
- **Data Preprocessing**:
  - Anomaly detection (sensor errors, statistical outliers, sudden changes)
  - Data cleaning and validation
  - Time series resampling and interpolation
- **Pattern Analysis**:
  - Daily and weekly glucose patterns
  - Trend analysis
  - Variability metrics
- **Prediction Models**:
  - Facebook Prophet for time series forecasting
  - Random Forest for feature-based predictions
  - 95% confidence intervals
- **Flexible Predictions**:
  - Predict for specific number of hours/days
  - Predict for custom date ranges
  - Predict up to one year ahead
- **Visualizations**:
  - Historical data plots
  - Prediction charts with confidence intervals
  - Anomaly detection visualizations
  - Daily/weekly pattern analysis
  - Comprehensive dashboards

## Installation

1. Install Python 3.12 or later

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 🚀 Quick Start - Web Interface (Recommended)

**Launch the web interface:**

```bash
streamlit run app.py
```

Or simply double-click `run_app.bat`

The interface will open in your browser at `http://localhost:8501`

**Features:**
- Select any patient ID
- Choose prediction time periods (24 hours to 1 year)
- View interactive charts and statistics
- Download predictions as CSV
- Switch between patients without restarting

See [USER_GUIDE.md](USER_GUIDE.md) for detailed instructions.

### Command Line Usage

Run the complete analysis:

```bash
py -3.12 main.py
```

This will:
1. Load data from the database
2. Detect and analyze anomalies
3. Train prediction models
4. Generate predictions for 7 days
5. Create visualizations
6. Save results to `./output/` directory

### Custom Usage

```python
from main import GlucosePredictionSystem

# Initialize system
system = GlucosePredictionSystem(patient_id=132)

# Load and preprocess data
system.load_data()
system.preprocess_data()

# Train models
system.train_model()

# Example 1: Predict next 24 hours
predictions_24h = system.predict(periods=24, freq='H')

# Example 2: Predict specific date range
predictions = system.predict_date_range(
    start_date='2025-11-10',
    end_date='2025-11-20'
)

# Example 3: Predict next year
predictions_year = system.predict(periods=365*24, freq='H')

# Generate visualizations
system.visualize_all(predictions=predictions_24h)

# Generate report
system.generate_report(predictions=predictions_24h)
```

## Project Structure

```
predictor/
├── config.py           # Configuration settings
├── database.py         # Database connection and data fetching
├── preprocessor.py     # Data preprocessing and anomaly detection
├── predictor.py        # Prediction models (Prophet & ML)
├── visualizer.py       # Visualization module
├── main.py            # Main application
├── requirements.txt   # Python dependencies
└── output/            # Generated visualizations and predictions
    ├── 00_dashboard.png
    ├── 01_historical_data.png
    ├── 02_predictions.png
    ├── 03_anomalies.png
    ├── 04_daily_patterns.png
    ├── 05_weekly_patterns.png
    ├── 06_distribution.png
    ├── predictions_custom_range.csv
    └── predictions_one_year.csv
```

## Configuration

Edit `config.py` to customize:

- Database connection settings
- Patient ID
- Anomaly detection thresholds
- Prediction intervals
- Visualization settings

## Prediction Model

The system uses Facebook Prophet, which:
- Handles missing data and outliers automatically
- Captures hourly and daily seasonality patterns
- Provides uncertainty intervals
- Accounts for trend changes
- Considers:
  - Previous glucose patterns
  - Rate of change
  - Gaps in data
  - Sudden spikes/drops
  - Anomalies and outliers

## Output

### Visualizations
- **Dashboard**: Comprehensive overview with all key metrics
- **Historical Data**: Clean glucose readings over time
- **Predictions**: Future glucose levels with confidence intervals
- **Anomalies**: Detected sensor errors, outliers, and sudden changes
- **Daily Patterns**: Average glucose by hour of day
- **Weekly Patterns**: Average glucose by day of week
- **Distribution**: Histogram and statistics

### CSV Files
- Predictions with timestamps and confidence bounds
- Can be imported into Excel or other tools for further analysis

## Performance Metrics

The system reports:
- **MAE** (Mean Absolute Error): Average prediction error in mg/dL
- **RMSE** (Root Mean Squared Error): Penalizes larger errors
- **MAPE** (Mean Absolute Percentage Error): Error as percentage

## Data Quality

The system handles:
- **Sensor Errors**: Values outside valid range (40-400 mg/dL)
- **Statistical Outliers**: Values beyond 3 standard deviations
- **Sudden Changes**: Glucose changes >5 mg/dL per minute
- **Data Gaps**: Missing data periods >15 minutes
- **Interpolation**: Fills gaps up to 24 hours

## Requirements

- Python 3.12+
- MySQL database access
- Dependencies listed in requirements.txt

## Notes

- Patient ID 132 has ~294,000 readings over 8 months
- Predictions account for all patterns and anomalies
- Model retrains automatically when run
- All visualizations saved as high-resolution PNG files

## Support

For issues or questions, refer to the code documentation or modify configurations as needed.
