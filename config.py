"""
Configuration file for database connection and model parameters
"""

# Database Configuration
DB_CONFIG = {
    'host': 'revival365ai-db.chisukc6ague.ap-south-1.rds.amazonaws.com',
    'port': 3306,
    'database': 'revival',
    'user': 'admin',
    'password': 'MvqHf1QnpP1F1UqT57Pr'
}

# Patient Configuration
PATIENT_ID = 132

# Model Parameters
PREDICTION_INTERVAL = 'H'  # Hourly predictions (can be 'H', 'D', 'W')
CONFIDENCE_INTERVAL = 0.95  # 95% confidence interval for predictions

# Anomaly Detection Parameters
ANOMALY_THRESHOLD_STD = 3  # Standard deviations for outlier detection
MIN_VALID_GLUCOSE = 40  # mg/dL - values below are likely sensor errors
MAX_VALID_GLUCOSE = 400  # mg/dL - values above are likely sensor errors

# Visualization Parameters
FIGURE_SIZE = (15, 8)
DPI = 100
