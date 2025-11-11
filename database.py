"""
Database connection and data retrieval module
"""

import mysql.connector
import pandas as pd
from config import DB_CONFIG, PATIENT_ID


class GlucoseDataFetcher:
    """Handles database connections and data retrieval"""

    def __init__(self, patient_id=PATIENT_ID):
        self.patient_id = patient_id
        self.connection = None

    def connect(self):
        """Establish database connection"""
        try:
            self.connection = mysql.connector.connect(**DB_CONFIG)
            print(f"[OK] Connected to database successfully")
            return True
        except Exception as e:
            print(f"[X] Database connection failed: {e}")
            return False

    def disconnect(self):
        """Close database connection"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("[OK] Database connection closed")

    def fetch_patient_data(self):
        """Fetch all glucose readings for the patient"""
        if not self.connection or not self.connection.is_connected():
            if not self.connect():
                return None

        try:
            query = """
                SELECT id, timestamp, value, actual_time
                FROM glucose_readings
                WHERE patient_id = %s
                ORDER BY timestamp
            """

            df = pd.read_sql(query, self.connection, params=(self.patient_id,))

            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            if df['actual_time'].notna().any():
                df['actual_time'] = pd.to_datetime(df['actual_time'])

            print(f"[OK] Fetched {len(df)} records for patient {self.patient_id}")
            print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"  Glucose range: {df['value'].min():.1f} - {df['value'].max():.1f} mg/dL")

            return df

        except Exception as e:
            print(f"[X] Error fetching data: {e}")
            return None

    def get_data_statistics(self):
        """Get statistical summary of patient data"""
        if not self.connection or not self.connection.is_connected():
            if not self.connect():
                return None

        try:
            query = """
                SELECT
                    COUNT(*) as total_records,
                    MIN(timestamp) as earliest_date,
                    MAX(timestamp) as latest_date,
                    AVG(value) as avg_glucose,
                    MIN(value) as min_glucose,
                    MAX(value) as max_glucose,
                    STDDEV(value) as std_glucose
                FROM glucose_readings
                WHERE patient_id = %s
            """

            cursor = self.connection.cursor()
            cursor.execute(query, (self.patient_id,))
            stats = cursor.fetchone()
            cursor.close()

            return {
                'total_records': stats[0],
                'earliest_date': stats[1],
                'latest_date': stats[2],
                'avg_glucose': stats[3],
                'min_glucose': stats[4],
                'max_glucose': stats[5],
                'std_glucose': stats[6]
            }

        except Exception as e:
            print(f"[X] Error fetching statistics: {e}")
            return None
