import mysql.connector
import pandas as pd

# Database connection
db_config = {
    'host': 'revival365ai-db.chisukc6ague.ap-south-1.rds.amazonaws.com',
    'port': 3306,
    'database': 'revival',
    'user': 'admin',
    'password': 'MvqHf1QnpP1F1UqT57Pr'
}

try:
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    # Get table structure
    cursor.execute("DESCRIBE glucose_readings")
    columns = cursor.fetchall()
    print("Table Structure:")
    print("-" * 80)
    for col in columns:
        print(f"{col[0]:20} | {col[1]:20} | Null: {col[2]:5} | Key: {col[3]:5}")

    print("\n" + "="*80)

    # Get sample data for patient 132
    cursor.execute("""
        SELECT * FROM glucose_readings
        WHERE patient_id = 132
        ORDER BY timestamp
        LIMIT 10
    """)

    sample_data = cursor.fetchall()
    print("\nSample Data for Patient 132 (first 10 records):")
    print("-" * 80)
    for row in sample_data:
        print(row)

    print("\n" + "="*80)

    # Get data count and date range
    cursor.execute("""
        SELECT
            COUNT(*) as total_records,
            MIN(timestamp) as earliest_date,
            MAX(timestamp) as latest_date,
            AVG(value) as avg_glucose,
            MIN(value) as min_glucose,
            MAX(value) as max_glucose
        FROM glucose_readings
        WHERE patient_id = 132
    """)

    stats = cursor.fetchone()
    print("\nData Statistics for Patient 132:")
    print("-" * 80)
    print(f"Total Records: {stats[0]}")
    print(f"Earliest Date: {stats[1]}")
    print(f"Latest Date: {stats[2]}")
    print(f"Average Glucose: {stats[3]:.2f} mg/dL")
    print(f"Min Glucose: {stats[4]:.2f} mg/dL")
    print(f"Max Glucose: {stats[5]:.2f} mg/dL")

    cursor.close()
    conn.close()

except Exception as e:
    print(f"Error: {e}")
