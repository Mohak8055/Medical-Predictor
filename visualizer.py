"""
Visualization module for glucose data and predictions
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from config import FIGURE_SIZE, DPI

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = FIGURE_SIZE
plt.rcParams['figure.dpi'] = DPI


class GlucoseVisualizer:
    """Handles all visualization tasks"""

    def __init__(self):
        self.figures = []

    def plot_historical_data(self, df, title="Historical Glucose Readings", save_path=None):
        """Plot historical glucose data with anomalies highlighted"""
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)

        # Plot glucose values
        ax.plot(df.index, df['glucose'], color='#2E86AB', linewidth=1.5, label='Glucose Level')

        # Add target range
        ax.axhline(y=70, color='green', linestyle='--', alpha=0.3, label='Target Range')
        ax.axhline(y=180, color='green', linestyle='--', alpha=0.3)
        ax.fill_between(df.index, 70, 180, alpha=0.1, color='green')

        # Formatting
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Glucose (mg/dL)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
            print(f"[OK] Saved plot to {save_path}")

        self.figures.append(fig)
        return fig

    def plot_predictions(self, historical_df, predictions_df, title="Glucose Predictions", save_path=None):
        """Plot historical data with future predictions"""
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)

        # Split predictions into historical and future
        last_historical_date = historical_df.index[-1]
        future_mask = predictions_df['timestamp'] > last_historical_date

        historical_pred = predictions_df[~future_mask]
        future_pred = predictions_df[future_mask]

        # Plot historical actual data
        ax.plot(historical_df.index, historical_df['glucose'],
                color='#2E86AB', linewidth=2, label='Historical Actual', alpha=0.8)

        # Plot historical predictions (fitted values)
        if len(historical_pred) > 0:
            ax.plot(historical_pred['timestamp'], historical_pred['predicted_glucose'],
                    color='#A23B72', linewidth=1.5, label='Model Fit', alpha=0.6, linestyle='--')

        # Plot future predictions
        if len(future_pred) > 0:
            ax.plot(future_pred['timestamp'], future_pred['predicted_glucose'],
                    color='#F18F01', linewidth=2.5, label='Future Prediction', marker='o', markersize=3)

            # Plot confidence interval
            ax.fill_between(future_pred['timestamp'],
                           future_pred['lower_bound'],
                           future_pred['upper_bound'],
                           alpha=0.2, color='#F18F01', label='95% Confidence Interval')

        # Add target range
        ax.axhline(y=70, color='green', linestyle='--', alpha=0.3, linewidth=1)
        ax.axhline(y=180, color='green', linestyle='--', alpha=0.3, linewidth=1)
        ax.fill_between([historical_df.index[0], future_pred['timestamp'].iloc[-1] if len(future_pred) > 0 else historical_df.index[-1]],
                       70, 180, alpha=0.05, color='green', label='Target Range')

        # Formatting
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Glucose (mg/dL)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
            print(f"[OK] Saved plot to {save_path}")

        self.figures.append(fig)
        return fig

    def plot_anomalies(self, df, anomalies, title="Anomaly Detection", save_path=None):
        """Plot glucose data with anomalies highlighted"""
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)

        # Plot normal data
        ax.plot(df.index, df['glucose'], color='#2E86AB', linewidth=1.5, label='Normal Readings', alpha=0.8)

        # Highlight different types of anomalies
        if 'sensor_errors' in anomalies and len(anomalies['sensor_errors']) > 0:
            sensor_errors = anomalies['sensor_errors']
            ax.scatter(sensor_errors['timestamp'], sensor_errors['value'],
                      color='red', s=50, alpha=0.7, label='Sensor Errors', marker='x', linewidths=2)

        if 'sudden_changes' in anomalies and len(anomalies['sudden_changes']) > 0:
            sudden = anomalies['sudden_changes']
            # Use 'value' column for raw anomaly data
            value_col = 'glucose' if 'glucose' in sudden.columns else 'value'
            x_vals = sudden.index if 'timestamp' not in sudden.columns else sudden['timestamp']
            ax.scatter(x_vals, sudden[value_col],
                      color='orange', s=30, alpha=0.6, label='Sudden Changes', marker='^')

        if 'statistical_outliers' in anomalies and len(anomalies['statistical_outliers']) > 0:
            outliers = anomalies['statistical_outliers']
            ax.scatter(outliers['timestamp'], outliers['value'],
                      color='purple', s=40, alpha=0.5, label='Statistical Outliers', marker='s')

        # Add target range
        ax.axhline(y=70, color='green', linestyle='--', alpha=0.3)
        ax.axhline(y=180, color='green', linestyle='--', alpha=0.3)

        # Formatting
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Glucose (mg/dL)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
            print(f"[OK] Saved plot to {save_path}")

        self.figures.append(fig)
        return fig

    def plot_daily_patterns(self, df, title="Daily Glucose Patterns", save_path=None):
        """Plot average glucose by hour of day"""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Calculate hourly averages
        hourly_avg = df.groupby(df.index.hour)['glucose'].agg(['mean', 'std'])

        # Plot with error bars
        hours = hourly_avg.index
        ax.plot(hours, hourly_avg['mean'], color='#2E86AB', linewidth=2.5, marker='o', markersize=8, label='Average')
        ax.fill_between(hours,
                       hourly_avg['mean'] - hourly_avg['std'],
                       hourly_avg['mean'] + hourly_avg['std'],
                       alpha=0.3, color='#2E86AB', label='±1 Std Dev')

        # Add target range
        ax.axhline(y=70, color='green', linestyle='--', alpha=0.3, label='Target Range')
        ax.axhline(y=180, color='green', linestyle='--', alpha=0.3)

        # Formatting
        ax.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Glucose (mg/dL)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(range(0, 24))
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
            print(f"[OK] Saved plot to {save_path}")

        self.figures.append(fig)
        return fig

    def plot_weekly_patterns(self, df, title="Weekly Glucose Patterns", save_path=None):
        """Plot average glucose by day of week"""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Calculate daily averages
        daily_avg = df.groupby(df.index.dayofweek)['glucose'].agg(['mean', 'std'])

        # Plot
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        ax.bar(days, daily_avg['mean'], color='#2E86AB', alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.errorbar(days, daily_avg['mean'], yerr=daily_avg['std'],
                   fmt='none', ecolor='black', capsize=5, capthick=2, alpha=0.6)

        # Add target range
        ax.axhline(y=70, color='green', linestyle='--', alpha=0.3, label='Target Range')
        ax.axhline(y=180, color='green', linestyle='--', alpha=0.3)

        # Formatting
        ax.set_xlabel('Day of Week', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Glucose (mg/dL)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
            print(f"[OK] Saved plot to {save_path}")

        self.figures.append(fig)
        return fig

    def plot_distribution(self, df, title="Glucose Distribution", save_path=None):
        """Plot glucose distribution histogram and statistics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Histogram
        ax1.hist(df['glucose'], bins=50, color='#2E86AB', alpha=0.7, edgecolor='black')
        ax1.axvline(df['glucose'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["glucose"].mean():.1f}')
        ax1.axvline(df['glucose'].median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {df["glucose"].median():.1f}')

        # Add target range
        ax1.axvline(70, color='green', linestyle='--', alpha=0.5)
        ax1.axvline(180, color='green', linestyle='--', alpha=0.5)

        ax1.set_xlabel('Glucose (mg/dL)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax1.set_title('Distribution of Glucose Levels', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # Box plot
        bp = ax2.boxplot([df['glucose']], widths=0.6, patch_artist=True,
                         boxprops=dict(facecolor='#2E86AB', alpha=0.7),
                         medianprops=dict(color='red', linewidth=2),
                         whiskerprops=dict(linewidth=1.5),
                         capprops=dict(linewidth=1.5))

        # Add target range
        ax2.axhline(70, color='green', linestyle='--', alpha=0.5, label='Target Range')
        ax2.axhline(180, color='green', linestyle='--', alpha=0.5)

        ax2.set_ylabel('Glucose (mg/dL)', fontsize=12, fontweight='bold')
        ax2.set_title('Box Plot', fontsize=12, fontweight='bold')
        ax2.set_xticklabels(['Glucose'])
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
            print(f"[OK] Saved plot to {save_path}")

        self.figures.append(fig)
        return fig

    def create_dashboard(self, df, predictions, anomalies, save_path=None):
        """Create a comprehensive dashboard with multiple plots"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # 1. Historical data with predictions
        ax1 = fig.add_subplot(gs[0, :])
        last_historical = df.index[-1]
        future_mask = predictions['timestamp'] > last_historical
        future_pred = predictions[future_mask]

        ax1.plot(df.index, df['glucose'], color='#2E86AB', linewidth=1.5, label='Historical', alpha=0.8)
        if len(future_pred) > 0:
            ax1.plot(future_pred['timestamp'], future_pred['predicted_glucose'],
                    color='#F18F01', linewidth=2, label='Prediction', marker='o', markersize=3)
            ax1.fill_between(future_pred['timestamp'],
                           future_pred['lower_bound'], future_pred['upper_bound'],
                           alpha=0.2, color='#F18F01', label='95% CI')
        ax1.axhline(y=70, color='green', linestyle='--', alpha=0.3)
        ax1.axhline(y=180, color='green', linestyle='--', alpha=0.3)
        ax1.set_title('Glucose Levels & Predictions', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Glucose (mg/dL)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 2. Daily patterns
        ax2 = fig.add_subplot(gs[1, 0])
        hourly_avg = df.groupby(df.index.hour)['glucose'].mean()
        ax2.plot(hourly_avg.index, hourly_avg.values, color='#2E86AB', linewidth=2, marker='o')
        ax2.axhline(y=70, color='green', linestyle='--', alpha=0.3)
        ax2.axhline(y=180, color='green', linestyle='--', alpha=0.3)
        ax2.set_title('Daily Pattern (Hourly Average)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Avg Glucose (mg/dL)')
        ax2.grid(True, alpha=0.3)

        # 3. Distribution
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.hist(df['glucose'], bins=40, color='#2E86AB', alpha=0.7, edgecolor='black')
        ax3.axvline(df['glucose'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["glucose"].mean():.1f}')
        ax3.axvline(70, color='green', linestyle='--', alpha=0.5)
        ax3.axvline(180, color='green', linestyle='--', alpha=0.5)
        ax3.set_title('Glucose Distribution', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Glucose (mg/dL)')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. Weekly patterns
        ax4 = fig.add_subplot(gs[2, 0])
        daily_avg = df.groupby(df.index.dayofweek)['glucose'].mean()
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        ax4.bar(days, daily_avg.values, color='#2E86AB', alpha=0.7, edgecolor='black')
        ax4.axhline(y=70, color='green', linestyle='--', alpha=0.3)
        ax4.axhline(y=180, color='green', linestyle='--', alpha=0.3)
        ax4.set_title('Weekly Pattern', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Avg Glucose (mg/dL)')
        ax4.grid(True, alpha=0.3, axis='y')
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 5. Statistics summary
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis('off')

        stats_text = f"""
        GLUCOSE STATISTICS

        Total Readings: {len(df):,}
        Date Range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}

        Average: {df['glucose'].mean():.1f} mg/dL
        Median: {df['glucose'].median():.1f} mg/dL
        Std Dev: {df['glucose'].std():.1f} mg/dL

        Min: {df['glucose'].min():.1f} mg/dL
        Max: {df['glucose'].max():.1f} mg/dL

        Time in Range (70-180): {((df['glucose'] >= 70) & (df['glucose'] <= 180)).sum() / len(df) * 100:.1f}%
        Time Below 70: {(df['glucose'] < 70).sum() / len(df) * 100:.1f}%
        Time Above 180: {(df['glucose'] > 180).sum() / len(df) * 100:.1f}%

        Total Anomalies: {len(anomalies['all']) if 'all' in anomalies else 0}
        """

        ax5.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.suptitle('Glucose Monitoring Dashboard', fontsize=16, fontweight='bold', y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
            print(f"[OK] Saved dashboard to {save_path}")

        self.figures.append(fig)
        return fig

    def show_all(self):
        """Display all generated figures"""
        plt.show()

    def close_all(self):
        """Close all figures"""
        plt.close('all')
        self.figures = []
