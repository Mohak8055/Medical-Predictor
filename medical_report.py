"""
Medical Report Generator for Glucose Predictions
Generates professional medical-style reports with reference ranges
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import pandas as pd
from datetime import datetime
import io
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


class GlucoseReportGenerator:
    """Generate medical-style glucose prediction reports"""

    # Medical reference ranges for glucose levels (mg/dL)
    REFERENCE_RANGES = {
        'Fasting Glucose': {
            'Normal': (70, 99),
            'Prediabetes': (100, 125),
            'Diabetes': (126, 400)
        },
        'Post-Meal Glucose (2hr)': {
            'Normal': (70, 139),
            'Prediabetes': (140, 199),
            'Diabetes': (200, 400)
        },
        'Random Glucose': {
            'Normal': (70, 140),
            'Elevated': (141, 199),
            'High': (200, 400)
        },
        'Average Glucose': {
            'Excellent': (70, 120),
            'Good': (121, 154),
            'Fair': (155, 180),
            'Poor': (181, 400)
        },
        'HbA1c Equivalent': {
            'Normal': (0, 5.6),
            'Prediabetes': (5.7, 6.4),
            'Diabetes': (6.5, 15)
        }
    }

    def __init__(self, patient_id, predictions_df, statistics=None):
        """
        Initialize report generator

        Args:
            patient_id: Patient identifier
            predictions_df: DataFrame with predictions (ds, yhat, yhat_lower, yhat_upper)
            statistics: Dict with summary statistics
        """
        self.patient_id = patient_id
        self.predictions_df = predictions_df
        self.statistics = statistics or {}
        self.report_date = datetime.now()

    def _get_status_color(self, value, metric_type='Random Glucose'):
        """Get color based on glucose value and reference ranges"""
        ranges = self.REFERENCE_RANGES.get(metric_type, self.REFERENCE_RANGES['Random Glucose'])

        for status, (min_val, max_val) in ranges.items():
            if min_val <= value <= max_val:
                if 'Normal' in status or 'Excellent' in status:
                    return colors.green, status
                elif 'Good' in status or 'Fair' in status:
                    return colors.orange, status
                else:
                    return colors.red, status

        return colors.red, 'Critical'

    def _get_status_for_range(self, value, metric_type='Random Glucose'):
        """Get status text for a value based on reference ranges"""
        ranges = self.REFERENCE_RANGES.get(metric_type, self.REFERENCE_RANGES['Random Glucose'])

        for status, (min_val, max_val) in ranges.items():
            if min_val <= value <= max_val:
                return status

        if value < list(ranges.values())[0][0]:
            return 'Low'
        return 'Critical High'

    def _calculate_metrics(self):
        """Calculate key metrics from predictions"""
        if 'yhat' not in self.predictions_df.columns:
            return {}

        glucose_values = self.predictions_df['yhat'].values

        metrics = {
            'Average Glucose': glucose_values.mean(),
            'Minimum Predicted': glucose_values.min(),
            'Maximum Predicted': glucose_values.max(),
            'Standard Deviation': glucose_values.std(),
            'Time in Range (70-180)': ((glucose_values >= 70) & (glucose_values <= 180)).sum() / len(glucose_values) * 100,
            'Time Below Range (<70)': (glucose_values < 70).sum() / len(glucose_values) * 100,
            'Time Above Range (>180)': (glucose_values > 180).sum() / len(glucose_values) * 100,
            'Estimated HbA1c': (glucose_values.mean() + 46.7) / 28.7  # ADAG formula
        }

        # Add first and last prediction values
        if len(glucose_values) > 0:
            metrics['First Prediction'] = glucose_values[0]
            metrics['Last Prediction'] = glucose_values[-1]

        return metrics

    def _create_mini_chart(self):
        """Create a small chart of predictions for the report"""
        fig, ax = plt.subplots(figsize=(6, 2.5))

        if 'ds' in self.predictions_df.columns and 'yhat' in self.predictions_df.columns:
            df = self.predictions_df.copy()
            df['ds'] = pd.to_datetime(df['ds'])

            # Plot predictions
            ax.plot(df['ds'], df['yhat'], color='#2E86AB', linewidth=2, label='Predicted')

            # Plot confidence intervals if available
            if 'yhat_lower' in df.columns and 'yhat_upper' in df.columns:
                ax.fill_between(df['ds'], df['yhat_lower'], df['yhat_upper'],
                               alpha=0.3, color='#A6CEE3', label='95% CI')

            # Reference lines
            ax.axhline(y=70, color='red', linestyle='--', alpha=0.5, linewidth=1)
            ax.axhline(y=180, color='red', linestyle='--', alpha=0.5, linewidth=1)
            ax.fill_between(df['ds'], 70, 180, alpha=0.1, color='green')

            ax.set_xlabel('Date/Time', fontsize=9)
            ax.set_ylabel('Glucose (mg/dL)', fontsize=9)
            ax.set_title('Predicted Glucose Levels', fontsize=10, fontweight='bold')
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)

            # Rotate x-axis labels
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.yticks(fontsize=8)

            plt.tight_layout()

        # Save to bytes buffer
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()

        return img_buffer

    def generate_pdf_report(self, filename):
        """
        Generate a professional PDF medical report

        Args:
            filename: Output PDF filename
        """
        doc = SimpleDocTemplate(filename, pagesize=letter,
                              rightMargin=0.75*inch, leftMargin=0.75*inch,
                              topMargin=0.75*inch, bottomMargin=0.75*inch)

        # Container for the 'Flowable' objects
        elements = []

        # Define styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=colors.HexColor('#1E3A8A'),
            spaceAfter=12,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )

        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=12,
            textColor=colors.HexColor('#1E3A8A'),
            spaceAfter=6,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        )

        normal_style = styles['Normal']
        normal_style.fontSize = 10

        # Header
        elements.append(Paragraph("GLUCOSE MONITORING REPORT", title_style))
        elements.append(Paragraph("Continuous Glucose Prediction Analysis",
                                 ParagraphStyle('subtitle', parent=normal_style,
                                              fontSize=11, alignment=TA_CENTER,
                                              textColor=colors.grey)))
        elements.append(Spacer(1, 0.3*inch))

        # Patient Information
        elements.append(Paragraph("PATIENT INFORMATION", heading_style))

        patient_data = [
            ['Patient ID:', str(self.patient_id), 'Report Date:', self.report_date.strftime('%Y-%m-%d %H:%M')],
            ['Report Type:', 'Predictive Analysis', 'Prediction Period:',
             f"{len(self.predictions_df)} hours" if len(self.predictions_df) < 168 else f"{len(self.predictions_df)//24} days"]
        ]

        patient_table = Table(patient_data, colWidths=[1.2*inch, 2*inch, 1.2*inch, 2*inch])
        patient_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#E5E7EB')),
            ('BACKGROUND', (2, 0), (2, -1), colors.HexColor('#E5E7EB')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        elements.append(patient_table)
        elements.append(Spacer(1, 0.2*inch))

        # Key Metrics
        elements.append(Paragraph("GLUCOSE ANALYSIS SUMMARY", heading_style))

        metrics = self._calculate_metrics()

        metrics_data = [
            ['Test Parameter', 'Value', 'Unit', 'Reference Range', 'Status'],
        ]

        # Add metrics with reference ranges
        test_params = [
            ('Average Glucose', metrics.get('Average Glucose', 0), 'mg/dL', 'Average Glucose'),
            ('Estimated HbA1c', metrics.get('Estimated HbA1c', 0), '%', 'HbA1c Equivalent'),
            ('Minimum Predicted', metrics.get('Minimum Predicted', 0), 'mg/dL', 'Random Glucose'),
            ('Maximum Predicted', metrics.get('Maximum Predicted', 0), 'mg/dL', 'Random Glucose'),
        ]

        for param_name, value, unit, ref_type in test_params:
            if value > 0:
                status = self._get_status_for_range(value, ref_type)
                ranges = self.REFERENCE_RANGES[ref_type]
                # Create shorter, wrapped reference range text
                range_lines = [f"{k}: {v[0]}-{v[1]}" for k, v in ranges.items()]
                # Use Paragraph for wrapping
                range_text = Paragraph('<br/>'.join(range_lines),
                                      ParagraphStyle('RangeText', parent=normal_style, fontSize=7))

                if unit == '%':
                    value_str = f"{value:.1f}"
                else:
                    value_str = f"{value:.0f}"

                metrics_data.append([param_name, value_str, unit, range_text, status])

        # Add Time in Range metrics
        metrics_data.append(['Time in Range (70-180)', f"{metrics.get('Time in Range (70-180)', 0):.1f}", '%',
                           '>70% recommended',
                           'Good' if metrics.get('Time in Range (70-180)', 0) > 70 else 'Needs Improvement'])
        metrics_data.append(['Time Below Range (<70)', f"{metrics.get('Time Below Range (<70)', 0):.1f}", '%',
                           '<4% recommended',
                           'Good' if metrics.get('Time Below Range (<70)', 0) < 4 else 'High'])
        metrics_data.append(['Time Above Range (>180)', f"{metrics.get('Time Above Range (>180)', 0):.1f}", '%',
                           '<25% recommended',
                           'Good' if metrics.get('Time Above Range (>180)', 0) < 25 else 'High'])

        metrics_table = Table(metrics_data, colWidths=[1.6*inch, 0.7*inch, 0.5*inch, 2.5*inch, 1.3*inch])

        # Style the metrics table
        table_style = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E3A8A')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('WORDWRAP', (0, 0), (-1, -1), True),
        ]

        # Color code status column
        for i in range(1, len(metrics_data)):
            status = metrics_data[i][4]
            if 'Good' in status or 'Normal' in status or 'Excellent' in status:
                table_style.append(('BACKGROUND', (4, i), (4, i), colors.lightgreen))
            elif 'Fair' in status or 'Prediabetes' in status:
                table_style.append(('BACKGROUND', (4, i), (4, i), colors.lightyellow))
            else:
                table_style.append(('BACKGROUND', (4, i), (4, i), colors.lightpink))

        metrics_table.setStyle(TableStyle(table_style))
        elements.append(metrics_table)
        elements.append(Spacer(1, 0.3*inch))

        # Add prediction chart
        elements.append(Paragraph("GLUCOSE PREDICTION TREND", heading_style))
        chart_buffer = self._create_mini_chart()
        chart_img = Image(chart_buffer, width=5.5*inch, height=2.3*inch)
        elements.append(chart_img)
        elements.append(Spacer(1, 0.2*inch))

        # Reference Ranges Legend
        elements.append(Paragraph("CLINICAL REFERENCE RANGES", heading_style))

        ref_text = """
        <b>Glucose Levels (mg/dL):</b><br/>
        • <b>Normal Fasting:</b> 70-99 mg/dL<br/>
        • <b>Prediabetes Fasting:</b> 100-125 mg/dL<br/>
        • <b>Diabetes Fasting:</b> ≥126 mg/dL<br/>
        • <b>Target Range (General):</b> 70-180 mg/dL<br/>
        <br/>
        <b>HbA1c (%):</b><br/>
        • <b>Normal:</b> Below 5.7%<br/>
        • <b>Prediabetes:</b> 5.7-6.4%<br/>
        • <b>Diabetes:</b> ≥6.5%<br/>
        <br/>
        <b>Time in Range Goals:</b><br/>
        • <b>In Range (70-180 mg/dL):</b> >70%<br/>
        • <b>Below Range (<70 mg/dL):</b> <4%<br/>
        • <b>Above Range (>180 mg/dL):</b> <25%
        """

        elements.append(Paragraph(ref_text, normal_style))
        elements.append(Spacer(1, 0.2*inch))

        # Disclaimer
        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=normal_style,
            fontSize=8,
            textColor=colors.grey,
            leftIndent=0.2*inch,
            rightIndent=0.2*inch
        )

        disclaimer = """
        <b>DISCLAIMER:</b> This report is generated by a predictive algorithm and is intended for
        informational purposes only. It should not be used as a substitute for professional medical
        advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified
        health provider with any questions you may have regarding a medical condition. The predictions
        are based on historical patterns and may not account for all individual factors affecting
        glucose levels.
        """

        elements.append(Paragraph(disclaimer, disclaimer_style))

        # Build PDF
        doc.build(elements)

        return filename

    def generate_html_report(self, filename):
        """Generate an HTML version of the report"""
        metrics = self._calculate_metrics()

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Glucose Monitoring Report - Patient {self.patient_id}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    max-width: 900px;
                    margin: 0 auto;
                }}
                h1 {{
                    color: #1E3A8A;
                    text-align: center;
                    border-bottom: 3px solid #1E3A8A;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #1E3A8A;
                    margin-top: 30px;
                    border-left: 4px solid #1E3A8A;
                    padding-left: 10px;
                }}
                .patient-info {{
                    background-color: #E5E7EB;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 20px 0;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th {{
                    background-color: #1E3A8A;
                    color: white;
                    padding: 12px;
                    text-align: left;
                }}
                td {{
                    padding: 10px;
                    border: 1px solid #ddd;
                }}
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                .status-normal {{
                    background-color: #90EE90;
                    padding: 5px 10px;
                    border-radius: 3px;
                    font-weight: bold;
                }}
                .status-warning {{
                    background-color: #FFD700;
                    padding: 5px 10px;
                    border-radius: 3px;
                    font-weight: bold;
                }}
                .status-high {{
                    background-color: #FFB6C1;
                    padding: 5px 10px;
                    border-radius: 3px;
                    font-weight: bold;
                }}
                .disclaimer {{
                    background-color: #FFF3CD;
                    border: 1px solid #FFC107;
                    padding: 15px;
                    border-radius: 5px;
                    margin-top: 30px;
                    font-size: 12px;
                    color: #856404;
                }}
                .reference-ranges {{
                    background-color: #E3F2FD;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 20px 0;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>GLUCOSE MONITORING REPORT</h1>
                <p style="text-align: center; color: #666;">Continuous Glucose Prediction Analysis</p>

                <div class="patient-info">
                    <h2>Patient Information</h2>
                    <table>
                        <tr>
                            <td><strong>Patient ID:</strong></td>
                            <td>{self.patient_id}</td>
                            <td><strong>Report Date:</strong></td>
                            <td>{self.report_date.strftime('%Y-%m-%d %H:%M')}</td>
                        </tr>
                        <tr>
                            <td><strong>Report Type:</strong></td>
                            <td>Predictive Analysis</td>
                            <td><strong>Prediction Period:</strong></td>
                            <td>{len(self.predictions_df)} hours ({len(self.predictions_df)//24} days)</td>
                        </tr>
                    </table>
                </div>

                <h2>Glucose Analysis Summary</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Test Parameter</th>
                            <th>Value</th>
                            <th>Unit</th>
                            <th>Reference Range</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
        """

        # Add metrics
        test_params = [
            ('Average Glucose', metrics.get('Average Glucose', 0), 'mg/dL', 'Average Glucose'),
            ('Estimated HbA1c', metrics.get('Estimated HbA1c', 0), '%', 'HbA1c Equivalent'),
            ('Minimum Predicted', metrics.get('Minimum Predicted', 0), 'mg/dL', 'Random Glucose'),
            ('Maximum Predicted', metrics.get('Maximum Predicted', 0), 'mg/dL', 'Random Glucose'),
        ]

        for param_name, value, unit, ref_type in test_params:
            if value > 0:
                status = self._get_status_for_range(value, ref_type)
                ranges = self.REFERENCE_RANGES[ref_type]
                range_text = ', '.join([f"{k}: {v[0]}-{v[1]}" for k, v in ranges.items()])

                status_class = 'status-normal' if 'Normal' in status or 'Good' in status or 'Excellent' in status else ('status-warning' if 'Fair' in status or 'Prediabetes' in status else 'status-high')

                if unit == '%':
                    value_str = f"{value:.1f}"
                else:
                    value_str = f"{value:.0f}"

                html_content += f"""
                        <tr>
                            <td>{param_name}</td>
                            <td>{value_str}</td>
                            <td>{unit}</td>
                            <td>{range_text}</td>
                            <td><span class="{status_class}">{status}</span></td>
                        </tr>
                """

        # Add Time in Range metrics
        tir_status = 'status-normal' if metrics.get('Time in Range (70-180)', 0) > 70 else 'status-high'
        tbr_status = 'status-normal' if metrics.get('Time Below Range (<70)', 0) < 4 else 'status-high'
        tar_status = 'status-normal' if metrics.get('Time Above Range (>180)', 0) < 25 else 'status-high'

        html_content += f"""
                        <tr>
                            <td>Time in Range (70-180)</td>
                            <td>{metrics.get('Time in Range (70-180)', 0):.1f}</td>
                            <td>%</td>
                            <td>>70% recommended</td>
                            <td><span class="{tir_status}">{'Good' if metrics.get('Time in Range (70-180)', 0) > 70 else 'Needs Improvement'}</span></td>
                        </tr>
                        <tr>
                            <td>Time Below Range (&lt;70)</td>
                            <td>{metrics.get('Time Below Range (<70)', 0):.1f}</td>
                            <td>%</td>
                            <td>&lt;4% recommended</td>
                            <td><span class="{tbr_status}">{'Good' if metrics.get('Time Below Range (<70)', 0) < 4 else 'High'}</span></td>
                        </tr>
                        <tr>
                            <td>Time Above Range (&gt;180)</td>
                            <td>{metrics.get('Time Above Range (>180)', 0):.1f}</td>
                            <td>%</td>
                            <td>&lt;25% recommended</td>
                            <td><span class="{tar_status}">{'Good' if metrics.get('Time Above Range (>180)', 0) < 25 else 'High'}</span></td>
                        </tr>
                    </tbody>
                </table>

                <div class="reference-ranges">
                    <h2>Clinical Reference Ranges</h2>
                    <p><strong>Glucose Levels (mg/dL):</strong></p>
                    <ul>
                        <li><strong>Normal Fasting:</strong> 70-99 mg/dL</li>
                        <li><strong>Prediabetes Fasting:</strong> 100-125 mg/dL</li>
                        <li><strong>Diabetes Fasting:</strong> ≥126 mg/dL</li>
                        <li><strong>Target Range (General):</strong> 70-180 mg/dL</li>
                    </ul>

                    <p><strong>HbA1c (%):</strong></p>
                    <ul>
                        <li><strong>Normal:</strong> Below 5.7%</li>
                        <li><strong>Prediabetes:</strong> 5.7-6.4%</li>
                        <li><strong>Diabetes:</strong> ≥6.5%</li>
                    </ul>

                    <p><strong>Time in Range Goals:</strong></p>
                    <ul>
                        <li><strong>In Range (70-180 mg/dL):</strong> >70%</li>
                        <li><strong>Below Range (&lt;70 mg/dL):</strong> &lt;4%</li>
                        <li><strong>Above Range (&gt;180 mg/dL):</strong> &lt;25%</li>
                    </ul>
                </div>

                <div class="disclaimer">
                    <strong>DISCLAIMER:</strong> This report is generated by a predictive algorithm and is intended for
                    informational purposes only. It should not be used as a substitute for professional medical
                    advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified
                    health provider with any questions you may have regarding a medical condition. The predictions
                    are based on historical patterns and may not account for all individual factors affecting
                    glucose levels.
                </div>
            </div>
        </body>
        </html>
        """

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return filename
