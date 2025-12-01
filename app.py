"""
Streamlit Web Interface for Glucose Prediction System
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from database import GlucoseDataFetcher
from preprocessor import GlucosePreprocessor
from predictor import GlucosePredictor
from config import DB_CONFIG
from medical_report import GlucoseReportGenerator

# Page configuration
st.set_page_config(
    page_title="Glucose Prediction System",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E86AB;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'system_ready' not in st.session_state:
    st.session_state.system_ready = False
    st.session_state.trained = False
    st.session_state.raw_data = None
    st.session_state.cleaned_data = None
    st.session_state.anomalies = None
    st.session_state.predictions = None
    st.session_state.patient_id = 132
    st.session_state.selected_algorithm = 'prophet'

# Header
st.markdown('<p class="main-header">🩸 Glucose Prediction System</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("🤖 Algorithm Selection")

    # Get all algorithms from predictor
    temp_predictor = GlucosePredictor()
    all_algorithms = temp_predictor.get_all_algorithms()

    # Create algorithm options
    algorithm_options = {
        'prophet': '📈 Facebook Prophet',
        'linear_regression': '📉 Linear Regression',
        'lstm': '🧠 LSTM',    # <--- NEW
        'gru': '⚡ GRU',      # new
        'hist_gradient_boosting': '📊 Hist. Gradient Boosting',
        'catboost': '🐱 CatBoost',
        'adaboost': '📈 AdaBoost',
        'ridge': '🔷 Ridge Regression',
        'lightgbm': '🚀 LightGBM',
        'lasso': '🔶 Lasso Regression',
        'random_forest': '🌲 Random Forest',
        'gradient_boosting': '⚡ Gradient Boosting',
        'xgboost': '🚀 XGBoost',
        'svr': '🎯 Support Vector Regression',
        'knn': '🔍 K-Nearest Neighbors',
        'decision_tree': '🌳 Decision Tree'
    }

    selected_algo = st.selectbox(
        "Select Prediction Algorithm",
        options=list(algorithm_options.keys()),
        format_func=lambda x: algorithm_options[x],
        index=list(algorithm_options.keys()).index(st.session_state.selected_algorithm),
        help="Choose the algorithm for glucose prediction"
    )

    # Update selected algorithm if changed
    if selected_algo != st.session_state.selected_algorithm:
        st.session_state.selected_algorithm = selected_algo
        st.session_state.trained = False  # Retrain when algorithm changes

    st.markdown("---")

    st.header("⚙️ Configuration")

    # Patient ID selection
    patient_id = st.number_input(
        "Patient ID",
        min_value=1,
        max_value=10000,
        value=st.session_state.patient_id,
        step=1,
        help="Enter the patient ID to analyze"
    )

    # Update patient ID if changed
    if patient_id != st.session_state.patient_id:
        st.session_state.patient_id = patient_id
        st.session_state.system_ready = False
        st.session_state.trained = False

    st.markdown("---")

    # Load data button
    if st.button("📥 Load Patient Data", type="primary", use_container_width=True):
        with st.spinner(f"Loading data for Patient {patient_id}..."):
            try:
                # Initialize components
                fetcher = GlucoseDataFetcher(patient_id)
                preprocessor = GlucosePreprocessor()

                # Fetch data
                raw_data = fetcher.fetch_patient_data()

                if raw_data is None or len(raw_data) == 0:
                    st.error(f"No data found for Patient {patient_id}")
                else:
                    st.session_state.raw_data = raw_data

                    # Preprocess data
                    st.session_state.anomalies = preprocessor.detect_anomalies(raw_data)
                    st.session_state.cleaned_data = preprocessor.clean_data(raw_data)
                    featured_data = preprocessor.add_features(st.session_state.cleaned_data)
                    preprocessor.analyze_patterns(st.session_state.cleaned_data)

                    st.session_state.preprocessor = preprocessor
                    st.session_state.featured_data = featured_data
                    st.session_state.system_ready = True
                    st.session_state.trained = False

                    st.success(f"✓ Loaded {len(raw_data):,} records!")

                fetcher.disconnect()

            except Exception as e:
                st.error(f"Error loading data: {str(e)}")

    st.markdown("---")

    # Train model button
    if st.session_state.system_ready:
        if st.button("🤖 Train Prediction Model", type="primary", use_container_width=True):
            with st.spinner(f"Training {all_algorithms[st.session_state.selected_algorithm]['name']} model..."):
                try:
                    predictor = GlucosePredictor(algorithm=st.session_state.selected_algorithm)

                    # Train based on algorithm type
                    if st.session_state.selected_algorithm == 'prophet':
                        predictor.train_prophet_model(st.session_state.cleaned_data)
                    else:
                        # Train ML model with features
                        if len(st.session_state.featured_data.columns) > 1:
                            predictor.train_ml_model(st.session_state.featured_data)
                        else:
                            st.warning("Not enough features for ML model. Try Prophet instead.")
                            predictor = None

                    if predictor is not None:
                        st.session_state.predictor = predictor
                        st.session_state.trained = True
                        st.success(f"✓ {all_algorithms[st.session_state.selected_algorithm]['name']} model trained successfully!")

                except Exception as e:
                    st.error(f"Error training model: {str(e)}")

    st.markdown("---")

    # About section
    with st.expander("ℹ️ About"):
        st.markdown("""
        **Glucose Prediction System v1.0**

        This system uses AI to predict glucose levels based on historical data.

        **Features:**
        - Pattern analysis
        - Anomaly detection
        - Future predictions
        - Interactive visualizations
        """)

# Main content area
# Display algorithm information at the top
st.header(f"📊 Selected Algorithm: {all_algorithms[st.session_state.selected_algorithm]['name']}")

# Algorithm information card
algo_info = all_algorithms[st.session_state.selected_algorithm]

# Simple analogy at the top
if 'simple_analogy' in algo_info:
    st.info(f"**💡 Quick Understanding:** {algo_info['simple_analogy']}")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📖 About this Algorithm")
    st.write(f"**Type:** {algo_info['type']}")
    st.write(algo_info['description'])

with col2:
    st.subheader("🎯 Best Used For")
    st.success(algo_info['best_for'])

    st.subheader("🏷️ Algorithm Type")
    if algo_info['type'] == 'Time Series':
        st.success("⏱️ Time Series Model")
    elif algo_info['type'] in ['Regression', 'Ensemble', 'Kernel-based', 'Instance-based', 'Tree-based']:
        st.success(f"🔢 {algo_info['type']} Model")

# Expandable detailed explanation
with st.expander("🔍 How Does This Algorithm Work? (Click to see detailed explanation)", expanded=False):
    if 'how_it_works' in algo_info:
        st.markdown(algo_info['how_it_works'])
    else:
        st.write("Detailed explanation coming soon!")

# Key advantages in a nice format
st.subheader("✨ Key Advantages")
cols = st.columns(2)
for idx, advantage in enumerate(algo_info['advantages']):
    with cols[idx % 2]:
        st.markdown(f"✅ {advantage}")

st.markdown("---")

if not st.session_state.system_ready:
    # Welcome screen with instructions
    st.subheader("🚀 Getting Started")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 1️⃣ Choose Algorithm
        Select your preferred prediction algorithm from the sidebar.
        Each algorithm has unique strengths.
        """)

    with col2:
        st.markdown("""
        ### 2️⃣ Load Data
        Enter a Patient ID and click "Load Patient Data" to fetch historical glucose readings.
        """)

    with col3:
        st.markdown("""
        ### 3️⃣ Train & Predict
        Train the model and generate predictions with confidence intervals.
        """)

else:
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard",
        "🔮 Predictions",
        "📈 Patterns",
        "⚠️ Anomalies",
        "📋 Data"
    ])

    # Tab 1: Dashboard
    with tab1:
        st.header(f"Patient {patient_id} - Overview")

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Readings",
                f"{len(st.session_state.raw_data):,}",
                help="Total glucose readings in database"
            )

        with col2:
            avg_glucose = st.session_state.cleaned_data['glucose'].mean()
            st.metric(
                "Average Glucose",
                f"{avg_glucose:.1f} mg/dL",
                help="Average glucose level"
            )

        with col3:
            in_range = ((st.session_state.cleaned_data['glucose'] >= 70) &
                       (st.session_state.cleaned_data['glucose'] <= 180)).sum()
            time_in_range = in_range / len(st.session_state.cleaned_data) * 100
            st.metric(
                "Time in Range",
                f"{time_in_range:.1f}%",
                help="Percentage of time in target range (70-180 mg/dL)"
            )

        with col4:
            total_anomalies = len(st.session_state.anomalies['all'])
            st.metric(
                "Anomalies Detected",
                f"{total_anomalies:,}",
                help="Total anomalies found in data"
            )

        st.markdown("---")

        # Glucose timeline chart
        st.subheader("Glucose Levels Over Time")

        fig = go.Figure()

        # Plot glucose data
        fig.add_trace(go.Scatter(
            x=st.session_state.cleaned_data.index,
            y=st.session_state.cleaned_data['glucose'],
            mode='lines',
            name='Glucose',
            line=dict(color='#2E86AB', width=2)
        ))

        # Add target range
        fig.add_hrect(y0=70, y1=180, fillcolor="green", opacity=0.1,
                     line_width=0, annotation_text="Target Range")

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Glucose (mg/dL)",
            hovermode='x unified',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        # Statistics
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Statistics")
            stats_df = pd.DataFrame({
                'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                'Value': [
                    f"{st.session_state.cleaned_data['glucose'].mean():.1f} mg/dL",
                    f"{st.session_state.cleaned_data['glucose'].median():.1f} mg/dL",
                    f"{st.session_state.cleaned_data['glucose'].std():.1f} mg/dL",
                    f"{st.session_state.cleaned_data['glucose'].min():.1f} mg/dL",
                    f"{st.session_state.cleaned_data['glucose'].max():.1f} mg/dL"
                ]
            })
            st.dataframe(stats_df, hide_index=True, use_container_width=True)

        with col2:
            st.subheader("Time in Range Breakdown")
            below = (st.session_state.cleaned_data['glucose'] < 70).sum()
            in_range_count = ((st.session_state.cleaned_data['glucose'] >= 70) &
                            (st.session_state.cleaned_data['glucose'] <= 180)).sum()
            above = (st.session_state.cleaned_data['glucose'] > 180).sum()
            total = len(st.session_state.cleaned_data)

            range_df = pd.DataFrame({
                'Range': ['Below (<70)', 'In Range (70-180)', 'Above (>180)'],
                'Percentage': [
                    f"{below/total*100:.1f}%",
                    f"{in_range_count/total*100:.1f}%",
                    f"{above/total*100:.1f}%"
                ],
                'Hours': [f"{below:,}", f"{in_range_count:,}", f"{above:,}"]
            })
            st.dataframe(range_df, hide_index=True, use_container_width=True)

    # Tab 2: Predictions
    with tab2:
        st.header("🔮 Glucose Predictions")

        if not st.session_state.trained:
            st.warning("⚠️ Please train the model first using the sidebar button.")
        else:
            # Prediction controls
            col1, col2 = st.columns(2)

            with col1:
                prediction_mode = st.radio(
                    "Prediction Mode",
                    ["Quick Forecast", "Custom Date Range"],
                    horizontal=True
                )

            if prediction_mode == "Quick Forecast":
                with col2:
                    time_period = st.selectbox(
                        "Time Period",
                        ["Next 24 Hours", "Next 3 Days", "Next Week", "Next 2 Weeks", "Next Month"]
                    )

                # Map time period to hours
                period_map = {
                    "Next 24 Hours": 24,
                    "Next 3 Days": 72,
                    "Next Week": 168,
                    "Next 2 Weeks": 336,
                    "Next Month": 720
                }
                periods = period_map[time_period]

                if st.button("Generate Predictions", type="primary"):
                    with st.spinner("Generating predictions..."):
                        try:
                            predictions = st.session_state.predictor.predict_future(
                                periods=periods,
                                freq='H'
                            )
                            st.session_state.predictions = predictions
                            st.success(f"✓ Generated predictions for {time_period}")
                        except Exception as e:
                            st.error(f"Error generating predictions: {str(e)}")

            else:  # Custom Date Range
                col2a, col2b = col2.columns(2)

                with col2a:
                    start_date = st.date_input(
                        "Start Date",
                        value=datetime.now().date() + timedelta(days=1),
                        min_value=datetime.now().date()
                    )

                with col2b:
                    end_date = st.date_input(
                        "End Date",
                        value=datetime.now().date() + timedelta(days=30),
                        min_value=start_date
                    )

                if st.button("Generate Custom Predictions", type="primary"):
                    with st.spinner("Generating custom predictions..."):
                        try:
                            predictions = st.session_state.predictor.predict_date_range(
                                start_date=start_date.strftime('%Y-%m-%d'),
                                end_date=end_date.strftime('%Y-%m-%d')
                            )
                            st.session_state.predictions = predictions
                            st.success(f"✓ Generated predictions from {start_date} to {end_date}")
                        except Exception as e:
                            st.error(f"Error generating predictions: {str(e)}")

            # Display predictions
            if st.session_state.predictions is not None:
                st.markdown("---")

                # Get future predictions
                last_date = st.session_state.cleaned_data.index[-1]
                future_mask = st.session_state.predictions['timestamp'] > last_date
                future_pred = st.session_state.predictions[future_mask]

                # User-friendly prediction summary
                avg_pred = future_pred['predicted_glucose'].mean()
                min_pred = future_pred['predicted_glucose'].min()
                max_pred = future_pred['predicted_glucose'].max()

                # Interpret the predictions
                if 70 <= avg_pred <= 180:
                    status_color = "green"
                    status_text = "Good! Your predicted glucose levels are in the healthy range."
                    status_icon = "✅"
                elif avg_pred < 70:
                    status_color = "red"
                    status_text = "⚠️ Warning! Predictions show low glucose levels (hypoglycemia risk)."
                    status_icon = "⚠️"
                else:
                    status_color = "orange"
                    status_text = "⚠️ Caution! Predictions show elevated glucose levels."
                    status_icon = "⚠️"

                st.success(f"{status_icon} **Prediction Summary:** {status_text}")

                # Simple explanation
                with st.expander("📚 What do these predictions mean?", expanded=True):
                    st.markdown(f"""
                    ### Understanding Your Predictions

                    The **{all_algorithms[st.session_state.selected_algorithm]['name']}** algorithm analyzed your glucose history and predicts:

                    **What we predict:**
                    - Your glucose will average around **{avg_pred:.0f} mg/dL** in the predicted period
                    - Lowest expected: **{min_pred:.0f} mg/dL**
                    - Highest expected: **{max_pred:.0f} mg/dL**

                    **What this means in simple terms:**
                    - 🟢 **Healthy range:** 70-180 mg/dL (target zone)
                    - 🔴 **Too low:** Below 70 mg/dL (may feel dizzy, shaky)
                    - 🟠 **Too high:** Above 180 mg/dL (increased thirst, frequent urination)

                    **Confidence interval** (the shaded area in the chart):
                    - Shows the range where your actual glucose is likely to fall
                    - Wider band = more uncertainty
                    - Narrower band = more confident prediction

                    **Different algorithms give different predictions because:**
                    - Prophet focuses on time patterns (daily/weekly cycles)
                    - ML models (Random Forest, XGBoost, etc.) focus on complex relationships between factors
                    - Each has strengths in different scenarios
                    """)

                # Prediction metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    delta_text = "Healthy" if 70 <= avg_pred <= 180 else ("Low" if avg_pred < 70 else "High")
                    st.metric(
                        "Predicted Average",
                        f"{avg_pred:.1f} mg/dL",
                        delta=delta_text
                    )

                with col2:
                    st.metric(
                        "Predicted Min",
                        f"{min_pred:.1f} mg/dL",
                        delta="Low Risk" if min_pred < 70 else "Safe"
                    )

                with col3:
                    st.metric(
                        "Predicted Max",
                        f"{max_pred:.1f} mg/dL",
                        delta="High Risk" if max_pred > 180 else "Safe"
                    )

                with col4:
                    st.metric(
                        "Total Predictions",
                        f"{len(future_pred):,}",
                        delta=f"Using {all_algorithms[st.session_state.selected_algorithm]['name']}"
                    )

                # Prediction chart
                st.subheader("Prediction Chart")

                fig = go.Figure()

                # Historical data
                fig.add_trace(go.Scatter(
                    x=st.session_state.cleaned_data.index,
                    y=st.session_state.cleaned_data['glucose'],
                    mode='lines',
                    name='Historical',
                    line=dict(color='#2E86AB', width=2)
                ))

                # Future predictions
                fig.add_trace(go.Scatter(
                    x=future_pred['timestamp'],
                    y=future_pred['predicted_glucose'],
                    mode='lines+markers',
                    name='Prediction',
                    line=dict(color='#F18F01', width=3),
                    marker=dict(size=4)
                ))

                # Confidence interval
                fig.add_trace(go.Scatter(
                    x=future_pred['timestamp'],
                    y=future_pred['upper_bound'],
                    mode='lines',
                    name='Upper Bound',
                    line=dict(width=0),
                    showlegend=False
                ))

                fig.add_trace(go.Scatter(
                    x=future_pred['timestamp'],
                    y=future_pred['lower_bound'],
                    mode='lines',
                    name='Lower Bound',
                    fill='tonexty',
                    fillcolor='rgba(241, 143, 1, 0.2)',
                    line=dict(width=0),
                    showlegend=True
                ))

                # Target range
                fig.add_hrect(y0=70, y1=180, fillcolor="green", opacity=0.1,
                             line_width=0, annotation_text="Target Range")

                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Glucose (mg/dL)",
                    hovermode='x unified',
                    height=500
                )

                st.plotly_chart(fig, use_container_width=True)

                # Download predictions
                st.subheader("Download Predictions")

                col1, col2, col3 = st.columns(3)

                with col1:
                    csv = future_pred.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"predictions_patient_{patient_id}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                with col2:
                    # Generate medical report button - PDF
                    if st.button("📄 Generate Medical Report (PDF)", use_container_width=True):
                        with st.spinner("Generating medical report..."):
                            try:
                                # Prepare predictions for report
                                report_df = st.session_state.predictions.copy()
                                report_df = report_df.rename(columns={
                                    'timestamp': 'ds',
                                    'predicted_glucose': 'yhat',
                                    'lower_bound': 'yhat_lower',
                                    'upper_bound': 'yhat_upper'
                                })

                                # Calculate statistics
                                stats = {
                                    'avg_glucose': future_pred['predicted_glucose'].mean(),
                                    'min_glucose': future_pred['predicted_glucose'].min(),
                                    'max_glucose': future_pred['predicted_glucose'].max(),
                                    'std_glucose': future_pred['predicted_glucose'].std()
                                }

                                # Generate report
                                report_gen = GlucoseReportGenerator(
                                    patient_id=patient_id,
                                    predictions_df=report_df[future_mask],
                                    statistics=stats
                                )

                                pdf_filename = f"glucose_report_patient_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                                report_gen.generate_pdf_report(pdf_filename)

                                # Read PDF and create download button
                                with open(pdf_filename, "rb") as pdf_file:
                                    pdf_bytes = pdf_file.read()

                                st.download_button(
                                    label="⬇️ Download PDF Report",
                                    data=pdf_bytes,
                                    file_name=pdf_filename,
                                    mime="application/pdf",
                                    use_container_width=True
                                )

                                st.success("✓ Medical report generated successfully!")

                            except Exception as e:
                                st.error(f"Error generating report: {str(e)}")

                with col3:
                    # Generate medical report button - HTML
                    if st.button("🌐 Generate Medical Report (HTML)", use_container_width=True):
                        with st.spinner("Generating HTML report..."):
                            try:
                                # Prepare predictions for report
                                report_df = st.session_state.predictions.copy()
                                report_df = report_df.rename(columns={
                                    'timestamp': 'ds',
                                    'predicted_glucose': 'yhat',
                                    'lower_bound': 'yhat_lower',
                                    'upper_bound': 'yhat_upper'
                                })

                                # Calculate statistics
                                stats = {
                                    'avg_glucose': future_pred['predicted_glucose'].mean(),
                                    'min_glucose': future_pred['predicted_glucose'].min(),
                                    'max_glucose': future_pred['predicted_glucose'].max(),
                                    'std_glucose': future_pred['predicted_glucose'].std()
                                }

                                # Generate report
                                report_gen = GlucoseReportGenerator(
                                    patient_id=patient_id,
                                    predictions_df=report_df[future_mask],
                                    statistics=stats
                                )

                                html_filename = f"glucose_report_patient_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                                report_gen.generate_html_report(html_filename)

                                # Read HTML and create download button
                                with open(html_filename, "r", encoding='utf-8') as html_file:
                                    html_content = html_file.read()

                                st.download_button(
                                    label="⬇️ Download HTML Report",
                                    data=html_content,
                                    file_name=html_filename,
                                    mime="text/html",
                                    use_container_width=True
                                )

                                st.success("✓ HTML report generated successfully!")

                            except Exception as e:
                                st.error(f"Error generating report: {str(e)}")

    # Tab 3: Patterns
    with tab3:
        st.header("📈 Glucose Patterns")

        col1, col2 = st.columns(2)

        with col1:
            # Daily pattern
            st.subheader("Daily Pattern (By Hour)")
            hourly_avg = st.session_state.cleaned_data.groupby(
                st.session_state.cleaned_data.index.hour
            )['glucose'].agg(['mean', 'std'])

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hourly_avg.index,
                y=hourly_avg['mean'],
                mode='lines+markers',
                name='Average',
                line=dict(color='#2E86AB', width=3),
                marker=dict(size=8)
            ))

            # Add std deviation band
            fig.add_trace(go.Scatter(
                x=hourly_avg.index,
                y=hourly_avg['mean'] + hourly_avg['std'],
                mode='lines',
                name='Upper Std',
                line=dict(width=0),
                showlegend=False
            ))

            fig.add_trace(go.Scatter(
                x=hourly_avg.index,
                y=hourly_avg['mean'] - hourly_avg['std'],
                mode='lines',
                name='Lower Std',
                fill='tonexty',
                fillcolor='rgba(46, 134, 171, 0.2)',
                line=dict(width=0),
                showlegend=True
            ))

            fig.add_hrect(y0=70, y1=180, fillcolor="green", opacity=0.1, line_width=0)
            fig.update_layout(
                xaxis_title="Hour of Day",
                yaxis_title="Average Glucose (mg/dL)",
                xaxis=dict(tickmode='linear', tick0=0, dtick=2),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

            # Peak/lowest hours
            peak_hour = hourly_avg['mean'].idxmax()
            lowest_hour = hourly_avg['mean'].idxmin()
            st.info(f"**Peak:** {peak_hour}:00 ({hourly_avg['mean'].max():.1f} mg/dL)  \n"
                   f"**Lowest:** {lowest_hour}:00 ({hourly_avg['mean'].min():.1f} mg/dL)")

        with col2:
            # Weekly pattern
            st.subheader("Weekly Pattern (By Day)")
            daily_avg = st.session_state.cleaned_data.groupby(
                st.session_state.cleaned_data.index.dayofweek
            )['glucose'].agg(['mean', 'std'])

            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=days,
                y=daily_avg['mean'],
                name='Average',
                marker_color='#2E86AB',
                error_y=dict(type='data', array=daily_avg['std'])
            ))

            fig.add_hrect(y0=70, y1=180, fillcolor="green", opacity=0.1, line_width=0)
            fig.update_layout(
                xaxis_title="Day of Week",
                yaxis_title="Average Glucose (mg/dL)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

            # Highest/lowest days
            highest_day_idx = daily_avg['mean'].idxmax()
            lowest_day_idx = daily_avg['mean'].idxmin()
            st.info(f"**Highest:** {days[highest_day_idx]} ({daily_avg['mean'].max():.1f} mg/dL)  \n"
                   f"**Lowest:** {days[lowest_day_idx]} ({daily_avg['mean'].min():.1f} mg/dL)")

        # Distribution
        st.markdown("---")
        st.subheader("Glucose Distribution")

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=st.session_state.cleaned_data['glucose'],
            nbinsx=50,
            name='Glucose',
            marker_color='#2E86AB',
            opacity=0.7
        ))

        fig.add_vline(
            x=st.session_state.cleaned_data['glucose'].mean(),
            line_dash="dash",
            line_color="red",
            annotation_text="Mean"
        )

        fig.update_layout(
            xaxis_title="Glucose (mg/dL)",
            yaxis_title="Frequency",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    # Tab 4: Anomalies
    with tab4:
        st.header("⚠️ Detected Anomalies")

        # Anomaly metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Sensor Errors", f"{len(st.session_state.anomalies['sensor_errors']):,}")

        with col2:
            st.metric("Statistical Outliers", f"{len(st.session_state.anomalies['statistical_outliers']):,}")

        with col3:
            st.metric("Sudden Changes", f"{len(st.session_state.anomalies['sudden_changes']):,}")

        with col4:
            st.metric("Data Gaps", f"{len(st.session_state.anomalies['data_gaps']):,}")

        st.markdown("---")

        # Anomaly type selector
        anomaly_type = st.selectbox(
            "View Anomaly Type",
            ["All Anomalies", "Sensor Errors", "Statistical Outliers", "Sudden Changes", "Data Gaps"]
        )

        # Map selection to data
        type_map = {
            "All Anomalies": 'all',
            "Sensor Errors": 'sensor_errors',
            "Statistical Outliers": 'statistical_outliers',
            "Sudden Changes": 'sudden_changes',
            "Data Gaps": 'data_gaps'
        }

        selected_anomalies = st.session_state.anomalies[type_map[anomaly_type]]

        if len(selected_anomalies) > 0:
            st.subheader(f"{anomaly_type} - {len(selected_anomalies):,} detected")

            # Show table
            display_cols = ['timestamp', 'value'] if 'value' in selected_anomalies.columns else list(selected_anomalies.columns[:5])
            st.dataframe(
                selected_anomalies[display_cols].head(100),
                use_container_width=True,
                height=300
            )

            # Download anomalies
            csv = selected_anomalies.to_csv(index=False)
            st.download_button(
                label=f"📥 Download {anomaly_type} (CSV)",
                data=csv,
                file_name=f"anomalies_{anomaly_type.lower().replace(' ', '_')}_patient_{patient_id}.csv",
                mime="text/csv"
            )
        else:
            st.info(f"No {anomaly_type.lower()} detected.")

    # Tab 5: Data
    with tab5:
        st.header("📋 Raw Data")

        col1, col2 = st.columns([3,1])

        with col1:
            st.subheader(f"Patient {patient_id} - Glucose Readings")

        with col2:
            data_view = st.selectbox("View", ["Cleaned Data", "Raw Data"])

        # Select data to display
        if data_view == "Cleaned Data":
            display_data = st.session_state.cleaned_data.reset_index()
        else:
            display_data = st.session_state.raw_data

        # Show data
        st.dataframe(display_data, use_container_width=True, height=500)

        # Download data
        csv = display_data.to_csv(index=False)
        st.download_button(
            label=f"📥 Download {data_view} (CSV)",
            data=csv,
            file_name=f"{data_view.lower().replace(' ', '_')}_patient_{patient_id}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

        # Data summary
        st.markdown("---")
        st.subheader("Data Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**Date Range**")
            if data_view == "Cleaned Data":
                st.write(f"From: {display_data['timestamp'].min()}")
                st.write(f"To: {display_data['timestamp'].max()}")
            else:
                st.write(f"From: {display_data['timestamp'].min()}")
                st.write(f"To: {display_data['timestamp'].max()}")

        with col2:
            st.write("**Record Count**")
            st.write(f"Total: {len(display_data):,}")
            duration = (display_data['timestamp'].max() - display_data['timestamp'].min()).days
            st.write(f"Duration: {duration} days")

        with col3:
            st.write("**Glucose Range**")
            value_col = 'glucose' if 'glucose' in display_data.columns else 'value'
            st.write(f"Min: {display_data[value_col].min():.1f} mg/dL")
            st.write(f"Max: {display_data[value_col].max():.1f} mg/dL")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>Glucose Prediction System v1.0 | Powered by Facebook Prophet & Machine Learning</p>
    </div>
    """,
    unsafe_allow_html=True
)
