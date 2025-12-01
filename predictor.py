"""
Glucose prediction module using multiple prediction algorithms
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Note: XGBoost not installed. Using alternatives.")

# ADDED: Deep Learning Imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False
    print("Note: TensorFlow not installed. Deep Learning models unavailable.")


class GlucosePredictor:
    """Handles glucose level predictions using multiple algorithms"""

    # def __init__(self, algorithm='prophet'):
    #     self.algorithm = algorithm
    #     self.prophet_model = None
    #     self.ml_model = None
    #     self.scaler = StandardScaler()
    #     self.training_data = None
    #     self.feature_cols = None

    def __init__(self, algorithm='prophet'):
        self.algorithm = algorithm
        self.prophet_model = None
        self.ml_model = None
        # Use MinMaxScaler for LSTMs (better for neural nets), StandardScaler for others
        if algorithm in ['lstm', 'gru']:
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()
            
        # Separate scaler for the target variable (y) - Critical for LSTMs!
        self.y_scaler = MinMaxScaler()
        
        self.training_data = None
        self.feature_cols = None

        # Algorithm information
        self.algorithms = {
            'prophet': {
                'name': 'Facebook Prophet',
                'type': 'Time Series',
                'description': 'Prophet is a time series forecasting model designed for business metrics with daily observations. It captures seasonality, trends, and holidays effectively.',
                'how_it_works': '''
**Think of it like a weather forecaster for your glucose levels!**

Prophet analyzes your glucose history like a meteorologist studies weather patterns. Here's how:

**Step 1: Finding the Trend**
- Looks at your glucose over days/weeks to see if it's generally going up, down, or staying stable
- Example: If your average glucose was 150 mg/dL last month and 145 mg/dL this month, it detects a downward trend

**Step 2: Discovering Daily Patterns**
- Identifies what happens at different times of day
- Example: "Every morning around 8 AM, glucose rises to ~180 mg/dL (breakfast spike)"
- "Every night around 11 PM, glucose drops to ~120 mg/dL (sleep)"

**Step 3: Finding Weekly Patterns**
- Notices differences between weekdays and weekends
- Example: "Saturday mornings show higher glucose (~160) than weekday mornings (~140)"

**Step 4: Making Predictions**
- Combines the trend + daily pattern + weekly pattern
- Example: "Next Tuesday at 8 AM: Based on your breakfast pattern (usually 180), downward trend (-5), I predict 175 mg/dL"

**Real Example with Your Data:**
- Input: 30 days of glucose readings (every hour = 720 readings)
- Prophet sees: Morning spike at 7-9 AM, afternoon dip at 2-3 PM, bedtime drop at 10 PM
- Prediction: "Tomorrow at 8 AM will be 172 mg/dL (confidence: 160-185 mg/dL)"
                ''',
                'advantages': [
                    'Excellent for capturing daily and weekly patterns',
                    'Handles missing data and outliers automatically',
                    'Provides uncertainty intervals for predictions',
                    'Easy to interpret and tune',
                    'Robust to missing data points'
                ],
                'best_for': 'Time series data with strong seasonal patterns',
                'simple_analogy': 'Like a smart calendar that learns your routine and predicts what will happen at specific times'
            },
            # ADDED: LSTM Algorithm Description
            'lstm': {
                'name': 'LSTM',
                'type': 'Recurrent Neural Network',
                'description': 'Long Short-Term Memory network. A type of deep learning model specifically designed to remember long-term patterns in time-series data.',
                'how_it_works': 'LSTMs process data sequentially, keeping an internal "memory state" that can store information over long periods. Unlike standard regression, they can decide what to remember and what to forget from previous glucose readings.',
                'advantages': ['Captures long-term dependencies', 'Great for complex sequential patterns', 'State-of-the-art for many time-series tasks'],
                'best_for': 'Large datasets with complex historical patterns',
                'simple_analogy': 'Like a doctor who remembers your entire medical history, not just your last visit.'
            },

            # ADD THIS BLOCK:
            'lightgbm': {
                'name': 'LightGBM',
                'type': 'Ensemble',
                'description': 'A gradient boosting framework that uses tree-based learning algorithms. It is designed to be distributed and efficient with fast training speed.',
                'how_it_works': 'Unlike other boosting algorithms that grow trees level-wise, LightGBM grows trees leaf-wise. It chooses the leaf with max delta loss to grow, leading to lower loss.',
                'advantages': ['Faster training speed and higher efficiency', 'Lower memory usage', 'Better accuracy than other boosting methods'],
                'best_for': 'Large datasets and high-speed requirements',
                'simple_analogy': 'Like a speed-reader who focuses only on the most important chapters of a book first.'
            },
            # ADD THIS BLOCK:
            'catboost': {
                'name': 'CatBoost',
                'type': 'Ensemble',
                'description': 'A high-performance open source library for gradient boosting on decision trees. It is known for its high quality without parameter tuning and robustness.',
                'how_it_works': 'It builds symmetric decision trees. During training, it focuses on reducing "prediction shift" (a common issue in boosting) to provide more stable and accurate forecasts.',
                'advantages': ['High accuracy', 'Robust to overfitting', 'Great defaults', 'Handles noisy data well'],
                'best_for': 'Complex datasets where other models fail to generalize',
                'simple_analogy': 'Like a wise teacher who corrects their own biases to give a fair grade.'
            },
            # ADD THIS BLOCK:
            'adaboost': {
                'name': 'AdaBoost',
                'type': 'Ensemble',
                'description': 'Adaptive Boosting. It fits a sequence of weak learners on repeatedly modified versions of the data.',
                'how_it_works': 'Unlike Random Forest, which builds trees independent of each other, AdaBoost builds trees sequentially. Each new tree focuses on correcting the errors made by the previous trees.',
                'advantages': ['Less prone to overfitting', 'Requires few parameters to tune', 'Improves accuracy of weak learners'],
                'best_for': 'Focusing on "hard-to-predict" cases',
                'simple_analogy': 'Like a student taking practice tests, but studying ONLY the questions they got wrong on the last test.'
            },
            # ADD THIS BLOCK:
            'hist_gradient_boosting': {
                'name': 'Hist. Gradient Boosting',
                'type': 'Ensemble',
                'description': 'Scikit-learn’s modern, optimized implementation of gradient boosting. It uses histogram-binning for speed and handles missing values natively.',
                'how_it_works': 'It groups data points into "bins" (histograms) to find split points much faster. It uses a "leaf-wise" growth strategy similar to LightGBM.',
                'advantages': ['Native handling of missing data', 'Much faster than standard GB', 'State-of-the-art accuracy'],
                'best_for': 'Large datasets with missing values or noise',
                'simple_analogy': 'A highly efficient organizer who groups detailed files into buckets to make decisions 10x faster.'
            },
            # ADDED: GRU Algorithm Description
            'gru': {
                'name': 'GRU',
                'type': 'Recurrent Neural Network',
                'description': 'Gated Recurrent Unit. A simplified, often faster version of LSTM that performs similarly well on smaller datasets.',
                'how_it_works': 'Similar to LSTM but with a streamlined architecture. It uses "gates" to control the flow of information, balancing previous context with new input.',
                'advantages': ['Faster training than LSTM', 'Often as accurate as LSTM', 'Efficient memory usage'],
                'best_for': 'Medium-sized datasets needing deep learning power',
                'simple_analogy': 'A "speed-reader" version of the LSTM doctor.'
            },
            'linear_regression': {
                'name': 'Linear Regression',
                'type': 'Regression',
                'description': 'A fundamental algorithm that models the relationship between features and target using a linear equation. Simple yet effective for many scenarios.',
                'how_it_works': '''
**Think of it like finding the "recipe" for your glucose!**

Linear Regression creates a mathematical formula that connects your activities to glucose levels.

**Step 1: Identifying Factors**
- Looks at time of day, day of week, recent glucose readings
- Example factors: Hour (0-23), Day (0-6), Previous glucose average

**Step 2: Finding Relationships**
- Discovers how each factor affects glucose
- Example: "For every hour after midnight, glucose increases by 2 mg/dL"
- "On weekends, glucose is 10 mg/dL higher on average"

**Step 3: Creating the Formula**
- Combines all factors into one equation
- Formula example: Glucose = 100 + (2 × Hour) + (10 × IsWeekend) + (0.5 × YesterdayAvg)

**Step 4: Making Predictions**
- Plugs in values for the time you want to predict
- Example: Tuesday at 8 AM, yesterday average was 140
- Prediction = 100 + (2×8) + (10×0) + (0.5×140) = 100 + 16 + 0 + 70 = 186 mg/dL

**Real Example with Your Data:**
- Input: Hour=8, Day=Tuesday(2), Weekend=No, Recent_Avg=145
- Model: Glucose = 95 + 3.2×Hour + 8×Weekend + 0.6×Recent_Avg
- Calculation: 95 + 3.2×8 + 8×0 + 0.6×145 = 95 + 25.6 + 0 + 87 = 207.6 mg/dL
- Final Prediction: 208 mg/dL
                ''',
                'advantages': [
                    'Simple and easy to understand',
                    'Fast training and prediction',
                    'Works well with linearly separable data',
                    'Low computational cost',
                    'Good baseline model'
                ],
                'best_for': 'Linear relationships between features',
                'simple_analogy': 'Like a simple math equation that adds up different factors to predict your glucose'
            },
            'ridge': {
                'name': 'Ridge Regression',
                'type': 'Regression',
                'description': 'Linear regression with L2 regularization that prevents overfitting by penalizing large coefficients.',
                'how_it_works': '''
**Think of it like Linear Regression with a "reality check"!**

Ridge is Linear Regression's smarter cousin that avoids making extreme predictions.

**Step 1: Same as Linear Regression**
- Creates formula: Glucose = Factor1 + Factor2 + Factor3...

**Step 2: Adding Balance (The Key Difference!)**
- Prevents any single factor from dominating
- Example: If regular regression says "Hour affects glucose by +50", Ridge says "that's too extreme, let's make it +15"

**Step 3: Creating Balanced Formula**
- All factors contribute, but none too much
- Result: More stable, reliable predictions

**Why This Matters:**
- Regular Linear Regression might say: "Weekend increases glucose by 100 mg/dL!" (too extreme)
- Ridge says: "Weekend increases glucose by 12 mg/dL" (more realistic)

**Real Example:**
- Input: Same glucose data
- Linear Regression: Wild swings in predictions (50-300 mg/dL)
- Ridge Regression: Smoother predictions (120-180 mg/dL) - more trustworthy!
                ''',
                'advantages': [
                    'Prevents overfitting better than linear regression',
                    'Handles multicollinearity well',
                    'More stable predictions',
                    'Good for high-dimensional data',
                    'Regularization improves generalization'
                ],
                'best_for': 'Datasets with correlated features',
                'simple_analogy': 'Like a balanced diet - no single ingredient dominates, everything in moderation'
            },
            'lasso': {
                'name': 'Lasso Regression',
                'type': 'Regression',
                'description': 'Linear regression with L1 regularization that can perform feature selection by driving some coefficients to zero.',
                'how_it_works': '''
**Think of it like a minimalist organizer for your glucose factors!**

Lasso automatically identifies which factors truly matter and ignores the rest.

**Step 1: Start with All Factors**
- Hour, day, weekend, recent average, rolling mean, etc. (maybe 20+ factors)

**Step 2: Eliminate Unimportant Ones**
- Lasso tests each factor and removes those that don't help
- Example: Maybe "month" doesn't affect glucose → Remove it (set to 0)
- Maybe "day of week" has minimal effect → Remove it too

**Step 3: Keep Only What Matters**
- Final formula uses only 5-8 key factors
- Example: Keeps "Hour", "Recent_Avg", "Weekend" but drops the rest

**Step 4: Make Predictions**
- Uses simple formula with only important factors
- Result: Clean, interpretable predictions

**Real Example:**
- Started with 20 factors
- Lasso found only 6 actually matter: Hour, IsWeekend, Previous_Hour_Glucose, Rolling_Mean_7day, Day_of_Month, Recent_Std
- Final prediction: Glucose = 85 + 2.5×Hour + 8×Weekend + 0.3×Previous + 0.4×Rolling_Mean
- This tells you: "Time of day and recent patterns are what really drive your glucose!"
                ''',
                'advantages': [
                    'Automatic feature selection',
                    'Produces sparse models',
                    'Good for high-dimensional data',
                    'Helps identify important features',
                    'Prevents overfitting'
                ],
                'best_for': 'Feature selection and sparse solutions',
                'simple_analogy': 'Like Marie Kondo for data - keeps only what "sparks joy" (actually matters)'
            },
            'random_forest': {
                'name': 'Random Forest',
                'type': 'Ensemble',
                'description': 'An ensemble learning method that constructs multiple decision trees and outputs their average prediction.',
                'how_it_works': '''
**Think of it like asking 100 doctors for their opinion, then taking the average!**

Random Forest creates many "decision trees" and combines their predictions.

**Step 1: Create Decision Tree #1**
- Asks: "Is it morning? YES → Is it weekend? NO → Predict 165 mg/dL"
- This tree uses random factors and makes a prediction

**Step 2: Create Decision Tree #2**
- Uses different random factors
- Asks: "Recent avg > 150? YES → Hour < 12? YES → Predict 172 mg/dL"

**Step 3: Create 98 More Trees**
- Each tree has its own "decision path"
- Each makes its own prediction

**Step 4: Vote & Average**
- Tree 1 says: 165 mg/dL
- Tree 2 says: 172 mg/dL
- Tree 3 says: 168 mg/dL
- ... (97 more predictions)
- Final prediction: Average of all 100 = 169 mg/dL

**Why This Works:**
- Each tree might make mistakes
- But 100 trees together are very accurate!
- Like asking many people for directions vs. just one person

**Real Example with Your Data:**
- Input: Tuesday 2 PM, Recent avg=155, Last reading=162
- Tree 1 path: Hour(14) → Day(2) → Recent(155) → Predicts 158
- Tree 2 path: Recent(155) → Weekend(No) → Hour(14) → Predicts 163
- ... 98 more trees predict values between 155-165
- Final: Average of 100 predictions = 160 mg/dL
                ''',
                'advantages': [
                    'Highly accurate and robust',
                    'Handles non-linear relationships well',
                    'Provides feature importance',
                    'Resistant to overfitting',
                    'Works with missing values'
                ],
                'best_for': 'Complex non-linear patterns',
                'simple_analogy': 'Like a committee of 100 experts voting on the answer - wisdom of crowds!'
            },
            'gradient_boosting': {
                'name': 'Gradient Boosting',
                'type': 'Ensemble',
                'description': 'Builds an ensemble of weak learners sequentially, with each tree correcting errors of the previous ones.',
                'how_it_works': '''
**Think of it like a teacher correcting homework, one mistake at a time!**

Gradient Boosting builds trees sequentially, where each tree fixes the previous tree's errors.

**Step 1: First Simple Prediction**
- Tree 1 makes a basic guess: "Everyone gets 150 mg/dL"
- Check errors: Some actual values were 180 (error: +30), some were 120 (error: -30)

**Step 2: Learn from Mistakes**
- Tree 2 focuses on fixing Tree 1's errors
- Learns: "Morning readings were 30 too low, add +30"
- New prediction = Tree 1 + Tree 2 = 150 + 30 = 180 (better!)

**Step 3: Fix Remaining Errors**
- Tree 3 focuses on what Tree 2 missed
- Learns: "Weekends need -5 adjustment"
- Prediction = Tree 1 + Tree 2 + Tree 3 = 180 - 5 = 175 (even better!)

**Step 4: Keep Improving**
- Trees 4, 5, 6... each fix smaller and smaller errors
- Final prediction = Sum of all trees' contributions

**Why This Works:**
- Each tree specializes in fixing specific mistakes
- Like a student improving grade from C → B → A → A+ through targeted practice

**Real Example:**
- Input: Wednesday 10 AM
- Tree 1: Base prediction = 150 mg/dL
- Tree 2: +15 (morning adjustment)
- Tree 3: +5 (weekday boost)
- Tree 4: -3 (mid-morning dip)
- Tree 5: +2 (fine-tuning)
- Final: 150+15+5-3+2 = 169 mg/dL
                ''',
                'advantages': [
                    'Often provides best accuracy',
                    'Handles mixed data types well',
                    'Captures complex patterns',
                    'Feature importance analysis',
                    'Flexible loss functions'
                ],
                'best_for': 'High accuracy requirements',
                'simple_analogy': 'Like learning from mistakes - each step corrects previous errors'
            },
            'xgboost': {
                'name': 'XGBoost',
                'type': 'Ensemble',
                'description': 'Extreme Gradient Boosting - an optimized gradient boosting framework that is highly efficient and accurate.',
                'how_it_works': '''
**Think of it like Gradient Boosting on steroids - faster and smarter!**

XGBoost is like Gradient Boosting but optimized for speed and accuracy.

**Same Core Idea as Gradient Boosting:**
- Builds trees sequentially, each fixing previous errors
- Tree 1 + Tree 2 + Tree 3... = Final prediction

**But with Superpowers:**

**1. Parallel Processing**
- Builds parts of trees simultaneously
- Like having 4 chefs cooking instead of 1 → 4× faster!

**2. Smart Regularization**
- Prevents overfitting automatically
- Won't memorize your data, learns real patterns

**3. Handles Missing Data**
- If some glucose readings are missing, XGBoost figures out the best path
- Example: "No reading at 3 AM? I'll use the pattern from similar days"

**4. Built-in Cross-Validation**
- Tests itself while training to ensure accuracy

**Real Example:**
- Same process as Gradient Boosting but executes in 1/4 the time
- Input: Thursday 3 PM, Recent pattern shows upward trend
- Tree 1: Base = 145
- Trees 2-5: Adjustments based on time/pattern = +20
- Final: 165 mg/dL (with 95% confidence it's correct!)
                ''',
                'advantages': [
                    'State-of-the-art performance',
                    'Fast training with parallelization',
                    'Handles missing values automatically',
                    'Built-in regularization',
                    'Excellent for competitions'
                ],
                'best_for': 'Maximum performance and speed',
                'simple_analogy': 'Like a Formula 1 race car version of Gradient Boosting - same destination, but faster!'
            },
            'svr': {
                'name': 'Support Vector Regression',
                'type': 'Kernel-based',
                'description': 'Uses support vector machines for regression by finding a function that deviates minimally from training data.',
                'how_it_works': '''
**Think of it like drawing the best-fit line, but with a "tolerance zone"!**

SVR finds a prediction line where most points fall within an acceptable error margin.

**Step 1: Create Tolerance Tube**
- Draws a tube around predictions
- Example: "I accept predictions within ±10 mg/dL of actual values"

**Step 2: Find Best Line Through Tube**
- Finds the line that keeps most glucose readings within the tube
- Points outside the tube are "support vectors" (they define the boundaries)

**Step 3: Handle Complex Patterns (Kernel Trick)**
- Can bend and curve the line for non-linear patterns
- Example: Morning spike pattern is curved, not straight

**Step 4: Make Predictions**
- New data point → Check where it falls on the line
- Predict glucose based on position

**Why This Works:**
- Focuses on extreme cases (support vectors) to define boundaries
- Robust to outliers (one weird reading won't mess up predictions)

**Real Example:**
- Input: 1000 glucose readings
- SVR identifies 50 "support vectors" (key boundary points)
- Creates prediction boundary based on these 50 critical points
- For Friday 7 AM: Position on curve → Predicts 178 mg/dL
- Tolerance: Could be 168-188 mg/dL (±10 margin)
                ''',
                'advantages': [
                    'Effective in high dimensions',
                    'Memory efficient',
                    'Versatile with different kernels',
                    'Robust to outliers',
                    'Good for non-linear data'
                ],
                'best_for': 'Non-linear patterns with outliers',
                'simple_analogy': 'Like drawing a flexible line that bends to fit your data pattern'
            },
            'knn': {
                'name': 'K-Nearest Neighbors',
                'type': 'Instance-based',
                'description': 'Predicts values based on the average of K nearest training examples in the feature space.',
                'how_it_works': '''
**Think of it like asking "What happened in similar situations before?"**

KNN finds the K most similar past situations and averages their outcomes.

**Step 1: You Want to Predict**
- Tuesday 9 AM, recent average 155, weekend=No

**Step 2: Find Similar Past Situations (K=5)**
- Looks through all history for most similar situations
- Found 5 similar cases:
  1. Last Tuesday 9 AM: 162 mg/dL
  2. Two weeks ago Tuesday 9:15 AM: 158 mg/dL
  3. Wednesday 8:45 AM: 165 mg/dL
  4. Last Monday 9:30 AM: 160 mg/dL
  5. Tuesday 8:30 AM three weeks ago: 157 mg/dL

**Step 3: Average These Similar Cases**
- Average = (162 + 158 + 165 + 160 + 157) / 5 = 160.4 mg/dL
- Prediction: 160 mg/dL

**Why This Works:**
- If similar situations led to similar glucose levels before, likely to happen again
- No complex math needed - just memory and similarity matching

**The "K" in K-NN:**
- K=5 means "look at 5 similar cases"
- K=10 means "look at 10 similar cases"
- More neighbors = smoother predictions, fewer neighbors = more responsive to local patterns

**Real Example:**
- Want to predict: Saturday 2 PM, recent avg 148
- KNN finds: 7 most similar Saturdays around 2 PM
- Their glucose values: 152, 155, 149, 158, 151, 153, 150
- Prediction: Average = 153 mg/dL
                ''',
                'advantages': [
                    'Simple and intuitive',
                    'No training phase required',
                    'Naturally handles multi-class problems',
                    'Adapts to local patterns',
                    'Non-parametric approach'
                ],
                'best_for': 'Local pattern recognition',
                'simple_analogy': 'Like asking your 5 closest friends what they would do in your situation'
            },
            'decision_tree': {
                'name': 'Decision Tree',
                'type': 'Tree-based',
                'description': 'Creates a tree-like model of decisions based on features, easy to visualize and interpret.',
                'how_it_works': '''
**Think of it like a flowchart of yes/no questions!**

Decision Tree makes predictions by asking a series of simple questions.

**The Tree Structure:**
Start Here
   ↓
Is it morning (Hour < 12)?
   YES ↓                    NO ↓
   Is it weekend?          Is recent avg > 150?
   YES ↓    NO ↓          YES ↓         NO ↓
   175     185            165           140
```

**Step 1: First Question (Root)**
- "Is it morning?" (Hour < 12)
- YES → Go left branch, NO → Go right branch

**Step 2: Second Question**
- Left branch: "Is it weekend?"
- Right branch: "Is recent average > 150?"

**Step 3: Keep Asking Until You Reach a Leaf**
- Each leaf has a prediction value

**Real Example - Predicting Tuesday 10 AM:**
1. Hour < 12? YES (10 < 12) → Go left
2. Is weekend? NO (Tuesday) → Go to "NO" branch
3. Recent avg > 145? YES (155 > 145) → Go to "YES" branch
4. Previous hour > 150? YES (162 > 150) → Go to "YES" branch
5. Reach leaf: Predict 182 mg/dL

**Why This Works:**
- Each question splits data into groups
- Finds the questions that best separate high vs low glucose
- Result: Clear rules like "Morning weekdays with high recent avg → expect 180+"

**Very Interpretable:**
- You can see exactly why it made that prediction
- Example: "Your glucose is predicted high because: It's morning (YES), It's a weekday (YES), Your recent average is elevated (YES)"
                ''',
                'advantages': [
                    'Easy to understand and interpret',
                    'Requires little data preparation',
                    'Handles non-linear relationships',
                    'Can visualize decision process',
                    'Works with categorical data'
                ],
                'best_for': 'Interpretable predictions',
                'simple_analogy': 'Like a choose-your-own-adventure book - follow the path to your answer'
            }
        }

    def get_algorithm_info(self):
        """Get information about the current algorithm"""
        return self.algorithms.get(self.algorithm, {})

    def get_all_algorithms(self):
        """Get information about all available algorithms"""
        return self.algorithms

    def prepare_prophet_data(self, df):
        """
        Prepare data for Prophet model
        Prophet requires columns: 'ds' (datetime) and 'y' (value)
        """
        prophet_df = pd.DataFrame({
            'ds': df.index,
            'y': df['glucose']
        })

        return prophet_df

    def train_prophet_model(self, df):
        """
        Train Facebook Prophet model
        Prophet is excellent for:
        - Capturing daily/weekly seasonality
        - Handling missing data and outliers
        - Providing uncertainty intervals
        - Making future predictions
        """
        print(f"\n[OK] Training Prophet model...")

        prophet_df = self.prepare_prophet_data(df)

        # Initialize Prophet with custom parameters
        self.prophet_model = Prophet(
            changepoint_prior_scale=0.05,  # Flexibility of trend changes
            seasonality_prior_scale=10.0,   # Strength of seasonality
            holidays_prior_scale=10.0,      # Holiday effects
            seasonality_mode='multiplicative',  # Seasonality type
            interval_width=0.95,            # 95% confidence interval
            daily_seasonality=True,         # Daily patterns
            weekly_seasonality=True,        # Weekly patterns
            yearly_seasonality=False        # Not enough data for yearly
        )

        # Add custom hourly seasonality
        self.prophet_model.add_seasonality(
            name='hourly',
            period=1,
            fourier_order=8
        )

        # Fit the model
        self.prophet_model.fit(prophet_df)

        print(f"  Prophet model trained on {len(prophet_df)} samples")

        return self.prophet_model

    def train_ml_model(self, df):
        """
        Train the selected ML model
        Uses engineered features for predictions
        """
        algo_info = self.get_algorithm_info()
        print(f"\n[OK] Training {algo_info.get('name', 'ML')} model...")

        # Prepare features (all columns except glucose)
        self.feature_cols = [col for col in df.columns if col not in ['glucose', 'glucose_std', 'glucose_min', 'glucose_max', 'reading_count']]

        if len(self.feature_cols) == 0:
            print("  Warning: No features available for ML model")
            return None

        X = df[self.feature_cols].values
        y = df['glucose'].values

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Initialize model based on algorithm
        # ADDED: LSTM and GRU Training Logic
        if self.algorithm in ['lstm', 'gru']:
            if not DL_AVAILABLE:
                raise ImportError("TensorFlow is required for LSTM/GRU. Please pip install tensorflow.")
            
            # --- 1. Scale the Target (y) to 0-1 range ---
            y_reshaped = y.reshape(-1, 1)
            y_scaled = self.y_scaler.fit_transform(y_reshaped)
            
            # --- 2. Reshape Inputs ---
            X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
            
            model = Sequential()
            
            # --- 3. Improved Deep Network Architecture ---
            if self.algorithm == 'lstm':
                # Layer 1: 128 units + Return Sequences
                model.add(LSTM(128, return_sequences=True, input_shape=(1, X_scaled.shape[1])))
                model.add(Dropout(0.2))
                # Layer 2: 64 units
                model.add(LSTM(64, return_sequences=False))
            else: # gru
                model.add(GRU(128, return_sequences=True, input_shape=(1, X_scaled.shape[1])))
                model.add(Dropout(0.2))
                model.add(GRU(64, return_sequences=False))
                
            model.add(Dropout(0.2))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(1)) # Output layer
            
            # --- 4. Slower Learning Rate for stability ---
            model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')
            
            # --- 5. Train with Early Stopping ---
            print(f"  Training Deep Learning model ({self.algorithm})...")
            early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
            
            # FIT ON SCALED TARGET (y_scaled)
            model.fit(X_reshaped, y_scaled, epochs=100, batch_size=32, verbose=0, callbacks=[early_stop])
            
            self.ml_model = model
            
            # --- 6. Calculate Score (Inverse Transform first) ---
            train_pred_scaled = model.predict(X_reshaped, verbose=0)
            train_pred = self.y_scaler.inverse_transform(train_pred_scaled).flatten()
            
            from sklearn.metrics import r2_score
            train_score = r2_score(y, train_pred)
            print(f"  Model trained (R² score: {train_score:.4f})")
            
            self.training_data = df
            return self.ml_model

        elif self.algorithm == 'linear_regression':
            self.ml_model = LinearRegression()
        elif self.algorithm == 'ridge':
            self.ml_model = Ridge(alpha=1.0, random_state=42)
        elif self.algorithm == 'lasso':
            self.ml_model = Lasso(alpha=1.0, random_state=42)
        elif self.algorithm == 'random_forest':
            self.ml_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.algorithm == 'gradient_boosting':
            self.ml_model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif self.algorithm == 'xgboost':
            if XGBOOST_AVAILABLE:
                self.ml_model = XGBRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                )
            else:
                print("  XGBoost not available, using Gradient Boosting instead")
                self.ml_model = GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                )
        # ADD THIS BLOCK:
        elif self.algorithm == 'lightgbm':
            self.ml_model = LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=20,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        # ... inside train_ml_model ...
        elif self.algorithm == 'adaboost':
            self.ml_model = AdaBoostRegressor(
                n_estimators=100,
                learning_rate=0.1,
                random_state=42
            )

        # ADD THIS BLOCK:
        elif self.algorithm == 'catboost':
            self.ml_model = CatBoostRegressor(
                iterations=500,
                learning_rate=0.05,
                depth=6,
                random_seed=42,
                verbose=0  # Silent training
            )

        elif self.algorithm == 'hist_gradient_boosting':
            self.ml_model = HistGradientBoostingRegressor(
                max_iter=100,
                learning_rate=0.1,
                max_depth=20,
                random_state=42,
                early_stopping=True
            )
            

        elif self.algorithm == 'svr':
            self.ml_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        elif self.algorithm == 'knn':
            self.ml_model = KNeighborsRegressor(n_neighbors=5)
        elif self.algorithm == 'decision_tree':
            self.ml_model = DecisionTreeRegressor(
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
        else:
            # Default to Random Forest
            self.ml_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            )

        # Train model
        self.ml_model.fit(X_scaled, y)

        # Calculate training accuracy
        train_score = self.ml_model.score(X_scaled, y)
        print(f"  Model trained (R² score: {train_score:.4f})")

        # Feature importance (for tree-based models)
        if hasattr(self.ml_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': self.ml_model.feature_importances_
            }).sort_values('importance', ascending=False)

            print(f"  Top 5 important features:")
            for idx, row in feature_importance.head(5).iterrows():
                print(f"    {row['feature']}: {row['importance']:.4f}")

        self.training_data = df

        return self.ml_model

    def predict_future(self, periods, freq='H'):
        """
        Predict future glucose levels

        Args:
            periods: Number of periods to predict
            freq: Frequency ('H' for hours, 'D' for days)

        Returns:
            DataFrame with predictions and confidence intervals
        """
        print(f"\n[OK] Generating predictions for {periods} {freq}...")

        # Check if using Prophet or ML model
        if self.algorithm == 'prophet':
            if self.prophet_model is None:
                raise ValueError("Prophet model not trained yet!")

            # Create future dataframe
            future = self.prophet_model.make_future_dataframe(
                periods=periods,
                freq=freq,
                include_history=True
            )

            # Make predictions
            forecast = self.prophet_model.predict(future)

            # Extract relevant columns
            predictions = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']].copy()
            predictions.columns = ['timestamp', 'predicted_glucose', 'lower_bound', 'upper_bound', 'trend']

        else:
            # ML model predictions
            if self.ml_model is None or self.training_data is None:
                raise ValueError("ML model not trained yet!")

            # Get last timestamp from training data
            last_timestamp = self.training_data.index[-1]

            # Generate future timestamps
            future_dates = pd.date_range(
                start=last_timestamp,
                periods=periods + len(self.training_data),
                freq=freq
            )

            # Create features for future dates
            future_df = pd.DataFrame(index=future_dates)
            future_df['hour'] = future_df.index.hour
            future_df['day_of_week'] = future_df.index.dayofweek
            future_df['day_of_month'] = future_df.index.day
            future_df['month'] = future_df.index.month
            future_df['is_weekend'] = (future_df.index.dayofweek >= 5).astype(int)

            # Use rolling average from recent history for continuous features
            recent_avg = self.training_data['glucose'].tail(24*7).mean()  # Last week average
            future_df['glucose_rolling_mean'] = recent_avg
            future_df['glucose_rolling_std'] = self.training_data['glucose'].tail(24*7).std()

            # Select only the features used in training
            available_features = [col for col in self.feature_cols if col in future_df.columns]
            missing_features = [col for col in self.feature_cols if col not in future_df.columns]

            # Fill missing features with 0 or appropriate values
            for feat in missing_features:
                future_df[feat] = 0

            # Reorder to match training features
            X_future = future_df[self.feature_cols].values

            # Scale features
            X_future_scaled = self.scaler.transform(X_future)
# Make predictions
            # ADDED: LSTM and GRU Prediction Logic
            if self.algorithm in ['lstm', 'gru']:
                # Reshape for DL
                X_future_reshaped = X_future_scaled.reshape((X_future_scaled.shape[0], 1, X_future_scaled.shape[1]))
                
                # Predict (Result is 0-1 because of MinMaxScaler)
                y_pred_scaled = self.ml_model.predict(X_future_reshaped, verbose=0)
                
                # CRITICAL FIX: Un-scale result back to real glucose values (e.g. 0.5 -> 120)
                y_pred = self.y_scaler.inverse_transform(y_pred_scaled).flatten()
            else:
                y_pred = self.ml_model.predict(X_future_scaled)

            # Create predictions dataframe
            predictions = pd.DataFrame({
                'timestamp': future_dates,
                'predicted_glucose': y_pred,
                'trend': y_pred  # For ML models, trend is same as prediction
            })

            # Calculate confidence intervals (using simple std-based approach)
            std_error = self.training_data['glucose'].std() * 0.2  # 20% of std as error estimate
            predictions['lower_bound'] = predictions['predicted_glucose'] - 2 * std_error
            predictions['upper_bound'] = predictions['predicted_glucose'] + 2 * std_error

        # Ensure predictions are within valid range
        predictions['predicted_glucose'] = predictions['predicted_glucose'].clip(lower=40, upper=400)
        predictions['lower_bound'] = predictions['lower_bound'].clip(lower=40, upper=400)
        predictions['upper_bound'] = predictions['upper_bound'].clip(lower=40, upper=400)

        print(f"  Generated {len(predictions)} predictions")
        print(f"  Prediction range: {predictions['predicted_glucose'].iloc[-periods:].min():.1f} - {predictions['predicted_glucose'].iloc[-periods:].max():.1f} mg/dL")

        return predictions

    def predict_date_range(self, start_date, end_date):
        """
        Predict glucose levels for a specific date range

        Args:
            start_date: Start date (string or datetime)
            end_date: End date (string or datetime)

        Returns:
            DataFrame with predictions for the specified range
        """
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        # Calculate periods needed
        periods = int((end_date - start_date).total_seconds() / 3600) + 1

        # Get predictions
        predictions = self.predict_future(periods=periods, freq='H')

        # Filter to requested date range
        mask = (predictions['timestamp'] >= start_date) & (predictions['timestamp'] <= end_date)
        filtered_predictions = predictions[mask].copy()

        print(f"  Filtered to {len(filtered_predictions)} predictions for date range")

        return filtered_predictions

    def evaluate_model(self, df, test_size=0.2):
        """
        Evaluate model performance using train-test split
        """
        print(f"\n[OK] Evaluating model performance...")

        # Split data
        split_idx = int(len(df) * (1 - test_size))
        train_data = df.iloc[:split_idx]
        test_data = df.iloc[split_idx:]

        # Train on training data only
        prophet_df = self.prepare_prophet_data(train_data)
        model = Prophet(
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
            seasonality_mode='multiplicative',
            interval_width=0.95,
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False
        )
        model.fit(prophet_df)

        # Predict on test data
        future = model.make_future_dataframe(periods=len(test_data), freq='H')
        forecast = model.predict(future)

        # Calculate metrics on test set
        test_predictions = forecast.iloc[-len(test_data):]['yhat'].values
        test_actual = test_data['glucose'].values

        mae = np.mean(np.abs(test_predictions - test_actual))
        rmse = np.sqrt(np.mean((test_predictions - test_actual) ** 2))
        mape = np.mean(np.abs((test_actual - test_predictions) / test_actual)) * 100

        print(f"  Test Set Performance:")
        print(f"    MAE (Mean Absolute Error): {mae:.2f} mg/dL")
        print(f"    RMSE (Root Mean Squared Error): {rmse:.2f} mg/dL")
        print(f"    MAPE (Mean Absolute Percentage Error): {mape:.2f}%")

        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'test_predictions': test_predictions,
            'test_actual': test_actual
        }
