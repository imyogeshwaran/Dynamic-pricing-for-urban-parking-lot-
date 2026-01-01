🚗 Parking Demand & Dynamic Pricing Model
📌 Project Overview
This project implements a full end-to-end Machine Learning pipeline to model parking demand and predict dynamic parking prices based on real-world factors such as occupancy, queue length, traffic conditions, time patterns, vehicle type, and special days.

The system combines:

Explainable rule-based demand pricing (baseline)

Advanced ML regression models for accurate price prediction

Strict data leakage prevention and time-based evaluation

🎯 Objective (Target Data Mining Task)
To predict parking price (continuous value) at a given time and location using historical parking and contextual data.

Why this matters:
Helps optimize parking revenue

Reduces congestion during peak hours

Enables fair and demand-aware pricing

Supports smart-city transportation planning

🧠 Key Outcomes
Built a robust ML pipeline covering the complete data science lifecycle

Achieved high predictive accuracy using Gradient Boosting and Random Forest

Ensured realistic evaluation using time-based train/validation/test splits

Prevented target leakage by separating demand-construction logic from ML features

Enabled manual prediction testing (interactive, CSV, and programmatic inputs)

Sample Test Performance (after leakage fixes):
Model	RMSE	MAE	R²
Gradient Boosting	~0.03–0.04	~0.02	~0.99
Random Forest	~0.04–0.05	~0.03	~0.99

⚠️ High accuracy is expected because prices are generated using structured demand logic; strict safeguards are applied to keep evaluation honest.

🔧 Pipeline Components
This project includes all major data mining and ML stages:

Data Cleaning

Missing value handling

Outlier treatment (IQR-based)

Timestamp standardization

Feature Engineering

Lag features (time-series safe)

Rolling statistics

Interaction features

Peak-hour indicators

Exploratory Data Analysis

Correlation analysis

Distribution plots

Residual analysis

Hypothesis Testing

T-test (Weekend vs Weekday pricing)

ANOVA (Day-of-week impact)

Chi-square tests (categorical relationships)

Feature Selection

Univariate statistical selection

Random Forest importance-based selection

Dimensionality Reduction

PCA (for analysis and variance understanding)

Proximity Measures

Euclidean

Cosine similarity

Mahalanobis distance

k-Nearest Neighbors (training-only)

Modeling

Random Forest Regressor

Gradient Boosting Regressor

Evaluation

RMSE, MAE, R², MAPE, sMAPE

Time-based validation

Inference & Manual Testing

Interactive input (Jupyter-friendly)

CSV-based batch prediction

Python dictionary input

📥 How to Test the Model (Jupyter Friendly)
Option 1: Interactive Input
Run the notebook/script and choose:

mathematica
Copy code
Manual testing options:
1) Interactive prompt
Enter values manually or press Enter to use defaults.

Option 2: CSV Input
Provide a CSV with feature columns:

python
Copy code
manual_predict(final_model, final_features, proximity_info, train_data, csv_path="sample_input.csv")
Option 3: Python Dictionary
python
Copy code
samples = [{
    "OccupancyRate_lag1": 0.75,
    "QueueNorm": 0.6,
    "TrafficLevel": 2,
    "IsWeekend": 0,
    "Hour": 18
}]
manual_predict(final_model, final_features, proximity_info, train_data, samples=samples)
📦 Libraries & Tools Used
Core Libraries
Python 3

NumPy

Pandas

Visualization
Matplotlib

Seaborn

Machine Learning
scikit-learn

RandomForestRegressor

GradientBoostingRegressor

PCA

SelectKBest

NearestNeighbors

Statistics
SciPy

