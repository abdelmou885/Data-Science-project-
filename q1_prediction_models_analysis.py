"""
===============================================================================
Q1 - PREDICTION MODELS ANALYSIS
===============================================================================

This file consolidates all analysis related to the PREDICTION MODELS developed
for Question 1 (Q1) of the project.

The code and results that follow belong to the prediction models built for Q1,
including the data preparation, model training, evaluation, and the comparison
of the different approaches that were explored:

    - Linear Regression
    - Polynomial Regression
    - Decision Tree
    - Gradient Boosting
    - XGBoost

Everything below this header is part of the Q1 prediction models analysis and
has been merged into this single file for submission.

Ps: before running the code make sure to adjust the file paths for the datasets (Production_Log_01.csv and
===============================================================================
"""


# =============================================================================
# 1. LINEAR REGRESSION
# =============================================================================

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#reading datasets
production_log = pd.read_csv('C:/Users/appen/OneDrive/Área de Trabalho/Furtwangen/05_Summer Semester_2026/Data Science/SbA1/Production_Log_01.csv')
machine_settings_log =  pd.read_csv('C:/Users/appen/OneDrive/Área de Trabalho/Furtwangen/05_Summer Semester_2026/Data Science/SbA1/Machine_Settings_Log_01.csv')

#merging production and machine settings
df = production_log.merge(machine_settings_log, on="configuration_log_ID", how="left")

df_clean = df.copy()

# Remove extreme / unrealistic width values
df_clean = df_clean[df_clean["width"] < 1000]

# Optional: remove extreme target outliers as well
df_clean = df_clean[df_clean["weight_in_kg"] < df_clean["weight_in_kg"].quantile(0.99)]

print("Original data:", df.shape)
print("Cleaned data:", df_clean.shape)

print(df_clean["weight_in_kg"].describe())
print(df_clean["width"].describe())

#First predictive model. Linear regression: weight in kg

# Target variable: what we want to predict
target = "weight_in_kg"

# Features: information available before or during production
features = [
    "width",
    "height",
    "ionizationclass",
    "FluxCompensation",
    "pressure",
    "karma",
    "modulation",
    "gear",
    "rotation_speed"
]

X = df_clean[features]
y = df_clean[target]

# Convert categorical variables into dummy variables
X = pd.get_dummies(X, drop_first=True)

print(X.head())
print(X.columns)

#train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training data:", X_train.shape)
print("Test data:", X_test.shape)

model = linear_model.LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Show intercept and coefficients
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

#make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

#evaluate the model
mae_train = mean_absolute_error(y_train, y_pred_train)
mae_test = mean_absolute_error(y_test, y_pred_test)

rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print("Training MAE:", mae_train)
print("Test MAE:", mae_test)

print("Training RMSE:", rmse_train)
print("Test RMSE:", rmse_test)

print("Training R²:", r2_train)
print("Test R²:", r2_test)

#visualize actual vs predicted weight
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test, alpha=0.5)

plt.xlabel("Actual weight_in_kg")
plt.ylabel("Predicted weight_in_kg")
plt.title("Linear Regression: Actual vs Predicted Weight")

# Reference line: perfect predictions
min_value = min(y_test.min(), y_pred_test.min())
max_value = max(y_test.max(), y_pred_test.max())
plt.plot([min_value, max_value], [min_value, max_value], color="red")

plt.show()


# =============================================================================
# 2. LINEAR REGRESSION WITH SUBSTITUTION
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import pandas as pd


#reading datasets
production_log = pd.read_csv('Production_Log_01.csv')
machine_settings_log =  pd.read_csv('Machine_Settings_Log_01.csv')

#merging production and machine settings
df = production_log.merge(machine_settings_log, on="configuration_log_ID", how="left")

df_clean = df.copy()

# Remove extreme / unrealistic width values
df_clean = df_clean[df_clean["width"] < 1000]

# Optional: remove extreme target outliers as well
df_clean = df_clean[df_clean["weight_in_kg"] < df_clean["weight_in_kg"].quantile(0.99)]
# --------------------------------------------------
# Linear Regression with substitution
# Predict weight_in_kg using width and width^2
# --------------------------------------------------

x = df_clean[["width"]]
z = df_clean["weight_in_kg"]

model = linear_model.LinearRegression()

# Create substituted variable: width squared
x2 = x["width"] ** 2

# Create dataframe like professor's example
model_data = pd.DataFrame()
model_data["width"] = x["width"]
model_data["width2"] = x2

# Train model
model.fit(model_data, z)

print("Coefficients:")
print(model.coef_)

print("Intercept:")
print(model.intercept_)

# Create values for plotting
x_plot = np.linspace(df_clean["width"].min(), df_clean["width"].max(), 100)

plot_data = pd.DataFrame()
plot_data["width"] = x_plot
plot_data["width2"] = x_plot ** 2

# Predict weight for the plot line
z_predicted = model.predict(plot_data)

plt.figure(figsize=(8, 6))
plt.scatter(df_clean["width"], df_clean["weight_in_kg"], alpha=0.4)
plt.plot(x_plot, z_predicted, color="red")

plt.xlabel("width")
plt.ylabel("weight_in_kg")
plt.title("Linear Regression with Substitution: width and width²")

plt.show()


# =============================================================================
# 3. POLYNOMIAL REGRESSION
# =============================================================================

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#reading datasets
production_log = pd.read_csv('Production_Log_01.csv')
machine_settings_log =  pd.read_csv('Machine_Settings_Log_01.csv')

#merging production and machine settings
df = production_log.merge(machine_settings_log, on="configuration_log_ID", how="left")

df_clean = df.copy()

# Remove extreme / unrealistic width values
df_clean = df_clean[df_clean["width"] < 1000]

# Optional: remove extreme target outliers as well
df_clean = df_clean[df_clean["weight_in_kg"] < df_clean["weight_in_kg"].quantile(0.99)]

print("Original data:", df.shape)
print("Cleaned data:", df_clean.shape)

print(df_clean["weight_in_kg"].describe())
print(df_clean["width"].describe())

#First predictive model. Linear regression: weight in kg

# Target variable: what we want to predict
target = "weight_in_kg"

# Features: information available before or during production
features = [
    "width",
    "height",
    "ionizationclass",
    "FluxCompensation",
    "pressure",
    "karma",
    "modulation",
    "gear",
    "rotation_speed"
]

X = df_clean[features]
y = df_clean[target]

# Convert categorical variables into dummy variables
X = pd.get_dummies(X, drop_first=True)

print(X.head())
print(X.columns)

#train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training data:", X_train.shape)
print("Test data:", X_test.shape)

results = []

for degree in [1, 2, 3]:

    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=False)

    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Train linear regression model on polynomial features
    model = linear_model.LinearRegression()
    model.fit(X_train_poly, y_train)

    # Predictions
    y_pred_train = model.predict(X_train_poly)
    y_pred_test = model.predict(X_test_poly)

    # Evaluation
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)

    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    results.append({
        "degree": degree,
        "train_MAE": mae_train,
        "test_MAE": mae_test,
        "train_RMSE": rmse_train,
        "test_RMSE": rmse_test,
        "train_R2": r2_train,
        "test_R2": r2_test,
        "number_of_features": X_train_poly.shape[1]
    })

results_df = pd.DataFrame(results)
print(results_df)

best_degree = results_df.sort_values("test_MAE").iloc[0]["degree"]

best_degree = int(best_degree)

print("Best polynomial degree based on test MAE:", best_degree)

#train final polynomial model
poly = PolynomialFeatures(degree=best_degree, include_bias=False)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, y_train)

y_pred_train = poly_model.predict(X_train_poly)
y_pred_test = poly_model.predict(X_test_poly)

#evaluate final polynomial model
mae_train = mean_absolute_error(y_train, y_pred_train)
mae_test = mean_absolute_error(y_test, y_pred_test)

rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print("Polynomial degree:", best_degree)

print("Training MAE:", mae_train)
print("Test MAE:", mae_test)

print("Training RMSE:", rmse_train)
print("Test RMSE:", rmse_test)

print("Training R²:", r2_train)
print("Test R²:", r2_test)

#plot actual vs predicted values
plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred_test, alpha=0.5)

plt.xlabel("Actual weight_in_kg")
plt.ylabel("Predicted weight_in_kg")
plt.title(f"Polynomial Regression Degree {best_degree}: Actual vs Predicted Weight")

min_value = min(y_test.min(), y_pred_test.min())
max_value = max(y_test.max(), y_pred_test.max())

plt.plot([min_value, max_value], [min_value, max_value], color="red")

plt.show()

# Choose degree 2 manually
degree_2 = 2

# Create polynomial features for degree 2
poly_degree_2 = PolynomialFeatures(degree=degree_2, include_bias=False)

X_train_poly_2 = poly_degree_2.fit_transform(X_train)
X_test_poly_2 = poly_degree_2.transform(X_test)

# Train degree 2 polynomial regression model
poly_model_2 = linear_model.LinearRegression()
poly_model_2.fit(X_train_poly_2, y_train)

# Make predictions
y_pred_train_2 = poly_model_2.predict(X_train_poly_2)
y_pred_test_2 = poly_model_2.predict(X_test_poly_2)

# Evaluate model
mae_train_2 = mean_absolute_error(y_train, y_pred_train_2)
mae_test_2 = mean_absolute_error(y_test, y_pred_test_2)

rmse_train_2 = np.sqrt(mean_squared_error(y_train, y_pred_train_2))
rmse_test_2 = np.sqrt(mean_squared_error(y_test, y_pred_test_2))

r2_train_2 = r2_score(y_train, y_pred_train_2)
r2_test_2 = r2_score(y_test, y_pred_test_2)

print("Polynomial degree:", degree_2)
print("Training MAE:", mae_train_2)
print("Test MAE:", mae_test_2)
print("Training RMSE:", rmse_train_2)
print("Test RMSE:", rmse_test_2)
print("Training R²:", r2_train_2)
print("Test R²:", r2_test_2)
print("Number of features:", X_train_poly_2.shape[1])

#plot for degree 2
plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred_test_2, alpha=0.5)

plt.xlabel("Actual weight_in_kg")
plt.ylabel("Predicted weight_in_kg")
plt.title("Recommended Model: Polynomial Regression Degree 2")

min_value = min(y_test.min(), y_pred_test_2.min())
max_value = max(y_test.max(), y_pred_test_2.max())

plt.plot([min_value, max_value], [min_value, max_value], color="red")

plt.show()


# =============================================================================
# 4. POLYNOMIAL DEGREE COMPARISON (degrees 1, 2, 3 side by side)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Load and clean (same pipeline as polynomial_regression.py) ---
production_log = pd.read_csv('Production_Log_01.csv')
machine_settings_log = pd.read_csv('Machine_Settings_Log_01.csv')

df = production_log.merge(machine_settings_log, on="configuration_log_ID", how="left")
df_clean = df.copy()
df_clean = df_clean[df_clean["width"] < 1000]
df_clean = df_clean[df_clean["weight_in_kg"] < df_clean["weight_in_kg"].quantile(0.99)]

target = "weight_in_kg"
features = [
    "width", "height", "ionizationclass", "FluxCompensation",
    "pressure", "karma", "modulation", "gear", "rotation_speed",
]

X = pd.get_dummies(df_clean[features], drop_first=True)
y = df_clean[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Fit each degree and collect test predictions + metrics ---
degrees = [1, 2, 3]
preds = {}
metrics = {}

for degree in degrees:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = linear_model.LinearRegression()
    model.fit(X_train_poly, y_train)

    y_pred_test = model.predict(X_test_poly)
    preds[degree] = y_pred_test
    metrics[degree] = {
        "MAE": mean_absolute_error(y_test, y_pred_test),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_test)),
        "R2": r2_score(y_test, y_pred_test),
        "n_feat": X_test_poly.shape[1],
    }

# --- Side-by-side plot ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

lo = float(y_test.min())
hi = float(y_test.max())

for ax, degree in zip(axes, degrees):
    y_pred = preds[degree]
    m = metrics[degree]

    ax.scatter(y_test, y_pred, alpha=0.4, s=15, edgecolor="none")
    ax.plot([lo, hi], [lo, hi], color="red", lw=2, label="Perfect prediction")

    ax.set_title(f"Degree {degree}  ({m['n_feat']} features)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Actual weight_in_kg")
    if degree == degrees[0]:
        ax.set_ylabel("Predicted weight_in_kg")

    text = (
        f"MAE  = {m['MAE']:,.0f}\n"
        f"RMSE = {m['RMSE']:,.0f}\n"
        f"R²   = {m['R2']:.5f}"
    )
    ax.text(
        0.05, 0.95, text, transform=ax.transAxes,
        va="top", ha="left", fontsize=11, family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax.legend(loc="lower right", fontsize=9)

fig.suptitle("Polynomial Regression: Actual vs Predicted (Test Set)", fontsize=15, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("degree_comparison.png", dpi=120)
print("Saved figure to degree_comparison.png")
plt.show()


# =============================================================================
# 5. RESIDUAL PLOT (plain linear model: width only)
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import pandas as pd

# --------------------------------------------------
# Read & merge data (same as the main script)
# --------------------------------------------------
production_log = pd.read_csv('Production_Log_01.csv')
machine_settings_log = pd.read_csv('Machine_Settings_Log_01.csv')

df = production_log.merge(machine_settings_log, on="configuration_log_ID", how="left")

df_clean = df.copy()
df_clean = df_clean[df_clean["width"] < 1000]
df_clean = df_clean[df_clean["weight_in_kg"] < df_clean["weight_in_kg"].quantile(0.99)]

# --------------------------------------------------
# Fit a PLAIN linear model: weight ~ width  (no width^2)
# This is the model that makes systematic mistakes,
# so its residuals reveal the U-shape.
# --------------------------------------------------
x = df_clean[["width"]]
z = df_clean["weight_in_kg"]

model = linear_model.LinearRegression()
model.fit(x, z)

# Predictions and residuals (residual = actual - predicted)
z_predicted = model.predict(x)
residuals = z - z_predicted

# --------------------------------------------------
# Residual plot: residuals vs predicted values
# A good model -> random scatter around zero.
# Here -> a clear U-shape (curvature the line cannot capture).
# --------------------------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(z_predicted, residuals, alpha=0.4)
plt.axhline(y=0, color="red", linewidth=2)

# Smooth trend line through the residuals to make the U-shape obvious
order = np.argsort(z_predicted)
trend = np.poly1d(np.polyfit(z_predicted, residuals, 2))
plt.plot(z_predicted[order], trend(z_predicted[order]),
         color="orange", linewidth=2, label="Residual trend (U-shape)")

plt.xlabel("Predicted weight_in_kg")
plt.ylabel("Residual (actual - predicted)")
plt.title("Residual Plot of Plain Linear Model (width only)\nU-shape = systematic error")
plt.legend()
plt.tight_layout()
plt.show()


# =============================================================================
# 6. RESIDUALS COMPARISON (before vs after adding width²)
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import pandas as pd

# --------------------------------------------------
# Read & merge data (same as the main script)
# --------------------------------------------------
production_log = pd.read_csv('Production_Log_01.csv')
machine_settings_log = pd.read_csv('Machine_Settings_Log_01.csv')

df = production_log.merge(machine_settings_log, on="configuration_log_ID", how="left")

df_clean = df.copy()
df_clean = df_clean[df_clean["width"] < 1000]
df_clean = df_clean[df_clean["weight_in_kg"] < df_clean["weight_in_kg"].quantile(0.99)]

z = df_clean["weight_in_kg"]


def residuals_for(features):
    """Fit a linear model on the given feature frame and return (predicted, residuals)."""
    m = linear_model.LinearRegression()
    m.fit(features, z)
    pred = m.predict(features)
    return pred, z - pred


# --------------------------------------------------
# BEFORE: plain model  weight ~ width
# AFTER:  substitution model  weight ~ width + width^2
# --------------------------------------------------
plain_features = df_clean[["width"]]

quad_features = pd.DataFrame()
quad_features["width"] = df_clean["width"]
quad_features["width2"] = df_clean["width"] ** 2

pred_plain, res_plain = residuals_for(plain_features)
pred_quad, res_quad = residuals_for(quad_features)


def plot_residuals(ax, pred, res, title):
    ax.scatter(pred, res, alpha=0.4)
    ax.axhline(y=0, color="red", linewidth=2)
    order = np.argsort(pred)
    trend = np.poly1d(np.polyfit(pred, res, 2))
    ax.plot(pred[order], trend(pred[order]), color="orange",
            linewidth=2, label="Residual trend")
    ax.set_xlabel("Predicted Weight (kg)")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title(title)
    ax.legend()


fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
plot_residuals(axes[0], pred_plain, res_plain,
               "BEFORE: width only\nU-shape = systematic error")
plot_residuals(axes[1], pred_quad, res_quad,
               "AFTER: width + width²\nflattened = curvature captured")

fig.suptitle("Residuals: adding width² removes the U-shape", fontsize=14)
plt.tight_layout()
plt.show()


# =============================================================================
# 7. DECISION TREE & XGBOOST (SmartBuild weight predictor)
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import time

# just some styling for the plots
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 8)

print("SmartBuild Weight Prediction")
print("-" * 40)

# load the data files
print("\nLoading data...")

production_data = pd.read_csv('Production_Log_01.csv')
machine_settings = pd.read_csv('Machine_Settings_Log_01.csv')
data = production_data.merge(machine_settings, on='configuration_log_ID', how='left')

print(f"Loaded {len(data):,} production records")

# these are measured AFTER production so we cant use them - would be cheating
post_production_features = [
    'weight_in_kg',      # this is what we're trying to predict
    'weight_in_g',       # same thing just in grams
    'distortion',
    'roughness',
    'nicesness',         # found this has 99.5% correlation with width, definitely post-production
    'smartness',
    'multideminsionality',
    'reflectionScore',
    'Quality',
    'error',
    'error_type'
]

# these are the features we CAN use - they're all known before production starts
pre_production_features = [
    'width', 'height',  # input dimensions
    'pressure', 'karma', 'modulation',  # machine settings
    'ionizationclass', 'FluxCompensation',
    'gear', 'rotation_speed'  # machine config
]

print(f"\nUsing {len(pre_production_features)} pre-production features")
print("Excluded post-production features to avoid data leakage")

# clean up outliers using IQR method
print("\nCleaning data...")

Q1 = data['weight_in_kg'].quantile(0.25)
Q3 = data['weight_in_kg'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 3 * IQR
upper_bound = Q3 + 3 * IQR

data_clean = data[(data['weight_in_kg'] >= lower_bound) & (data['weight_in_kg'] <= upper_bound)].copy()
outliers_removed = len(data) - len(data_clean)

print(f"Removed {outliers_removed} extreme outliers")
print(f"Clean dataset: {len(data_clean):,} samples")

# encode the categorical features
data_clean['ionizationclass'] = data_clean['ionizationclass'].map({'A': 0, 'B': 1, 'C': 2})
data_clean['FluxCompensation'] = data_clean['FluxCompensation'].map({'I': 1, 'II': 2, 'III': 3, 'IV': 4})

# drop any rows with missing values in our features
data_clean = data_clean.dropna(subset=pre_production_features + ['weight_in_kg'])

# split into features and target
X = data_clean[pre_production_features]
y = data_clean['weight_in_kg']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining samples: {len(X_train):,}")
print(f"Testing samples: {len(X_test):,}")

# baseline - just predicting the mean every time
print("\nCalculating baseline (mean prediction)...")

baseline_pred = np.full(len(y_test), y_train.mean())
baseline_mae = mean_absolute_error(y_test, baseline_pred)
baseline_r2 = r2_score(y_test, baseline_pred)

print(f"Baseline MAE: {baseline_mae:.0f} kg, R2: {baseline_r2:.4f}")

# train decision tree
print("\nTraining Decision Tree...")

start = time.time()
dt_model = DecisionTreeRegressor(max_depth=10, min_samples_split=20, min_samples_leaf=10, random_state=42)
dt_model.fit(X_train, y_train)
dt_time = time.time() - start

dt_train_pred = dt_model.predict(X_train)
dt_test_pred = dt_model.predict(X_test)

dt_train_r2 = r2_score(y_train, dt_train_pred)
dt_test_r2 = r2_score(y_test, dt_test_pred)
dt_mae = mean_absolute_error(y_test, dt_test_pred)
dt_rmse = np.sqrt(mean_squared_error(y_test, dt_test_pred))

dt_cv_scores = cross_val_score(dt_model, X_train, y_train, cv=5, scoring='r2')
dt_cv_mean = dt_cv_scores.mean()
dt_cv_std = dt_cv_scores.std()

print(f"Training time: {dt_time:.2f}s")
print(f"Train R2: {dt_train_r2:.4f} ({dt_train_r2*100:.2f}%)")
print(f"Test R2: {dt_test_r2:.4f} ({dt_test_r2*100:.2f}%)")
print(f"Cross-Val R2: {dt_cv_mean:.4f} +/- {dt_cv_std:.4f}")
print(f"MAE: {dt_mae:.0f} kg, RMSE: {dt_rmse:.0f} kg")

# train xgboost - usually does better
print("\nTraining XGBoost...")

start = time.time()
xgb_model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
xgb_model.fit(X_train, y_train)
xgb_time = time.time() - start

xgb_train_pred = xgb_model.predict(X_train)
xgb_test_pred = xgb_model.predict(X_test)

xgb_train_r2 = r2_score(y_train, xgb_train_pred)
xgb_test_r2 = r2_score(y_test, xgb_test_pred)
xgb_mae = mean_absolute_error(y_test, xgb_test_pred)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_test_pred))

xgb_cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='r2')
xgb_cv_mean = xgb_cv_scores.mean()
xgb_cv_std = xgb_cv_scores.std()

print(f"Training time: {xgb_time:.2f}s")
print(f"Train R2: {xgb_train_r2:.4f} ({xgb_train_r2*100:.2f}%)")
print(f"Test R2: {xgb_test_r2:.4f} ({xgb_test_r2*100:.2f}%)")
print(f"Cross-Val R2: {xgb_cv_mean:.4f} +/- {xgb_cv_std:.4f}")
print(f"MAE: {xgb_mae:.0f} kg, RMSE: {xgb_rmse:.0f} kg")

# results comparison
print("\n" + "-" * 40)
print("RESULTS SUMMARY")
print("-" * 40)

comparison = pd.DataFrame({
    'Metric': ['R2 Score (Test)', 'R2 (Cross-Val)', 'MAE (kg)', 'RMSE (kg)', 'Training Time (s)'],
    'Baseline': [f"{baseline_r2:.4f}", "N/A", f"{baseline_mae:.0f}", "N/A", "0.00"],
    'Decision Tree': [
        f"{dt_test_r2:.4f}",
        f"{dt_cv_mean:.4f}+/-{dt_cv_std:.4f}",
        f"{dt_mae:.0f}",
        f"{dt_rmse:.0f}",
        f"{dt_time:.2f}"
    ],
    'XGBoost': [
        f"{xgb_test_r2:.4f}",
        f"{xgb_cv_mean:.4f}+/-{xgb_cv_std:.4f}",
        f"{xgb_mae:.0f}",
        f"{xgb_rmse:.0f}",
        f"{xgb_time:.2f}"
    ]
})

print("\n" + comparison.to_string(index=False))

# feature importance
print("\n" + "-" * 40)
print("FEATURE IMPORTANCE")
print("-" * 40)

importance_df = pd.DataFrame({
    'Feature': pre_production_features,
    'Decision_Tree': dt_model.feature_importances_,
    'XGBoost': xgb_model.feature_importances_
}).sort_values('XGBoost', ascending=False)

print("\n" + importance_df.to_string(index=False, float_format='%.4f'))

# check for overfitting
dt_gap = dt_train_r2 - dt_test_r2
xgb_gap = xgb_train_r2 - xgb_test_r2

print("\n" + "-" * 40)
print("OVERFITTING CHECK")
print("-" * 40)
print(f"Decision Tree gap: {dt_gap:.4f} ({dt_gap*100:.2f}%) - {'OK' if dt_gap < 0.05 else 'might be overfitting'}")
print(f"XGBoost gap: {xgb_gap:.4f} ({xgb_gap*100:.2f}%) - {'OK' if xgb_gap < 0.05 else 'might be overfitting'}")

# final summary
print("\n" + "-" * 40)
print("DONE!")
print("-" * 40)
print(f"Dataset: {len(data_clean):,} samples")
print(f"Features: {len(pre_production_features)} pre-production only")
print(f"Best Model: {'XGBoost' if xgb_test_r2 > dt_test_r2 else 'Decision Tree'}")
print(f"Best R2: {max(xgb_test_r2, dt_test_r2):.4f} ({max(xgb_test_r2, dt_test_r2)*100:.2f}%)")
print(f"Best MAE: {min(xgb_mae, dt_mae):.0f} kg")

# now make some plots
print("\nGenerating visualizations...")

# plot 1 - model comparison (4 panels)
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Model Performance Comparison (No Data Leakage)', fontsize=16, fontweight='bold')

# r2 comparison
ax1 = axes[0, 0]
models = ['Baseline', 'Decision Tree', 'XGBoost']
r2_scores = [baseline_r2, dt_test_r2, xgb_test_r2]
colors = ['#ff7f0e', '#2ca02c', '#1f77b4']
bars = ax1.bar(models, r2_scores, color=colors, alpha=0.7, edgecolor='black')
ax1.set_ylabel('R2 Score', fontsize=12, fontweight='bold')
ax1.set_title('R2 Score Comparison (Test Set)', fontsize=13, fontweight='bold')
ax1.set_ylim([min(r2_scores)-0.05, 1.0])
ax1.grid(axis='y', alpha=0.3)
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{r2_scores[i]:.4f}\n({r2_scores[i]*100:.2f}%)',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# mae comparison
ax2 = axes[0, 1]
mae_scores = [baseline_mae, dt_mae, xgb_mae]
bars = ax2.bar(models, mae_scores, color=colors, alpha=0.7, edgecolor='black')
ax2.set_ylabel('Mean Absolute Error (kg)', fontsize=12, fontweight='bold')
ax2.set_title('MAE Comparison (Test Set)', fontsize=13, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{mae_scores[i]:.0f} kg',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# feature importance
ax3 = axes[1, 0]
importance_sorted = importance_df.sort_values('XGBoost', ascending=True).tail(10)
y_pos = np.arange(len(importance_sorted))
ax3.barh(y_pos, importance_sorted['XGBoost'], alpha=0.7, label='XGBoost', color='#1f77b4')
ax3.barh(y_pos, importance_sorted['Decision_Tree'], alpha=0.5, label='Decision Tree', color='#2ca02c')
ax3.set_yticks(y_pos)
ax3.set_yticklabels(importance_sorted['Feature'])
ax3.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
ax3.set_title('Top 10 Feature Importance', fontsize=13, fontweight='bold')
ax3.legend()
ax3.grid(axis='x', alpha=0.3)

# actual vs predicted scatter
ax4 = axes[1, 1]
ax4.scatter(y_test, xgb_test_pred, alpha=0.5, s=20, color='#1f77b4', label='XGBoost')
ax4.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
ax4.set_xlabel('Actual Weight (kg)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Predicted Weight (kg)', fontsize=12, fontweight='bold')
ax4.set_title('Actual vs Predicted Weight (XGBoost)', fontsize=13, fontweight='bold')
ax4.legend()
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison_visualization.png', dpi=300, bbox_inches='tight')
print("Saved: model_comparison_visualization.png")

# plot 2 - residuals
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle('Prediction Residuals Analysis', fontsize=16, fontweight='bold')

dt_residuals = y_test - dt_test_pred
ax1 = axes[0]
ax1.scatter(dt_test_pred, dt_residuals, alpha=0.5, s=20, color='#2ca02c')
ax1.axhline(y=0, color='r', linestyle='--', lw=2)
ax1.set_xlabel('Predicted Weight (kg)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Residuals (kg)', fontsize=12, fontweight='bold')
ax1.set_title(f'Decision Tree Residuals (MAE: {dt_mae:.0f} kg)', fontsize=13, fontweight='bold')
ax1.grid(alpha=0.3)

xgb_residuals = y_test - xgb_test_pred
ax2 = axes[1]
ax2.scatter(xgb_test_pred, xgb_residuals, alpha=0.5, s=20, color='#1f77b4')
ax2.axhline(y=0, color='r', linestyle='--', lw=2)
ax2.set_xlabel('Predicted Weight (kg)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Residuals (kg)', fontsize=12, fontweight='bold')
ax2.set_title(f'XGBoost Residuals (MAE: {xgb_mae:.0f} kg)', fontsize=13, fontweight='bold')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('residuals_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: residuals_analysis.png")

# plot 3 - cross validation scores boxplot
fig, ax = plt.subplots(figsize=(10, 6))
cv_data = {
    'Decision Tree': dt_cv_scores,
    'XGBoost': xgb_cv_scores
}
positions = [1, 2]
bp = ax.boxplot([dt_cv_scores, xgb_cv_scores], positions=positions, widths=0.6,
                 patch_artist=True, labels=['Decision Tree', 'XGBoost'])
colors = ['#2ca02c', '#1f77b4']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel('R2 Score', fontsize=12, fontweight='bold')
ax.set_title('5-Fold Cross-Validation Scores Distribution', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for i, (model_name, scores) in enumerate(cv_data.items(), 1):
    ax.text(i, scores.mean(), f'{scores.mean():.4f}+/-{scores.std():.4f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig('cross_validation_scores.png', dpi=300, bbox_inches='tight')
print("Saved: cross_validation_scores.png")

# plot 4 - metrics heatmap
fig, ax = plt.subplots(figsize=(10, 6))
metrics_data = pd.DataFrame({
    'Decision Tree': [dt_test_r2, dt_cv_mean, dt_mae, dt_rmse, dt_time],
    'XGBoost': [xgb_test_r2, xgb_cv_mean, xgb_mae, xgb_rmse, xgb_time]
}, index=['R2 (Test)', 'R2 (CV)', 'MAE (kg)', 'RMSE (kg)', 'Time (s)'])

im = ax.imshow(metrics_data.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=metrics_data.values.max())
ax.set_xticks(np.arange(len(metrics_data.columns)))
ax.set_yticks(np.arange(len(metrics_data.index)))
ax.set_xticklabels(metrics_data.columns)
ax.set_yticklabels(metrics_data.index)

for i in range(len(metrics_data.index)):
    for j in range(len(metrics_data.columns)):
        text = ax.text(j, i, f'{metrics_data.values[i, j]:.2f}',
                      ha="center", va="center", color="black", fontweight='bold')

ax.set_title('Model Metrics Heatmap', fontsize=14, fontweight='bold')
fig.colorbar(im, ax=ax, label='Value')
plt.tight_layout()
plt.savefig('metrics_heatmap.png', dpi=300, bbox_inches='tight')
print("Saved: metrics_heatmap.png")

print("\nAll done! Generated 4 visualization files.")
plt.show()
