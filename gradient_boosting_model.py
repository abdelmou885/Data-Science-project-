# -*- coding: utf-8 -*-
"""
Created on Wed May  6 17:08:11 2026

@author: appen
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
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

# --------------------------------------------------
# 1. Create Gradient Boosting model
# --------------------------------------------------

gb_model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

# --------------------------------------------------
# 2. Train model
# --------------------------------------------------

gb_model.fit(X_train, y_train)

# --------------------------------------------------
# 3. Predict
# --------------------------------------------------

y_pred_train_gb = gb_model.predict(X_train)
y_pred_test_gb = gb_model.predict(X_test)

# --------------------------------------------------
# 4. Evaluate
# --------------------------------------------------

mae_train_gb = mean_absolute_error(y_train, y_pred_train_gb)
mae_test_gb = mean_absolute_error(y_test, y_pred_test_gb)

rmse_train_gb = np.sqrt(mean_squared_error(y_train, y_pred_train_gb))
rmse_test_gb = np.sqrt(mean_squared_error(y_test, y_pred_test_gb))

r2_train_gb = r2_score(y_train, y_pred_train_gb)
r2_test_gb = r2_score(y_test, y_pred_test_gb)

print("Gradient Boosting Regressor")
print("Training MAE:", mae_train_gb)
print("Test MAE:", mae_test_gb)
print("Training RMSE:", rmse_train_gb)
print("Test RMSE:", rmse_test_gb)
print("Training R²:", r2_train_gb)
print("Test R²:", r2_test_gb)

# --------------------------------------------------
# 5. Actual vs predicted plot
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred_test_gb, alpha=0.5)

plt.xlabel("Actual weight_in_kg")
plt.ylabel("Predicted weight_in_kg")
plt.title("Gradient Boosting Regressor: Actual vs Predicted Weight")

min_value = min(y_test.min(), y_pred_test_gb.min())
max_value = max(y_test.max(), y_pred_test_gb.max())

plt.plot([min_value, max_value], [min_value, max_value], color="red")

plt.show()

# --------------------------------------------------
# 6. Feature importance
# --------------------------------------------------

feature_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": gb_model.feature_importances_
})

feature_importance = feature_importance.sort_values("importance", ascending=False)

print(feature_importance)

plt.figure(figsize=(10, 6))

plt.barh(feature_importance["feature"], feature_importance["importance"])

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Gradient Boosting Feature Importance")
plt.gca().invert_yaxis()

plt.show()