# -*- coding: utf-8 -*-
"""
Created on Fri May  1 15:23:11 2026

@author: appen
"""

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





