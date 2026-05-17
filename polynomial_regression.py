# -*- coding: utf-8 -*-
"""
Created on Wed May  6 16:36:33 2026

@author: appen
"""

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
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