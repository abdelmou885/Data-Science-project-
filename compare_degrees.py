# -*- coding: utf-8 -*-
"""
Side-by-side comparison of polynomial regression degrees 1, 2, 3.
Plots Actual vs Predicted (test set) for each degree with metrics annotated.
"""

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
