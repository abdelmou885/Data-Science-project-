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

plt.xlabel("Predicted Weight (kg)")
plt.ylabel("Residual (actual - predicted)")
plt.title("Residual Plot of Plain Linear Model (width only)\nU-shape = systematic error")
plt.legend()
plt.tight_layout()
plt.show()
