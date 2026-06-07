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
