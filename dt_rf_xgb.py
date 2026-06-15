"""
EU-Park: Prediction of attraction wait times
==============================================
Goal: Predict wait times and understand what drives them.

Models:
- Decision Tree Regressor (simple, interpretable)
- Random Forest Regressor (robust baseline + feature importance)
- XGBoost Regressor (high accuracy + SHAP explainability)

Features kept simple: Hour, Month, DayOfWeek, Attraction, Season,
Temperature, Rain (one-hot encoding, no cyclical/derived features).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor
import shap

sns.set_theme(style="whitegrid")

# ----------------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------------
data = pd.read_csv("EU-park.csv")

print("Shape:", data.shape)
print(data.head())
print(data["WaitTime"].describe())

# ----------------------------------------------------------------------
# 2. PREPROCESSING / DATA CLEANING
# ----------------------------------------------------------------------

# --- 2.1 Negative wait times are impossible -> set to 0 ---
n_negative = (data["WaitTime"] < 0).sum()
print(f"\nNegative WaitTime values found: {n_negative} -> set to 0")
data["WaitTime"] = data["WaitTime"].clip(lower=0)

# --- 2.2 Outlier handling: winsorization (clip) instead of deletion ---
# IQR method would remove ~6.5% of rows (mostly legitimate "long queue" cases
# on busy days), which would distort the target distribution and remove
# exactly the high-demand situations EU-Park cares about.
# -> Clip extreme values at the 99th percentile instead of dropping rows.
Q1 = data["WaitTime"].quantile(0.25)
Q3 = data["WaitTime"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
print(f"IQR bounds: {lower_bound:.1f} - {upper_bound:.1f}")
print(f"Rows above IQR upper bound: {(data['WaitTime'] > upper_bound).sum()} "
      f"({(data['WaitTime'] > upper_bound).mean()*100:.1f}%) -> NOT removed")

cap = data["WaitTime"].quantile(0.99)
n_capped = (data["WaitTime"] > cap).sum()
print(f"Capping WaitTime at 99th percentile ({cap:.1f} min): {n_capped} rows affected")
data["WaitTime"] = data["WaitTime"].clip(upper=cap)

# --- 2.3 Missing values check ---
print("\nMissing values per column:")
print(data.isna().sum())

# --- 2.4 Rain to int ---
data["Rain"] = data["Rain"].astype(int)

# --- 2.5 Order DayOfWeek for nicer plots ---
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
data["DayOfWeek"] = pd.Categorical(data["DayOfWeek"], categories=day_order, ordered=True)

# ----------------------------------------------------------------------
# 3. EXPLORATORY PLOTS
# ----------------------------------------------------------------------

# 3.1 Distribution of WaitTime (after cleaning)
plt.figure(figsize=(8, 5))
sns.histplot(data["WaitTime"], bins=50, kde=True)
plt.title("Distribution of Wait Times (after cleaning)")
plt.xlabel("Wait Time (min)")
plt.tight_layout()
plt.savefig("plot_01_waittime_distribution.png", dpi=150)
plt.close()

# 3.2 Boxplot of WaitTime
plt.figure(figsize=(6, 5))
sns.boxplot(y=data["WaitTime"])
plt.title("Boxplot of Wait Times (after cleaning)")
plt.ylabel("Wait Time (min)")
plt.tight_layout()
plt.savefig("plot_02_waittime_boxplot.png", dpi=150)
plt.close()

# 3.3 Average wait time by hour
plt.figure(figsize=(9, 5))
sns.lineplot(data=data, x="Hour", y="WaitTime", estimator="mean", errorbar=("ci", 95))
plt.title("Average Wait Time by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Wait Time (min)")
plt.tight_layout()
plt.savefig("plot_03_waittime_by_hour.png", dpi=150)
plt.close()

# 3.4 Average wait time by day of week
plt.figure(figsize=(9, 5))
sns.barplot(data=data, x="DayOfWeek", y="WaitTime", estimator="mean", errorbar=("ci", 95))
plt.title("Average Wait Time by Day of Week")
plt.xlabel("")
plt.ylabel("Wait Time (min)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("plot_04_waittime_by_dayofweek.png", dpi=150)
plt.close()

# 3.5 Average wait time by attraction
plt.figure(figsize=(9, 5))
order = data.groupby("Attraction")["WaitTime"].mean().sort_values(ascending=False).index
sns.barplot(data=data, x="Attraction", y="WaitTime", order=order, estimator="mean")
plt.title("Average Wait Time by Attraction")
plt.xlabel("")
plt.ylabel("Wait Time (min)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("plot_05_waittime_by_attraction.png", dpi=150)
plt.close()

# 3.6 Wait time vs Temperature, split by season
plt.figure(figsize=(9, 5))
sns.scatterplot(data=data.sample(min(5000, len(data)), random_state=42),
                 x="Temperature", y="WaitTime", hue="Season", alpha=0.4)
plt.title("Wait Time vs Temperature")
plt.tight_layout()
plt.savefig("plot_06_waittime_vs_temperature.png", dpi=150)
plt.close()

# 3.7 Wait time: rain vs no rain
plt.figure(figsize=(6, 5))
sns.boxplot(data=data, x="Rain", y="WaitTime")
plt.title("Wait Time: Rain vs No Rain")
plt.xticks([0, 1], ["No Rain", "Rain"])
plt.tight_layout()
plt.savefig("plot_07_waittime_rain.png", dpi=150)
plt.close()

# 3.8 Heatmap: average wait time by hour and day of week
pivot = data.pivot_table(values="WaitTime", index="DayOfWeek", columns="Hour",
                          aggfunc="mean", observed=False)
plt.figure(figsize=(12, 5))
sns.heatmap(pivot, cmap="YlOrRd")
plt.title("Average Wait Time: Day of Week x Hour")
plt.tight_layout()
plt.savefig("plot_08_heatmap_day_hour.png", dpi=150)
plt.close()

print("\nExploratory plots saved.")

# ----------------------------------------------------------------------
# 4. ENCODING + FEATURE/TARGET SETUP (kept simple, no derived features)
# ----------------------------------------------------------------------

data_encoded = pd.get_dummies(
    data,
    columns=["Attraction", "Season", "DayOfWeek"],
    drop_first=False
)

X = data_encoded.drop(columns=["WaitTime", "Date"])
y = data_encoded["WaitTime"]

print(f"\nNumber of features after encoding: {X.shape[1]}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# ----------------------------------------------------------------------
# 5. MODELS
# ----------------------------------------------------------------------

models = {
    "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
    ),
    "XGBoost": XGBRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        random_state=42, n_jobs=-1
    ),
}

results = {}
fitted_models = {}

# Baseline: predict the mean
baseline_pred = np.repeat(y_train.mean(), len(y_test))
mae_baseline = mean_absolute_error(y_test, baseline_pred)
print(f"\nBaseline (predict mean) MAE: {mae_baseline:.2f} min")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    improvement = (mae_baseline - mae) / mae_baseline * 100

    results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2, "Improvement_vs_baseline_%": improvement}
    fitted_models[name] = model

    print(f"\n{name}")
    print(f"  MAE  : {mae:.2f} min")
    print(f"  RMSE : {rmse:.2f} min")
    print(f"  R2   : {r2:.3f}")
    print(f"  Improvement vs baseline: {improvement:.1f}%")

results_df = pd.DataFrame(results).T
print("\nModel comparison:")
print(results_df)

# ----------------------------------------------------------------------
# 6. MODEL COMPARISON PLOT
# ----------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, metric in zip(axes, ["MAE", "RMSE", "R2"]):
    sns.barplot(x=results_df.index, y=results_df[metric], ax=ax)
    ax.set_title(metric)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
plt.suptitle("Model Performance Comparison")
plt.tight_layout()
plt.savefig("plot_09_model_comparison.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------
# 7. ACTUAL VS PREDICTED (best model)
# ----------------------------------------------------------------------

best_model_name = results_df["R2"].idxmax()
print(f"\nBest model based on R2: {best_model_name}")
best_model = fitted_models[best_model_name]
y_pred_best = best_model.predict(X_test)

plt.figure(figsize=(7, 7))
sample_idx = np.random.choice(len(y_test), size=min(3000, len(y_test)), replace=False)
plt.scatter(y_test.iloc[sample_idx], y_pred_best[sample_idx], alpha=0.3)
max_val = max(y_test.max(), y_pred_best.max())
plt.plot([0, max_val], [0, max_val], "r--")
plt.xlabel("Actual Wait Time (min)")
plt.ylabel("Predicted Wait Time (min)")
plt.title(f"Actual vs Predicted Wait Time - {best_model_name}")
plt.tight_layout()
plt.savefig("plot_10_actual_vs_predicted.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------
# 8. FEATURE IMPORTANCE - Random Forest (built-in)
# ----------------------------------------------------------------------

rf_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": fitted_models["Random Forest"].feature_importances_
}).sort_values("Importance", ascending=False)

print("\n=== Top 10 features - Random Forest ===")
print(rf_importance.head(10))

plt.figure(figsize=(10, 7))
plt.barh(rf_importance["Feature"].head(15), rf_importance["Importance"].head(15))
plt.gca().invert_yaxis()
plt.title("Feature Importance - Random Forest")
plt.tight_layout()
plt.savefig("plot_11_feature_importance_rf.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------
# 9. FEATURE IMPORTANCE - XGBoost (total gain)
# ----------------------------------------------------------------------

xgb_model = fitted_models["XGBoost"]
importance = xgb_model.get_booster().get_score(importance_type="total_gain")
imp_frame = pd.DataFrame(importance.items(), columns=["Feature", "Importance"])
imp_frame = imp_frame.sort_values("Importance", ascending=False)

print("\n=== Top 10 features - XGBoost (total gain) ===")
print(imp_frame.head(10))

plt.figure(figsize=(10, 7))
plt.barh(imp_frame["Feature"].head(15), imp_frame["Importance"].head(15))
plt.gca().invert_yaxis()
plt.title("Feature Importance - XGBoost (Total Gain)")
plt.tight_layout()
plt.savefig("plot_12_feature_importance_xgb.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------
# 10. PERMUTATION IMPORTANCE (XGBoost, on a sample for speed)
# ----------------------------------------------------------------------

# Use a subsample of the test set - permutation importance is expensive
# (n_repeats * n_features full predict passes)
sample_size = min(5000, len(X_test))
X_test_sample = X_test.sample(sample_size, random_state=42)
y_test_sample = y_test.loc[X_test_sample.index]

perm_importance = permutation_importance(
    xgb_model, X_test_sample, y_test_sample,
    n_repeats=5, random_state=42, scoring="neg_mean_absolute_error", n_jobs=-1
)

perm_df = pd.DataFrame({
    "Feature": X_test_sample.columns,
    "Importance": perm_importance.importances_mean
}).sort_values("Importance", ascending=False)

print("\n=== Top 10 features - Permutation Importance (XGBoost) ===")
print(perm_df.head(10))

plt.figure(figsize=(10, 7))
plt.barh(perm_df["Feature"].head(15), perm_df["Importance"].head(15))
plt.gca().invert_yaxis()
plt.title("Permutation Importance - XGBoost (sample of test set)")
plt.tight_layout()
plt.savefig("plot_13_permutation_importance.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------
# 11. SHAP VALUES (XGBoost, on a sample for speed)
# ----------------------------------------------------------------------

shap_sample = X_test.sample(min(1000, len(X_test)), random_state=42)
explainer = shap.Explainer(xgb_model)
shap_values = explainer(shap_sample)

# Beeswarm: overall feature impact
plt.figure()
shap.plots.beeswarm(shap_values, max_display=15, show=False)
plt.tight_layout()
plt.savefig("plot_14_shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()

# Waterfall: explanation for a single prediction
plt.figure()
shap.plots.waterfall(shap_values[0], max_display=10, show=False)
plt.tight_layout()
plt.savefig("plot_15_shap_waterfall_example.png", dpi=150, bbox_inches="tight")
plt.close()

# ----------------------------------------------------------------------
# 12. DECISION TREE VISUALIZATION (interpretable model)
# ----------------------------------------------------------------------

plt.figure(figsize=(20, 10))
plot_tree(
    fitted_models["Decision Tree"],
    feature_names=X.columns,
    filled=True,
    max_depth=3,
    fontsize=8
)
plt.title("Decision Tree (top levels) - Wait Time Prediction")
plt.tight_layout()
plt.savefig("plot_16_decision_tree.png", dpi=150)
plt.close()

print("\nAll plots saved. Done.")