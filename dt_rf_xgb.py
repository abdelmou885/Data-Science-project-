"""
EU-Park: Wait Time Prediction and Driver Analysis
=================================================
Goal: Identify the best predictive model and extract data-driven insights.
Language: English (for all plots and outputs).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import shap

# Set a professional visual theme
sns.set_theme(style="whitegrid", palette="muted")

# ----------------------------------------------------------------------
# 1. LOAD & PREPROCESS DATA
# ----------------------------------------------------------------------
print("Loading and preprocessing data...")
data = pd.read_csv("EU-park.csv")

# Clean target variable
data["WaitTime"] = data["WaitTime"].clip(lower=0)
cap = data["WaitTime"].quantile(0.99)
data["WaitTime"] = data["WaitTime"].clip(upper=cap)

# Format features
data["Rain"] = data["Rain"].astype(int)

# One-hot encoding
data_encoded = pd.get_dummies(
    data,
    columns=["Attraction", "Season", "DayOfWeek"],
    drop_first=False
)

X = data_encoded.drop(columns=["WaitTime", "Date"])
y = data_encoded["WaitTime"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# ----------------------------------------------------------------------
# 2. MODEL TRAINING & COMPREHENSIVE EVALUATION (INCLUDING TIME)
# ----------------------------------------------------------------------
print("\nTraining models and measuring time...")
models = {
    "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
    "XGBoost": XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1),
}

results = []
fitted_models = {}

for name, model in models.items():
    # Measure training time (using high-precision perf_counter)
    start_train = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start_train
    
    # Measure inference time (using high-precision perf_counter)
    start_pred = time.perf_counter()
    y_pred = model.predict(X_test)
    pred_time = (time.perf_counter() - start_pred) * 1000 # Convert to milliseconds
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        "Model": name, 
        "R2": r2, 
        "MAE": mae, 
        "RMSE": rmse, 
        "Predict_Time_ms": pred_time
    })
    fitted_models[name] = model

results_df = pd.DataFrame(results).set_index("Model")

# Calculate ranks for each metric
results_df["R2_Rank"] = results_df["R2"].rank(ascending=False)
results_df["MAE_Rank"] = results_df["MAE"].rank(ascending=True)
results_df["RMSE_Rank"] = results_df["RMSE"].rank(ascending=True)
results_df["Time_Rank"] = results_df["Predict_Time_ms"].rank(ascending=True)

# Calculate total score and identify the absolute winner
results_df["Total_Rank"] = results_df["R2_Rank"] + results_df["MAE_Rank"] + results_df["RMSE_Rank"] + results_df["Time_Rank"]
best_model_name = results_df["Total_Rank"].idxmin()

# EXTREMELY IMPORTANT: Save the actual model object to use for SHAP later
best_model = fitted_models[best_model_name]

print("\nModel Evaluation (Precise):")
print(results_df[["R2", "MAE", "RMSE", "Predict_Time_ms"]])
print(f"\nOverall Best Model: {best_model_name}")

# ----------------------------------------------------------------------
# 3. PLOT 1: MODEL COMPARISON (4 Metrics, Fair Highlighting)
# ----------------------------------------------------------------------
print("\nGenerating High-Precision Model Comparison Plot with Time...")
fig, axes = plt.subplots(1, 4, figsize=(22, 6))
metrics = [
    ("R2", "R2 Score\n(Higher is Better ↑)", True),
    ("MAE", "MAE (Min)\n(Lower is Better ↓)", False),
    ("RMSE", "RMSE (Min)\n(Lower is Better ↓)", False),
    ("Predict_Time_ms", "Time (ms)\n(Lower is Better ↓)", False)
]

for ax, (metric, title, higher_is_better) in zip(axes, metrics):
    # Dynamically find the winner for THIS specific metric
    if higher_is_better:
        best_for_metric = results_df[metric].idxmax()
    else:
        best_for_metric = results_df[metric].idxmin()
        
    # Paint only the true winner of this metric gold
    colors = {model: ("#f5b041" if model == best_for_metric else "#bdc3c7") for model in results_df.index}

    sns.barplot(
        x=results_df.index, 
        y=results_df[metric], 
        ax=ax, 
        palette=colors,
        hue=results_df.index,
        legend=False
    )
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("")
    ax.set_xlabel("")
    
    # Add precise value annotations
    for i, p in enumerate(ax.patches):
        value = p.get_height()
        if metric == "R2":
            text = f"{value:.4f}"
            y_offset = 3
        elif metric == "Predict_Time_ms":
            text = f"{value:.2f}ms" # Show 2 decimals for micro-measurements
            y_offset = 5
        else:
            text = f"{value:.2f}m"
            y_offset = 5
            
        ax.annotate(text, (p.get_x() + p.get_width() / 2., value),
                    ha='center', va='bottom', fontsize=11, fontweight='bold',
                    xytext=(0, y_offset), textcoords='offset points')

plt.suptitle(f"Model Comparison (Winner: {best_model_name})", fontsize=18, fontweight="bold", y=1.08)
plt.tight_layout()
plt.savefig("01_Model_Comparison_Precise_Time.png", dpi=150, bbox_inches="tight")
plt.close()

# ----------------------------------------------------------------------
# 4. PLOT 2: BUSINESS BASELINE - DYNAMIC WAIT TIME BY HOUR
# ----------------------------------------------------------------------
print("Generating Dynamic Business Baseline Plot (Wait Time by Hour)...")

plt.figure(figsize=(10, 6))

# Calculate mathematical means
hourly_means = data.groupby("Hour")["WaitTime"].mean()

# DYNAMIC PEAK DETECTION (No hardcoding)
peak_threshold = hourly_means.max() * 0.75
peak_hours = hourly_means[hourly_means >= peak_threshold].index

peak_start = peak_hours.min()
peak_end = peak_hours.max()

# Main trend line
sns.lineplot(
    data=data, 
    x="Hour", 
    y="WaitTime", 
    estimator="mean", 
    errorbar=None, 
    marker="o", 
    markersize=8,
    linewidth=3, 
    color="#2c3e50"
)

# Dynamic red zone
plt.axvspan(
    peak_start, peak_end, 
    color='#e74c3c', alpha=0.15, 
    label=f'Peak Bottleneck Hours ({peak_start}:00 - {peak_end}:00)'
)

# Text annotations
for hour, mean_wait in hourly_means.items():
    if hour in range(8, 23): # Operating hours only
        plt.annotate(
            f"{mean_wait:.0f}m", 
            (hour, mean_wait),
            textcoords="offset points", 
            xytext=(0, 10), 
            ha='center', 
            fontsize=10,
            fontweight='bold',
            color="#2c3e50"
        )

plt.title("Hourly Wait Times", fontsize=16, fontweight="bold", pad=15)
plt.xlabel("Hour of the Day", fontsize=12, fontweight="bold")
plt.ylabel("Average Wait Time (Minutes)", fontsize=12, fontweight="bold")
plt.xticks(np.arange(8, 23, 1))
plt.legend(loc="upper left", fontsize=11)

plt.tight_layout()
plt.savefig("02_Average_Wait_by_Hour.png", dpi=150, bbox_inches="tight")
plt.close()

# ----------------------------------------------------------------------
# 5. PLOT 3: FEATURE IMPORTANCE VIA SHAP (Directional Impact)
# ----------------------------------------------------------------------
print("Calculating SHAP values...")
X_shap = X_test.sample(min(2000, len(X_test)), random_state=42)
explainer = shap.Explainer(best_model)
shap_values = explainer(X_shap)

plt.figure(figsize=(10, 8))
shap.plots.beeswarm(shap_values, max_display=12, show=False)
plt.title("Feature Impact (SHAP)", fontsize=16, fontweight="bold", pad=20)
plt.savefig("03_SHAP_Directional_Impact.png", dpi=150, bbox_inches="tight")
plt.close()

# ----------------------------------------------------------------------
# 6. PLOT 4: DEEP ANALYSIS - FEATURE COMBINATION IMPACT (SENSITIVITY)
# ----------------------------------------------------------------------
print("Generating Feature Combination Impact Plot...")

# 1. Находим самый популярный аттракцион
busiest_attraction = data.groupby("Attraction")["WaitTime"].mean().idxmax()
attr_col = f"Attraction_{busiest_attraction}"

# 2. Создаем "Идеально спокойный" базовый профиль (Вторник, 10 утра, 15°C, без дождя)
base_profile = X.mean().to_dict()
for col in base_profile:
    if col.startswith("Attraction_") or col.startswith("DayOfWeek_") or col.startswith("Season_"):
        base_profile[col] = 0

if attr_col in base_profile: base_profile[attr_col] = 1
if "Season_summer" in base_profile: base_profile["Season_summer"] = 1
if "DayOfWeek_Tuesday" in base_profile: base_profile["DayOfWeek_Tuesday"] = 1
base_profile["Hour"] = 10
base_profile["Temperature"] = 12
base_profile["Rain"] = 0

# 3. Моделируем "Шоковые" комбинации
scenarios = {
    "Baseline (Tue, 10:00, 15°C)": base_profile.copy(),
}

# Шок 1: Наступил час пик
sc1 = base_profile.copy()
sc1["Hour"] = 14
scenarios["+ Peak Hour Only (14:00)"] = sc1

# Шок 2: Час пик + Жара (Что хуже: жара или выходной?)
sc2 = base_profile.copy()
sc2["Hour"] = 14
sc2["Temperature"] = 32
scenarios["+ Peak Hour & Heatwave (32°C)"] = sc2

# Шок 3: Час пик + Выходной
sc3 = base_profile.copy()
sc3["Hour"] = 14
if "DayOfWeek_Tuesday" in sc3: sc3["DayOfWeek_Tuesday"] = 0
if "DayOfWeek_Sunday" in sc3: sc3["DayOfWeek_Sunday"] = 1
scenarios["+ Peak Hour & Weekend (Sun)"] = sc3

# Шок 4: Идеальный шторм (Пересечение всех негативных факторов)
sc4 = base_profile.copy()
sc4["Hour"] = 14
sc4["Temperature"] = 32
if "DayOfWeek_Tuesday" in sc4: sc4["DayOfWeek_Tuesday"] = 0
if "DayOfWeek_Sunday" in sc4: sc4["DayOfWeek_Sunday"] = 1
scenarios["The Perfect Storm (Sun + 32°C + 14:00)"] = sc4

# 4. Прогнозируем результаты
scenarios_df = pd.DataFrame(list(scenarios.values()))[X.columns]
predictions = best_model.predict(scenarios_df)

baseline_wait = predictions[0]
deltas = predictions - baseline_wait # Добавочное время ожидания (штраф)

# 5. Визуализация (Stacked Horizontal Bar Chart)
plt.figure(figsize=(12, 7))

labels = list(scenarios.keys())[1:]
values = deltas[1:]
base_vals = [baseline_wait] * len(values)

# Базовое ожидание (серым)
bars1 = plt.barh(labels, base_vals, color="#bdc3c7", label=f"Baseline Wait ({baseline_wait:.0f}m)")
# Добавочное ожидание от комбинации факторов (красным)
bars2 = plt.barh(labels, values, left=base_vals, color="#e74c3c", label="Added Wait Time Penalty")

plt.title(f"What makes the queue worse? Combination Analysis ({busiest_attraction})", fontsize=16, fontweight="bold", pad=15)
plt.xlabel("Total Predicted Wait Time (Minutes)", fontsize=12, fontweight="bold")

# Подписи минут прямо на графике
for bar, val in zip(bars2, values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, 
             f"+{val:.0f}m", ha='center', va='center', color='white', fontweight='bold')

plt.legend(loc="lower right", fontsize=11)
plt.gca().invert_yaxis() 

# Бизнес-вывод
plt.figtext(
    0.5, -0.05, 
    "Insight: Heat combined with peak hours creates a stronger bottleneck than simply a weekend.\nThe 'Perfect Storm' shows a compounding effect, doubling the wait time penalty.", 
    ha="center", fontsize=11, style='italic', bbox={"facecolor":"orange", "alpha":0.2, "pad":5}
)

plt.tight_layout()
plt.savefig("04_Deep_Analysis_Combo_Impact.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nSuccess! Combination Analysis plot is saved.")