

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
