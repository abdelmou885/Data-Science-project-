"""
EU-Park Wait Time Prediction - Regression Models
Student Project: Comparing different regression approaches

We're testing 3 models:
- Linear Regression
- Polynomial Degree 2
- Polynomial Degree 3

Goal: See which one works best for predicting wait times
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)

print("="*80)
print("Regression Models Comparison for Wait Time Prediction")
print("="*80)

# Load the data
print("\nLoading dataset...")
data_path = r"C:\uni wokr\datascine\learning data science 2\EU-park.csv"
df = pd.read_csv(data_path)
print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")

# Data preprocessing
print("\nPreparing data...")
df_model = df.copy()

# Drop date column since it's too specific
df_model = df_model.drop(['Date'], axis=1)

# Convert boolean to numeric
df_model['Rain'] = df_model['Rain'].map({False: 0, True: 1})

# Encode categorical variables
categorical_cols = ['Season', 'DayOfWeek', 'Attraction']
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df_model[col + '_encoded'] = le.fit_transform(df_model[col])
    encoders[col] = le

df_model = df_model.drop(categorical_cols, axis=1)

# Split into features and target
X = df_model.drop(['WaitTime'], axis=1)
y = df_model['WaitTime']

print(f"Features: {X.shape[1]}, Total samples: {X.shape[0]}")

# Train-test split (80-20)
print("\nSplitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# Train models
print("\nTraining models...")
models = {}
predictions = {}
metrics = {}

# Linear model
print("  Training Linear Regression...")
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
models['Linear (Degree 1)'] = linear_model

y_train_pred = linear_model.predict(X_train)
y_test_pred = linear_model.predict(X_test)
predictions['Linear (Degree 1)'] = {'train': y_train_pred, 'test': y_test_pred}

# Calculate metrics
metrics['Linear (Degree 1)'] = {
    'train_r2': r2_score(y_train, y_train_pred),
    'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
    'train_mae': mean_absolute_error(y_train, y_train_pred),
    'test_r2': r2_score(y_test, y_test_pred),
    'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
    'test_mae': mean_absolute_error(y_test, y_test_pred),
}

# Cross validation
cv_r2 = cross_val_score(linear_model, X, y, cv=5, scoring='r2')
cv_rmse = -cross_val_score(linear_model, X, y, cv=5, scoring='neg_root_mean_squared_error')
metrics['Linear (Degree 1)']['cv_r2_mean'] = cv_r2.mean()
metrics['Linear (Degree 1)']['cv_r2_std'] = cv_r2.std()
metrics['Linear (Degree 1)']['cv_rmse_mean'] = cv_rmse.mean()
metrics['Linear (Degree 1)']['cv_rmse_std'] = cv_rmse.std()

print(f"    Done. Test R2: {metrics['Linear (Degree 1)']['test_r2']:.4f}")

# Polynomial models (degree 2 and 3)
for degree in [2, 3]:
    model_name = f'Polynomial (Degree {degree})'
    print(f"  Training Polynomial Regression (degree {degree})...")

    # Create pipeline
    poly_model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('linear', LinearRegression())
    ])

    poly_model.fit(X_train, y_train)
    models[model_name] = poly_model

    y_train_pred = poly_model.predict(X_train)
    y_test_pred = poly_model.predict(X_test)
    predictions[model_name] = {'train': y_train_pred, 'test': y_test_pred}

    # Calculate metrics
    metrics[model_name] = {
        'train_r2': r2_score(y_train, y_train_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
    }

    # Cross validation
    cv_r2 = cross_val_score(poly_model, X, y, cv=5, scoring='r2')
    cv_rmse = -cross_val_score(poly_model, X, y, cv=5, scoring='neg_root_mean_squared_error')
    metrics[model_name]['cv_r2_mean'] = cv_r2.mean()
    metrics[model_name]['cv_r2_std'] = cv_r2.std()
    metrics[model_name]['cv_rmse_mean'] = cv_rmse.mean()
    metrics[model_name]['cv_rmse_std'] = cv_rmse.std()

    print(f"    Done. Test R2: {metrics[model_name]['test_r2']:.4f}")

print("\nAll models trained!")

# Compare results
print("\n" + "="*80)
print("Results:")
print("="*80)

model_names = ['Linear (Degree 1)', 'Polynomial (Degree 2)', 'Polynomial (Degree 3)']

for model_name in model_names:
    m = metrics[model_name]
    print(f"\n{model_name}:")
    print(f"  Train R2: {m['train_r2']:.4f}, Test R2: {m['test_r2']:.4f}")
    print(f"  Test RMSE: {m['test_rmse']:.2f} min, Test MAE: {m['test_mae']:.2f} min")
    print(f"  CV R2: {m['cv_r2_mean']:.4f} (+/- {m['cv_r2_std']*2:.4f})")

# Find best model
best_model = max(metrics.keys(), key=lambda k: metrics[k]['test_r2'])
print(f"\nBest model: {best_model}")
print(f"Test R2: {metrics[best_model]['test_r2']:.4f}")
print(f"Test RMSE: {metrics[best_model]['test_rmse']:.2f} minutes")

# Create visualizations
print("\nCreating plots...")
output_path = r"C:\uni wokr\datascine\learning data science 2\analysis"

# Plot 1: Actual vs Predicted
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for idx, model_name in enumerate(model_names):
    y_pred = predictions[model_name]['test']
    m = metrics[model_name]

    axes[idx].scatter(y_test, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5, s=20)
    axes[idx].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                   'r--', lw=2, label='Perfect Prediction')
    axes[idx].set_xlabel('Actual Wait Time (minutes)', fontsize=11)
    axes[idx].set_ylabel('Predicted Wait Time (minutes)', fontsize=11)
    axes[idx].set_title(f'{model_name}\nR2 = {m["test_r2"]:.4f}, RMSE = {m["test_rmse"]:.2f}',
                       fontsize=11, fontweight='bold')
    axes[idx].legend(fontsize=9)
    axes[idx].grid(True, alpha=0.3)

plt.suptitle('Actual vs Predicted Wait Times - Model Comparison (Test Set)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"{output_path}/combined_actual_vs_predicted_all_models.png", dpi=300)
print("  Saved: combined_actual_vs_predicted_all_models.png")
plt.close()

# Plot 2: Metrics comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

metric_list = [
    ('test_r2', 'R2 Score (Test Set)', True),
    ('test_rmse', 'RMSE - minutes (Test Set)', False),
    ('test_mae', 'MAE - minutes (Test Set)', False),
    ('cv_r2_mean', 'Cross-Validation R2', True)
]

colors = ['#3498db', '#e74c3c', '#2ecc71']

for idx, (metric_key, title, higher_better) in enumerate(metric_list):
    ax = axes[idx // 2, idx % 2]

    values = [metrics[name][metric_key] for name in model_names]
    bars = ax.bar(range(len(model_names)), values, color=colors,
                  edgecolor='black', alpha=0.7, width=0.6)

    # Highlight best
    best_idx = values.index(max(values) if higher_better else min(values))
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)

    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(['Linear\n(Deg 1)', 'Polynomial\n(Deg 2)', 'Polynomial\n(Deg 3)'], fontsize=10)
    ax.set_ylabel(title, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add values on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}' if 'r2' in metric_key.lower() else f'{val:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.suptitle('Performance Metrics Comparison', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f"{output_path}/combined_metrics_comparison_all_models.png", dpi=300)
print("  Saved: combined_metrics_comparison_all_models.png")
plt.close()

print("\nDone!")
print("="*80)
