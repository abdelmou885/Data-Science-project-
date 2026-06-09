import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, roc_curve, auc, roc_auc_score
)
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. LOAD & MERGE DATASETS
# ==========================================
prod = pd.read_csv('Production_Log_01.csv')
machine = pd.read_csv('Machine_Settings_Log_01.csv')

prod.columns = prod.columns.str.strip()
machine.columns = machine.columns.str.strip()

df = prod.merge(machine, on='configuration_log_ID', how='left')
print(f"[1] Loaded {df.shape[0]} records, {df.shape[1]} columns.")

# ==========================================
# 2. DATA CLEANING
# ==========================================
df = df[df['error'].notna() & ~df['error'].isin(['None', 'none', ''])]
df = df[df['width'] < 1000]

print(f"[2] Cleaned dataset: {df.shape[0]} records.")
print(f"    Error distribution: {df['error'].value_counts().to_dict()}\n")

# ==========================================
# 3. FEATURE SELECTION & ENGINEERING
# ==========================================
base_features = [
    'width', 'height', 'ionizationclass', 'FluxCompensation',
    'pressure', 'karma', 'modulation', 'gear', 'rotation_speed'
]

X = df[base_features].copy()
y_raw = df['error'].copy()

# Feature Engineering
X['width_height'] = X['width'] * X['height']
X['width_pressure'] = X['width'] * X['pressure']
X['width_karma'] = X['width'] * X['karma']
X['width_ratio'] = X['width'] / (X['height'] + 1e-6)
X['pressure_mod_ratio'] = X['pressure'] / (X['modulation'] + 1e-6)
X['volume_proxy'] = X['width'] * X['height'] * X['rotation_speed']

print(f"[3] Engineered {len(X.columns)} features (pre-production only).")

num_cols = X.select_dtypes(include='number').columns.tolist()
cat_cols = X.select_dtypes(include='object').columns.tolist()

le = LabelEncoder()
y = le.fit_transform(y_raw)
print(f"    Target mapping: {dict(zip(le.classes_, [0, 1]))}")

# ==========================================
# 4. PREPROCESSING PIPELINE
# ==========================================
numeric_pipe = Pipeline([('imputer', SimpleImputer(strategy='median'))])
categorical_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipe, num_cols),
    ('cat', categorical_pipe, cat_cols)
])

# ==========================================
# 5. TRAIN/TEST SPLIT
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"[5] Split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")

# ==========================================
# 6. TRAIN ALL MODELS IN ONE LOOP
# ==========================================
models = {
    "Decision Tree": DecisionTreeClassifier(
        max_depth=8,
        min_samples_leaf=5,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        min_samples_leaf=5,
        subsample=0.9,
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42
    ),
    "MLP Neural Network": MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation='relu',
        solver='adam',
        alpha=0.001,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        verbose=False
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight='balanced',
        solver='lbfgs',
        random_state=42
    )
}

import time
results = []
trained_pipelines = {}

predictions_df = X_test.copy()
predictions_df["Actual_Error"] = y_test
predictions_df["Actual_Error_Label"] = predictions_df["Actual_Error"].map({0: "no", 1: "yes"})

for name, model in models.items():
    if name in ("MLP Neural Network", "Logistic Regression"):
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("scaler", StandardScaler()),
            ("classifier", model)
        ])
    else:
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])

    print(f"\nTraining {name}...")
    t_start = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - t_start
    trained_pipelines[name] = pipeline
    print(f"  Training time: {train_time:.2f}s")

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_proba)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "F1 Score": f1,
        "ROC AUC": auc_score,
        "Train Time (s)": round(train_time, 2)
    })

    predictions_df[f"{name}_Label"] = pd.Series(y_pred).map({0: "no", 1: "yes"}).values
    predictions_df[f"{name}_Probability_Error"] = y_proba

    print(f"\n=== {name.upper()} RESULTS ===")
    print(classification_report(y_test, y_pred, target_names=["no (OK)", "yes (Faulty)"], zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{pd.DataFrame(cm, index=['Actual: no', 'Actual: yes'], columns=['Pred: no', 'Pred: yes'])}")

# ==========================================
# 7. MODEL COMPARISON
# ==========================================
comparison_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)

print("\n=== MODEL COMPARISON (with training time) ===")
print(comparison_df.to_string(index=False))

print(f"Fastest Model: {comparison_df.sort_values('Train Time (s)').iloc[0]['Model']} "
      f"({comparison_df['Train Time (s)'].min():.2f}s)")
print(f"Slowest Model: {comparison_df.sort_values('Train Time (s)').iloc[-1]['Model']} "
      f"({comparison_df['Train Time (s)'].max():.2f}s)")

plt.figure(figsize=(13, 6))
plt.bar(comparison_df["Model"], comparison_df["Accuracy"], color='steelblue', edgecolor='black')
plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.title("Accuracy Comparison: 6 Models (DT, RF, GB, XGBoost, MLP, Logistic Regression)")
y_min = max(0, comparison_df["Accuracy"].min() - 0.05)
plt.ylim(y_min, 1.0)
plt.xticks(rotation=15, ha='right')
plt.grid(axis="y", alpha=0.3)
for i, v in enumerate(comparison_df["Accuracy"]):
    plt.text(i, v + 0.002, f"{v:.5f}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig('plot_01_accuracy_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("[6] Saved: plot_01_accuracy_comparison.png")

# Training time comparison
time_df = comparison_df.sort_values("Train Time (s)", ascending=False)
plt.figure(figsize=(10, 4))
bars = plt.barh(time_df["Model"], time_df["Train Time (s)"], color='coral', edgecolor='black')
for bar, val in zip(bars, time_df["Train Time (s)"]):
    plt.text(val + 0.05, bar.get_y() + bar.get_height()/2,
             f"{val:.2f}s", va='center', fontsize=8, fontweight='bold')
plt.xlabel("Training Time (seconds)", fontsize=9)
plt.title("Training Time Comparison: 6 Models", fontsize=11)
plt.yticks(fontsize=8)
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig('plot_02_training_time.png', dpi=150, bbox_inches='tight')
plt.close()
print("[7] Saved: plot_02_training_time.png")

# ==========================================
# 8. DETAILED EVALUATION — LOGISTIC REGRESSION
# ==========================================
lr_pipeline = trained_pipelines["Logistic Regression"]

y_pred = lr_pipeline.predict(X_test)
y_proba = lr_pipeline.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n=== LOGISTIC REGRESSION — DETAILED RESULTS ===")
print(classification_report(y_test, y_pred, target_names=['no (OK)', 'yes (Faulty)'], zero_division=0))
print(f"Accuracy: {acc:.3f} | F1: {f1:.3f}")

# ROC Curve — Logistic Regression
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('plot_03_roc_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("[8] Saved: plot_03_roc_curve.png")

# ==========================================
# 9. FEATURE IMPORTANCE — LOGISTIC REGRESSION
# ==========================================
cat_feature_names = lr_pipeline.named_steps['preprocessor'].named_transformers_['cat']\
    .named_steps['onehot'].get_feature_names_out(cat_cols)
all_features = num_cols + cat_feature_names.tolist()

classifier = lr_pipeline.named_steps['classifier']
importances = np.abs(classifier.coef_[0])

feat_df = pd.DataFrame({'feature': all_features, 'importance': importances})\
    .sort_values('importance', ascending=False)

print("\n=== TOP 10 FEATURE IMPORTANCE (Logistic Regression) ===")
print(feat_df.head(10).to_string(index=False))

plt.figure(figsize=(8, 6))
sns.barplot(data=feat_df.head(10), x='importance', y='feature')
plt.title('Top 10 Feature Importance (Logistic Regression)')
plt.xlabel('|Coefficient|')
plt.tight_layout()
plt.savefig('plot_04_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("[9] Saved: plot_04_feature_importance.png")

# ==========================================
# 10. EXTENDED MODEL COMPARISON (F1 + AUC)
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
metrics = ["Accuracy", "F1 Score", "ROC AUC"]
colors = ["dodgerblue", "hotpink", "aquamarine"]

for ax, metric, color in zip(axes, metrics, colors):
    sorted_df = comparison_df.sort_values(by=metric, ascending=False)
    ax.bar(sorted_df["Model"], sorted_df[metric], color=color, edgecolor='black', alpha=0.85)
    y_min = max(0, sorted_df[metric].min() - 0.05)
    ax.set_ylim(y_min, 0.96)
    ax.set_title(f"{metric} Comparison", fontsize=12, fontweight='bold')
    ax.set_ylabel(metric, fontsize=10, labelpad=8, rotation=90, va='center')
    ax.set_xticks(range(len(sorted_df)))
    ax.set_xticklabels(sorted_df["Model"], rotation=30, ha='right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(sorted_df[metric]):
        ax.text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=8)

plt.suptitle("6-Model Comparison: Accuracy, F1 Score, ROC AUC", fontsize=14, fontweight='bold')
plt.subplots_adjust(bottom=0.22, left=0.06, wspace=0.35)
plt.savefig('plot_05_extended_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("[10] Saved: plot_05_extended_comparison.png")

# ==========================================
# 11. ESTIMATED SAVINGS — LOGISTIC REGRESSION
# ==========================================
cost_faulty  = 150  # EUR: faulty product slips through (FN)
cost_discard = 10   # EUR: good product discarded by mistake (FP)

n_faulty        = y_test.sum()
status_quo_cost = n_faulty * cost_faulty

lr_model_cost = fp * cost_discard + fn * cost_faulty
lr_savings    = status_quo_cost - lr_model_cost

print("\n" + "="*60)
print("ESTIMATED SAVINGS — LOGISTIC REGRESSION")
print("="*60)
print(f"  Test set size:          {len(y_test)} samples")
print(f"  Actual faulty products: {n_faulty}")
print(f"")
print(f"  --- Without model (status quo) ---")
print(f"  All faulty products slip through")
print(f"  Status quo cost:        {status_quo_cost} EUR")
print(f"")
print(f"  --- With Logistic Regression ---")
print(f"  TP (correctly flagged faulty): {tp}")
print(f"  FP (good → wrongly discarded): {fp}  × {cost_discard} EUR = {fp * cost_discard} EUR")
print(f"  FN (faulty → missed):          {fn}  × {cost_faulty} EUR = {fn * cost_faulty} EUR")
print(f"  Total model cost:       {lr_model_cost} EUR")
print(f"")
print(f"  Estimated savings:      {lr_savings} EUR")
print(f"  Savings per sample:     {lr_savings / len(y_test):.2f} EUR")
print(f"  Cost reduction:         {lr_savings / status_quo_cost * 100:.1f}%")
print("="*60)

# Bar chart: status quo vs model cost
plt.figure(figsize=(7, 5))
bars = plt.bar(
    ["Status Quo\n(No Model)", "Logistic Regression"],
    [status_quo_cost, lr_model_cost],
    color=["#E74C3C", "#2ECC71"],
    edgecolor="black",
    width=0.5
)
for bar, val in zip(bars, [status_quo_cost, lr_model_cost]):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
             f"{val} EUR", ha="center", fontsize=11, fontweight="bold")
plt.ylabel("Total Cost (EUR)")
plt.title(f"Estimated Savings with Logistic Regression\nsavings: {lr_savings} EUR  ({lr_savings/status_quo_cost*100:.1f}% reduction)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig('plot_06_savings_logistic_regression.png', dpi=150, bbox_inches='tight')
plt.close()
print("[11] Saved: plot_06_savings_logistic_regression.png")

