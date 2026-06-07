# -*- coding: utf-8 -*-
"""
Q2 Alternative: Binary Classifier for 'error' (yes/no)
Predicts if a product will be faulty BEFORE machine usage.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
# 6. TRAIN ALL MODELS IN ONE LOOP (no duplicates)
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

results = []
trained_pipelines = {}  # store trained pipelines for later use

predictions_df = X_test.copy()
predictions_df["Actual_Error"] = y_test
predictions_df["Actual_Error_Label"] = predictions_df["Actual_Error"].map({0: "no", 1: "yes"})

for name, model in models.items():
    # MLP and Logistic Regression require feature scaling
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
    pipeline.fit(X_train, y_train)
    trained_pipelines[name] = pipeline  # save for reuse

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_proba)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "F1 Score": f1,
        "ROC AUC": auc_score
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

print("\n=== MODEL ACCURACY COMPARISON ===")
print(comparison_df.to_string(index=False))

best_model_name = comparison_df.iloc[0]["Model"]
best_accuracy = comparison_df.iloc[0]["Accuracy"]
print(f"\nBest Model: {best_model_name}")
print(f"Best Accuracy: {best_accuracy:.3f}")

plt.figure(figsize=(13, 6))
bars = plt.bar(comparison_df["Model"], comparison_df["Accuracy"], color='steelblue', edgecolor='black')
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
plt.show()

# ==========================================
# 8. DETAILED EVALUATION ON BEST MODEL
# ==========================================
best_pipeline = trained_pipelines[best_model_name]

y_pred = best_pipeline.predict(X_test)
y_proba = best_pipeline.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n=== BEST MODEL: {best_model_name} ===")
print(classification_report(y_test, y_pred, target_names=['no (OK)', 'yes (Faulty)'], zero_division=0))
print(f"Accuracy: {acc:.3f} | F1: {f1:.3f}")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve - {best_model_name}')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ==========================================
# 9. BUSINESS IMPACT / EXPECTED VALUE
# ==========================================
def calculate_expected_loss(threshold, y_true, y_prob):
    y_pred_thresh = (y_prob >= threshold).astype(int)
    cm_thresh = confusion_matrix(y_true, y_pred_thresh)
    tn_t, fp_t, fn_t, tp_t = cm_thresh.ravel()
    loss = (tp_t + fp_t) * 10 + fn_t * 150
    return loss, tn_t, fp_t, fn_t, tp_t

base_loss, _, _, _, _ = calculate_expected_loss(0.5, y_test, y_proba)
print("\n=== BUSINESS IMPACT (Default Threshold 0.5) ===")
print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
print(f"Estimated Loss: {base_loss:.0f} EUR")
print(f"Loss per sample: {base_loss/len(y_test):.2f} EUR")

# Threshold Optimization
thresh_range = np.linspace(0.05, 0.95, 100)
losses = [calculate_expected_loss(t, y_test, y_proba)[0] for t in thresh_range]

best_idx = np.argmin(losses)
best_thresh = thresh_range[best_idx]
best_loss = losses[best_idx]
_, tn_opt, fp_opt, fn_opt, tp_opt = calculate_expected_loss(best_thresh, y_test, y_proba)

print(f"\n=== OPTIMIZED BUSINESS IMPACT (Threshold = {best_thresh:.2f}) ===")
print(f"TN: {tn_opt}, FP: {fp_opt}, FN: {fn_opt}, TP: {tp_opt}")
print(f"Optimized Loss: {best_loss:.0f} EUR")
print(f"Loss per sample: {best_loss/len(y_test):.2f} EUR")
print(f"Improvement: {base_loss - best_loss:.0f} EUR saved on test set")

plt.figure(figsize=(8, 5))
plt.plot(thresh_range, losses, color='crimson', lw=2)
plt.axvline(x=best_thresh, color='green', linestyle='--', label=f'Optimal Threshold ({best_thresh:.2f})')
plt.xlabel('Prediction Threshold')
plt.ylabel('Expected Financial Loss (EUR)')
plt.title('Expected Loss vs Decision Threshold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ==========================================
# 10. FEATURE IMPORTANCE (best model only)
# ==========================================
cat_feature_names = best_pipeline.named_steps['preprocessor'].named_transformers_['cat']\
    .named_steps['onehot'].get_feature_names_out(cat_cols)
all_features = num_cols + cat_feature_names.tolist()

classifier = best_pipeline.named_steps['classifier']

if hasattr(classifier, 'feature_importances_'):
    importances = classifier.feature_importances_
elif hasattr(classifier, 'coef_'):
    importances = np.abs(classifier.coef_[0])
else:
    importances = np.zeros(len(all_features))

feat_df = pd.DataFrame({'feature': all_features, 'importance': importances})\
    .sort_values('importance', ascending=False)

print("\n=== TOP 10 FEATURE IMPORTANCE ===")
print(feat_df.head(10).to_string(index=False))

plt.figure(figsize=(8, 6))
sns.barplot(data=feat_df.head(10), x='importance', y='feature')
plt.title(f'Top 10 Feature Importance ({best_model_name})')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()

# ==========================================
# 11. SAMPLE PREDICTIONS TABLE
# ==========================================
print("\n=== SAMPLE PREDICTIONS (first 30) ===")
print(predictions_df[[
    "Actual_Error_Label",
    "Decision Tree_Label", "Decision Tree_Probability_Error",
    "Random Forest_Label", "Random Forest_Probability_Error",
    "Gradient Boosting_Label", "Gradient Boosting_Probability_Error",
    "XGBoost_Label", "XGBoost_Probability_Error",
    "MLP Neural Network_Label", "MLP Neural Network_Probability_Error",
    "Logistic Regression_Label", "Logistic Regression_Probability_Error"
]].head(30).to_string())

# ==========================================
# 11b. EXTENDED MODEL COMPARISON (F1 + AUC)
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
metrics = ["Accuracy", "F1 Score", "ROC AUC"]
colors = ["steelblue", "darkorange", "seagreen"]

for ax, metric, color in zip(axes, metrics, colors):
    sorted_df = comparison_df.sort_values(by=metric, ascending=False)
    bars = ax.bar(sorted_df["Model"], sorted_df[metric], color=color, edgecolor='black', alpha=0.85)
    y_min = max(0, sorted_df[metric].min() - 0.05)
    ax.set_ylim(y_min, 1.0)
    ax.set_title(f"{metric} Comparison", fontsize=12, fontweight='bold')
    ax.set_ylabel(metric)
    ax.tick_params(axis='x', rotation=20)
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(sorted_df[metric]):
        ax.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=8)

plt.suptitle("6-Model Comparison: Accuracy, F1 Score, ROC AUC", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ==========================================
# 12. EXECUTIVE SUMMARY
# ==========================================
print("\n" + "="*60)
print("EXECUTIVE SUMMARY (Q2 Alternative: Binary Error Prediction)")
print("="*60)
print(f"  TASK: Predict if product will be faulty (yes/no) BEFORE machine runs.")
print(f"  BEST MODEL: {best_model_name}")
print(f"  PERFORMANCE: Accuracy {acc:.1%}, ROC-AUC {roc_auc:.3f}")
print(f"  BUSINESS: Default threshold loss = {base_loss/len(y_test):.2f} EUR/sample")
print(f"  OPTIMIZED: Threshold {best_thresh:.2f} reduces loss to {best_loss/len(y_test):.2f} EUR/sample")
print(f"  TOP FEATURE: {feat_df.iloc[0]['feature']} (importance: {feat_df.iloc[0]['importance']:.3f})")
print("="*60)

# ==========================================
# 13. ESTIMATED SAVINGS — LOGISTIC REGRESSION
# ==========================================
cost_faulty  = 150  # EUR: faulty product slips through (FN)
cost_discard = 10   # EUR: good product discarded by mistake (FP)

lr_pipeline = trained_pipelines["Logistic Regression"]
lr_pred     = lr_pipeline.predict(X_test)
lr_proba    = lr_pipeline.predict_proba(X_test)[:, 1]

lr_cm               = confusion_matrix(y_test, lr_pred)
lr_tn, lr_fp, lr_fn, lr_tp = lr_cm.ravel()

# Baseline: no model at all — every faulty product slips through
n_faulty        = y_test.sum()
status_quo_cost = n_faulty * cost_faulty

# With Logistic Regression
lr_model_cost = lr_fp * cost_discard + lr_fn * cost_faulty
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
print(f"  TP (correctly flagged faulty): {lr_tp}")
print(f"  FP (good → wrongly discarded): {lr_fp}  × {cost_discard} EUR = {lr_fp * cost_discard} EUR")
print(f"  FN (faulty → missed):          {lr_fn}  × {cost_faulty} EUR = {lr_fn * cost_faulty} EUR")
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
plt.ylabel("Total Cost  (EUR)")
plt.title(f"Estimated Savings with Logistic Regression\n savings: {lr_savings} EUR  ({lr_savings/status_quo_cost*100:.1f}% reduction)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()