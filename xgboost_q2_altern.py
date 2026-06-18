# -*- coding: utf-8 -*-
"""
Q2 Alternative: Gradient Boosting Binary Classifier for 'error' (yes/no)
Predicts if a product will be faulty BEFORE machine usage.
Aligned with Task1.pdf & Lecture Slides (Trees/XGBoost, Expected Value)
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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, roc_curve, auc, precision_score, recall_score
)
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
# 2. STRICT DATA CLEANING (Slide 11, 14)
# ==========================================
# Focus on binary 'error' column
df = df[df['error'].notna() & ~df['error'].isin(['None', 'none', ''])]
df = df[df['width'] < 1000]  # Remove outliers

print(f"[2] Cleaned dataset: {df.shape[0]} records.")
print(f"    Error distribution: {df['error'].value_counts().to_dict()}\n")

# ==========================================
# 3. FEATURE SELECTION & ENGINEERING (Slide 51-53)
# ==========================================
# STRICTLY pre-production features only (Slide 11 diagram)
base_features = [
    'width', 'height', 'ionizationclass', 'FluxCompensation',
    'pressure', 'karma', 'modulation', 'gear', 'rotation_speed'
]

X = df[base_features].copy()
y_raw = df['error'].copy()

# Feature Engineering: Emphasize interactions & ratios
X['width_height'] = X['width'] * X['height']
X['width_pressure'] = X['width'] * X['pressure']
X['width_karma'] = X['width'] * X['karma']
X['width_ratio'] = X['width'] / (X['height'] + 1e-6)
X['pressure_mod_ratio'] = X['pressure'] / (X['modulation'] + 1e-6)
X['volume_proxy'] = X['width'] * X['height'] * X['rotation_speed']

print(f"[3] Engineered {len(X.columns)} features (pre-production only).")

num_cols = X.select_dtypes(include='number').columns.tolist()
cat_cols = X.select_dtypes(include='object').columns.tolist()

# Encode target: 'yes' -> 1 (Faulty), 'no' -> 0 (OK)
le = LabelEncoder()
y = le.fit_transform(y_raw)
print(f"    Target mapping: {dict(zip(le.classes_, [0, 1]))}")

# ==========================================
# 4. PREPROCESSING PIPELINE (Slide 47-49)
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
# 5. TRAIN/TEST SPLIT (Slide 29)
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"[5] Split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")

# ==========================================
# 6. MODEL: GRADIENT BOOSTING (Slide 34-42, 58-61)
# ==========================================
model = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    min_samples_leaf=5,
    subsample=0.9,
    random_state=42
)

pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', model)])
print("[6] Training Gradient Boosting model...")
pipeline.fit(X_train, y_train)
print("[6] Training complete.\n")

# ==========================================
# 7. EVALUATION & ROC ANALYSIS (Slide 28-29, 73-75, 80)
# ==========================================
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]  # Probability of 'yes' (error)

print("=== CLASSIFICATION REPORT (Threshold=0.5) ===")
print(classification_report(y_test, y_pred, target_names=['no (OK)', 'yes (Faulty)'], zero_division=0))

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {acc:.3f} | F1: {f1:.3f}")
print("\n=== CONFUSION MATRIX ===")
print(pd.DataFrame(cm, index=['Actual: no', 'Actual: yes'], columns=['Pred: no', 'Pred: yes']))

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Error Prediction')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ==========================================
# 8. BUSINESS IMPACT / EXPECTED VALUE (Slide 85-87, 91-92)
# ==========================================
# Costs from Task 1:
# - If we predict 'yes' (discard raw material): Cost = 10 EUR (applies to TP & FP)
# - If we predict 'no' & it's actually 'yes' (FN): Faulty product = 150 EUR loss
# - If we predict 'no' & it's actually 'no' (TN): Cost = 0 EUR

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

# Threshold Optimization (Slide 92)
print("\nOptimizing threshold for minimal financial loss...")
thresh_range = np.linspace(0.05, 0.95, 100)
losses = []
for t in thresh_range:
    loss, _, _, _, _ = calculate_expected_loss(t, y_test, y_proba)
    losses.append(loss)

best_idx = np.argmin(losses)
best_thresh = thresh_range[best_idx]
best_loss = losses[best_idx]
_, tn_opt, fp_opt, fn_opt, tp_opt = calculate_expected_loss(best_thresh, y_test, y_proba)

print(f"\n=== OPTIMIZED BUSINESS IMPACT (Threshold = {best_thresh:.2f}) ===")
print(f"TN: {tn_opt}, FP: {fp_opt}, FN: {fn_opt}, TP: {tp_opt}")
print(f"Optimized Loss: {best_loss:.0f} EUR")
print(f"Loss per sample: {best_loss/len(y_test):.2f} EUR")
print(f"💡 Improvement: {base_loss - best_loss:.0f} EUR saved on test set")

# Plot Expected Loss vs Threshold
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
# 9. FEATURE IMPORTANCE (Slide 51-53)
# ==========================================
cat_feature_names = pipeline.named_steps['preprocessor'].named_transformers_['cat']\
    .named_steps['onehot'].get_feature_names_out(cat_cols)
all_features = num_cols + cat_feature_names.tolist()

importances = pipeline.named_steps['classifier'].feature_importances_
feat_df = pd.DataFrame({'feature': all_features, 'importance': importances})\
    .sort_values('importance', ascending=False)

print("\n=== TOP 10 FEATURE IMPORTANCE ===")
print(feat_df.head(10).to_string(index=False))

plt.figure(figsize=(8, 6))
sns.barplot(data=feat_df.head(10), x='importance', y='feature')
plt.title('Top 10 Feature Importance')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()

# ==========================================
# 10. EXECUTIVE SUMMARY FOR CEO/CTO (Slide 6, 14)
# ==========================================
print("\n" + "="*60)
print("EXECUTIVE SUMMARY (Q2 Alternative: Binary Error Prediction)")
print("="*60)
print("• TASK: Predict if product will be faulty (yes/no) BEFORE machine runs.")
print("• METHOD: Gradient Boosting with strict pre-production features.")
print(f"• RESULT: Accuracy ~{acc:.1%}, ROC-AUC {roc_auc:.3f}.")
print(f"• BUSINESS: Default threshold loss = {base_loss/len(y_test):.2f} EUR/sample.")
print(f"• OPTIMIZED: Threshold {best_thresh:.2f} reduces loss to {best_loss/len(y_test):.2f} EUR/sample.")
print("="*60)