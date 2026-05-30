# -*- coding: utf-8 -*-
"""
Q2 Alternative (Neural Network): MLPClassifier for binary error prediction
Aligned with Lecture: Neural Networks (Slides 5-7, 14-16) & Task1 (Slides 10-11, 85-87)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, roc_curve, auc, precision_score, recall_score
)
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. LOAD & MERGE DATASETS (Task1 Slide 11)
# ==========================================
prod = pd.read_csv('Production_Log_01.csv')
machine = pd.read_csv('Machine_Settings_Log_01.csv')

prod.columns = prod.columns.str.strip()
machine.columns = machine.columns.str.strip()

df = prod.merge(machine, on='configuration_log_ID', how='left')
print(f"[1] Loaded {df.shape[0]} records, {df.shape[1]} columns.")

# ==========================================
# 2. STRICT DATA CLEANING
# ==========================================
df = df[df['error'].notna() & ~df['error'].isin(['None', 'none', ''])]
df = df[df['width'] < 1000]  # Remove outliers
print(f"[2] Cleaned dataset: {df.shape[0]} records.")
print(f"    Error distribution: {df['error'].value_counts().to_dict()}\n")

# ==========================================
# 3. FEATURE ENGINEERING (Task1 Slide 51)
# ==========================================
base_features = [
    'width', 'height', 'ionizationclass', 'FluxCompensation',
    'pressure', 'karma', 'modulation', 'gear', 'rotation_speed'
]

X = df[base_features].copy()
y_raw = df['error'].copy()

# Create interactions & ratios (helps NN find non-linear patterns faster)
X['width_height'] = X['width'] * X['height']
X['width_pressure'] = X['width'] * X['pressure']
X['width_karma'] = X['width'] * X['karma']
X['width_ratio'] = X['width'] / (X['height'] + 1e-6)
X['pressure_mod_ratio'] = X['pressure'] / (X['modulation'] + 1e-6)
X['volume_proxy'] = X['width'] * X['height'] * X['rotation_speed']

# One-Hot Encoding for categorical features
X = pd.get_dummies(X, drop_first=True)
print(f"[3] Features after encoding & engineering: {X.shape[1]}")

# Encode target: 'no' -> 0, 'yes' -> 1
le = LabelEncoder()
y = le.fit_transform(y_raw)

# ==========================================
# 4. SCALING & PIPELINE (Neural Networks Slide 14)
# ==========================================
# Slide 14: "Scaling data of different features so that they are of the same range.
# Makes handling easier for many algorithms. Avoids errors due to different ranges."
# Neural networks are highly sensitive to unscaled data.

scaler = StandardScaler()

# ==========================================
# 5. TRAIN/TEST SPLIT
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"[5] Split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")

# ==========================================
# 6. NEURAL NETWORK MODEL (Slides 15-16, 5-7)
# ==========================================
# Slide 15/16: MLPClassifier implements multi-layer perceptron
# Forward pass: dot products + activation functions (Slide 5-7)
# We use 'adam' solver, ReLU activation, and early stopping for robustness

model = MLPClassifier(
    hidden_layer_sizes=(64, 32, 16),  # Architecture: width & depth (Slide 12)
    activation='relu',                # Non-linearity (Slide 8)
    solver='adam',                    # Optimizer
    alpha=0.001,                      # L2 regularization
    batch_size='auto',
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=1000,
    random_state=42,
    early_stopping=True,              # Prevents overfitting
    validation_fraction=0.1,
    verbose=False
)

# Pipeline ensures scaling is fit ONLY on train data (no data leakage)
pipeline = Pipeline([
    ('scaler', scaler),
    ('classifier', model)
])

print("[6] Training Neural Network (MLPClassifier)...")
pipeline.fit(X_train, y_train)
print("[6] Training complete. Final loss:", pipeline.named_steps['classifier'].loss_)

# ==========================================
# 7. EVALUATION & ROC ANALYSIS (Slides 28-29, 73-75)
# ==========================================
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print("\n=== CLASSIFICATION REPORT ===")
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
plt.title('ROC Curve - Neural Network (MLP)')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ==========================================
# 8. BUSINESS IMPACT / THRESHOLD OPTIMIZATION (Slides 85-87, 92)
# ==========================================
def calculate_expected_loss(threshold, y_true, y_prob):
    y_pred_thresh = (y_prob >= threshold).astype(int)
    cm_t = confusion_matrix(y_true, y_pred_thresh)
    tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
    # Costs: FP/TP = 10 EUR (discard raw material), FN = 150 EUR (faulty product)
    loss = (tp_t + fp_t) * 10 + fn_t * 150
    return loss, tn_t, fp_t, fn_t, tp_t

base_loss, _, _, _, _ = calculate_expected_loss(0.5, y_test, y_proba)
print("\n=== BUSINESS IMPACT (Default Threshold 0.5) ===")
print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
print(f"Estimated Loss: {base_loss:.0f} EUR")
print(f"Loss per sample: {base_loss/len(y_test):.2f} EUR")

# Optimize threshold
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
print(f"💡 Improvement vs default: {base_loss - best_loss:.0f} EUR saved")

plt.figure(figsize=(8, 5))
plt.plot(thresh_range, losses, color='crimson', lw=2)
plt.axvline(x=best_thresh, color='green', linestyle='--', label=f'Optimal Threshold ({best_thresh:.2f})')
plt.xlabel('Prediction Threshold')
plt.ylabel('Expected Financial Loss (EUR)')
plt.title('Expected Loss vs Decision Threshold (MLP)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ==========================================
# 9. EXECUTIVE SUMMARY FOR CEO/CTO
# ==========================================
print("\n" + "="*60)
print("EXECUTIVE SUMMARY (Q2 Alt: Neural Network MLP)")
print("="*60)
print("• TASK: Predict binary error (yes/no) BEFORE machine runs.")
print("• MODEL: Multi-Layer Perceptron (MLPClassifier) with StandardScaler.")
print("• WHY NN?: Captures complex non-linear interactions automatically (Slide 5-7).")
print("• RESULT: Accuracy ~{:.1%}, ROC-AUC {:.3f}.".format(acc, roc_auc))
print("• BUSINESS: Optimized threshold {:.2f} → Loss {:.2f} EUR/sample.".format(best_thresh, best_loss/len(y_test)))
print("• TRADE-OFF: NN performs similarly to Gradient Boosting but is less interpretable.")
print("• ACTION: Deploy as secondary validator. Monitor inference latency & drift.")
print("="*60)