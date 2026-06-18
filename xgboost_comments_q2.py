# -*- coding: utf-8 -*-
"""
Q2: XGBoost Multi-Class Classifier for error_type
Aligned with lecture slides: 28-29, 34-42, 44-49, 51-53, 58-61, 85-88, 91-92
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. LOAD AND MERGE DATASETS
# ==========================================
# Specify correct file paths
production_log = pd.read_csv('Production_Log_01.csv')
machine_settings_log = pd.read_csv('Machine_Settings_Log_01.csv')

df = production_log.merge(machine_settings_log, on="configuration_log_ID", how="left")
print(f"[1] Initial dataset size: {df.shape}")

# ==========================================
# 2. DATA CLEANING 
# ==========================================
df_clean = df.copy()

# Remove rows with missing or invalid error information
df_clean = df_clean[df_clean['error_type'].notna()]
df_clean = df_clean[~df_clean['error_type'].isin(['None', 'none', ''])]

# Remove obvious outliers/input errors (same as Q1)
df_clean = df_clean[df_clean["width"] < 1000]

print(f"[2] Size after cleaning: {df_clean.shape}")
print(f"[2] Unique error classes: {df_clean['error_type'].unique()}")
print(f"[2] Class distribution:\n{df_clean['error_type'].value_counts()}\n")

# ==========================================
# 3. FEATURE AND TARGET PREPARATION 
# ==========================================
# Features available BEFORE or DURING production
features = [
    "width", "height", "ionizationclass", "FluxCompensation", 
    "pressure", "karma", "modulation", "gear", "rotation_speed"
]

X = df_clean[features].copy()
y_raw = df_clean['error_type'].copy()

# 3.1 Automatically separate numeric and categorical columns
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = X.select_dtypes(include=['number']).columns.tolist()

# 3.2 One-Hot Encoding for categorical features (Slides 47-49)
if cat_cols:
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# 3.3 Fill missing values with median ONLY in numeric columns (fixes TypeError)
for col in num_cols:
    if col in X.columns:
        X[col] = X[col].fillna(X[col].median())

# 3.4 Encode target variable using Label Encoding (Slides 44-45)
le = LabelEncoder()
y = le.fit_transform(y_raw)
num_classes = len(le.classes_)

print(f"[3] Encoded classes: {dict(zip(le.classes_, range(num_classes)))}")
print(f"[3] Final feature set: {X.shape[1]} columns\n")

# ==========================================
# 4. TRAIN/TEST SPLIT 
# ==========================================
# stratify=y is critical for multi-class tasks: preserves class proportions
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, shuffle=True, stratify=y
)

print(f"[4] Train: {X_train.shape}, Test: {X_test.shape}")

# ==========================================
# 5. XGBOOST MODEL CREATION 
# ==========================================
model = XGBClassifier(
    objective='multi:softmax',      # Multi-class classification (one-vs-all, Slide 88)
    num_class=num_classes,          # Number of classes
    max_depth=5,                    # Forward pruning (Slide 41)
    learning_rate=0.1,              # eta: learning step in boosting (Slide 58)
    n_estimators=200,               # Chain length of models
    reg_lambda=1.0,                 # L2 regularization (lambda)
    random_state=42,
    eval_metric='mlogloss'
)

# Training
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
print("[5] Model trained.\n")

# ==========================================
# 6. PREDICTIONS AND EVALUATION 
# ==========================================
y_pred = model.predict(X_test)

print("=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred, target_names=le.classes_))

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}\n")

# Confusion Matrix (Slides 28-29)
cm = confusion_matrix(y_test, y_pred)
print("=== CONFUSION MATRIX ===")
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix (XGBoost Multi-Class)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()

# ==========================================
# 7. BUSINESS IMPACT / EXPECTED VALUE 
# ==========================================
# From assignment: Faulty product = 150 EUR, Raw material discard = 10 EUR
# Logic: Incorrect error_type classification -> wrong setup/missed defect -> 150 EUR loss
# Correct prediction -> 0 additional loss
total_samples = np.sum(cm)
correct_predictions = np.trace(cm)
incorrect_predictions = total_samples - correct_predictions

cost_per_misclassification = 150.0  # EUR
total_business_loss = incorrect_predictions * cost_per_misclassification
loss_per_sample = total_business_loss / total_samples

print("=== BUSINESS IMPACT (Expected Value Framework) ===")
print(f"Total test samples: {total_samples}")
print(f"Correct predictions: {correct_predictions} ({correct_predictions/total_samples:.1%})")
print(f"Incorrect predictions: {incorrect_predictions}")
print(f"Estimated financial loss: {total_business_loss:.2f} EUR")
print(f"Loss per sample: {loss_per_sample:.2f} EUR")
print("💡 Recommendation: If loss_per_sample > 10 EUR, consider stricter preprocessing, "
      "feature engineering, or collecting additional sensor data.\n")

# ==========================================
# 8. FEATURE IMPORTANCE 
# ==========================================
feat_imp = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("=== TOP 10 FEATURE IMPORTANCE ===")
print(feat_imp.head(10))

plt.figure(figsize=(10, 6))
sns.barplot(data=feat_imp.head(10), x='importance', y='feature')
plt.title('XGBoost Feature Importance (Top 10)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()