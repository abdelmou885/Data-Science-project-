# If XGBoost is not installed, run this once:
# pip install xgboost

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc
)

from xgboost import XGBClassifier, plot_importance

# ============================================================
# 2. LOAD DATA
# ============================================================

data = pd.read_csv("EU-Park-Customers.csv")

print("First five rows:")
print(data.head())

print("\nDataset shape:")
print(data.shape)

print("\nColumn information:")
print(data.info())

print("\nMissing values:")
print(data.isnull().sum())

print("\nTarget distribution before preprocessing:")
print(data["Pass_Type"].value_counts(dropna=False))

# Replace missing target values with the class "No Pass"
data["Pass_Type"] = data["Pass_Type"].fillna("No Pass")

print("\nTarget distribution after preprocessing:")
print(data["Pass_Type"].value_counts())

# Replace missing target values with the class "No Pass"
data["Pass_Type"] = data["Pass_Type"].fillna("No Pass")

print("\nTarget distribution after preprocessing:")
print(data["Pass_Type"].value_counts())


# ============================================================
# 3. FEATURE SELECTION
# ============================================================

features = [
    "Age",
    "Family_Members_with_Passes",
    "Previously_Owned_Passes",
    "Club_Member",
    "Distance_from_Park_km"
]

X = data[features].copy()
y = data["Pass_Type"].copy()

# Convert Boolean values into numeric values
X["Club_Member"] = X["Club_Member"].astype(int)

print("\nPredictor variables:")
print(X.head())

print("\nTarget variable:")
print(y.head())

# ============================================================
# 4. TARGET ENCODING
# ============================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("Class encoding:")

for class_number, class_name in enumerate(label_encoder.classes_):
    print(class_number, "=", class_name)


# ============================================================
# 5. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    shuffle=True,
    stratify=y_encoded
)

print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)

print("\nTraining class distribution:")
print(pd.Series(y_train).value_counts(normalize=True).sort_index())

print("\nTest class distribution:")
print(pd.Series(y_test).value_counts(normalize=True).sort_index())

# ============================================================
# 6. CREATE XGBOOST CLASSIFIER
# ============================================================

xgb_model = XGBClassifier(
    objective="multi:softprob",
    num_class=len(label_encoder.classes_),

    # Number of sequential trees
    n_estimators=200,

    # Controls the complexity of individual trees
    max_depth=3,

    # Contribution made by each new tree
    learning_rate=0.05,

    # Randomly use 80% of training rows for each tree
    subsample=0.80,

    # Randomly use 80% of features for each tree
    colsample_bytree=0.80,

    # Evaluation metric for multiclass classification
    eval_metric="mlogloss",

    random_state=42,
    n_jobs=-1
)

# Train the model
xgb_model.fit(X_train, y_train)

# ============================================================
# 7. MAKE PREDICTIONS
# ============================================================

# Predicted numerical class
y_train_pred = xgb_model.predict(X_train)
y_test_pred = xgb_model.predict(X_test)

# Predicted probabilities for each class
y_train_probability = xgb_model.predict_proba(X_train)
y_test_probability = xgb_model.predict_proba(X_test)

print("First five predicted classes:")
print(y_test_pred[:5])

print("\nFirst five probability predictions:")
print(y_test_probability[:5])

# ============================================================
# 8. OVERALL MODEL EVALUATION
# ============================================================

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

train_macro_f1 = f1_score(
    y_train,
    y_train_pred,
    average="macro"
)

test_macro_f1 = f1_score(
    y_test,
    y_test_pred,
    average="macro"
)

train_weighted_f1 = f1_score(
    y_train,
    y_train_pred,
    average="weighted"
)

test_weighted_f1 = f1_score(
    y_test,
    y_test_pred,
    average="weighted"
)

print("=" * 60)
print("XGBOOST CLASSIFICATION RESULTS")
print("=" * 60)

print(f"Training accuracy:    {train_accuracy:.4f}")
print(f"Test accuracy:        {test_accuracy:.4f}")

print(f"\nTraining Macro F1:    {train_macro_f1:.4f}")
print(f"Test Macro F1:        {test_macro_f1:.4f}")

print(f"\nTraining Weighted F1: {train_weighted_f1:.4f}")
print(f"Test Weighted F1:     {test_weighted_f1:.4f}")

print(f"\nAccuracy gap:         {train_accuracy - test_accuracy:.4f}")
print(f"Macro F1 gap:         {train_macro_f1 - test_macro_f1:.4f}")

# ============================================================
# 8. OVERALL MODEL EVALUATION
# ============================================================

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

train_macro_f1 = f1_score(
    y_train,
    y_train_pred,
    average="macro"
)

test_macro_f1 = f1_score(
    y_test,
    y_test_pred,
    average="macro"
)

train_weighted_f1 = f1_score(
    y_train,
    y_train_pred,
    average="weighted"
)

test_weighted_f1 = f1_score(
    y_test,
    y_test_pred,
    average="weighted"
)

print("=" * 60)
print("XGBOOST CLASSIFICATION RESULTS")
print("=" * 60)

print(f"Training accuracy:    {train_accuracy:.4f}")
print(f"Test accuracy:        {test_accuracy:.4f}")

print(f"\nTraining Macro F1:    {train_macro_f1:.4f}")
print(f"Test Macro F1:        {test_macro_f1:.4f}")

print(f"\nTraining Weighted F1: {train_weighted_f1:.4f}")
print(f"Test Weighted F1:     {test_weighted_f1:.4f}")

print(f"\nAccuracy gap:         {train_accuracy - test_accuracy:.4f}")
print(f"Macro F1 gap:         {train_macro_f1 - test_macro_f1:.4f}")

# ============================================================
# 9. CLASSIFICATION REPORT
# ============================================================

print("\nClassification report – Test Data:\n")

print(
    classification_report(
        y_test,
        y_test_pred,
        target_names=label_encoder.classes_,
        digits=3
    )
)

# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

conf_matrix = confusion_matrix(y_test, y_test_pred)

conf_matrix_df = pd.DataFrame(
    conf_matrix,
    index=[
        f"Actual {class_name}"
        for class_name in label_encoder.classes_
    ],
    columns=[
        f"Predicted {class_name}"
        for class_name in label_encoder.classes_
    ]
)

print("Confusion Matrix:")
print(conf_matrix_df)


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

conf_matrix = confusion_matrix(y_test, y_test_pred)

conf_matrix_df = pd.DataFrame(
    conf_matrix,
    index=[
        f"Actual {class_name}"
        for class_name in label_encoder.classes_
    ],
    columns=[
        f"Predicted {class_name}"
        for class_name in label_encoder.classes_
    ]
)

print("Confusion Matrix:")
print(conf_matrix_df)

plt.figure(figsize=(8, 6))

plt.imshow(conf_matrix)

plt.title("XGBoost Confusion Matrix")
plt.xlabel("Predicted Pass Type")
plt.ylabel("Actual Pass Type")

plt.xticks(
    ticks=np.arange(len(label_encoder.classes_)),
    labels=label_encoder.classes_
)

plt.yticks(
    ticks=np.arange(len(label_encoder.classes_)),
    labels=label_encoder.classes_
)

# Write the values inside the matrix
for row in range(conf_matrix.shape[0]):
    for column in range(conf_matrix.shape[1]):
        plt.text(
            column,
            row,
            conf_matrix[row, column],
            horizontalalignment="center",
            verticalalignment="center"
        )

plt.colorbar()
plt.tight_layout()
plt.show()

# ============================================================
# 11. NORMALISED CONFUSION MATRIX
# ============================================================

conf_matrix_normalised = confusion_matrix(
    y_test,
    y_test_pred,
    normalize="true"
)

plt.figure(figsize=(8, 6))

plt.imshow(conf_matrix_normalised)

plt.title("Normalised XGBoost Confusion Matrix")
plt.xlabel("Predicted Pass Type")
plt.ylabel("Actual Pass Type")

plt.xticks(
    ticks=np.arange(len(label_encoder.classes_)),
    labels=label_encoder.classes_
)

plt.yticks(
    ticks=np.arange(len(label_encoder.classes_)),
    labels=label_encoder.classes_
)

for row in range(conf_matrix_normalised.shape[0]):
    for column in range(conf_matrix_normalised.shape[1]):
        plt.text(
            column,
            row,
            f"{conf_matrix_normalised[row, column]:.2f}",
            horizontalalignment="center",
            verticalalignment="center"
        )

plt.colorbar()
plt.tight_layout()
plt.show()

# ============================================================
# 12. MULTICLASS ROC CURVES
# ============================================================

# Convert test labels into one-hot format
y_test_one_hot = pd.get_dummies(y_test).reindex(
    columns=range(len(label_encoder.classes_)),
    fill_value=0
).values

plt.figure(figsize=(9, 7))

for class_number, class_name in enumerate(label_encoder.classes_):

    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        y_test_one_hot[:, class_number],
        y_test_probability[:, class_number]
    )

    class_auc = auc(
        false_positive_rate,
        true_positive_rate
    )

    plt.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"{class_name}: AUC = {class_auc:.3f}"
    )

# Random-classifier reference line
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random prediction"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("One-vs-Rest ROC Curves – XGBoost")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


multiclass_auc = roc_auc_score(
    y_test,
    y_test_probability,
    multi_class="ovr",
    average="macro"
)

print(f"Multiclass ROC-AUC: {multiclass_auc:.4f}")


# ============================================================
# 13. FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": xgb_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("Feature Importance:")
print(feature_importance)


# ============================================================
# 13. FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": xgb_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("Feature Importance:")
print(feature_importance)

plt.figure(figsize=(9, 6))

plot_importance(
    xgb_model,
    importance_type="gain",
    show_values=True
)

plt.title("XGBoost Feature Importance Based on Gain")
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 6))

plot_importance(
    xgb_model,
    importance_type="gain",
    show_values=True
)

plt.title("XGBoost Feature Importance Based on Gain")
plt.tight_layout()
plt.show()


# ============================================================
# 14. DETAILED PREDICTION RESULTS
# ============================================================

results = X_test.copy()

results["Actual_Pass_Type"] = label_encoder.inverse_transform(y_test)
results["Predicted_Pass_Type"] = label_encoder.inverse_transform(
    y_test_pred.astype(int)
)

# Add one probability column per class
for class_number, class_name in enumerate(label_encoder.classes_):
    results[f"Probability_{class_name}"] = (
        y_test_probability[:, class_number]
    )

results["Correct_Prediction"] = (
    results["Actual_Pass_Type"]
    == results["Predicted_Pass_Type"]
)




print(results.head(10))



incorrect_predictions = results[
    results["Correct_Prediction"] == False
]

print("\nIncorrect predictions:")
print(incorrect_predictions.head(20))

# ============================================================
# 15. CUSTOMER TARGETING TABLE
# ============================================================

results["Highest_Probability"] = y_test_probability.max(axis=1)

def assign_marketing_segment(row):

    predicted_pass = row["Predicted_Pass_Type"]
    probability = row["Highest_Probability"]

    if probability >= 0.80:
        return f"High-confidence {predicted_pass}"

    elif probability >= 0.55:
        return f"Moderate-confidence {predicted_pass}"

    else:
        return "Uncertain customer"

results["Marketing_Segment"] = results.apply(
    assign_marketing_segment,
    axis=1
)

print(
    results[
        [
            "Actual_Pass_Type",
            "Predicted_Pass_Type",
            "Highest_Probability",
            "Marketing_Segment"
        ]
    ].head(20)
)

