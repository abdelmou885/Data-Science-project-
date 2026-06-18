import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Load data
df = pd.read_csv("EU-Park-Customers.csv")

# Missing Pass_Type means no season pass
df["Pass_Type"] = df["Pass_Type"].fillna("No Pass")

# Remove identifier
X = df.drop(
    columns=["Pass_Type", "Telephone_Number"]
).copy()

y = df["Pass_Type"]

# Convert Boolean to integer
X["Club_Member"] = X["Club_Member"].astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Build model pipeline
logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            random_state=42
        )
    )
])

# Train model
logistic_model.fit(X_train, y_train)

# Predictions
y_train_pred = logistic_model.predict(X_train)
y_test_pred = logistic_model.predict(X_test)

# Metrics
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

print("=" * 55)
print("MULTINOMIAL LOGISTIC REGRESSION RESULTS")
print("=" * 55)

print(f"Training accuracy: {train_accuracy:.4f}")
print(f"Test accuracy:     {test_accuracy:.4f}")

print()

print(f"Training Macro F1: {train_macro_f1:.4f}")
print(f"Test Macro F1:     {test_macro_f1:.4f}")

print()

print(f"Accuracy gap: {train_accuracy - test_accuracy:.4f}")
print(f"Macro F1 gap: {train_macro_f1 - test_macro_f1:.4f}")

print("\nClassification report – Test Data:\n")

print(
    classification_report(
        y_test,
        y_test_pred,
        digits=3
    )
)

# Confusion matrix
class_order = ["Gold", "Silver", "No Pass"]

cm = confusion_matrix(
    y_test,
    y_test_pred,
    labels=class_order
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_order
)

fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=False
)

plt.title("Logistic Regression – Confusion Matrix")
plt.tight_layout()
plt.show()

# Coefficients
classifier = logistic_model.named_steps["classifier"]

coefficients = pd.DataFrame(
    classifier.coef_,
    index=classifier.classes_,
    columns=X.columns
)

print("\nStandardised coefficients:")
print(coefficients.round(3))