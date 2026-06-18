import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# Set a seed for reproducibility
np.random.seed(42)

# Load the customer dataset
data = pd.read_csv("EU-Park-Customers.csv")

# Customers with no recorded pass are non-buyers -> label them "No Pass"
data["Pass_Type"] = data["Pass_Type"].fillna("No Pass")

# Display the first rows
print(data.head())

# Check the dataset structure
print("Dataset shape:", data.shape)
print("\nData types:")
print(data.dtypes)

print("\nMissing values:")
print(data.isna().sum())


# Select reasonable predictive features
feature_columns = [
    "Age",
    "Family_Members_with_Passes",
    "Previously_Owned_Passes",
    "Club_Member",
    "Distance_from_Park_km"
]

X = data[feature_columns].copy()
y = data["Pass_Type"].copy()

# Convert Boolean values to numeric values
X["Club_Member"] = X["Club_Member"].astype(int)

print("Input features:")
print(X.head())

print("\nTarget:")
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    shuffle=True,
    stratify=y
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True).round(3))

print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True).round(3))

# Create an unrestricted Decision Tree
dt_full = DecisionTreeClassifier(
    criterion="entropy",
    random_state=42
)

# Train the model
dt_full.fit(X_train, y_train)

# Make predictions
y_train_pred_full = dt_full.predict(X_train)
y_test_pred_full = dt_full.predict(X_test)

# Calculate accuracy
train_accuracy_full = accuracy_score(y_train, y_train_pred_full)
test_accuracy_full = accuracy_score(y_test, y_test_pred_full)

print("Unrestricted Decision Tree")
print("--------------------------------")
print(f"Training accuracy: {train_accuracy_full:.4f}")
print(f"Testing accuracy:  {test_accuracy_full:.4f}")
print(f"Accuracy gap:      {train_accuracy_full - test_accuracy_full:.4f}")

print("\nTree depth:", dt_full.get_depth())
print("Number of leaves:", dt_full.get_n_leaves())

# Create a forward-pruned Decision Tree
dt_model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=3,
    random_state=42
)

# Train the model
dt_model.fit(X_train, y_train)

# Make predictions
y_train_pred = dt_model.predict(X_train)
y_test_pred = dt_model.predict(X_test)

# Compute accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Pruned Decision Tree")
print("--------------------------------")
print(f"Training accuracy: {train_accuracy:.4f}")
print(f"Testing accuracy:  {test_accuracy:.4f}")
print(f"Accuracy gap:      {train_accuracy - test_accuracy:.4f}")

print("\nTree depth:", dt_model.get_depth())
print("Number of leaves:", dt_model.get_n_leaves())

plt.figure(figsize=(24, 12))

plot_tree(
    dt_model,
    feature_names=X_train.columns.tolist(),
    class_names=dt_model.classes_.tolist(),
    filled=True,
    rounded=True,
    fontsize=10,
    proportion=True,
    precision=2
)

plt.title("Decision Tree: Season-Pass Purchase Prediction")
plt.tight_layout()
plt.savefig("decision_tree_plot.jpg", dpi=150, bbox_inches="tight")
plt.show()

plt.figure(figsize=(24, 12))

plot_tree(
    dt_model,
    feature_names=X_train.columns.tolist(),
    class_names=dt_model.classes_.tolist(),
    filled=True,
    rounded=True,
    fontsize=10,
    proportion=True,
    precision=2
)

plt.title("Decision Tree: Season-Pass Purchase Prediction")
plt.tight_layout()
plt.savefig("decision_tree_plot_2.jpg", dpi=150, bbox_inches="tight")
plt.show()

# Define a consistent class order
class_order = ["Gold", "Silver", "No Pass"]

# Create confusion matrix
conf_matrix = confusion_matrix(
    y_test,
    y_test_pred,
    labels=class_order
)

print("Confusion Matrix:")
print(conf_matrix)

# Display it as a labelled table
confusion_table = pd.DataFrame(
    conf_matrix,
    index=[f"Actual {label}" for label in class_order],
    columns=[f"Predicted {label}" for label in class_order]
)

print(confusion_table)

# Define a consistent class order
class_order = ["Gold", "Silver", "No Pass"]

# Create confusion matrix
conf_matrix = confusion_matrix(
    y_test,
    y_test_pred,
    labels=class_order
)

print("Confusion Matrix:")
print(conf_matrix)

# Display it as a labelled table
confusion_table = pd.DataFrame(
    conf_matrix,
    index=[f"Actual {label}" for label in class_order],
    columns=[f"Predicted {label}" for label in class_order]
)

print(confusion_table)

print("Classification Report:")
print(
    classification_report(
        y_test,
        y_test_pred,
        labels=class_order,
        digits=3,
        zero_division=0
    )
)

comparison = pd.DataFrame({
    "Model": [
        "Unrestricted Decision Tree",
        "Pruned Decision Tree"
    ],
    "Tree Depth": [
        dt_full.get_depth(),
        dt_model.get_depth()
    ],
    "Number of Leaves": [
        dt_full.get_n_leaves(),
        dt_model.get_n_leaves()
    ],
    "Training Accuracy": [
        train_accuracy_full,
        train_accuracy
    ],
    "Testing Accuracy": [
        test_accuracy_full,
        test_accuracy
    ]
})

comparison["Train-Test Gap"] = (
    comparison["Training Accuracy"]
       - comparison["Testing Accuracy"]
)

print(comparison.round(4))

comparison_plot = comparison.set_index("Model")[
    ["Training Accuracy", "Testing Accuracy"]
]

comparison_plot.plot(
    kind="bar",
    figsize=(9, 5)
)

plt.title("Training and Testing Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("")
plt.ylim(0, 1.05)
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("accuracy_comparison.jpg", dpi=150, bbox_inches="tight")
plt.show()

#Feature importance

feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": dt_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print(feature_importance)

feature_importance_sorted = feature_importance.sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(9, 5))

plt.barh(
    feature_importance_sorted["Feature"],
    feature_importance_sorted["Importance"]
)

plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.title("Decision Tree Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.jpg", dpi=150, bbox_inches="tight")
plt.show()

#test different tree depths
depth_results = []

for depth in range(1, 16):

    model = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    depth_results.append({
        "Max Depth": depth,
        "Actual Tree Depth": model.get_depth(),
        "Leaves": model.get_n_leaves(),
        "Training Accuracy": accuracy_score(y_train, train_pred),
        "Testing Accuracy": accuracy_score(y_test, test_pred)
    })

depth_results = pd.DataFrame(depth_results)

depth_results["Accuracy Gap"] = (
    depth_results["Training Accuracy"]
    - depth_results["Testing Accuracy"]
)

print(depth_results.round(4))

plt.figure(figsize=(10, 6))

plt.plot(
    depth_results["Max Depth"],
    depth_results["Training Accuracy"],
    marker="o",
    label="Training accuracy"
)

plt.plot(
    depth_results["Max Depth"],
    depth_results["Testing Accuracy"],
    marker="o",
    label="Testing accuracy"
)

plt.xlabel("Maximum tree depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree Complexity and Model Performance")
plt.xticks(depth_results["Max Depth"])
plt.ylim(0, 1.02)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("depth_vs_accuracy.jpg", dpi=150, bbox_inches="tight")
plt.show()

best_depth_row = depth_results.loc[
    depth_results["Testing Accuracy"].idxmax()
]

print("Best depth based on the test comparison:")
print(best_depth_row)

#More rigorous depth selection with cross-validation

from sklearn.model_selection import GridSearchCV

parameter_grid = {
    "max_depth": range(2, 11),
    "min_samples_split": [2, 10, 25, 50],
    "min_samples_leaf": [1, 10, 25, 50]
}

base_tree = DecisionTreeClassifier(
    criterion="entropy",
    random_state=42
)

grid_search = GridSearchCV(
    estimator=base_tree,
    param_grid=parameter_grid,
    scoring="f1_macro",
    cv=5,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Best parameters:")
print(grid_search.best_params_)

print("\nBest cross-validation macro F1:")
print(round(grid_search.best_score_, 4))

best_tree = grid_search.best_estimator_

best_test_pred = best_tree.predict(X_test)

print("Optimised Decision Tree")
print("--------------------------------")
print(
    "Test accuracy:",
    round(accuracy_score(y_test, best_test_pred), 4)
)

print("\nClassification report:")
print(
    classification_report(
        y_test,
        best_test_pred,
        labels=class_order,
        digits=3,
        zero_division=0
    )
)

plt.figure(figsize=(26, 14))

plot_tree(
    best_tree,
    feature_names=X_train.columns.tolist(),
    class_names=best_tree.classes_.tolist(),
    filled=True,
    rounded=True,
    fontsize=9,
    proportion=True,
    precision=2
)

plt.title("Cross-Validated Decision Tree")
plt.tight_layout()
plt.show()