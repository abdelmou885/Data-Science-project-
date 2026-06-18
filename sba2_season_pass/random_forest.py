# ============================================================
# EU-PARK: SEASON-PASS PURCHASE PREDICTION
# RANDOM FOREST CLASSIFIER
# ============================================================

# 1. Import libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.inspection import permutation_importance


# ============================================================
# 2. Load the dataset
# ============================================================

df = pd.read_csv("EU-Park-Customers.csv")

print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# 3. Prepare the target variable
# ============================================================

# Assumption:
# Missing Pass_Type means that the customer did not buy a pass.
df["Pass_Type"] = df["Pass_Type"].fillna("No Pass")

print("\nTarget distribution:")
print(df["Pass_Type"].value_counts())

print("\nTarget distribution in percentages:")
print(df["Pass_Type"].value_counts(normalize=True).mul(100).round(2))


# ============================================================
# 4. Remove irrelevant identifier
# ============================================================

# Telephone number is an identifier and should not be used
# for predicting customer behaviour.
df = df.drop(columns=["Telephone_Number"])


# ============================================================
# 5. Define predictors and target
# ============================================================

X = df.drop(columns=["Pass_Type"])
y = df["Pass_Type"]

print("\nPredictor variables:")
print(X.columns.tolist())

print("\nTarget classes:")
print(y.unique())


# ============================================================
# 6. Convert Boolean variable to numeric
# ============================================================

# Random Forest can often process Boolean values directly,
# but converting them to 0 and 1 makes the data explicit.
X["Club_Member"] = X["Club_Member"].astype(int)

print("\nPrepared predictor data:")
print(X.head())


# ============================================================
# 7. Split into training and test sets
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:", X_train.shape)
print("Test data shape:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True).round(3))

print("\nTest target distribution:")
print(y_test.value_counts(normalize=True).round(3))


# ============================================================
# 8. Create and train the Random Forest model
# ============================================================

random_forest = RandomForestClassifier(
    n_estimators=300,       # Number of decision trees
    max_depth=10,           # Limits tree complexity
    min_samples_split=10,   # Minimum observations required to split a node
    min_samples_leaf=5,     # Minimum observations in each final leaf
    max_features="sqrt",    # Number of features considered at each split
    random_state=42,
    n_jobs=-1               # Uses all available processor cores
)

random_forest.fit(X_train, y_train)


# ============================================================
# 9. Make predictions
# ============================================================

y_train_pred = random_forest.predict(X_train)
y_test_pred = random_forest.predict(X_test)


# ============================================================
# 10. Evaluate training and test performance
# ============================================================

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

train_f1 = f1_score(
    y_train,
    y_train_pred,
    average="macro"
)

test_f1 = f1_score(
    y_test,
    y_test_pred,
    average="macro"
)

print("\n" + "=" * 55)
print("RANDOM FOREST RESULTS")
print("=" * 55)

print(f"Training accuracy: {train_accuracy:.4f}")
print(f"Test accuracy:     {test_accuracy:.4f}")

print(f"\nTraining Macro F1: {train_f1:.4f}")
print(f"Test Macro F1:     {test_f1:.4f}")

print(f"\nAccuracy gap: {train_accuracy - test_accuracy:.4f}")
print(f"Macro F1 gap: {train_f1 - test_f1:.4f}")


# ============================================================
# 11. Detailed classification report
# ============================================================

print("\nClassification report – Test Data:")
print(
    classification_report(
        y_test,
        y_test_pred,
        digits=3
    )
)


# ============================================================
# 12. Additional overall metrics
# ============================================================

test_precision_macro = precision_score(
    y_test,
    y_test_pred,
    average="macro"
)

test_recall_macro = recall_score(
    y_test,
    y_test_pred,
    average="macro"
)

test_f1_weighted = f1_score(
    y_test,
    y_test_pred,
    average="weighted"
)

print("Overall test metrics:")
print(f"Accuracy:            {test_accuracy:.4f}")
print(f"Macro precision:     {test_precision_macro:.4f}")
print(f"Macro recall:        {test_recall_macro:.4f}")
print(f"Macro F1-score:      {test_f1:.4f}")
print(f"Weighted F1-score:   {test_f1_weighted:.4f}")


# ============================================================
# 13. Confusion matrix
# ============================================================

class_order = ["Gold", "Silver", "No Pass"]

cm = confusion_matrix(
    y_test,
    y_test_pred,
    labels=class_order
)

print("\nConfusion matrix:")
print(cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_order
)

display.plot(values_format="d")
plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.show()


# ============================================================
# 14. Normalised confusion matrix
# ============================================================

cm_normalized = confusion_matrix(
    y_test,
    y_test_pred,
    labels=class_order,
    normalize="true"
)

display_normalized = ConfusionMatrixDisplay(
    confusion_matrix=cm_normalized,
    display_labels=class_order
)

display_normalized.plot(values_format=".2f")
plt.title("Normalised Random Forest Confusion Matrix")
plt.tight_layout()
plt.show()


# ============================================================
# 15. Standard Random Forest feature importance
# ============================================================

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": random_forest.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
).reset_index(drop=True)

print("\nRandom Forest feature importance:")
print(feature_importance)

plt.figure(figsize=(9, 5))
plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# ============================================================
# 16. Permutation importance
# ============================================================

# Permutation importance measures how strongly model performance
# decreases when the values of one feature are shuffled.

permutation_results = permutation_importance(
    random_forest,
    X_test,
    y_test,
    scoring="f1_macro",
    n_repeats=20,
    random_state=42,
    n_jobs=-1
)

permutation_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance_Mean": permutation_results.importances_mean,
    "Importance_Std": permutation_results.importances_std
})

permutation_df = permutation_df.sort_values(
    by="Importance_Mean",
    ascending=False
).reset_index(drop=True)

print("\nPermutation importance:")
print(permutation_df)

plt.figure(figsize=(9, 5))
plt.barh(
    permutation_df["Feature"],
    permutation_df["Importance_Mean"],
    xerr=permutation_df["Importance_Std"]
)
plt.xlabel("Decrease in Macro F1 after permutation")
plt.ylabel("Feature")
plt.title("Permutation Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# ============================================================
# 17. Predicted probabilities
# ============================================================

probabilities = random_forest.predict_proba(X_test)

probability_df = pd.DataFrame(
    probabilities,
    columns=random_forest.classes_,
    index=X_test.index
)

probability_df = probability_df.rename(
    columns={
        "Gold": "Probability_Gold",
        "Silver": "Probability_Silver",
        "No Pass": "Probability_No_Pass"
    }
)

results_df = X_test.copy()

results_df["Actual_Pass_Type"] = y_test
results_df["Predicted_Pass_Type"] = y_test_pred

results_df = results_df.join(probability_df)

results_df["Highest_Probability"] = probability_df.max(axis=1)

print("\nSample customer predictions:")
print(results_df.head(10))


# ============================================================
# 18. Identify uncertain customers
# ============================================================

# Customers with a maximum probability below 65% are considered
# less certain predictions and may be interesting marketing targets.

uncertain_customers = results_df[
    results_df["Highest_Probability"] < 0.65
].copy()

uncertain_customers = uncertain_customers.sort_values(
    by="Highest_Probability"
)

print("\nNumber of uncertain customers:")
print(len(uncertain_customers))

print("\nSample uncertain customers:")
print(uncertain_customers.head(10))


# ============================================================
# 19. Export results
# ============================================================

results_df.to_csv(
    "random_forest_customer_predictions.csv",
    index=False
)

feature_importance.to_csv(
    "random_forest_feature_importance.csv",
    index=False
)

permutation_df.to_csv(
    "random_forest_permutation_importance.csv",
    index=False
)

print("\nResults exported successfully.")