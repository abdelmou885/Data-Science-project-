# ============================================================
# EU-PARK: SEASON-PASS PURCHASE PREDICTION
# MODEL COMPARISON
#
# Compares the four classifiers on a common, consistent setup:
#   - Decision Tree        (decision_tree.py)
#   - Logistic Regression  (logistic_regression.py)
#   - Random Forest        (random_forest.py)
#   - XGBoost              (xgboost_model.py)
#
# All models use the same five predictors, the same train/test
# split, and each keeps the hyperparameters from its own script.
# ============================================================

# 1. IMPORT LIBRARIES
# ------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from xgboost import XGBClassifier

np.random.seed(42)


# ============================================================
# 2. LOAD AND PREPARE DATA
# ============================================================

df = pd.read_csv("EU-Park-Customers.csv")

# Missing Pass_Type means the customer did not buy a pass.
df["Pass_Type"] = df["Pass_Type"].fillna("No Pass")

# The five predictors shared by all four model scripts.
feature_columns = [
    "Age",
    "Family_Members_with_Passes",
    "Previously_Owned_Passes",
    "Club_Member",
    "Distance_from_Park_km",
]

X = df[feature_columns].copy()
y = df["Pass_Type"].copy()

# Boolean -> integer so every model receives numeric input.
X["Club_Member"] = X["Club_Member"].astype(int)

# Consistent display order for the classes.
class_order = ["Gold", "Silver", "No Pass"]


# ============================================================
# 3. SINGLE SHARED TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    shuffle=True,
    stratify=y,
)

print("Training data shape:", X_train.shape)
print("Test data shape:    ", X_test.shape)


# ============================================================
# 4. DEFINE THE MODELS
#    (hyperparameters mirror each individual script)
# ============================================================

# Logistic Regression is wrapped in a scaler pipeline because it
# is the only model that is sensitive to feature scale.
logistic_regression = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
])

decision_tree = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=3,
    random_state=42,
)

random_forest = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
)

xgboost_model = XGBClassifier(
    objective="multi:softprob",
    num_class=len(class_order),
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.80,
    colsample_bytree=0.80,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1,
)

models = {
    "Logistic Regression": logistic_regression,
    "Decision Tree": decision_tree,
    "Random Forest": random_forest,
    "XGBoost": xgboost_model,
}


# ============================================================
# 5. TRAIN, PREDICT AND COLLECT METRICS
# ============================================================

# XGBoost needs integer class labels, so we encode the target
# once and decode its predictions back to the original strings.
label_encoder = LabelEncoder().fit(y_train)


def fit_predict(name, model):
    """Train a model and return string-label predictions."""
    if name == "XGBoost":
        model.fit(X_train, label_encoder.transform(y_train))
        train_pred = label_encoder.inverse_transform(model.predict(X_train))
        test_pred = label_encoder.inverse_transform(model.predict(X_test))
    else:
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
    return train_pred, test_pred


results = []
predictions = {}

for name, model in models.items():
    train_pred, test_pred = fit_predict(name, model)
    predictions[name] = test_pred

    results.append({
        "Model": name,
        "Train Accuracy": accuracy_score(y_train, train_pred),
        "Test Accuracy": accuracy_score(y_test, test_pred),
        "Macro Precision": precision_score(
            y_test, test_pred, average="macro", zero_division=0
        ),
        "Macro Recall": recall_score(
            y_test, test_pred, average="macro", zero_division=0
        ),
        "Macro F1": f1_score(y_test, test_pred, average="macro"),
        "Weighted F1": f1_score(y_test, test_pred, average="weighted"),
    })

comparison = pd.DataFrame(results)
comparison["Accuracy Gap"] = (
    comparison["Train Accuracy"] - comparison["Test Accuracy"]
)

# Rank by test macro F1 (a balanced metric for the imbalanced classes).
comparison = comparison.sort_values(
    by="Macro F1", ascending=False
).reset_index(drop=True)


# ============================================================
# 6. PRINT AND EXPORT THE COMPARISON TABLE
# ============================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON – TEST SET PERFORMANCE")
print("=" * 70)
print(comparison.round(4).to_string(index=False))

best_model = comparison.iloc[0]["Model"]
print(f"\nBest model by test Macro F1: {best_model}")

comparison.round(4).to_csv("model_comparison_results.csv", index=False)
print("\nSaved: model_comparison_results.csv")


# ============================================================
# 7. GROUPED BAR CHART – ACCURACY VS MACRO F1
# ============================================================

plot_data = comparison.set_index("Model")[["Test Accuracy", "Macro F1"]]

ax = plot_data.plot(kind="bar", figsize=(10, 6))
ax.set_title("Model Comparison – Test Accuracy and Macro F1")
ax.set_ylabel("Score")
ax.set_xlabel("")
ax.set_ylim(0, 1.05)
plt.xticks(rotation=0)
plt.legend(loc="lower right")

# Annotate each bar with its value.
for container in ax.containers:
    ax.bar_label(container, fmt="%.3f", padding=2, fontsize=8)

plt.tight_layout()
plt.savefig("model_comparison_scores.jpg", dpi=150, bbox_inches="tight")
print("Saved: model_comparison_scores.jpg")
plt.show()


# ============================================================
# 8. HORIZONTAL RANKING CHART – MACRO F1 (BEST TO WORST)
# ============================================================

# `comparison` is already sorted by Macro F1 (best first). For a
# horizontal bar chart we reverse it so the best model sits on top.
ranking = comparison.iloc[::-1].reset_index(drop=True)

# Per-bar colours: Random Forest highlighted, Decision Tree greyed
# out to read as the clearly weaker model, everyone else neutral.
def bar_color(model):
    if model == "Random Forest":
        return "#2ca02c"          # highlight (green)
    if model == "Decision Tree":
        return "#cccccc"          # de-emphasised (light grey)
    return "#4c72b0"              # neutral (blue)

colors = [bar_color(m) for m in ranking["Model"]]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(ranking["Model"], ranking["Macro F1"], color=colors)

# Hatch + dashed edge to visually separate Decision Tree as weaker.
for bar, model in zip(bars, ranking["Model"]):
    if model == "Decision Tree":
        bar.set_hatch("//")
        bar.set_edgecolor("#888888")
        bar.set_linestyle("--")
        bar.set_linewidth(1.2)

ax.set_title("Model Ranking – Macro F1 (best to worst)")
ax.set_xlabel("Macro F1")
ax.set_xlim(0, 1.12)

# Main label: Macro F1. Smaller secondary label: Test Accuracy.
for bar, f1, acc in zip(bars, ranking["Macro F1"], ranking["Test Accuracy"]):
    y = bar.get_y() + bar.get_height() / 2
    x = bar.get_width()
    ax.text(x + 0.01, y, f"{f1:.3f}",
            va="center", ha="left", fontsize=11, fontweight="bold")
    ax.text(x + 0.07, y, f"acc {acc:.3f}",
            va="center", ha="left", fontsize=8, color="#555555")

# Legend explaining the colour coding.
from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor="#2ca02c", label="Random Forest (best)"),
    Patch(facecolor="#4c72b0", label="Other models"),
    Patch(facecolor="#cccccc", hatch="//", edgecolor="#888888",
          label="Decision Tree (weakest)"),
]
ax.legend(handles=legend_handles, loc="lower right", fontsize=8)

plt.tight_layout()
plt.savefig("model_comparison_ranking.jpg", dpi=150, bbox_inches="tight")
print("Saved: model_comparison_ranking.jpg")
plt.show()


# ============================================================
# 9. CONFUSION MATRICES FOR ALL MODELS
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, (name, test_pred) in zip(axes.ravel(), predictions.items()):
    cm = confusion_matrix(y_test, test_pred, labels=class_order)
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_order,
    ).plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(name)

fig.suptitle("Confusion Matrices – Test Set", fontsize=14)
plt.tight_layout()
plt.savefig("model_comparison_confusion_matrices.jpg", dpi=150, bbox_inches="tight")
print("Saved: model_comparison_confusion_matrices.jpg")
plt.show()
