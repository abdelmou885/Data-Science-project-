import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    mean_absolute_error,
    mean_squared_error,
    silhouette_score
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans


# ==========================================
# 1. LOAD DATA
# ==========================================
prod = pd.read_csv("Production_Log_01.csv")

ms = pd.read_csv(
    "Machine_Settings_Log_01.csv",
    names=["configuration_log_ID", "gear", "rotationspeed"],
    header=0
)

prod.columns = prod.columns.str.strip()
ms.columns = ms.columns.str.strip()

print("Production shape:", prod.shape)
print("Machine settings shape:", ms.shape)
print("\nProduction columns:", prod.columns.tolist())
print("Machine settings columns:", ms.columns.tolist())

# ==========================================
# 2. MERGE
# ==========================================
prod["configuration_log_ID"] = pd.to_numeric(prod["configuration_log_ID"], errors="coerce")
ms["configuration_log_ID"] = pd.to_numeric(ms["configuration_log_ID"], errors="coerce")

prod = prod.dropna(subset=["configuration_log_ID"])
ms = ms.dropna(subset=["configuration_log_ID"])
ms = ms.drop_duplicates(subset=["configuration_log_ID"])

merged = pd.merge(prod, ms, on="configuration_log_ID", how="inner")

print("\nMerged shape:", merged.shape)
print(merged.head())

# ==========================================
# 3. TARGET CLEANING
# ==========================================
df = merged.dropna(subset=["error_type"]).copy()

for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("\nFinal modeling shape:", df.shape)
print("\nError type distribution:")
print(df["error_type"].value_counts())

# ==========================================
# 4. BASIC DESCRIPTION
# ==========================================
print("\nData info:")
print(df.info())

print("\nNumerical description:")
print(df.describe(include=[np.number]).T)

print("\nCategorical description:")
print(df.describe(include=["object"]).T)

# ==========================================
# 5. CORRELATION ANALYSIS
# ==========================================
le_corr = LabelEncoder()
df["error_type_encoded"] = le_corr.fit_transform(df["error_type"])

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
drop_for_corr = ["configuration_log_ID", "error_type_encoded"]
corr_features = [c for c in numeric_cols if c not in drop_for_corr]

corr_df = df[corr_features + ["error_type_encoded"]].corr()

plt.figure(figsize=(16, 10))
sns.heatmap(corr_df, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("plot_01_correlation_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plot_01_correlation_heatmap.png")

target_corr = corr_df["error_type_encoded"].drop("error_type_encoded").sort_values(key=np.abs, ascending=False)

print("\nTop features correlated with error_type:")
print(target_corr.head(15))

plt.figure(figsize=(10, 6))
target_corr.head(15).sort_values().plot(kind="barh", color="teal")
plt.title("Top 15 Features Correlated with error_type")
plt.xlabel("Correlation with error_type_encoded")
plt.tight_layout()
plt.savefig("plot_02_target_correlation.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plot_02_target_correlation.png")

# ==========================================
# 6. FEATURE SELECTION
# ==========================================
input_features = [
    "width", "height", "ionizationclass", "FluxCompensation",
    "pressure", "karma", "modulation",
    "gear", "rotationspeed"
]

available_input_features = [col for col in input_features if col in df.columns]
print("\nSelected features:", available_input_features)

X = df[available_input_features].copy()
y = df["error_type"].copy()

# ==========================================
# 7. PREPROCESSING
# ==========================================
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

# ==========================================
# 8. TRAIN & EVALUATE MODELS
# ==========================================
models = {
    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5
    ),
    "Random Forest": RandomForestClassifier(
        random_state=42,
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42,
        n_estimators=150,
        learning_rate=0.08,
        max_depth=3
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    )
}

results = []

for name, model in models.items():
    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MAE": mae,
        "RMSE": rmse
    })

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)
    print("Accuracy:", acc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_")
    plt.savefig(f"plot_cm_{safe_name}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: plot_cm_{safe_name}.png")

# ==========================================
# 9. MODEL COMPARISON
# ==========================================
results_df = pd.DataFrame(results).sort_values(by=["F1", "Accuracy"], ascending=False)

print("\nFinal model comparison:")
print(results_df)

best_two = results_df.head(2)
print("\nBest two models:")
print(best_two)

plt.figure(figsize=(10, 5))
sns.barplot(data=results_df, x="F1", y="Model", palette="viridis")
plt.title("Model Comparison by F1 Score")
plt.tight_layout()
plt.savefig("plot_03_comparison_f1.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plot_03_comparison_f1.png")

plt.figure(figsize=(10, 5))
sns.barplot(data=results_df, x="RMSE", y="Model", palette="magma")
plt.title("Model Comparison by RMSE")
plt.tight_layout()
plt.savefig("plot_04_comparison_rmse.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plot_04_comparison_rmse.png")

