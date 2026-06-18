import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
data = pd.read_csv(r"Production_Log_01.csv")
machine_settings = pd.read_csv(r"Machine_Settings_Log_01.csv")

merged_data = pd.merge(
    machine_settings,
    data,
    left_on='configuration_log_ID',
    right_on='id',
    how='inner'
)

print("\nMerged Dataset:")
print(merged_data.head())

errors = merged_data.dropna(subset=['error', 'error_type'])

merged_data['error_binary'] = merged_data['error'].map({
    'yes': 1,
    'no': 0
})

# Select numeric columns
numeric_cols = merged_data.select_dtypes(include='number')

# Correlation with errors
correlations = numeric_cols.corr()['error_binary']

# Remove self-correlation
correlations = correlations.drop('error_binary')

# Sort strongest effects
correlations = correlations.abs().sort_values(ascending=False)

print(correlations)

print(errors[['error', 'error_type']])

errors = errors[
    (errors['error'].astype(str).str.strip() != '') &
    (errors['error_type'].astype(str).str.strip() != '')
]

errors = errors[['error', 'error_type']]
print("\nRows containing errors:")
print(errors)

# Convert error column to binary
# yes = 1, no = 0
data['error_binary'] = data['error'].map({'yes': 1, 'no': 0})

# Select only numeric columns
numeric_cols = data.select_dtypes(include='number')

# Calculate correlation with error
correlations = numeric_cols.corr()['error_binary']

# Remove self-correlation
correlations = correlations.drop('error_binary')

# Sort by strongest effect
correlations = correlations.abs().sort_values(ascending=False)

print("Factors affecting errors:")
print(correlations)