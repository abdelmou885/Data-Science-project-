import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import pandas as pd


#reading datasets
production_log = pd.read_csv('Production_Log_01.csv')
machine_settings_log =  pd.read_csv('Machine_Settings_Log_01.csv')

#merging production and machine settings
df = production_log.merge(machine_settings_log, on="configuration_log_ID", how="left")

df_clean = df.copy()

# Remove extreme / unrealistic width values
df_clean = df_clean[df_clean["width"] < 1000]

# Optional: remove extreme target outliers as well
df_clean = df_clean[df_clean["weight_in_kg"] < df_clean["weight_in_kg"].quantile(0.99)]
# --------------------------------------------------
# Linear Regression with substitution
# Predict weight_in_kg using width and width^2
# --------------------------------------------------

x = df_clean[["width"]]
z = df_clean["weight_in_kg"]

model = linear_model.LinearRegression()

# Create substituted variable: width squared
x2 = x["width"] ** 2

# Create dataframe like professor's example
model_data = pd.DataFrame()
model_data["width"] = x["width"]
model_data["width2"] = x2

# Train model
model.fit(model_data, z)

print("Coefficients:")
print(model.coef_)

print("Intercept:")
print(model.intercept_)

# Create values for plotting
x_plot = np.linspace(df_clean["width"].min(), df_clean["width"].max(), 100)

plot_data = pd.DataFrame()
plot_data["width"] = x_plot
plot_data["width2"] = x_plot ** 2

# Predict weight for the plot line
z_predicted = model.predict(plot_data)

plt.figure(figsize=(8, 6))
plt.scatter(df_clean["width"], df_clean["weight_in_kg"], alpha=0.4)
plt.plot(x_plot, z_predicted, color="red")

plt.xlabel("width")
plt.ylabel("weight_in_kg")
plt.title("Linear Regression with Substitution: width and width²")

plt.show()

