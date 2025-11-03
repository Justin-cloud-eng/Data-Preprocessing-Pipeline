import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set a clean style for all plots
sns.set_style("whitegrid")

# --- 1. CONFIGURATION AND DATA LOADING ---
FILE_NAME = "C:\\Users\\USER\\OneDrive\\Documents\\sample_data.csv"

if not os.path.exists(FILE_NAME):
    print(f"ERROR: '{FILE_NAME}' not found. Please ensure the NTSB data file is in the same directory")
    exit ()

# Load the original data and model results
df_raw = pd.read_csv(FILE_NAME)
try:
    y_true = np.load("y_target.npy").flatten()
    y_hat = np.load("y_hat_predictions.npy").flatten()
except FileNotFoundError:
    print("ERROR: Missing model result files (y_target.npy, y_hat_predictions.npy). Please run Projects 1 and 2 first.")
    exit()

# Reapply cleaning steps to get a clean DataFrame to match y_true and y_hat indices
# This is crucial for matching the original categorical columns
WEATHER_COL = "Weather_Condition"
FATALITIES_COL = "Total_Fatalities"

# Fill missing fatalities (as done in preprocessing)
median_fatalities = df_raw[FATALITIES_COL].median()
df_clean = df_raw.copy()
df_clean.fillna({FATALITIES_COL: median_fatalities}, inplace = True)

# Drop rows where WEATHER_COL is missing (as done in Project 1)
df_clean = df_clean.dropna(subset = [WEATHER_COL], inplace = False).reset_index(drop = True)

# --- 2. MERGE PREDICTIONS INTO DATAFRAME ---
# Add the true and predicted values to the clean DataFrame
df_viz = df_clean[["Aircraft_Category", "Engine_Type", "Flight_Phase", FATALITIES_COL]].copy()
df_viz["True_Fatalities"] = y_true
df_viz["Predicted_Fatalities"] = y_hat
print(df_viz.to_string())

print("*** Data Visualization Setup Complete ***")

# --- 3. PLOT 1: CATEGORICAL EDA (seaborn) ---
# Visualize the average number of fatalities by the type of Engine.
plt.figure(figsize = (8, 5))
# Use a bar plot to show the central tendency (mean) and its variability (confidence interval)
sns.barplot(data = df_viz, x = "Engine_Type", y = "True_Fatalities",  palette = "viridis")
plt.title("Average Total Fatalities by Engine Type (EDA)", 
          fontsize = 14, fontweight = "bold", family = "serif")
plt.xlabel("Aircraft Engine Type", fontsize = 12)
plt.ylabel("Average Total Fatalites")
plt.xticks(rotation = 15)
plt.tight_layout()
plt.show()
#print(df_viz.to_string())

# --- 4. PLOT 2: MODEL DIAGNOSTICS (Matplotlib) ---
# Visualize performance by comparing True vs Predicted Fatalities
# For a perfect model all points would lie on the  y = x line
plt.figure(figsize = (7, 7))

# Scatter plot of True vs Predicted values
plt.scatter(df_viz["True_Fatalities"], df_viz["Predicted_Fatalities"], 
            alpha = .6, color = "darkblue", edgecolor = "black", label = "Predicted points")
"""
# 2. Plot the ideal line (y=x) for comparison
max_val = max(df_viz["True_Fatalities"].max(), df_viz["Predicted_Fatalities"].max()) * 1.05
plt.plot([0, max_val], [0, max_val] , # define the x = y line
         color = "red", linestyle = "--", linewidth = 2, label = "Ideal Prediction line")
plt.title("True vs Predicted Total Fatalites (Model Diagnosis)", fontsize = 14, 
          fontweight = "bold", family = "arial")
plt.xlabel("True Total Fatalities", fontsize = 12)
plt.ylabel("Predicted Total Fatalities", fontsize = 12)
plt.legend()
plt.grid(True, linestyle = ":", alpha = 0.7)
"""
plt.tight_layout()
plt.show()

print("\nSUCCESS: Two diagnostic plots generated for data visualization and model performance evaluation. Visualization Complete.")