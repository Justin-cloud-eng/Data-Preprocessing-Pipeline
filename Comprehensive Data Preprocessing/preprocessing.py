import pandas as pd
import numpy as np
FILE_NAME = "C:\\Users\\USER\\OneDrive\\Documents\\sample_data.csv"

#1. Load the DataSet
print(f"*****1. Loading Data: {FILE_NAME}******")
df = pd.read_csv(FILE_NAME)

print(f"Initial shape: {df.shape}")
print(f"Initial missing values: \n{df.isnull().sum()[df.isnull().sum() > 0]}")
print("-" * 30)


#2. DATA CLEANING: Handling the missing Valuess
# Target Column: Total Fatalities (Numerical)
# Strategy: Impute missing numerical values with the median, a robust measure.
FATALITIES_COL = "Total_Fatalities"
median_fatalities = df[FATALITIES_COL].median()
#df[FATALITIES_COL].fillna(median_fatalities, inplace = True)
df.fillna({FATALITIES_COL: median_fatalities}, inplace = True)


# Target Column: Weather Condtion (Categorical /Text)
# Strategy: Drops rows where a critical feature like weather condition is missing.
WEATHER_COL = "Weather_Condition"
df.dropna(subset = [WEATHER_COL], inplace = True)


print("***** 2. Data Cleaning Summary *****")
print(f"Missing {FATALITIES_COL} filled with median: {median_fatalities}")
print(f"Rows dropped due to missing {WEATHER_COL}. New shape: {df.shape}")
print(f"Remaining Missing Values Check: {df.isnull().sum().max()}") # Should be 0 or very small
print("-" * 30)


# 3. Feature Transformation : Categorical to Numerical
# Target Column: "Aircraft Category" (Categorical Text)
# Strategy: Convert Categorical Data into numerical binary flags (One-Hot Encoding via get_dummies)
# We only keep the required columns for our regression model
# (assuming we predict fatalities based on Engine Type and Flight Phase)
FEATURES = ["Engine_Type", "Flight_Phase"]
TARGET = FATALITIES_COL
# Convert categorical features into dummy variables
df_encoded = pd.get_dummies(df[FEATURES],drop_first = True,  prefix = FEATURES)
# Combine the Target Column and the new features
df_final = pd.concat([df_encoded, df[TARGET]], axis = 1)

print("***** 3. Feature Transformation Summary *****")
print(f"Original features transformed: {FEATURES}")
print(f"Final Dataframe Columns: \n{df_final.columns.tolist()}")
print("-" * 30) 

# 4. ---Bridging to NumPy and Data Slicing---
# Separate the processed data into X (Features) and Y (Target)
X_cols = df_final.columns.drop(TARGET)
X = df_final[X_cols].values # Convert to NumPy arrays for Project 2
y = df_final[TARGET].values # Convert to NumPy array



# Report final data types
print("***** 4. Bridging to NumPy Array *****")
print(f"X (Features) created with shape: {X.shape} and dtype: {X.dtype}")
print(f"y (Target) created with shape: {y.shape} and dtype: {y.dtype}")

# Save the clean NumPy array for Project 2
np.save("X_clean.npy", X)
np.save("y_target.npy", y)

print("\nSUCCESS: Data Pipeline complete. Saved X_clean.npy and y_target.npy for Project 2")


