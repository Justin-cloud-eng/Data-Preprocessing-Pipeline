import numpy as np
import pandas as pd
import os

# --- 1. CONFIGURATION AND DATA LOADING ---
# Load the target vector (y) from Project 1 and the results from Project 2
try:
    # y is the true target values from Project 1
    y_true = np.load("y_target.npy").reshape(-1, 1)
    # theta is the calculated parameter from project 2
    theta_final = np.load("theta_final.npy")
    # y_hat is the predicted values from Project 2
    y_hat_predictions = np.load("y_hat_predictions.npy")

    print("*** 1. Required Data Loaded ***")
    print("-" * 30)

except FileNotFoundError:
    print("ERROR: Missing required files (y_target.npy, theta_final.npy, y_hat_predictions.npy). ")
    print("Please ensure that Project 1 and Project 2 were run successfully. ")
    exit()

# --- 2. COST FUNCTION: MEAN SQUARED ERROR (MSE) ---
def calculate_mse(y_true, y_predicted):
    """
    Implements the Mean Squared Error (MSE) cost function using NumPy vectorization.
    MSE = (1/m) * sum((y_hat - y**2)
    """
    # Number of data points (m)
    m = y_true.shape[0]

    # Calculate the squared residuals (y_hat - y)**2
    squared_errors = (y_predicted - y_true)**2

    # Calculate the mean of the squared errors
    mse = np.sum(squared_errors)/m

    return mse

# Calculate the final MSE score
final_mse_score = calculate_mse(y_true, y_hat_predictions)

print("*** 2. Model Cost Function Calculation ***")
print(f"Mean Squared Error (MSE): {final_mse_score:.4f}")
print("-" * 30)

# --- 3. PARAMETER REPORTING (Pandas Focus) ---
# Since we used the encoded features from Project 1, we need to create plausible labels.
# This step assumes a certain structure of the features created in Project 1's get_dummies step.

# 1. Define feature labels (Must match due the order of columns in X_b from Project 2)
# The first label must be the intercept
# The remaining labels correspond to the binary columns created via get_dummies.
features_labels = ['Intercept (Bias Column)','Engine_Type_Turbo-shaft', 'Engine_Type_Turbojet', 'Flight_Phase_Cruise','Flight_Phase_Landing',
                    'Flight_Phase_Maneuvering', 'Flight_Phase_Takeoff']

# Ensure the number of labels matches the number of parameters (theta)
if len(features_labels)!= theta_final[0].shape:
    print("WARNING: Feature label count does not match the size of theta_final.")
    print(len(features_labels))
    print(theta_final[0].shape)
# Exit or proceed with a default index if mismatch occurs

# 2. Convert the theta vector into a Pandas Series with descriptive labels
parameters_series = pd.Series(theta_final.flatten(), index = features_labels, name = "Calculated Parameter (theta)")

print("*** 3. Parameter Reporting (Pandas Series) ***")
print(parameters_series.to_string())
print("-" * 30)

# --- 4. CONCLUSION AND PORTFOLIO OUTPUT ---
print("\n Model Evaluation Summary")
print(f"The Linear Regression model fit was evaluated against the training data.")
print(f"Final Mode Performance (Lower is better): MSE = {final_mse_score:.4f}")
print(f"The coefficients (weights) for each feature are listed above, indicating their influence on predicting 'Total Fatalites'")