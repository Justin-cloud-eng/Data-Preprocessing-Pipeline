import numpy as np
# --- 1. CONFIGURATION AND DATA LOADING ---
# Load the clean NumPy arrays created by Project 1
try:
    X = np.load("X_clean.npy")
    y = np.load("y_target.npy")

    # Ensure y is a column vector for matrix multiplication consistency
    y = y.reshape(-1, 1)
    print("*** 1. Data Loaded Successfully ***")
    print(f"Features (X) Shape: {X.shape}")
    print(f"Target (y) Shape: {y.shape}")
    print("-" * 30)

except FileNotFoundError:
    print("ERROR: Could not find 'X_clean.npy' or 'y_clean.npy'. " )
    print("Please ensure Project 1 was run successfully and the files exist. ")
    exit()

# --- 2. DESIGN MATRIX CREATION (Adding the bias term) ---
# The Linear Regression equation (y = X * theta) requires an intercept or bias term
# This is added by creatin a column of ones and horizontally stacking it with X.
print("*** 2. Design Matrix Creation ***")
# 1. Create a column vector of ones (the bias column)
ones = np.ones((X.shape[0], 1))

# 2. Horizontally stack the ones column with the feature matrix X
X_b = np.hstack((ones, X))
print(f"Design Matrix (X_b) Shape: {X_b.shape}")
print("-" * 30)

#--- 3. THE NORMAL EQUATION SOLVER (NumPy Core) ---
# The Normal Equation provides the closed form solution for the optimal parameter vector (theta)
# theta = (X_b^T * X_b)^-1 * X_b^T * y
print("*** 3. Solving for Parameters (theta) using Normal Equaiton ***")
# Step A: Calculate (X_b^T * X_b)
# Use .T for transpose and @ for matrix multiplication
X_T_X = X_b.T @ X_b
print(X_T_X.shape)
# Step B: Calculate the inverse of (X_b^T * X_b)
# Use np.linalg.inv() for the matrix inverse
X_T_X_inv = np.linalg.inv(X_T_X)
print(X_T_X_inv.shape)

# Step C: Calculate X_b^T * y
X_T_y = X_b.T @ y
print(X_T_y.shape)

# Step D: Final calculation for theta 
theta = X_T_X_inv @ X_T_y
print(f"Calculated Parameter Vector (theta) shape: {theta.shape}")
print(theta)
print("-" * 30)


# --- 4. VECTORIZED PREDICTION FUNCTION ---
# This function demonstrates how the model will make predictions using the calculated theta.
# Prediciton: y_hat = X_b * theta
def predict(X_input, theta_vector):
    
    # Generates predicted target values (y_hat) using the calcualted parameters.
    
    # The input X_input msut already include the bias term (column of ones).
    return X_input @ theta_vector

# Generate predictions for the training data
y_hat = predict(X_b, theta)
print(y_hat)
print(y_hat.shape)

# --- 5. SAVING OUTPUT FOR PROJECT 3 ---
# Save the final parameter vector (theta) and the predictions (y_hat) for Project 3 (Evaluation)
np.save("theta_final.npy", theta)
np.save("y_hat_predictions.npy", y_hat)

print("SUCCESS: Linear Regression Solver complete.")
print("Final parameters (theta_final.npy) and predictions (y_hat_predictions.npy) saved for Project 3. ")
