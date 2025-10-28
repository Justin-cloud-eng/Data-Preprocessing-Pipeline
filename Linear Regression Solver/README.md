# 2. Linear Regression Solver from Scratch (NumPy Core)
## Project Goal
This project is the core demonstration of fundamental machine learning knowledge.
It implements a **Linear Regrssion** model entirely from scratch using only **NumPy** proving an in-depth understanding of the  algorithm's mathematical foundation.

**Focus Tool:** NumPy (for vectorized operations and Linear Algebra)
## Methodology: The Normal Equation 
Instead of using iterative optimization (like Gradient Descent), this solver uses the **Normal Equation**, which
provides  a direct, closed-form solution for the optimal model parameteres.
The Normal Equation is defined as:
    theta = (X^T * X)^-1 * X_T * y

Where:
    * theta is the parameter vector (the model's coefficients).
    * X is the design matrix (features with the added bias column)
    * y is the target vector.

## Key Implementation Steps (NumPy Showcase)
### 1. Design Matrix Creation
* **Goal:** Prepare the feature matrix (X) for the linear algebra calculation
* **Tools:** "np.ones()", "np.hstack()"
* **Action:** A column of ones (the **bias term**) is added to the feature matrix (X), imported from Project 1, creating the desing matrix (X_b)

### 2. Solving the Normal Equation
* **Tool:** "np.linalg.inv()", @
* **Action:** The closed form equation si solved directly using NumPy's highly optimized linear algebra functions, demonstrating **vectorization** and effecient matrix manipulation.

### 3. Vectorized Prediction
* **Tool:** "@" (matrix multiplication)
* **Action:** A reusable "predict()" function is defined, which takes the Desing Matrix and the calculated theta and generates the predicted target values using the formula: y_hat = X_b * theta

## Final Output
The script successfully computed the model parameters and predictions, saving  them for Project 3:
* "theta_final.npy" : The calculated parameter vector (including the intercept term)
* "y_hat_predictions.npy" : The predicted target values (y_hat) for the training data

This output serves as the necessary input for Project 3, where the model's perfromance will be quantified using the Mean Squared Error (MSE)
