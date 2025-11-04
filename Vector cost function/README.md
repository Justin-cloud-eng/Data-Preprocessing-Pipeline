# Vectorized Cost Function and Evaluation (Bridging Math/ Code)
## Project Goal
This final project demonstrates the crucial step of **quantifying model performance** and **reporting results** in a clear, professional format. It validates the effectiveness of the parameters (theta) calculated in Project 2

**Focus Tools:** NumPy (Vectorization for Cost) and Pandas (Professional Reporting)
## Mean Squared Error (MSE)
The **Cost Function** implemented is the Mean Squared Error(MSE). This function measures the average squared difference between the predicted values (y_hat) and the true values (y). A lower MSE indicates a better fit of the model to the data.

MSE = sum((y_hat - y)**)/m

### Implementation Details:
* **Vectorization:** The MSE function is implemented using pure NumPy operations.
This showcases efficient, vectorized computation, which is standard practice in high-performance machine learning code.

## Parameter Reporting via Pandas
The calculated parameter vector (theta) from Project 2 is a simple NumPy array of numbers. To make the output useful and interpretable, it is converted into a **Pandas Series**

### Key Reporting Features:
* **Descriptive Labels:** Each coefficient is assigned a meaningful label (e.g., "Intercept", "Engine_Type", etc) that corresponds to the features created in Project 1.
* **Clarity:** This reporting format is essential for communicating the model's learned weights to stakeholders, making the raw mathematical output immediately understandable.

## Final Model Summary
The script outputs the two most important pieces of information for model analysis:
1. **Final MSE Score:** The quantitative measure of model error.
2. **Parameter Series:** The feature weights that define the linear relationship.

**Final MSE** - Calculated by the script. It is the average squared error of the predictions.
**Intercept** - Calcualted by the script. The predicted target value when all the features are zero.
**Feature weights** - Calculated by the script. They are the changes in the target variable for a one-unit change in that feature (holding others constant).