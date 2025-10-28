# Comprehensive Data Preprocessing Pipeline (Pandas Focus)
## Project Goal
This project is the foundational step for a machine learning model, desinged to demonstrate mastery over the most critical skill in data science: **data cleaning and transformation**.
It processes a real-world dataset, handles common data quality issues, and prepares the output for mathematical modeling in Project 2.

**Focus Tools:** Pandas and fundamental data manipulation techniques.
## Dataset and Context
The project utilizes a sample of the **NTSB Aviation Accident Dataset**, which is notorious for conataining missing values and various categorical text fields.
This aligns with a real-world scenario where data is often incomplete or messy.

## Key Pipeline Steps
The Python script "preprocessing.py" implements the following components:

### 1. Data Ingestion and Initial Assesment
* **Tool:** "pd.read_csv()
* **Action:** Loads the raw "NTSB_Accidents_Sample.csv" file into a Pandas Dataframe.
* **Showcase:** Initial ".shape" and ".isnull().sum()" reports the dimensionality and the extent of missing data.

### 2. Data Cleaning: Handling Missing Values
* **Numerical Imputation:** Missing values in the "Total_Fatalities" column were filled using the **median** of the column to maintain the integrity of the data distribution.
* **Row removal:** Rows missing the critical "Weather_Condition" were dropped using ".dropna()" to ensure data quality for subsequent analysis.

### 3. Feature Transformation: Categorical Encoding
* **Tool:** ".values" attribute and NumPy slicing.
* **Output:** The final, clean, and numerical Pandas DataFrame is converted into two NumPy arrays: **X (Features)** and **y (Target)**.
These are saved as .npy files and are the required input for Project 2 (Linear Regression Solver)

## Final Output
The pipeline successfully produced two clean, numerical NumPy arrays:
* "X_clean.npy": **Shape: (m, n)** - The feature matrix
* "y_target.npy": **Shape: (m, 1)** - The target variable (Total Fatalities)