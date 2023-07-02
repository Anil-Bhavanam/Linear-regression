# Linear-regression
The code you provided performs various tasks related to data analysis, linear regression, classification, and cross-validation. Here's a description of each section:

# Data Preparation:

The code imports the necessary libraries such as pandas, numpy, matplotlib, missingno, and sklearn.
It reads the "iris.data" file using read_csv() from pandas and assigns column names to the DataFrame.
The independent variables are extracted into the independent_value DataFrame, and the dependent variable is stored in dependent_value.
The missingno library is used to visualize missing values in the dataset.
The dependent variable is label encoded using sklearn's LabelEncoder and stored in dependent_value.
# Linear Regression:

The code defines a function linear_regression(n, y) to perform linear regression.
The independent variables are extracted from the input DataFrame n.
A column of ones is added as the first column in the independent variable matrix to represent the intercept term.
The function calculates the beta values (coefficients) using the normal equation.
The function predicts the dependent variable values using the calculated beta values.
The function returns the R-squared score as a performance measure.
# Classification:

The code calculates the Euclidean distance for each data point in the independent variables.
The distances and the corresponding flower labels are stored in a DataFrame.
The DataFrame is sorted based on the distances in ascending order.
The top three rows of the sorted DataFrame are selected.
The most frequent flower label among the selected rows is determined as the classification result.
# Cross Validation:

No specific code for cross-validation is provided in the code snippet you shared. It seems to be missing from the code.
Please note that the code snippet you provided has some indentation issues, which may cause errors when running it. Ensure that the code is properly indented to maintain the correct structure and avoid indentation errors.
