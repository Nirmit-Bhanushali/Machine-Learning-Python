from sklearn.metrics import r2_score

# Actual and Predicted values
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

# R² Score
r2 = r2_score(y_true, y_pred)

# Adjusted R²
n = len(y_true)      # number of samples
p = 1                # number of independent variables

adj_r2 = 1 - (1-r2)*(n-1)/(n-p-1)

# Output
print("R² Score:", r2)
print("Adjusted R² Score:", adj_r2)