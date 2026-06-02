from sklearn.metrics import mean_absolute_error

# Actual and Predicted values
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

# MAE
mae = mean_absolute_error(y_true, y_pred)

# Output
print("Mean Absolute Error:", mae)
