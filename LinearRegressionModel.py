import numpy as np
from sklearn.linear_model import LinearRegression

# Sample dataset
X = np.array([1000, 1500, 2000, 2500, 3000]).reshape(-1,1)
y = np.array([200, 300, 400, 500, 600])

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Prediction
new_size = np.array([[3500]])

prediction = model.predict(new_size)

print("Predicted Price:", prediction[0])