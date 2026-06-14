from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()

X = iris.data
y = iris.target

# Create Neural Network model
model = MLPClassifier(hidden_layer_sizes=(5), max_iter=1000)

# Train model
model.fit(X, y)

# Prediction
prediction = model.predict([X[0]])

print("Predicted Class:", prediction)