import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Plot Density Plot
df['sepal length (cm)'].plot(kind='density')

plt.title("Density Plot of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.show()