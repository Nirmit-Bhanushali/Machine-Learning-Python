import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from pandas.plotting import scatter_matrix

# Load dataset
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Scatter Matrix Plot
scatter_matrix(df, figsize=(8, 8))

plt.suptitle("Scatter Matrix Plot of Iris Dataset")
plt.show()