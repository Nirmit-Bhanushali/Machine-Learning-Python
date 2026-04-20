import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Plot Box and Whisker Plot
df.plot(kind='box')

plt.title("Box Plot of Iris Features")
plt.ylabel("Values")
plt.show()