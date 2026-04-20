import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load standard dataset from sklearn
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Display first 5 rows
print("Iris Dataset:")
print(df.head())

# Plot Histogram
plt.hist(df['sepal length (cm)'], bins=10, edgecolor='black')
plt.title('Histogram of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.show()