import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add target (species)
df['species'] = iris.target

# Calculate average sepal length for each species
avg_values = df.groupby('species')['sepal length (cm)'].mean()

# Plot Bar Graph
avg_values.plot(kind='bar')

plt.title("Average Sepal Length per Species")
plt.xlabel("Species")
plt.ylabel("Sepal Length (cm)")
plt.show()