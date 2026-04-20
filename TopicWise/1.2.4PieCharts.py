import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add species column
df['species'] = iris.target

# Count number of samples in each species
species_count = df['species'].value_counts()

# Plot Pie Chart
plt.pie(species_count, labels=species_count.index, autopct='%1.1f%%')

plt.title("Distribution of Species")
plt.show()