import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add species names
df['species'] = iris.target
df['species'] = df['species'].map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})

# Create Count Plot
sns.countplot(x='species', data=df)

plt.title("Count Plot - Species Distribution")
plt.xlabel("Species")
plt.ylabel("Count")
plt.show()