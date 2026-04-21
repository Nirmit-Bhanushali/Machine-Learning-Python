import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add species names instead of numbers
df['species'] = iris.target
df['species'] = df['species'].map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})

# Create Bar Plot
sns.barplot(x='species', y='sepal length (cm)', data=df)

plt.title("Bar Plot - Average Sepal Length per Species")
plt.xlabel("Species")
plt.ylabel("Sepal Length (cm)")
plt.show()