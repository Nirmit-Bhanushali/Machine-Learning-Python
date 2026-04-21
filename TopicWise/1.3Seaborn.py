import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add species names
df['species'] = iris.target
df['species'] = df['species'].map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})

# Loop for menu
while True:
    print("\nChoose Seaborn Plot:")
    print("1. Bar Plot")
    print("2. Count Plot")
    print("3. Distribution Plot")
    print("4. Point Plot")
    print("5. Facet Grid")
    print("6. Pair Grid")
    print("0. Exit")

    choice = int(input("Enter your choice (0-6): "))

    match choice:

        case 1:
            sns.barplot(x='species', y='sepal length (cm)', data=df)
            plt.title("Bar Plot")
            plt.show()

        case 2:
            sns.countplot(x='species', data=df)
            plt.title("Count Plot")
            plt.show()

        case 3:
            sns.histplot(df['sepal length (cm)'], kde=True)
            plt.title("Distribution Plot")
            plt.show()

        case 4:
            sns.pointplot(x='species', y='sepal length (cm)', data=df)
            plt.title("Point Plot")
            plt.show()

        case 5:
            g = sns.FacetGrid(df, col="species")
            g.map(sns.histplot, "sepal length (cm)")
            plt.show()

        case 6:
            g = sns.PairGrid(df, hue="species")
            g.map_diag(sns.histplot)
            g.map_offdiag(sns.scatterplot)
            g.add_legend()
            plt.show()

        case 0:
            print("Exiting program...")
            break

        case _:
            print("Invalid choice! Please try again.")