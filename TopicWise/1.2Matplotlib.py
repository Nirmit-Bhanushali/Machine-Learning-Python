import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Menu
print("Choose Visualization:")
print("1. Histogram")
print("2. Bar Graph")
print("3. Density Plot")
print("4. Pie Chart")
print("5. Box Plot")
print("6. Scatter Plot")
print("7. Correlation Matrix")
print("8. Scatter Matrix")

choice = int(input("Enter your choice (1-8): "))

match choice:
    
    case 1:
        plt.hist(df['sepal length (cm)'])
        plt.title("Histogram")
        plt.xlabel("Sepal Length")
        plt.ylabel("Frequency")
        plt.show()

    case 2:
        df.groupby('species')['sepal length (cm)'].mean().plot(kind='bar')
        plt.title("Bar Graph")
        plt.xlabel("Species")
        plt.ylabel("Average Sepal Length")
        plt.show()

    case 3:
        df['sepal length (cm)'].plot(kind='density')
        plt.title("Density Plot")
        plt.show()

    case 4:
        counts = df['species'].value_counts()
        plt.pie(counts, labels=['Setosa','Versicolor','Virginica'], autopct='%1.1f%%')
        plt.title("Pie Chart")
        plt.show()

    case 5:
        df.plot(kind='box')
        plt.title("Box Plot")
        plt.show()

    case 6:
        plt.scatter(df['sepal length (cm)'], df['petal length (cm)'])
        plt.title("Scatter Plot")
        plt.xlabel("Sepal Length")
        plt.ylabel("Petal Length")
        plt.show()

    case 7:
        plt.matshow(df.corr())
        plt.title("Correlation Matrix")
        plt.colorbar()
        plt.show()

    case 8:
        pd.plotting.scatter_matrix(df, figsize=(8,8))
        plt.show()

    case _:
        print("Invalid choice")