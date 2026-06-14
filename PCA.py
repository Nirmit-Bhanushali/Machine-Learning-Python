from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import pandas as pd

# Load dataset
iris = load_iris()

X = iris.data

# Apply PCA
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X)

# Display transformed data
df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])

print(df.head())