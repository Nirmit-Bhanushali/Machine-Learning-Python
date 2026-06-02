from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()

X = iris.data

# Reduce dimensions from 4 to 2
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X)

print(X_pca[:5])
