from sklearn.cluster import KMeans
import numpy as np

# Sample data
X = np.array([[1,2], [1,4], [1,0],
              [10,2], [10,4], [10,0]])

# Create K-Means model
model = KMeans(n_clusters=2)

# Train model
model.fit(X)

# Cluster labels
print(model.labels_)