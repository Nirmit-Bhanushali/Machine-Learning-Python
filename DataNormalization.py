from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Sample data
data = np.array([[10], [20], [30], [40], [50]])

# Min-Max Normalization
scaler = MinMaxScaler()

normalized_data = scaler.fit_transform(data)

print(normalized_data)