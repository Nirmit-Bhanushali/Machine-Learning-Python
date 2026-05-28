import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Sample dataset
data = {'Marks': [50, 60, 70, 80, 90]}

df = pd.DataFrame(data)

# Normalization
scaler = MinMaxScaler()

df['Normalized Marks'] = scaler.fit_transform(df[['Marks']])

print(df)