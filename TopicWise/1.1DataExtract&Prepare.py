import pandas as pd

data = {
    "country": ["Brazil", "Russia", "India", "China", "South Africa"],
    "capital": ["Brasilia", "Moscow", "New Delhi", "Beijing", "Pretoria"],
    "area": [8.516, 17.10, 3.286, 9.597, 1.221],
    "population": [200.4, 143.5, 1252, 1357, 52.98]
}

# Create DataFrame
df = pd.DataFrame(data)

# Display table
print("\nData Table:")
print(df)

# Split dataset
train_size = int(0.7 * len(df))

train_data = df[:train_size]
test_data = df[train_size:]

print("\nTraining Data:\n", train_data)
print("\nTesting Data:\n", test_data)