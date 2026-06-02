import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Dataset
data = {
    'Outlook': ['Sunny','Sunny','Overcast','Rain','Rain'],
    'Temperature': ['Hot','Hot','Hot','Mild','Cool'],
    'Humidity': ['High','High','High','High','Normal'],
    'Wind': ['Weak','Strong','Weak','Weak','Weak'],
    'PlayTennis': ['No','No','Yes','Yes','Yes']
}

# Create DataFrame
df = pd.DataFrame(data)

# Encode categorical data
le = LabelEncoder()

for col in df.columns:
    df[col] = le.fit_transform(df[col])

# Features and target
X = df.drop('PlayTennis', axis=1)
y = df['PlayTennis']

# Create ID3 Decision Tree
model = DecisionTreeClassifier(criterion='entropy')

# Train model
model.fit(X, y)

# Prediction
prediction = model.predict([[2, 1, 0, 1]])

print("Prediction:", prediction)
