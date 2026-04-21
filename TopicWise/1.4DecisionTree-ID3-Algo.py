import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

# -------------------------------
# Step 1: Create standard dataset
# -------------------------------
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain',
                'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast',
                'Overcast', 'Rain'],
    
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool',
                    'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Mild',
                    'Hot', 'Mild'],
    
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal',
                 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High',
                 'Normal', 'High'],
    
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong',
             'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong',
             'Weak', 'Strong'],
    
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No',
                   'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes',
                   'Yes', 'No']
}

df = pd.DataFrame(data)

print("Original Dataset:\n")
print(df)

# -----------------------------------
# Step 2: Encode categorical features
# -----------------------------------
encoders = {}
for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    encoders[column] = le

print("\nEncoded Dataset:\n")
print(df)

# -------------------------------
# Step 3: Split into X and y
# -------------------------------
X = df.drop('PlayTennis', axis=1)
y = df['PlayTennis']

# ----------------------------------------
# Step 4: Build Decision Tree using ID3
# ----------------------------------------
model = DecisionTreeClassifier(criterion='entropy', random_state=0)
model.fit(X, y)

# ----------------------------------------
# Step 5: Classify a new sample
# Example: Outlook=Sunny, Temperature=Cool,
#          Humidity=High, Wind=Strong
# ----------------------------------------
new_sample = pd.DataFrame({
    'Outlook': ['Sunny'],
    'Temperature': ['Cool'],
    'Humidity': ['High'],
    'Wind': ['Strong']
})

# Encode new sample using same encoders
for column in new_sample.columns:
    new_sample[column] = encoders[column].transform(new_sample[column])

prediction = model.predict(new_sample)

predicted_label = encoders['PlayTennis'].inverse_transform(prediction)

print("\nNew Sample:")
print("Outlook = Sunny, Temperature = Cool, Humidity = High, Wind = Strong")
print("Predicted Class:", predicted_label[0])

# -------------------------------
# Step 6: Display Decision Tree
# -------------------------------
plt.figure(figsize=(12, 8))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=encoders['PlayTennis'].classes_,
    filled=True
)
plt.title("Decision Tree using ID3 Algorithm")
plt.show()