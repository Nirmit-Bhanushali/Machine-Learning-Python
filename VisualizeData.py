import pandas as pd
import matplotlib.pyplot as plt

# Import CSV file
data = pd.read_csv("ML\students.csv")

print("Student Dataset")
print(data)

# -------- Visualization 1: Bar Chart --------
subject_marks = data.groupby("Subject")["Marks"].mean()

subject_marks.plot(kind='bar')
plt.title("Average Marks per Subject")
plt.xlabel("Subject")
plt.ylabel("Average Marks")
plt.show()

# -------- Visualization 2: Pie Chart --------
grade_counts = data["Grade"].value_counts()

plt.pie(grade_counts, labels=grade_counts.index, autopct='%1.1f%%')
plt.title("Grade Distribution")
plt.show()

# -------- Visualization 3: Histogram --------
plt.hist(data["Marks"], bins=5)
plt.title("Distribution of Marks")
plt.xlabel("Marks")
plt.ylabel("Number of Students")
plt.show()

# -------- Visualization 4: Scatter Plot --------
plt.scatter(data["Student_ID"], data["Marks"])
plt.title("Student Marks Scatter Plot")
plt.xlabel("Student ID")
plt.ylabel("Marks")
plt.show()