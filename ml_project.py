import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("student_performance.csv")

# Fill missing values
df['parent_education'] = df['parent_education'].fillna(df['parent_education'].mode()[0])

# Convert text into numbers
le = LabelEncoder()

df['gender'] = le.fit_transform(df['gender'])
df['parent_education'] = le.fit_transform(df['parent_education'])
df['internet_access'] = le.fit_transform(df['internet_access'])
df['extracurricular'] = le.fit_transform(df['extracurricular'])
df['passed'] = le.fit_transform(df['passed'])

# Features and target
X = df[['study_hours_per_week', 'attendance_rate', 'previous_score']]
y = df['passed']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Plot graph
sns.heatmap(cm, annot=True, fmt='d')

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()
