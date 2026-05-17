import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("student_performance.csv")

# Fill missing values
df['parent_education'] = df['parent_education'].fillna(df['parent_education'].mode()[0])

# Convert text columns into numbers
le = LabelEncoder()

df['gender'] = le.fit_transform(df['gender'])
df['parent_education'] = le.fit_transform(df['parent_education'])
df['internet_access'] = le.fit_transform(df['internet_access'])
df['extracurricular'] = le.fit_transform(df['extracurricular'])
df['passed'] = le.fit_transform(df['passed'])

# Correlation matrix
correlation = df.corr(numeric_only=True)

# Create heatmap
sns.heatmap(correlation, annot=True)

# Title
plt.title("Correlation Heatmap")

# Show graph
plt.show()
