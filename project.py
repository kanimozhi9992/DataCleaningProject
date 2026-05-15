import pandas as pd

# Load dataset
df = pd.read_csv("student_performance.csv")

print(df.head())

print("Dataset Info:")
print(df.info())

# Show missing values before cleaning
print("Missing Values Before Cleaning:")
print(df.isnull().sum())

# Fill missing values with mode
df['parent_education'] = df['parent_education'].fillna(df['parent_education'].mode()[0])

# Show missing values after cleaning
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())
