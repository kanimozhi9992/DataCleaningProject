import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("student_performance.csv")

# Show first 5 rows
print(df.head())

# Dataset information
print("\nDataset Info:")
print(df.info())

# Statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Study hours vs final score
sns.scatterplot(
    x='study_hours_per_week',
    y='final_score',
    data=df
)

plt.title("Study Hours vs Final Score")

plt.show()
