import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_name = "Copy of EduPro Online Platform.xlsx"

print("Step 1: Loading data for EDA...")
# Hum sirf 3 main sheets load kar rahe hain jo analysis ke liye zaroori hain
courses = pd.read_excel(file_name, sheet_name='Courses')
teachers = pd.read_excel(file_name, sheet_name='Teachers')
transactions = pd.read_excel(file_name, sheet_name='Transactions')

print("Step 2: Merging Datasets...")
# Transactions table me Course ki details jod rahe hain (CourseID ke basis par)
df_merged = pd.merge(transactions, courses, on='CourseID', how='left')

# Ab usme Teacher ki details bhi jod rahe hain (TeacherID ke basis par)
df_merged = pd.merge(df_merged, teachers, on='TeacherID', how='left')

print("\n--- Data Merging Successful ---")
print(f"Master Dataset ban gaya hai! Total Rows (Transactions): {df_merged.shape[0]}, Total Columns: {df_merged.shape[1]}")

print("\nStep 3: Checking for Missing Values...")
# Yeh batayega ki kis column me kitni values missing (khali) hain
missing_values = df_merged.isnull().sum()
print(missing_values[missing_values > 0]) # Sirf wahi columns dikhayega jisme missing values hain

print("\nStep 4: Generating Visualization...")
# Hum dekhenge ki kis category me sabse zyada enrollments hain
plt.figure(figsize=(10, 6))
# Seaborn ka countplot har category ke transactions count karega
sns.countplot(data=df_merged, y='CourseCategory', order=df_merged['CourseCategory'].value_counts().index, palette='viridis')

plt.title('Total Enrollments per Course Category (Demand Analysis)')
plt.xlabel('Number of Enrollments')
plt.ylabel('Course Category')
plt.tight_layout()

# Yeh line graph ko screen par display karegi
plt.show()