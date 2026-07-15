import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

file_name = "Copy of EduPro Online Platform.xlsx"

print("Step 1: Loading raw data...")
courses = pd.read_excel(file_name, sheet_name='Courses')
teachers = pd.read_excel(file_name, sheet_name='Teachers')
transactions = pd.read_excel(file_name, sheet_name='Transactions')

# ==========================================
# REQUIREMENT 1: Aggregate data at course level
# ==========================================
print("Step 2: Aggregating transaction data (Creating Target Variables)...")
# Hum har course ka total enrollment (count) aur total revenue (sum) nikal rahe hain
course_targets = transactions.groupby('CourseID').agg(
    EnrollmentCount=('TransactionID', 'count'),
    CourseRevenue=('Amount', 'sum')
).reset_index()

# Har course ko kaunsa teacher padha raha hai (mode nikal rahe hain)
course_teacher = transactions.groupby('CourseID')['TeacherID'].agg(lambda x: x.mode()[0]).reset_index()

# Ab in sabko main Courses table me merge kar denge
df_model = pd.merge(courses, course_targets, on='CourseID', how='left')
df_model = pd.merge(df_model, course_teacher, on='CourseID', how='left')
df_model = pd.merge(df_model, teachers, on='TeacherID', how='left')

# ==========================================
# REQUIREMENT 2: Handle missing ratings
# ==========================================
print("Step 3: Handling missing values...")
# Agar kisi course ka enrollment zero hai (left join ki wajah se), to NaN ko 0 kar do
df_model['EnrollmentCount'] = df_model['EnrollmentCount'].fillna(0)
df_model['CourseRevenue'] = df_model['CourseRevenue'].fillna(0)

# Teacher ki missing ratings aur experience ko median (middle value) se fill karenge
df_model['TeacherRating'] = df_model['TeacherRating'].fillna(df_model['TeacherRating'].median())
df_model['YearsOfExperience'] = df_model['YearsOfExperience'].fillna(df_model['YearsOfExperience'].median())

# ==========================================
# REQUIREMENT 3 & 4: Remove Redundant Features
# ==========================================
print("Step 4: Removing redundant/text features...")
# ML model ke liye IDs aur Names ki zaroorat nahi hai
columns_to_drop = ['CourseID', 'CourseName', 'TeacherID', 'TeacherName', 'Email']
# Jo columns dataset me hain, sirf unhi ko drop karenge
existing_drops = [col for col in columns_to_drop if col in df_model.columns]
df_model = df_model.drop(columns=existing_drops)

# ==========================================
# REQUIREMENT 5: Encode Categorical Variables
# ==========================================
print("Step 5: Encoding categorical variables...")
# Text columns ko numbers me badalna (One-Hot / Label Encoding)
categorical_cols = ['CourseCategory', 'CourseType', 'CourseLevel', 'Gender', 'Expertise']
le = LabelEncoder()
for col in categorical_cols:
    if col in df_model.columns:
        df_model[col] = le.fit_transform(df_model[col].astype(str))

# ==========================================
# REQUIREMENT 6: Normalize numerical features
# ==========================================
print("Step 6: Normalizing numerical features...")
numerical_cols = ['CoursePrice', 'CourseDuration', 'CourseRating', 'Age', 'YearsOfExperience', 'TeacherRating']
scaler = StandardScaler()
for col in numerical_cols:
    if col in df_model.columns:
        df_model[col] = scaler.fit_transform(df_model[[col]])

# Final check
print("\n--- Preprocessing Complete ---")
print(f"Final Data Shape for Machine Learning: {df_model.shape[0]} rows, {df_model.shape[1]} columns")

# Is clean data ko save kar lete hain taki models me direct use kar sakein
df_model.to_csv("clean_model_data.csv", index=False)
print("Saved clean data as 'clean_model_data.csv' in your folder!")