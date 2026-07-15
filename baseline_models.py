import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("Step 1: Loading clean data...")
# Hum apna saaf kiya hua data load kar rahe hain
df = pd.read_csv("clean_model_data.csv")

print("Step 2: Splitting Data into Features (X) and Target (y)...")
# Humein 'EnrollmentCount' predict karna hai, toh hum baaki sabko Features (X) bana denge
# Note: Hum 'CourseRevenue' ko bhi hata rahe hain kyunki real-world me future revenue pehle se nahi pata hota
X = df.drop(columns=['EnrollmentCount', 'CourseRevenue'])
y = df['EnrollmentCount']

# Data ko 80% Training aur 20% Testing me baant rahe hain
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")

print("\nStep 3: Training Baseline Models...")
# Dono models ko initialize kar rahe hain
lr_model = LinearRegression()
ridge_model = Ridge(alpha=1.0)

# Models ko training data par seekha rahe hain (fit kar rahe hain)
lr_model.fit(X_train, y_train)
ridge_model.fit(X_train, y_train)

print("\nStep 4: Making Predictions on Testing Data...")
# Ab test data par predictions nikal rahe hain
lr_predictions = lr_model.predict(X_test)
ridge_predictions = ridge_model.predict(X_test)

# ==========================================
# EVALUATION METRICS (As per Technical Doc)
# ==========================================
def evaluate_model(model_name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"--- {model_name} Performance ---")
    print(f"MAE (Average Error) : {mae:.2f}")
    print(f"RMSE (Large Errors) : {rmse:.2f}")
    print(f"R2 Score (Accuracy) : {r2:.4f}\n")

print("\nStep 5: Evaluating Models...")
evaluate_model("Linear Regression", y_test, lr_predictions)
evaluate_model("Ridge Regression", y_test, ridge_predictions)