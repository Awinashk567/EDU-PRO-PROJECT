import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("Step 1: Loading clean data...")
df = pd.read_csv("clean_model_data.csv")

# Setting up Features (X) and Target (y)
X = df.drop(columns=['EnrollmentCount', 'CourseRevenue'])
y = df['EnrollmentCount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Step 2: Training Advanced Models (This might take a few seconds)...")
# Initializing Advanced Models
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)

# Training the models
rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)

# Making predictions
rf_predictions = rf_model.predict(X_test)
gb_predictions = gb_model.predict(X_test)

# Evaluation Function
def evaluate_model(model_name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"--- {model_name} Performance ---")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R2   : {r2:.4f}\n")

print("\nStep 3: Evaluating Advanced Models...")
evaluate_model("Random Forest", y_test, rf_predictions)
evaluate_model("Gradient Boosting", y_test, gb_predictions)

# ==========================================
# FEATURE IMPORTANCE ANALYSIS (Tech Doc Requirement)
# ==========================================
print("Step 4: Generating Feature Importance Graph...")

# Random Forest se feature importance nikal rahe hain
feature_importances = rf_model.feature_importances_

# DataFrame bana rahe hain taaki graph plot kar sakein
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Plotting the graph
plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x='Importance', y='Feature', palette='magma')
plt.title('Feature Importance: What drives Course Demand?')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()

# Graph ko save karna ek good practice hai report ke liye
plt.savefig('feature_importance.png')
print("Graph saved as 'feature_importance.png' in your folder!")

plt.show()