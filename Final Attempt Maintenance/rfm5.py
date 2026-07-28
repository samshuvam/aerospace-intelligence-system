# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (ensure your dataset file is in the same directory or provide the correct path)
df = pd.read_csv("aircraft_rating_dataset.csv")

# Display the first few rows to confirm the structure
print("Dataset Overview:")
print(df.head())

# Check for missing values
if df.isnull().sum().any():
    print("\nHandling Missing Values...")
    df = df.fillna(df.median(numeric_only=True))  # Fill numeric NaNs with median
    df = df.fillna("Unknown")  # Fill categorical NaNs with "Unknown"

# Identify the target column and drop non-feature columns
target_column = "Aircraft_Rating"  # Replace with your actual target column name
non_feature_columns = ["Aircraft_ID"]  # Columns not to be used as features

# Features (X) and Target (y)
X = df.drop(columns=[target_column] + non_feature_columns, errors="ignore")  # Drop target & non-feature columns
y = df[target_column]

# Detect categorical columns (object or category types)
categorical_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()

# Encode categorical columns
if categorical_columns:
    print(f"\nEncoding Categorical Columns: {categorical_columns}")
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
print("\nTraining the Random Forest Model...")
rf_model.fit(X_train, y_train)

# Make predictions
y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

# Evaluate the model
print("\nModel Evaluation on Training Data:")
print(f"Mean Squared Error: {mean_squared_error(y_train, y_pred_train):.2f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_train, y_pred_train):.2f}")
print(f"R-squared Score: {r2_score(y_train, y_pred_train):.2f}")

print("\nModel Evaluation on Test Data:")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_test):.2f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred_test):.2f}")
print(f"R-squared Score: {r2_score(y_test, y_pred_test):.2f}")

# Feature importance analysis
feature_importances = rf_model.feature_importances_
importance_df = pd.DataFrame({"Feature": X.columns, "Importance": feature_importances})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

# Plot feature importances
plt.figure(figsize=(12, 6))
sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis")
plt.title("Feature Importance in Random Forest Model")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()

# Visualize predicted vs actual ratings
plt.figure(figsize=(8, 8))
sns.scatterplot(x=y_test, y=y_pred_test, color="blue", alpha=0.7)
plt.title("Predicted vs Actual Aircraft Ratings")
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.axline((0, 0), slope=1, color="red", linestyle="--")  # Reference line
plt.grid()
plt.show()