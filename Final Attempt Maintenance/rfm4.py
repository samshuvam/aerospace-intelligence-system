# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("aircraft_rating_dataset.csv")

# Check for missing values and handle them
if df.isnull().sum().any():
    print("\nHandling Missing Values...")
    df = df.fillna(df.median(numeric_only=True))  # Fill numeric NaNs with median
    df = df.fillna("Unknown")  # Fill categorical NaNs with "Unknown"

# Define the target column and non-feature columns
target_column = "Aircraft_Rating"  # Replace with your actual target column
non_feature_columns = ["Aircraft_ID"]  # Non-feature columns to drop

# Prepare features (X) and target (y)
X = df.drop(columns=[target_column] + non_feature_columns, errors="ignore")
y = df[target_column]

# Encode categorical columns
categorical_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()
if categorical_columns:
    print(f"\nEncoding Categorical Columns: {categorical_columns}")
    X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 1: Simplify the model
rf_model = RandomForestRegressor(
    n_estimators=150,  # Balanced number of trees
    max_depth=10,  # Limit tree depth
    min_samples_split=10,  # Require more samples to split
    min_samples_leaf=4,  # Require more samples per leaf
    max_features="sqrt",  # Consider a subset of features
    random_state=42
)

# Train the simplified model
print("\nTraining the Simplified Random Forest Model...")
rf_model.fit(X_train, y_train)

# Evaluate the simplified model
y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

print("\nSimplified Model Performance Metrics:")
print(f"Training R²: {r2_score(y_train, y_pred_train):.3f}")
print(f"Testing R²: {r2_score(y_test, y_pred_test):.3f}")
print(f"Testing RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.3f}")

# Step 2: Feature importance analysis and reduction
feature_importances = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nTop 10 Features by Importance:")
print(feature_importances.head(10))

# Retain only important features (threshold: Importance > 0.01)
important_features = feature_importances[feature_importances["Importance"] > 0.01]["Feature"]
X_train_reduced = X_train[important_features]
X_test_reduced = X_test[important_features]

# Train the model again with reduced features
rf_model.fit(X_train_reduced, y_train)
y_pred_test_reduced = rf_model.predict(X_test_reduced)

print("\nPerformance After Reducing Features:")
print(f"Testing R²: {r2_score(y_test, y_pred_test_reduced):.3f}")
print(f"Testing RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test_reduced)):.3f}")

# Step 3: Cross-validation
cv_scores = cross_val_score(rf_model, X_train_reduced, y_train, cv=5, scoring="r2")
print("\nCross-Validation R² Scores:")
print(cv_scores)
print(f"Mean CV R²: {np.mean(cv_scores):.3f}")

# Visualize feature importance
plt.figure(figsize=(12, 6))
sns.barplot(x="Importance", y="Feature", data=feature_importances, palette="viridis")
plt.title("Feature Importance in Simplified Random Forest Model")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()

# Visualize predicted vs actual ratings
plt.figure(figsize=(8, 8))
sns.scatterplot(x=y_test, y=y_pred_test_reduced, color="blue", alpha=0.7)
plt.title("Predicted vs Actual Aircraft Ratings (Reduced Features)")
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.axline((0, 0), slope=1, color="red", linestyle="--")  # Reference line
plt.grid()
plt.show()