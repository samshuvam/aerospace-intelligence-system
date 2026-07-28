# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("aircraft_rating_dataset.csv")

# Check for missing values and handle them
print("\nChecking Missing Values...")
if df.isnull().sum().any():
    print("Handling Missing Values...")
    df.fillna(df.median(numeric_only=True), inplace=True)  # Fill numeric NaNs with median
    df.fillna("Unknown", inplace=True)  # Fill categorical NaNs with "Unknown"

# Identify target and non-feature columns
target_column = "Aircraft_Rating"  # Replace with your actual target column
non_feature_columns = ["Aircraft_ID"]  # Replace or add irrelevant columns

# Features (X) and target (y)
X = df.drop(columns=[target_column] + non_feature_columns, errors="ignore")  # Drop target and non-feature columns
y = df[target_column]

# Detect categorical columns
categorical_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()

# Encode categorical columns
print(f"\nEncoding Categorical Columns: {categorical_columns}")
if categorical_columns:
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Optimize Random Forest Regressor using GridSearchCV
print("\nOptimizing Random Forest Model...")
param_grid = {
    "n_estimators": [200, 300, 500],
    "max_depth": [10, 20, 30],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", "log2"],
}
grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1, bootstrap=True, oob_score=True),
    param_grid=param_grid,
    cv=3,
    scoring="r2",
    verbose=1,
)
grid_search.fit(X_train, y_train)

# Use the best model from GridSearchCV
best_model = grid_search.best_estimator_
print("\nBest Model Parameters:")
print(grid_search.best_params_)

# Train the model on the training data
print("\nTraining Optimized Random Forest Model...")
best_model.fit(X_train, y_train)

# Make predictions
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

# Evaluate the model
print("\nModel Evaluation on Training Data:")
print(f"R-squared (R²): {r2_score(y_train, y_pred_train):.3f}")
print(f"Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(y_train, y_pred_train)):.3f}")

print("\nModel Evaluation on Test Data:")
print(f"R-squared (R²): {r2_score(y_test, y_pred_test):.3f}")
print(f"Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(y_test, y_pred_test)):.3f}")

# # Feature importance
# feature_importances = best_model.feature_importances_
# importance_df = pd.DataFrame({"Feature": X.columns, "Importance": feature_importances})
# importance_df = importance_df.sort_values(by="Importance", ascending=False)

# # Plot feature importances
# plt.figure(figsize=(12, 6))
# sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis")
# plt.title("Feature Importance in Optimized Random Forest Model")
# plt.xlabel("Importance")
# plt.ylabel("Features")
# plt.show()

# # Visualize predicted vs actual ratings
# plt.figure(figsize=(8, 8))
# sns.scatterplot(x=y_test, y=y_pred_test, color="blue", alpha=0.7)
# plt.title("Predicted vs Actual Aircraft Ratings (Optimized)")
# plt.xlabel("Actual Ratings")
# plt.ylabel("Predicted Ratings")
# plt.axline((0, 0), slope=1, color="red", linestyle="--")  # Reference line
# plt.grid()
# plt.show()

# Save the model and other necessary components
import joblib

# Save the trained model
joblib.dump(best_model, "aircraft_maintenance_model.pkl")

# Save the scaler (if applicable, based on your dataset)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()  # Apply scaling to match the Streamlit app input logic
scaler.fit(X)
joblib.dump(scaler, "scaler.pkl")

# Save feature ranges for UI scaling
feature_ranges = {col: (X[col].min(), X[col].max()) for col in X.columns}
joblib.dump(feature_ranges, "feature_ranges.pkl")

print("Model, scaler, and feature ranges saved successfully.")