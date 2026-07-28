# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the dataset
df = pd.read_csv("aircraft_rating_dataset.csv")

# Handle missing values
print("Checking for missing values...")
if df.isnull().sum().any():
    print("Handling missing values...")
    df.fillna(df.median(numeric_only=True), inplace=True)  # Numeric columns
    df.fillna("Unknown", inplace=True)  # Categorical columns

# Specify target and non-feature columns
target_column = "Aircraft_Rating"  # Update as per your dataset
non_feature_columns = ["Aircraft_ID"]  # Add irrelevant columns if necessary

# Prepare features (X) and target (y)
X = df.drop(columns=[target_column] + non_feature_columns, errors="ignore")
y = df[target_column]

# Encode categorical columns
categorical_columns = X.select_dtypes(include=["object", "category"]).columns
if categorical_columns.any():
    print(f"Encoding categorical columns: {categorical_columns}")
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le  # Save encoder for future use

# Scale features for consistency
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# GridSearchCV for hyperparameter tuning
print("Optimizing model using GridSearchCV...")
param_grid = {
    "n_estimators": [200, 300, 500],
    "max_depth": [10, 20, 30],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", "log2"],
}
grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring="r2",
    verbose=1,
    n_jobs=-1,
)
grid_search.fit(X_train, y_train)

# Select the best model
best_model = grid_search.best_estimator_
print(f"Best model parameters: {grid_search.best_params_}")

# Evaluate the model
def evaluate_model(model, X, y, dataset_name=""):
    predictions = model.predict(X)
    r2 = r2_score(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    print(f"{dataset_name} R²: {r2:.3f}, RMSE: {rmse:.3f}")

print("\nEvaluating on training data...")
evaluate_model(best_model, X_train, y_train, "Training")
print("\nEvaluating on test data...")
evaluate_model(best_model, X_test, y_test, "Test")

# Save the trained model, scaler, and feature metadata
joblib.dump(best_model, "aircraft_maintenance_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump({col: (X[col].min(), X[col].max()) for col in X.columns}, "feature_ranges.pkl")

print("Model, scaler, and feature ranges saved successfully.")