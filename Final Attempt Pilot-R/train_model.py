import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load the dataset
average_dataset = pd.read_csv("average_pilot_dataset.csv")

# Prepare features (X) and target variable (y)
X = average_dataset.drop(columns=["Pilot ID", "Performance Level"])  # Features
y = average_dataset["Performance Score"]  # Target variable

# Record the original feature ranges for reference
feature_ranges = {col: (X[col].min(), X[col].max()) for col in X.columns}

# Scale features to 0-100 using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 100))
X_scaled = scaler.fit_transform(X)

# Scale the target variable (Performance Score)
target_scaler = MinMaxScaler(feature_range=(0, 100))
y_scaled = target_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42
)

# Train the model
rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Model Performance Metrics:")
print(f"R-squared (R²): {r2:.3f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")

# Save the model, scaler, and feature ranges
joblib.dump(rf_model, "scaled_random_forest_pilot_rating_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(target_scaler, "target_scaler.pkl")
joblib.dump(feature_ranges, "feature_ranges.pkl")
print("Model, scalers, and feature ranges have been saved.")