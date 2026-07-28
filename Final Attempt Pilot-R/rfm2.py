import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
average_pilot_dataset = pd.read_csv("average_pilot_dataset.csv")

# Separate features and target variable
X = average_pilot_dataset.drop(columns=["Performance Score"])
y = average_pilot_dataset["Performance Score"]

# Convert non-numeric columns to numeric using Label Encoding
for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Add Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Define Random Forest model
rf = RandomForestRegressor(random_state=42)

# Set up parameter grid for GridSearchCV
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

# Perform hyperparameter tuning
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring="r2", verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters from GridSearchCV
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Train the final model with the best parameters
best_rf = grid_search.best_estimator_
best_rf.fit(X_train, y_train)

# Predictions on test set
y_pred = best_rf.predict(X_test)

# Calculate metrics
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Final Model Performance:")
print(f"R-squared (R²): {r2}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Plot feature importances
importances = best_rf.feature_importances_
feature_names = poly.get_feature_names_out(X.columns)

# Plot top 10 important features
sorted_idx = np.argsort(importances)[::-1][:10]
plt.figure(figsize=(10, 6))
plt.barh([feature_names[i] for i in sorted_idx], importances[sorted_idx], color="skyblue")
plt.xlabel("Feature Importance")
plt.title("Top 10 Feature Importances")
plt.gca().invert_yaxis()
plt.show()
