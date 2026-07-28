import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
average_dataset = pd.read_csv("average_pilot_dataset.csv")

# Prepare the features (X) and target variable (y)
# Exclude non-numeric columns like 'Performance Level' and 'Pilot ID' for training
X = average_dataset.drop(columns=["Pilot ID", "Performance Level"])
y = average_dataset["Performance Score"]  # Use 'Performance Score' as the target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor with conservative settings to reduce overfitting
rf_model = RandomForestRegressor(
    n_estimators=100,      # Number of trees in the forest
    max_depth=10,          # Limit the depth of each tree
    min_samples_split=5,   # Minimum samples required to split an internal node
    min_samples_leaf=2,    # Minimum samples required at each leaf node
    max_features='sqrt',   # Number of features to consider when looking for the best split
    random_state=42
)

# Train the model
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Model Performance Metrics:")
print(f"R-squared (R²): {r2:.3f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")


# Save the model for future use
import joblib
joblib.dump(rf_model, "random_forest_pilot_rating_model.pkl")

print("Random Forest Model has been saved as 'random_forest_pilot_rating_model.pkl'")