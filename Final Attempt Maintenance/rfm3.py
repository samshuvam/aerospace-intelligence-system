import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("aircraft_rating_dataset.csv")

# Display dataset structure
print("Dataset Overview:")
print(df.head())

# Handle missing values
if df.isnull().sum().any():
    print("\nHandling Missing Values...")
    df.fillna(df.median(numeric_only=True), inplace=True)  # Numeric columns
    df.fillna("Unknown", inplace=True)  # Categorical columns

# Identify features (X) and target (y)
target_column = "Aircraft_Rating"  # Replace with the actual target column name
non_feature_columns = ["Aircraft_ID"]  # Exclude non-feature columns

X = df.drop(columns=[target_column] + non_feature_columns, errors="ignore")
y = df[target_column]

# Encode categorical variables
categorical_columns = X.select_dtypes(include=["object", "category"]).columns
if not categorical_columns.empty:
    print(f"Encoding Categorical Columns: {categorical_columns.tolist()}")
    X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

# Scale numeric features
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning using GridSearchCV
rf_model = RandomForestRegressor(random_state=42)
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
}

print("\nTuning Hyperparameters...")
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, scoring="r2", cv=3, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)
best_rf_model = grid_search.best_estimator_

# Evaluate the tuned model
y_pred_train = best_rf_model.predict(X_train)
y_pred_test = best_rf_model.predict(X_test)

print("\nModel Performance Metrics:")
print(f"Training R²: {r2_score(y_train, y_pred_train):.3f}")
print(f"Testing R²: {r2_score(y_test, y_pred_test):.3f}")
print(f"Testing RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.3f}")

# Visualize feature importance
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": best_rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis")
plt.title("Feature Importance in Random Forest Model")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()

# Visualize predicted vs. actual values
plt.figure(figsize=(8, 8))
sns.scatterplot(x=y_test, y=y_pred_test, color="blue", alpha=0.7)
plt.title("Predicted vs Actual Aircraft Ratings")
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.axline((0, 0), slope=1, color="red", linestyle="--")
plt.grid()
plt.show()