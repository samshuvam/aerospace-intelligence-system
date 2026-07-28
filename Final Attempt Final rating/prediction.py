from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd

# Load the synthetic flight dataset
flight_df = pd.read_csv('synthetic_flight_data.csv')

# Step 1: Randomly assign pilot ratings and aircraft ratings
np.random.seed(42)
pilot_rating_choices = ['Poor', 'Good', 'Excellent']
aircraft_rating_choices = ['Poor', 'Good', 'Excellent']

# Add random ratings for pilots and aircraft
flight_df['Pilot Rating'] = np.random.choice(pilot_rating_choices, size=len(flight_df))
flight_df['Aircraft Rating'] = np.random.choice(aircraft_rating_choices, size=len(flight_df))

# Map ratings to numerical values
rating_map = {'Poor': 0, 'Good': 1, 'Excellent': 2}
flight_df['Pilot Rating'] = flight_df['Pilot Rating'].map(rating_map)
flight_df['Aircraft Rating'] = flight_df['Aircraft Rating'].map(rating_map)

# Create hazard categories with more randomness
def hazard_category(row):
    random_factor = np.random.rand()
    if row['Weather Temperature (°C)'] > 35 and row['Weather Wind Speed (km/h)'] > 50 and random_factor > 0.7:
        return 3  # Critical
    elif row['Pilot Rating'] == 0 or row['Aircraft Rating'] == 0 and random_factor > 0.5:
        return 2  # Stress
    elif row['Weather Precipitation (mm)'] > 20 and random_factor > 0.3:
        return 1  # Turbulence
    else:
        return 0  # No Hazard

# Apply the hazard categorization logic
flight_df['Hazard Category'] = flight_df.apply(hazard_category, axis=1)

# Step 2: Select features for the model
features = ['Duration (min)', 'Weather Temperature (°C)', 'Weather Wind Speed (km/h)', 'Weather Precipitation (mm)',
            'Aircraft Age (years)', 'Aircraft Maintenance Score', 'Pilot Experience (years)', 
            'Pilot Fatigue Level (1-10)', 'Fuel Consumption Rate (kg/min)', 'Landing Rate (ft/min)', 
            'Pilot Rating', 'Aircraft Rating']

# Target variable
target = 'Hazard Category'

# Prepare the data
X = flight_df[features]
y = flight_df[target]

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a Random Forest model with regularization
rf_classifier = RandomForestClassifier(
    n_estimators=50,  # Fewer trees
    max_depth=10,  # Limit tree depth
    max_features='sqrt',  # Limit features considered for splitting
    random_state=42
)
rf_classifier.fit(X_train, y_train)

# Step 5: Cross-Validation to Check Performance
cv_scores = cross_val_score(rf_classifier, X_train, y_train, cv=5)
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Average CV Score: {np.mean(cv_scores)}")

# Step 6: Make predictions
y_pred = rf_classifier.predict(X_test)

# Step 7: Evaluate the model
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save the updated dataset and model
flight_df.to_csv('updated_flight_data_with_hazard.csv', index=False)
import joblib
joblib.dump(rf_classifier, 'flight_hazard_classifier.pkl')