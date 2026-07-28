import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('synthetic_pilot_ratings.csv')

# Preprocess the data
X = df.drop(columns=['Final_Rating', 'Pilot_ID'])
y = df['Final_Rating']

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best model from GridSearchCV
best_model = grid_search.best_estimator_

# Evaluate the model
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Function to input new pilot data and predict the rating
def rate_new_pilot():
    print("Enter the following details for the new pilot (values out of 100):")
    flight_hours = float(input("Flight Hours: "))
    training_scores = float(input("Training Scores: "))
    incident_reports = float(input("Incident Reports: "))
    flight_maneuver_proficiency = float(input("Flight Maneuver Proficiency: "))
    physiological_data = float(input("Physiological Data: "))
    cognitive_performance = float(input("Cognitive Performance: "))
    stress_levels = float(input("Stress Levels: "))
    communication_skills = float(input("Communication Skills: "))
    weather_adaptability = float(input("Weather Adaptability: "))
    peer_reviews = float(input("Peer Reviews: "))
    simulator_performance = float(input("Simulator Performance: "))
    fatigue_management = float(input("Fatigue Management: "))

    new_pilot_data = {
        'Flight_Hours': [flight_hours],
        'Training_Scores': [training_scores],
        'Incident_Reports': [incident_reports],
        'Flight_Maneuver_Proficiency': [flight_maneuver_proficiency],
        'Physiological_Data': [physiological_data],
        'Cognitive_Performance': [cognitive_performance],
        'Stress_Levels': [stress_levels],
        'Communication_Skills': [communication_skills],
        'Weather_Adaptability': [weather_adaptability],
        'Peer_Reviews': [peer_reviews],
        'Simulator_Performance': [simulator_performance],
        'Fatigue_Management': [fatigue_management]
    }

    new_pilot_df = pd.DataFrame(new_pilot_data)
    new_pilot_df_scaled = scaler.transform(new_pilot_df)
    new_pilot_rating = best_model.predict(new_pilot_df_scaled)
    print(f'Predicted Final Rating for New Pilot: {new_pilot_rating[0]}')

# Call the function to rate a new pilot
rate_new_pilot()
