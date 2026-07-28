import pandas as pd
import numpy as np

# Define the number of pilots
num_pilots = 6000

# Generate random data for each feature, scaled to 100
np.random.seed(42)  # For reproducibility
flight_hours = np.random.randint(0, 101, num_pilots)
training_scores = np.random.randint(0, 101, num_pilots)
incident_reports = np.random.randint(0, 101, num_pilots)
flight_maneuver_proficiency = np.random.randint(0, 101, num_pilots)
physiological_data = np.random.randint(0, 101, num_pilots)
cognitive_performance = np.random.randint(0, 101, num_pilots)
stress_levels = np.random.randint(0, 101, num_pilots)
communication_skills = np.random.randint(0, 101, num_pilots)
weather_adaptability = np.random.randint(0, 101, num_pilots)
peer_reviews = np.random.randint(0, 101, num_pilots)
simulator_performance = np.random.randint(0, 101, num_pilots)
fatigue_management = np.random.randint(0, 101, num_pilots)

# Calculate the final rating based on the features
final_rating = (
    0.2 * flight_hours +
    0.2 * training_scores +
    0.1 * (100 - incident_reports) +  # Lower incident reports should increase the rating
    0.1 * flight_maneuver_proficiency +
    0.1 * physiological_data +
    0.1 * cognitive_performance +
    0.05 * (100 - stress_levels) +  # Lower stress levels should increase the rating
    0.05 * communication_skills +
    0.05 * weather_adaptability +
    0.05 * peer_reviews +
    0.05 * simulator_performance +
    0.05 * fatigue_management
)

# Ensure the final rating is between 0 and 100
final_rating = np.clip(final_rating, 0, 100)

# Create the DataFrame
data = {
    'Pilot_ID': np.arange(1, num_pilots + 1),
    'Flight_Hours': flight_hours,
    'Training_Scores': training_scores,
    'Incident_Reports': incident_reports,
    'Flight_Maneuver_Proficiency': flight_maneuver_proficiency,
    'Physiological_Data': physiological_data,
    'Cognitive_Performance': cognitive_performance,
    'Stress_Levels': stress_levels,
    'Communication_Skills': communication_skills,
    'Weather_Adaptability': weather_adaptability,
    'Peer_Reviews': peer_reviews,
    'Simulator_Performance': simulator_performance,
    'Fatigue_Management': fatigue_management,
    'Final_Rating': final_rating
}

df = pd.DataFrame(data)

# Display the first few rows of the dataset
print(df.head())

# Save the dataset to a CSV file
df.to_csv('realistic_pilot_ratings.csv', index=False)
