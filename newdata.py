import pandas as pd
import numpy as np

# Define the number of pilots and the number of entries per pilot
num_pilots = 500
entries_per_pilot = 12  # This will give us 6000 rows (500 pilots * 12 entries each)

# Generate random data for each feature, scaled to 100
data = {
    'Pilot_ID': np.repeat(np.arange(1, num_pilots + 1), entries_per_pilot),
    'Flight_Hours': np.random.randint(0, 101, num_pilots * entries_per_pilot),
    'Training_Scores': np.random.randint(0, 101, num_pilots * entries_per_pilot),
    'Incident_Reports': np.random.randint(0, 101, num_pilots * entries_per_pilot),
    'Flight_Maneuver_Proficiency': np.random.randint(0, 101, num_pilots * entries_per_pilot),
    'Physiological_Data': np.random.randint(0, 101, num_pilots * entries_per_pilot),
    'Cognitive_Performance': np.random.randint(0, 101, num_pilots * entries_per_pilot),
    'Stress_Levels': np.random.randint(0, 101, num_pilots * entries_per_pilot),
    'Communication_Skills': np.random.randint(0, 101, num_pilots * entries_per_pilot),
    'Weather_Adaptability': np.random.randint(0, 101, num_pilots * entries_per_pilot),
    'Peer_Reviews': np.random.randint(0, 101, num_pilots * entries_per_pilot),
    'Simulator_Performance': np.random.randint(0, 101, num_pilots * entries_per_pilot),
    'Fatigue_Management': np.random.randint(0, 101, num_pilots * entries_per_pilot),
    'Final_Rating': np.random.randint(0, 101, num_pilots * entries_per_pilot)
}

# Create a DataFrame
df = pd.DataFrame(data)

# Display the first few rows of the dataset
print(df.head())

# Save the dataset to a CSV file
df.to_csv('synthetic_pilot_ratings.csv', index=False)
