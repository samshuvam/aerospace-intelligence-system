import pandas as pd
import numpy as np
import random
import os

# Load original features from the dataset
original_data = pd.read_csv("realistic_pilot_ratings.csv")

# Directory to store individual pilot datasets
output_dir = "pilot_datasets"
os.makedirs(output_dir, exist_ok=True)

# Function to create synthetic pilot-specific data with original and new features
def generate_pilot_data(pilot_id, original_features, num_records=100):
    data = []
    for _ in range(num_records):
        entry = {
            "PilotID": pilot_id,
            # Existing features from original dataset
            **{feature: np.random.choice(original_features[feature].dropna()) for feature in original_features.columns},
            # New Features
            "HoursFlownPerYear": np.random.normal(loc=800, scale=50),
            "StressManagementScore": np.random.randint(6, 10),
            "Certifications": random.choice(["IFR", "Night Flight", "Multi-Engine"]),
            "AircraftTypeExperience": random.choice(["Boeing 737", "Airbus A320", "Cessna 172"]),
            "WeatherAdaptabilityScore": np.random.randint(5, 10),
            "AccidentFreeYears": np.random.randint(1, 10),
            "TrainingHoursLastYear": np.random.normal(loc=30, scale=5),
            "FlightReviewScore": np.random.uniform(7, 10)
        }
        data.append(entry)
    return pd.DataFrame(data)

# Generate and save dataset for each pilot
for pilot_id in range(1, 101):
    pilot_data = generate_pilot_data(pilot_id, original_data)
    pilot_data.to_csv(f"{output_dir}/pilot_{pilot_id}_data.csv", index=False)
