import pandas as pd
import numpy as np

# Function to generate a random pilot dataset
def generate_pilot_data(num_pilots=100, entries_per_pilot=100):
    np.random.seed(42)  # For reproducibility

    # Define the performance levels and their distribution
    performance_levels = {
        "Excellent": 10,
        "Very Good": 25,
        "Good": 20,
        "OK": 15,
        "Needs Improvement": 20,
        "Unacceptable": 10
    }
    levels = []
    for level, count in performance_levels.items():
        levels.extend([level] * count)

    # Shuffle the levels to randomize assignment
    np.random.shuffle(levels)

    # Features to generate
    feature_ranges = {
        "Training Hours": (300, 1500),
        "Simulator Hours": (100, 800),
        "Flight Experience (Years)": (1, 40),
        "Training Score": (60, 100),
        "Performance Score": (0, 100),
        "Incidents": (0, 5),
        "Accidents": (0, 2),
        "Aircraft Types Flown": (1, 10),
        "Last Check Ride Score": (50, 100),
        "Reaction Time (ms)": (200, 500),
        "Stress Level": (1, 10),  # 1=Low stress, 10=High stress
        "Sleep Quality": (1, 10),  # 1=Poor, 10=Excellent
        "Cognitive Flexibility Score": (50, 100),
        "Psychological Condition Index": (50, 100)  # Higher is better
    }

    # Generate variability for each performance level
    def adjust_feature_by_performance(base_value, level):
        # Scale the features based on performance level
        if level == "Excellent":
            return base_value * np.random.uniform(1.1, 1.3)
        elif level == "Very Good":
            return base_value * np.random.uniform(1.0, 1.1)
        elif level == "Good":
            return base_value * np.random.uniform(0.9, 1.0)
        elif level == "OK":
            return base_value * np.random.uniform(0.8, 0.9)
        elif level == "Needs Improvement":
            return base_value * np.random.uniform(0.7, 0.8)
        elif level == "Unacceptable":
            return base_value * np.random.uniform(0.5, 0.7)
        else:
            return base_value

    # Create a DataFrame for all pilots
    all_pilots_data = []
    for pilot_id in range(1, num_pilots + 1):
        performance_level = levels[pilot_id - 1]
        for _ in range(entries_per_pilot):
            pilot_data = {
                "Pilot ID": f"P{pilot_id:03d}",
                "Performance Level": performance_level
            }
            for feature, (low, high) in feature_ranges.items():
                base_value = np.random.uniform(low, high)
                pilot_data[feature] = round(adjust_feature_by_performance(base_value, performance_level), 2)
            all_pilots_data.append(pilot_data)

    # Convert to DataFrame
    df = pd.DataFrame(all_pilots_data)
    return df

# Generate the dataset
num_pilots = 100
entries_per_pilot = 100
pilot_dataset = generate_pilot_data(num_pilots, entries_per_pilot)

# Save to CSV for easy access
pilot_dataset.to_csv("pilot_dataset.csv", index=False)

print("Dataset generated and saved as 'pilot_dataset.csv'")
