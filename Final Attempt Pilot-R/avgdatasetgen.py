import pandas as pd

# Load the pilot dataset
pilot_dataset = pd.read_csv("pilot_dataset.csv")

# Group by 'Pilot ID' and calculate the mean for numeric columns only
average_dataset = (
    pilot_dataset.groupby("Pilot ID")
    .mean(numeric_only=True)
    .reset_index()
)

# Include the 'Performance Level' as the most frequent category for each pilot
performance_level = (
    pilot_dataset.groupby("Pilot ID")["Performance Level"]
    .agg(lambda x: x.mode()[0])  # Get the most frequent value
    .reset_index()
)

# Merge numeric averages and performance level
average_dataset = average_dataset.merge(performance_level, on="Pilot ID")

# Save the averaged dataset to a new CSV file
average_dataset.to_csv("average_pilot_dataset.csv", index=False)

print("Averaged dataset generated and saved as 'average_pilot_dataset.csv'")