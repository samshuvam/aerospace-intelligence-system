import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of data entries
num_entries = 1000

# Generate synthetic data
data = {
    # Unique Aircraft ID
    "Aircraft_ID": [f"AC-{str(i).zfill(4)}" for i in range(1, num_entries + 1)],

    # Aircraft age in years (range: 1-30)
    "Aircraft_Age": np.random.randint(1, 31, num_entries),

    # Total flight hours (5000 to 50,000 hours, random variation)
    "Total_Flight_Hours": np.random.normal(25000, 10000, num_entries).clip(5000, 50000),

    # Number of flight cycles (1,000 to 20,000)
    "Total_Cycles": np.random.randint(1000, 20001, num_entries),

    # Days since last maintenance (1 to 300)
    "Days_Since_Last_Maintenance": np.random.randint(1, 301, num_entries),

    # Type of last maintenance performed
    "Last_Maintenance_Type": np.random.choice(
        ["Routine Check", "Component Replacement", "Major Overhaul", "Inspection"], num_entries
    ),

    # Number of unresolved logged issues
    "Unresolved_Issues": np.random.randint(0, 11, num_entries),

    # Total logged issues in the aircraft's lifetime
    "Total_Logged_Issues": np.random.randint(0, 50, num_entries),

    # Pressurization cycles (500 to 10,000 cycles)
    "Pressurization_Cycles": np.random.randint(500, 10001, num_entries),

    # Wear and tear score (scale of 0.1 to 1.0, uniform distribution)
    "Wear_Tear_Score": np.random.uniform(0.1, 1.0, num_entries),

    # Environmental exposure level (1 to 5)
    "Environmental_Exposure": np.random.randint(1, 6, num_entries),

    # Maintenance priority (Low, Medium, High)
    "Maintenance_Priority": np.random.choice(["Low", "Medium", "High"], num_entries, p=[0.5, 0.3, 0.2]),

    # Engine type
    "Engine_Type": np.random.choice(["Turbofan", "Turboprop", "Piston"], num_entries, p=[0.6, 0.3, 0.1]),

    # Structural integrity score (scale of 0.5 to 1.0)
    "Structural_Integrity_Score": np.random.uniform(0.5, 1.0, num_entries),

    # Operational region
    "Operational_Region": np.random.choice(
        ["Asia", "Europe", "North America", "South America", "Africa", "Oceania"], num_entries
    ),
}

# Create a base DataFrame
df = pd.DataFrame(data)

# Add correlated columns with slight variations to mimic real-world relationships
df["Maintenance_Cost_Per_Hour"] = (
    1000 + (50 * df["Aircraft_Age"]) + np.random.randint(-500, 500, num_entries)
).clip(1000, 5000)  # Maintenance cost in $ per hour
df["Overall_Aircraft_Usage"] = (
    df["Total_Flight_Hours"] / df["Total_Cycles"] + np.random.uniform(0.8, 1.2, num_entries)
).round(2)  # Average flight hours per cycle

# Add target variable: Aircraft Rating (scale: 1 to 10)
# Combine multiple factors to generate a weighted score
df["Aircraft_Rating"] = (
    (1 - df["Wear_Tear_Score"]) * 3  # Wear and tear inversely affects rating
    + df["Structural_Integrity_Score"] * 3  # Structural integrity is a major factor
    - (df["Unresolved_Issues"] / 10)  # More unresolved issues lower the rating
    - (df["Days_Since_Last_Maintenance"] / 100)  # Longer time since maintenance reduces rating
    + np.random.uniform(-0.5, 0.5, num_entries)  # Random noise for variation
).clip(1, 10).round(2)  # Rating scale from 1 (poor) to 10 (excellent)

# Display the dataset
print(df.head())

# Save to CSV
df.to_csv("aircraft_rating_dataset.csv", index=False)
print("Aircraft rating dataset generated and saved as 'aircraft_rating_dataset.csv'.")