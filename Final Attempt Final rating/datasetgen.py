import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Generate synthetic data
def generate_synthetic_data(num_records):
    flight_ids = [i for i in range(1, num_records + 1)]
    aircraft_ids = [random.randint(1, 50) for _ in range(num_records)]
    pilot_ids = [f"{random.randint(1, 50)},{random.randint(1, 50)}" for _ in range(num_records)]
    routes = ["Vijayawada to Hyderabad"] * num_records
    flight_dates = [datetime.now() - timedelta(days=random.randint(1, 365)) for _ in range(num_records)]
    flight_durations = [random.randint(120, 180) for _ in range(num_records)]
    weather_temp = [round(random.uniform(25, 40), 2) for _ in range(num_records)]
    weather_wind_speed = [random.randint(30, 80) for _ in range(num_records)]
    weather_precipitation = [round(random.uniform(0, 5), 2) for _ in range(num_records)]
    aircraft_ages = [random.randint(5, 30) for _ in range(num_records)]
    maintenance_scores = [random.randint(1, 10) for _ in range(num_records)]
    pilot_experience = [random.randint(5, 25) for _ in range(num_records)]
    pilot_fatigue = [random.randint(1, 10) for _ in range(num_records)]
    fuel_consumption_rate = [round(random.uniform(0.6, 1.2), 2) for _ in range(num_records)]
    landing_rate = [random.randint(150, 400) for _ in range(num_records)]
    flight_safety_rating = ["Critical" if random.random() < 0.2 else "Safe" for _ in range(num_records)]

    # Create the DataFrame
    df = pd.DataFrame({
        'Flight ID': flight_ids,
        'Aircraft ID': aircraft_ids,
        'Pilot IDs': pilot_ids,
        'Flight Date/Time': flight_dates,
        'Route': routes,
        'Duration (min)': flight_durations,
        'Weather Temperature (°C)': weather_temp,
        'Weather Wind Speed (km/h)': weather_wind_speed,
        'Weather Precipitation (mm)': weather_precipitation,
        'Aircraft Age (years)': aircraft_ages,
        'Aircraft Maintenance Score': maintenance_scores,
        'Pilot Experience (years)': pilot_experience,
        'Pilot Fatigue Level (1-10)': pilot_fatigue,
        'Fuel Consumption Rate (kg/min)': fuel_consumption_rate,
        'Landing Rate (ft/min)': landing_rate,
        'Flight Safety Rating': flight_safety_rating
    })

    # Save to CSV
    df.to_csv("synthetic_flight_data.csv", index=False)
    print(f"Dataset of {num_records} rows has been saved to 'synthetic_flight_data.csv'")

# Generate and save 1000+ records
generate_synthetic_data(1000)
