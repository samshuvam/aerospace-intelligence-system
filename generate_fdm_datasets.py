import pandas as pd
import numpy as np
import random

# Helper function to generate random data
def generate_random_data(size, category=False, values=None):
    if category and values:
        return [random.choice(values) for _ in range(size)]
    elif category:
        return [random.choice(['Yes', 'No']) for _ in range(size)]
    else:
        return [round(random.uniform(0, 10), 2) for _ in range(size)]

# General parameters for all datasets
size = 10000
aircraft_types = ['A320', 'B737', 'A380', 'B787', 'A350']
weather_conditions = ['Clear', 'Rain', 'Fog', 'Snow', 'Thunderstorm']

# 1. Dataset for Predicting Chances of Accidents
accident_data = pd.DataFrame({
    'FlightID': range(1, size+1),
    'AircraftType': generate_random_data(size, category=True, values=aircraft_types),
    'RouteAdherence': generate_random_data(size),
    'LandingQuality': generate_random_data(size),
    'PerformanceRating': generate_random_data(size),
    'ErrorsDetected': generate_random_data(size, category=True),
    'WeatherConditions': generate_random_data(size, category=True, values=weather_conditions),
    'AirTrafficControlIssues': generate_random_data(size, category=True),
    'PilotExperience': generate_random_data(size),
    'FlightDuration': generate_random_data(size),
    'AccidentOccurred': generate_random_data(size, category=True)
})

# 2. Dataset for Predicting Maintenance Requirements
maintenance_data = pd.DataFrame({
    'FlightID': range(1, size+1),
    'AircraftType': generate_random_data(size, category=True, values=aircraft_types),
    'EngineThrust': generate_random_data(size),
    'PressurizationCycles': generate_random_data(size),
    'ErrorsDetected': generate_random_data(size, category=True),
    'MaintenanceHistory': generate_random_data(size),
    'FlightHoursSinceLastCheck': generate_random_data(size),
    'MaintenanceRequired': generate_random_data(size, category=True)
})

# 3. Dataset for Optimized Flight Operations
operations_data = pd.DataFrame({
    'FlightID': range(1, size+1),
    'AircraftType': generate_random_data(size, category=True, values=aircraft_types),
    'FuelPlanned': generate_random_data(size),
    'FuelActual': generate_random_data(size),
    'RouteAdherence': generate_random_data(size),
    'PerformanceRating': generate_random_data(size),
    'GoAround': generate_random_data(size, category=True),
    'WeatherConditions': generate_random_data(size, category=True, values=weather_conditions),
    'PilotEfficiency': generate_random_data(size),
    'OptimizedOperation': generate_random_data(size, category=True)
})

# Save datasets to CSV files
accident_data.to_csv('accident_data.csv', index=False)
maintenance_data.to_csv('maintenance_data.csv', index=False)
operations_data.to_csv('operations_data.csv', index=False)

print("Datasets generated and saved to CSV files.")