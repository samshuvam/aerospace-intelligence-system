import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Define parameters for synthetic dataset
num_waypoints = 10  # Number of waypoints to simulate
num_aircraft = 50  # Number of unique aircraft to simulate
days_of_data = 365  # Simulate data for one year

# Generate random waypoint coordinates (latitude and longitude) and names
waypoints = {
    f'WP{str(i).zfill(3)}': {
        'latitude': round(random.uniform(-90, 90), 6),
        'longitude': round(random.uniform(-180, 180), 6)
    }
    for i in range(num_waypoints)
}

# Generate a list of random aircraft ICAO24 identifiers
aircraft_ids = [f'ICAO24_{str(i).zfill(5)}' for i in range(num_aircraft)]

# Generate synthetic flight data
data = []

for day in range(days_of_data):
    date = datetime.now() - timedelta(days=day)
    num_flights = random.randint(5, 20)  # Number of flights per day

    for _ in range(num_flights):
        aircraft_id = random.choice(aircraft_ids)
        waypoint_id, waypoint = random.choice(list(waypoints.items()))
        
        # Simulate passing data
        timestamp = date + timedelta(hours=random.randint(0, 23), minutes=random.randint(0, 59))
        altitude = random.randint(3000, 40000)  # in feet
        speed = random.randint(200, 600)  # in knots
        distance_to_waypoint = round(random.uniform(0, 1), 3)  # in nautical miles

        data.append({
            'aircraft_id': aircraft_id,
            'waypoint': waypoint_id,
            'latitude': waypoint['latitude'],
            'longitude': waypoint['longitude'],
            'timestamp': timestamp,
            'altitude': altitude,
            'speed': speed,
            'distance_to_waypoint': distance_to_waypoint
        })

# Convert data to DataFrame
df_synthetic = pd.DataFrame(data)

# Display the synthetic dataset
df_synthetic.head(), df_synthetic.info()
