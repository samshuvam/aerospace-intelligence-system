import pandas as pd
import random
import json

# Function to generate synthetic data for Optimal Route Selection
def generate_optimal_route_data(num_rows=2000):
    synthetic_data = []
    
    for i in range(num_rows):
        # Randomly generate flight details
        flight_id = f"FL{i+1000}"
        origin = random.choice(["VGA", "HYD", "DEL", "BLR", "BOM"])
        destination = random.choice(["HYD", "VGA", "MAA", "DEL", "BLR", "BOM"])
        while destination == origin:  # Ensure origin and destination are different
            destination = random.choice(["HYD", "VGA", "MAA", "DEL", "BLR", "BOM"])

        # Generate random route distance
        route_distance = random.randint(150, 1500)  # in kilometers

        # Generate waypoints data
        waypoints = [{"lat": round(random.uniform(-90, 90), 4), "long": round(random.uniform(-180, 180), 4), 
                      "alt": random.choice([10000, 15000, 20000, 25000, 30000]), 
                      "time": f"{random.randint(0, 23)}:{str(random.randint(0, 59)).zfill(2)}"} for _ in range(random.randint(5, 15))]
        
        # Intersecting flights along the route
        traffic = [{"flight_id": f"INT{random.randint(1000, 9999)}", 
                    "alt": random.choice([10000, 15000, 20000, 25000]), 
                    "time": f"{random.randint(0, 23)}:{str(random.randint(0, 59)).zfill(2)}", 
                    "wp": random.choice(["WP1", "WP2", "WP3", "WP4"])} for _ in range(random.randint(1, 5))]
        
        # Wind details
        tailwind = random.randint(0, 30)  # in knots
        headwind = random.randint(0, 30)  # in knots
        
        # Jet stream impact
        jet_stream_impact = random.choice(["Positive", "Negative", "No Impact"])
        
        # Restrictions data
        restrictions = [{"type": random.choice(["Govt", "War Zone", "Noise Sensitive"]), 
                         "wp": f"WP{random.randint(1, 5)}"} for _ in range(random.randint(1, 3))]

        # Drones and Air Taxis
        drones_air_taxis = [{"alt": random.randint(100, 500), "time": f"{random.randint(0, 23)}:{str(random.randint(0, 59)).zfill(2)}"} 
                            for _ in range(random.randint(0, 3))]
        
        # Bird movement
        bird_movement = [{"zone": random.choice(["Near Airport", "City Area"]), 
                          "alt": [random.randint(1000, 3000), random.randint(3000, 6000)], 
                          "time_range": f"{random.randint(6, 10)}:{str(random.randint(0, 59)).zfill(2)}-{random.randint(18, 22)}:{str(random.randint(0, 59)).zfill(2)}"}]

        # Other relevant fields
        airspace_density = random.choice(["Low", "Medium", "High"])
        seasonal_pattern = random.choice(["Winter", "Summer", "Monsoon"])
        fuel_cost_segment = round(random.uniform(0.3, 0.8), 2)  # cost per segment in USD or currency unit
        historical_delay = random.randint(0, 30)  # in minutes

        row = {
            "Flight ID": flight_id,
            "Origin": origin,
            "Destination": destination,
            "Route Distance (km)": route_distance,
            "Waypoints (List)": json.dumps(waypoints),
            "Traffic (Intersecting Flights)": json.dumps(traffic),
            "Tailwind (knots)": tailwind,
            "Headwind (knots)": headwind,
            "Jet Stream Impact": jet_stream_impact,
            "Restrictions (Details)": json.dumps(restrictions),
            "Drones/Air Taxis": json.dumps(drones_air_taxis),
            "Bird Movement": json.dumps(bird_movement),
            "Airspace Density": airspace_density,
            "Seasonal Pattern": seasonal_pattern,
            "Fuel Cost (Segment)": fuel_cost_segment,
            "Historical Delay (mins)": historical_delay
        }
        
        synthetic_data.append(row)
    
    return pd.DataFrame(synthetic_data)

# Generate the synthetic dataset
optimal_route_df = generate_optimal_route_data(num_rows=2000)

# Save to CSV in the current directory
optimal_route_df.to_csv("synthetic_optimal_route_dataset.csv", index=False)


