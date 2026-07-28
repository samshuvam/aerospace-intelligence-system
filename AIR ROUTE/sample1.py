import pandas as pd
import random

# Reference data (based on provided inputs)
reference_data = {
    "Flight ID": ["6E7224", "IX2883", "6E2264", "6E7439", "6E7288", "6E7251", "6E7392", "6E5184", "AI599", 
                  "6E7704", "IX2516", "6E7206", "6E7139", "AI468", "6E7284"],
    "Destination": ["Bengaluru (BLR)", "Hyderabad (HYD)", "Delhi (DEL)", "Tirupati (TIR)", "Hyderabad (HYD)", 
                    "Visakhapatnam (VTZ)", "Hyderabad (HYD)", "Mumbai (BOM)", "Mumbai (BOM)", "Bengaluru (BLR)", 
                    "Visakhapatnam (VTZ)", "Hyderabad (HYD)", "Chennai (MAA)", "Delhi (DEL)", "Hyderabad (HYD)"],
    "Airline": ["IndiGo", "Air India Express", "SmartLynx", "IndiGo", "IndiGo", "IndiGo", "IndiGo", "IndiGo", 
                "Air India", "IndiGo", "Air India Express", "IndiGo", "IndiGo", "Air India", "IndiGo"],
    "Aircraft Type": ["AT76", "B38M", "A320", "AT76", "AT76", "AT76", "AT76", "A20N", "A20N", "AT76", "B38M", 
                      "AT76", "AT76", "A319", "AT76"],
    "Scheduled Departure": ["12:45 PM", "12:55 PM", "2:10 PM", "2:45 PM", "4:50 PM", "5:45 PM", "6:10 PM", 
                            "6:40 PM", "7:10 PM", "7:15 PM", "7:55 PM", "8:15 PM", "9:05 PM", "9:10 PM", "9:25 PM"],
    "Actual Departure": ["12:46 PM", "3:37 PM", "2:21 PM", "3:03 PM", "4:48 PM", "6:18 PM", "6:08 PM", 
                         "6:59 PM", "7:04 PM", "7:19 PM", "7:53 PM", "8:37 PM", "9:10 PM", "9:21 PM", "9:26 PM"]
}

# Convert the reference data into a DataFrame for ease of manipulation
ref_df = pd.DataFrame(reference_data)

# Function to generate synthetic data based on provided specifications
def generate_large_synthetic_data(df, num_rows=3000):
    synthetic_data = []
    for i in range(num_rows):
        base_idx = i % len(df)  # Cycle through reference data
        scheduled_departure = pd.to_datetime(df["Scheduled Departure"][base_idx], format="%I:%M %p")
        actual_departure = pd.to_datetime(df["Actual Departure"][base_idx], format="%I:%M %p")

        row = {
            "Flight ID": df["Flight ID"][base_idx],
            "Destination": df["Destination"][base_idx],
            "Airline": df["Airline"][base_idx],
            "Aircraft Type": df["Aircraft Type"][base_idx],
            "Scheduled Departure": scheduled_departure.strftime("%I:%M %p"),
            "Scheduled Arrival": (scheduled_departure + pd.Timedelta(minutes=random.randint(60, 120))).strftime("%I:%M %p"),
            "Actual Departure": actual_departure.strftime("%I:%M %p"),
            "Actual Arrival": (actual_departure + pd.Timedelta(minutes=random.randint(60, 120))).strftime("%I:%M %p"),
            "Traffic Volume (Nearby Flights ±5 mins)": f"{random.randint(1, 6)} flights scheduled within +/-5 mins",
            "Sector Congestion (Intersections)": f"{random.randint(7, 10)}:{random.randint(10, 59)} - Flight XYZ-ABC at {random.randint(15000, 30000)} ft",
            "Special Event (Rating out of 10)": random.choice([0, 5, 7, 10]),
            "Day of the Week": random.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]),
            "Time of the Day": random.choice(["Morning", "Afternoon", "Evening"]),
            "Weather Condition": random.choice(["Clear", "Fog", "Rain"]),
            "Airport Operating Hours (Origin)": "05:00-23:00",
            "Airport Operating Hours (Destination)": "05:00-23:00",
            "Actual Travel Distance (Time Taken in mins)": random.randint(45, 65),
            "Suggested Travel Distance (Optimal Time in mins)": random.randint(45, 55)
        }
        
        synthetic_data.append(row)
    
    return pd.DataFrame(synthetic_data)

# Generate the synthetic dataset
large_synthetic_df = generate_large_synthetic_data(ref_df, num_rows=3000)

# Save to CSV
output_path = "/mnt/data/synthetic_vijayawada_airport_dataset.csv"
large_synthetic_df.to_csv(output_path, index=False)

output_path
