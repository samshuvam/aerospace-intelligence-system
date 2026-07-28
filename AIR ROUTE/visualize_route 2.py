import pandas as pd
import json
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, LineString

# Load the dataset
df = pd.read_csv("synthetic_optimal_route_dataset.csv")

# Convert JSON fields to usable Python objects
df["Waypoints (List)"] = df["Waypoints (List)"].apply(lambda x: json.loads(x) if isinstance(x, str) else [])
df["Traffic (Intersecting Flights)"] = df["Traffic (Intersecting Flights)"].apply(lambda x: json.loads(x) if isinstance(x, str) else [])

# Initialize an empty list to store GeoDataFrame rows
gdf_list = []

# Process each flight's waypoints to create flight paths
for index, row in df.iterrows():
    waypoints = row["Waypoints (List)"]
    if waypoints:  # Only process if there are waypoints
        # Convert waypoints to Point objects and create a LineString for visualization
        points = [Point(wp["long"], wp["lat"]) for wp in waypoints]
        flight_path = LineString(points)
        
        # Append the row as a dictionary to the list
        gdf_list.append({"Flight ID": row["Flight ID"], "geometry": flight_path})

# Create a GeoDataFrame from the list
gdf = gpd.GeoDataFrame(gdf_list, crs="EPSG:4326")

# Load the world map shapefile (Update this path to where you saved the downloaded .shp file)
world = gpd.read_file("ne_110m_admin_0_countries.shp")

# Plot the flight paths on a world map
fig, ax = plt.subplots(figsize=(12, 8))
world.plot(ax=ax, color="lightgray")
gdf.plot(ax=ax, color="blue", linewidth=1, alpha=0.7)
plt.title("Flight Paths of Synthetic Dataset")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# Visualize intersecting traffic for a sample flight
sample_flight_id = df["Flight ID"].iloc[0]
sample_flight_traffic = df[df["Flight ID"] == sample_flight_id]["Traffic (Intersecting Flights)"].values[0]

print(f"Intersecting Flights for Flight ID {sample_flight_id}:\n")
for intersect in sample_flight_traffic:
    print(f"Intersecting Flight ID: {intersect['flight_id']}, Altitude: {intersect['alt']} ft, "
          f"Time: {intersect['time']}, Waypoint: {intersect['wp']}")

# Visualize wind data distributions
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
df["Tailwind (knots)"].plot(kind="hist", bins=15, ax=axs[0], color="skyblue", edgecolor="black")
axs[0].set_title("Tailwind Distribution (knots)")
axs[0].set_xlabel("Tailwind (knots)")

df["Headwind (knots)"].plot(kind="hist", bins=15, ax=axs[1], color="salmon", edgecolor="black")
axs[1].set_title("Headwind Distribution (knots)")
axs[1].set_xlabel("Headwind (knots)")

plt.tight_layout()
plt.show()