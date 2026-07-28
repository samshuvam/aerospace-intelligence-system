import pandas as pd
import json
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, LineString

# Load the dataset
df = pd.read_csv("synthetic_optimal_route_dataset.csv")

# Convert JSON fields to usable Python objects
df["Waypoints (List)"] = df["Waypoints (List)"].apply(json.loads)
df["Traffic (Intersecting Flights)"] = df["Traffic (Intersecting Flights)"].apply(json.loads)

# Initialize an empty GeoDataFrame for visualization with geopandas
gdf = gpd.GeoDataFrame(columns=["Flight ID", "geometry"])

# Parse waypoints and add them as geometries
for index, row in df.iterrows():
    waypoints = row["Waypoints (List)"]
    
    # Convert waypoints to Point objects and create a LineString for visualization
    points = [Point(wp["long"], wp["lat"]) for wp in waypoints]
    flight_path = LineString(points)
    
    # Add the flight path to the GeoDataFrame
    gdf = gdf.append({"Flight ID": row["Flight ID"], "geometry": flight_path}, ignore_index=True)

# Set a coordinate reference system for plotting
gdf.set_crs("EPSG:4326", inplace=True)  # WGS84 latitude/longitude

# Plot the flight paths on a simple world map
world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
fig, ax = plt.subplots(figsize=(12, 8))
world.plot(ax=ax, color="lightgray")
gdf.plot(ax=ax, color="blue", linewidth=1, alpha=0.7)
plt.title("Flight Paths of Synthetic Dataset")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# Visualize intersecting traffic at waypoints
# Let's pick a random flight to analyze intersecting flights
sample_flight_id = df["Flight ID"].iloc[0]
sample_flight_traffic = df[df["Flight ID"] == sample_flight_id]["Traffic (Intersecting Flights)"].values[0]

print(f"Intersecting Flights for Flight ID {sample_flight_id}:\n")
for intersect in sample_flight_traffic:
    print(f"Intersecting Flight ID: {intersect['flight_id']}, Altitude: {intersect['alt']} ft, "
          f"Time: {intersect['time']}, Waypoint: {intersect['wp']}")

# Visualize tailwind, headwind, and other metrics
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
df["Tailwind (knots)"].plot(kind="hist", bins=15, ax=axs[0], color="skyblue", edgecolor="black")
axs[0].set_title("Tailwind Distribution (knots)")
axs[0].set_xlabel("Tailwind (knots)")

df["Headwind (knots)"].plot(kind="hist", bins=15, ax=axs[1], color="salmon", edgecolor="black")
axs[1].set_title("Headwind Distribution (knots)")
axs[1].set_xlabel("Headwind (knots)")

plt.tight_layout()
plt.show()
