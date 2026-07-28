import gpxpy

# Open the GPX file
with open('Users\elite\Downloads', 'r') as file:
    gpx = gpxpy.parse(file)

# Iterate through tracks and track points
for track in gpx.tracks:
    for segment in track.segments:
        for point in segment.points:
            print(f"Latitude: {point.latitude}, Longitude: {point.longitude}, Elevation: {point.elevation}, Time: {point.time}")