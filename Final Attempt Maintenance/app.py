import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Define synthetic reference data (representing scaled training data for each feature)
synthetic_data = {
    "Aircraft_Age": [10, 20, 30, 50, 70, 90],  # Example synthetic values for Aircraft Age
    "Unresolved_Issues": [0, 1, 2, 3, 4, 5],  # Example synthetic values for Unresolved Issues
    "Total_Flight_Hours": [1000, 2000, 3000, 5000, 7000, 10000],  # Flight hours
    "Total_Cycles": [200, 400, 600, 1000, 1400, 2000],  # Example Cycles
    "Wear_Tear_Score": [1, 2, 3, 4, 5, 6],  # Wear and tear score
    "Natural_Integrity_Score": [1, 2, 3, 4, 5, 6],  # Integrity score
    "Pressurization_Cycles": [5, 10, 15, 20, 25, 30],  # Pressurization cycles
    "Aircraft_Usage": [10, 20, 30, 40, 50, 60],  # Aircraft usage
    "Maintenance_Cost_Per_Hour": [100, 150, 200, 250, 300, 350],  # Maintenance cost
    "Total_Logged_Issues": [1, 2, 3, 4, 5, 6],  # Total logged issues
    "Operational_Region": [1, 2, 3, 4, 5, 6],  # Operational region scale (1-6)
    "Environmental_Exposure": [0, 1, 2, 3, 4, 5],  # Environmental exposure scale
    "Maintenance_Type": [1, 2, 3, 4, 5, 6],  # Maintenance type scale (1-6)
    "Engine_Type": [1, 2, 3, 4, 5, 6],  # Engine type scale (1-6)
}

# Create synthetic DataFrame
synthetic_df = pd.DataFrame(synthetic_data)

# MinMax Scaler to scale features to 0-1 range
scaler = MinMaxScaler()

# Fit the scaler on the synthetic data
scaled_data = scaler.fit_transform(synthetic_df)

# Replace original synthetic data with the scaled data
scaled_df = pd.DataFrame(scaled_data, columns=synthetic_df.columns)

# Simulate the prediction function (replace this with your real prediction logic)
def predict_rating(features):
    # Apply scaling on input features based on synthetic data scale
    scaled_features = scaler.transform([features])
    # Fake prediction formula: Sum of scaled features multiplied by random weight (for demo purpose)
    rating = np.dot(scaled_features, np.random.rand(scaled_features.shape[1]))
    return rating[0]

# Streamlit UI code
def main():
    st.title("Aircraft Maintenance Rating Prediction")

    # Create sliders for each feature, with synthetic ranges
    aircraft_age = st.slider("Aircraft Age", min_value=10, max_value=90, value=10)
    unresolved_issues = st.slider("Unresolved Issues", min_value=0, max_value=5, value=0)
    total_flight_hours = st.slider("Total Flight Hours", min_value=1000, max_value=10000, value=1000)
    total_cycles = st.slider("Total Cycles", min_value=200, max_value=2000, value=200)
    wear_tear_score = st.slider("Wear and Tear Score", min_value=1, max_value=6, value=1)
    natural_integrity_score = st.slider("Natural Integrity Score", min_value=1, max_value=6, value=1)
    pressurization_cycles = st.slider("Pressurization Cycles", min_value=5, max_value=30, value=5)
    aircraft_usage = st.slider("Aircraft Usage", min_value=10, max_value=60, value=10)
    maintenance_cost_per_hour = st.slider("Maintenance Cost per Hour", min_value=100, max_value=350, value=100)
    total_logged_issues = st.slider("Total Logged Issues", min_value=1, max_value=6, value=1)
    operational_region = st.slider("Operational Region", min_value=1, max_value=6, value=1)
    environmental_exposure = st.slider("Environmental Exposure", min_value=0, max_value=5, value=0)
    maintenance_type = st.slider("Maintenance Type", min_value=1, max_value=6, value=1)
    engine_type = st.slider("Engine Type", min_value=1, max_value=6, value=1)

    # Bundle the input features into a list
    features = [
        aircraft_age,
        unresolved_issues,
        total_flight_hours,
        total_cycles,
        wear_tear_score,
        natural_integrity_score,
        pressurization_cycles,
        aircraft_usage,
        maintenance_cost_per_hour,
        total_logged_issues,
        operational_region,
        environmental_exposure,
        maintenance_type,
        engine_type
    ]

    # Display the predicted rating when user clicks button
    if st.button('Predict Rating'):
        rating = predict_rating(features)
        st.write(f"Predicted Rating: {rating:.2f}")

if __name__ == "__main__":
    main()