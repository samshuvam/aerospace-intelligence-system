import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load('flight_hazard_classifier.pkl')

# Define the feature columns
feature_columns = [
    'Duration (min)', 'Weather Temperature (°C)', 'Weather Wind Speed (km/h)', 
    'Weather Precipitation (mm)', 'Aircraft Age (years)', 'Aircraft Maintenance Score', 
    'Pilot Experience (years)', 'Pilot Fatigue Level (1-10)', 'Fuel Consumption Rate (kg/min)', 
    'Landing Rate (ft/min)', 'Pilot Rating', 'Aircraft Rating'
]

# Define the rating map
rating_map = {0: 'Poor', 1: 'Good', 2: 'Excellent'}

# Streamlit App
st.title("Enhanced Flight Hazard and Suggestion System")
st.markdown("""
This app predicts the **Hazard Category** for a given flight and provides additional insights and suggestions 
for operational safety, efficiency, and decision-making.
""")

# Collect user inputs for prediction
st.sidebar.header("Input Flight Details")

def get_user_input():
    duration = st.sidebar.slider('Duration (min)', 10, 500, 120)
    weather_temp = st.sidebar.slider('Weather Temperature (°C)', -10, 50, 25)
    weather_wind = st.sidebar.slider('Weather Wind Speed (km/h)', 0, 100, 15)
    weather_precip = st.sidebar.slider('Weather Precipitation (mm)', 0, 100, 5)
    aircraft_age = st.sidebar.slider('Aircraft Age (years)', 0, 30, 10)
    maintenance_score = st.sidebar.slider('Aircraft Maintenance Score', 1, 10, 7)
    pilot_exp = st.sidebar.slider('Pilot Experience (years)', 0, 40, 15)
    pilot_fatigue = st.sidebar.slider('Pilot Fatigue Level (1-10)', 1, 10, 5)
    fuel_rate = st.sidebar.slider('Fuel Consumption Rate (kg/min)', 50, 500, 200)
    landing_rate = st.sidebar.slider('Landing Rate (ft/min)', -1000, 0, -500)
    pilot_rating = st.sidebar.selectbox('Pilot Rating', options=['Poor', 'Good', 'Excellent'])
    aircraft_rating = st.sidebar.selectbox('Aircraft Rating', options=['Poor', 'Good', 'Excellent'])

    # Map ratings to numerical values
    pilot_rating_num = {v: k for k, v in rating_map.items()}[pilot_rating]
    aircraft_rating_num = {v: k for k, v in rating_map.items()}[aircraft_rating]

    # Combine all inputs into a single dataframe
    data = {
        'Duration (min)': duration,
        'Weather Temperature (°C)': weather_temp,
        'Weather Wind Speed (km/h)': weather_wind,
        'Weather Precipitation (mm)': weather_precip,
        'Aircraft Age (years)': aircraft_age,
        'Aircraft Maintenance Score': maintenance_score,
        'Pilot Experience (years)': pilot_exp,
        'Pilot Fatigue Level (1-10)': pilot_fatigue,
        'Fuel Consumption Rate (kg/min)': fuel_rate,
        'Landing Rate (ft/min)': landing_rate,
        'Pilot Rating': pilot_rating_num,
        'Aircraft Rating': aircraft_rating_num
    }
    return pd.DataFrame([data])

# Get user input
input_df = get_user_input()

st.subheader("Input Parameters")
st.write(input_df)

# Predict Hazard Category
if st.button("Predict and Get Suggestions"):
    prediction = model.predict(input_df)
    hazard_category = int(prediction[0])

    # Hazard prediction result
    st.subheader("Prediction Result")
    hazard_mapping = {
        0: "**No Hazard**: The flight is safe.",
        1: "**Turbulence Hazard**: Mild risk due to weather conditions.",
        2: "**Stress Hazard**: Moderate risk caused by operational issues.",
        3: "**Critical Hazard**: Severe risk detected!"
    }
    st.write(hazard_mapping[hazard_category])

    # Suggestions based on input
    st.subheader("Operational Suggestions")
    suggestions = []

    # Weather-related suggestions
    if input_df['Weather Temperature (°C)'].iloc[0] > 40:
        suggestions.append("High temperature detected. Ensure extra cooling for the engines before takeoff.")
    if input_df['Weather Wind Speed (km/h)'].iloc[0] > 60:
        suggestions.append("High wind speed detected. Suggest using headwind direction for better lift.")
    if input_df['Weather Precipitation (mm)'].iloc[0] > 50:
        suggestions.append("Heavy precipitation detected. Perform thorough de-icing before departure.")

    # Aircraft-related suggestions
    if input_df['Aircraft Age (years)'].iloc[0] > 20:
        suggestions.append("Older aircraft detected. Ensure recent maintenance checks are verified.")
    if input_df['Aircraft Maintenance Score'].iloc[0] < 5:
        suggestions.append("Low maintenance score. Double-check pre-flight checks for safety.")

    # Pilot-related suggestions
    if input_df['Pilot Experience (years)'].iloc[0] < 5:
        suggestions.append("Inexperienced pilot detected. Recommend adding a co-pilot with higher experience.")
    if input_df['Pilot Fatigue Level (1-10)'].iloc[0] > 7:
        suggestions.append("Pilot fatigue detected. Consider a short delay for proper rest.")
    if input_df['Pilot Rating'].iloc[0] == 0:
        suggestions.append("Pilot rating is 'Poor'. Consider reviewing their current readiness to fly.")

    # Traffic-related suggestions (randomized for demo purposes)
    traffic_conditions = np.random.choice(['low', 'moderate', 'high'])
    if traffic_conditions == 'high':
        suggestions.append("High air traffic detected. Expect possible delays or holding patterns.")

    # Final list of suggestions
    if suggestions:
        for suggestion in suggestions:
            st.write(f"- {suggestion}")
    else:
        st.write("No additional suggestions. Flight conditions look optimal!")

    st.subheader("Conclusion")
    st.write("Ensure compliance with all standard operating procedures and monitor conditions throughout the flight.")