import streamlit as st
import pandas as pd
import joblib

# Load necessary files
model = joblib.load("scaled_random_forest_pilot_rating_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_ranges = joblib.load("feature_ranges.pkl")

# General information
st.title("Pilot Performance Rating Prediction")
st.markdown(
    """
    ### Note:
    - All inputs are scaled between 0-100 for simplicity.
    - For this model, **higher values are considered better**, even for features like "Stress Level" and "Reaction Time".
    """
)

# Styling for improved UI
st.markdown(
    """
    <style>
    .feature-box {
        padding: 15px;
        border: 1px solid #ddd;
        border-radius: 5px;
        margin-bottom: 15px;
        background-color: black;
    }
    .feature-title {
        font-size: 16px;
        font-weight: bold;
        margin-bottom: 5px;
    }

    .divider {
        margin: 15px 0;
        border-top: 1px solid #ccc;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Input collection
st.markdown("### Input Pilot Data")
input_data = {}
for feature, (min_val, max_val) in feature_ranges.items():
    unit_change = (max_val - min_val) / 100  # Scale adjustment
    input_scaled_value = st.slider(
        f"{feature} (Scaled Range: 0 to 100)",
        0,
        100,
        50,
        help=f"Original range: [{min_val:.2f}, {max_val:.2f}]",
    )
    # Convert scaled value back to the original range
    original_value = min_val + (input_scaled_value / 100) * (max_val - min_val)
    st.markdown(
        f"""
        <div class="feature-box">
            <div class="feature-title">{feature}</div>
            <div>Entered Scaled Value: {input_scaled_value}</div>
            <div>Corresponding Original Value: {original_value:.2f}</div>
        </div>
        <div class="divider"></div>
        """,
        unsafe_allow_html=True,
    )
    input_data[feature] = original_value  # Save the original value

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data])

# Scale input data to match model's training scale
scaled_input = scaler.transform(input_df)

# Predict performance score and category
scaled_prediction = model.predict(scaled_input)[0]  # Scaled prediction
performance_score = scaled_prediction  # Assume output is scaled to 0-100

# Determine performance level
if performance_score < 40:
    performance_level = "Needs Improvement"
elif 40 <= performance_score < 60:
    performance_level = "Average"
elif 60 <= performance_score < 80:
    performance_level = "Good"
else:
    performance_level = "Excellent"

# Display results
st.markdown("### Prediction Results")
st.success(f"The predicted pilot performance score is: {performance_score:.2f}")
st.info(f"The predicted performance level is: {performance_level}")