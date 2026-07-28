import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

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

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get values from sliders
        features = [
            float(request.form['Aircraft_Age']),
            float(request.form['Unresolved_Issues']),
            float(request.form['Total_Flight_Hours']),
            float(request.form['Total_Cycles']),
            float(request.form['Wear_Tear_Score']),
            float(request.form['Natural_Integrity_Score']),
            float(request.form['Pressurization_Cycles']),
            float(request.form['Aircraft_Usage']),
            float(request.form['Maintenance_Cost_Per_Hour']),
            float(request.form['Total_Logged_Issues']),
            float(request.form['Operational_Region']),
            float(request.form['Environmental_Exposure']),
            float(request.form['Maintenance_Type']),
            float(request.form['Engine_Type']),
        ]

        # Call the synthetic prediction function
        rating = predict_rating(features)
        return render_template('index.html', rating=rating)

    return render_template('index.html', rating=None)

@app.route('/get_feature_info', methods=['GET'])
def get_feature_info():
    # Return feature information for UI (so sliders can be populated with example ranges)
    feature_info = {
        "Aircraft_Age": [10, 90],
        "Unresolved_Issues": [0, 5],
        "Total_Flight_Hours": [1000, 10000],
        "Total_Cycles": [200, 2000],
        "Wear_Tear_Score": [1, 6],
        "Natural_Integrity_Score": [1, 6],
        "Pressurization_Cycles": [5, 30],
        "Aircraft_Usage": [10, 60],
        "Maintenance_Cost_Per_Hour": [100, 350],
        "Total_Logged_Issues": [1, 6],
        "Operational_Region": [1, 6],
        "Environmental_Exposure": [0, 5],
        "Maintenance_Type": [1, 6],
        "Engine_Type": [1, 6]
    }
    return jsonify(feature_info)

if __name__ == '__main__':
    app.run(debug=True)