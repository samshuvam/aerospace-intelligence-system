from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('pilot_rating_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    feature_names = ['Flight_Hours', 'Training_Scores', 'Incident_Reports', 'Flight_Maneuver_Proficiency',
                     'Physiological_Data', 'Cognitive_Performance', 'Stress_Levels', 'Communication_Skills',
                     'Weather_Adaptability', 'Peer_Reviews', 'Simulator_Performance', 'Fatigue_Management']
    input_data = pd.DataFrame([features], columns=feature_names)
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    return render_template('index.html', prediction_text=f'Predicted Final Rating for New Pilot: {prediction[0]:.2f}')

if __name__ == "__main__":
    app.run(debug=True)