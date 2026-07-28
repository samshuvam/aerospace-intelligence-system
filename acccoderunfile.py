import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# Step 1: Load the dataset
df = pd.read_csv('s_acc_data.csv')

# Encode categorical variables
label_encoders = {}
for column in ['AircraftType', 'LandingQuality', 'WeatherConditions', 'TailStrike']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Splitting dataset
X = df.drop(columns=['FlightID', 'AccidentRisk'])
y = df['AccidentRisk']

# Encode the target column
y = LabelEncoder().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 2: Training with XGBoost Classifier
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb_model.fit(X_train, y_train)

# Step 3: Predictions and Classification Report
y_pred = xgb_model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['Highly Likely to Crash', 'Less Likely to Crash', 'Inconclusive']))

# # Step 4: Function to Predict Accident Risk based on input features
# def predict_accident_risk(input_data):
#     input_df = pd.DataFrame([input_data])
    
#     # Encode input data the same way as training data
#     for column, le in label_encoders.items():
#         input_df[column] = le.transform(input_df[column])
    
#     prediction = xgb_model.predict(input_df)
#     risk_category = ['Highly Likely to Crash', 'Less Likely to Crash', 'Inconclusive'][prediction[0]]
    
#     return risk_category

# # Example usage
# example_input = {
#     'AircraftType': ['Type A'],
#     'RouteAdherence': [95],
#     'LandingQuality': ['Good'],
#     'PerformanceRating': [90],
#     'ErrorsDetected': [1],
#     'WeatherConditions': ['Clear'],
#     'AirTrafficControlIssues': [1],
#     'PilotExperience': [12000],
#     'FlightDuration': [6.5],
#     'TailStrike': ['No']
# }

# # Predict risk category
# predicted_risk = predict_accident_risk(example_input)
# print(f"The predicted accident risk is: {predicted_risk}")


import pandas as pd

# Function to Predict Accident Risk based on input features
def predict_accident_risk(input_data):
    input_df = pd.DataFrame([input_data])  # Create DataFrame with one row of input data
    
    # Encode input data the same way as training data
    for column, le in label_encoders.items():
        input_df[column] = le.transform(input_df[column])
    
    prediction = xgb_model.predict(input_df)
    risk_category = ['Highly Likely to Crash', 'Less Likely to Crash', 'Inconclusive'][prediction[0]]
    
    return risk_category

# Example usage with scalar values for prediction
example_input = {
    'AircraftType': 'Type A',  # Pass scalar instead of list
    'RouteAdherence': 95,
    'LandingQuality': 'Good',
    'PerformanceRating': 90,
    'ErrorsDetected': 1,
    'WeatherConditions': 'Clear',
    'AirTrafficControlIssues': 1,
    'PilotExperience': 12000,
    'FlightDuration': 6.5,
    'TailStrike': 'No'
}

# Predict risk category
predicted_risk = predict_accident_risk(example_input)
print(f"The predicted accident risk is: {predicted_risk}")
