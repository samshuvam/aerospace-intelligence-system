import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load the dataset
df_maintenance = pd.read_csv('maintenance_data.csv')

# Define features and target
X_maintenance = df_maintenance[['EngineThrust', 'PressurizationCycles', 'ErrorsDetected', 'MaintenanceHistory', 'FlightHoursSinceLastCheck']]
y_maintenance = df_maintenance['MaintenanceRequired'].apply(lambda x: 1 if x == 'Yes' else 0)

# Split the data
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_maintenance, y_maintenance, test_size=0.2, random_state=42)

# Train the model
model_maintenance = RandomForestClassifier(random_state=42)
model_maintenance.fit(X_train_m, y_train_m)

# Make predictions
y_pred_m = model_maintenance.predict(X_test_m)

# Evaluate the model
print("Maintenance Dataset - Classification Report:\n", classification_report(y_test_m, y_pred_m))
