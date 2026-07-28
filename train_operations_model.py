import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load the dataset
df_operations = pd.read_csv('operations_data.csv')

# Define features and target
X_operations = df_operations[['FuelPlanned', 'FuelActual', 'RouteAdherence', 'PerformanceRating', 'GoAround', 'WeatherConditions', 'PilotEfficiency']]
y_operations = df_operations['OptimizedOperation'].apply(lambda x: 1 if x == 'Yes' else 0)

# Split the data
X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(X_operations, y_operations, test_size=0.2, random_state=42)

# Train the model
model_operations = RandomForestClassifier(random_state=42)
model_operations.fit(X_train_o, y_train_o)

# Make predictions
y_pred_o = model_operations.predict(X_test_o)

# Evaluate the model
print("Operations Dataset - Classification Report:\n", classification_report(y_test_o, y_pred_o))
