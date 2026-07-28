import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the dataset
data_path = 'comprehensive_pilot_data.csv'
pilot_data = pd.read_csv(data_path)

# Step 1: Aggregate data for each pilot by averaging metrics
aggregated_data = pilot_data.groupby("PilotID").mean()

# Define numerical and categorical columns
numerical_cols = ['HoursFlownPerYear', 'StressManagementScore', 'WeatherAdaptabilityScore', 
                  'AccidentFreeYears', 'TrainingHoursLastYear', 'FlightReviewScore', 
                  'FlightDisciplineRating', 'ReactionTimeScore', 'CommunicationSkills', 
                  'DecisionMakingScore']
categorical_cols = ['Certifications', 'AircraftTypeExperience']

# Drop irrelevant columns (PilotID column not needed after aggregation)
aggregated_data = aggregated_data.drop(columns=['PilotID'], errors='ignore')

# Step 2: Preprocessing
# Setup preprocessor with StandardScaler and OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

# Step 3: Dimensionality Reduction - Use PCA to reduce dimensions while retaining 95% variance
pca = PCA(n_components=0.95, random_state=42)

# Step 4: Model Selection - Random Forest for regression
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('pca', pca),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Step 5: Define features and target
# Here, we simulate a synthetic score based on available metrics as target
aggregated_data['SyntheticScore'] = aggregated_data[numerical_cols].mean(axis=1)
X = aggregated_data.drop(columns=['SyntheticScore'])
y = aggregated_data['SyntheticScore']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Model training
model.fit(X_train, y_train)

# Step 7: Model evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Model Mean Squared Error: {mse}")

# Step 8: Final scoring for all pilots
aggregated_data['PredictedScore'] = model.predict(X)

# Export the final pilot scores to CSV
output_path = 'final_pilot_scores.csv'
aggregated_data[['PredictedScore']].to_csv(output_path)

print(f"Modeling complete. Final pilot scores saved to {output_path}.")