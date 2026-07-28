import joblib

# Assuming model and scaler are already trained
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
