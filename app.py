from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model and scaler
model = joblib.load('voting_classifier_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = [
        data['no_of_dependents'],
        data['education'],
        data['self_employed'],
        data['income_annum'],
        data['loan_amount'],
        data['loan_term'],
        data['cibil_score'],
        data['residential_assets_value'],
        data['commercial_assets_value'],
        data['luxury_assets_value'],
        data['bank_asset_value']
    ]

    features = np.array([features])
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)
    loan_status = 'Approved' if prediction[0] == 0 else 'Rejected'

    return jsonify({'loan_status': loan_status})
