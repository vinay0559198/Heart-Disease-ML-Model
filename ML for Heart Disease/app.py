# app.py
from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('heart_disease_model.pkl')

# Function to classify test levels
def classify_level(value, thresholds):
    if value <= thresholds[0]:
        return "Good"
    elif value <= thresholds[1]:
        return "Okay"
    elif value <= thresholds[2]:
        return "Bad"
    else:
        return "Extremely Bad"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    try:
        # Extract and process input data
        inputs = np.array([
            float(data['age']),
            float(data['sex']),
            float(data['chest_pain_type']),
            float(data['resting_blood_pressure']),
            float(data['serum_cholesterol']),
            float(data['fasting_blood_sugar']),
            float(data['resting_ecg']),
            float(data['max_heart_rate']),
            float(data['exercise_induced_angina']),
            float(data['oldpeak']),
            float(data['st_segment']),
            float(data['major_vessels']),
            float(data['thal'])
        ]).reshape(1, -1)

        # Make a prediction
        prediction = model.predict(inputs)[0]
        probabilities = model.predict_proba(inputs)[0]

        # Test level classifications
        levels = {
            "age": classify_level(float(data['age']), [30, 50, 65]),
            "resting_blood_pressure": classify_level(float(data['resting_blood_pressure']), [120, 140, 160]),
            "serum_cholesterol": classify_level(float(data['serum_cholesterol']), [200, 240, 300]),
            "max_heart_rate": classify_level(float(data['max_heart_rate']), [100, 140, 180]),
            "oldpeak": classify_level(float(data['oldpeak']), [1, 2, 4])
        }

        result = {
            "prediction": "Positive for Heart Disease" if prediction == 1 else "Negative for Heart Disease",
            "probability": f"{probabilities[1] * 100:.2f}%",
            "test_levels": levels
        }

        return render_template('index.html', result=result)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
