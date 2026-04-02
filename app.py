# app.py
# Backend API for SugarTrace using Flask
# This handles the /predict endpoint

from flask import Flask, request, jsonify, send_from_directory
import pickle
import numpy as np
import os

app = Flask(__name__, static_folder='frontend')

# ---- Load models when server starts ----
# We load everything once so we don't reload on every request

print("Loading models...")

# check if models exist
if not os.path.exists('models/random_forest.pkl'):
    print("WARNING: Models not found. Please run train_model.py first!")

try:
    with open('models/logistic_regression.pkl', 'rb') as f:
        lr_model = pickle.load(f)
    print("Logistic Regression model loaded.")
except:
    lr_model = None
    print("Could not load Logistic Regression model.")

try:
    with open('models/random_forest.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    print("Random Forest model loaded.")
except:
    rf_model = None
    print("Could not load Random Forest model.")

try:
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("Scaler loaded.")
except:
    scaler = None
    print("Could not load scaler.")

try:
    with open('models/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    print("Feature names loaded:", feature_names)
except:
    # fallback feature names in case file not found
    feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                     'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    print("Using default feature names.")

try:
    with open('models/results.pkl', 'rb') as f:
        model_results = pickle.load(f)
    print("Model results loaded.")
except:
    model_results = {}
    print("Could not load model results.")


# ---- Helper function to get feature importances ----
def get_feature_importance():
    if rf_model is None:
        return {}
    importances = rf_model.feature_importances_
    importance_dict = {}
    for i, name in enumerate(feature_names):
        importance_dict[name] = round(float(importances[i]), 4)
    return importance_dict


# ---- Routes ----

# Serve the main HTML page
@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')

# Serve static files (css, js)
@app.route('/frontend/<path:filename>')
def serve_static(filename):
    return send_from_directory('frontend', filename)


# Main prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # get data from request
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data received'}), 400

        # pick which model to use (default is random forest)
        model_choice = data.get('model', 'random_forest')

        # extract feature values in the correct order
        input_features = []
        for fname in feature_names:
            val = data.get(fname)
            if val is None:
                return jsonify({'error': f'Missing feature: {fname}'}), 400
            input_features.append(float(val))

        # convert to numpy array and reshape for sklearn
        input_array = np.array(input_features).reshape(1, -1)

        # scale the input using the saved scaler
        if scaler is not None:
            input_scaled = scaler.transform(input_array)
        else:
            input_scaled = input_array  # fallback if scaler missing

        # choose model
        if model_choice == 'logistic_regression' and lr_model is not None:
            selected_model = lr_model
            model_name = 'Logistic Regression'
        elif rf_model is not None:
            selected_model = rf_model
            model_name = 'Random Forest'
        else:
            return jsonify({'error': 'No model available. Run train_model.py first.'}), 500

        # make prediction
        prediction = selected_model.predict(input_scaled)[0]
        probability = selected_model.predict_proba(input_scaled)[0][1]

        # convert to python native types (not numpy types)
        prediction = int(prediction)
        probability = float(probability)

        # build result label
        if prediction == 1:
            result_label = "Diabetic"
        else:
            result_label = "Not Diabetic"

        # simple risk level based on probability
        if probability >= 0.7:
            risk_level = "High"
        elif probability >= 0.4:
            risk_level = "Moderate"
        else:
            risk_level = "Low"

        # get feature importances only for random forest
        importances = {}
        if model_choice == 'random_forest':
            importances = get_feature_importance()

        response = {
            'prediction': prediction,
            'result': result_label,
            'probability': round(probability * 100, 2),  # as percentage
            'risk_level': risk_level,
            'model_used': model_name,
            'feature_importance': importances
        }

        return jsonify(response)

    except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify({'error': str(e)}), 500


# Endpoint to get model performance stats
@app.route('/model-info', methods=['GET'])
def model_info():
    if not model_results:
        return jsonify({'error': 'Model results not available'}), 500

    # format it nicely
    info = {}
    for model_name, metrics in model_results.items():
        if model_name == 'feature_importance':
            continue
        info[model_name] = {
            'accuracy': round(metrics['accuracy'] * 100, 2),
            'recall': round(metrics['recall'] * 100, 2),
            'roc_auc': round(metrics['roc_auc'] * 100, 2)
        }

    return jsonify(info)


if __name__ == '__main__':
    print("\nStarting SugarTrace server...")
    print("Open http://localhost:5000 in your browser\n")
    app.run(debug=True, port=5000)
