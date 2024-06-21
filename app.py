from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained logistic regression model
with open('heart_log_regr.sav', 'rb') as model_file:
    lgr_model = pickle.load(model_file)

# Load the scaler used during training
with open('scaler.sav', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

def predict_heart_disease(input_data):
    # Scale the input data using the loaded scaler
    input_data_scaled = scaler.transform(input_data)
    
    # Make a prediction using the logistic regression model
    prediction = lgr_model.predict(input_data_scaled)
    
    # Return the prediction (1 = heart disease, 0 = no heart disease)
    return prediction[0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    data = request.get_json(force=True)
    
    # Extract features from the JSON data
    features = [
        data['age'], data['sex'], data['cp'], data['trestbps'],
        data['chol'], data['fbs'], data['restecg'], data['thalach'],
        data['exang'], data['oldpeak'], data['slope'], data['ca'], data['thal']
    ]
    input_data = np.array([features])
    
    # Predict heart disease
    prediction = predict_heart_disease(input_data)
    
    return jsonify({'prediction': int(prediction)})

if __name__ == "__main__":
    app.run(debug=True)
