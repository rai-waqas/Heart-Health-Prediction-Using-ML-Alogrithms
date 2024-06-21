import numpy as np
import pickle

# Load the trained logistic regression model
with open('heart_log_regr.sav', 'rb') as model_file:
    lgr_model = pickle.load(model_file)

# Load the scaler used during training
with open('scaler.sav', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

def predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    # Create a numpy array from the input parameters
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    print("Input data:", input_data)
    
    # Scale the input data using the loaded scaler
    input_data_scaled = scaler.transform(input_data)
    print("Scaled input data:", input_data_scaled)
    
    # Make a prediction using the logistic regression model
    prediction = lgr_model.predict(input_data_scaled)
    print("Prediction:", prediction)
    
    # Return the prediction (1 = heart disease, 0 = no heart disease)
    return prediction[0]

age = 54
sex = 0
cp = 2
trestbps = 108
chol = 267
fbs = 0
restecg = 0
thalach = 167
exang = 0
oldpeak = 0
slope = 2
ca = 0
thal = 2
# Example usage:
print(predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal))
