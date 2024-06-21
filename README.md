# Machine Learning based Web Application to predict Heart Disease
Developed a machine learning model to predict heart disease.<br>
I have obtained the dataset from Kaggle. It has patient medical records with 13 parameters/columns useful to predict heart health.<br>
```
The 13 parameters which are given in the datset used to predict if a person has heart disease or not are:
- age : The person's age in years
- sex : The person's sex (1 = male, 0 = female)
- cp : The chest pain experienced (Value 1: typical angina, Value 2: atypical angina, Value 3: non-anginal pain, Value 4: asymptomatic)
- trestbps : The person's resting blood pressure (mm Hg on admission to the hospital)
- chol : The person's cholesterol measurement in mg/dl
- fbs : The person's fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false)
- restecg : Resting electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria)
- thalach : The person's maximum heart rate achieved
- exang : Exercise induced angina (1 = yes; 0 = no)
- oldpeak : ST depression induced by exercise relative to rest ('ST' relates to positions on the ECG plot)
- slope : the slope of the peak exercise ST segment (Value 1: upsloping, Value 2: flat, Value 3: downsloping)
- ca : The number of major vessels (0-3)
- thal : A blood disorder called thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect) 
```
<br>
The last column in the dataset "target" represents if a person has hear disease or not (1 = Heart disease, 0 = No Heart disease).<br><br>
After training and testing the dataset with KNN, Logistic Regression and Linear Regression and Naive Bayes and comparing their accuracies, chose to use Logistic Regression for implementation.<br><br>
Then saved this model and using Flask for backend.
<br><br><br>

How to RUN the Project:
- Install necessary packages such as Flask, Numpy etc.
- Then type in the command
``` python app.py ```
<br><br><br>