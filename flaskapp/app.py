from flask import Flask, render_template, request
import pickle
import pandas as pd
import os
import numpy as np

# Preparing the Classifier
cur_dir = os.path.dirname(__file__)


app = Flask(__name__)


#from google.cloud import aiplatform
#from typing import Dict, List, Union
#from google.oauth2 import service_account

#credentials = service_account.Credentials.from_service_account_file("telefonica-369122-7ee103eac228.json")

@app.route('/')
def index():
	return render_template('vertical-modern-menu-template/Index.html')

@app.route('/registro')
def registro():
	return render_template('vertical-modern-menu-template/FORMULARIO MODELO.html')

@app.route('/dashboard')
def dashboard():
	return render_template('vertical-modern-menu-template/DASHBOARD.html')
#from google.cloud import bigquery
#from google.oauth2 import service_account
#import pandas_gbq
import joblib

from datetime import datetime
@app.route('/results', methods=['POST'])
def predict():
    #credentials = service_account.Credentials.from_service_account_file("telefonica-369122-7ee103eac228.json")
    nombre = request.form['firstName1']
    apellido = request.form['lastName1']
    dni = request.form['DNI']
    sexo = int(request.form['sexo'])
    edad = int(request.form['edad'])
    
    tp3= "" #resfinal_diabetes
    smoking =  int(request.form['smoking'])
    yellow_Fingers =  int(request.form['yellow_Fingers'])
    anxiety =  int(request.form['anxiety'])
    peer_pressure =  int(request.form['peer_pressure'])
    chronic_disease =  int(request.form['chronic_disease'])
    fatigue =  int(request.form['fatigue'])
    swallowing_difficulty =  int(request.form['swallowing_difficulty'])
    allergy =  int(request.form['allergy'])
    wheezing =  int(request.form['wheezing'])
    alcohol_consuming =  int(request.form['alcohol_consuming'])
    coughing =  int(request.form['coughing'])
    shortness_breath =  int(request.form['shortness_breath'])
    chest_pain =  int(request.form['chest_pain'])
    final_lung = pd.DataFrame([[sexo-1,edad,smoking,yellow_Fingers,
                                anxiety,peer_pressure,chronic_disease,fatigue,
                                swallowing_difficulty,allergy,wheezing,
                                alcohol_consuming,coughing,shortness_breath,
                                chest_pain]],columns=    ['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY',
    'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING',
    'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH',
    'SWALLOWING DIFFICULTY', 'CHEST PAIN'])
    lung = [[sexo-1,edad,smoking,yellow_Fingers,
                                anxiety,peer_pressure,chronic_disease,fatigue,
                                swallowing_difficulty,allergy,wheezing,
                                alcohol_consuming,coughing,shortness_breath,
                                chest_pain]]
    import mlflow
    
    
    df = pd.read_csv("data/final_model/model.txt",header=None)
    my_clf = mlflow.pyfunc.load_model(df[0].values[0])
    resfinal_lung = my_clf.predict(final_lung)
    
    #"" #  predict_custom_trained_model_sample(project="214503334282", endpoint_id="1386163105230225408",
    #                                               location="us-central1", instances=lung)
    tp4 = resfinal_lung
    resfinal_lung=np.where(resfinal_lung[0] > 0.6, "Tiene un alto grado de padecer cancer al pulmon", "Tiene pulmones sanos")

    now = datetime.now()
    print("modelo_pulmon", resfinal_lung)
    return render_template('vertical-modern-menu-template/RECOMENDACION MODELO.html', lung= resfinal_lung)

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=9999,debug=True)