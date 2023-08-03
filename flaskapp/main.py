from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
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

'''def predict_custom_trained_model_sample(
    project: str,
    endpoint_id: str,
    instances: Union[Dict, List[Dict]],
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
    """
    `instances` can be either single instance of type dict or a list
    of instances.
    """
    #api_endpoint = "us-central1-aiplatform.googleapis.com"
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options,credentials=credentials )
    # The format of each instance should conform to the deployed model's prediction input schema.
    #instances = instances if type(instances) == list else [instances]
    #instances = [
    #    json_format.ParseDict(instance_dict, Value()) for instance_dict in instances
    #]
    print(instances)
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances
    )
    print("response")
    print(" deployed_model_id:", response.deployed_model_id)
    # The predictions are a google.protobuf.Value representation of the model's predictions.
    predictions = response.predictions
    print(predictions)
    return predictions
'''
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
    red_blood_cell_count = float(request.form['red_blood_cell_count'])
    packed_cell_volume = float(request.form['packed_cell_volume'])
    mean_cell_volume = float(request.form['mean_cell_volume'])
    mean_cell_hemoglobin = float(request.form['mean_cell_hemoglobin'])
    mCHC = float(request.form['mCHC'])
    red_cell_distribution_width = float(request.form['red_cell_distribution_width'])
    white_blood_cell = float(request.form['white_blood_cell'])
    platelet = float(request.form['platelet'])
    input_data = [[edad,  sexo,  red_blood_cell_count, packed_cell_volume,  mean_cell_volume, mean_cell_hemoglobin,  mCHC,
				    red_cell_distribution_width,  white_blood_cell,platelet]]

    resfinal =  "" # predict_custom_trained_model_sample(project="214503334282",endpoint_id="4930496011970805760",location="us-central1",instances=input_data)
    print(resfinal)
    tp1 = resfinal
    resfinal = np.where(resfinal[0]==3,"ANEMIA GRAVE",
                        np.where(resfinal[0]==2,"ANEMIA MEDIA",
                                 np.where(resfinal[0]==1,"ANEMIA LEVE","SIN ANEMIA")))
    project = 'telefonica-369122'
    schema = 'Aplicativo'
    final =pd.DataFrame(input_data,columns=["edad",  "sexo",  "red_blood_cell_count", "packed_cell_volume",  "mean_cell_volume", "mean_cell_hemoglobin",  "mCHC",
				    "red_cell_distribution_width",  "white_blood_cell","platelet"])
    final['nombre'] =nombre
    final['apellido'] =apellido
    final['dni'] =dni
    final['prediccion_anemia'] = resfinal

    cat_cols = ['sex', 'exng', 'caa', 'cp', 'fbs', 'restecg', 'slp', 'thall']
    con_cols = ["age", "trtbps", "chol", "thalachh", "oldpeak"]

    chest_pain = int(request.form['chest_pain'])
    resting_blood_pressure = float(request.form['resting_blood_pressure'])
    cholestoral = float(request.form['cholestoral'])
    fasting_blood_sugar = int(request.form['fasting_blood_sugar'])
    number_vessels = float(request.form['number_vessels'])
    thalium = int(request.form['thalium'])
    electrocardiographic = int(request.form['electrocardiographic'])
    maximum_heart_rate = float(request.form['maximum_heart_rate'])
    previous_peek = float(request.form['previous_peek'])
    slope = int(request.form['slope'])
    induced_angina = int(request.form['induced_angina'])

    final['chest_pain'] =  chest_pain
    final['resting_blood_pressure'] =  resting_blood_pressure
    final['cholestoral'] = cholestoral
    final['prediccion_anemia'] = fasting_blood_sugar
    final['prediccion_anemia'] = number_vessels
    final['prediccion_anemia'] = thalium
    final['prediccion_anemia'] = electrocardiographic
    final['prediccion_anemia'] = maximum_heart_rate
    final['prediccion_anemia'] = previous_peek
    final['prediccion_anemia'] = maximum_heart_rate
    final['prediccion_anemia'] = slope
    final_hatack= pd.DataFrame([[edad, sexo-1,chest_pain, resting_blood_pressure, cholestoral, fasting_blood_sugar,number_vessels,thalium,electrocardiographic,maximum_heart_rate,previous_peek,slope,induced_angina]],
                               columns=['age','sex','cp','trtbps','chol','fbs','caa','thall','restecg','thalachh','oldpeak','slp','exng'])
    print(final_hatack.values)
    df1 = pd.get_dummies(final_hatack, columns=cat_cols, drop_first=True)

    df2 = pd.DataFrame([[ 0.59259259,  0.75      , -0.11023622, -0.09230769,  0.9375    ,
        1.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  1.        ,  1.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  1.        ,
        0.        ,  0.        ]],columns=['age', 'trtbps', 'chol', 'thalachh', 'oldpeak', 'sex_1', 'exng_1',
       'caa_1', 'caa_2', 'caa_3', 'caa_4', 'cp_1', 'cp_2', 'cp_3', 'fbs_1',
       'restecg_1', 'restecg_2', 'slp_1', 'slp_2', 'thall_1', 'thall_2',
       'thall_3'])
    df2= df2.append(df1).fillna(0)
    scaler = joblib.load("scaler.save")
    df2[con_cols] = scaler.transform(df2[con_cols])
    #resfinal_HA = predict_custom_trained_model_sample(project="214503334282", endpoint_id="6755579760962699264",
    #                                               location="us-central1", instances=[df2.iloc[1,:].values.tolist()])
    #tp2=resfinal_HA
    pregnancies = int(request.form['pregnancies'])
    glucose = int(request.form['glucose'])
    bloodpressure = int(request.form['bloodpressure'])
    skinthickness = int(request.form['skinthickness'])
    insulin = int(request.form['insulin'])
    BMI = float(request.form['BMI'])
    diabetespedigreefunction = float(request.form['diabetespedigreefunction'])


    final_diabetes =pd.DataFrame([[pregnancies,glucose,bloodpressure,skinthickness,insulin,BMI,diabetespedigreefunction,edad]],
                                 columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age'])
    diabetes =[[pregnancies,glucose,bloodpressure,skinthickness,insulin,BMI,diabetespedigreefunction,edad]]
    resfinal_diabetes = ""#predict_custom_trained_model_sample(project="214503334282", endpoint_id="2460271616358088704",
                                                   location="us-central1", instances=diabetes)
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
    resfinal_lung = "" #  predict_custom_trained_model_sample(project="214503334282", endpoint_id="1386163105230225408",
                                                   location="us-central1", instances=lung)
    tp4 = resfinal_lung
    resfinal_lung=np.where(resfinal_lung[0] > 0.6, "Tiene un alto grado de padecer cancer al pulmon", "Tiene pulmones sanos")

    now = datetime.now()
    print("modelo_HA",resfinal_HA )
    print("modelo_diabetes", resfinal_diabetes)
    print("modelo_pulmon", resfinal_lung)
    final['fecha'] = str(now)[0:10]
    print(final)
    #pandas_gbq.context.credentials = credentials
    #input_data

    resfinal_HA = np.where(resfinal_HA[0] == 1, "Tiene un alto grado de padecer un ataque al corazón", "Tiene un corazon sano")
    resfinal_diabetes = np.where(resfinal_diabetes[0] == 1, "Tiene un alto grado de padecer diabetes", "Tiene baja probabilidad de tener diabetes")
    print(resfinal,resfinal_HA, resfinal_diabetes,resfinal_lung)
    tmp = pd.DataFrame([[nombre,apellido, dni,edad, sexo,  resting_blood_pressure, cholestoral, fasting_blood_sugar, number_vessels, thalium,
      electrocardiographic, maximum_heart_rate, previous_peek, slope, induced_angina, red_blood_cell_count, packed_cell_volume,  mean_cell_volume, mean_cell_hemoglobin,  mCHC,
				    red_cell_distribution_width,  white_blood_cell,platelet,pregnancies,glucose,bloodpressure,skinthickness,insulin,BMI,diabetespedigreefunction,smoking,yellow_Fingers,
                                anxiety,peer_pressure,chronic_disease,fatigue,
                                swallowing_difficulty,allergy,wheezing,
                                alcohol_consuming,coughing,shortness_breath,
                                chest_pain,tp1[0],tp2[0],tp3[0],tp4[0]]],columns=["nombre","apellido", "dni","edad", "sexo", "resting_blood_pressure", "cholestoral", "fasting_blood_sugar", "number_vessels", "thalium",
      "electrocardiographic", "maximum_heart_rate", "previous_peek", "slope", "induced_angina",  "red_blood_cell_count", "packed_cell_volume",  "mean_cell_volume", "mean_cell_hemoglobin",  "mCHC",
				    "red_cell_distribution_width",  "white_blood_cell","platelet","pregnancies","glucose","bloodpressure","skinthickness","insulin","BMI","diabetespedigreefunction","smoking","yellow_Fingers",
                                "anxiety","peer_pressure","chronic_disease","fatigue",
                                "swallowing_difficulty","allergy","wheezing",
                                "alcohol_consuming","coughing","shortness_breath",
                                "chest_pain","resfinal","resfinal_HA", "resfinal_diabetes","resfinal_lung"])

    tmp['fecha'] = str(now)[0:10]
    #client = bigquery.Client(credentials=credentials, project=credentials.project_id, )
    print(tmp)#.to_gbq(schema + '.' + "DATA_APLICATIVO", project_id=project, if_exists='append')

    #resfinal_lung = np.where(resfinal_lung[0] == 1, "Tiene un alto grado de padecer cancer al pulmon", "Tiene pulmones sanos")
    return render_template('vertical-modern-menu-template/RECOMENDACION MODELO.html', res=resfinal,ha =resfinal_HA,dia= resfinal_diabetes,lung= resfinal_lung)

if __name__ == '__main__':
	app.run(host='127.0.0.1', port=8080, debug=True)