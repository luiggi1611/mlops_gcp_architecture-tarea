##################################################################

## LOCAL SERVER
## ############
# pip install virtualenv
# mlflow models serve -m s3://datapath-mlops-1/10/b9bfebd99bc3403bad43c1d35fdace67/artifacts/logistic_regression_model -p 1234


import requests
import pandas as pd

url = "http://127.0.0.1:1234/invocations"

data = pd.read_csv("../data/output_data/x_test.csv")
print('Total Rows:', len(data))

input_data = data[:10].values.tolist()
body = {"inputs": input_data}
headers={ 'ContentType':'application/json'}

response = requests.post(url, headers=headers, json=body)

# Manejar la respuesta del modelo
if response.status_code == 200:
    result = response.json()
    print("Respuesta del modelo:", result)
else:
    print("Error al realizar la consulta:", response.text)

