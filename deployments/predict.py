import boto3
import json
import pandas as pd
 
region = 'us-west-1'
deployment_name = 'mlops-model-datapath-a'

# Connection
runtime = boto3.Session().client(
    'sagemaker-runtime', 
    region_name=region)
 
# Prepare data
data = pd.read_csv("../data/output_data/x_test.csv")
input_data = data[:3].values.tolist()
payload = json.dumps({"inputs": input_data})

# Send image via InvokeEndpoint API
response = runtime.invoke_endpoint(
    EndpointName=deployment_name, 
    ContentType='application/json', 
    Body=payload)

# Unpack response
result = json.loads(response['Body'].read().decode())
print(result)




# response = runtime.invoke_endpoint(
#     EndpointName=deployment_name,
#     ContentType="application/json", 
#     format="pandas-split",
#     Body=’{“columns”:[“text”],”data”:[[“This is terrible weather”],
#                                      [“This is great weather”]]}’)
# print(str(json.loads(response[‘Body’].read())))