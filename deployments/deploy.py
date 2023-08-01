# DEPLOYMENT
############
# https://mlflow.org/docs/latest/python_api/mlflow.sagemaker.html

from mlflow.deployments import get_deploy_client
# Configuración de AWS
region = 'us-west-1'
aws_id = '114899394146'
bucket = 'datapath-mlops-1'
role_arn = 'arn:aws:iam::114899394146:role/aws-sagemaker-for-deploy-ml-model'
# URI del modelo en MLflow
# mlflow_experiment_id = '2'
# mlflow_run_id = '41afa1b68e744837b9728ae4f6954743'
model_name = 'prd_logreg_model'
# model_uri = f's3://{bucket}/{mlflow_experiment_id}/{mlflow_run_id}/artifacts/{model_name}'
model_uri = f'models:/{model_name}/Production'

# Deployment
image_name = 'mlops-model-datapath-a'
deployment_name = 'mlops-model-datapath-a'
tag_id = '2.4.2'
image_url = f'{aws_id}.dkr.ecr.{region}.amazonaws.com/{image_name}:{tag_id}'

config = dict(
    assume_role_arn=role_arn, execution_role_arn=role_arn, bucket=bucket, 
    image_url=image_url, region_name=region, archive=False,
    instance_count=1, synchronous=True, timeout_seconds=300
)

client = get_deploy_client(f"sagemaker:{region}")
client.create_deployment(
    deployment_name,
    model_uri=model_uri,
    flavor="python_function",
    config=config,
)
