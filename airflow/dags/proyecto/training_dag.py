import os
import sys
from datetime import timedelta
from loguru import logger
import pandas as pd
import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking.client import MlflowClient

# The DAG object; we'll need this to instantiate a DAG
from airflow.decorators import dag, task
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from sklearn.model_selection import train_test_split
# sys.path.append("/opt/airflow/dags/")
from proyecto.utils import (
    check_keys,homologate_values_dummy, get_read_data, get_split_train_test,
    get_preprocess_data, get_model_1,get_model_2, get_val_performance,
    get_cv_performance, get_predictions,
    VARS_TO_DROP    
)

PATH_DIR_DATA = "./data"
PATH_DIR_DATA_INPUT = PATH_DIR_DATA + '/input_data'
PATH_DIR_DATA_OUTPUT = PATH_DIR_DATA + '/output_data_model'
PATH_DIR_DATA_OUTPUT_MODEL = PATH_DIR_DATA + '/final_model'
if not os.path.exists(PATH_DIR_DATA_OUTPUT):
    os.makedirs(PATH_DIR_DATA_OUTPUT)

if not os.path.exists(PATH_DIR_DATA_OUTPUT_MODEL):
    os.makedirs(PATH_DIR_DATA_OUTPUT_MODEL)


_data_files_ = {
    'input_raw_data_file': os.path.join(PATH_DIR_DATA_INPUT, 'survey lung cancer.csv'),
    'raw_data_file': os.path.join(PATH_DIR_DATA_OUTPUT + '/data.csv'),
    'raw_data_file_homologate': os.path.join(PATH_DIR_DATA_OUTPUT + '/data_homologate.csv'),
    'raw_x_train_file': os.path.join(PATH_DIR_DATA_OUTPUT + '/x_train_raw.csv'),
    'raw_x_test_file': os.path.join(PATH_DIR_DATA_OUTPUT + '/x_test_raw.csv'),
    'raw_y_train_file': os.path.join(PATH_DIR_DATA_OUTPUT + '/y_train_raw.csv'),
    'raw_y_test_file': os.path.join(PATH_DIR_DATA_OUTPUT + '/y_test_raw.csv'),
    'transformed_x_train_file': os.path.join(PATH_DIR_DATA_OUTPUT + '/x_train.csv'),
    'transformed_y_train_file': os.path.join(PATH_DIR_DATA_OUTPUT + '/y_train.csv'),
    'transformed_x_test_file': os.path.join(PATH_DIR_DATA_OUTPUT + '/x_test.csv'),
    'transformed_y_test_file': os.path.join(PATH_DIR_DATA_OUTPUT + '/y_test.csv'),
}

model_name1 = 'prd_logreg_model_lung'
experiment_name1 = "prd-lung-prediction"
type_model1 = 'logistic_regression'
model_name2 = 'prd_RF_model_lung'
experiment_name2 = "prd-RF-prediction"
type_model2 = 'RF'
models_mod = [model_name1 ,model_name2]
args={
    'owner' : 'datapath',
    'depends_on_past': False,
    'retries': 1,
    'start_date':days_ago(1)  #1 means yesterday
}

@dag(
    dag_id='ci_ml_train_model', ## Name of DAG run
    default_args=args,
    description='ML pipeline Training Model lung cancer',
    schedule = None 
)
def mydag():

    @task
    def read_data(data_files):

        data = get_read_data(data_files['input_raw_data_file'])
        logger.info(f"Total rows: {len(data)}")

        data.to_csv(data_files['raw_data_file'], index=False)
        logger.success(f"File {data_files['raw_data_file']} was saved successfully.")

    @task
    def homologate_values(data_files):
        data = pd.read_csv(data_files['raw_data_file'])
        data = homologate_values_dummy(data)
        data.to_csv(data_files['raw_data_file_homologate'], index=False)
        logger.success(f"File {data_files['raw_data_file_homologate']} was saved successfully.")

    @task
    def split_train_test(data_files):

        data = pd.read_csv(data_files['raw_data_file_homologate'])
        x_train, x_test, y_train, y_test = get_split_train_test(data)

        x_train.to_csv(data_files['raw_x_train_file'], index=False)
        x_test.to_csv(data_files['raw_x_test_file'], index=False)
        y_train.to_csv(data_files['raw_y_train_file'], index=False)
        y_test.to_csv(data_files['raw_y_test_file'], index=False)
        logger.success("Initial dataset was split and saved successfully.")

    @task
    def preprocess_data(data_files):

        x_train = pd.read_csv(data_files['raw_x_train_file'])
        x_test = pd.read_csv(data_files['raw_x_test_file'])
        y_train = pd.read_csv(data_files['raw_y_train_file'])
        y_test = pd.read_csv(data_files['raw_y_test_file'])

        x_train, x_test, y_train, y_test = get_preprocess_data(x_train, x_test, y_train, y_test)

        x_train.to_csv(data_files['transformed_x_train_file'], index=False)
        x_test.to_csv(data_files['transformed_x_test_file'], index=False)
        pd.DataFrame(y_train).to_csv(data_files['transformed_y_train_file'], index=False)
        pd.DataFrame(y_test).to_csv(data_files['transformed_y_test_file'], index=False)

        logger.success("Data were transformed and saved successfully.")

    @task
    def train_model1(data_files, experiment_name, model_name,
                    track_cv_performance=True) -> dict[str, str]:

        x_train = pd.read_csv(data_files['transformed_x_train_file'])
        x_test = pd.read_csv(data_files['transformed_x_test_file'])
        y_train = pd.read_csv(data_files['transformed_y_train_file']).values  # get np array
        y_test = pd.read_csv(data_files['transformed_y_test_file']).values  # get np array

        # Get the untrained model
        clf = get_model_1()

        # MLflow: experiment name
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run() as run:
            # MLflow: print run specific info
            run_id = run.info.run_id
            logger.info(f"\nActive run_id: {run_id}")

            # Train
            clf.fit(x_train, y_train)

            # MLflow: track model parameters
            mlflow.log_params(clf.get_params())

            # MLflow: track CV performance
            if track_cv_performance is True:
                cv_accuracy, cv_f1 = get_cv_performance(x_train, y_train, clf)
                metrics = {"cv_accuracy": cv_accuracy, "cv_f1": cv_f1}
                mlflow.log_metrics(metrics)

            # MLflow: track performance on validation
            if x_test is not None and y_test is not None:
                y_pred = get_predictions(x_test, clf)
                val_accuracy, val_f1 = get_val_performance(y_test, y_pred)
                metrics = {"val_accuracy": val_accuracy, "val_f1": val_f1}
                mlflow.log_metrics(metrics)

            # MLflow log the model
            mlflow.sklearn.log_model(clf, model_name)
            model_uri = mlflow.get_artifact_uri(model_name)

        logger.info(f"run_id: {run_id}")
        logger.info(f"model_uri: {model_uri}")
        
        return {'run_id': run_id, 'model_uri': model_uri}
    
    @task
    def train_model2(data_files, experiment_name, model_name,
                    track_cv_performance=True) -> dict[str, str]:

        x_train = pd.read_csv(data_files['transformed_x_train_file'])
        x_test = pd.read_csv(data_files['transformed_x_test_file'])
        y_train = pd.read_csv(data_files['transformed_y_train_file']).values  # get np array
        y_test = pd.read_csv(data_files['transformed_y_test_file']).values  # get np array
        xtrain, x_valid, ytrain, y_valid = train_test_split(x_train,y_train,test_size=0.3,
        random_state=123)
        # Get the untrained model
        clf = get_model_2(xtrain, x_valid, ytrain, y_valid)

        # MLflow: experiment name
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run() as run:
            # MLflow: print run specific info
            run_id = run.info.run_id
            logger.info(f"\nActive run_id: {run_id}")

            # Train
            clf.fit(x_train, y_train)

            # MLflow: track model parameters
            mlflow.log_params(clf.get_params())

            # MLflow: track CV performance
            if track_cv_performance is True:
                cv_accuracy, cv_f1 = get_cv_performance(x_train, y_train, clf)
                metrics = {"cv_accuracy": cv_accuracy, "cv_f1": cv_f1}
                mlflow.log_metrics(metrics)

            # MLflow: track performance on validation
            if x_test is not None and y_test is not None:
                y_pred = get_predictions(x_test, clf)
                val_accuracy, val_f1 = get_val_performance(y_test, y_pred)
                metrics = {"val_accuracy": val_accuracy, "val_f1": val_f1}
                mlflow.log_metrics(metrics)

            # MLflow log the model
            mlflow.sklearn.log_model(clf, model_name)
            model_uri = mlflow.get_artifact_uri(model_name)

        logger.info(f"run_id: {run_id}")
        logger.info(f"model_uri: {model_uri}")
        
        return {'run_id': run_id, 'model_uri': model_uri}

    def check_registered_model(data_files, reg_model, new_model_uri,register_model,register_model_by_comparison,stop):
        reg_model_latest = f"models:/{reg_model}/latest"

        try:
            # Verificamos si existe un modelo registrado
            old_model = mlflow.pyfunc.load_model(reg_model_latest)
            logger.info(f"Existe un modelo {reg_model} previamente registrado!")
        except MlflowException:
            logger.info(f"No existe un modelos registrado {reg_model}.")
            return register_model 
        #'register_model1'
        
        x_test = pd.read_csv(data_files['transformed_x_test_file'])
        y_test = pd.read_csv(data_files['transformed_y_test_file']).values

        # First check if the requested model and version exist
        new_model = mlflow.pyfunc.load_model(model_uri=new_model_uri)

        new_y_pred = get_predictions(x_test, new_model)
        old_y_pred = get_predictions(x_test, old_model)

        new_val_accuracy, _ = get_val_performance(y_test, new_y_pred)
        old_val_accuracy, _ = get_val_performance(y_test, old_y_pred)

        logger.info(f'Old Model: {old_val_accuracy}')
        logger.info(f'New Model: {new_val_accuracy}')

        if new_val_accuracy >= old_val_accuracy:
            logger.info(f"Restraremos una nueva version del modelo {reg_model}.")
            return register_model_by_comparison
        else:
            logger.info(f"El nuevo modelo {reg_model} no presenta un mejor performance.")
            return stop


    def register_model(model_name, model_uri):
        mv = mlflow.register_model(model_uri, model_name)
        logger.success(f"Model {model_name} registered.")


    def register_model_by_comparison(model_name, model_uri, push_to_production=False):
        mv = mlflow.register_model(model_uri, model_name)
        logger.success(f"Model {model_name} registered.")
        
        if push_to_production is True:
            
            model_version = mv.version
            client = MlflowClient()
            # Get all registered models of given model_name tagged with "Production"
            try:
                prod_model_versions = client.get_latest_versions(model_name, stages=["Production"])
            except:
                prod_model_versions = []

            # Change their tags to "Archived"
            for prod_model_version in prod_model_versions:
                print(f"Archiving model: {prod_model_version}")
                client.transition_model_version_stage(
                    name=prod_model_version.name,
                    version=prod_model_version.version,
                    stage="Archived"
                )

            print(f"Promoting model: {model_name} as version: {model_version}")
            client.transition_model_version_stage(
                name=model_name,
                version=model_version,
                stage="Production"
            )
    @task
    def check_final_model(data_files, models = models_mod):
        models_comparators = []
        for reg_model in models:
            models_comparators.append(f"models:/{reg_model}/latest")

        #reg_model_latest = f"models:/{reg_model}/latest"
        #reg_model_latest = f"models:/{reg_model}/latest"
        x_test = pd.read_csv(data_files['transformed_x_test_file'])
        y_test = pd.read_csv(data_files['transformed_y_test_file']).values

        # First check if the requested model and version exist
        i = 0
        val = 0
        for x in range(len(models_comparators)):
            new_model = mlflow.pyfunc.load_model(model_uri=models_comparators[x])
            new_y_pred = get_predictions(x_test, new_model)
            new_val_accuracy, _ = get_val_performance(y_test, new_y_pred)
            logger.info(f'New Model: {new_val_accuracy}')
            if new_val_accuracy >= val:
                val = new_val_accuracy
                i=x
                logger.info(f"Restraremos una nueva version del modelo {models[i]}.")
            else:
                logger.info(f"El nuevo modelo {models[i]} no presenta un mejor performance.")
        
        with open(PATH_DIR_DATA_OUTPUT_MODEL + '/model.txt', 'w') as file:
            file.write(models_comparators[i])

        return 'stopfinal'


    # Execution nodes
    node_start = EmptyOperator(task_id='Starting_the_process', retries=1)  
    
    end_node = EmptyOperator(task_id="end", trigger_rule="all_done")
    node_read_data = read_data(_data_files_)
    node_homologate_data = homologate_values(_data_files_)
    node_split_data = split_train_test(_data_files_)
    node_preprocess_data = preprocess_data(_data_files_)
    node_train_model1 = train_model1(
        _data_files_,
        experiment_name = experiment_name1,
        model_name = model_name1,
        track_cv_performance = True)
    
    node_check_registered_model1 = BranchPythonOperator(
        task_id='check_registered_model1',
        python_callable=check_registered_model,
        op_kwargs={
            'data_files': _data_files_,
            'reg_model': model_name1,
            'new_model_uri': node_train_model1['model_uri'],
            'register_model':'register_model1',
            'register_model_by_comparison':'register_model_by_comparison1',
            'stop':'stop1'

            
        }
    )

    node_register_model1 = PythonOperator(
        task_id='register_model1',
        
        python_callable=register_model,
        op_kwargs={
            'model_name': model_name1,
            'model_uri': node_train_model1['model_uri']
        }
    )

    node_register_model_by_comparison1 = PythonOperator(
        task_id='register_model_by_comparison1',
        python_callable=register_model_by_comparison,
        op_kwargs={
            'model_name': model_name1,
            'model_uri': node_train_model1['model_uri'],
            'push_to_production': True,
        },
    )
    node_train_model2 = train_model2(
            _data_files_,
            experiment_name = experiment_name2,
            model_name = model_name2,
            track_cv_performance = True)
    
    node_check_registered_model2 = BranchPythonOperator(
        task_id='check_registered_model2',
        python_callable=check_registered_model,
        op_kwargs={
            'data_files': _data_files_,
            'reg_model': model_name1,
            'new_model_uri': node_train_model2['model_uri'],
                        'register_model':'register_model2',
            'register_model_by_comparison':'register_model_by_comparison2',
            'stop':'stop2'
        }
    )

    node_register_model2 = PythonOperator(
        task_id='register_model2',
        
        python_callable=register_model,
        op_kwargs={
            'model_name': model_name2,
            'model_uri': node_train_model2['model_uri']
        }
    )

    node_register_model_by_comparison2 = PythonOperator(
        task_id='register_model_by_comparison2',
        python_callable=register_model_by_comparison,
        op_kwargs={
            'model_name': model_name2,
            'model_uri': node_train_model2['model_uri'],
            'push_to_production': True,
        },
    )

    node_stop1 = EmptyOperator(
        task_id='stop1',
        trigger_rule="all_done",
    )
    node_stop2 = EmptyOperator(
        task_id='stop2',
        trigger_rule="all_done",
    )

    node_stopfinal = EmptyOperator(
        task_id='wait_models',
        trigger_rule="all_done",
    )

    node_model_final = check_final_model(_data_files_, models = models_mod)

    node_start >> node_read_data >> \
        node_homologate_data>> \
    node_split_data >> node_preprocess_data >> \
            node_train_model1 >> node_check_registered_model1 >> \
                [node_register_model1, 
                 node_register_model_by_comparison1, 
                 node_stop1] >>  node_stopfinal
    
    node_preprocess_data >> node_train_model2 >> node_check_registered_model2 >> \
                [node_register_model2, 
                 node_register_model_by_comparison2, 
                 node_stop2]  >> node_stopfinal
    
    node_stopfinal >> node_model_final >> end_node


etl_dag = mydag()
