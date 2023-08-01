import os, sys
import pytz
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

# Obtener la zona horaria deseada
TZ = os.getenv('TZ')
# TZ="America/Lima"
timezone = pytz.timezone(TZ)
# timezone = pytz.timezone('America/Lima')

default_args = {
    'owner': 'datapath',
    'retries': 5,
    'retry_delay': timedelta(minutes=2)
}

with DAG(
    dag_id='test_dag',
    default_args=default_args,
    description='DAG de prueba, asignamos una zona horaria',
    start_date=datetime(2023, 6, 25, tzinfo=timezone),
    schedule_interval='0 8 * * *', #'@daily'
    catchup=False,
    # schedule=None,
) as dag:
    task1 = BashOperator(
        task_id='first_task',
        bash_command="echo hello world, this is the first task! [$TZ]"
    )

    task2 = BashOperator(
        task_id='second_task',
        bash_command="echo hey, I am task2 and will be running after task1!"
    )

    task3 = BashOperator(
        task_id='thrid_task',
        bash_command="echo hey, I am task3 and will be running after task1 at the same time as task2!"
    )

    # Task dependency method 1
    # task1.set_downstream(task2)
    # task1.set_downstream(task3)

    # Task dependency method 2
    # task1 >> task2
    # task1 >> task3

    # Task dependency method 3
    task1 >> [task2, task3]
