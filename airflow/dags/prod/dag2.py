import os
from datetime import datetime, timedelta

from airflow.decorators import dag, task
from airflow.operators.empty import EmptyOperator


default_args = {
    'owner': 'datapath',
    # 'retries': 5,
    # 'retry_delay': timedelta(minutes=5)
}

@dag(
    dag_id='dag_2', 
    default_args=default_args, 
    start_date=datetime(2023, 6, 28), 
    schedule_interval='@once',
    catchup=False,
    schedule=None,
)
def hello_world_etl():

    node_start = EmptyOperator(task_id='Starting_the_process', retries=1)  
    node_end = EmptyOperator(task_id="end", trigger_rule="all_done")

    @task(multiple_outputs=True)
    def get_name():
        return {'first_name': 'Jerry', 'last_name': 'Fridman'}

    @task()
    def get_age():
        return 19

    @task()
    def greet(first_name, last_name, age):
        print(f"Hello World! My name is {first_name} {last_name} "
              f"and I am {age} years old!")
    
    name_dict = get_name()
    age = get_age()
    result = greet(
        first_name=name_dict['first_name'], 
        last_name=name_dict['last_name'],
        age=age
    )

    node_start >> name_dict
    result >> node_end

greet_dag = hello_world_etl()