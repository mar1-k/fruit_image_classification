# A simple Airflow DAG to check GPU availability.

from datetime import datetime

from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.decorators import task
from airflow.utils.edgemodifier import Label

# Docker library from PIP
import docker

# Simple DAG
with DAG(
    "gpu_test", 
    schedule_interval="@daily", 
    start_date=datetime(2022, 1, 1), 
    catchup=False, 
    tags=['test']
) as dag:


    @task(task_id='check_gpu')
    def start_gpu_container(**kwargs):

         # get the docker params from the environment
         client = docker.from_env()
          
         # run the container
         response = client.containers.run(

             # The container you wish to call
             'pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime',

             # The command to run inside the container
             'nvidia-smi',

             # Passing the GPU access
             device_requests=[
                 docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])
             ]
         )

         return str(response)

    check_gpu = start_gpu_container()


    # Dummy functions
    start = DummyOperator(task_id='start')
    end   = DummyOperator(task_id='end')


    # Create a simple workflow
    start >> check_gpu >> end