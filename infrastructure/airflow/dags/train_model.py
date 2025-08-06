from __future__ import annotations
import pendulum
from airflow.decorators import task, dag
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.models.param import Param
from airflow.hooks.base import BaseHook
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

# --- Define Constants ---
DOCKER_IMAGE = "my-pytorch-trainer:latest"
MLFLOW_CONN_ID = "mlflow_default" # Assumes you have an MLflow connection in Airflow
REGISTERED_MODEL_NAME = "fruit-classifier"

# --- Task to Register the Model ---
@task
def register_model(mlflow_conn_id: str, experiment_name: str, run_name: str, registered_model_name: str):
    """
    Finds a specific run by its name and registers its model artifact to the
    MLflow Model Registry. This task is idempotent.
    """
    print(f"Searching for run with name: '{run_name}' in experiment: '{experiment_name}'")
    
    # THE FIX IS HERE: We get the connection URI from Airflow and instantiate the client directly.
    # This avoids any provider-specific hook issues.
    mlflow_conn = BaseHook.get_connection(mlflow_conn_id)
    tracking_uri = mlflow_conn.host
    client = MlflowClient(tracking_uri=tracking_uri)
    
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found.")
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'"
    )
    
    if not runs:
        raise ValueError(f"Run with name '{run_name}' not found in experiment '{experiment_name}'.")
        
    run_id = runs[0].info.run_id
    model_uri = f"runs:/{run_id}/model"
    
    print(f"Found run: {run_id}")
    print(f"Registering model '{registered_model_name}' from URI: {model_uri}")
    
    try:
        client.create_registered_model(name=registered_model_name)
        print(f"Created new registered model: '{registered_model_name}'")
    except MlflowException as e:
        if "RESOURCE_ALREADY_EXISTS" in str(e):
            print(f"Registered model '{registered_model_name}' already exists.")
        else:
            raise e

    client.create_model_version(
        name=registered_model_name,
        source=model_uri,
        run_id=run_id,
        description=f"Model from run {run_name}"
    )
    print("Model version created and registered successfully.")


# --- Define the DAG ---
@dag(
    dag_id="train_model",
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    schedule=None,
    catchup=False,
    is_paused_upon_creation=False,
    params={
        "s3_bucket": Param("data", type="string"),
        "data_prefix": Param("dataset/", type="string"),
        "num_epochs": Param(5, type="integer"),
        "learning_rate": Param(0.001, type="number"),
        "batch_size": Param(32, type="integer"),
        "class_mapping": Param({}, type=["object", "null"]),
    },
    tags=["training", "ml", "docker", "mlflow"],
)
def train_and_register_dag():
    
    class_mapping_from_xcom = "{{ dag_run.conf.get('class_mapping', '{}') | tojson }}"
    run_name = "training-run-{{ run_id }}"
    experiment_name = "Fruit Classification"

    train_model_task = DockerOperator(
        task_id="train_model",
        image=DOCKER_IMAGE,
        command="python -u train.py",
        environment={
            "PYTHONUNBUFFERED": "1",
            "S3_ENDPOINT_URL": "http://minio:9000",
            "S3_ACCESS_KEY": "minio_user",
            "S3_SECRET_KEY": "minio_password",
            "S3_BUCKET": "{{ params.s3_bucket }}",
            "DATA_PREFIX": "{{ params.data_prefix }}",
            "MLFLOW_TRACKING_URI": "http://mlflow-server:5000",
            "MLFLOW_EXPERIMENT_NAME": experiment_name,
            "MLFLOW_RUN_NAME": run_name,
            "NUM_EPOCHS": "{{ params.num_epochs }}",
            "LEARNING_RATE": "{{ params.learning_rate }}",
            "BATCH_SIZE": "{{ params.batch_size }}",
            "CLASS_MAPPING": class_mapping_from_xcom,
        },
        device_requests=[{"driver": "nvidia", "count": -1, "capabilities": [["gpu"]] }],
        shm_size='6g',
        auto_remove='success',
        docker_url="unix://var/run/docker.sock",
        network_mode="infrastructure_default",
        # THE FIX IS HERE: The tty parameter has been removed.
        # This forces the container into a non-interactive mode that is
        # more reliable for log streaming in Airflow.
    )

    registration_task = register_model(
        mlflow_conn_id=MLFLOW_CONN_ID,
        experiment_name=experiment_name,
        run_name=run_name,
        registered_model_name=REGISTERED_MODEL_NAME
    )

    train_model_task >> registration_task

# Instantiate the DAG
train_and_register_dag()
