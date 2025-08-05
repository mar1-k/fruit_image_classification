from __future__ import annotations
import pendulum
from airflow.models.dag import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.models.param import Param
from docker.types import Mount

DOCKER_IMAGE = "my-pytorch-trainer:latest"

with DAG(
    dag_id="train_model",
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    schedule=None,
    catchup=False,
    params={
        "s3_bucket": Param("data", type="string"),
        "data_prefix": Param("dataset/", type="string"),
        "num_epochs": Param(5, type="integer"),
        "learning_rate": Param(0.001, type="number"),
        "batch_size": Param(32, type="integer"),
        "class_mapping": Param({}, type=["object", "null"]),
    },
    tags=["training", "ml", "docker", "mlflow"],
) as dag:
    
    class_mapping_from_xcom = "{{ dag_run.conf.get('class_mapping', '{}') | tojson }}"

    train_model_task = DockerOperator(
        task_id="train_model",
        image=DOCKER_IMAGE,
        command="python train.py",
        environment={
            # S3 Credentials
            "S3_ENDPOINT_URL": "http://minio:9000",
            "S3_ACCESS_KEY": "minio_user",
            "S3_SECRET_KEY": "minio_password",
            "S3_BUCKET": "{{ params.s3_bucket }}",
            "DATA_PREFIX": "{{ params.data_prefix }}",
            # MLflow Configuration
            "MLFLOW_TRACKING_URI": "http://mlflow-tracking-server:5000",
            "MLFLOW_EXPERIMENT_NAME": "Fruit Classification",
            # Hyperparameters
            "NUM_EPOCHS": "{{ params.num_epochs }}",
            "LEARNING_RATE": "{{ params.learning_rate }}",
            "BATCH_SIZE": "{{ params.batch_size }}",
            "CLASS_MAPPING": class_mapping_from_xcom,
        },
        device_requests=[{"driver": "nvidia", "count": -1, "capabilities": [["gpu"]] }],
        # This is crucial for PyTorch's DataLoader with num_workers > 0.
        shm_size='6g', # Allocate 6 gigabytes of shared memory
        auto_remove='success',
        docker_url="unix://var/run/docker.sock",
        network_mode="infrastructure_default",
    )
