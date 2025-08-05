from __future__ import annotations
import pendulum
from airflow.models.dag import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.models.param import Param

# Import the Mount type from the docker library
# This is no longer needed but kept for reference
from docker.types import Mount

# Define the Docker image to use for training
DOCKER_IMAGE = "my-pytorch-trainer:latest"

with DAG(
    dag_id="train_model",
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    schedule=None,
    catchup=False,
    doc_md="""
    ### Model Training DAG
    This DAG trains a fruit classification model using data from S3.
    It is triggered after the preprocessing DAG successfully generates the class mapping.
    """,
    params={
        "s3_bucket": Param("data", type="string"),
        "data_prefix": Param("dataset/", type="string"),
        "model_prefix": Param("models/", type="string"),
        "num_epochs": Param(5, type="integer", minimum=1),
        "learning_rate": Param(0.001, type="number"),
        "batch_size": Param(32, type="integer"),
        "class_mapping": Param({}, type=["object", "null"]),
    },
    tags=["training", "ml", "docker"],
) as dag:
    
    # THE FIX IS HERE: We apply the `tojson` filter to ensure the dictionary
    # is passed as a valid JSON string.
    class_mapping_from_xcom = "{{ dag_run.conf.get('class_mapping', '{}') | tojson }}"

    train_model_task = DockerOperator(
        task_id="train_model",
        image=DOCKER_IMAGE,
        # The mounts and mount_tmp_dir parameters are removed
        # because the script is now part of the image.
        command="python train.py", # The command is simpler as we are already in the /app working dir
        environment={
            "S3_ENDPOINT_URL": "http://minio:9000",
            "S3_ACCESS_KEY": "minio_user",
            "S3_SECRET_KEY": "minio_password",
            "S3_BUCKET": "{{ params.s3_bucket }}",
            "DATA_PREFIX": "{{ params.data_prefix }}",
            "MODEL_PREFIX": "{{ params.model_prefix }}",
            "NUM_EPOCHS": "{{ params.num_epochs }}",
            "LEARNING_RATE": "{{ params.learning_rate }}",
            "BATCH_SIZE": "{{ params.batch_size }}",
            "CLASS_MAPPING": class_mapping_from_xcom,
        },
        device_requests=[
            {
                "driver": "nvidia",
                "count": -1,
                "capabilities": [["gpu"]],
            }
        ],
        working_dir="/app",
        auto_remove='success',
        docker_url="unix://var/run/docker.sock",
        network_mode="infrastructure_default",
    )
