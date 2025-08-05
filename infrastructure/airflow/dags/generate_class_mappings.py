from __future__ import annotations

import json
from datetime import datetime
from airflow.decorators import task
from airflow.models.dag import DAG
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

# --- Define Constants for the S3 data location ---
S3_CONN_ID = "minio_conn"
S3_BUCKET = "data"
# Path to the training data directory in S3
TRAIN_DATA_PREFIX = "dataset/train/"


@task
def generate_class_mapping(conn_id: str, bucket: str, prefix: str) -> dict:
    """
    Scans an S3 prefix to find subdirectories, treating them as class names,
    and creates a class-to-index mapping, similar to torchvision's ImageFolder.
    The resulting mapping is pushed to XComs automatically.
    """
    print(f"Scanning S3 path: s3://{bucket}/{prefix}")
    s3_hook = S3Hook(aws_conn_id=conn_id)

    # list_prefixes with a delimiter gives us the "subdirectories"
    class_folders = s3_hook.list_prefixes(bucket_name=bucket, prefix=prefix, delimiter="/")

    if not class_folders:
        raise ValueError(f"No class folders found at S3 path: s3://{bucket}/{prefix}")

    # The prefixes look like 'dataset/train/apple/'. We need to extract 'apple'.
    # We sort them to ensure the mapping is consistent every time.
    class_names = sorted([folder.split("/")[-2] for folder in class_folders])

    # Create the mapping
    class_to_idx = {class_name: i for i, class_name in enumerate(class_names)}

    print(f"Found {len(class_names)} classes.")
    print("Class to index mapping created:")
    # Using json.dumps for pretty printing in logs
    print(json.dumps(class_to_idx, indent=4))

    # The dictionary is automatically returned and pushed to XComs by the TaskFlow API
    return class_to_idx


with DAG(
    dag_id="preprocess_generate_class_mapping",
    start_date=datetime(2025, 8, 5),
    schedule_interval="@once",
    catchup=False,
    is_paused_upon_creation=False,
    doc_md="""
    ### Generate Class Mapping DAG

    This DAG scans the S3 directory structure of the training data to create a
    class-to-index mapping. This mapping is essential for the model training
    process and is saved to XComs for downstream tasks.
    """,
    tags=["preprocessing", "ml"],
) as dag:
    # Instantiate the task, passing in our constants
    generate_class_mapping(
        conn_id=S3_CONN_ID, bucket=S3_BUCKET, prefix=TRAIN_DATA_PREFIX
    )