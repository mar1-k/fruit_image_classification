from datetime import datetime
from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator
from airflow.providers.amazon.aws.transfers.local_to_s3 import LocalFilesystemToS3Operator

# Define the default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

# Define the DAG
with DAG(
    dag_id='ingest_fruit_images_to_s3',
    default_args=default_args,
    description='A DAG to download fruit images from GitHub and upload to MinIO S3',
    schedule_interval='@once',
    start_date=datetime(2025, 8, 5), # Use a static start date
    is_paused_upon_creation=False,
    catchup=False,
    tags=['data_engineering', 'ingestion', 's3', 'minio', 'images'],
) as dag:
    # Define the local data path within the Airflow container
    LOCAL_DATA_PATH = '/opt/airflow/data'
    ZIP_FILE_NAME = 'dataset.zip'
    LOCAL_ZIP_PATH = f"{LOCAL_DATA_PATH}/{ZIP_FILE_NAME}"

    # Define the BashOperator to create the directory and download the file
    # The -L flag is important to follow redirects from GitHub.
    download_task = BashOperator(
        task_id='download_dataset_from_github',
        bash_command=(
            f'mkdir -p {LOCAL_DATA_PATH} && '
            f'curl -L -o {LOCAL_ZIP_PATH} '
            'https://github.com/mar1-k/fruit_image_classification/raw/refs/heads/main/data/dataset.zip'
        )
    )

    # Define the BashOperator to unzip the file
    # The -o flag overwrites files without prompting.
    unzip_task = BashOperator(
        task_id='unzip_dataset',
        bash_command=f'unzip -o {LOCAL_ZIP_PATH} -d {LOCAL_DATA_PATH}/'
    )

    # Note: We assume the zip file contains a single top-level directory (e.g., 'dataset').
    # This task will upload that entire directory and its contents recursively.
    # The path to upload is '/opt/airflow/data/dataset' if the zip extracts to a 'dataset' folder.
    # If the zip extracts its contents directly, change the filename to '/opt/airflow/data/'
    upload_to_s3_task = LocalFilesystemToS3Operator(
        task_id='upload_images_to_minio',
        aws_conn_id='minio_conn',       # The connection ID for your MinIO instance
        dest_key='data/',               # The destination folder/prefix in your bucket
        filename=f'{LOCAL_DATA_PATH}/dataset', # Path to the local directory to upload
        replace=True,                   # Overwrite files in S3 if they exist
    )

    # Set the task dependencies
    download_task >> unzip_task >> upload_to_s3_task