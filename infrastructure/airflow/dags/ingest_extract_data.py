import os
import concurrent.futures
from functools import partial
from datetime import datetime

from airflow.decorators import task
from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

# --- Define Constants for easy configuration ---
S3_CONN_ID = 'minio_conn'
S3_BUCKET = 'data'
S3_DEST_KEY = 'dataset/'

LOCAL_DATA_PATH = '/opt/airflow/data'
ZIP_FILE_NAME = 'dataset.zip'
LOCAL_ZIP_PATH = f"{LOCAL_DATA_PATH}/{ZIP_FILE_NAME}"
LOCAL_UNZIPPED_PATH = f'{LOCAL_DATA_PATH}/dataset'

# --- Define the default arguments for the DAG ---
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

# --- MODIFIED: Python function for the parallel upload task ---
def _upload_directory_parallel(local_directory, s3_prefix, bucket_name, conn_id, max_workers=20):
    """
    Recursively walks a local directory and uploads all files to S3 in parallel
    using a thread pool.
    """
    s3_hook = S3Hook(aws_conn_id=conn_id)
    print(f"Starting parallel upload of directory '{local_directory}' to 's3://{bucket_name}/{s3_prefix}' with {max_workers} workers.")

    # Create a partial function for the worker thread. This pre-fills some arguments
    # to the load_file method, making it easier to call from the thread pool.
    upload_worker = partial(
        s3_hook.load_file, bucket_name=bucket_name, replace=True
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        # Walk the directory to find all files
        for root, _, files in os.walk(local_directory):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, local_directory)
                s3_key = os.path.join(s3_prefix, relative_path).replace("\\", "/")

                # Submit the upload task to the thread pool and store the future
                futures.append(
                    executor.submit(upload_worker, filename=local_path, key=s3_key)
                )

        # Wait for all futures to complete and check for exceptions
        for future in concurrent.futures.as_completed(futures):
            try:
                # result() will be None on success but will raise an exception on failure
                future.result()
            except Exception as e:
                print(f"Upload failed with an exception: {e}")
                # Re-raise the exception to make the Airflow task fail
                raise e
    print(f"Parallel upload of {len(futures)} files complete.")


# --- Branching Task Functions (unchanged) ---

@task.branch(task_id="check_if_data_exists_in_s3")
def check_s3_data_existence(conn_id, bucket, prefix):
    s3_hook = S3Hook(aws_conn_id=conn_id)
    keys = s3_hook.list_keys(bucket_name=bucket, prefix=prefix, max_items=1)
    if keys:
        return 'skip_ingestion_task'
    else:
        return 'check_if_local_dir_exists'

@task.branch(task_id="check_if_local_dir_exists")
def check_local_dir_existence(local_unzipped_path):
    if os.path.isdir(local_unzipped_path):
        return 'upload_images_to_s3'
    else:
        return 'check_if_local_zip_exists'

@task.branch(task_id="check_if_local_zip_exists")
def check_local_zip_existence(local_zip_path):
    if os.path.exists(local_zip_path):
        return 'unzip_dataset'
    else:
        return 'download_dataset_from_github'

# --- Define the DAG ---
with DAG(
    dag_id='ingest_dataset_to_s3',
    default_args=default_args,
    description='A DAG to download and upload fruit images with multiple checks.',
    schedule_interval='@once',
    start_date=datetime(2025, 8, 5),
    is_paused_upon_creation=False,
    catchup=False,
    tags=['data_engineering', 'ingestion', 's3', 'images'],
) as dag:

    # --- Task Definitions ---
    check_s3_task = check_s3_data_existence(conn_id=S3_CONN_ID, bucket=S3_BUCKET, prefix=S3_DEST_KEY)
    skip_ingestion_task = EmptyOperator(task_id='skip_ingestion_task')
    check_local_dir_task = check_local_dir_existence(local_unzipped_path=LOCAL_UNZIPPED_PATH)
    check_local_zip_task = check_local_zip_existence(local_zip_path=LOCAL_ZIP_PATH)

    download_task = BashOperator(
        task_id='download_dataset_from_github',
        bash_command=(
            f'mkdir -p {LOCAL_DATA_PATH} && '
            f'curl -L -o {LOCAL_ZIP_PATH} '
            'https://github.com/mar1-k/fruit_image_classification/raw/refs/heads/main/data/dataset.zip'
        )
    )
    unzip_task = BashOperator(
        task_id='unzip_dataset',
        bash_command=f'unzip -o {LOCAL_ZIP_PATH} -d {LOCAL_DATA_PATH}/',
        trigger_rule='one_success'
    )
    
    # MODIFIED: This task now calls the parallel upload function
    upload_to_s3_task = PythonOperator(
        task_id='upload_images_to_s3',
        python_callable=_upload_directory_parallel, # Changed to the parallel function
        op_kwargs={
            'local_directory': LOCAL_UNZIPPED_PATH,
            's3_prefix': S3_DEST_KEY,
            'bucket_name': S3_BUCKET,
            'conn_id': S3_CONN_ID,
            'max_workers': 20  # Tune this number based on your worker's resources
        },
        trigger_rule='one_success'
    )

    # --- Set Task Dependencies (unchanged) ---
    check_s3_task >> [skip_ingestion_task, check_local_dir_task]
    check_local_dir_task >> [upload_to_s3_task, check_local_zip_task]
    check_local_zip_task >> [unzip_task, download_task]
    download_task >> unzip_task
    unzip_task >> upload_to_s3_task