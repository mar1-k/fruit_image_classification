import os
import concurrent.futures
from functools import partial
from datetime import datetime

from airflow.decorators import task
from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
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

# --- Python function for the parallel upload task ---
def _upload_directory_parallel(local_directory, s3_prefix, bucket_name, conn_id, max_workers=20):
    """
    Recursively walks a local directory and uploads all files to S3 in parallel
    using a thread pool.
    """
    s3_hook = S3Hook(aws_conn_id=conn_id)
    print(f"Starting parallel upload of directory '{local_directory}' to 's3://{bucket_name}/{s3_prefix}' with {max_workers} workers.")

    upload_worker = partial(
        s3_hook.load_file, bucket_name=bucket_name, replace=True
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for root, _, files in os.walk(local_directory):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, local_directory)
                s3_key = os.path.join(s3_prefix, relative_path).replace("\\", "/")
                futures.append(
                    executor.submit(upload_worker, filename=local_path, key=s3_key)
                )
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Upload failed with an exception: {e}")
                raise e
    print(f"Parallel upload of {len(futures)} files complete.")


# --- Branching Task Functions ---

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
    description='A DAG to ingest data and trigger the preprocessing pipeline.',
    schedule_interval='@once',
    start_date=datetime(2025, 8, 5),
    is_paused_upon_creation=False,
    catchup=False,
    tags=['pipeline-starter', 'ingestion', 's3'],
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
    )
    
    upload_to_s3_task = PythonOperator(
        task_id='upload_images_to_s3',
        python_callable=_upload_directory_parallel,
        op_kwargs={
            'local_directory': LOCAL_UNZIPPED_PATH,
            's3_prefix': S3_DEST_KEY,
            'bucket_name': S3_BUCKET,
            'conn_id': S3_CONN_ID,
            'max_workers': 20
        },
        trigger_rule='one_success'
    )

    trigger_preprocessing_dag = TriggerDagRunOperator(
        task_id="trigger_preprocessing_dag",
        trigger_dag_id="preprocess_generate_class_mapping",
        # This task should run if any of its parents succeed, making it the final join point.
        trigger_rule='one_success'
    )

    # --- Set Task Dependencies (Corrected) ---
    check_s3_task >> [skip_ingestion_task, check_local_dir_task]
    
    check_local_dir_task >> [upload_to_s3_task, check_local_zip_task]
    
    check_local_zip_task >> [unzip_task, download_task]
    
    download_task >> unzip_task
    
    unzip_task >> upload_to_s3_task
    
    # now lead to the final trigger task.
    skip_ingestion_task >> trigger_preprocessing_dag
    upload_to_s3_task >> trigger_preprocessing_dag
