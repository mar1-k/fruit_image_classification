import os
import json
import torch
import boto3
import mlflow
import mlflow.pytorch
import logging
import sys # Import the sys module
import concurrent.futures
from functools import partial
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.metrics import f1_score

# --- THE FIX IS HERE: Setup Verbose, Unbuffered Logging ---
# Get the root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Remove any existing handlers to prevent conflicts with MLflow's handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Create a new handler that streams directly to standard output
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)

# Create a formatter and add it to the handler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Add the new, unbuffered handler to the root logger
logger.addHandler(handler)


# --- Configuration from Environment Variables ---
def get_config():
    """Reads configuration from environment variables."""
    config = {
        's3_endpoint_url': os.environ.get('S3_ENDPOINT_URL'),
        's3_access_key': os.environ.get('S3_ACCESS_KEY'),
        's3_secret_key': os.environ.get('S3_SECRET_KEY'),
        's3_bucket': os.environ.get('S3_BUCKET'),
        'data_prefix': os.environ.get('DATA_PREFIX'),
        'class_mapping_str': os.environ.get('CLASS_MAPPING'),
        'mlflow_tracking_uri': os.environ.get('MLFLOW_TRACKING_URI'),
        'mlflow_experiment_name': os.environ.get('MLFLOW_EXPERIMENT_NAME', 'Fruit Classification'),
        'mlflow_run_name': os.environ.get('MLFLOW_RUN_NAME'),
        'num_epochs': int(os.environ.get('NUM_EPOCHS', 5)),
        'batch_size': int(os.environ.get('BATCH_SIZE', 32)),
        'learning_rate': float(os.environ.get('LEARNING_RATE', 0.001)),
    }
    required_vars = ['s3_endpoint_url', 's3_access_key', 's3_secret_key', 's3_bucket', 'data_prefix', 'class_mapping_str', 'mlflow_tracking_uri']
    if not all(config[var] for var in required_vars):
        raise ValueError("One or more required environment variables are not set.")
    
    config['class_mapping'] = json.loads(config['class_mapping_str'])
    return config

# --- Parallel S3 Download Function ---
def download_s3_directory_parallel(s3_client, bucket_name, s3_prefix, local_dir, max_workers=20):
    """Recursively downloads a directory from S3 in parallel."""
    logging.info(f"Starting parallel download from s3://{bucket_name}/{s3_prefix} to {local_dir} with {max_workers} workers.")
    keys_to_download = []
    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
        if "Contents" in page:
            for obj in page["Contents"]:
                if not obj["Key"].endswith('/'):
                    keys_to_download.append(obj["Key"])
    def download_worker(s3_key):
        relative_path = os.path.relpath(s3_key, s3_prefix)
        local_file_path = os.path.join(local_dir, relative_path)
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        s3_client.download_file(bucket_name, s3_key, local_file_path)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_key = {executor.submit(download_worker, key): key for key in keys_to_download}
        for future in concurrent.futures.as_completed(future_to_key):
            try:
                future.result()
            except Exception as exc:
                logging.error(f'{future_to_key[future]} generated an exception: {exc}')
                raise
    logging.info(f"Parallel download of {len(keys_to_download)} files complete.")

# --- Model Definition ---
def create_model(num_classes: int):
    """Creates a pre-trained EfficientNet model."""
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    classifier_input_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=True), nn.Linear(classifier_input_features, num_classes))
    logging.info(f"Model created: efficientnet_b0. Number of classes: {num_classes}")
    return model

# --- Data Loading ---
def get_dataloaders(local_data_path: str, transform: transforms.Compose, batch_size: int):
    """Creates DataLoaders for training and validation from a LOCAL path."""
    logging.info(f"Loading datasets from local path: {local_data_path}")
    train_dataset = datasets.ImageFolder(root=os.path.join(local_data_path, 'train'), transform=transform)
    val_dataset = datasets.ImageFolder(root=os.path.join(local_data_path, 'val'), transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    logging.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    return train_loader, val_loader

# --- Training and Evaluation Loop ---
def train(model, train_loader, val_loader, epochs, learning_rate, device):
    """Runs the training and validation loop with explicit metric logging."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate)
    model.to(device)
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        logging.info(f'[Epoch {epoch + 1}] Training Loss: {avg_train_loss:.3f}')
        mlflow.log_metric('train_loss', avg_train_loss, step=epoch)

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        logging.info(f'Epoch {epoch + 1} - Validation Loss: {avg_val_loss:.3f}, Accuracy: {accuracy:.2f}%, F1-Score: {f1:.3f}')
        mlflow.log_metric('validation_loss', avg_val_loss, step=epoch)
        mlflow.log_metric('validation_accuracy', accuracy, step=epoch)
        mlflow.log_metric('validation_f1_score', f1, step=epoch)

# --- Main Execution ---
if __name__ == "__main__":
    try:
        config = get_config()

        os.environ['AWS_ACCESS_KEY_ID'] = config['s3_access_key']
        os.environ['AWS_SECRET_ACCESS_KEY'] = config['s3_secret_key']
        os.environ['MLFLOW_S3_ENDPOINT_URL'] = config['s3_endpoint_url']
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {device}")
        local_data_dir = "/app/dataset/"

        mlflow.set_tracking_uri(config['mlflow_tracking_uri'])
        mlflow.set_experiment(config['mlflow_experiment_name'])
        mlflow.pytorch.autolog(log_models=True, disable=True, exclusive=True)

        with mlflow.start_run(run_name=config.get('mlflow_run_name')) as run:
            logging.info(f"Started MLflow run: {run.info.run_id}")
            
            logging.info("Logging parameters to MLflow...")
            mlflow.log_param('model_name', 'efficientnet_b0')
            mlflow.log_param('num_classes', len(config['class_mapping']))
            mlflow.log_param('num_epochs', config['num_epochs'])
            mlflow.log_param('batch_size', config['batch_size'])
            mlflow.log_param('learning_rate', config['learning_rate'])
            
            # THE FIX IS HERE: Check for the pre-loaded data directory before downloading.
            if os.path.isdir(local_data_dir):
                logging.info(f"Found pre-loaded data at {local_data_dir}. Skipping S3 download.")
            else:
                logging.warning(f"Local data not found at {local_data_dir}. Attempting to download from S3.")
                s3_client = boto3.client('s3', endpoint_url=config['s3_endpoint_url'])
                download_s3_directory_parallel(s3_client, config['s3_bucket'], config['data_prefix'], local_data_dir)

            model = create_model(len(config['class_mapping']))
            data_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            train_loader, val_loader = get_dataloaders(local_data_dir, data_transform, config['batch_size'])

            train(model, train_loader, val_loader, config['num_epochs'], config['learning_rate'], device)

            logging.info("Training finished. Model artifacts have been automatically saved to the MLflow artifact store (S3).")

    except Exception as e:
        logging.error(f"An error occurred during the training process: {e}", exc_info=True)
        raise
