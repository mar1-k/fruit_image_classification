import os
import json
import torch
import boto3
import mlflow
import mlflow.pytorch
import logging
import concurrent.futures
from functools import partial
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
# NEW: Import scikit-learn for F1 score calculation
from sklearn.metrics import f1_score

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        'mlflow_run_name': os.environ.get('MLFLOW_RUN_NAME'), # New: Specific run name
        'num_epochs': int(os.environ.get('NUM_EPOCHS', 5)),
        'batch_size': int(os.environ.get('BATCH_SIZE', 32)),
        'learning_rate': float(os.environ.get('LEARNING_RATE', 0.001)),
    }
    required_vars = ['s3_endpoint_url', 's3_access_key', 's3_secret_key', 's3_bucket', 'data_prefix', 'class_mapping_str', 'mlflow_tracking_uri']
    if not all(config[var] for var in required_vars):
        raise ValueError("One or more required environment variables are not set.")
    
    config['class_mapping'] = json.loads(config['class_mapping_str'])
    return config

# --- Parallel S3 Download Function (unchanged) ---
def download_s3_directory_parallel(s3_client, bucket_name, s3_prefix, local_dir, max_workers=20):
    # This function is unchanged.
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
            key = future_to_key[future]
            try:
                future.result()
            except Exception as exc:
                logging.error(f'{key} generated an exception: {exc}')
                raise
    logging.info(f"Parallel download of {len(keys_to_download)} files complete.")


# --- Model Definition (unchanged) ---
def create_model(num_classes: int):
    # This function is unchanged.
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    classifier_input_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=True), nn.Linear(classifier_input_features, num_classes))
    logging.info(f"Model created. Number of classes: {num_classes}")
    return model

# --- Data Loading (unchanged) ---
def get_dataloaders(local_data_path: str, transform: transforms.Compose, batch_size: int):
    # This function is unchanged.
    logging.info(f"Loading datasets from local path: {local_data_path}")
    train_dataset = datasets.ImageFolder(root=os.path.join(local_data_path, 'train'), transform=transform)
    val_dataset = datasets.ImageFolder(root=os.path.join(local_data_path, 'val'), transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    logging.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    return train_loader, val_loader

# --- Training and Evaluation Loop (MODIFIED) ---
def train(model, train_loader, val_loader, epochs, learning_rate, device):
    """
    Runs the training and validation loop, now with explicit metric logging for MLflow.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate)
    model.to(device)
    
    for epoch in range(epochs):
        # Training phase
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

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                # Calculate validation loss
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Collect predictions and labels for F1 score
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate and log metrics for the epoch
        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        logging.info(f'Epoch {epoch + 1} - Validation Loss: {avg_val_loss:.3f}, Accuracy: {accuracy:.2f}%, F1-Score: {f1:.3f}')
        mlflow.log_metric('validation_loss', avg_val_loss, step=epoch)
        mlflow.log_metric('validation_accuracy', accuracy, step=epoch)
        mlflow.log_metric('validation_f1_score', f1, step=epoch)


# --- Main Execution (unchanged) ---
if __name__ == "__main__":
    try:
        config = get_config()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {device}")

        mlflow.set_tracking_uri(config['mlflow_tracking_uri'])
        mlflow.set_experiment(config['mlflow_experiment_name'])
        mlflow.pytorch.autolog()

        with mlflow.start_run(run_name=config.get('mlflow_run_name')) as run:
            logging.info(f"Started MLflow run: {run.info.run_id}")
            
            mlflow.log_param('num_classes', len(config['class_mapping']))

            s3_client = boto3.client('s3', endpoint_url=config['s3_endpoint_url'], aws_access_key_id=config['s3_access_key'], aws_secret_access_key=config['s3_secret_key'])
            local_data_dir = "/tmp/data"
            download_s3_directory_parallel(s3_client, config['s3_bucket'], config['data_prefix'], local_data_dir)

            model = create_model(len(config['class_mapping']))
            data_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            train_loader, val_loader = get_dataloaders(local_data_dir, data_transform, config['batch_size'])

            train(model, train_loader, val_loader, config['num_epochs'], config['learning_rate'], device)

            logging.info("Training finished. Autologging has saved the model.")

    except Exception as e:
        logging.error(f"An error occurred during the training process: {e}", exc_info=True)
        raise
