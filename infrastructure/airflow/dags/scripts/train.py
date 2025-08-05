import os
import json
import torch
import boto3
import logging
import concurrent.futures
from functools import partial
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

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
        'model_prefix': os.environ.get('MODEL_PREFIX', 'models/'),
        'class_mapping_str': os.environ.get('CLASS_MAPPING'),
        'num_epochs': int(os.environ.get('NUM_EPOCHS', 5)),
        'batch_size': int(os.environ.get('BATCH_SIZE', 32)),
        'learning_rate': float(os.environ.get('LEARNING_RATE', 0.001)),
    }
    if not all([config['s3_endpoint_url'], config['s3_access_key'], config['s3_secret_key'], config['s3_bucket'], config['data_prefix'], config['class_mapping_str']]):
        raise ValueError("One or more required environment variables are not set.")
    
    config['class_mapping'] = json.loads(config['class_mapping_str'])
    return config

# --- NEW: Function for parallel S3 download ---
def download_s3_directory_parallel(s3_client, bucket_name, s3_prefix, local_dir, max_workers=8):
    """Recursively downloads a directory from S3 in parallel using a thread pool."""
    logging.info(f"Starting parallel download from s3://{bucket_name}/{s3_prefix} to {local_dir} with {max_workers} workers.")
    
    # Get a list of all objects to download
    keys_to_download = []
    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
        if "Contents" in page:
            for obj in page["Contents"]:
                if not obj["Key"].endswith('/'): # Exclude folder placeholders
                    keys_to_download.append(obj["Key"])

    # Define a worker function to download a single file
    def download_worker(s3_key):
        relative_path = os.path.relpath(s3_key, s3_prefix)
        local_file_path = os.path.join(local_dir, relative_path)
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        s3_client.download_file(bucket_name, s3_key, local_file_path)

    # Use a ThreadPoolExecutor to download files in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a future for each download task
        future_to_key = {executor.submit(download_worker, key): key for key in keys_to_download}
        
        for future in concurrent.futures.as_completed(future_to_key):
            key = future_to_key[future]
            try:
                future.result() # Check for exceptions
            except Exception as exc:
                logging.error(f'{key} generated an exception: {exc}')
                raise
    
    logging.info(f"Parallel download of {len(keys_to_download)} files complete.")


# --- Model Definition ---
def create_model(num_classes: int):
    """Creates a pre-trained EfficientNet model with a new classifier head."""
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    classifier_input_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(classifier_input_features, num_classes),
    )
    logging.info(f"Model created. Number of classes: {num_classes}")
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
    """Runs the training and validation loop."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate)
    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                logging.info(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        logging.info(f'Epoch {epoch + 1} - Validation Accuracy: {accuracy:.2f}%')

# --- Main Execution ---
if __name__ == "__main__":
    try:
        # 1. Configuration
        config = get_config()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {device}")

        # 2. Setup S3 client and local data directory
        s3_client = boto3.client(
            's3',
            endpoint_url=config['s3_endpoint_url'],
            aws_access_key_id=config['s3_access_key'],
            aws_secret_access_key=config['s3_secret_key']
        )
        local_data_dir = "/tmp/data"

        # 3. Download data from S3 (now in parallel)
        download_s3_directory_parallel(s3_client, config['s3_bucket'], config['data_prefix'], local_data_dir)

        # 4. Model
        num_classes = len(config['class_mapping'])
        model = create_model(num_classes)
        
        # 5. Data Loaders
        data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        train_loader, val_loader = get_dataloaders(local_data_dir, data_transform, config['batch_size'])

        # 6. Training
        train(model, train_loader, val_loader, config['num_epochs'], config['learning_rate'], device)

        # 7. Save and Upload Model
        logging.info("Training finished. Saving model...")
        model_save_path = "trained_model.pth"
        torch.save(model.state_dict(), model_save_path)
        model_s3_key = f"{config['model_prefix']}fruit_classifier_v1.pth"
        s3_client.upload_file(model_save_path, config['s3_bucket'], model_s3_key)
        logging.info(f"Model successfully uploaded to s3://{config['s3_bucket']}/{model_s3_key}")

    except Exception as e:
        logging.error(f"An error occurred during the training process: {e}", exc_info=True)
        raise
