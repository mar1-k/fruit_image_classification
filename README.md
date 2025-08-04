# Fruit Image Classification with PyTorch Vision Models

This is an MLOPs project that uses machine learning to classify images of various fruits. A PyTorch-trained Convolutional Neural Network (CNN) analyzes an image and identifies which fruit it contains from a predefined set of categories.

---

## ✨ Problem Description

The goal of this project is to build an end-to-end MLOps pipeline to implmement a multi-class image classification model. Given an image of a single fruit, the model should accurately predict its type (e.g., Apple, Banana, Cherry). This serves as a practical application of computer vision and deep learning principles.

### Why work on this?
- **Learning Value:** A hands-on way to learn how core MLOps components like Airflow, MLflow, and MinIO interact in a real-world scenario.
- **Practical Application:** Models like this are foundational for real-world systems like automated grocery checkouts, agricultural sorting robots, and dietary tracking apps.
- **Exploring Architectures:** It provides an excellent opportunity to compare different neural network architectures, from simple custom CNNs to complex, pre-trained models like ResNet.


---

## 🏛️ Architecture & Core Components

The environment is composed of several key services that work together to provide a complete MLOps platform.

| Service | Technology | Role |
| :--- | :--- | :--- |
| **Orchestrator** | Apache Airflow | Schedules, runs, and monitors complex data and ML workflows (DAGs). |
| **Experiment Tracker** | MLflow | Logs and tracks ML experiments, including parameters, metrics, and models. |
| **Artifact Store** | MinIO | Provides an S3-compatible object storage for MLflow artifacts (models, plots, etc.). |
| **Metadata DB** | PostgreSQL | Acts as the persistent backend for both Airflow and MLflow metadata. |
| **Containerization** | Docker | Each of the services used to facilitate this pipeline is implemented via docker containers |
| **Cloud Deployment** | Google Cloud | Infrastructure to Google Cloud |
| **Infrastructure as Code** | Terraform | Used to deploy infrastructure to Google Cloud via code|


---


## 💾 Dataset

The model was trained on this **Fruit Classification Dataset** https://www.kaggle.com/datasets/icebearogo/fruit-classification-dataset from Kaggle, which contains 100 different classes of fruits.

The dataset is structured with a clear split for training, validation, and testing, providing a solid foundation for building and evaluating robust classification models.

- **Number of Classes (Fruits):** 100
- **Training Set:** ~400 images per class
- **Validation Set:** 50 images per class
- **Test Set:** 50 images per class
- **Total Images:** ~50,000

Each image is of a single fruit, captured against a plain background, which helps the model focus on the features of the fruit itself.

---

## 🧠 Model Details

Several models were trained and evaluated to find the best-performing architecture for this task. 

---

## 📊 Model Performance and Conclusions

### Model Performance Metrics



### Conclusions


## 📂 Project Structure
