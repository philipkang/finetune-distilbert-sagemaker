README.MD
This project demonstrates the process of fine-tuning a DistilBERT model for multi-class text classification using Amazon SageMaker. The process can be summarized in the following steps:

Data Preparation and Model Definition
The script.py file defines the core components:
•	Data loading from an S3 bucket
•	Custom NewsDataset class for handling the text data
•	DistilBERTClass model architecture
•	Training and validation functions
The model architecture includes:
•	Pre-trained DistilBERT base
•	Additional linear layers for classification
•	Dropout for regularization

Training Setup
The TrainingNotebook.ipynb sets up the SageMaker training job:
•	Installs required libraries
•	Configures the SageMaker session and role
•	Creates a HuggingFace estimator with specified hyperparameters
Key hyperparameters include:
•	Number of epochs: 2
•	Training batch size: 4
•	Validation batch size: 2
•	Learning rate: 1e-05
The training job is configured to use:
•	A single ml.p2.xlarge instance
•	PyTorch 1.8 and Transformers 4.6

Model Training
The training process in script.py includes:
•	Splitting data into training and validation sets
•	Creating DataLoader objects for efficient batching
•	Training loop with loss calculation and optimization
•	Validation loop for performance evaluation
The model is trained to classify text into four categories:
•	Business, Science, Entertainment, and Health

Model Deployment
After training, the Deployment.ipynb notebook handles model deployment:
•	Creates a HuggingFaceModel object with the trained model artifacts
•	Deploys the model to a SageMaker endpoint (ml.m5.xlarge instance)
The inference.py script defines functions for:
•	Loading the model
•	Processing input data
•	Making predictions
•	Formatting output

Inference
The deployed model can be used for real-time inference:
•	Input text is sent to the endpoint
•	The model returns the predicted category and probabilities

Load Testing
•	Generate load testing benchmarking data with load-test.py
•	Create new load testing with SageMaker Inference Recommender

API Access
Enable API access via API Gateway trigged a Lambda Function 
•	Create Lambda Function ‘llm-endpoint-invoke-function’
•	Create API Gateway to integrate request to invoke the lambda


This fine-tuning process leverages SageMaker's capabilities to streamline the training, deployment, and inference stages of developing a custom text classification model based on DistilBERT.
