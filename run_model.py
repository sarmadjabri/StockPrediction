import pandas as pd
from data_preparation import prepare_data
from model_training import train_model
from prediction import load_model
import logging
import os

logging.info("Starting the model run process.")
    

# Load your data here (this is a placeholder)
data_file_path = 'sample_data.csv'  # Specify the actual data source
if not os.path.exists(data_file_path):
    logging.error(f"Data file not found: {data_file_path}")
    raise FileNotFoundError(f"Data file not found: {data_file_path}")
    
data = pd.read_csv(data_file_path)
logging.info("Data loaded successfully.")

# Prepare the data
logging.info("Preparing the data.")
prepared_data = prepare_data(data)
logging.info("Data preparation completed successfully.")

# Train the model
logging.info("Starting model training.")
model, history = train_model(prepared_data)
logging.info("Model training completed successfully.")

# Save the model
model_path = 'trained_model.joblib'
logging.info(f"Saving the model to {model_path}.")
joblib.dump(model, model_path)

# Load the model
loaded_model = load_model(model_path)
try:
    loaded_model = load_model(model_path)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise

logging.info("Model training and loading completed successfully.")
