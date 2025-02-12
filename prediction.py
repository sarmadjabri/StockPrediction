import joblib
import pandas as pd
from logging_config import setup_logging
setup_logging()
import logging

def load_model(model_path):
    """Load the model from the specified path."""
    try:
        logging.info(f"Loading model from {model_path}.")
        model = joblib.load(model_path)
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading model from {model_path}: {e}")
        return None

def predict_with_all_features(model, combined_features):
    """Make predictions using the trained model and combined features."""
    logging.info("Making predictions with the model.")
    logging.info(f"Making predictions with combined features of shape: {combined_features.shape}.")
    predictions = model.predict(combined_features)
    logging.info(f"Predictions made: {predictions}")
    return predictions
