import joblib
from model_definition import create_model
from logging_config import setup_logging
setup_logging()
import logging

def train_model(combined_features):
    """Train the model using the provided combined features."""
    
    logging.info("Starting model training.")
    
    
    # Define input shape based on the combined features.
    if len(combined_features.shape) < 3:
        logging.error("Combined features must have at least 3 dimensions.")
        raise ValueError("Combined features must have at least 3 dimensions.")
    input_shape = (combined_features.shape[1], combined_features.shape[2])
    model = create_model(input_shape)
    
    # Fit the model
    logging.info(f"Training model with input shape: {input_shape}.")
    model.fit(combined_features, epochs=100, batch_size=32, verbose=2)
    
    # Save the model
    joblib.dump(model, 'trained_model.pkl')
    logging.info("Model training completed and saved.")
    
    
    return model
