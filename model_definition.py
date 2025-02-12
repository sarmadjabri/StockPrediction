from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import numpy as np
from logging_config import setup_logging
setup_logging()

def create_model(input_shape, units=50, dropout_rate=0.2):
    """Create and return an RNN model."""
    logging.info("Creating model with input shape: %s", input_shape)
    
    model = Sequential()
    model.add(LSTM(units, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))  # Output layer for regression

    logging.info("Model created successfully.")
    return model
