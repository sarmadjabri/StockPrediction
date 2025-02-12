import pandas as pd
from sklearn.preprocessing import StandardScaler
# Removed logging setup as it may be redundant
import logging

def prepare_data(data):
    """Prepare the data for modeling."""
    
    # Scale the data
    scaler = StandardScaler()
    data = data.drop(columns=['Date'])  # Drop the date column
    scaled_data = scaler.fit_transform(data)
    
    logging.info("Data preparation completed.")
    logging.info(f"Data shape after preparation: {scaled_data.shape}")
    return scaled_data
