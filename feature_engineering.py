import pandas as pd
from logging_config import setup_logging
setup_logging()
import logging

def combine_features(stock_data, correlated_data, news_data):
    """Combine stock data, correlated stocks data, and news sentiment data into a single DataFrame."""
    logging.info("Combining features from stock data, correlated stocks, and news sentiment.")
    
    # Check for empty DataFrames and log a warning
    if stock_data.empty or correlated_data.empty or news_data.empty:
        logging.warning("One or more input DataFrames are empty. Please check the input data.")
        return pd.DataFrame()  # Return an empty DataFrame if any input is empty
    
    # Combine stock data and correlated data
    combined_data = stock_data.merge(correlated_data, how='outer', left_index=True, right_index=True)
    
    # Combine with news sentiment data
    combined_data = combined_data.merge(news_data.set_index('publishedAt'), how='outer', left_index=True, right_index=True)
    
    # Drop rows with NaN values
    combined_data = combined_data.dropna()  # Drop rows with NaN values
    logging.info(f"Shape of combined data after dropping NaNs: {combined_data.shape}")
    logging.info(f"Combined DataFrame columns: {combined_data.columns.tolist()}")
    
    return combined_data