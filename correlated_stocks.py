import yfinance as yf
import pandas as pd
from logging_config import setup_logging
setup_logging()
import logging

def download_correlated_stocks(target_ticker, stock_list, start_date, end_date, threshold=0.5):
    """Download correlated stocks and return their data along with correlation with the target ticker."""
    logging.info(f"Stock list being used: {stock_list}")
    if target_ticker not in stock_list:
        logging.error(f"Target ticker {target_ticker} is not in the stock list.")
        return pd.DataFrame()  # Return an empty DataFrame if the target ticker is not found
    
    logging.info(f"Downloading correlated stocks for {target_ticker}.")
    correlated_data = {}
    
    for stock in stock_list:
        logging.info(f"Downloading data for {stock}.")
        try:
            data = yf.download(stock, start=start_date, end=end_date)
            correlated_data[stock] = data['Close'].values.flatten()  # Ensure data is 1D
        except Exception as e:
            logging.error(f"Error downloading data for {stock}: {e}")
    
    # Check if correlated_data is empty before creating DataFrame
    if not correlated_data:
        logging.error("No correlated data was collected.")
        return pd.DataFrame()  # Return an empty DataFrame if no data was collected

    logging.info(f"Correlated data collected: {correlated_data}")
    df_correlated = pd.DataFrame.from_dict(correlated_data, orient='columns')
    correlation_matrix = df_correlated.corrwith(df_correlated[target_ticker])
    logging.info(f"Correlation with {target_ticker}: {correlation_matrix}")
    
    # Filter stocks based on correlation threshold
    correlated_stocks = correlation_matrix[correlation_matrix > threshold].index.tolist()
    logging.info(f"Correlated stocks above threshold {threshold}: {correlated_stocks}")
    
    return df_correlated  # Return the DataFrame of correlated stock data
