import yfinance as yf
from logging_config import setup_logging
setup_logging()
import logging
from model_definition import create_model

def download_data(ticker, start_date, end_date):  
    logging.info(f"Starting download for {ticker} from {start_date} to {end_date}.")
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        logging.info(f"Successfully downloaded data for {ticker}.")
        return data
    except Exception as e:
        logging.error(f"Error downloading data for {ticker}: {e}")
        return None
