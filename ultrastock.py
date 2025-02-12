import pandas as pd
from logging_config import setup_logging
from correlated_stocks import download_correlated_stocks
from data_download import download_data
from data_scraping import scrape_news_data
from feature_engineering import combine_features
from model_training import train_model
from prediction import predict_with_all_features
setup_logging()
import numpy as np
import logging
import yfinance as yf

def main(user_response='default_response'):
    """Main function to orchestrate data collection, processing, and prediction."""
    stock_ticker = 'TSLA'
    start_date = '2021-01-01'
    end_date = '2022-02-26'
    
    # Data Collection
    logging.info(f"User response for Black-Scholes: {user_response}")
    stock_data = download_data(stock_ticker, start_date, end_date)
    logging.info(f"Shape of stock_data: {stock_data.shape}")
    news_data = scrape_news_data(stock_ticker, start_date, end_date)
    logging.info(f"Shape of news_data: {news_data.shape}")
    
    # Data Processing
    correlated_data = download_correlated_stocks(stock_ticker, ['F', 'GM', 'NIO', 'TSLA'], start_date, end_date)
    logging.info(f"Shape of correlated_data: {correlated_data.shape}")
    
    # Check shapes before combining
    logging.info(f"Checking shapes: stock_data: {stock_data.shape}, correlated_data: {correlated_data.shape}, news_data: {news_data.shape}")
    if stock_data.empty or correlated_data.empty or news_data.empty:
        logging.error("One or more input DataFrames are empty. Please check the input data.")
        return
    
    try:
        combined_features = combine_features(stock_data, correlated_data, news_data)
        logging.info(f"Shape of combined features: {combined_features.shape}")
    except Exception as e:
        logging.error(f"Error combining features: {e}")
        return
    
    # Model Training
    model = train_model(combined_features)
    logging.info("Model training completed successfully.")
    
    # Prediction
    predictions = predict_with_all_features(model, combined_features)
    logging.info(f"Predictions made: {predictions}")
    print(predictions)

if __name__ == "__main__":
    main()
