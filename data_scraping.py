import requests
from bs4 import BeautifulSoup
import pandas as pd
from logging_config import setup_logging
setup_logging()
import logging

def scrape_news_data(ticker, start_date, end_date):
    """Scrape news articles related to the specified stock ticker."""
    url = f"https://newsapi.org/v2/everything?q={ticker}&from={start_date}&to={end_date}&sortBy=popularity&apiKey=YOUR_API_KEY"
    response = requests.get(url)
    if response.status_code != 200:
        logging.error(f"Failed to fetch news data: {response.status_code}")
        return pd.DataFrame()  # Return an empty DataFrame on failure

    articles = response.json().get('articles', [])
    news_data = []
    for article in articles:
        news_data.append({
            'title': article['title'],
            'description': article['description'],
            'publishedAt': article['publishedAt'],
            'url': article['url']
        })
    
    return pd.DataFrame(news_data)

def process_news_sentiment(news_df):
    """Process news articles to extract sentiment scores."""
    from textblob import TextBlob
    news_df['sentiment'] = news_df['description'].apply(lambda x: TextBlob(x).sentiment.polarity)
    return news_df
