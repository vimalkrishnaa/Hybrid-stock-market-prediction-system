"""
Pipeline Orchestrator - Full Data Collection, Preprocessing, Training, and Prediction
This module orchestrates the complete pipeline for fetching new data, preprocessing,
training, and making predictions.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import logging
import json
import pickle
from typing import Dict, Any, Optional
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import required libraries
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance not available")

try:
    from nsepy import get_history
    NSEPY_AVAILABLE = True
except ImportError:
    NSEPY_AVAILABLE = False
    logger.warning("NSEpy not available")

try:
    from sklearn.preprocessing import MinMaxScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.error("sklearn not available")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Attention, Permute, Multiply, Lambda, Concatenate
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    TF_AVAILABLE = True
except ImportError as e:
    TF_AVAILABLE = False
    logger.error(f"TensorFlow not available: {e}")

# Sentiment analysis libraries
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    FINBERT_AVAILABLE = True
except ImportError:
    FINBERT_AVAILABLE = False
    logger.warning("FinBERT (transformers) not available")

# Fallback sentiment analyzer (VADER - lightweight, no heavy dependencies)
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
    vader_analyzer = SentimentIntensityAnalyzer()
except ImportError:
    VADER_AVAILABLE = False
    vader_analyzer = None
    logger.warning("VADER sentiment analyzer not available - install with: pip install vaderSentiment")

try:
    import tweepy
    TWEEPY_AVAILABLE = True
except ImportError:
    TWEEPY_AVAILABLE = False
    logger.warning("Tweepy not available")

try:
    from newsapi import NewsApiClient
    NEWSAPI_AVAILABLE = True
except ImportError:
    NEWSAPI_AVAILABLE = False
    logger.warning("NewsAPI not available")

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    logger.warning("arch library not available - GARCH will use simplified method")


class PipelineOrchestrator:
    """Orchestrates the complete pipeline: collect -> preprocess -> train -> predict"""
    
    def __init__(self, symbol: str, start_date: str, end_date: str):
        self.symbol = symbol
        # Normalize dates to timezone-naive to avoid comparison issues
        self.start_date = pd.to_datetime(start_date).tz_localize(None) if pd.to_datetime(start_date).tz is not None else pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date).tz_localize(None) if pd.to_datetime(end_date).tz is not None else pd.to_datetime(end_date)
        self.status = {
            'step': 'initialized',
            'progress': 0,
            'message': 'Pipeline initialized',
            'errors': []
        }
        
        # Paths
        self.raw_path = 'data/extended/raw'
        self.processed_path = 'data/extended/processed'
        self.models_path = 'data/extended/models'
        self.temp_path = f'data/extended/temp_{symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        
        # Create directories
        os.makedirs(self.raw_path, exist_ok=True)
        os.makedirs(self.processed_path, exist_ok=True)
        os.makedirs(self.models_path, exist_ok=True)
        os.makedirs(self.temp_path, exist_ok=True)
        
        # Load API keys from environment or file
        self.news_api_key = os.getenv('NEWS_API_KEY', '')
        self.twitter_bearer_token = os.getenv('TWITTER_BEARER_TOKEN', '')
        
        # Store news articles with sentiment for frontend display
        self.news_articles = []
        
        # Try loading from api_keys.txt if not in environment
        if not self.news_api_key or not self.twitter_bearer_token:
            try:
                with open('api_keys.txt', 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('NEWS_API_KEY=') and not self.news_api_key:
                            self.news_api_key = line.split('=', 1)[1].strip()
                            logger.info(f"Loaded NewsAPI key from api_keys.txt (length: {len(self.news_api_key)})")
                        elif line.startswith('TWITTER_BEARER_TOKEN=') and not self.twitter_bearer_token:
                            self.twitter_bearer_token = line.split('=', 1)[1].strip()
                            logger.info(f"Loaded Twitter token from api_keys.txt (length: {len(self.twitter_bearer_token)})")
            except FileNotFoundError:
                logger.warning("api_keys.txt not found. Sentiment collection will be limited.")
            except Exception as e:
                logger.warning(f"Error loading api_keys.txt: {e}")
        
        # Log final API key status
        logger.info(f"API keys loaded - NewsAPI: {'✓' if self.news_api_key else '✗'}, Twitter: {'✓' if self.twitter_bearer_token else '✗'}")
        
        # Initialize FinBERT model (lazy loading)
        self.finbert_model = None
        self.finbert_tokenizer = None
        
    def update_status(self, step: str, progress: int, message: str):
        """Update pipeline status"""
        self.status = {
            'step': step,
            'progress': progress,
            'message': message,
            'errors': self.status.get('errors', [])
        }
        logger.info(f"[{step}] {progress}% - {message}")
    
    def collect_stock_data(self) -> pd.DataFrame:
        """Step 1: Collect stock price data for the symbol and date range"""
        self.update_status('collecting', 10, f'Collecting stock data for {self.symbol}...')
        
        try:
            # Indian stocks - try NSEpy first
            indian_stocks = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'HINDUNILVR',
                           'BHARTIARTL', 'KOTAKBANK', 'SBIN', 'ITC', 'AXISBANK']
            
            df = None
            
            if self.symbol in indian_stocks and NSEPY_AVAILABLE:
                try:
                    self.update_status('collecting', 15, f'Fetching from NSE for {self.symbol}...')
                    # NSEpy requires date objects
                    nse_data = get_history(
                        symbol=self.symbol,
                        start=self.start_date.date(),
                        end=self.end_date.date()
                    )
                    if not nse_data.empty:
                        df = nse_data.reset_index()
                        df['Symbol'] = self.symbol
                        df['Source'] = 'NSE'
                        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Symbol', 'Source']]
                except Exception as e:
                    logger.warning(f"NSEpy failed for {self.symbol}: {e}")
            
            # Fallback to yfinance
            if df is None or df.empty:
                if not YFINANCE_AVAILABLE:
                    raise Exception("yfinance not available and NSEpy failed")
                
                self.update_status('collecting', 20, f'Fetching from yfinance for {self.symbol}...')
                ticker_symbol = self.symbol
                if self.symbol in indian_stocks:
                    ticker_symbol = f"{self.symbol}.NS"  # NSE suffix for yfinance
                
                ticker = yf.Ticker(ticker_symbol)
                # yfinance end_date is exclusive, so add 1 day to include the end date
                end_date_inclusive = self.end_date + timedelta(days=1)
                df = ticker.history(start=self.start_date, end=end_date_inclusive)
                
                if df.empty:
                    raise Exception(f"No data returned for {ticker_symbol}")
                
                df = df.reset_index()
                df['Symbol'] = self.symbol
                df['Source'] = 'yfinance'
                df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Symbol', 'Source']]
            
            if df.empty:
                raise Exception(f"No data collected for {self.symbol}")
            
            # Save raw data
            raw_file = f"{self.temp_path}/raw_data_{self.symbol}.csv"
            df.to_csv(raw_file, index=False)
            
            self.update_status('collecting', 30, f'Collected {len(df)} records for {self.symbol}')
            return df
            
        except Exception as e:
            error_msg = f"Data collection failed: {str(e)}"
            self.status['errors'].append(error_msg)
            logger.error(error_msg)
            raise
    
    def load_existing_twitter_sentiment(self) -> pd.DataFrame:
        """Load Twitter tweets from cache and analyze sentiment with FinBERT"""
        try:
            twitter_cache_file = 'data/raw/sentiment/twitter_cache.json'
            if not os.path.exists(twitter_cache_file):
                logger.warning(f"Twitter cache file not found: {twitter_cache_file}")
                return pd.DataFrame({'date': [], 'sentiment_score': []})
            
            # Load Twitter cache
            with open(twitter_cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            tweets = cache_data.get('tweets', [])
            if len(tweets) == 0:
                logger.warning("No tweets found in Twitter cache")
                return pd.DataFrame({'date': [], 'sentiment_score': []})
            
            logger.info(f"Loaded {len(tweets)} tweets from Twitter cache")
            
            # Initialize FinBERT if not already loaded
            if FINBERT_AVAILABLE and self.finbert_model is None:
                try:
                    model_name = 'yiyanghkust/finbert-tone'
                    self.finbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
                    self.finbert_model.eval()
                    logger.info("FinBERT model loaded for Twitter sentiment analysis")
                except Exception as e:
                    logger.warning(f"Failed to load FinBERT: {e}")
                    return pd.DataFrame({'date': [], 'sentiment_score': []})
            
            # Analyze sentiment for each tweet
            sentiment_data = []
            if FINBERT_AVAILABLE and self.finbert_model:
                for tweet in tweets:
                    try:
                        text = tweet.get('text', '')
                        if not text:
                            continue
                        
                        # Skip retweets (they start with "RT @")
                        if text.startswith('RT @'):
                            continue
                        
                        # Tokenize and predict
                        inputs = self.finbert_tokenizer(
                            text,
                            return_tensors='pt',
                            truncation=True,
                            max_length=512,
                            padding=True
                        )
                        
                        with torch.no_grad():
                            outputs = self.finbert_model(**inputs)
                            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                        
                        # Calculate sentiment score: P(positive) - P(negative)
                        positive_prob = probs[0][0].item()
                        negative_prob = probs[0][2].item()
                        sentiment_score = positive_prob - negative_prob
                        
                        # Extract date from created_at
                        created_at = tweet.get('created_at', '')
                        if created_at:
                            try:
                                date = pd.to_datetime(created_at).date()
                                sentiment_data.append({
                                    'date': date,
                                    'sentiment_score': sentiment_score
                                })
                            except Exception as e:
                                logger.warning(f"Error parsing date {created_at}: {e}")
                                continue
                    except Exception as e:
                        logger.warning(f"Error analyzing tweet sentiment: {e}")
                        continue
            
            if len(sentiment_data) > 0:
                twitter_df = pd.DataFrame(sentiment_data)
                logger.info(f"Analyzed sentiment for {len(twitter_df)} tweets from cache")
                return twitter_df
            else:
                logger.warning("No valid sentiment data extracted from Twitter cache")
                return pd.DataFrame({'date': [], 'sentiment_score': []})
                
        except Exception as e:
            logger.warning(f"Failed to load Twitter cache: {e}")
            return pd.DataFrame({'date': [], 'sentiment_score': []})
    
    def get_cached_news_sentiment(self, query: str, date: str) -> Optional[list]:
        """Check if NewsAPI data is cached (within 6 hours)"""
        cache_dir = 'data/extended/cache/newsapi'
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = f"{cache_dir}/{query}_{date}.json"
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                cache_time = datetime.fromisoformat(cache_data['timestamp'])
                # Check if cache is less than 6 hours old
                if (datetime.now() - cache_time).total_seconds() < 6 * 3600:
                    logger.info(f"Using cached NewsAPI data for {query} on {date}")
                    return cache_data['articles']
            except Exception as e:
                logger.warning(f"Error reading cache: {e}")
        return None
    
    def cache_news_sentiment(self, query: str, date: str, articles: list):
        """Cache NewsAPI data with timestamp"""
        cache_dir = 'data/extended/cache/newsapi'
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = f"{cache_dir}/{query}_{date}.json"
        
        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'articles': articles
                }, f)
        except Exception as e:
            logger.warning(f"Error caching NewsAPI data: {e}")
    
    def collect_and_analyze_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Collect news (NewsAPI) and reuse existing Twitter sentiment, analyze using FinBERT"""
        try:
            sentiment_data = []
            dates = df['Date'].dt.date.unique()
            
            # NewsAPI free tier only covers last month - filter dates
            from datetime import datetime, timedelta
            today = datetime.now().date()
            one_month_ago = today - timedelta(days=30)  # date - timedelta = date (no need for .date())
            
            logger.info(f"NewsAPI date filtering: Today={today}, One month ago={one_month_ago}")
            logger.info(f"Total dates in dataset: {len(dates)}, Date range: {min(dates) if len(dates) > 0 else 'N/A'} to {max(dates) if len(dates) > 0 else 'N/A'}")
            logger.info(f"Dataset end date: {max(dates) if len(dates) > 0 else 'N/A'}, Days from today: {(today - max(dates)).days if len(dates) > 0 else 'N/A'}")
            
            recent_dates = [d for d in dates if d >= one_month_ago]
            
            if len(recent_dates) < len(dates):
                logger.info(f"NewsAPI: Filtering dates - {len(recent_dates)} recent dates (last 30 days) out of {len(dates)} total dates")
                if recent_dates:
                    logger.info(f"NewsAPI: Will collect news for dates from {min(recent_dates)} to {max(recent_dates)}")
                else:
                    logger.warning(f"NewsAPI: No recent dates found! All dates are older than 30 days.")
            else:
                logger.info(f"NewsAPI: All {len(dates)} dates are within last 30 days - will collect news for all")
            
            # Use only recent dates for news collection
            dates_for_news = recent_dates if recent_dates else []
            
            # Initialize FinBERT if available (don't fail if it doesn't load - we can still collect news)
            if FINBERT_AVAILABLE and self.finbert_model is None:
                try:
                    self.update_status('preprocessing', 51, 'Loading FinBERT model...')
                    model_name = 'yiyanghkust/finbert-tone'
                    self.finbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
                    self.finbert_model.eval()
                    logger.info("FinBERT model loaded successfully")
                except Exception as e:
                    logger.warning(f"Failed to load FinBERT: {e}")
                    logger.warning("Continuing without FinBERT - will use neutral sentiment for news articles")
                    # Don't return early - continue to collect news even without FinBERT
            
            # Twitter sentiment removed - only using NewsAPI for sentiment
            # This prevents the issue where Twitter sentiment was incorrectly applied to many dates
            logger.info("Twitter sentiment disabled - using only NewsAPI for sentiment collection")
            
            # Collect fresh news articles from NewsAPI (with 6-hour cache)
            news_texts = []
            logger.info(f"NewsAPI availability check: NEWSAPI_AVAILABLE={NEWSAPI_AVAILABLE}, API key present={bool(self.news_api_key)}")
            if not NEWSAPI_AVAILABLE:
                logger.warning("NewsAPI library not available - news articles will not be collected")
            elif not self.news_api_key:
                logger.warning(f"NewsAPI key not configured (key length: {len(self.news_api_key) if self.news_api_key else 0}) - news articles will not be collected")
            else:
                try:
                    self.update_status('preprocessing', 53, 'Fetching news articles (NewsAPI)...')
                    newsapi = NewsApiClient(api_key=self.news_api_key)
                    
                    # Search query based on symbol
                    query = self.symbol
                    if self.symbol in ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'HINDUNILVR', 
                                     'BHARTIARTL', 'KOTAKBANK', 'SBIN', 'ITC', 'AXISBANK']:
                        query = f"{self.symbol} stock India"
                    
                    if not dates_for_news:
                        logger.warning(f"NewsAPI: No recent dates (within last 30 days) to fetch news for")
                        # Fallback: Try to get news for the most recent dates in the dataset (even if older than 30 days)
                        if len(dates) > 0:
                            most_recent_dates = sorted(dates, reverse=True)[:10]  # Top 10 most recent dates
                            logger.info(f"NewsAPI: Fallback - trying to collect news for {len(most_recent_dates)} most recent dates: {most_recent_dates[0]} to {most_recent_dates[-1]}")
                            dates_for_news = most_recent_dates
                    
                    if dates_for_news:
                        logger.info(f"Searching NewsAPI for: {query} across {len(dates_for_news)} dates")
                        articles_found = 0
                        
                        # Try multiple query variations if first one fails
                        query_variations = [query]
                        if " stock India" in query:
                            # Add simpler query without "stock India"
                            query_variations.append(query.replace(" stock India", ""))
                            query_variations.append(self.symbol)  # Just the symbol
                        
                        for date in dates_for_news:
                            date_str = date.strftime('%Y-%m-%d')
                            articles_data = []
                            
                            # Check cache first
                            cached_articles = self.get_cached_news_sentiment(query, date_str)
                            if cached_articles:
                                articles_data = cached_articles
                                logger.info(f"Using cached articles for {date_str}: {len(articles_data)} articles")
                            else:
                                # Try each query variation until we get results
                                for query_var in query_variations:
                                    try:
                                        logger.info(f"Fetching news from NewsAPI for {date_str} with query: '{query_var}'...")
                                        response = newsapi.get_everything(
                                            q=query_var,
                                            from_param=date_str,
                                            to=(date + timedelta(days=1)).strftime('%Y-%m-%d'),
                                            language='en',
                                            sort_by='relevancy',
                                            page_size=10
                                        )
                                        articles_data = response.get('articles', [])
                                        total_results = response.get('totalResults', 0)
                                        
                                        if len(articles_data) > 0:
                                            logger.info(f"✓ Found {len(articles_data)} articles for {date_str} with query '{query_var}' (total results: {total_results})")
                                            articles_found += len(articles_data)
                                            # Cache with original query key
                                            self.cache_news_sentiment(query, date_str, articles_data)
                                            break  # Success, stop trying other queries
                                        else:
                                            logger.debug(f"No articles for {date_str} with query '{query_var}' (total results: {total_results})")
                                    except Exception as e:
                                        logger.warning(f"NewsAPI error for {date_str} with query '{query_var}': {e}")
                                        continue
                                
                                if len(articles_data) == 0:
                                    logger.warning(f"No articles found for {date_str} with any query variation")
                                    # Cache empty result to avoid repeated API calls
                                    self.cache_news_sentiment(query, date_str, [])
                            
                            # Process articles
                            if not articles_data:
                                logger.warning(f"No articles to process for {date_str} - articles_data is empty")
                            else:
                                processed_count = 0
                                skipped_count = 0
                                for article in articles_data[:5]:  # Top 5 articles
                                    if article.get('title') and article.get('description'):
                                        text = f"{article['title']} {article['description']}"
                                        news_texts.append({
                                            'date': date, 
                                            'text': text, 
                                            'source': 'news',
                                            'article': article  # Store full article for frontend
                                        })
                                        processed_count += 1
                                    else:
                                        skipped_count += 1
                                        logger.debug(f"Skipping article (missing title/description): {article.get('title', 'N/A')[:50]}")
                                logger.info(f"Processed {processed_count} articles for {date_str} (skipped {skipped_count} without title/description)")
                        
                        logger.info(f"Total news articles collected: {len(news_texts)} from {articles_found} API results")
                        if len(news_texts) == 0:
                            logger.error(f"❌ No news articles collected for {self.symbol}!")
                            logger.error(f"   This could be due to:")
                            logger.error(f"   1. No articles available for the date range")
                            logger.error(f"   2. Query '{query}' too specific")
                            logger.error(f"   3. NewsAPI free tier limitations")
                            logger.error(f"   4. Articles missing title/description fields")
                        else:
                            logger.info(f"✓ Successfully collected {len(news_texts)} news articles for processing")
                except Exception as e:
                    logger.warning(f"NewsAPI collection failed: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Analyze news sentiment using FinBERT and store articles
            if not news_texts:
                logger.error(f"❌ No news_texts collected! news_texts length: {len(news_texts)}")
                logger.error(f"   This means articles were not added to news_texts list during collection")
            else:
                # Store articles even if FinBERT fails (with neutral sentiment)
                logger.info(f"✓ Processing {len(news_texts)} news_texts items for sentiment analysis and storage")
                
                sentiment_analyzed_count = 0
                
                if FINBERT_AVAILABLE and self.finbert_model:
                    self.update_status('preprocessing', 54, f'Analyzing {len(news_texts)} news articles with FinBERT...')
                    logger.info(f"Analyzing sentiment for {len(news_texts)} news articles with FinBERT")
                elif VADER_AVAILABLE and vader_analyzer:
                    self.update_status('preprocessing', 54, f'Analyzing {len(news_texts)} news articles with VADER (FinBERT unavailable)...')
                    logger.info(f"Analyzing sentiment for {len(news_texts)} news articles with VADER (FinBERT not available)")
                else:
                    self.update_status('preprocessing', 54, f'Processing {len(news_texts)} news articles (no sentiment analyzer available, using neutral sentiment)...')
                    logger.warning("Neither FinBERT nor VADER available - storing articles with neutral sentiment (0.0)")
                
                for idx, item in enumerate(news_texts):
                    sentiment_score = 0.0  # Default neutral
                    sentiment_method = "neutral (fallback)"
                    
                    # Try FinBERT analysis first (best quality)
                    if FINBERT_AVAILABLE and self.finbert_model:
                        try:
                            # Tokenize and predict
                            inputs = self.finbert_tokenizer(
                                item['text'],
                                return_tensors='pt',
                                truncation=True,
                                max_length=512,
                                padding=True
                            )
                            
                            with torch.no_grad():
                                outputs = self.finbert_model(**inputs)
                                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                            
                            # Calculate sentiment score: P(positive) - P(negative)
                            positive_prob = probs[0][0].item()
                            negative_prob = probs[0][2].item()
                            sentiment_score = positive_prob - negative_prob
                            sentiment_analyzed_count += 1
                            sentiment_method = "FinBERT"
                            
                            sentiment_data.append({
                                'date': item['date'],
                                'sentiment_score': sentiment_score
                            })
                        except Exception as e:
                            logger.warning(f"FinBERT analysis error for article {idx+1}: {e}")
                            # Fall through to VADER
                    
                    # Fallback to VADER if FinBERT failed or unavailable
                    if sentiment_score == 0.0 and VADER_AVAILABLE and vader_analyzer:
                        try:
                            scores = vader_analyzer.polarity_scores(item['text'])
                            sentiment_score = scores['compound']  # Range: -1.0 to +1.0
                            sentiment_analyzed_count += 1
                            sentiment_method = "VADER"
                            
                            sentiment_data.append({
                                'date': item['date'],
                                'sentiment_score': sentiment_score
                            })
                        except Exception as e:
                            logger.warning(f"VADER analysis error for article {idx+1}: {e}")
                            sentiment_method = "neutral (error)"
                    
                    # Log first few articles for debugging
                    if idx < 3:
                        logger.debug(f"Article {idx+1} sentiment: {sentiment_score:.3f} ({sentiment_method})")
                    
                    # Store news article for frontend display (regardless of FinBERT success)
                    if 'article' in item:
                        article = item['article']
                        article_data = {
                            'title': article.get('title', ''),
                            'description': article.get('description', ''),
                            'url': article.get('url', ''),
                            'source': article.get('source', {}).get('name', 'Unknown'),
                            'publishedAt': article.get('publishedAt', ''),
                            'date': item['date'].strftime('%Y-%m-%d') if hasattr(item['date'], 'strftime') else str(item['date']),
                            'sentiment_score': float(sentiment_score),
                            'sentiment_label': 'positive' if sentiment_score > 0.1 else ('negative' if sentiment_score < -0.1 else 'neutral')
                        }
                        self.news_articles.append(article_data)
                        if idx < 3:  # Log first 3 articles for debugging
                            logger.debug(f"Stored article {idx+1}: {article_data['title'][:50]}... (sentiment: {sentiment_score:.3f})")
                    else:
                        logger.error(f"❌ Item {idx+1} missing 'article' key! Keys: {list(item.keys())}")
                
                logger.info(f"✓ Successfully stored {len(self.news_articles)} news articles in self.news_articles")
                logger.info(f"  - Sentiment analyzed: {sentiment_analyzed_count}/{len(news_texts)}")
                if len(self.news_articles) == 0:
                    logger.error(f"❌ CRITICAL: No articles stored in self.news_articles despite having {len(news_texts)} news_texts!")
                    logger.error(f"   Check if 'article' key exists in news_texts items")
            
            # Aggregate sentiment by date
            if sentiment_data:
                sentiment_df = pd.DataFrame(sentiment_data)
                
                # Log raw sentiment data before aggregation
                logger.info(f"Raw sentiment data: {len(sentiment_df)} records")
                if len(sentiment_df) > 0:
                    logger.info(f"Raw sentiment date range: {sentiment_df['date'].min()} to {sentiment_df['date'].max()}")
                    logger.info(f"Raw sentiment score range: {sentiment_df['sentiment_score'].min():.6f} to {sentiment_df['sentiment_score'].max():.6f}")
                    logger.info(f"Raw sentiment unique dates: {sentiment_df['date'].nunique()}")
                
                daily_sentiment = sentiment_df.groupby('date')['sentiment_score'].mean().reset_index()
                daily_sentiment.columns = ['Date', 'sentiment_score']
                # Convert to datetime and normalize timezone
                daily_sentiment['Date'] = pd.to_datetime(daily_sentiment['Date']).dt.tz_localize(None)
                logger.info(f"Collected sentiment for {len(daily_sentiment)} dates (NewsAPI only)")
                if len(daily_sentiment) > 0:
                    logger.info(f"Sentiment date range: {daily_sentiment['Date'].min()} to {daily_sentiment['Date'].max()}")
                    logger.info(f"Sentiment score range: {daily_sentiment['sentiment_score'].min():.6f} to {daily_sentiment['sentiment_score'].max():.6f}")
                    logger.info(f"Sentiment unique values: {daily_sentiment['sentiment_score'].nunique()}")
                    # Check if all values are the same (potential bug)
                    if daily_sentiment['sentiment_score'].nunique() == 1:
                        logger.warning(f"⚠️ WARNING: All sentiment scores are the same value: {daily_sentiment['sentiment_score'].iloc[0]:.6f}")
                        logger.warning(f"   This suggests Twitter sentiment might be incorrectly aggregated")
            else:
                # Return empty DataFrame with Date column
                daily_sentiment = pd.DataFrame({'Date': pd.to_datetime([]), 'sentiment_score': []})
                logger.warning("No sentiment data collected, using neutral (0.0)")
            
            return daily_sentiment
            
        except Exception as e:
            logger.error(f"Sentiment collection failed: {e}")
            import traceback
            traceback.print_exc()
            # Return empty DataFrame with proper date format
            try:
                dates_only = df['Date'].dt.date if hasattr(df['Date'].dt, 'date') else df['Date']
            except:
                dates_only = df['Date']
            return pd.DataFrame({'Date': dates_only, 'sentiment_score': 0.0})
    
    def calculate_garch_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate proper GARCH(1,1) conditional volatility"""
        try:
            # Group by symbol and calculate GARCH for each
            garch_results = []
            
            for symbol in df['Symbol'].unique():
                symbol_data = df[df['Symbol'] == symbol].copy()
                symbol_data = symbol_data.sort_values('Date').reset_index(drop=True)
                
                # Extract returns
                returns = symbol_data['Returns'].values
                
                # Remove NaN and inf values
                valid_mask = ~(np.isnan(returns) | np.isinf(returns))
                returns_clean = returns[valid_mask]
                dates_clean = symbol_data['Date'].iloc[valid_mask].values
                
                if len(returns_clean) < 30:
                    # Insufficient data - use rolling std
                    logger.warning(f"Insufficient data for GARCH({symbol}): {len(returns_clean)} points, using rolling std")
                    vol_values = symbol_data['Returns'].rolling(window=min(30, len(symbol_data)), min_periods=1).std().fillna(0).values
                    # Normalize dates
                    dates_normalized = pd.to_datetime(symbol_data['Date']).dt.tz_localize(None)
                    garch_results.append(pd.DataFrame({
                        'Date': dates_normalized,
                        'garch_volatility': vol_values
                    }))
                    continue
                
                # Try to fit GARCH(1,1) model
                if ARCH_AVAILABLE:
                    try:
                        # Scale returns to percentage for better numerical stability
                        returns_scaled = returns_clean * 100
                        
                        # Fit GARCH(1,1) model
                        model = arch_model(returns_scaled, vol='Garch', p=1, q=1, rescale=False)
                        result = model.fit(disp='off', show_warning=False)
                        
                        # Extract conditional volatility (convert back from percentage)
                        conditional_vol = result.conditional_volatility / 100
                        
                        # Create DataFrame with dates and volatility
                        # Normalize dates to remove timezone (dates_clean is numpy array)
                        dates_clean_series = pd.Series(dates_clean)
                        dates_clean_normalized = pd.to_datetime(dates_clean_series).dt.tz_localize(None)
                        vol_df = pd.DataFrame({
                            'Date': dates_clean_normalized,
                            'garch_volatility': conditional_vol
                        })
                        
                        # Align with original dates (some may be missing due to NaN/inf removal)
                        # Normalize symbol_data dates too
                        symbol_dates_normalized = pd.to_datetime(symbol_data['Date']).dt.tz_localize(None)
                        full_vol_df = pd.DataFrame({
                            'Date': symbol_dates_normalized
                        })
                        full_vol_df = full_vol_df.merge(vol_df, on='Date', how='left')
                        
                        # Forward fill and backward fill for missing values
                        full_vol_df['garch_volatility'] = full_vol_df['garch_volatility'].bfill().ffill()
                        
                        # If still missing, use rolling std
                        if full_vol_df['garch_volatility'].isna().any():
                            rolling_vol = symbol_data['Returns'].rolling(window=30, min_periods=1).std().fillna(0)
                            full_vol_df['garch_volatility'] = full_vol_df['garch_volatility'].fillna(rolling_vol.values)
                        
                        garch_results.append(full_vol_df[['Date', 'garch_volatility']])
                        logger.info(f"✓ GARCH(1,1) fitted for {symbol}: mean vol={conditional_vol.mean():.6f}")
                        
                    except Exception as e:
                        logger.warning(f"GARCH fitting failed for {symbol}: {e}, using rolling std")
                        # Fallback to rolling std
                        vol_values = symbol_data['Returns'].rolling(window=30, min_periods=1).std().fillna(0).values
                        # Normalize dates
                        dates_normalized = pd.to_datetime(symbol_data['Date']).dt.tz_localize(None)
                        garch_results.append(pd.DataFrame({
                            'Date': dates_normalized,
                            'garch_volatility': vol_values
                        }))
                else:
                    # arch library not available - use rolling std
                    logger.warning(f"arch library not available, using rolling std for {symbol}")
                    vol_values = symbol_data['Returns'].rolling(window=30, min_periods=1).std().fillna(0).values
                    # Normalize dates
                    dates_normalized = pd.to_datetime(symbol_data['Date']).dt.tz_localize(None)
                    garch_results.append(pd.DataFrame({
                        'Date': dates_normalized,
                        'garch_volatility': vol_values
                    }))
            
            # Combine all results
            if garch_results:
                combined_garch = pd.concat(garch_results, ignore_index=True)
                return combined_garch
            else:
                # Fallback: return rolling std for all dates
                return pd.DataFrame({
                    'Date': df['Date'],
                    'garch_volatility': df['Returns'].rolling(window=30, min_periods=1).std().fillna(0).values
                })
                
        except Exception as e:
            logger.error(f"GARCH calculation failed: {e}")
            # Fallback to rolling std
            return pd.DataFrame({
                'Date': df['Date'],
                'garch_volatility': df['Returns'].rolling(window=30, min_periods=1).std().fillna(0).values
            })
    
    def preprocess_data(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Step 2: Preprocess data - add features, normalize, add sentiment/volatility"""
        self.update_status('preprocessing', 40, 'Preprocessing data...')
        
        try:
            df = raw_df.copy()
            # Convert Date to datetime and remove timezone to avoid merge issues
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
            df = df.sort_values('Date').reset_index(drop=True)
            
            # Calculate technical indicators
            self.update_status('preprocessing', 45, 'Calculating basic technical indicators...')
            df['Returns'] = df['Close'].pct_change()
            df['MA_5'] = df['Close'].rolling(window=5).mean()
            df['MA_10'] = df['Close'].rolling(window=10).mean()
            df['MA_20'] = df['Close'].rolling(window=20).mean()
            df['Volatility'] = df['Returns'].rolling(window=30).std()
            df['Momentum'] = df['Close'].pct_change(periods=5)
            
            # Advanced technical indicators for better directional prediction
            self.update_status('preprocessing', 46, 'Calculating advanced technical indicators (RSI, MACD, Bollinger Bands)...')
            
            # RSI (Relative Strength Index) - 14 period
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD (Moving Average Convergence Divergence)
            ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
            ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = ema_12 - ema_26
            df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_hist'] = df['MACD'] - df['MACD_signal']
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            bb_middle = df['Close'].rolling(window=bb_period).mean()
            bb_std_val = df['Close'].rolling(window=bb_period).std()
            df['BB_upper'] = bb_middle + (bb_std_val * bb_std)
            df['BB_lower'] = bb_middle - (bb_std_val * bb_std)
            df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / bb_middle
            df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
            
            # Additional momentum indicators
            df['Momentum_10'] = df['Close'].pct_change(periods=10)
            df['Momentum_20'] = df['Close'].pct_change(periods=20)
            
            # Volume indicators
            df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()
            df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
            df['Volume_ratio'] = df['Volume'] / (df['Volume_MA_20'] + 1e-8)  # Avoid division by zero
            
            # Price position relative to range
            df['High_Low_ratio'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-8)  # Avoid division by zero
            
            # MA crossovers (important for direction)
            df['MA5_MA10_cross'] = df['MA_5'] - df['MA_10']  # Positive = bullish
            df['MA10_MA20_cross'] = df['MA_10'] - df['MA_20']  # Positive = bullish
            
            # Additional directional indicators for better accuracy
            self.update_status('preprocessing', 47, 'Calculating additional directional indicators...')
            
            # Stochastic Oscillator (K% and D%) - momentum indicator
            low_14 = df['Low'].rolling(window=14).min()
            high_14 = df['High'].rolling(window=14).max()
            df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14 + 1e-8))
            df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
            
            # Williams %R - momentum indicator (inverted, -100 to 0)
            df['Williams_R'] = -100 * ((high_14 - df['Close']) / (high_14 - low_14 + 1e-8))
            
            # Commodity Channel Index (CCI) - trend indicator
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            sma_tp = typical_price.rolling(window=20).mean()
            mad = typical_price.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
            df['CCI'] = (typical_price - sma_tp) / (0.015 * mad + 1e-8)
            # Clip CCI to reasonable range (-200 to 200) to avoid infinity
            df['CCI'] = df['CCI'].clip(-200, 200)
            
            # Average Directional Index (ADX) - trend strength
            # Simplified ADX calculation
            high_diff = df['High'].diff()
            low_diff = -df['Low'].diff()
            plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
            minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
            atr = df['High'].rolling(window=14).max() - df['Low'].rolling(window=14).min()
            atr = atr.replace(0, np.nan).bfill().fillna(1.0)  # Better ATR handling
            plus_di = 100 * (plus_dm.rolling(window=14).mean() / (atr + 1e-8))
            minus_di = 100 * (minus_dm.rolling(window=14).mean() / (atr + 1e-8))
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
            df['ADX'] = dx.rolling(window=14).mean()
            # Clip ADX to reasonable range (0 to 100)
            df['ADX'] = df['ADX'].clip(0, 100)
            
            # Price Rate of Change (ROC) - momentum
            df['ROC_10'] = df['Close'].pct_change(periods=10) * 100
            df['ROC_20'] = df['Close'].pct_change(periods=20) * 100
            
            # Price position features (for support/resistance)
            df['Price_to_MA5'] = df['Close'] / (df['MA_5'] + 1e-8)
            df['Price_to_MA20'] = df['Close'] / (df['MA_20'] + 1e-8)
            df['Price_to_High'] = df['Close'] / (df['High'].rolling(window=20).max() + 1e-8)
            df['Price_to_Low'] = df['Close'] / (df['Low'].rolling(window=20).min() + 1e-8)
            
            # Volume-based momentum
            df['Volume_Momentum'] = df['Volume'].pct_change(periods=5)
            df['Price_Volume_Trend'] = (df['Close'].pct_change() * df['Volume']).rolling(window=10).sum()
            
            # Trend strength indicators
            df['Trend_Strength'] = np.abs(df['MA_5'] - df['MA_20']) / (df['Close'] + 1e-8)
            df['Volatility_Trend'] = df['Volatility'].rolling(window=10).mean()
            
            # Fill NaN values and handle infinity (using pandas 2.0+ compatible method)
            df = df.replace([np.inf, -np.inf], np.nan)  # Replace infinity with NaN first
            df = df.bfill().ffill().fillna(0)
            
            # Additional safety: clip extreme values to prevent overflow
            # Clip all numeric columns to reasonable ranges
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col not in ['Date', 'Symbol', 'Source']:  # Skip non-numeric columns
                    # Clip to prevent extreme values (use percentiles as bounds)
                    q01 = df[col].quantile(0.01)
                    q99 = df[col].quantile(0.99)
                    if not (np.isnan(q01) or np.isnan(q99)):
                        df[col] = df[col].clip(lower=q01 - 10*abs(q01), upper=q99 + 10*abs(q99))
            
            # Final check: replace any remaining infinity or NaN
            df = df.replace([np.inf, -np.inf], 0)
            df = df.fillna(0)
            
            # Add sentiment_score from NewsAPI only (Twitter removed)
            self.update_status('preprocessing', 50, 'Collecting and analyzing sentiment (NewsAPI only)...')
            sentiment_scores = self.collect_and_analyze_sentiment(df)
            if not sentiment_scores.empty:
                # Ensure Date columns are same type (remove timezone if present)
                df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
                sentiment_scores['Date'] = pd.to_datetime(sentiment_scores['Date']).dt.tz_localize(None)
                
                # Log sentiment scores before merge for debugging
                logger.info(f"Merging sentiment: {len(sentiment_scores)} sentiment dates with {len(df)} stock dates")
                logger.info(f"Stock date range: {df['Date'].min()} to {df['Date'].max()}")
                logger.info(f"Sentiment date range: {sentiment_scores['Date'].min()} to {sentiment_scores['Date'].max()}")
                
                # Check for duplicate column names after merge (pandas adds _x, _y suffixes)
                df = df.merge(sentiment_scores, on='Date', how='left', suffixes=('', '_new'))
                
                # If merge created sentiment_score_new, use it and drop the old one
                if 'sentiment_score_new' in df.columns:
                    logger.warning("Merge created sentiment_score_new column - using it")
                    df['sentiment_score'] = df['sentiment_score_new']
                    df = df.drop(columns=['sentiment_score_new'])
                
                # Log merge results BEFORE fillna
                matched_count = df['sentiment_score'].notna().sum()
                missing_count = df['sentiment_score'].isna().sum()
                logger.info(f"Sentiment merge: {matched_count} dates matched, {missing_count} dates missing (will be filled with 0.0)")
                
                # Check if matched sentiment values are all the same (potential bug)
                if matched_count > 0:
                    matched_values = df[df['sentiment_score'].notna()]['sentiment_score']
                    logger.info(f"Matched sentiment values: min={matched_values.min():.6f}, max={matched_values.max():.6f}, unique={matched_values.nunique()}")
                    if matched_values.nunique() == 1:
                        logger.warning(f"⚠️ WARNING: All {matched_count} matched sentiment values are identical: {matched_values.iloc[0]:.6f}")
                        logger.warning(f"   This suggests sentiment might be incorrectly applied to multiple dates")
                
                # CRITICAL: Fill missing with 0.0 (neutral) - do NOT forward-fill
                # This is the correct approach: no news = neutral sentiment (0.0)
                # After normalization, 0.0 becomes 0.5, which correctly represents "no news available"
                df['sentiment_score'] = df['sentiment_score'].fillna(0.0)  # Fill missing with neutral
                
                # Verify fillna worked correctly
                zero_count = (df['sentiment_score'] == 0.0).sum()
                non_zero_count = (df['sentiment_score'] != 0.0).sum()
                logger.info(f"After fillna: {zero_count} dates with 0.0 (neutral), {non_zero_count} dates with sentiment")
                if non_zero_count > 0:
                    non_zero_values = df[df['sentiment_score'] != 0.0]['sentiment_score']
                    logger.info(f"Non-zero sentiment range: {non_zero_values.min():.6f} to {non_zero_values.max():.6f}")
                
                self.update_status('preprocessing', 50, f'✓ Sentiment added: {matched_count} dates with sentiment, {missing_count} with neutral (0.0)')
                logger.info(f"Sentiment feature added: {matched_count} non-null values, {missing_count} filled with 0.0")
            else:
                df['sentiment_score'] = 0.0  # No sentiment data available
                self.update_status('preprocessing', 50, '⚠ Sentiment: No data available (using neutral 0.0)')
                logger.warning("Sentiment feature set to 0.0 (no data)")
            
            # Add GARCH volatility (proper GARCH(1,1) modeling)
            self.update_status('preprocessing', 55, 'Calculating GARCH(1,1) conditional volatility...')
            garch_volatility = self.calculate_garch_volatility(df)
            if not garch_volatility.empty:
                # Ensure Date columns are same type (remove timezone if present)
                df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
                garch_volatility['Date'] = pd.to_datetime(garch_volatility['Date']).dt.tz_localize(None)
                df = df.merge(garch_volatility, on='Date', how='left')
                df['garch_volatility'] = df['garch_volatility'].fillna(df['Volatility'])  # Fallback to rolling std
                self.update_status('preprocessing', 55, f'✓ GARCH volatility added: {df["garch_volatility"].notna().sum()} values')
                logger.info(f"GARCH volatility feature added: mean={df['garch_volatility'].mean():.6f}")
            else:
                df['garch_volatility'] = df['Volatility']  # Use rolling std if GARCH failed
                self.update_status('preprocessing', 55, '⚠ GARCH: Using rolling std (fallback)')
                logger.warning("GARCH volatility feature using rolling std fallback")
            
            # Normalize features per symbol - now with enhanced directional indicators
            feature_cols = [
                # Market data (5)
                'Open', 'High', 'Low', 'Close', 'Volume',
                # Basic technical indicators (6)
                'Returns', 'MA_5', 'MA_10', 'MA_20', 'Volatility', 'Momentum',
                # Advanced technical indicators (12)
                'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
                'BB_upper', 'BB_lower', 'BB_width', 'BB_position',
                'Momentum_10', 'Momentum_20',
                'Volume_ratio', 'High_Low_ratio',
                # MA crossovers (2)
                'MA5_MA10_cross', 'MA10_MA20_cross',
                # Additional directional indicators (15)
                'Stoch_K', 'Stoch_D',  # Stochastic Oscillator
                'Williams_R',  # Williams %R
                'CCI',  # Commodity Channel Index
                'ADX',  # Average Directional Index
                'ROC_10', 'ROC_20',  # Rate of Change
                'Price_to_MA5', 'Price_to_MA20',  # Price position relative to MAs
                'Price_to_High', 'Price_to_Low',  # Price position relative to range
                'Volume_Momentum', 'Price_Volume_Trend',  # Volume-based momentum
                'Trend_Strength', 'Volatility_Trend',  # Trend indicators
                # Hybrid features (2)
                'sentiment_score', 'garch_volatility'
            ]
            # Total: 42 features (was 27) - Enhanced for better directional accuracy
            
            # Verify all features exist
            missing_features = [col for col in feature_cols if col not in df.columns]
            if missing_features:
                raise Exception(f"Missing features: {missing_features}")
            
            self.update_status('preprocessing', 60, f'Normalizing {len(feature_cols)} features (including sentiment & GARCH)...')
            logger.info(f"Features to normalize: {feature_cols}")
            logger.info(f"Feature statistics before normalization:")
            for col in ['sentiment_score', 'garch_volatility']:
                if col in df.columns:
                    logger.info(f"  {col}: min={df[col].min():.6f}, max={df[col].max():.6f}, mean={df[col].mean():.6f}")
            
            # Normalize sentiment separately with fixed range (-1 to 1) to avoid identical normalized values
            # This ensures neutral (0.0) normalizes to 0.5, and dates with news have varied values
            # This prevents the issue where 127 dates with 0.0 all normalize to the same value (0.186216)
            sentiment_raw = df['sentiment_score'].copy()
            
            # Debug: Check sentiment values before normalization
            logger.info(f"BEFORE normalization - Sentiment stats: min={sentiment_raw.min():.6f}, max={sentiment_raw.max():.6f}, mean={sentiment_raw.mean():.6f}")
            logger.info(f"BEFORE normalization - Zero count: {(sentiment_raw == 0.0).sum()}, Non-zero count: {(sentiment_raw != 0.0).sum()}")
            if (sentiment_raw != 0.0).sum() > 0:
                non_zero_raw = sentiment_raw[sentiment_raw != 0.0]
                logger.info(f"BEFORE normalization - Non-zero values: min={non_zero_raw.min():.6f}, max={non_zero_raw.max():.6f}, unique={non_zero_raw.nunique()}")
            
            # Normalize sentiment from [-1, 1] to [0, 1] using fixed range
            # This way: -1 (most negative) → 0.0, 0 (neutral) → 0.5, +1 (most positive) → 1.0
            df['sentiment_score'] = (sentiment_raw + 1) / 2.0
            
            # Debug: Check sentiment values after normalization
            logger.info(f"AFTER normalization - Sentiment stats: min={df['sentiment_score'].min():.6f}, max={df['sentiment_score'].max():.6f}, mean={df['sentiment_score'].mean():.6f}")
            logger.info(f"AFTER normalization - 0.5 count (neutral): {(df['sentiment_score'] == 0.5).sum()}, Non-0.5 count: {(df['sentiment_score'] != 0.5).sum()}")
            if (df['sentiment_score'] != 0.5).sum() > 0:
                non_05_normalized = df[df['sentiment_score'] != 0.5]['sentiment_score']
                logger.info(f"AFTER normalization - Non-0.5 values: min={non_05_normalized.min():.6f}, max={non_05_normalized.max():.6f}, unique={non_05_normalized.nunique()}")
            
            logger.info(f"Sentiment normalized separately: raw range [{sentiment_raw.min():.6f}, {sentiment_raw.max():.6f}] → normalized [0.0, 1.0]")
            logger.info(f"  Neutral (0.0) → 0.5, Non-zero values will vary (e.g., -0.18→0.41, +0.80→0.90)")
            
            # Normalize all other features together (excluding sentiment to preserve its separate normalization)
            feature_cols_without_sentiment = [col for col in feature_cols if col != 'sentiment_score']
            scaler = MinMaxScaler()
            df[feature_cols_without_sentiment] = scaler.fit_transform(df[feature_cols_without_sentiment])
            # Note: sentiment_score is already normalized above and excluded from MinMaxScaler
            
            self.update_status('preprocessing', 65, f'✓ Normalized {len(feature_cols)} features successfully')
            
            # Save scaler
            scaler_file = f"{self.temp_path}/scaler_{self.symbol}.pkl"
            with open(scaler_file, 'wb') as f:
                pickle.dump(scaler, f)
            
            # Final verification before saving CSV
            logger.info(f"FINAL CHECK before saving CSV - Sentiment stats: min={df['sentiment_score'].min():.6f}, max={df['sentiment_score'].max():.6f}, mean={df['sentiment_score'].mean():.6f}")
            logger.info(f"FINAL CHECK - 0.5 count: {(df['sentiment_score'] == 0.5).sum()}, Non-0.5 count: {(df['sentiment_score'] != 0.5).sum()}")
            if (df['sentiment_score'] != 0.5).sum() > 0:
                final_non_05 = df[df['sentiment_score'] != 0.5]['sentiment_score']
                logger.info(f"FINAL CHECK - Non-0.5 values: min={final_non_05.min():.6f}, max={final_non_05.max():.6f}, unique={final_non_05.nunique()}")
                logger.info(f"FINAL CHECK - Sample non-0.5 dates: {df[df['sentiment_score'] != 0.5][['Date', 'sentiment_score']].head(5).to_dict('records')}")
            
            # Save preprocessed data
            processed_file = f"{self.temp_path}/processed_data_{self.symbol}.csv"
            df.to_csv(processed_file, index=False)
            logger.info(f"CSV saved to {processed_file} with {len(df)} rows")
            
            self.update_status('preprocessing', 70, f'Preprocessed {len(df)} records')
            return df, scaler
            
        except Exception as e:
            error_msg = f"Preprocessing failed: {str(e)}"
            self.status['errors'].append(error_msg)
            logger.error(error_msg)
            raise
    
    def create_sequences(self, df: pd.DataFrame, lookback: int = 60) -> tuple:
        """Step 3: Create sequences for LSTM training with dual outputs (price + direction)"""
        self.update_status('preparing', 75, 'Creating sequences for training...')
        
        try:
            # Updated feature list with enhanced directional indicators
            feature_cols = [
                # Market data (5)
                'Open', 'High', 'Low', 'Close', 'Volume',
                # Basic technical indicators (6)
                'Returns', 'MA_5', 'MA_10', 'MA_20', 'Volatility', 'Momentum',
                # Advanced technical indicators (12)
                'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
                'BB_upper', 'BB_lower', 'BB_width', 'BB_position',
                'Momentum_10', 'Momentum_20',
                'Volume_ratio', 'High_Low_ratio',
                # MA crossovers (2)
                'MA5_MA10_cross', 'MA10_MA20_cross',
                # Additional directional indicators (15)
                'Stoch_K', 'Stoch_D',  # Stochastic Oscillator
                'Williams_R',  # Williams %R
                'CCI',  # Commodity Channel Index
                'ADX',  # Average Directional Index
                'ROC_10', 'ROC_20',  # Rate of Change
                'Price_to_MA5', 'Price_to_MA20',  # Price position relative to MAs
                'Price_to_High', 'Price_to_Low',  # Price position relative to range
                'Volume_Momentum', 'Price_Volume_Trend',  # Volume-based momentum
                'Trend_Strength', 'Volatility_Trend',  # Trend indicators
                # Hybrid features (2)
                'sentiment_score', 'garch_volatility'
            ]
            
            features = df[feature_cols].values
            close_idx = feature_cols.index('Close')
            
            X, y = [], []
            for i in range(lookback, len(features)):
                X.append(features[i-lookback:i])
                # Price target: current day's close (standard approach)
                y.append(features[i, close_idx])
            
            X = np.array(X)
            y = np.array(y)
            
            # Train/test split (80/20)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            self.update_status('preparing', 80, f'Created {len(X_train)} train and {len(X_test)} test sequences')
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            error_msg = f"Sequence creation failed: {str(e)}"
            self.status['errors'].append(error_msg)
            logger.error(error_msg)
            raise
    
    def train_model(self, X_train, X_test, y_train, y_test) -> tf.keras.Model:
        """Step 4: Train enhanced Hybrid LSTM model with 42 features for improved directional accuracy"""
        self.update_status('training', 85, 'Building optimized Hybrid LSTM model...')
        
        if not TF_AVAILABLE:
            raise Exception("TensorFlow not available")
        
        try:
            n_features = X_train.shape[2]
            lookback = X_train.shape[1]
            
            # Build enhanced model with attention mechanism for better accuracy
            self.update_status('training', 86, 'Building enhanced model architecture with attention...')
            
            # Use Functional API for attention mechanism
            inputs = Input(shape=(lookback, n_features), name='input')
            
            # First LSTM layer - capture long-term patterns
            lstm1 = LSTM(128, return_sequences=True, name='lstm_1')(inputs)
            lstm1 = Dropout(0.2, name='dropout_1')(lstm1)
            
            # Second LSTM layer - capture medium-term patterns
            lstm2 = LSTM(64, return_sequences=True, name='lstm_2')(lstm1)
            lstm2 = Dropout(0.2, name='dropout_2')(lstm2)
            
            # Third LSTM layer - final feature extraction
            lstm3 = LSTM(64, return_sequences=True, name='lstm_3')(lstm2)  # Increased from 32 to 64
            lstm3 = Dropout(0.2, name='dropout_3')(lstm3)
            
            # Attention mechanism - helps model focus on important time steps
            # Compute attention scores for each time step (batch, timesteps, features) -> (batch, timesteps, 1)
            attention_scores = Dense(1, activation='tanh', name='attention_scores')(lstm3)
            # Flatten to (batch, timesteps) and apply softmax to get attention weights
            attention_weights = Lambda(lambda x: tf.nn.softmax(tf.squeeze(x, axis=-1), axis=1), name='attention_weights')(attention_scores)
            # Expand dimensions to (batch, timesteps, 1) for broadcasting
            attention_weights = Lambda(lambda x: tf.expand_dims(x, axis=-1), name='attention_expand')(attention_weights)
            # Apply attention weights to LSTM outputs
            attended = Multiply(name='attention_apply')([lstm3, attention_weights])
            # Sum over time dimension to get weighted features (batch, features)
            attended = Lambda(lambda x: tf.reduce_sum(x, axis=1), name='attention_sum')(attended)
            
            # Dense layers for prediction (increased capacity for 42 features)
            dense1 = Dense(256, activation='relu', name='dense_1')(attended)  # Increased for 42 features
            dense1 = Dropout(0.25, name='dropout_4')(dense1)
            dense2 = Dense(128, activation='relu', name='dense_2')(dense1)  # Increased capacity
            dense2 = Dropout(0.2, name='dropout_5')(dense2)
            dense3 = Dense(64, activation='relu', name='dense_3')(dense2)  # Additional layer
            dense3 = Dropout(0.15, name='dropout_6')(dense3)
            dense4 = Dense(32, activation='relu', name='dense_4')(dense3)  # Final feature extraction
            dense4 = Dropout(0.1, name='dropout_7')(dense4)
            output = Dense(1, name='output')(dense4)
            
            model = tf.keras.Model(inputs=inputs, outputs=output, name='enhanced_lstm_attention')
            
            # Compile with Huber loss (more robust to outliers) and optimized learning rate
            optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)  # Slightly higher LR for faster convergence
            model.compile(
                optimizer=optimizer,
                loss=tf.keras.losses.Huber(delta=1.0),  # Huber loss is more robust to outliers than MSE
                metrics=['mae', 'mse']
            )
            
            # Enhanced callbacks with better settings for improved accuracy
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=30,  # Increased patience for better convergence
                    restore_best_weights=True,
                    min_delta=1e-6,  # Smaller delta for finer convergence
                    verbose=0
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.3,  # More aggressive LR reduction
                    patience=8,  # Reduced patience for faster adaptation
                    min_lr=1e-8,  # Lower minimum LR
                    verbose=0
                ),
                ModelCheckpoint(
                    filepath=f"{self.temp_path}/best_model_{self.symbol}.h5",
                    monitor='val_loss',
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=0
                )
            ]
            
            # Train model with more epochs and better early stopping
            # Adjust batch size based on dataset size for optimal training
            dataset_size = len(X_train)
            if dataset_size > 400:
                batch_size = 64  # Larger batch for large datasets
                epochs = 200  # More epochs for large datasets
                self.update_status('training', 90, f'Training enhanced model ({dataset_size} sequences, batch_size={batch_size}, epochs={epochs})...')
            elif dataset_size > 200:
                batch_size = 48
                epochs = 180
                self.update_status('training', 90, f'Training enhanced model ({dataset_size} sequences, batch_size={batch_size}, epochs={epochs})...')
            else:
                batch_size = 32
                epochs = 150
                self.update_status('training', 90, f'Training enhanced model ({dataset_size} sequences, batch_size={batch_size}, epochs={epochs})...')
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                callbacks=callbacks
            )
            
            # Load best model weights if checkpoint was saved
            best_model_path = f"{self.temp_path}/best_model_{self.symbol}.h5"
            if os.path.exists(best_model_path):
                model = tf.keras.models.load_model(best_model_path)
                logger.info("Loaded best model from checkpoint for improved accuracy")
            
            # Log final metrics
            final_val_loss = history.history.get('val_loss', [0])
            final_val_mae = history.history.get('val_mae', [0])
            if final_val_loss:
                logger.info(f"Final validation loss: {final_val_loss[-1]:.6f}")
            if final_val_mae:
                logger.info(f"Final validation MAE: {final_val_mae[-1]:.6f}")
            
            # Save model
            model_file = f"{self.temp_path}/model_{self.symbol}.h5"
            model.save(model_file)
            
            self.update_status('training', 95, 'Model training completed')
            return model
            
        except Exception as e:
            error_msg = f"Model training failed: {str(e)}"
            self.status['errors'].append(error_msg)
            logger.error(error_msg)
            raise
    
    def predict_next_day(self, model: tf.keras.Model, df: pd.DataFrame, scaler: MinMaxScaler, raw_df: pd.DataFrame = None) -> Dict[str, Any]:
        """Step 5: Make prediction for next day"""
        self.update_status('predicting', 98, 'Making prediction...')
        
        try:
            lookback = 60
            # Updated feature list with enhanced directional indicators
            feature_cols = [
                # Market data (5)
                'Open', 'High', 'Low', 'Close', 'Volume',
                # Basic technical indicators (6)
                'Returns', 'MA_5', 'MA_10', 'MA_20', 'Volatility', 'Momentum',
                # Advanced technical indicators (12)
                'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
                'BB_upper', 'BB_lower', 'BB_width', 'BB_position',
                'Momentum_10', 'Momentum_20',
                'Volume_ratio', 'High_Low_ratio',
                # MA crossovers (2)
                'MA5_MA10_cross', 'MA10_MA20_cross',
                # Additional directional indicators (15)
                'Stoch_K', 'Stoch_D',  # Stochastic Oscillator
                'Williams_R',  # Williams %R
                'CCI',  # Commodity Channel Index
                'ADX',  # Average Directional Index
                'ROC_10', 'ROC_20',  # Rate of Change
                'Price_to_MA5', 'Price_to_MA20',  # Price position relative to MAs
                'Price_to_High', 'Price_to_Low',  # Price position relative to range
                'Volume_Momentum', 'Price_Volume_Trend',  # Volume-based momentum
                'Trend_Strength', 'Volatility_Trend',  # Trend indicators
                # Hybrid features (2)
                'sentiment_score', 'garch_volatility'
            ]
            
            features = df[feature_cols].values
            
            if len(features) < lookback:
                raise Exception(f"Need at least {lookback} days of data, got {len(features)}")
            
            # Get last sequence
            X_pred = features[-lookback:].reshape(1, lookback, len(feature_cols))
            
            # Predict (single-output: price)
            prediction_normalized = model.predict(X_pred, verbose=0)[0][0]
            
            # Inverse transform prediction
            # Note: scaler was fitted on features WITHOUT sentiment, so we need to exclude sentiment
            feature_cols_without_sentiment = [col for col in feature_cols if col != 'sentiment_score']
            close_idx_scaler = feature_cols_without_sentiment.index('Close')
            dummy_pred = np.zeros((1, len(feature_cols_without_sentiment)))
            dummy_pred[0, close_idx_scaler] = prediction_normalized
            predicted_close = scaler.inverse_transform(dummy_pred)[0, close_idx_scaler]
            
            # Get last actual close - use raw data if available for accuracy
            if raw_df is not None and len(raw_df) > 0:
                # Use raw data for accurate last close (not normalized)
                raw_df_sorted = raw_df.sort_values('Date').reset_index(drop=True)
                # Normalize dates to timezone-naive for comparison
                raw_df_sorted['Date'] = pd.to_datetime(raw_df_sorted['Date']).dt.tz_localize(None)
                # Filter to end_date or earlier to ensure we get the correct last date
                raw_df_filtered = raw_df_sorted[raw_df_sorted['Date'] <= self.end_date]
                if len(raw_df_filtered) > 0:
                    last_close = float(raw_df_filtered['Close'].iloc[-1])
                    last_date = raw_df_filtered['Date'].iloc[-1]
                    logger.info(f"Using raw data for last_close: ₹{last_close:.2f} on {last_date}")
                else:
                    # Fallback to last available
                    last_close = float(raw_df_sorted['Close'].iloc[-1])
                    logger.info(f"Using last available raw data for last_close: ₹{last_close:.2f}")
            else:
                # Fallback: inverse transform from normalized data
                close_idx = feature_cols.index('Close')
                last_close_normalized = features[-1, close_idx]
                # Scaler expects features without sentiment
                dummy_last = np.zeros((1, len(feature_cols_without_sentiment)))
                dummy_last[0, close_idx_scaler] = last_close_normalized
                last_close = scaler.inverse_transform(dummy_last)[0, close_idx_scaler]
                logger.info(f"Using inverse transform for last_close: {last_close:.2f}")
            
            # Calculate metrics on test set
            X_test = features[:-1]
            close_idx = feature_cols.index('Close')  # Index in full feature list (with sentiment)
            y_test = features[1:, close_idx]
            
            test_sequences = []
            test_targets = []
            for i in range(lookback, len(X_test)):
                test_sequences.append(X_test[i-lookback:i])
                test_targets.append(y_test[i])
            
            if len(test_sequences) > 0:
                X_eval = np.array(test_sequences)
                y_eval = np.array(test_targets)
                y_pred_eval = model.predict(X_eval, verbose=0).flatten()
                
                # Inverse transform for metrics
                # Scaler expects features without sentiment
                y_eval_actual = []
                y_pred_actual = []
                for i in range(len(y_eval)):
                    dummy_eval = np.zeros((1, len(feature_cols_without_sentiment)))
                    dummy_eval[0, close_idx_scaler] = y_eval[i]
                    y_eval_actual.append(scaler.inverse_transform(dummy_eval)[0, close_idx_scaler])
                    
                    dummy_pred_eval = np.zeros((1, len(feature_cols_without_sentiment)))
                    dummy_pred_eval[0, close_idx_scaler] = y_pred_eval[i]
                    y_pred_actual.append(scaler.inverse_transform(dummy_pred_eval)[0, close_idx_scaler])
                
                y_eval_actual = np.array(y_eval_actual)
                y_pred_actual = np.array(y_pred_actual)
                
                rmse = np.sqrt(mean_squared_error(y_eval_actual, y_pred_actual))
                mae = mean_absolute_error(y_eval_actual, y_pred_actual)
                mape = np.mean(np.abs((y_eval_actual - y_pred_actual) / y_eval_actual)) * 100
                r2 = r2_score(y_eval_actual, y_pred_actual)
                
                # Directional accuracy: calculate from price movements
                direction_actual = np.diff(y_eval_actual) > 0
                direction_pred = np.diff(y_pred_actual) > 0
                directional_accuracy = np.mean(direction_actual == direction_pred) * 100
                logger.info(f"Directional accuracy: {directional_accuracy:.2f}%")
            else:
                rmse = mae = mape = r2 = directional_accuracy = 0.0
            
            # Prepare recent data for chart - use raw data if available for accuracy
            recent_data = []
            if raw_df is not None and len(raw_df) > 0:
                # Use raw data for accurate prices (not normalized)
                raw_df_sorted = raw_df.sort_values('Date').reset_index(drop=True)
                # Normalize dates to timezone-naive
                raw_df_sorted['Date'] = pd.to_datetime(raw_df_sorted['Date']).dt.tz_localize(None)
                recent_window_raw = raw_df_sorted.tail(30)
                for _, row in recent_window_raw.iterrows():
                    date_str = pd.to_datetime(row['Date']).strftime('%Y-%m-%d')
                    recent_data.append({
                        'date': date_str,
                        'close': float(row['Close']),
                        'predicted': False
                    })
            else:
                # Fallback: inverse transform from normalized data
                for i in range(max(0, len(df) - 30), len(df)):
                    date_str = df.iloc[i]['Date'].strftime('%Y-%m-%d')
                    close_normalized = df.iloc[i]['Close']
                    # Scaler expects features without sentiment
                    dummy_recent = np.zeros((1, len(feature_cols_without_sentiment)))
                    dummy_recent[0, close_idx_scaler] = close_normalized
                    close_actual = scaler.inverse_transform(dummy_recent)[0, close_idx_scaler]
                    
                    recent_data.append({
                        'date': date_str,
                        'close': float(close_actual),
                        'predicted': False
                    })
            
            # Determine currency
            indian_stocks = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'HINDUNILVR',
                           'BHARTIARTL', 'KOTAKBANK', 'SBIN', 'ITC', 'AXISBANK']
            currency = '₹' if self.symbol in indian_stocks else '$'
            
            # Log news articles count for debugging
            logger.info(f"Returning {len(self.news_articles)} news articles in result")
            
            result = {
                'symbol': self.symbol,
                'predicted_close': float(predicted_close),
                'last_close': float(last_close),
                'currency': currency,
                'rmse': float(rmse),
                'mae': float(mae),
                'mape': float(mape),
                'r2': float(r2),
                'directional_accuracy': float(directional_accuracy),
                'date_predicted_for': (self.end_date + timedelta(days=1)).strftime('%Y-%m-%d'),
                'recent_data': recent_data,
                'news_articles': self.news_articles if self.news_articles else []  # Always include, even if empty
            }
            
            self.update_status('complete', 100, 'Pipeline completed successfully')
            return result
            
        except Exception as e:
            error_msg = f"Prediction failed: {str(e)}"
            self.status['errors'].append(error_msg)
            logger.error(error_msg)
            raise
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """Run the complete pipeline"""
        try:
            # Step 1: Collect data
            raw_df = self.collect_stock_data()
            
            # Step 2: Preprocess
            processed_df, scaler = self.preprocess_data(raw_df)
            
            # Step 3: Create sequences
            X_train, X_test, y_train, y_test = self.create_sequences(processed_df)
            
            # Step 4: Train model (single-output, optimized)
            model = self.train_model(X_train, X_test, y_train, y_test)
            
            # Step 5: Predict (pass raw_df for accurate last_close)
            result = self.predict_next_day(model, processed_df, scaler, raw_df)
            
            # Add status to result
            result['status'] = self.status
            
            return result
            
        except Exception as e:
            self.status['step'] = 'error'
            self.status['message'] = f"Pipeline failed: {str(e)}"
            return {
                'status': self.status,
                'error': str(e)
            }

