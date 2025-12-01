import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import data collection libraries
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    logger.warning("yfinance not available. Please install: pip install yfinance")
    YFINANCE_AVAILABLE = False

try:
    from nsepy import get_history
    NSEPY_AVAILABLE = True
except ImportError:
    logger.warning("NSEpy not available. Please install: pip install nsepy")
    NSEPY_AVAILABLE = False


class ExtendedDataCollector:
    def __init__(self, output_path='data/extended'):
        self.output_path = output_path
        self.raw_path = f"{output_path}/raw"
        self.processed_path = f"{output_path}/processed"
        
        # Create output directories
        os.makedirs(self.raw_path, exist_ok=True)
        os.makedirs(self.processed_path, exist_ok=True)
        
        # Define symbols to collect
        self.indian_stocks = [
            'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'HINDUNILVR',
            'BHARTIARTL', 'KOTAKBANK', 'SBIN', 'ITC', 'AXISBANK'
        ]
        
        self.global_stocks = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 
            'META', 'NFLX', 'ADBE', 'CRM'
        ]
        
        self.indices = [
            '^GSPC',  # S&P 500
            '^IXIC',  # NASDAQ
            '^DJI',   # Dow Jones
            '^FTSE',  # FTSE 100
            '^GDAXI', # DAX
            '^N225',  # Nikkei 225
            '^HSI'    # Hang Seng
        ]
        
        self.crypto = ['BTC-USD', 'ETH-USD']
        self.commodities = ['GC=F', 'CL=F']  # Gold, Crude Oil
        
        # Date range: 6 months of historical data
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=180)  # 6 months
        
        logger.info(f"Data collection period: {self.start_date.date()} to {self.end_date.date()}")
        
    def collect_yfinance_data(self, symbol, max_retries=3):
        """Collect data using yfinance with retries"""
        for attempt in range(max_retries):
            try:
                logger.info(f"  Attempting to fetch {symbol} (attempt {attempt + 1}/{max_retries})...")
                
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=self.start_date, end=self.end_date)
                
                if df.empty:
                    logger.warning(f"  No data returned for {symbol}")
                    continue
                
                # Reset index to get Date as column
                df.reset_index(inplace=True)
                
                # Standardize column names
                df.rename(columns={
                    'Open': 'Open',
                    'High': 'High',
                    'Low': 'Low',
                    'Close': 'Close',
                    'Volume': 'Volume'
                }, inplace=True)
                
                # Add symbol column
                df['Symbol'] = symbol
                df['Source'] = 'yfinance'
                
                # Select relevant columns
                df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Symbol', 'Source']]
                
                logger.info(f"  ✓ Collected {len(df)} records for {symbol}")
                return df
                
            except Exception as e:
                logger.warning(f"  Attempt {attempt + 1} failed for {symbol}: {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2)
                continue
        
        logger.error(f"  ✗ Failed to collect data for {symbol} after {max_retries} attempts")
        return None
    
    def collect_nsepy_data(self, symbol):
        """Collect Indian stock data using NSEpy"""
        try:
            logger.info(f"  Fetching {symbol} from NSE...")
            
            df = get_history(
                symbol=symbol,
                start=self.start_date,
                end=self.end_date
            )
            
            if df.empty:
                logger.warning(f"  No data returned for {symbol}")
                return None
            
            # Reset index
            df.reset_index(inplace=True)
            
            # Standardize column names
            df.rename(columns={
                'Date': 'Date',
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume'
            }, inplace=True)
            
            # Add metadata
            df['Symbol'] = symbol
            df['Source'] = 'NSE'
            
            # Select relevant columns
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Symbol', 'Source']]
            
            logger.info(f"  ✓ Collected {len(df)} records for {symbol}")
            return df
            
        except Exception as e:
            logger.warning(f"  NSEpy failed for {symbol}: {e}")
            return None
    
    def collect_all_data(self):
        """Collect data for all symbols"""
        logger.info("=" * 80)
        logger.info("EXTENDED DATA COLLECTION - 6 MONTHS HISTORICAL DATA")
        logger.info("=" * 80)
        
        all_data = []
        
        # Collect Indian stocks
        if NSEPY_AVAILABLE:
            logger.info(f"\n📊 Collecting Indian Stocks (NSE) - {len(self.indian_stocks)} symbols")
            logger.info("-" * 80)
            
            for symbol in self.indian_stocks:
                # Try NSEpy first
                df = self.collect_nsepy_data(symbol)
                
                # Fallback to yfinance with .NS suffix
                if df is None and YFINANCE_AVAILABLE:
                    logger.info(f"  Trying yfinance fallback for {symbol}...")
                    df = self.collect_yfinance_data(f"{symbol}.NS")
                    if df is not None:
                        df['Symbol'] = symbol
                        df['Source'] = 'yfinance_NSE'
                
                if df is not None:
                    all_data.append(df)
        
        # Collect global stocks
        if YFINANCE_AVAILABLE:
            logger.info(f"\n📊 Collecting Global Stocks - {len(self.global_stocks)} symbols")
            logger.info("-" * 80)
            
            for symbol in self.global_stocks:
                df = self.collect_yfinance_data(symbol)
                if df is not None:
                    all_data.append(df)
            
            # Collect indices
            logger.info(f"\n📊 Collecting Market Indices - {len(self.indices)} symbols")
            logger.info("-" * 80)
            
            for symbol in self.indices:
                df = self.collect_yfinance_data(symbol)
                if df is not None:
                    all_data.append(df)
            
            # Collect crypto
            logger.info(f"\n📊 Collecting Cryptocurrencies - {len(self.crypto)} symbols")
            logger.info("-" * 80)
            
            for symbol in self.crypto:
                df = self.collect_yfinance_data(symbol)
                if df is not None:
                    all_data.append(df)
            
            # Collect commodities
            logger.info(f"\n📊 Collecting Commodities - {len(self.commodities)} symbols")
            logger.info("-" * 80)
            
            for symbol in self.commodities:
                df = self.collect_yfinance_data(symbol)
                if df is not None:
                    all_data.append(df)
        
        return all_data
    
    def save_data(self, all_data):
        """Save collected data in multiple formats"""
        logger.info("\n" + "=" * 80)
        logger.info("SAVING COLLECTED DATA")
        logger.info("=" * 80)
        
        if not all_data:
            logger.error("No data to save!")
            return
        
        # Combine all data
        logger.info("\nCombining all collected data...")
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Sort by symbol and date
        combined_df.sort_values(['Symbol', 'Date'], inplace=True)
        
        # Remove duplicates
        initial_rows = len(combined_df)
        combined_df.drop_duplicates(subset=['Symbol', 'Date'], keep='first', inplace=True)
        duplicates_removed = initial_rows - len(combined_df)
        
        logger.info(f"  - Total records: {len(combined_df):,}")
        logger.info(f"  - Unique symbols: {combined_df['Symbol'].nunique()}")
        logger.info(f"  - Duplicates removed: {duplicates_removed}")
        logger.info(f"  - Date range: {combined_df['Date'].min()} to {combined_df['Date'].max()}")
        
        # Save as CSV
        csv_file = f"{self.raw_path}/extended_stock_data_{datetime.now().strftime('%Y%m%d')}.csv"
        combined_df.to_csv(csv_file, index=False)
        logger.info(f"\n✓ CSV saved: {csv_file}")
        logger.info(f"  Size: {os.path.getsize(csv_file) / 1024:.2f} KB")
        
        # Save as Parquet (more efficient)
        try:
            parquet_file = f"{self.raw_path}/extended_stock_data_{datetime.now().strftime('%Y%m%d')}.parquet"
            combined_df.to_parquet(parquet_file, index=False, compression='snappy')
            logger.info(f"\n✓ Parquet saved: {parquet_file}")
            logger.info(f"  Size: {os.path.getsize(parquet_file) / 1024:.2f} KB")
        except Exception as e:
            logger.warning(f"Could not save Parquet format: {e}")
        
        # Save individual symbol files
        logger.info("\n📁 Saving individual symbol files...")
        symbols_dir = f"{self.raw_path}/by_symbol"
        os.makedirs(symbols_dir, exist_ok=True)
        
        for symbol in combined_df['Symbol'].unique():
            symbol_df = combined_df[combined_df['Symbol'] == symbol].copy()
            symbol_file = f"{symbols_dir}/{symbol.replace('^', '').replace('=', '_')}.csv"
            symbol_df.to_csv(symbol_file, index=False)
        
        logger.info(f"  ✓ Saved {combined_df['Symbol'].nunique()} individual symbol files")
        
        return combined_df
    
    def generate_statistics(self, df):
        """Generate and display statistics"""
        logger.info("\n" + "=" * 80)
        logger.info("DATA COLLECTION STATISTICS")
        logger.info("=" * 80)
        
        logger.info("\n📊 OVERALL STATISTICS:")
        logger.info("-" * 80)
        logger.info(f"Total Records: {len(df):,}")
        logger.info(f"Unique Symbols: {df['Symbol'].nunique()}")
        logger.info(f"Date Range: {df['Date'].min()} to {df['Date'].max()}")
        logger.info(f"Total Days: {(df['Date'].max() - df['Date'].min()).days}")
        
        logger.info("\n📊 RECORDS PER SYMBOL (Top 20):")
        logger.info("-" * 80)
        symbol_counts = df.groupby('Symbol').size().sort_values(ascending=False)
        for symbol, count in symbol_counts.head(20).items():
            logger.info(f"  {symbol:15s}: {count:4d} records")
        
        logger.info("\n📊 DATA COMPLETENESS:")
        logger.info("-" * 80)
        logger.info(f"Missing Open: {df['Open'].isna().sum()}")
        logger.info(f"Missing High: {df['High'].isna().sum()}")
        logger.info(f"Missing Low: {df['Low'].isna().sum()}")
        logger.info(f"Missing Close: {df['Close'].isna().sum()}")
        logger.info(f"Missing Volume: {df['Volume'].isna().sum()}")
        
        logger.info("\n📊 PRICE STATISTICS:")
        logger.info("-" * 80)
        logger.info(f"Mean Close Price: ${df['Close'].mean():.2f}")
        logger.info(f"Median Close Price: ${df['Close'].median():.2f}")
        logger.info(f"Min Close Price: ${df['Close'].min():.2f}")
        logger.info(f"Max Close Price: ${df['Close'].max():.2f}")
        
        logger.info("\n📊 DATA SOURCES:")
        logger.info("-" * 80)
        source_counts = df.groupby('Source').size()
        for source, count in source_counts.items():
            logger.info(f"  {source:20s}: {count:6d} records ({count/len(df)*100:.1f}%)")
        
        # Create summary report
        summary_file = f"{self.output_path}/data_collection_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("EXTENDED DATA COLLECTION SUMMARY\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("1. COLLECTION PERIOD\n")
            f.write("-" * 80 + "\n")
            f.write(f"Start Date: {self.start_date.date()}\n")
            f.write(f"End Date: {self.end_date.date()}\n")
            f.write(f"Duration: 6 months\n\n")
            
            f.write("2. OVERALL STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Records: {len(df):,}\n")
            f.write(f"Unique Symbols: {df['Symbol'].nunique()}\n")
            f.write(f"Actual Date Range: {df['Date'].min()} to {df['Date'].max()}\n")
            f.write(f"Total Days Covered: {(df['Date'].max() - df['Date'].min()).days}\n\n")
            
            f.write("3. RECORDS PER SYMBOL\n")
            f.write("-" * 80 + "\n")
            for symbol, count in symbol_counts.items():
                f.write(f"  {symbol:15s}: {count:4d} records\n")
            f.write("\n")
            
            f.write("4. DATA SOURCES\n")
            f.write("-" * 80 + "\n")
            for source, count in source_counts.items():
                f.write(f"  {source:20s}: {count:6d} records ({count/len(df)*100:.1f}%)\n")
            f.write("\n")
            
            f.write("5. OUTPUT FILES\n")
            f.write("-" * 80 + "\n")
            f.write(f"  - {self.raw_path}/extended_stock_data_*.csv\n")
            f.write(f"  - {self.raw_path}/extended_stock_data_*.parquet\n")
            f.write(f"  - {self.raw_path}/by_symbol/*.csv (individual files)\n")
            f.write(f"  - {summary_file}\n")
        
        logger.info(f"\n✓ Summary report saved: {summary_file}")


def main():
    """Main data collection pipeline"""
    logger.info("\n" + "=" * 80)
    logger.info("EXTENDED DATA COLLECTION PIPELINE")
    logger.info("6 MONTHS HISTORICAL OHLCV DATA")
    logger.info("=" * 80 + "\n")
    
    if not YFINANCE_AVAILABLE:
        logger.error("yfinance is required for this script. Please install: pip install yfinance")
        return
    
    # Initialize collector
    collector = ExtendedDataCollector()
    
    # Collect all data
    all_data = collector.collect_all_data()
    
    if not all_data:
        logger.error("\nNo data collected! Check API connections and symbol availability.")
        return
    
    # Save data
    combined_df = collector.save_data(all_data)
    
    # Generate statistics
    collector.generate_statistics(combined_df)
    
    logger.info("\n" + "=" * 80)
    logger.info("DATA COLLECTION COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info(f"\nCollected data for {combined_df['Symbol'].nunique()} symbols")
    logger.info(f"Total records: {len(combined_df):,}")
    logger.info(f"Ready for preprocessing and model training!\n")


if __name__ == "__main__":
    main()

