"""
Add Direction-Specific Technical Indicators
Adds RSI, MACD, and Bollinger Bands to improve directional prediction
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_rsi(df, period=14):
    """Calculate Relative Strength Index"""
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(df, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def calculate_bollinger_bands(df, period=20, num_std=2):
    """Calculate Bollinger Bands"""
    sma = df['Close'].rolling(window=period).mean()
    std = df['Close'].rolling(window=period).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    width = (upper - lower) / sma
    position = (df['Close'] - lower) / (upper - lower)  # 0 = lower band, 1 = upper band
    return upper, lower, width, position


def add_direction_features(input_file, output_file):
    """Add direction-specific technical indicators to dataset"""
    logger.info("=" * 80)
    logger.info("ADDING DIRECTION-SPECIFIC TECHNICAL INDICATORS")
    logger.info("=" * 80)
    
    # Load data
    logger.info(f"Loading data from: {input_file}")
    df = pd.read_csv(input_file)
    df['Date'] = pd.to_datetime(df['Date'])
    
    logger.info(f"✓ Loaded {len(df):,} records")
    logger.info(f"✓ Symbols: {df['Symbol'].nunique()}")
    logger.info("")
    
    # Process each symbol separately
    logger.info("Processing indicators for each symbol...")
    all_features = []
    
    for symbol in df['Symbol'].unique():
        symbol_data = df[df['Symbol'] == symbol].copy().sort_values('Date')
        
        if len(symbol_data) < 30:  # Need at least 30 days for indicators
            logger.warning(f"  ⚠️  {symbol}: Insufficient data ({len(symbol_data)} days)")
            all_features.append(symbol_data)
            continue
        
        # Calculate RSI
        symbol_data['RSI'] = calculate_rsi(symbol_data)
        
        # Calculate MACD
        macd, macd_signal, macd_hist = calculate_macd(symbol_data)
        symbol_data['MACD'] = macd
        symbol_data['MACD_signal'] = macd_signal
        symbol_data['MACD_hist'] = macd_hist
        
        # Calculate Bollinger Bands
        bb_upper, bb_lower, bb_width, bb_position = calculate_bollinger_bands(symbol_data)
        symbol_data['BB_upper'] = bb_upper
        symbol_data['BB_lower'] = bb_lower
        symbol_data['BB_width'] = bb_width
        symbol_data['BB_position'] = bb_position
        
        # Additional direction features
        # Price momentum (enhanced)
        symbol_data['price_momentum_5'] = symbol_data['Close'].pct_change(5)
        symbol_data['price_momentum_10'] = symbol_data['Close'].pct_change(10)
        
        # Volume-price relationship
        symbol_data['volume_price_trend'] = (symbol_data['Volume'] * symbol_data['Close']).pct_change()
        
        # Support/Resistance levels
        symbol_data['price_vs_20d_high'] = symbol_data['Close'] / symbol_data['High'].rolling(20).max()
        symbol_data['price_vs_20d_low'] = symbol_data['Close'] / symbol_data['Low'].rolling(20).min()
        
        all_features.append(symbol_data)
        logger.info(f"  ✓ {symbol}: Added {len([c for c in symbol_data.columns if c not in df.columns])} new features")
    
    # Combine all symbols
    df_enhanced = pd.concat(all_features, ignore_index=True)
    df_enhanced = df_enhanced.sort_values(['Symbol', 'Date']).reset_index(drop=True)
    
    # Fill NaN values (from rolling windows)
    new_features = ['RSI', 'MACD', 'MACD_signal', 'MACD_hist', 
                    'BB_upper', 'BB_lower', 'BB_width', 'BB_position',
                    'price_momentum_5', 'price_momentum_10', 
                    'volume_price_trend', 'price_vs_20d_high', 'price_vs_20d_low']
    
    for feature in new_features:
        if feature in df_enhanced.columns:
            df_enhanced[feature] = df_enhanced.groupby('Symbol')[feature].fillna(method='bfill').fillna(method='ffill')
    
    # Remove rows where indicators couldn't be calculated
    df_enhanced = df_enhanced.dropna(subset=['RSI', 'MACD'])
    
    logger.info("")
    logger.info(f"✓ Enhanced dataset: {len(df_enhanced):,} records")
    logger.info(f"✓ New features added: {len(new_features)}")
    logger.info(f"✓ Total features: {len(df_enhanced.columns)}")
    logger.info("")
    
    # Save enhanced dataset
    logger.info(f"Saving to: {output_file}")
    df_enhanced.to_csv(output_file, index=False)
    logger.info(f"✓ Saved successfully")
    logger.info("")
    
    # Summary statistics
    logger.info("📊 Feature Statistics:")
    logger.info("-" * 80)
    for feature in new_features:
        if feature in df_enhanced.columns:
            logger.info(f"  {feature:25s}: Mean={df_enhanced[feature].mean():8.4f}, "
                       f"Std={df_enhanced[feature].std():8.4f}, "
                       f"Range=[{df_enhanced[feature].min():6.2f}, {df_enhanced[feature].max():6.2f}]")
    logger.info("")
    
    return df_enhanced


def main():
    logger.info("\n" + "=" * 80)
    logger.info("ADD DIRECTION-SPECIFIC FEATURES")
    logger.info("=" * 80)
    logger.info("Adding RSI, MACD, Bollinger Bands, and other direction indicators")
    logger.info("=" * 80 + "\n")
    
    # File paths
    input_file = 'data/extended/processed/hybrid_data_with_sentiment_volatility.csv'
    output_file = 'data/extended/processed/hybrid_data_with_direction_features.csv'
    
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        logger.error("Please run preprocessing and sentiment/volatility integration first")
        return
    
    # Add features
    df_enhanced = add_direction_features(input_file, output_file)
    
    # Final summary
    logger.info("=" * 80)
    logger.info("🎉 DIRECTION FEATURES ADDED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info(f"\n📦 Output:")
    logger.info(f"   - Enhanced dataset: {output_file}")
    logger.info(f"   - New features: {len([c for c in df_enhanced.columns if 'RSI' in c or 'MACD' in c or 'BB' in c])}")
    logger.info(f"   - Total records: {len(df_enhanced):,}")
    logger.info("\n✅ Ready for model retraining with direction features!")
    logger.info("=" * 80 + "\n")


if __name__ == "__main__":
    main()


