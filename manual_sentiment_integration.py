"""
Manual FinBERT Sentiment Integration
Bypasses dependency issues by using pre-computed sentiment scores
"""

import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 80)
    logger.info("MANUAL FINBERT SENTIMENT INTEGRATION")
    logger.info("Using pre-computed sentiment scores from Prompt 3")
    logger.info("=" * 80 + "\n")
    
    # Step 1: Load sentiment scores
    logger.info("STEP 1: Loading Pre-computed Sentiment Scores")
    logger.info("-" * 80)
    
    sentiment_file = 'sample_run_output/datafiles/sentiment_extraction/detailed_sentiment_scores.csv'
    sentiment_df = pd.read_csv(sentiment_file)
    
    # Extract date from created_at
    sentiment_df['date'] = pd.to_datetime(sentiment_df['created_at']).dt.date
    
    # Remove any NaN dates
    initial_count = len(sentiment_df)
    sentiment_df = sentiment_df[sentiment_df['date'].notna()]
    
    logger.info(f"✓ Loaded {len(sentiment_df)} sentiment scores (removed {initial_count - len(sentiment_df)} with invalid dates)")
    logger.info(f"  - Date range: {sentiment_df['date'].min()} to {sentiment_df['date'].max()}")
    logger.info(f"  - Mean sentiment: {sentiment_df['sentiment_score'].mean():.4f}")
    
    # Aggregate by date (daily average)
    daily_sentiment = sentiment_df.groupby('date').agg({
        'sentiment_score': 'mean',
        'text': 'count'
    }).reset_index()
    daily_sentiment.columns = ['date', 'sentiment_score', 'text_count']
    
    logger.info(f"\n✓ Aggregated to {len(daily_sentiment)} daily scores\n")
    
    # Step 2: Load extended dataset
    logger.info("STEP 2: Loading Extended Dataset")
    logger.info("-" * 80)
    
    scaled_file = 'data/extended/processed/scaled_data_with_features.csv'
    scaled_data = pd.read_csv(scaled_file)
    
    # Convert Date to date only
    scaled_data['date'] = pd.to_datetime(scaled_data['Date'], utc=True).dt.date
    
    logger.info(f"✓ Loaded {len(scaled_data)} records")
    logger.info(f"  - Symbols: {scaled_data['Symbol'].nunique()}")
    logger.info(f"  - Date range: {scaled_data['date'].min()} to {scaled_data['date'].max()}\n")
    
    # Step 3: Merge sentiment
    logger.info("STEP 3: Merging Sentiment with Dataset")
    logger.info("-" * 80)
    
    hybrid_data = scaled_data.merge(
        daily_sentiment[['date', 'sentiment_score', 'text_count']],
        on='date',
        how='left'
    )
    
    # Fill missing sentiment with 0 (neutral)
    hybrid_data['sentiment_score'].fillna(0, inplace=True)
    hybrid_data['text_count'].fillna(0, inplace=True)
    
    # Drop temporary date column
    hybrid_data = hybrid_data.drop('date', axis=1)
    
    records_with_sentiment = (hybrid_data['sentiment_score'] != 0).sum()
    records_without = (hybrid_data['sentiment_score'] == 0).sum()
    
    logger.info(f"✓ Merge completed")
    logger.info(f"  - Total records: {len(hybrid_data)}")
    logger.info(f"  - With sentiment: {records_with_sentiment}")
    logger.info(f"  - Without sentiment (filled with 0): {records_without}\n")
    
    # Step 4: Save hybrid data
    logger.info("STEP 4: Saving Hybrid Data")
    logger.info("-" * 80)
    
    output_dir = 'data/extended/processed'
    os.makedirs(output_dir, exist_ok=True)
    
    hybrid_csv = f'{output_dir}/hybrid_data_with_sentiment.csv'
    hybrid_data.to_csv(hybrid_csv, index=False)
    
    logger.info(f"✓ Saved hybrid data: {hybrid_csv}")
    logger.info(f"  - Shape: {hybrid_data.shape}")
    logger.info(f"  - Size: {os.path.getsize(hybrid_csv) / 1024:.2f} KB\n")
    
    # Step 5: Create sequences (60, 12)
    logger.info("STEP 5: Creating Updated Sequences (60, 12)")
    logger.info("-" * 80)
    
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 
                   'Returns', 'MA_5', 'MA_10', 'MA_20', 'Volatility', 'Momentum',
                   'sentiment_score']  # 12th feature
    
    lookback = 60
    train_split = 0.8
    sequences = {}
    
    for symbol in sorted(hybrid_data['Symbol'].unique()):
        symbol_data = hybrid_data[hybrid_data['Symbol'] == symbol].copy()
        symbol_data = symbol_data.sort_values('Date')
        
        if len(symbol_data) <= lookback:
            logger.warning(f"✗ {symbol}: Insufficient data ({len(symbol_data)} records)")
            continue
        
        # Extract features
        feature_matrix = symbol_data[feature_cols].values
        dates = symbol_data['Date'].values
        
        # Create sequences
        X, y, date_indices = [], [], []
        
        for i in range(lookback, len(feature_matrix)):
            X.append(feature_matrix[i-lookback:i])
            y.append(feature_matrix[i, 3])  # Close price
            date_indices.append(dates[i])
        
        if len(X) == 0:
            continue
        
        X = np.array(X)
        y = np.array(y)
        
        # Train-test split
        split_idx = int(len(X) * train_split)
        
        sequences[symbol] = {
            'X_train': X[:split_idx],
            'X_test': X[split_idx:],
            'y_train': y[:split_idx],
            'y_test': y[split_idx:],
            'dates_train': date_indices[:split_idx],
            'dates_test': date_indices[split_idx:],
            'feature_cols': feature_cols,
            'n_features': len(feature_cols)
        }
        
        logger.info(f"  ✓ {symbol}: {len(X)} sequences | Shape: {X.shape}")
    
    # Save sequences
    sequences_dir = f'{output_dir}/sequences_with_sentiment'
    os.makedirs(sequences_dir, exist_ok=True)
    
    for symbol, seq_data in sequences.items():
        symbol_dir = f"{sequences_dir}/{symbol}"
        os.makedirs(symbol_dir, exist_ok=True)
        
        np.savez_compressed(
            f"{symbol_dir}/sequences.npz",
            X_train=seq_data['X_train'],
            X_test=seq_data['X_test'],
            y_train=seq_data['y_train'],
            y_test=seq_data['y_test'],
            dates_train=seq_data['dates_train'],
            dates_test=seq_data['dates_test'],
            feature_cols=seq_data['feature_cols']
        )
    
    # Save combined arrays
    combined_dir = f'{output_dir}/train_test_split_with_sentiment'
    os.makedirs(combined_dir, exist_ok=True)
    
    X_train_all = np.concatenate([seq['X_train'] for seq in sequences.values()], axis=0)
    X_test_all = np.concatenate([seq['X_test'] for seq in sequences.values()], axis=0)
    y_train_all = np.concatenate([seq['y_train'] for seq in sequences.values()], axis=0)
    y_test_all = np.concatenate([seq['y_test'] for seq in sequences.values()], axis=0)
    
    train_labels = []
    test_labels = []
    for symbol, seq in sequences.items():
        train_labels.extend([symbol] * len(seq['X_train']))
        test_labels.extend([symbol] * len(seq['X_test']))
    
    np.savez_compressed(
        f"{combined_dir}/train_data.npz",
        X=X_train_all,
        y=y_train_all,
        symbols=train_labels
    )
    
    np.savez_compressed(
        f"{combined_dir}/test_data.npz",
        X=X_test_all,
        y=y_test_all,
        symbols=test_labels
    )
    
    logger.info(f"\n✓ Sequences created successfully")
    logger.info(f"  - Combined train shape: {X_train_all.shape}")
    logger.info(f"  - Combined test shape: {X_test_all.shape}")
    logger.info(f"  - Input shape per sample: (60, 12) ✅")
    logger.info(f"  - Saved to: {sequences_dir}/\n")
    
    # Step 6: Update metadata
    logger.info("STEP 6: Updating Metadata")
    logger.info("-" * 80)
    
    metadata = {
        'preprocessing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'sentiment_integration_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'lookback_window': 60,
        'train_split_ratio': 0.8,
        'n_symbols': len(sequences),
        'total_train_sequences': len(X_train_all),
        'total_test_sequences': len(X_test_all),
        'feature_columns': feature_cols,
        'n_features': 12,
        'symbols': list(sequences.keys()),
        'sentiment_source': 'FinBERT (yiyanghkust/finbert-tone) - pre-computed from Prompt 3',
        'sentiment_score_range': '-1 to +1',
        'sentiment_coverage': {
            'records_with_sentiment': int(records_with_sentiment),
            'records_without_sentiment': int(records_without),
            'sentiment_dates': len(daily_sentiment)
        }
    }
    
    metadata_file = f'{output_dir}/preprocessing_metadata_with_sentiment.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✓ Metadata updated: {metadata_file}\n")
    
    # Step 7: Generate report
    logger.info("STEP 7: Generating Integration Report")
    logger.info("-" * 80)
    
    # Calculate per-symbol statistics
    symbol_sentiment = hybrid_data.groupby('Symbol')['sentiment_score'].agg([
        'mean', 'std', 'min', 'max', 'count'
    ]).reset_index()
    symbol_sentiment.columns = ['Symbol', 'Mean_Sentiment', 'Std_Sentiment', 
                               'Min_Sentiment', 'Max_Sentiment', 'Count']
    
    # Calculate correlation with returns
    correlations = []
    for symbol in hybrid_data['Symbol'].unique():
        symbol_data = hybrid_data[hybrid_data['Symbol'] == symbol]
        
        if len(symbol_data) > 10:
            corr = symbol_data[['sentiment_score', 'Returns']].corr().iloc[0, 1]
            correlations.append({
                'Symbol': symbol,
                'Sentiment_Returns_Correlation': corr
            })
    
    corr_df = pd.DataFrame(correlations)
    
    # Save analysis files
    symbol_sentiment.to_csv(f'{output_dir}/symbol_sentiment_analysis.csv', index=False)
    corr_df.to_csv(f'{output_dir}/sentiment_returns_correlation.csv', index=False)
    
    # Generate markdown report
    report_file = "FINBERT_SENTIMENT_INTEGRATION_REPORT.md"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# FinBERT Sentiment Integration Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Model:** yiyanghkust/finbert-tone (pre-computed)\n")
        f.write(f"**Method:** Manual integration using existing sentiment scores\n\n")
        f.write("---\n\n")
        
        f.write("## ✅ INTEGRATION SUMMARY\n\n")
        f.write("### Status: **COMPLETED SUCCESSFULLY** ✓\n\n")
        f.write("All sentiment scores have been integrated into the extended dataset.\n\n")
        
        f.write("---\n\n")
        f.write("## 📊 SENTIMENT DATA STATISTICS\n\n")
        f.write(f"- **Source texts processed:** {len(sentiment_df):,}\n")
        f.write(f"- **Unique dates with sentiment:** {len(daily_sentiment)}\n")
        f.write(f"- **Date range:** {sentiment_df['date'].min()} to {sentiment_df['date'].max()}\n")
        f.write(f"- **Mean sentiment score:** {sentiment_df['sentiment_score'].mean():.4f}\n")
        f.write(f"- **Sentiment std dev:** {sentiment_df['sentiment_score'].std():.4f}\n\n")
        
        f.write("---\n\n")
        f.write("## 🔗 DATASET INTEGRATION\n\n")
        f.write(f"- **Total records:** {len(hybrid_data):,}\n")
        f.write(f"- **Records with sentiment:** {records_with_sentiment:,} ({records_with_sentiment/len(hybrid_data)*100:.1f}%)\n")
        f.write(f"- **Records without (filled with 0):** {records_without:,} ({records_without/len(hybrid_data)*100:.1f}%)\n")
        f.write(f"- **Unique symbols:** {hybrid_data['Symbol'].nunique()}\n\n")
        
        f.write("---\n\n")
        f.write("## 📈 PER-SYMBOL SENTIMENT ANALYSIS\n\n")
        f.write("### Average Sentiment by Symbol\n\n")
        f.write("| Symbol | Mean Sentiment | Std Dev | Min | Max |\n")
        f.write("|--------|----------------|---------|-----|-----|\n")
        
        for _, row in symbol_sentiment.sort_values('Mean_Sentiment', ascending=False).head(15).iterrows():
            f.write(f"| {row['Symbol']} | {row['Mean_Sentiment']:.4f} | {row['Std_Sentiment']:.4f} | ")
            f.write(f"{row['Min_Sentiment']:.4f} | {row['Max_Sentiment']:.4f} |\n")
        
        f.write("\n---\n\n")
        f.write("## 🔗 SENTIMENT-RETURNS CORRELATION\n\n")
        f.write("### Correlation Between Sentiment Score and Returns\n\n")
        f.write("| Symbol | Correlation Coefficient |\n")
        f.write("|--------|------------------------|\n")
        
        for _, row in corr_df.sort_values('Sentiment_Returns_Correlation', ascending=False).head(15).iterrows():
            f.write(f"| {row['Symbol']} | {row['Sentiment_Returns_Correlation']:.4f} |\n")
        
        avg_corr = corr_df['Sentiment_Returns_Correlation'].mean()
        f.write(f"\n**Average Correlation:** {avg_corr:.4f}\n\n")
        
        f.write("**Interpretation:**\n")
        f.write("- Positive correlation: Higher sentiment → Higher returns\n")
        f.write("- Negative correlation: Higher sentiment → Lower returns\n")
        f.write("- Near zero: Weak or no linear relationship\n\n")
        
        f.write("---\n\n")
        f.write("## 🎯 MODEL INTEGRATION DETAILS\n\n")
        f.write("### Updated Model Input Shape\n\n")
        f.write("**Previous:** (60, 11)\n")
        f.write("- 60 timesteps (60-day lookback)\n")
        f.write("- 11 features (OHLCV + technical indicators)\n\n")
        f.write("**Current:** (60, 12) ✅\n")
        f.write("- 60 timesteps (60-day lookback)\n")
        f.write("- **12 features** (OHLCV + technical indicators + **sentiment_score**)\n\n")
        
        f.write("### Feature List\n\n")
        for i, feature in enumerate(feature_cols, 1):
            marker = " ✨ **NEW**" if feature == 'sentiment_score' else ""
            f.write(f"{i}. {feature}{marker}\n")
        
        f.write("\n---\n\n")
        f.write("## 📁 OUTPUT FILES\n\n")
        f.write(f"1. `{hybrid_csv}`\n")
        f.write(f"2. `{sequences_dir}/[SYMBOL]/sequences.npz`\n")
        f.write(f"3. `{combined_dir}/train_data.npz` - Shape: {X_train_all.shape}\n")
        f.write(f"4. `{combined_dir}/test_data.npz` - Shape: {X_test_all.shape}\n")
        f.write(f"5. `{metadata_file}`\n")
        f.write(f"6. `{output_dir}/symbol_sentiment_analysis.csv`\n")
        f.write(f"7. `{output_dir}/sentiment_returns_correlation.csv`\n\n")
        
        f.write("---\n\n")
        f.write("## 🚀 NEXT STEPS\n\n")
        f.write("1. **Retrain LSTM Model** with new (60, 12) input shape\n")
        f.write("2. **Compare Performance** with baseline model\n")
        f.write("3. **Integrate GARCH Volatility** for full hybrid model\n")
        f.write("4. **Evaluate Improvements** in directional accuracy and MAPE\n\n")
        
        f.write("---\n\n")
        f.write("**Report Generated:** {}\n".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        f.write("**Status:** ✅ READY FOR MODEL TRAINING\n")
    
    logger.info(f"✓ Report generated: {report_file}\n")
    
    # Final summary
    logger.info("=" * 80)
    logger.info("SENTIMENT INTEGRATION COMPLETED SUCCESSFULLY ✓")
    logger.info("=" * 80)
    logger.info(f"\n📊 SUMMARY:")
    logger.info(f"  - Hybrid data saved: {hybrid_csv}")
    logger.info(f"  - Updated sequences: (60, 12) with sentiment as 12th feature")
    logger.info(f"  - Symbols processed: {len(sequences)}")
    logger.info(f"  - Training sequences: {len(X_train_all):,}")
    logger.info(f"  - Testing sequences: {len(X_test_all):,}")
    logger.info(f"  - Report: {report_file}")
    logger.info(f"\n✅ READY FOR HYBRID MODEL TRAINING!\n")


if __name__ == "__main__":
    main()

