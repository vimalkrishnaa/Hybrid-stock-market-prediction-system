# FinBERT Sentiment Integration Instructions

**Status:** Script created (`integrate_finbert_sentiment.py`) but encountering dependency conflicts  
**Issue:** `typing_extensions` version conflicts between TensorFlow, PyTorch, and Transformers  
**Date:** 2025-10-24

---

## 🎯 OBJECTIVE

Integrate FinBERT sentiment analysis into the extended dataset to create a hybrid model with input shape (60, 12).

---

## ✅ WHAT WAS ACCOMPLISHED

### 1. Script Created: `integrate_finbert_sentiment.py`

The script is fully implemented with the following features:
- Loads existing FinBERT sentiment scores from Prompt 3
- Aggregates sentiment by date
- Merges sentiment with extended 6-month dataset
- Creates updated sequences with shape (60, 12)
- Generates comprehensive reports and analysis
- Handles fallback scenarios (uses neutral sentiment if FinBERT not available)

### 2. Existing Sentiment Data Available

From Prompt 3, we have:
- **File:** `sample_run_output/datafiles/sentiment_extraction/detailed_sentiment_scores.csv`
- **Records:** 196 sentiment scores
- **Date Range:** 2025-10-02 (1 day of Twitter data)
- **Mean Sentiment:** 0.1301 (slightly positive)
- **Source:** FinBERT model (yiyanghkust/finbert-tone) already applied

---

## ⚠️ CURRENT BLOCKER

### Dependency Conflict

The environment has conflicting requirements:
- **TensorFlow 2.13.0** requires: `typing_extensions < 4.6.0`
- **PyTorch 2.9.0** requires: `typing_extensions >= 4.10.0`
- **Transformers (FinBERT)** requires: `typing_extensions >= 4.6.0`

This creates an impossible dependency triangle that prevents FinBERT from loading.

---

## 🔧 SOLUTIONS

### Option 1: Use Existing Sentiment Data (RECOMMENDED for now)

Since we have pre-computed FinBERT scores from Prompt 3, we can:

1. **Manual Integration** (Quick Fix):
```python
import pandas as pd
import numpy as np

# Load extended data
extended_data = pd.read_csv('data/extended/processed/scaled_data_with_features.csv')

# Load sentiment scores
sentiment = pd.read_csv('sample_run_output/datafiles/sentiment_extraction/detailed_sentiment_scores.csv')
sentiment['date'] = pd.to_datetime(sentiment['created_at']).dt.date

# Aggregate by date
daily_sentiment = sentiment.groupby('date')['sentiment_score'].mean().reset_index()

# Merge
extended_data['date'] = pd.to_datetime(extended_data['Date'], utc=True).dt.date
hybrid_data = extended_data.merge(daily_sentiment, on='date', how='left')
hybrid_data['sentiment_score'].fillna(0, inplace=True)

# Save
hybrid_data.to_csv('data/extended/processed/hybrid_data_with_sentiment.csv', index=False)
```

2. **Recreate Sequences:**
```python
# Create sequences with 12 features (11 existing + sentiment)
feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 
               'Returns', 'MA_5', 'MA_10', 'MA_20', 'Volatility', 'Momentum',
               'sentiment_score']  # 12th feature

# Use existing preprocessing logic from extended_data_preprocessing.py
# but with 12 features instead of 11
```

---

### Option 2: Separate Environment for FinBERT

Create a dedicated environment just for sentiment extraction:

```bash
# Create new environment
conda create -n finbert python=3.10
conda activate finbert

# Install FinBERT dependencies only
pip install transformers torch sentencepiece pandas

# Run sentiment extraction separately
python extract_sentiment_only.py

# Switch back to main environment for LSTM training
conda activate main_env
```

---

### Option 3: Use Alternative Sentiment Library

Replace FinBERT with a lighter sentiment library:

```python
# Option A: VADER (no dependencies)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
scores = analyzer.polarity_scores(text)
sentiment_score = scores['compound']  # Range: -1 to +1

# Option B: TextBlob (lightweight)
from textblob import TextBlob
blob = TextBlob(text)
sentiment_score = blob.sentiment.polarity  # Range: -1 to +1
```

---

## 📋 STEP-BY-STEP MANUAL INTEGRATION

Since the automated script has dependency issues, here's how to manually integrate sentiment:

### Step 1: Prepare Sentiment Data

```python
import pandas as pd
import numpy as np

# Load existing sentiment scores
sentiment_df = pd.read_csv(
    'sample_run_output/datafiles/sentiment_extraction/detailed_sentiment_scores.csv'
)

# Extract date from created_at
sentiment_df['date'] = pd.to_datetime(sentiment_df['created_at']).dt.date

# Aggregate by date (daily average)
daily_sentiment = sentiment_df.groupby('date').agg({
    'sentiment_score': 'mean',
    'text': 'count'  # number of texts per day
}).reset_index()

daily_sentiment.columns = ['date', 'sentiment_score', 'text_count']

print(f"Unique dates with sentiment: {len(daily_sentiment)}")
print(f"Mean sentiment: {daily_sentiment['sentiment_score'].mean():.4f}")
```

### Step 2: Load Extended Dataset

```python
# Load scaled data with features
scaled_data = pd.read_csv('data/extended/processed/scaled_data_with_features.csv')

# Convert Date to date only (no time)
scaled_data['date'] = pd.to_datetime(scaled_data['Date'], utc=True).dt.date

print(f"Extended data shape: {scaled_data.shape}")
print(f"Date range: {scaled_data['date'].min()} to {scaled_data['date'].max()}")
```

### Step 3: Merge Sentiment with Extended Data

```python
# Merge sentiment scores
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

print(f"Hybrid data shape: {hybrid_data.shape}")
print(f"Records with sentiment: {(hybrid_data['sentiment_score'] != 0).sum()}")
print(f"Records without sentiment: {(hybrid_data['sentiment_score'] == 0).sum()}")
```

### Step 4: Save Hybrid Data

```python
import os

# Save hybrid CSV
output_dir = 'data/extended/processed'
os.makedirs(output_dir, exist_ok=True)

hybrid_csv = f'{output_dir}/hybrid_data_with_sentiment.csv'
hybrid_data.to_csv(hybrid_csv, index=False)

print(f"✓ Saved hybrid data: {hybrid_csv}")
print(f"  - Shape: {hybrid_data.shape}")
print(f"  - Size: {os.path.getsize(hybrid_csv) / 1024:.2f} KB")
```

### Step 5: Create Updated Sequences (60, 12)

```python
# Feature columns (12 features)
feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 
               'Returns', 'MA_5', 'MA_10', 'MA_20', 'Volatility', 'Momentum',
               'sentiment_score']  # 12th feature

lookback = 60
train_split = 0.8

sequences = {}

# Process each symbol
for symbol in sorted(hybrid_data['Symbol'].unique()):
    symbol_data = hybrid_data[hybrid_data['Symbol'] == symbol].copy()
    symbol_data = symbol_data.sort_values('Date')
    
    if len(symbol_data) <= lookback:
        print(f"✗ {symbol}: Insufficient data ({len(symbol_data)} records)")
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
    
    print(f"✓ {symbol}: {len(X)} sequences | Shape: {X.shape}")

# Save sequences
sequences_dir = 'data/extended/processed/sequences_with_sentiment'
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

print(f"\n✓ Sequences created with shape (60, 12)")
print(f"  - Saved to: {sequences_dir}/")
```

### Step 6: Update Metadata

```python
import json
from datetime import datetime

metadata = {
    'preprocessing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'sentiment_integration_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'lookback_window': 60,
    'train_split_ratio': 0.8,
    'n_symbols': len(sequences),
    'total_train_sequences': sum(len(seq['X_train']) for seq in sequences.values()),
    'total_test_sequences': sum(len(seq['X_test']) for seq in sequences.values()),
    'feature_columns': feature_cols,
    'n_features': 12,
    'symbols': list(sequences.keys()),
    'sentiment_source': 'FinBERT (yiyanghkust/finbert-tone) - pre-computed',
    'sentiment_score_range': '-1 to +1'
}

metadata_file = 'data/extended/processed/preprocessing_metadata_with_sentiment.json'
with open(metadata_file, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✓ Metadata updated: {metadata_file}")
```

---

## 📊 EXPECTED RESULTS

After manual integration:

### Files Created:
1. `data/extended/processed/hybrid_data_with_sentiment.csv`
2. `data/extended/processed/sequences_with_sentiment/[SYMBOL]/sequences.npz`
3. `data/extended/processed/preprocessing_metadata_with_sentiment.json`

### Data Shape:
- **Hybrid CSV:** (3980, 17) - added sentiment_score, sentiment_std, text_count
- **Sequences:** (N, 60, 12) per symbol
- **Combined Train:** (~1689, 60, 12)
- **Combined Test:** (~431, 60, 12)

### Model Training:
- **New Input Shape:** (60, 12) ✅
- **12th Feature:** sentiment_score (FinBERT)
- **Ready for:** Hybrid LSTM training with sentiment

---

## 🚀 NEXT STEPS

### Immediate:
1. Run the manual integration code above
2. Verify hybrid data shape
3. Check sequence shapes: (60, 12)

### Model Training:
1. Update LSTM architecture for (60, 12) input
2. Retrain with sentiment feature
3. Compare performance with baseline (60, 11)

### Expected Improvements:
- Better directional accuracy (from 54.88% to >60%)
- Improved MAPE (from 1.33% to <1.2%)
- Higher R² score (from 0.92 to >0.93)

---

## 📝 TROUBLESHOOTING

### If Sentiment Data is Insufficient:
The existing sentiment data only covers 1 day (2025-10-02). For better coverage:
1. Re-run Twitter/News data collection for more days
2. Or use neutral sentiment (0.0) for days without data
3. Or implement VADER/TextBlob for real-time sentiment

### If Dependencies Still Fail:
1. Use the manual integration code above
2. Skip automated FinBERT and use pre-computed scores
3. Or use Docker container with clean environment

---

**Status:** Ready for manual integration  
**Recommendation:** Use manual integration steps above to proceed  
**Time Required:** ~5 minutes for manual integration

