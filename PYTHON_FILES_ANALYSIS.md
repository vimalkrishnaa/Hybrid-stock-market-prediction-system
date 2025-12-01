# Python Files Analysis - Required vs Not Required

**Generated:** 2025-10-24  
**Purpose:** Identify Python files that are not required for the main pipeline

---

## 📋 ALL PYTHON FILES IN PROJECT

### Total Files: 11 Python scripts

---

## ✅ REQUIRED FILES (Core Pipeline)

These files are essential for the main data collection, preprocessing, and training pipeline:

### 1. **`extended_data_collection.py`** ✅ REQUIRED
- **Size:** 15,925 bytes
- **Purpose:** Collects 6 months of historical OHLCV data (Prompt 6)
- **Used For:** Primary data collection with extended historical data
- **Outputs:** 
  - `data/extended/raw/extended_stock_data_*.csv`
  - `data/extended/raw/extended_stock_data_*.parquet`
- **Status:** Currently in use for extended dataset
- **Dependencies:** yfinance, nsepy

---

### 2. **`extended_data_preprocessing.py`** ✅ REQUIRED
- **Size:** 25,091 bytes
- **Purpose:** Preprocesses extended 6-month dataset (Prompt 7)
- **Used For:** 
  - Data cleaning
  - Feature engineering (technical indicators)
  - MinMaxScaler normalization
  - 60-day sequence creation
  - 80/20 train-test split
- **Outputs:**
  - `data/extended/processed/sequences/` (per-symbol sequences)
  - `data/extended/processed/train_test_split/` (combined train/test arrays)
  - `data/extended/processed/scalers/feature_scalers.pkl`
- **Status:** Core preprocessing for current pipeline
- **Dependencies:** pandas, numpy, sklearn

---

### 3. **`train_baseline_lstm.py`** ✅ REQUIRED
- **Size:** 30,093 bytes
- **Purpose:** Trains baseline LSTM model (Prompt 8)
- **Used For:**
  - Building LSTM architecture (64→32 units)
  - Training with batch_size=32, epochs=50
  - Model evaluation and metrics calculation
  - Generating reports and plots
- **Outputs:**
  - `data/extended/models/baseline_lstm_model.h5`
  - `data/extended/models/training_history.csv`
  - `data/extended/models/plots/`
- **Status:** Main training script for baseline LSTM
- **Dependencies:** tensorflow, keras, sklearn, matplotlib

---

## ⚠️ NOT REQUIRED (Legacy/Verification Files)

These files were created for earlier prompts or verification purposes and are **not required** for the current main pipeline:

### 4. **`stock_data_collector.py`** ⚠️ NOT REQUIRED
- **Size:** 29,025 bytes
- **Purpose:** Initial data collection (Prompt 1) with 1 month of data
- **Why Not Required:**
  - Collects only ~1 month of data (insufficient for 60-day lookback)
  - Replaced by `extended_data_collection.py` (6 months)
  - Includes Twitter/News API integration (optional features)
  - Output data has only 20-30 records per symbol
- **Original Outputs:**
  - `data/raw/nse/nse_data.csv`
  - `data/raw/yfinance/yfinance_data.csv`
  - `data/raw/sentiment/raw_sentiment.json`
- **Status:** Superseded by extended data collection
- **Keep?** Yes, for reference/comparison but not used in pipeline

---

### 5. **`data_preprocessing.py`** ⚠️ NOT REQUIRED
- **Size:** 16,772 bytes
- **Purpose:** Preprocesses initial 1-month dataset (Prompt 2)
- **Why Not Required:**
  - Works with limited 1-month data
  - Replaced by `extended_data_preprocessing.py`
  - Output insufficient for 60-day lookback window
- **Original Outputs:**
  - `data/processed/merged_price_data.csv`
  - `data/processed/cleaned_sentiment.json`
  - `data/processed/scaled_price_data.csv`
  - `data/processed/scaler.pkl`
- **Status:** Superseded by extended preprocessing
- **Keep?** Yes, for reference but not used in current pipeline

---

### 6. **`verify_preprocessing.py`** ⚠️ NOT REQUIRED
- **Size:** 15,574 bytes
- **Purpose:** Verification script for Prompt 2a
- **Why Not Required:**
  - Quality check/verification script only
  - Used once to verify preprocessing output
  - Not part of production pipeline
- **Function:** 
  - Loads and displays preprocessed data
  - Generates verification plots
  - Prints data quality checks
- **Outputs:**
  - `sample_run_output/datafiles/preprocessing/verification_plots/`
  - Verification reports
- **Status:** One-time verification, not needed for pipeline
- **Keep?** Yes, useful for debugging but not required

---

### 7. **`sentiment_extraction.py`** ⚠️ NOT REQUIRED (Currently)
- **Size:** 20,811 bytes
- **Purpose:** Extracts FinBERT sentiment scores (Prompt 3)
- **Why Not Required (Currently):**
  - Works with 1-month dataset
  - Not yet integrated with extended 6-month dataset
  - Outputs not used in current baseline LSTM
  - **Will be needed** for hybrid model (future integration)
- **Original Outputs:**
  - `sample_run_output/datafiles/sentiment_extraction/merged_with_sentiment.csv`
  - `sample_run_output/datafiles/sentiment_extraction/detailed_sentiment_scores.csv`
- **Status:** Awaiting integration with extended dataset
- **Keep?** **YES - Required for future hybrid model**
- **Note:** Will need to be adapted for extended dataset

---

### 8. **`verify_sentiment_extraction.py`** ⚠️ NOT REQUIRED
- **Size:** 21,725 bytes
- **Purpose:** Verification script for Prompt 3a
- **Why Not Required:**
  - Verification/quality check only
  - Used once to verify sentiment extraction
  - Not part of production pipeline
- **Function:**
  - Plots sentiment vs price correlations
  - Displays sample sentiment scores
  - Generates correlation heatmaps
- **Outputs:**
  - `sample_run_output/datafiles/sentiment_verification/`
  - Correlation plots and reports
- **Status:** One-time verification
- **Keep?** Yes, useful for debugging but not required

---

### 9. **`volatility_modeling.py`** ⚠️ NOT REQUIRED (Currently)
- **Size:** 29,146 bytes
- **Purpose:** GARCH volatility modeling (Prompt 4)
- **Why Not Required (Currently):**
  - Works with 1-month dataset
  - Not yet integrated with extended 6-month dataset
  - Outputs not used in current baseline LSTM
  - **Will be needed** for hybrid model (future integration)
- **Original Outputs:**
  - `sample_run_output/datafiles/volatility_modeling/final_with_volatility.csv`
  - `sample_run_output/datafiles/volatility_modeling/garch_model_statistics.csv`
- **Status:** Awaiting integration with extended dataset
- **Keep?** **YES - Required for future hybrid model**
- **Note:** Will need to be adapted for extended dataset

---

### 10. **`verify_volatility_integration.py`** ⚠️ NOT REQUIRED
- **Size:** 27,885 bytes
- **Purpose:** Verification script for Prompt 4a
- **Why Not Required:**
  - Verification/quality check only
  - Used once to verify volatility features
  - Not part of production pipeline
- **Function:**
  - Plots price vs conditional volatility
  - Shows correlation analysis
  - Identifies volatility spikes
- **Outputs:**
  - `sample_run_output/datafiles/volatility_verification/`
  - Volatility analysis plots
- **Status:** One-time verification
- **Keep?** Yes, useful for debugging but not required

---

### 11. **`baseline_lstm_model.py`** ⚠️ NOT REQUIRED
- **Size:** 27,486 bytes
- **Purpose:** Early LSTM training script (Prompt 5)
- **Why Not Required:**
  - Works with limited 1-month dataset (insufficient data)
  - Used 10-day lookback (not 60-day as required)
  - Trained on BTC-USD only due to data limitations
  - Replaced by `train_baseline_lstm.py` with proper 6-month data
- **Issues:**
  - Only 30 records for BTC-USD → insufficient
  - Had to reduce lookback to 10 days
  - Had to reduce batch size to 8
  - Not meeting Prompt 8 specifications
- **Status:** Superseded by `train_baseline_lstm.py`
- **Keep?** Yes, for reference but not used in pipeline

---

## 📊 SUMMARY

### Required for Current Pipeline (3 files):
1. ✅ `extended_data_collection.py` - Data collection
2. ✅ `extended_data_preprocessing.py` - Preprocessing
3. ✅ `train_baseline_lstm.py` - Model training

### Required for Future Hybrid Model (2 files):
4. 🔜 `sentiment_extraction.py` - FinBERT sentiment (needs adaptation)
5. 🔜 `volatility_modeling.py` - GARCH volatility (needs adaptation)

### Not Required - Legacy/1-Month Data (3 files):
6. ⚠️ `stock_data_collector.py` - Superseded by extended collection
7. ⚠️ `data_preprocessing.py` - Superseded by extended preprocessing
8. ⚠️ `baseline_lstm_model.py` - Superseded by train_baseline_lstm.py

### Not Required - Verification Scripts (3 files):
9. ⚠️ `verify_preprocessing.py` - One-time verification
10. ⚠️ `verify_sentiment_extraction.py` - One-time verification
11. ⚠️ `verify_volatility_integration.py` - One-time verification

---

## 🎯 RECOMMENDATION

### Files to Keep Active:
**Current Pipeline (3 files):**
- `extended_data_collection.py`
- `extended_data_preprocessing.py`
- `train_baseline_lstm.py`

**Future Integration (2 files):**
- `sentiment_extraction.py` (will need updating for 6-month data)
- `volatility_modeling.py` (will need updating for 6-month data)

### Files Not Required (Can be archived/ignored) - 6 files:
**Legacy (3 files):**
- `stock_data_collector.py`
- `data_preprocessing.py`
- `baseline_lstm_model.py`

**Verification (3 files):**
- `verify_preprocessing.py`
- `verify_sentiment_extraction.py`
- `verify_volatility_integration.py`

---

## 💡 NOTES

1. **Don't Delete:** Keep all files for reference and documentation
2. **Archive Folder:** Consider moving non-required files to `archive/` or `legacy/` folder
3. **Future Work:** `sentiment_extraction.py` and `volatility_modeling.py` will be needed for hybrid model but need to be updated to work with extended 6-month dataset
4. **Verification Scripts:** Useful for debugging but not part of production pipeline

---

## 📝 PIPELINE FLOW (Required Files Only)

```
1. Data Collection
   └── extended_data_collection.py
       └── Output: extended_stock_data_*.csv (6 months, 31 symbols)

2. Preprocessing
   └── extended_data_preprocessing.py
       └── Output: sequences (60-day lookback), scalers

3. Model Training
   └── train_baseline_lstm.py
       └── Output: baseline_lstm_model.h5 (92% R², 1.33% MAPE)

4. Future: Hybrid Integration
   ├── sentiment_extraction.py (needs update)
   └── volatility_modeling.py (needs update)
```

---

**Analysis Date:** 2025-10-24  
**Total Python Files:** 11  
**Required for Pipeline:** 3  
**Required for Future:** 2  
**Not Required:** 6 (3 legacy + 3 verification)

