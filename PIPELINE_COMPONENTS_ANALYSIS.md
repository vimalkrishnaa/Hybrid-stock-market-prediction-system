# 🔍 Complete Pipeline Components Analysis

## 📋 Executive Summary

The **Pipeline Orchestrator** is a comprehensive end-to-end machine learning pipeline that performs:
1. **Data Collection** (Stock prices, News, Twitter)
2. **Data Preprocessing** (Feature engineering, Sentiment analysis, Volatility modeling)
3. **Model Training** (Hybrid LSTM deep learning)
4. **Prediction** (Next-day price forecasting with metrics)

---

## 🏗️ Architecture Overview

```
User Request (Frontend)
    ↓
API Endpoint: POST /train_and_predict
    ↓
PipelineOrchestrator.run_full_pipeline()
    ↓
┌─────────────────────────────────────────────────┐
│  STEP 1: Data Collection (10-30%)               │
│  STEP 2: Preprocessing (40-70%)                  │
│  STEP 3: Sequence Creation (75-80%)             │
│  STEP 4: Model Training (85-95%)                │
│  STEP 5: Prediction (98-100%)                   │
└─────────────────────────────────────────────────┘
    ↓
Return Results (Prediction + Metrics)
```

---

## 📦 Component 1: Data Collection Module

### **1.1 Stock Price Data Collection** (`collect_stock_data()`)

**Purpose:** Fetch historical OHLCV (Open, High, Low, Close, Volume) data

**Data Sources:**
- **Primary for Indian Stocks:** NSEpy (National Stock Exchange Python library)
- **Fallback/Global Stocks:** yfinance (Yahoo Finance API)

**Process:**
1. Identifies if symbol is Indian stock (RELIANCE, TCS, INFY, etc.)
2. Attempts NSEpy first (if available)
3. Falls back to yfinance with `.NS` suffix for Indian stocks
4. Collects data for user-specified date range
5. Saves raw data to temporary directory

**Output:**
- CSV file: `data/extended/temp_{symbol}_{timestamp}/raw_data_{symbol}.csv`
- DataFrame with columns: `Date, Open, High, Low, Close, Volume, Symbol, Source`

**Progress:** 10-30%

---

### **1.2 Sentiment Data Collection** (`collect_and_analyze_sentiment()`)

**Purpose:** Collect and analyze sentiment from news and social media

#### **1.2.1 Twitter Sentiment** (`load_existing_twitter_sentiment()`)

**Source:** `data/raw/sentiment/twitter_cache.json`

**Process:**
1. Loads cached Twitter data (permanent reuse to avoid API limits)
2. Filters out retweets (skips tweets starting with "RT @")
3. Analyzes each tweet with **FinBERT** model
4. Calculates sentiment score: `P(positive) - P(negative)`
5. Extracts dates from `created_at` timestamps
6. Returns sentiment scores by date

**FinBERT Model:**
- Model: `yiyanghkust/finbert-tone` (Hugging Face)
- Purpose: Financial sentiment analysis
- Output: Sentiment score range [-1.0, +1.0]
- Processing: Tokenization → FinBERT forward pass → Softmax → Score calculation

**Progress:** 52% ("Loading and analyzing Twitter sentiment from cache...")

---

#### **1.2.2 News Sentiment** (NewsAPI Integration)

**Source:** NewsAPI (with 6-hour caching)

**Process:**
1. **Cache Check:** `get_cached_news_sentiment()`
   - Checks `data/extended/cache/newsapi/{query}_{date}.json`
   - Validates cache age (< 6 hours)
   - Returns cached data if valid

2. **Fresh Fetch:** If cache expired/missing
   - Calls NewsAPI `get_everything()` endpoint
   - Query: `{symbol} stock India` for Indian stocks
   - Language: English
   - Sort: Relevancy
   - Page size: 10 articles per date
   - Top 5 articles selected per date

3. **Cache Storage:** `cache_news_sentiment()`
   - Saves fetched articles with timestamp
   - Enables 6-hour reuse window

4. **Sentiment Analysis:**
   - Analyzes each article with FinBERT
   - Extracts title + description text
   - Calculates sentiment score
   - Aggregates by date

**Progress:** 53% ("Fetching news articles (NewsAPI)...")

---

#### **1.2.3 Sentiment Aggregation**

**Process:**
1. Combines Twitter + News sentiment
2. Groups by date
3. Calculates daily average sentiment
4. Normalizes dates (removes timezone)
5. Merges with stock data

**Progress:** 54% ("Analyzing news sentiment with FinBERT...")

**Output:** DataFrame with `Date, sentiment_score`

---

## 📦 Component 2: Data Preprocessing Module

### **2.1 Technical Indicators** (`preprocess_data()`)

**Purpose:** Calculate technical analysis features

**Indicators Calculated:**

1. **Returns** (Price Change)
   - Formula: `Close.pct_change()`
   - Type: Percentage return

2. **Moving Averages**
   - **MA_5:** 5-day moving average
   - **MA_10:** 10-day moving average
   - **MA_20:** 20-day moving average
   - Formula: `Close.rolling(window=N).mean()`

3. **Volatility** (Rolling Standard Deviation)
   - Formula: `Returns.rolling(window=30).std()`
   - Purpose: Basic volatility measure

4. **Momentum**
   - Formula: `Close.pct_change(periods=5)`
   - Purpose: 5-day price momentum

**Progress:** 45% ("Calculating technical indicators...")

---

### **2.2 Sentiment Integration**

**Process:**
1. Merges sentiment scores with stock data
2. Fills missing sentiment with 0.0 (neutral)
3. Normalizes date columns (removes timezone)

**Progress:** 50% ("Collecting and analyzing sentiment...")

---

### **2.3 GARCH Volatility Modeling** (`calculate_garch_volatility()`)

**Purpose:** Calculate proper GARCH(1,1) conditional volatility

**Model Specification:**
```
Variance Equation:
σ²_t = ω + α·ε²_t-1 + β·σ²_t-1

Where:
- ω (omega): Baseline volatility
- α (alpha): Sensitivity to recent shocks
- β (beta): Volatility persistence
- σ²_t: Conditional variance at time t
```

**Process:**
1. **Per-Symbol Processing:**
   - Extracts returns for each symbol
   - Removes NaN/inf values
   - Validates data sufficiency (≥30 points)

2. **GARCH Fitting:**
   - Scales returns to percentage (×100)
   - Fits GARCH(1,1) using `arch` library
   - Extracts conditional volatility
   - Converts back to decimal (÷100)

3. **Fallback Handling:**
   - Insufficient data → Rolling std
   - Fitting failure → Rolling std
   - Library unavailable → Rolling std

4. **Date Alignment:**
   - Normalizes dates (removes timezone)
   - Aligns volatility with original dates
   - Forward/backward fills missing values

**Progress:** 55% ("Calculating GARCH(1,1) volatility...")

**Output:** DataFrame with `Date, garch_volatility`

---

### **2.4 Feature Normalization**

**Purpose:** Scale all features to [0, 1] range

**Method:** MinMaxScaler (per symbol)

**Features Normalized (13 total):**
1. Open
2. High
3. Low
4. Close
5. Volume
6. Returns
7. MA_5
8. MA_10
9. MA_20
10. Volatility
11. Momentum
12. sentiment_score
13. garch_volatility

**Process:**
1. Fits MinMaxScaler on all 13 features
2. Transforms features to [0, 1] range
3. Saves scaler to `scaler_{symbol}.pkl` for inverse transformation

**Progress:** 60% ("Normalizing features...")

**Output:** Normalized DataFrame + Saved Scaler

---

## 📦 Component 3: Sequence Creation Module

### **3.1 LSTM Sequence Generation** (`create_sequences()`)

**Purpose:** Create time-series sequences for LSTM training

**Configuration:**
- **Lookback Window:** 60 days
- **Target:** Next-day Close price
- **Features:** 13 features per timestep

**Process:**
1. Extracts 13 feature columns
2. Creates sliding windows:
   - Input: 60 days of features
   - Output: Close price on day 61
3. Generates sequences: `(N, 60, 13)`
4. Splits into train/test (80/20)

**Output Shapes:**
- `X_train`: (train_samples, 60, 13)
- `X_test`: (test_samples, 60, 13)
- `y_train`: (train_samples,)
- `y_test`: (test_samples,)

**Progress:** 75-80% ("Creating sequences for training...")

---

## 📦 Component 4: Model Training Module

### **4.1 Hybrid LSTM Architecture** (`train_model()`)

**Model Structure:**

```python
Input(shape=(60, 13))
    ↓
LSTM(128 units, return_sequences=True)
    ↓
Dropout(0.2)
    ↓
LSTM(64 units, return_sequences=True)
    ↓
Dropout(0.2)
    ↓
LSTM(32 units, return_sequences=False)
    ↓
Dropout(0.2)
    ↓
Dense(16, activation='relu')
    ↓
Dense(1)  # Output: Next-day Close price
```

**Total Parameters:** ~200K-300K (depending on input)

**Training Configuration:**
- **Optimizer:** Adam (learning_rate=0.001)
- **Loss Function:** MSE (Mean Squared Error)
- **Metrics:** MAE (Mean Absolute Error)
- **Batch Size:** 32
- **Epochs:** 50 (with early stopping)
- **Validation Split:** 20% (from test set)

**Callbacks:**
1. **EarlyStopping:**
   - Monitors: `val_loss`
   - Patience: 10 epochs
   - Restores best weights

2. **ReduceLROnPlateau:**
   - Monitors: `val_loss`
   - Factor: 0.5
   - Patience: 5 epochs
   - Min learning rate: 1e-7

**Progress:** 85-95% ("Training Hybrid LSTM model...")

**Output:** Trained model saved to `model_{symbol}.h5`

---

## 📦 Component 5: Prediction Module

### **5.1 Next-Day Prediction** (`predict_next_day()`)

**Process:**
1. **Sequence Creation:**
   - Takes last 60 days of features
   - Reshapes to (1, 60, 13)

2. **Model Prediction:**
   - Runs through trained LSTM
   - Output: Normalized Close price [0, 1]

3. **Inverse Transformation:**
   - Uses saved MinMaxScaler
   - Converts normalized prediction to actual price
   - Converts last close to actual price

4. **Metrics Calculation:**
   - **RMSE:** Root Mean Squared Error
   - **MAE:** Mean Absolute Error
   - **MAPE:** Mean Absolute Percentage Error
   - **R²:** Coefficient of Determination
   - **Directional Accuracy:** % of correct direction predictions

5. **Chart Data Preparation:**
   - Extracts last 30 days
   - Inverse transforms prices
   - Formats for frontend visualization

6. **Currency Detection:**
   - Indian stocks: ₹ (Rupee)
   - Global stocks: $ (Dollar)

**Progress:** 98-100% ("Making prediction...")

**Output:** Dictionary with prediction, metrics, and chart data

---

## 🔧 Supporting Components

### **A. Status Tracking** (`update_status()`)

**Purpose:** Real-time progress updates

**Status Structure:**
```python
{
    'step': 'collecting' | 'preprocessing' | 'training' | 'predicting' | 'complete',
    'progress': 0-100,
    'message': 'Human-readable status',
    'errors': []
}
```

**Usage:** Frontend displays progress bar and status messages

---

### **B. Error Handling**

**Strategy:**
- Try-except blocks at each step
- Graceful fallbacks (e.g., rolling std if GARCH fails)
- Error logging and status updates
- Returns error details in response

---

### **C. Caching System**

**1. NewsAPI Cache:**
- Location: `data/extended/cache/newsapi/`
- Format: JSON files with timestamp
- TTL: 6 hours
- Key: `{query}_{date}.json`

**2. Twitter Cache:**
- Location: `data/raw/sentiment/twitter_cache.json`
- Purpose: Permanent reuse (API limit reached)
- Format: JSON with tweets array

---

### **D. API Key Management**

**Sources (in order):**
1. Environment variables (`NEWS_API_KEY`, `TWITTER_BEARER_TOKEN`)
2. `api_keys.txt` file
3. Falls back gracefully if missing

---

## 📊 Complete Feature Set (13 Features)

### **Market Data (5 features):**
1. **Open** - Opening price
2. **High** - Highest price
3. **Low** - Lowest price
4. **Close** - Closing price (target variable)
5. **Volume** - Trading volume

### **Technical Indicators (6 features):**
6. **Returns** - Price change percentage
7. **MA_5** - 5-day moving average
8. **MA_10** - 10-day moving average
9. **MA_20** - 20-day moving average
10. **Volatility** - 30-day rolling standard deviation
11. **Momentum** - 5-day price momentum

### **Hybrid Features (2 features):**
12. **sentiment_score** - FinBERT sentiment analysis (-1 to +1)
    - Source: Twitter (cached) + NewsAPI (fresh)
    - Model: yiyanghkust/finbert-tone
    - Aggregation: Daily average

13. **garch_volatility** - GARCH(1,1) conditional volatility
    - Model: GARCH(p=1, q=1)
    - Library: arch
    - Fallback: Rolling std if unavailable

---

## 🔄 Complete Pipeline Flow

```
1. INITIALIZATION (0%)
   ├─ Load API keys
   ├─ Initialize FinBERT (lazy)
   └─ Create temp directories

2. DATA COLLECTION (10-30%)
   ├─ Collect stock prices (NSEpy/yfinance)
   └─ Save raw data

3. PREPROCESSING (40-70%)
   ├─ Calculate technical indicators (45%)
   ├─ Load Twitter sentiment from cache (52%)
   ├─ Fetch NewsAPI news (with 6h cache) (53%)
   ├─ Analyze sentiment with FinBERT (54%)
   ├─ Calculate GARCH(1,1) volatility (55%)
   ├─ Normalize features (60%)
   └─ Save preprocessed data

4. SEQUENCE CREATION (75-80%)
   ├─ Create 60-day lookback sequences
   └─ Split train/test (80/20)

5. MODEL TRAINING (85-95%)
   ├─ Build Hybrid LSTM architecture
   ├─ Train with callbacks
   └─ Save trained model

6. PREDICTION (98-100%)
   ├─ Make next-day prediction
   ├─ Calculate metrics
   ├─ Prepare chart data
   └─ Return results
```

---

## 📈 Output Metrics

### **Prediction Metrics:**
- **RMSE:** Root Mean Squared Error
- **MAE:** Mean Absolute Error
- **MAPE:** Mean Absolute Percentage Error (%)
- **R²:** Coefficient of Determination
- **Directional Accuracy:** % of correct up/down predictions

### **Response Structure:**
```json
{
  "symbol": "RELIANCE",
  "predicted_close": 2764.23,
  "last_close": 2750.10,
  "currency": "₹",
  "rmse": 45.32,
  "mae": 38.21,
  "mape": 1.39,
  "r2": 0.85,
  "directional_accuracy": 65.5,
  "date_predicted_for": "2025-11-08",
  "recent_data": [...],
  "status": {
    "step": "complete",
    "progress": 100,
    "message": "Pipeline completed successfully"
  }
}
```

---

## 🛠️ Dependencies & Libraries

### **Data Collection:**
- `yfinance` - Stock price data
- `nsepy` - Indian stock exchange data
- `newsapi-python` - News articles
- `tweepy` - Twitter API (not used, cache reused)

### **Sentiment Analysis:**
- `transformers` - Hugging Face models
- `torch` - PyTorch (for FinBERT)
- `finbert-tone` - Pre-trained financial sentiment model

### **Volatility Modeling:**
- `arch` - GARCH modeling library

### **Machine Learning:**
- `tensorflow` - Deep learning framework
- `keras` - High-level neural network API
- `scikit-learn` - MinMaxScaler, metrics

### **Data Processing:**
- `pandas` - Data manipulation
- `numpy` - Numerical computing

---

## ⚙️ Configuration Parameters

### **Model Architecture:**
- Lookback window: **60 days**
- Features per timestep: **13**
- LSTM layers: **3** (128 → 64 → 32 units)
- Dropout rate: **0.2**
- Dense layers: **2** (16 → 1)

### **Training:**
- Epochs: **50** (with early stopping)
- Batch size: **32**
- Learning rate: **0.001** (Adam optimizer)
- Train/test split: **80/20**

### **Caching:**
- NewsAPI cache TTL: **6 hours**
- Twitter: **Permanent cache** (no API calls)

### **Data Requirements:**
- Minimum data points: **61 days** (60 lookback + 1 target)
- GARCH minimum: **30 days** (falls back to rolling std if less)

---

## 🎯 Key Features

### **1. Multi-Source Data Collection**
- Stock prices: NSEpy (Indian) + yfinance (Global)
- News: NewsAPI with smart caching
- Twitter: Reused from permanent cache

### **2. Advanced Feature Engineering**
- Technical indicators (6 features)
- Sentiment analysis (FinBERT)
- Volatility modeling (GARCH(1,1))

### **3. Robust Error Handling**
- Graceful fallbacks at every step
- Comprehensive error logging
- Status tracking for debugging

### **4. Efficient Caching**
- NewsAPI: 6-hour cache window
- Twitter: Permanent cache reuse
- Reduces API calls and costs

### **5. Real-Time Progress**
- Status updates at each step
- Progress percentage (0-100%)
- Human-readable messages

---

## 📝 Summary

The pipeline is a **complete end-to-end ML system** that:

✅ **Collects** fresh stock data from multiple sources  
✅ **Integrates** sentiment from news and Twitter  
✅ **Models** volatility using GARCH(1,1)  
✅ **Trains** a deep learning model from scratch  
✅ **Predicts** next-day prices with comprehensive metrics  
✅ **Handles** errors gracefully with fallbacks  
✅ **Caches** data intelligently to reduce API costs  
✅ **Tracks** progress in real-time  

**Total Components:** 5 major modules + 4 supporting systems  
**Total Features:** 13 (5 market + 6 technical + 2 hybrid)  
**Processing Time:** 2-5 minutes (depending on data size)

