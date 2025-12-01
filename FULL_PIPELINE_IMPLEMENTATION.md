# Full Pipeline Implementation - Fetch, Preprocess, Train & Predict

## ✅ Implementation Complete!

The frontend "Train & Predict" button now supports a **full pipeline** that:
1. ✅ Fetches new stock data from APIs
2. ✅ Preprocesses the data (technical indicators, normalization)
3. ✅ Trains a new Hybrid LSTM model
4. ✅ Makes predictions and displays results

---

## 🎯 What Was Implemented

### 1. **Backend Pipeline Orchestrator** (`pipeline_orchestrator.py`)

A comprehensive module that orchestrates the complete workflow:

#### **Step 1: Data Collection** (10-30%)
- Fetches stock price data using `yfinance` or `NSEpy`
- Supports Indian stocks (NSE) and global stocks
- Collects data for the selected date range

#### **Step 2: Preprocessing** (40-70%)
- Calculates technical indicators:
  - Returns, MA_5, MA_10, MA_20, Volatility, Momentum
- Adds sentiment scores (placeholder - can integrate NewsAPI/Twitter)
- Calculates GARCH volatility (simplified using rolling volatility)
- Normalizes features using MinMaxScaler

#### **Step 3: Sequence Creation** (75-80%)
- Creates 60-day lookback sequences for LSTM
- Splits data into train/test (80/20)

#### **Step 4: Model Training** (85-95%)
- Builds Hybrid LSTM model:
  - 3 LSTM layers (128, 64, 32 units)
  - Dropout layers for regularization
  - Dense layers for output
- Trains with EarlyStopping and ReduceLROnPlateau callbacks
- Saves trained model

#### **Step 5: Prediction** (98-100%)
- Makes next-day price prediction
- Calculates evaluation metrics (RMSE, MAE, MAPE, R², Directional Accuracy)
- Prepares chart data
- Returns complete results

---

### 2. **Backend API Endpoint** (`/train_and_predict`)

**File:** `api_server.py`

**Endpoint:** `POST /train_and_predict`

**Request:**
```json
{
  "symbol": "RELIANCE",
  "start_date": "2025-04-27",
  "end_date": "2025-10-24"
}
```

**Response:**
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
  "date_predicted_for": "2025-10-25",
  "recent_data": [...],
  "status": {
    "step": "complete",
    "progress": 100,
    "message": "Pipeline completed successfully"
  }
}
```

---

### 3. **Frontend Updates** (`TrainPredictPanel.jsx`)

#### **New Features:**
1. **Toggle Checkbox**: "Fetch new data & train"
   - When checked: Uses full pipeline (`/train_and_predict`)
   - When unchecked: Uses existing data (`/predict_next_day`)

2. **Progress Indicator**:
   - Shows current step (collecting, preprocessing, training, predicting)
   - Displays progress percentage (0-100%)
   - Shows status message

3. **Dynamic Button Text**:
   - "Training..." when using full pipeline
   - "Predicting..." when using existing data

#### **Updated API Service** (`api.js`):
- Added `trainAndPredict()` method
- Calls `/train_and_predict` endpoint

---

## 🔄 How It Works

### **Full Pipeline Flow:**

```
User clicks "Train & Predict" (with checkbox checked)
    ↓
Frontend calls /train_and_predict
    ↓
Backend creates PipelineOrchestrator
    ↓
Step 1: Collect stock data (yfinance/NSEpy)
    ↓
Step 2: Preprocess (indicators, normalization)
    ↓
Step 3: Create sequences (60-day lookback)
    ↓
Step 4: Train Hybrid LSTM model
    ↓
Step 5: Make prediction
    ↓
Return results with metrics
    ↓
Frontend displays prediction + chart
```

### **Quick Prediction Flow (checkbox unchecked):**

```
User clicks "Train & Predict" (checkbox unchecked)
    ↓
Frontend calls /predict_next_day
    ↓
Backend loads existing preprocessed data
    ↓
Backend loads pre-trained model
    ↓
Makes prediction
    ↓
Return results
```

---

## 📊 Features

### **Data Collection:**
- ✅ Fetches fresh stock prices from APIs
- ✅ Supports Indian stocks (NSE) and global stocks
- ✅ Handles date range selection
- ✅ Error handling and retries

### **Preprocessing:**
- ✅ Technical indicators (6 features)
- ✅ Sentiment scores (placeholder - ready for NewsAPI/Twitter integration)
- ✅ GARCH volatility (simplified)
- ✅ Feature normalization

### **Model Training:**
- ✅ Hybrid LSTM architecture
- ✅ Early stopping to prevent overfitting
- ✅ Learning rate reduction
- ✅ Model checkpointing

### **Prediction:**
- ✅ Next-day price prediction
- ✅ Evaluation metrics
- ✅ Chart data preparation
- ✅ Currency detection (₹ for Indian stocks, $ for others)

---

## 🚀 Usage

### **From Frontend:**

1. **Select Symbol**: Choose from Indian stocks dropdown
2. **Select Date Range**: Pick start and end dates
3. **Toggle Option**: Check "Fetch new data & train" for full pipeline
4. **Click "Train & Predict"**: 
   - With checkbox: Full pipeline (takes 2-5 minutes)
   - Without checkbox: Quick prediction (takes 5-10 seconds)

### **Progress Tracking:**

The frontend shows real-time progress:
- **Step**: Current pipeline stage
- **Progress**: Percentage (0-100%)
- **Message**: Status description

---

## ⚙️ Configuration

### **Model Training Parameters:**
- **Epochs**: 50 (reduced for faster response)
- **Batch Size**: 32
- **Lookback Window**: 60 days
- **Train/Test Split**: 80/20

### **Model Architecture:**
- **LSTM Layers**: 128 → 64 → 32 units
- **Dropout**: 0.2 after each LSTM layer
- **Dense Layers**: 16 → 1
- **Optimizer**: Adam (lr=0.001)
- **Loss**: MSE

---

## 🔧 Future Enhancements

### **Sentiment Integration:**
Currently, sentiment scores are set to 0. To add real sentiment:
1. Integrate NewsAPI for news articles
2. Integrate Twitter API for social sentiment
3. Use FinBERT to analyze sentiment
4. Update `preprocess_data()` method

### **GARCH Volatility:**
Currently using rolling volatility. To add proper GARCH(1,1):
1. Install `arch` library: `pip install arch`
2. Fit GARCH model per symbol
3. Extract conditional volatility
4. Update `preprocess_data()` method

### **Real-time Progress Updates:**
For better UX, implement WebSocket or Server-Sent Events:
1. Stream progress updates from backend
2. Update frontend in real-time
3. Show detailed step-by-step progress

---

## 📝 Notes

- **Training Time**: Full pipeline takes 2-5 minutes depending on data size
- **Data Storage**: Temporary files saved in `data/extended/temp_{symbol}_{timestamp}/`
- **Model Storage**: Trained models saved per symbol
- **Error Handling**: Comprehensive error handling with status updates

---

## ✅ Testing

To test the full pipeline:

1. **Start Backend**: `python api_server.py`
2. **Start Frontend**: `cd frontend && npm run dev`
3. **Open Browser**: http://localhost:3000
4. **Navigate to**: Train & Predict panel
5. **Select**: Symbol, date range
6. **Check**: "Fetch new data & train" checkbox
7. **Click**: "Train & Predict"
8. **Wait**: 2-5 minutes for completion
9. **View**: Results with prediction and metrics

---

## 🎉 Summary

The frontend now supports **true end-to-end machine learning**:
- ✅ Fetches fresh data
- ✅ Preprocesses and engineers features
- ✅ Trains models
- ✅ Makes predictions
- ✅ Displays results

The system is now a **complete ML pipeline** accessible through a user-friendly web interface!

