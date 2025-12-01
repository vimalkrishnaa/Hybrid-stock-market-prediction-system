# What Happens When You Click "Train & Predict"?

## ⚠️ Important Clarification

**The button says "Train & Predict" but it does NOT actually:**
- ❌ Fetch new data from APIs (NewsAPI, Twitter, yfinance)
- ❌ Train a new model
- ❌ Collect fresh sentiment data
- ❌ Update stock prices in real-time

## ✅ What It Actually Does

### Step-by-Step Process:

1. **Loads Existing Preprocessed Data**
   - Reads from: `data/extended/processed/hybrid_data_with_sentiment_volatility.csv`
   - This file contains data collected on **October 24, 2025**
   - Data is already preprocessed with sentiment and volatility

2. **Filters Data by Your Selected Date Range**
   - Takes the symbol you selected (e.g., RELIANCE)
   - Filters data between your start_date and end_date
   - Uses only the data that already exists in the CSV file

3. **Loads Pre-Trained Model**
   - Loads: `data/extended/models/hybrid_lstm_best.h5`
   - This model was trained **once** on the full dataset
   - Model weights are **NOT retrained** - they're fixed

4. **Creates Prediction Sequence**
   - Takes the last 60 days from your filtered date range
   - Creates a sequence with all 13 features (including sentiment & volatility)
   - Uses this sequence as input to the pre-trained model

5. **Makes Prediction**
   - Model predicts the next-day closing price
   - Uses the pre-trained weights (no training happens)

6. **Calculates Metrics**
   - Evaluates the model on your selected date range
   - Calculates RMSE, MAPE, R², Directional Accuracy
   - These metrics show how well the model performs on your date window

7. **Returns Results**
   - Predicted price for next day
   - Evaluation metrics
   - Recent price data for chart

---

## 📊 Data Flow Diagram

```
User clicks "Train & Predict"
         ↓
Frontend sends: {symbol, start_date, end_date}
         ↓
Backend API receives request
         ↓
Loads CSV file (preprocessed data from Oct 24, 2025)
         ↓
Filters data by symbol and date range
         ↓
Loads pre-trained model (hybrid_lstm_best.h5)
         ↓
Creates 60-day sequence from filtered data
         ↓
Model.predict(sequence) → Prediction
         ↓
Inverse transform to get actual price
         ↓
Calculate metrics on filtered window
         ↓
Return prediction + metrics to frontend
```

---

## 🔍 Key Points

### Data Source
- **Stock Prices**: From `extended_stock_data_20251024.csv` (collected Oct 24, 2025)
- **Sentiment**: Pre-computed FinBERT scores (from earlier data collection)
- **Volatility**: Pre-computed GARCH(1,1) values (from earlier processing)
- **No new API calls** are made

### Model Training
- Model was trained **once** on the full 6-month dataset
- Training happened when you ran `train_hybrid_lstm.py`
- The button does **NOT retrain** the model
- It only uses the **pre-trained weights**

### Date Range Selection
- The date range you select is used to:
  1. Filter which historical data to use
  2. Calculate evaluation metrics on that window
  3. Create the 60-day sequence for prediction

---

## 💡 Why "Train & Predict" is Misleading

The button name suggests it trains a new model, but it doesn't. It should be called:
- "Run Prediction"
- "Predict Next Day"
- "Get Prediction"

The "training" part is misleading because:
- No model training happens
- No new data is collected
- No weights are updated

---

## 🚀 If You Want to Actually Train on New Data

To truly train on new data, you would need to:

1. **Collect New Data**
   ```python
   python extended_data_collection.py  # Fetch new stock prices
   # Collect new sentiment from NewsAPI/Twitter
   # Calculate new GARCH volatility
   ```

2. **Preprocess New Data**
   ```python
   python extended_data_preprocessing.py
   python manual_sentiment_integration.py
   python integrate_garch_volatility.py
   ```

3. **Retrain Model**
   ```python
   python train_hybrid_lstm.py  # Train on new data
   ```

4. **Then Use Prediction**
   - The "Train & Predict" button will use the newly trained model

---

## 📝 Summary

**Current Behavior:**
- ✅ Uses existing preprocessed data
- ✅ Uses pre-trained model
- ✅ Filters by your date range
- ✅ Makes prediction
- ❌ Does NOT fetch new data
- ❌ Does NOT train model
- ❌ Does NOT collect new sentiment

**The button should be renamed to "Predict" or "Run Prediction" to be more accurate.**

