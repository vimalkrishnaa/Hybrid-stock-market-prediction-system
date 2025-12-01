# Directional Accuracy Improvement Guide

## Current Status
- **Baseline Directional Accuracy:** ~54.88% (only slightly better than random 50%)
- **Target:** >60% (ideally 65-70%+)

## Why Directional Accuracy is Low

1. **Regression-focused training**: Model optimized for price prediction (MSE), not direction
2. **Limited direction signals**: Current features may not capture direction patterns well
3. **Data quality**: Sentiment data only available for ~1% of dates
4. **Market noise**: Stock prices are inherently noisy and hard to predict

## Strategies to Improve Directional Accuracy

### ✅ Strategy 1: Dual-Output Model (IMPLEMENTED)

**What it does:**
- Model predicts both price (regression) and direction (classification)
- Direction loss weighted 3x more than price loss
- Training focuses on maximizing directional accuracy

**Implementation:** `improve_directional_accuracy.py`

**Expected improvement:** +5-10%

---

### 🔧 Strategy 2: Add Direction-Specific Technical Indicators

**Indicators to add:**

1. **RSI (Relative Strength Index)**
   - Measures momentum (0-100)
   - RSI > 70 = overbought (likely to go down)
   - RSI < 30 = oversold (likely to go up)

2. **MACD (Moving Average Convergence Divergence)**
   - Shows trend direction and momentum
   - MACD line crossing signal line = direction change

3. **Bollinger Bands**
   - Price near upper band = likely to go down
   - Price near lower band = likely to go up

4. **ADX (Average Directional Index)**
   - Measures trend strength
   - Higher ADX = stronger trend direction

5. **Price Change Ratios**
   - Percentage change from previous day
   - Momentum indicators

**Implementation:**
```python
def add_direction_features(df):
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    
    # Bollinger Bands
    sma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    df['BB_upper'] = sma20 + (std20 * 2)
    df['BB_lower'] = sma20 - (std20 * 2)
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / sma20
    df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    
    return df
```

**Expected improvement:** +3-7%

---

### 🔧 Strategy 3: Custom Loss Function for Direction

**Idea:** Penalize wrong direction predictions more heavily

```python
def directional_loss(y_true, y_pred):
    # Price prediction error
    price_error = tf.keras.losses.mse(y_true, y_pred)
    
    # Direction error penalty
    true_direction = tf.sign(y_true[1:] - y_true[:-1])
    pred_direction = tf.sign(y_pred[1:] - y_pred[:-1])
    direction_error = tf.cast(tf.not_equal(true_direction, pred_direction), tf.float32)
    
    # Weighted combination
    return price_error + 5.0 * tf.reduce_mean(direction_error)
```

**Expected improvement:** +2-5%

---

### 🔧 Strategy 4: Feature Engineering for Direction

**New features to create:**

1. **Price change momentum** (already have, but enhance)
2. **Volume-price divergence** (volume increasing but price decreasing = reversal signal)
3. **Support/Resistance levels** (price near recent highs/lows)
4. **Candle patterns** (doji, hammer, etc.)
5. **Cross-sectional momentum** (how stock moves relative to market)

**Implementation:**
```python
# Volume-price divergence
df['volume_price_divergence'] = (df['Volume'].diff() > 0) & (df['Close'].diff() < 0)

# Support/Resistance
df['price_near_high'] = (df['Close'] / df['High'].rolling(20).max()) > 0.95
df['price_near_low'] = (df['Close'] / df['Low'].rolling(20).min()) < 1.05

# Relative momentum
df['relative_momentum'] = df['Returns'] - df['^GSPC_Returns']  # Relative to market
```

**Expected improvement:** +2-5%

---

### 🔧 Strategy 5: Ensemble Methods

**Combine multiple models:**

1. **LSTM + GRU Ensemble**
   - Train both models
   - Average predictions
   - Often improves robustness

2. **Direction-Specific Models**
   - Separate models for up/down predictions
   - Train on filtered data (only up days or only down days)

3. **Voting Classifier**
   - Multiple models vote on direction
   - Majority wins

**Expected improvement:** +3-8%

---

### 🔧 Strategy 6: Better Sentiment Data

**Current issue:** Sentiment only available for 1% of dates

**Solutions:**
1. Collect more news/social media data
2. Use alternative sentiment sources (Reddit, Twitter, news APIs)
3. Interpolate sentiment scores for missing dates
4. Use daily aggregated sentiment (average of all news for the day)

**Expected improvement:** +2-4%

---

### 🔧 Strategy 7: Hyperparameter Optimization

**Parameters to tune:**

1. **Loss weights** (price vs direction)
2. **Learning rate** (try 1e-4, 5e-4, 1e-3)
3. **Batch size** (16, 32, 64)
4. **Layer sizes** (LSTM units: 64, 128, 256)
5. **Dropout rates** (0.1, 0.2, 0.3)
6. **Lookback window** (30, 60, 90 days)

**Tool:** Use Optuna or Hyperopt for automated tuning

**Expected improvement:** +2-5%

---

### 🔧 Strategy 8: Post-Processing Threshold

**Idea:** Add confidence threshold for direction predictions

```python
def predict_with_threshold(model, X, confidence_threshold=0.55):
    price_pred, direction_prob = model.predict(X)
    
    # Only predict direction if confidence > threshold
    direction_pred = (direction_prob > confidence_threshold).astype(int)
    
    # For low confidence, use recent trend
    low_confidence = (direction_prob < confidence_threshold) & (direction_prob > (1 - confidence_threshold))
    if low_confidence.any():
        # Use moving average trend
        direction_pred[low_confidence] = (recent_trend > 0).astype(int)
    
    return price_pred, direction_pred
```

**Expected improvement:** +1-3%

---

### 🔧 Strategy 9: Market Regime Detection

**Idea:** Different models for different market conditions

1. **Bull market** (trending up)
2. **Bear market** (trending down)
3. **Sideways/volatile market**

Train separate models for each regime, then:
- Detect current regime
- Use appropriate model for prediction

**Expected improvement:** +3-6%

---

### 🔧 Strategy 10: Longer Training Data

**Current:** 6 months of data

**Target:** 1-2 years of data

**Why it helps:**
- More diverse market conditions
- Better generalization
- More training samples

**Expected improvement:** +2-4%

---

## Quick Wins (Easy to Implement)

### Priority 1: Dual-Output Model ✅
- **File:** `improve_directional_accuracy.py`
- **Time:** 30-60 minutes
- **Expected:** +5-10%

### Priority 2: Add RSI and MACD
- **Time:** 15-30 minutes
- **Expected:** +3-5%

### Priority 3: Optimize Loss Weights
- **Time:** 10 minutes
- **Expected:** +2-3%

### Priority 4: Post-Processing Threshold
- **Time:** 15 minutes
- **Expected:** +1-2%

---

## Combined Expected Improvement

If all strategies are implemented:
- **Baseline:** 54.88%
- **Target:** 70-75%+
- **Combined improvement:** +15-20%

---

## Implementation Priority

1. ✅ **Run `improve_directional_accuracy.py`** (Dual-output model)
2. **Add RSI/MACD features** to preprocessing
3. **Tune hyperparameters** (loss weights, learning rate)
4. **Implement ensemble** (combine models)
5. **Collect more sentiment data**
6. **Add market regime detection**

---

## Testing & Validation

After implementing improvements:
1. Run `evaluate_hybrid_lstm.py` to get baseline metrics
2. Run `improve_directional_accuracy.py` to get improved metrics
3. Compare directional accuracy
4. Test on out-of-sample data
5. Validate on different time periods

---

## Notes

- Directional accuracy is more important for trading than absolute price accuracy
- A model with 60%+ directional accuracy can be profitable with proper risk management
- Consider transaction costs when evaluating trading strategies
- Always validate on unseen data before deploying

---

**Last Updated:** 2025-10-24
**Status:** Strategy 1 (Dual-Output Model) implemented


