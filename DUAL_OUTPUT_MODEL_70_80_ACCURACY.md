# Dual-Output Model for 70-80% Directional Accuracy

## Implementation Summary

Implemented a **dual-output LSTM model** that predicts both **price** and **direction** separately, with heavy weighting on directional accuracy.

## Key Changes

### 1. **Dual-Output Architecture** ✅
- **Branch 1**: Price prediction (regression) - Linear output
- **Branch 2**: Direction prediction (classification) - Sigmoid output (0=down, 1=up)
- **Shared layers**: LSTM layers extract features for both tasks

### 2. **Weighted Loss Function** ✅
- **Price loss**: MSE (weight = 1.0)
- **Direction loss**: Binary Crossentropy (weight = **5.0**)
- **Total loss**: `1.0 * price_loss + 5.0 * direction_loss`
- **Why 5x?**: Heavily penalizes wrong direction predictions to achieve 70-80% accuracy

### 3. **Direction Labels** ✅
- Created direction labels: `1` if price goes up, `0` if down
- Based on actual next-day price movements
- Used for training and evaluation

### 4. **Training Optimization** ✅
- **Monitor**: `val_direction_prediction_binary_accuracy` (not just loss)
- **Early Stopping**: Patience 25, mode='max' (maximize accuracy)
- **Epochs**: 200 (more time to learn direction patterns)
- **Learning Rate**: 0.0005 (stable convergence)

### 5. **Enhanced Metrics** ✅
- Uses model's explicit direction predictions (not derived from price)
- More accurate directional accuracy calculation
- Logs direction probability for transparency

## Model Architecture

```
Input (60, 27)
    ↓
LSTM(128) → Dropout(0.25)
    ↓
LSTM(64) → Dropout(0.25)
    ↓
LSTM(32) → Dropout(0.2)
    ↓
Shared Dense(64) → Dropout(0.2)
    ↓
Shared Dense(32)
    ↓
    ├─→ Price Branch: Dense(16) → Dense(1, linear) [Price]
    └─→ Direction Branch: Dense(16) → Dropout(0.1) → Dense(1, sigmoid) [Direction]
```

## Expected Results

### Target Metrics
- **Directional Accuracy**: 42.5% → **70-80%** (target)
- **R² Score**: -0.08 → 0.2-0.4 (target)
- **RMSE**: 41.16 → 25-35 (target)
- **Price Accuracy**: Maintained or improved

### Why This Should Work

1. **Explicit Direction Learning**
   - Model directly learns to classify direction
   - Not just a byproduct of price prediction
   - Binary classification is easier than regression

2. **Heavy Weighting**
   - 5x weight on direction loss
   - Model prioritizes getting direction right
   - Penalizes wrong directions heavily

3. **27 Features**
   - RSI, MACD, Bollinger Bands provide strong directional signals
   - MA crossovers are classic trend indicators
   - Volume confirms price movements

4. **Optimized Training**
   - Monitors direction accuracy directly
   - Early stopping on accuracy (not loss)
   - More epochs for convergence

## Additional Strategies for 70-80% Accuracy

### If Still Below 70%:

1. **Increase Direction Weight**
   - Try 7x or 10x instead of 5x
   - More aggressive direction focus

2. **Use More Historical Data**
   - Current: 6 months
   - Target: 2-3 years minimum
   - More patterns = better learning

3. **Ensemble Methods**
   - Train 3-5 models with different seeds
   - Vote on direction predictions
   - Average price predictions

4. **Feature Selection**
   - Remove noisy features
   - Focus on strongest directional indicators
   - RSI, MACD, MA crossovers are key

5. **Market Regime Detection**
   - Separate models for bull/bear/volatile markets
   - Better specialization

6. **Hyperparameter Tuning**
   - Grid search for optimal weights
   - Optimal learning rate
   - Best architecture size

## Testing

After restarting servers:
1. Run prediction
2. Check directional accuracy (should be 70-80%)
3. Monitor direction probability in logs
4. If below 70%, increase direction weight to 7x or 10x

## Notes

- **70-80% directional accuracy is ambitious** but achievable with:
  - Dual-output model ✅
  - Heavy direction weighting ✅
  - 27 features with strong signals ✅
  - More historical data (recommended)
  - Ensemble methods (if needed)

- **Realistic expectations**:
  - 60-65% is very good for daily predictions
  - 70%+ requires excellent features and data
  - 80%+ is exceptional and may indicate overfitting

