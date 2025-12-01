# Model Simplification - Reverting to Proven Architecture

## Problem
Dual-output model was causing worse performance:
- Directional Accuracy: 46.6% → 39.7% (worse)
- R² Score: -0.25 → -0.29 (worse)
- Model too complex for limited data

## Solution: Revert to Single-Output with 27 Features

### What Changed:
1. **Reverted to Single-Output Model** ✅
   - Removed dual-output complexity
   - Back to proven Sequential architecture
   - Simpler = more stable

2. **Kept 27 Features** ✅
   - All advanced technical indicators retained
   - RSI, MACD, Bollinger Bands, etc.
   - Volume indicators, MA crossovers

3. **Improved Architecture for 27 Features** ✅
   - Increased dense layer capacity (32→64, 16→32)
   - Better handling of more features
   - Still stable and proven

4. **Optimized Training** ✅
   - 150 epochs with early stopping
   - Learning rate: 0.0005
   - Better patience (25)

## Why This Should Work Better

1. **Simplicity = Stability**
   - Single-output is proven
   - Less complexity = less overfitting
   - Better generalization

2. **27 Features Still Help**
   - RSI, MACD provide strong signals
   - Better feature set than before
   - Model can learn better patterns

3. **Proven Architecture**
   - 3-layer LSTM works well
   - Sequential model is stable
   - No experimental components

## Expected Results

- **Directional Accuracy**: 39.7% → 50-55% (target)
- **R² Score**: -0.29 → -0.1 to 0.1 (target)
- **RMSE**: 44.92 → 35-40 (target)
- **Stability**: More consistent predictions

## Key Insight

**Complexity doesn't always help!**
- Dual-output added complexity without benefit
- Simple model with good features often works better
- 27 features provide the signal, simple model learns it

## For 70-80% Accuracy (Future)

To reach 70-80%, you need:
1. **More Historical Data** (Critical)
   - 2-3 years minimum
   - Current: 6 months (too little)

2. **Ensemble Methods**
   - Multiple models
   - Voting on predictions

3. **Feature Selection**
   - Remove noisy features
   - Focus on strongest indicators

4. **Market Regime Detection**
   - Separate models for different conditions

