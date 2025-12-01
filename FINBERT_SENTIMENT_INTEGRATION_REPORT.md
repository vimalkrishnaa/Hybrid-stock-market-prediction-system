# FinBERT Sentiment Integration Report

**Generated:** 2025-10-24 18:28:47
**Model:** yiyanghkust/finbert-tone (pre-computed)
**Method:** Manual integration using existing sentiment scores

---

## ✅ INTEGRATION SUMMARY

### Status: **COMPLETED SUCCESSFULLY** ✓

All sentiment scores have been integrated into the extended dataset.

---

## 📊 SENTIMENT DATA STATISTICS

- **Source texts processed:** 99
- **Unique dates with sentiment:** 1
- **Date range:** 2025-10-02 to 2025-10-02
- **Mean sentiment score:** 0.1301
- **Sentiment std dev:** 0.8328

---

## 🔗 DATASET INTEGRATION

- **Total records:** 3,980
- **Records with sentiment:** 31 (0.8%)
- **Records without (filled with 0):** 3,949 (99.2%)
- **Unique symbols:** 31

---

## 📈 PER-SYMBOL SENTIMENT ANALYSIS

### Average Sentiment by Symbol

| Symbol | Mean Sentiment | Std Dev | Min | Max |
|--------|----------------|---------|-----|-----|
| ^N225 | 0.0011 | 0.0118 | 0.0000 | 0.1301 |
| HDFCBANK | 0.0010 | 0.0117 | 0.0000 | 0.1301 |
| ^HSI | 0.0010 | 0.0117 | 0.0000 | 0.1301 |
| TCS | 0.0010 | 0.0117 | 0.0000 | 0.1301 |
| SBIN | 0.0010 | 0.0117 | 0.0000 | 0.1301 |
| RELIANCE | 0.0010 | 0.0117 | 0.0000 | 0.1301 |
| ITC | 0.0010 | 0.0117 | 0.0000 | 0.1301 |
| INFY | 0.0010 | 0.0117 | 0.0000 | 0.1301 |
| HINDUNILVR | 0.0010 | 0.0117 | 0.0000 | 0.1301 |
| KOTAKBANK | 0.0010 | 0.0117 | 0.0000 | 0.1301 |
| AXISBANK | 0.0010 | 0.0117 | 0.0000 | 0.1301 |
| BHARTIARTL | 0.0010 | 0.0117 | 0.0000 | 0.1301 |
| GOOGL | 0.0010 | 0.0116 | 0.0000 | 0.1301 |
| NVDA | 0.0010 | 0.0116 | 0.0000 | 0.1301 |
| ^IXIC | 0.0010 | 0.0116 | 0.0000 | 0.1301 |

---

## 🔗 SENTIMENT-RETURNS CORRELATION

### Correlation Between Sentiment Score and Returns

| Symbol | Correlation Coefficient |
|--------|------------------------|
| ^N225 | 0.1337 |
| AXISBANK | 0.1329 |
| ADBE | 0.1275 |
| KOTAKBANK | 0.1218 |
| BHARTIARTL | 0.1207 |
| ^FTSE | 0.1141 |
| BTC-USD | 0.0691 |
| CRM | 0.0684 |
| META | 0.0512 |
| ETH-USD | 0.0507 |
| AMZN | 0.0330 |
| AAPL | 0.0269 |
| SBIN | 0.0256 |
| NVDA | 0.0215 |
| HINDUNILVR | 0.0193 |

**Average Correlation:** 0.0169

**Interpretation:**
- Positive correlation: Higher sentiment → Higher returns
- Negative correlation: Higher sentiment → Lower returns
- Near zero: Weak or no linear relationship

---

## 🎯 MODEL INTEGRATION DETAILS

### Updated Model Input Shape

**Previous:** (60, 11)
- 60 timesteps (60-day lookback)
- 11 features (OHLCV + technical indicators)

**Current:** (60, 12) ✅
- 60 timesteps (60-day lookback)
- **12 features** (OHLCV + technical indicators + **sentiment_score**)

### Feature List

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
12. sentiment_score ✨ **NEW**

---

## 📁 OUTPUT FILES

1. `data/extended/processed/hybrid_data_with_sentiment.csv`
2. `data/extended/processed/sequences_with_sentiment/[SYMBOL]/sequences.npz`
3. `data/extended/processed/train_test_split_with_sentiment/train_data.npz` - Shape: (1689, 60, 12)
4. `data/extended/processed/train_test_split_with_sentiment/test_data.npz` - Shape: (431, 60, 12)
5. `data/extended/processed/preprocessing_metadata_with_sentiment.json`
6. `data/extended/processed/symbol_sentiment_analysis.csv`
7. `data/extended/processed/sentiment_returns_correlation.csv`

---

## 🚀 NEXT STEPS

1. **Retrain LSTM Model** with new (60, 12) input shape
2. **Compare Performance** with baseline model
3. **Integrate GARCH Volatility** for full hybrid model
4. **Evaluate Improvements** in directional accuracy and MAPE

---

**Report Generated:** 2025-10-24 18:28:47
**Status:** ✅ READY FOR MODEL TRAINING
