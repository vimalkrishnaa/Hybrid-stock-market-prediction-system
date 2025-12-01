# Why More Historical Data is Needed

## Current Situation

You select a date range in the frontend (e.g., 27-04-2025 to 07-11-2025 = ~6 months).

**What happens:**
1. Pipeline collects data for that exact range
2. Creates sequences (60-day lookback)
3. Splits into train/test (80/20)
4. Trains on ~4.8 months, tests on ~1.2 months

## The Problem

### With 6 Months of Data:
- **Total days**: ~180 days
- **After 60-day lookback**: ~120 usable sequences
- **Train set (80%)**: ~96 sequences
- **Test set (20%)**: ~24 sequences

**This is TOO LITTLE for deep learning!**
- LSTM models need hundreds/thousands of examples
- 96 training sequences = insufficient
- Model can't learn complex patterns
- High variance, poor generalization

### With 2-3 Years of Data:
- **Total days**: ~500-750 days
- **After 60-day lookback**: ~440-690 usable sequences
- **Train set (80%)**: ~350-550 sequences
- **Test set (20%)**: ~90-140 sequences

**This is MUCH BETTER!**
- 350-550 training sequences = sufficient
- Model can learn complex patterns
- Better generalization
- More stable predictions

## Solution Options

### Option 1: Use Historical Data for Training, Selected Range for Prediction
- **Training**: Use 2-3 years of historical data (more examples)
- **Prediction**: Still predict for your selected end_date
- **Benefit**: Better trained model, predicts on your date

### Option 2: Extend Training Window
- **Training**: Use data from (start_date - 2 years) to end_date
- **Prediction**: Predict for end_date + 1
- **Benefit**: More training data, still uses your range

### Option 3: Sliding Window Training
- **Training**: Use all available historical data
- **Prediction**: Use your selected range for final prediction
- **Benefit**: Maximum training data

## Recommended Approach

**Modify the pipeline to:**
1. **Training**: Use extended historical data (e.g., start_date - 2 years to end_date)
2. **Prediction**: Still predict for your selected end_date + 1
3. **User Experience**: User still selects their range, but model trains on more data

This way:
- ✅ User still controls the prediction date
- ✅ Model trains on sufficient data (2-3 years)
- ✅ Better accuracy and generalization
- ✅ More stable predictions

## Example

**User selects:**
- Start: 27-04-2025
- End: 07-11-2025

**What we should do:**
- **Collect**: 27-04-2023 to 07-11-2025 (2 years + user range)
- **Train on**: 27-04-2023 to 07-11-2025 (all data)
- **Predict for**: 08-11-2025 (next day after end_date)

**Result:**
- Training: ~500 sequences (much better!)
- Prediction: Still for your selected date
- Accuracy: Should improve significantly

## Implementation

We can modify `collect_stock_data()` to:
- Accept an optional `training_history_years` parameter
- Extend start_date backwards by that many years
- Still predict for the user's selected end_date

This gives you the best of both worlds:
- User controls prediction date
- Model trains on sufficient data

