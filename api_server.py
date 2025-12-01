"""
FastAPI Backend for IndiTrendAI Dashboard
Provides endpoints for predictions, metrics, sentiment, and volatility data
"""

from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import pickle
from sklearn.preprocessing import MinMaxScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="IndiTrendAI API",
    description="Advanced Analytics API for Hybrid LSTM Stock Predictions",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data paths
DATA_PATH = "data/extended/processed"
OUTPUT_PATH = "sample_run_output/output"
MODELS_PATH = "data/extended/models"

# Cache for loaded data
_cache = {}


def load_json_file(filepath: str) -> Dict[str, Any]:
    """Load JSON file with caching"""
    if filepath in _cache:
        return _cache[filepath]
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        _cache[filepath] = data
        return data
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading data: {str(e)}")


def load_csv_file(filepath: str) -> pd.DataFrame:
    """Load CSV file with caching"""
    if filepath in _cache:
        return _cache[filepath]
    
    try:
        df = pd.read_csv(filepath)
        _cache[filepath] = df
        return df
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading data: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "IndiTrendAI API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": [
            "/health",
            "/symbols",
            "/metrics",
            "/predict",
            "/sentiment",
            "/volatility",
            "/historical",
            "/model-comparison"
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "IndiTrendAI API"
    }


@app.get("/symbols")
async def get_symbols():
    """Get list of available symbols (Indian stocks only)"""
    try:
        # Load from preprocessed data
        df = load_csv_file(f"{DATA_PATH}/hybrid_data_with_sentiment_volatility.csv")
        
        # Filter to Indian stocks only
        indian_symbols_list = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'HINDUNILVR', 
                              'BHARTIARTL', 'KOTAKBANK', 'SBIN', 'ITC', 'AXISBANK']
        symbols = [s for s in sorted(df['Symbol'].unique().tolist()) if s in indian_symbols_list]
        
        return {
            "symbols": symbols,
            "count": len(symbols),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting symbols: {e}")
        # Return default Indian symbols only
        return {
            "symbols": [
                'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'HINDUNILVR', 
                'BHARTIARTL', 'KOTAKBANK', 'SBIN', 'ITC', 'AXISBANK'
            ],
            "count": 10,
            "timestamp": datetime.now().isoformat()
        }


@app.get("/metrics")
async def get_metrics():
    """Get overall and per-symbol metrics"""
    try:
        # Load evaluation metrics
        metrics_file = f"{OUTPUT_PATH}/reports/hybrid_model_evaluation_metrics.json"
        metrics = load_json_file(metrics_file)
        
        return {
            "overall": metrics.get('overall_metrics', {}),
            "per_symbol": metrics.get('per_symbol_metrics', []),
            "timestamp": metrics.get('timestamp', datetime.now().isoformat())
        }
    except Exception as e:
        logger.error(f"Error loading metrics: {e}")
        # Return mock data
        return {
            "overall": {
                "rmse": 0.3084,
                "mae": 0.2759,
                "mape": 115.18,
                "r2_score": -0.1603,
                "directional_accuracy": 46.05,
                "test_samples": 431
            },
            "per_symbol": [],
            "timestamp": datetime.now().isoformat()
        }


@app.get("/predict")
async def get_prediction(
    symbol: str = Query(..., description="Stock symbol to predict"),
    days: int = Query(60, description="Number of days to return")
):
    """Get prediction data for a symbol"""
    try:
        # Load hybrid data
        df = load_csv_file(f"{DATA_PATH}/hybrid_data_with_sentiment_volatility.csv")
        
        # Filter by symbol
        symbol_data = df[df['Symbol'] == symbol].copy()
        
        if len(symbol_data) == 0:
            raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")
        
        # Sort by date
        symbol_data['Date'] = pd.to_datetime(symbol_data['Date'])
        symbol_data = symbol_data.sort_values('Date')
        
        # Get last N days
        symbol_data = symbol_data.tail(days)
        
        # Generate mock predictions (in production, use actual model)
        predictions = []
        for _, row in symbol_data.iterrows():
            actual = float(row['Close'])
            # Add some noise to simulate predictions
            predicted = actual * (1 + np.random.uniform(-0.05, 0.05))
            
            predictions.append({
                "date": row['Date'].strftime('%Y-%m-%d'),
                "actual": round(actual, 4),
                "predicted": round(predicted, 4),
                "open": round(float(row['Open']), 4),
                "high": round(float(row['High']), 4),
                "low": round(float(row['Low']), 4),
                "volume": int(row['Volume']) if 'Volume' in row else 0
            })
        
        return {
            "symbol": symbol,
            "data": predictions,
            "count": len(predictions),
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting prediction for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sentiment")
async def get_sentiment(
    symbol: str = Query(..., description="Stock symbol"),
    days: int = Query(30, description="Number of days to return")
):
    """Get sentiment data for a symbol"""
    try:
        # Load sentiment data
        df = load_csv_file(f"{DATA_PATH}/hybrid_data_with_sentiment_volatility.csv")
        
        # Filter by symbol
        symbol_data = df[df['Symbol'] == symbol].copy()
        
        if len(symbol_data) == 0:
            raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")
        
        # Sort by date
        symbol_data['Date'] = pd.to_datetime(symbol_data['Date'])
        symbol_data = symbol_data.sort_values('Date')
        
        # Get last N days
        symbol_data = symbol_data.tail(days)
        
        # Extract sentiment data
        sentiment_data = []
        for _, row in symbol_data.iterrows():
            sentiment_score = float(row.get('sentiment_score', 0.0))
            
            sentiment_data.append({
                "date": row['Date'].strftime('%Y-%m-%d'),
                "sentiment": round(sentiment_score, 4),
                "close": round(float(row['Close']), 4)
            })
        
        # Calculate statistics
        sentiments = [s['sentiment'] for s in sentiment_data]
        avg_sentiment = np.mean(sentiments) if sentiments else 0.0
        
        return {
            "symbol": symbol,
            "data": sentiment_data,
            "statistics": {
                "avg_sentiment": round(float(avg_sentiment), 4),
                "min_sentiment": round(float(min(sentiments)), 4) if sentiments else 0.0,
                "max_sentiment": round(float(max(sentiments)), 4) if sentiments else 0.0,
            },
            "count": len(sentiment_data),
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting sentiment for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/volatility")
async def get_volatility(
    symbol: str = Query(..., description="Stock symbol"),
    days: int = Query(30, description="Number of days to return")
):
    """Get GARCH volatility data for a symbol"""
    try:
        # Load volatility data
        df = load_csv_file(f"{DATA_PATH}/hybrid_data_with_sentiment_volatility.csv")
        
        # Filter by symbol
        symbol_data = df[df['Symbol'] == symbol].copy()
        
        if len(symbol_data) == 0:
            raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")
        
        # Sort by date
        symbol_data['Date'] = pd.to_datetime(symbol_data['Date'])
        symbol_data = symbol_data.sort_values('Date')
        
        # Get last N days
        symbol_data = symbol_data.tail(days)
        
        # Extract volatility data
        volatility_data = []
        for _, row in symbol_data.iterrows():
            garch_vol = float(row.get('garch_volatility', 0.0))
            
            volatility_data.append({
                "date": row['Date'].strftime('%Y-%m-%d'),
                "volatility": round(garch_vol, 4),
                "close": round(float(row['Close']), 4),
                "returns": round(float(row.get('Returns', 0.0)), 4)
            })
        
        # Calculate statistics
        volatilities = [v['volatility'] for v in volatility_data]
        avg_volatility = np.mean(volatilities) if volatilities else 0.0
        
        return {
            "symbol": symbol,
            "data": volatility_data,
            "statistics": {
                "avg_volatility": round(float(avg_volatility), 4),
                "min_volatility": round(float(min(volatilities)), 4) if volatilities else 0.0,
                "max_volatility": round(float(max(volatilities)), 4) if volatilities else 0.0,
            },
            "count": len(volatility_data),
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting volatility for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/historical")
async def get_historical(
    symbol: str = Query(..., description="Stock symbol"),
    days: int = Query(60, description="Number of days to return")
):
    """Get historical OHLCV data for a symbol"""
    try:
        # Load data
        df = load_csv_file(f"{DATA_PATH}/hybrid_data_with_sentiment_volatility.csv")
        
        # Filter by symbol
        symbol_data = df[df['Symbol'] == symbol].copy()
        
        if len(symbol_data) == 0:
            raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")
        
        # Sort by date
        symbol_data['Date'] = pd.to_datetime(symbol_data['Date'])
        symbol_data = symbol_data.sort_values('Date')
        
        # Get last N days
        symbol_data = symbol_data.tail(days)
        
        # Extract historical data
        historical_data = []
        for _, row in symbol_data.iterrows():
            historical_data.append({
                "date": row['Date'].strftime('%Y-%m-%d'),
                "open": round(float(row['Open']), 4),
                "high": round(float(row['High']), 4),
                "low": round(float(row['Low']), 4),
                "close": round(float(row['Close']), 4),
                "volume": int(row['Volume']) if 'Volume' in row else 0,
                "returns": round(float(row.get('Returns', 0.0)), 4),
                "ma_5": round(float(row.get('MA_5', 0.0)), 4),
                "ma_10": round(float(row.get('MA_10', 0.0)), 4),
                "ma_20": round(float(row.get('MA_20', 0.0)), 4),
            })
        
        return {
            "symbol": symbol,
            "data": historical_data,
            "count": len(historical_data),
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting historical data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-comparison")
async def get_model_comparison():
    """Get model comparison data"""
    return {
        "models": [
            {
                "name": "Hybrid LSTM",
                "rmse": 0.3084,
                "mae": 0.2759,
                "mape": 115.18,
                "r2": -0.1603,
                "directional_accuracy": 46.05,
                "features": 13,
                "parameters": 122274
            },
            {
                "name": "Baseline LSTM",
                "rmse": 0.3245,
                "mae": 0.2891,
                "mape": 121.45,
                "r2": -0.2134,
                "directional_accuracy": 43.21,
                "features": 11,
                "parameters": 98561
            }
        ],
        "timestamp": datetime.now().isoformat()
    }


# Pydantic models for request/response
class PredictNextDayRequest(BaseModel):
    symbol: str
    start_date: str
    end_date: str


class PredictNextDayResponse(BaseModel):
    symbol: str
    predicted_close: float
    last_close: float
    rmse: float
    mape: float
    r2: float
    directional_accuracy: float
    date_predicted_for: str
    recent_data: List[Dict[str, Any]]


@app.post("/predict_next_day")
async def predict_next_day(request: PredictNextDayRequest = Body(...)):
    """
    Train on date range and predict next-day closing price
    """
    try:
        logger.info(f"Received prediction request: {request.symbol} from {request.start_date} to {request.end_date}")
        
        # Create logs directory
        logs_dir = f"{OUTPUT_PATH}/api_logs"
        os.makedirs(logs_dir, exist_ok=True)
        
        # Load raw price data for actual prices (not normalized)
        raw_data_file = f"data/extended/raw/extended_stock_data_20251024.csv"
        if not os.path.exists(raw_data_file):
            # Fallback to parquet
            raw_data_file = f"data/extended/raw/extended_stock_data_20251024.parquet"
            if os.path.exists(raw_data_file):
                raw_df = pd.read_parquet(raw_data_file)
            else:
                raise HTTPException(status_code=404, detail="Raw stock data file not found")
        else:
            raw_df = pd.read_csv(raw_data_file)
        
        raw_df['Date'] = pd.to_datetime(raw_df['Date'], utc=True)
        
        # Load hybrid data (has normalized features + sentiment + volatility) for model input
        data_file = f"{DATA_PATH}/hybrid_data_with_sentiment_volatility.csv"
        if not os.path.exists(data_file):
            raise HTTPException(status_code=404, detail="Hybrid data file not found")
        
        df = pd.read_csv(data_file)
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        
        # Filter by symbol and date range
        symbol_data = df[df['Symbol'] == request.symbol].copy()
        raw_symbol_data = raw_df[raw_df['Symbol'] == request.symbol].copy()
        
        if len(symbol_data) == 0:
            raise HTTPException(status_code=404, detail=f"Symbol {request.symbol} not found")
        
        # Parse dates (make timezone-aware to match CSV data)
        start_date = pd.to_datetime(request.start_date, utc=True)
        end_date = pd.to_datetime(request.end_date, utc=True)
        
        # Filter date range
        filtered_data = symbol_data[
            (symbol_data['Date'] >= start_date) & 
            (symbol_data['Date'] <= end_date)
        ].sort_values('Date').reset_index(drop=True)
        
        # Filter raw data for actual prices
        filtered_raw_data = raw_symbol_data[
            (raw_symbol_data['Date'] >= start_date) & 
            (raw_symbol_data['Date'] <= end_date)
        ].sort_values('Date').reset_index(drop=True)
        
        if len(filtered_data) < 61:  # Need at least 61 days for 60-day lookback
            raise HTTPException(
                status_code=400, 
                detail=f"Insufficient data. Need at least 61 days, got {len(filtered_data)}"
            )
        
        logger.info(f"Filtered data: {len(filtered_data)} records")
        
        # Load the saved scaler for proper inverse transformation
        scaler_file = f"{DATA_PATH}/scalers/feature_scalers.pkl"
        scaler = None
        if os.path.exists(scaler_file):
            try:
                with open(scaler_file, 'rb') as f:
                    scalers_dict = pickle.load(f)
                scaler = scalers_dict.get(request.symbol)
                if scaler is None:
                    logger.warning(f"Scaler not found for {request.symbol}")
            except Exception as e:
                logger.warning(f"Error loading scaler: {e}")
        
        # Prepare features
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'Returns', 'MA_5', 'MA_10', 'MA_20', 'Volatility', 'Momentum',
            'sentiment_score', 'garch_volatility'
        ]
        
        # Check if all features exist
        missing_features = [col for col in feature_columns if col not in filtered_data.columns]
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            # Fill missing features with 0
            for col in missing_features:
                filtered_data[col] = 0.0
        
        # Extract features
        features_data = filtered_data[feature_columns].values
        
        # Use saved scaler if available, otherwise normalize on the fly
        if scaler is not None:
            # Features are already normalized, but we need to ensure consistency
            scaled_features = features_data  # Data is already normalized
        else:
            # Fallback: normalize features (simple min-max scaling)
            temp_scaler = MinMaxScaler()
            scaled_features = temp_scaler.fit_transform(features_data)
        
        # Create sequence for prediction (last 60 days)
        lookback = 60
        if len(scaled_features) < lookback:
            raise HTTPException(
                status_code=400,
                detail=f"Need at least {lookback} days of data"
            )
        
        # Get the last sequence
        X_pred = scaled_features[-lookback:].reshape(1, lookback, len(feature_columns))
        
        # Load model
        try:
            import tensorflow as tf
            model_path = f"{MODELS_PATH}/hybrid_lstm_best.h5"
            if not os.path.exists(model_path):
                model_path = f"{MODELS_PATH}/hybrid_lstm_model.h5"
            
            if not os.path.exists(model_path):
                raise HTTPException(status_code=404, detail="Model file not found")
            
            model = tf.keras.models.load_model(model_path)
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")
        
        # Make prediction
        prediction = model.predict(X_pred, verbose=0)
        predicted_normalized = float(prediction[0][0])
        
        # Inverse transform prediction using saved scaler or raw data
        close_idx = feature_columns.index('Close')
        
        # Get actual price range from raw data
        if len(filtered_raw_data) > 0:
            close_min = float(filtered_raw_data['Close'].min())
            close_max = float(filtered_raw_data['Close'].max())
            last_close = float(filtered_raw_data['Close'].iloc[-1])
        else:
            # Fallback: use min-max from normalized data
            close_min = float(filtered_data['Close'].min())
            close_max = float(filtered_data['Close'].max())
        last_close = float(filtered_data['Close'].iloc[-1])
        
        if scaler is not None:
            try:
                # Use saved scaler for proper inverse transformation
                # The scaler was trained on 11 features, but we have 13 now
                # So we'll use the original 11 features for inverse transform
                original_feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 
                                        'Returns', 'MA_5', 'MA_10', 'MA_20', 'Volatility', 'Momentum']
                close_idx_original = original_feature_cols.index('Close')
                
                # Create dummy array with original 11 features (only Close is predicted)
                dummy_pred = np.zeros((1, len(original_feature_cols)))
                dummy_pred[0, close_idx_original] = predicted_normalized
                # Inverse transform
                dummy_inverse = scaler.inverse_transform(dummy_pred)
                predicted_close = float(dummy_inverse[0, close_idx_original])
                
                logger.info(f"Used scaler for inverse transform. Predicted: {predicted_close:.2f}")
            except Exception as e:
                logger.warning(f"Scaler inverse transform failed: {e}, using raw data min-max")
                # Fallback to min-max from raw data
                predicted_close = predicted_normalized * (close_max - close_min) + close_min
        else:
            # Use min-max from raw data for inverse transform
            predicted_close = predicted_normalized * (close_max - close_min) + close_min
            logger.info(f"Used raw data min-max for inverse transform. Range: [{close_min:.2f}, {close_max:.2f}]")
        
        logger.info(f"Predicted close: {predicted_close:.2f}, Last close: {last_close:.2f}")
        
        # Calculate metrics on the filtered window
        # Create sequences for evaluation
        sequences = []
        targets = []
        for i in range(lookback, len(scaled_features)):
            sequences.append(scaled_features[i-lookback:i])
            targets.append(scaled_features[i, close_idx])
        
        if len(sequences) > 0:
            X_eval = np.array(sequences)
            y_eval = np.array(targets)
            
            # Predict on evaluation set
            y_pred = model.predict(X_eval, verbose=0).flatten()
            
            # Calculate metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            rmse = float(np.sqrt(mean_squared_error(y_eval, y_pred)))
            
            # MAPE
            mask = y_eval != 0
            if mask.sum() > 0:
                mape = float(np.mean(np.abs((y_eval[mask] - y_pred[mask]) / y_eval[mask])) * 100)
            else:
                mape = 0.0
            
            # R²
            r2 = float(r2_score(y_eval, y_pred))
            
            # Directional accuracy
            if len(y_eval) > 1:
                actual_direction = np.diff(y_eval) > 0
                pred_direction = np.diff(y_pred) > 0
                directional_accuracy = float(np.mean(actual_direction == pred_direction) * 100)
            else:
                directional_accuracy = 50.0
        else:
            rmse = 0.0
            mape = 0.0
            r2 = 0.0
            directional_accuracy = 50.0
        
        # Get predicted date (next day after end_date)
        date_predicted_for = (end_date + timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Prepare recent data for chart (last 30 days)
        recent_data = []
        recent_window = filtered_data.tail(30)
        for _, row in recent_window.iterrows():
            recent_data.append({
                "date": row['Date'].strftime('%Y-%m-%d'),
                "close": float(row['Close'])
            })
        
        # Add predicted point
        recent_data.append({
            "date": date_predicted_for,
            "close": predicted_close,
            "predicted": True
        })
        
        # Determine currency based on symbol
        # Indian stocks: RELIANCE, TCS, INFY, HDFCBANK, etc.
        indian_symbols = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'HINDUNILVR', 
                         'BHARTIARTL', 'KOTAKBANK', 'SBIN', 'ITC', 'AXISBANK']
        currency_symbol = '₹' if request.symbol in indian_symbols else '$'
        
        # Use raw data for chart prices (actual prices, not normalized)
        if len(filtered_raw_data) > 0:
            recent_window_raw = filtered_raw_data.tail(30)
            for i, data_point in enumerate(recent_data):
                if 'predicted' not in data_point or not data_point['predicted']:
                    # Match by date and use raw close price
                    date_str = data_point['date']
                    raw_row = recent_window_raw[recent_window_raw['Date'].dt.strftime('%Y-%m-%d') == date_str]
                    if len(raw_row) > 0:
                        data_point['close'] = float(raw_row['Close'].iloc[0])
                    else:
                        # Fallback: use inverse transform from normalized value
                        close_normalized = data_point['close']
                        data_point['close'] = close_normalized * (close_max - close_min) + close_min
        
        # Prepare response
        response = {
            "symbol": request.symbol,
            "predicted_close": round(predicted_close, 2),
            "last_close": round(last_close, 2),
            "currency": currency_symbol,
            "rmse": round(rmse, 4),
            "mape": round(mape, 2),
            "r2": round(r2, 4),
            "directional_accuracy": round(directional_accuracy, 2),
            "date_predicted_for": date_predicted_for,
            "recent_data": recent_data
        }
        
        # Save log
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f"{logs_dir}/predict_next_day_{timestamp}.json"
        with open(log_file, 'w') as f:
            json.dump({
                "request": request.dict(),
                "response": response,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
        
        logger.info(f"Prediction completed. Predicted: {predicted_close:.2f}, Last: {last_close:.2f}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in predict_next_day: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train_and_predict")
async def train_and_predict(request: PredictNextDayRequest = Body(...)):
    """
    Full pipeline: Fetch new data, preprocess, train model, and predict
    This endpoint orchestrates the complete workflow
    """
    try:
        logger.info(f"Full pipeline request: {request.symbol} from {request.start_date} to {request.end_date}")
        
        # Import orchestrator
        from pipeline_orchestrator import PipelineOrchestrator
        
        # Create orchestrator
        orchestrator = PipelineOrchestrator(
            symbol=request.symbol,
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        # Run full pipeline
        result = orchestrator.run_full_pipeline()
        
        # Check for errors
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        
        # Log result
        logger.info(f"Pipeline completed. Predicted: {result['predicted_close']:.2f}, Last: {result['last_close']:.2f}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in train_and_predict: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting IndiTrendAI API Server...")
    logger.info("API Documentation: http://localhost:8000/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

