## Hybrid Stock Market Prediction (IndiTrendAI)

IndiTrendAI is an end‑to‑end ML pipeline and dashboard for **next‑day stock price prediction** using a **Hybrid LSTM + Attention** model with **technical indicators, news sentiment, and GARCH(1,1) volatility**.  
It automates data collection, feature engineering, model training/evaluation, and provides an interactive React dashboard for visual analysis.

---

### 1. Goals and Motivation

- **Short‑term stock prediction**: Focus on predicting the **next trading day close price** for large‑cap stocks (e.g., `RELIANCE`).
- **Hybrid information sources**:
  - **Market micro‑structure** via OHLCV + 40+ technical indicators
  - **News / Twitter sentiment** via FinBERT
  - **Volatility clustering** via GARCH(1,1) conditional volatility
- **Model interpretability**: Metrics, error analysis, and (frontend) feature importance visualizations to understand model behavior.

---

### 2. High‑Level Architecture

- **Backend (Python + FastAPI)**
  - Main orchestrator: `pipeline_orchestrator.py`
  - API server: `api_server.py`
  - Auxiliary scripts: `train_hybrid_lstm.py`, `test_prediction.py`, `improve_directional_accuracy.py`, `verify_finbert_integration.py`, `integrate_garch_volatility.py` (legacy).

- **ML Core (Hybrid LSTM + Attention)**
  - LSTM layers capture temporal dependencies over a **60‑day lookback window**.
  - Attention layer learns **dynamic feature/time importance** for the prediction.
  - Trained and evaluated per symbol with rich metrics.

- **Frontend (React + Vite + Tailwind)**
  - `Dashboard.jsx` renders:
    - Price vs prediction charts
    - Evaluation metrics cards
    - Sentiment and volatility trends
    - Feature importance (currently using mock values in the UI)
  - `ModelComparison.jsx` compares hybrid vs baseline models visually.

- **Artifacts & Reports**
  - `sample_run_output/` contains:
    - Preprocessing, sentiment, volatility, and training reports (Markdown / txt)
    - Saved models (`.h5`), scalers (`.pkl`), logs, and plots.

---

### 3. Data & Features

#### 3.1. Raw Inputs

- **Price data (OHLCV)**: Open, High, Low, Close, Volume.
- **Symbol metadata**: e.g., `RELIANCE`, `TCS`, etc.
- **Sentiment sources**:
  - Financial news / tweets fetched and scored using **FinBERT**.
- **Volatility modeling input**:
  - Daily log returns for GARCH(1,1) fitting.

#### 3.2. Engineered Features (Examples)

The pipeline builds **42 features** (see `pipeline_orchestrator.py`) including:

- **Trend / Level**
  - **Returns**: percentage change of close price
  - **Moving Averages**: `MA_5`, `MA_10`, `MA_20`
  - **Price to MA ratios**: `Price_to_MA5`, `Price_to_MA20`
  - **Crossovers**: `MA5_MA10_cross`, `MA10_MA20_cross`

- **Momentum & Oscillators**
  - **Momentum** (various lookbacks)
  - **RSI (Relative Strength Index)**
  - **MACD**, **MACD_signal**, **MACD_hist**
  - **Stochastic Oscillator**: `%K`, `%D`
  - **Williams %R**
  - **CCI (Commodity Channel Index)**
  - **ROC (Rate of Change)**: `ROC_10`, `ROC_20`

- **Volatility & Range**
  - **Rolling volatility** of returns
  - **Bollinger Bands**: `BB_upper`, `BB_lower`, `BB_width`, `BB_position`
  - **High_Low_ratio`
  - **GARCH(1,1) conditional volatility**: `garch_volatility`

- **Volume & Trend Features**
  - `Volume_MA_5`, `Volume_MA_20`, `Volume_ratio`
  - `Volume_Momentum`
  - `Price_Volume_Trend`
  - `Trend_Strength`, `Volatility_Trend`

- **Sentiment**
  - Daily **sentiment_score** from FinBERT, normalized to a fixed range and merged on date.

All features are assembled into a modeling DataFrame and **normalized** using `MinMaxScaler` before sequence generation.

---

### 4. ML Pipeline (PipelineOrchestrator)

The main workflow is implemented in `pipeline_orchestrator.py` and exposed via `run_full_pipeline()`:

1. **Data Collection**
   - Fetch OHLCV data for a given `symbol`, `start_date`, and `end_date`.
   - Optionally load cached sentiment and volatility if present.

2. **Feature Engineering**
   - Compute all technical indicators and derived features.
   - Integrate **GARCH(1,1)** volatility per symbol.
   - Merge **FinBERT sentiment** at daily granularity.
   - Handle missing values, outliers, and alignment between price and sentiment/volatility data.

3. **Normalization**
   - Fit a `MinMaxScaler` on the training window for the 42 feature columns.
   - Persist scalers for later inverse‑transform in evaluation and plotting.

4. **Sequence Creation**
   - Build sequences with **lookback = 60**:
     - `X.shape = (num_samples, 60, num_features)`
     - `y.shape = (num_samples,)` for next‑day close.
   - Split into train/test sets (e.g., 80/20 chronological split).

5. **Model Training (Hybrid LSTM + Attention)**
   - Stack LSTM layers (e.g., 128 and 64 units) with `return_sequences=True`.
   - Apply an attention mechanism over the time dimension.
   - Dense layers map attended representation to a **single scalar prediction (next‑day close)**.
   - Train with loss such as **Huber / MSE** and optimizer like **Adam**, using early stopping.

6. **Prediction & Evaluation**
   - Generate predictions on the test set and the last window for **next‑day forecast**.
   - Inverse‑transform predictions and targets back to actual price scale.
   - Compute metrics (see below) and build recent history + prediction series for the dashboard.

7. **Output**
   - Returns a JSON‑serializable dictionary with:
     - `symbol`, `predicted_close`, `last_close`, `currency`
     - `rmse`, `mae`, `mape`, `r2`, `directional_accuracy`
     - `recent_data` (time series used for plotting)
     - `news_articles` (if available)

---

### 5. Evaluation Metrics

Metrics are computed in `pipeline_orchestrator.py` and `evaluate_hybrid_lstm.py`:

- **RMSE (Root Mean Squared Error)**
- **MAE (Mean Absolute Error)**
- **MAPE (Mean Absolute Percentage Error)**
- **R² Score**
- **Directional Accuracy** (percentage of days where the model correctly predicts up/down movement)
- **MSE (Mean Squared Error)**
- **Pearson Correlation Coefficient** between actual and predicted returns/prices
- **Residual statistics**: mean, standard deviation, distribution plots (in reports)

These are logged and saved to:

- `sample_run_output/output/reports/hybrid_model_evaluation_metrics.json`
- `sample_run_output/output/reports/hybrid_model_per_symbol_metrics.csv`
- Various **plots** under `sample_run_output/output/plots/`.

---

### 6. API Layer (FastAPI)

The **FastAPI** app in `api_server.py` exposes several endpoints (examples):

- **Health & Metadata**
  - `GET /health` – basic health check.
  - `GET /symbols` – available stock symbols.

- **Metrics & Historical Data**
  - `GET /metrics` – overall and per‑symbol performance metrics.
  - `GET /historical` – historical price, sentiment, and volatility slices.

- **Predictions**
  - `GET /predict` – returns recent predictions for a symbol and time window.
  - `POST /predict_next_day` – core endpoint used by the dashboard to:
    - Run (or reuse) the pipeline
    - Return next‑day prediction + metrics + recent data.
  - `POST /train_and_predict` – train the hybrid model and return prediction in a single call.

- **Analysis**
  - `GET /sentiment` – sentiment time series.
  - `GET /volatility` – GARCH volatility time series.
  - `GET /model-comparison` – comparison between hybrid and baseline LSTM models.

All responses are JSON‑serializable and consumed by the React frontend (`frontend/src/utils/api.js`).

---

### 7. Frontend Dashboard

Located in the `frontend/` directory (React + Vite + Tailwind):

- **Key Pages**
  - `Dashboard.jsx`
    - Symbol selector and date range inputs
    - **Price vs Prediction chart** (`StockChart`)
    - **Metrics cards** (`MetricsCard`) showing RMSE, MAPE, R², Directional Accuracy, etc.
    - **SentimentChart** – time series of sentiment_score vs price.
    - **VolatilityChart** – time series of GARCH volatility vs price.
    - **Feature importance** panel (currently using mock data, e.g., “Close Price = 95%”).
  - `ModelComparison.jsx`
    - Visual comparison of baseline vs hybrid models (e.g., RMSE, MAPE).
  - `Reports.jsx`
    - Links / summaries for generated Markdown and text reports in `sample_run_output/`.

- **Components**
  - `StockChart`, `SentimentChart`, `VolatilityChart`, `PerformanceBarChart`
  - `Navbar`, `Footer`, `TrainPredictPanel`, `MetricsCard`

The frontend communicates only with the FastAPI backend (CORS‑enabled in `api_server.py`).

---

### 8. Getting Started

#### 8.1. Clone the Repository

```bash
git clone https://github.com/vimalkrishnaa/Hybrid-Stock-Market-Prediction.git
cd Hybrid-Stock-Market-Prediction
```

#### 8.2. Backend Setup

```bash
# (Optional) Create and activate virtual environment on Windows
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start API server
python api_server.py
```

Backend runs at `http://localhost:8000`  
Interactive docs available at `http://localhost:8000/docs`.

#### 8.3. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Frontend runs by default at `http://localhost:3000`.

#### 8.4. One‑Click Startup (Windows)

From the project root:

```bash
start_servers.bat
```

This opens **two terminals**:

- FastAPI backend on port **8000**
- React frontend on port **3000**

---

### 9. Reproducing Experiments

- Use `train_hybrid_lstm.py` to run full training for selected symbols.
- Inspect generated artifacts in `sample_run_output/`:
  - **Reports**: training, evaluation, sentiment, and volatility reports.
  - **Plots**: training curves, prediction vs actual, error heatmaps, volatility/sentiment verification.
  - **Models**: saved `.h5` networks and `.pkl` scalers.
- For quick tests, use:
  - `test_prediction.py` – call API and print next‑day prediction + metrics.
  - `verify_finbert_integration.py` – sanity‑check sentiment extraction pipeline.

---

### 10. Repository Structure (High‑Level)

- `api_server.py` – FastAPI app exposing prediction, metrics, sentiment, volatility, and historical endpoints.  
- `pipeline_orchestrator.py` – main end‑to‑end pipeline (data → features → model → prediction).  
- `integrate_garch_volatility.py` – legacy standalone GARCH(1,1) volatility integration & analysis.  
- `train_hybrid_lstm.py`, `test_prediction.py`, `improve_directional_accuracy.py`, `manual_sentiment_integration.py`, `verify_finbert_integration.py` – helper / experiment scripts.  
- `frontend/` – React + Vite + Tailwind dashboard (charts, metrics, model comparison, reports UI).  
- `sample_run_output/` – example outputs (reports, plots, models, logs, verification files).  
- `requirements.txt` – Python dependencies for backend/ML.  
- `start_servers.bat`, `start_backend.bat`, `start_frontend.bat` – helper scripts to run servers (Windows).

---

### 11. Notes and Limitations

- The project is **research / educational** and not intended for live trading.
- Performance depends on:
  - Symbol liquidity and history length
  - Quality and coverage of news/sentiment data
  - Hyperparameter choices (lookback, LSTM size, learning rate, etc.)
- Frontend **feature importance values** are currently **mocked** and should be replaced with:
  - Attention‑based importance, SHAP values, or permutation importance from the trained model.

---

This README documents the **hybrid LSTM + Attention stock prediction system**, its data pipeline, ML architecture, APIs, and how to run and reproduce the main experiments locally.



