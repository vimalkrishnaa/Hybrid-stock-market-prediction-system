## Hybrid Stock Market Prediction (IndiTrendAI)

IndiTrendAI is an end‑to‑end ML pipeline and dashboard for **next‑day stock price prediction** using a **Hybrid LSTM + Attention** model with **technical indicators, sentiment, and GARCH(1,1) volatility**.  
The project includes automated data collection, feature engineering, model training/evaluation, and an interactive React dashboard.

### 1. Project Overview

- **Backend (FastAPI, Python)**  
  - Orchestrates the full pipeline via `pipeline_orchestrator.py`  
  - Exposes REST endpoints in `api_server.py` for prediction, metrics, sentiment, volatility, and historical data  
  - Integrates **FinBERT** sentiment and **GARCH(1,1)** conditional volatility into the feature set

- **Core ML Pipeline (`pipeline_orchestrator.py`)**
  - Data collection (OHLCV) for a symbol and date range
  - **Feature engineering (42 features)**: returns, moving averages (MA_5, MA_10, MA_20), volatility, momentum, RSI, MACD, Bollinger Bands, Stochastic Oscillator, Williams %R, CCI, ADX, ROC, price‑to‑MA ratios, volume features, trend/volatility indicators, sentiment score, and `garch_volatility`
  - Normalization via `MinMaxScaler`
  - Sequence creation (default **60‑day lookback**) for LSTM
  - Hybrid **LSTM + Attention** model training
  - Next‑day close price prediction + evaluation metrics:
    - RMSE, MAE, MAPE, R², Directional Accuracy, Pearson correlation, residual statistics

- **Frontend (React + Vite + Tailwind)**
  - `Dashboard.jsx` shows:
    - Price chart and prediction vs actual
    - Model metrics and performance comparison
    - Sentiment and volatility charts
    - (Mocked) feature importance visualization
  - `ModelComparison.jsx` compares hybrid vs baseline models

- **Analysis / Utility Scripts**
  - `integrate_garch_volatility.py`: legacy script demonstrating separate GARCH(1,1) volatility integration and reporting
  - `manual_sentiment_integration.py`, `improve_directional_accuracy.py`, `train_hybrid_lstm.py`, `test_prediction.py`, `verify_finbert_integration.py`
  - `sample_run_output/`: example reports, plots, models, and logs from a full run

### 2. Tech Stack

- **Backend / ML**
  - Python, FastAPI, NumPy, Pandas, scikit‑learn, TensorFlow / Keras
  - `arch` (GARCH(1,1) volatility modeling)
  - Hugging Face / FinBERT (financial sentiment)
- **Frontend**
  - React, Vite, Tailwind CSS, Victory charts
- **Other**
  - Git, JSON/CSV data files, batch scripts to start servers

### 3. Getting Started

#### 3.1. Clone the Repository

```bash
git clone https://github.com/vimalkrishnaa/Hybrid-Stock-Market-Prediction.git
cd Hybrid-Stock-Market-Prediction
```

#### 3.2. Backend Setup

```bash
# Create and activate virtual environment (example for Windows)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start API server
python api_server.py
```

Backend runs by default at `http://localhost:8000` (docs at `http://localhost:8000/docs`).

#### 3.3. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Frontend runs by default at `http://localhost:3000`.

Alternatively, from the project root you can use:

```bash
start_servers.bat
```

to start both backend and frontend in separate windows.

### 4. Usage Workflow

1. Call the FastAPI endpoint `/predict_next_day` (or use `test_prediction.py`) with:
   - `symbol` (e.g., `RELIANCE`)
   - `start_date`, `end_date`
2. `PipelineOrchestrator`:
   - Fetches and preprocesses data
   - Engineers all features and normalizes them
   - Trains/loads the Hybrid LSTM with Attention
   - Predicts **next‑day close price** and computes evaluation metrics
3. View predictions, metrics, sentiment, and volatility in the **React dashboard**.

### 5. Repository Structure (High‑Level)

- `api_server.py` – FastAPI app exposing prediction & metrics endpoints  
- `pipeline_orchestrator.py` – main end‑to‑end pipeline (data → features → model → prediction)  
- `integrate_garch_volatility.py` – standalone GARCH(1,1) volatility integration & analysis  
- `frontend/` – React + Vite + Tailwind dashboard  
- `sample_run_output/` – example outputs (reports, plots, models, logs)  
- `requirements.txt` – Python dependencies  
- `start_servers.bat`, `start_backend.bat`, `start_frontend.bat` – helper scripts to run servers

---

This README describes the **hybrid LSTM + Attention stock prediction system**, its pipeline, and how to run the backend and frontend locally.


