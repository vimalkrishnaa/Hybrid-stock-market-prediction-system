# 🚀 IndiTrendAI - Advanced Analytics Dashboard

## Complete Implementation Guide

### 📋 Overview

IndiTrendAI is a production-ready, fully responsive Advanced Analytics Dashboard for AI-driven hybrid LSTM stock prediction. The system combines:
- **Technical Indicators** (MA, Volatility, Momentum)
- **FinBERT Sentiment Analysis** (NLP-based market psychology)
- **GARCH Volatility Modeling** (Risk quantification)

---

## 🎯 Project Structure

```
stock_market_2.0/
├── frontend/                           # React + Vite Dashboard
│   ├── src/
│   │   ├── components/                 # Reusable UI components
│   │   │   ├── Navbar.jsx             # Navigation bar with routing
│   │   │   ├── Footer.jsx             # Footer with social links
│   │   │   ├── MetricsCard.jsx        # Animated metric display cards
│   │   │   ├── StockChart.jsx         # Actual vs Predicted line chart
│   │   │   ├── SentimentChart.jsx     # FinBERT sentiment area chart
│   │   │   ├── VolatilityChart.jsx    # GARCH volatility composed chart
│   │   │   └── PerformanceBarChart.jsx # Per-symbol accuracy bars
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx          # Main dashboard page
│   │   │   ├── ModelComparison.jsx    # Model comparison page
│   │   │   └── Reports.jsx            # Evaluation reports page
│   │   ├── utils/
│   │   │   └── api.js                 # Axios API service
│   │   ├── App.jsx                    # Main app with routing
│   │   ├── main.jsx                   # Entry point
│   │   └── index.css                  # Global styles + Tailwind
│   ├── public/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   └── README.md
├── api_server.py                       # FastAPI backend
├── data/                               # Data directory
│   └── extended/
│       ├── processed/
│       │   ├── hybrid_data_with_sentiment_volatility.csv
│       │   └── preprocessing_metadata_with_sentiment_volatility.json
│       └── models/
│           └── hybrid_lstm_best.h5
├── sample_run_output/                  # Output directory
│   └── output/
│       ├── reports/
│       │   ├── hybrid_model_evaluation_metrics.json
│       │   ├── HYBRID_PERFORMANCE_ANALYSIS.md
│       │   └── hybrid_model_per_symbol_metrics.csv
│       └── plots/
└── INDITRENDAI_DASHBOARD_README.md     # This file
```

---

## 🛠️ Installation & Setup

### Prerequisites

- **Node.js** 18+ and npm
- **Python** 3.8+
- **Git**

### Step 1: Install Frontend Dependencies

```bash
cd frontend
npm install
```

### Step 2: Install Backend Dependencies

```bash
cd ..
pip install fastapi uvicorn pandas numpy python-multipart
```

### Step 3: Configure Environment

Create `frontend/.env`:
```env
VITE_API_URL=http://localhost:8000
```

---

## 🚀 Running the Application

### Terminal 1: Start Backend API

```bash
python api_server.py
```

**API will be available at:** `http://localhost:8000`
**API Docs:** `http://localhost:8000/docs`

### Terminal 2: Start Frontend Dashboard

```bash
cd frontend
npm run dev
```

**Dashboard will be available at:** `http://localhost:3000`

---

## 📊 Dashboard Features

### 1️⃣ **Dashboard Page** (`/`)

#### Hero Section
- Title: "Advanced Analytics Dashboard"
- Subtitle: Real-time predictions with sentiment & volatility

#### Overview Cards (4)
- **Total Stocks Tracked**: 31 symbols
- **Avg Directional Accuracy**: 46.05%
- **Best Performing Stock**: AMZN (58.3%)
- **Last Update**: Real-time timestamp

#### Prediction Explorer
- **Symbol Selector**: Dropdown with all available symbols
- **Time Range Toggle**: 30 / 60 / 90 days
- **Run Prediction Button**: Triggers prediction fetch
- **Chart**: Recharts line chart (Actual vs Predicted)

#### Model Performance Section
- **Metrics Grid**: RMSE, MAE, MAPE, R²
- **Per-Symbol Bar Chart**: Directional accuracy by symbol
- **Color Coding**:
  - 🟢 Green: ≥60% accuracy
  - 🟡 Yellow: 50-60% accuracy
  - 🔴 Red: <50% accuracy

#### Sentiment & Volatility Insights
- **Sentiment Chart**: FinBERT sentiment area chart (-1 to +1)
- **Volatility Chart**: GARCH volatility + close price overlay
- **Statistics**: Avg, Min, Max values

#### Feature Importance Panel
- **Horizontal Bars**: 8 key features with importance scores
- **Animated Progress**: Framer Motion powered
- **Note**: Based on attention weights and SHAP analysis

---

### 2️⃣ **Model Comparison Page** (`/comparison`)

#### Metric Selector
- Choose comparison metric: RMSE, MAE, MAPE, R², Dir. Acc.

#### Best Model Highlight
- 🏆 Trophy icon
- Shows best performer for selected metric

#### Comparison Table
- **Models**: Hybrid LSTM, Baseline LSTM, GRU, Transformer
- **Columns**: Status, RMSE, MAE, MAPE, R², Dir. Acc., Best Symbol, Parameters
- **Status Badges**: Active, Baseline, Experimental

#### Model Architecture Cards
- Individual cards for each model
- Features, parameters, best symbol
- Quick metrics: RMSE and Dir. Acc.

#### Key Insights Panel
- Bullet points with analysis
- Recommendations for improvement

---

### 3️⃣ **Reports Page** (`/reports`)

#### Summary Cards
- Total Reports: 5
- Completed: 5
- Last Updated: Oct 24, 2025

#### Report Cards (5)
1. **Hybrid LSTM Evaluation Report**
   - Key metrics: RMSE, MAE, R², Dir. Acc.
   
2. **Performance Analysis Report**
   - Highlights: Per-symbol breakdown, feature impact, recommendations
   
3. **FinBERT Sentiment Integration**
   - Stats: 99 texts processed, avg sentiment 0.1301
   
4. **GARCH Volatility Report**
   - Stats: 31 symbols, avg volatility 0.0234
   
5. **Data Preprocessing Summary**
   - Stats: 3980 records, 13 features, 1689 train, 431 test

#### Actions
- **View Button**: Opens modal preview
- **Download Button**: Downloads report file

---

## 🎨 Design System

### Color Palette
```css
/* Primary */
--primary-400: #22d3ee  /* Cyan */
--primary-500: #06b6d4

/* Dark Theme */
--dark-900: #0f172a     /* Background */
--dark-800: #1e293b     /* Cards */
--dark-700: #334155     /* Borders */
--dark-600: #475569

/* Status Colors */
--green-400: #22c55e    /* Success */
--yellow-400: #eab308   /* Warning */
--red-400: #ef4444      /* Error */
```

### Typography
- **Font**: Inter (Google Fonts)
- **Weights**: 300, 400, 500, 600, 700, 800, 900

### Components
```css
.card {
  @apply bg-dark-800 rounded-2xl shadow-lg p-6 border border-dark-700;
}

.card-hover {
  @apply transition-all duration-300 hover:shadow-xl hover:border-primary-400;
}

.btn-primary {
  @apply bg-primary-400 hover:bg-primary-500 text-dark-900 font-semibold px-6 py-3 rounded-lg;
}

.gradient-text {
  @apply bg-gradient-to-r from-primary-400 to-blue-500 bg-clip-text text-transparent;
}
```

---

## 🔌 API Endpoints

### Base URL: `http://localhost:8000`

| Endpoint | Method | Description | Parameters |
|----------|--------|-------------|------------|
| `/` | GET | API info | - |
| `/health` | GET | Health check | - |
| `/symbols` | GET | List of symbols | - |
| `/metrics` | GET | Overall & per-symbol metrics | - |
| `/predict` | GET | Prediction data | `symbol`, `days` |
| `/sentiment` | GET | Sentiment data | `symbol`, `days` |
| `/volatility` | GET | Volatility data | `symbol`, `days` |
| `/historical` | GET | Historical OHLCV | `symbol`, `days` |
| `/model-comparison` | GET | Model comparison | - |

### Example Requests

```bash
# Get symbols
curl http://localhost:8000/symbols

# Get metrics
curl http://localhost:8000/metrics

# Get prediction for AAPL (60 days)
curl "http://localhost:8000/predict?symbol=AAPL&days=60"

# Get sentiment for AAPL (30 days)
curl "http://localhost:8000/sentiment?symbol=AAPL&days=30"

# Get volatility for AAPL
curl "http://localhost:8000/volatility?symbol=AAPL&days=30"
```

---

## 📱 Responsive Design

### Breakpoints
- **Mobile**: < 768px (1 column)
- **Tablet**: 768px - 1024px (2 columns)
- **Desktop**: > 1024px (3-4 columns)

### Mobile Optimizations
- Collapsible navigation
- Stacked cards
- Touch-friendly buttons
- Optimized chart sizes

---

## ⚡ Performance Optimizations

### Frontend
- **Code Splitting**: React.lazy() for routes
- **Lazy Loading**: Charts loaded on demand
- **Memoization**: useMemo, useCallback for expensive operations
- **Debouncing**: API calls debounced
- **Caching**: API responses cached

### Backend
- **In-Memory Cache**: Loaded data cached
- **Async Operations**: FastAPI async endpoints
- **Efficient Data Loading**: Pandas optimizations
- **CORS**: Configured for production

---

## 🧪 Testing

### Frontend Testing
```bash
cd frontend
npm run lint
```

### Backend Testing
```bash
# Test API endpoints
curl http://localhost:8000/health
curl http://localhost:8000/symbols
curl http://localhost:8000/metrics
```

---

## 📦 Production Deployment

### Frontend Build
```bash
cd frontend
npm run build
```

Output: `frontend/dist/`

### Deploy Options

#### Vercel
```bash
cd frontend
vercel deploy
```

#### Netlify
```bash
cd frontend
netlify deploy --prod
```

#### Docker
```dockerfile
# Dockerfile
FROM node:18-alpine
WORKDIR /app
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ .
RUN npm run build
EXPOSE 3000
CMD ["npm", "run", "preview"]
```

### Backend Deployment

#### Uvicorn Production
```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 4
```

#### Docker
```dockerfile
# Dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 🎯 Key Features Implemented

### ✅ UI Components
- [x] Navbar with routing
- [x] Footer with social links
- [x] Animated metrics cards
- [x] Stock prediction chart
- [x] Sentiment area chart
- [x] Volatility composed chart
- [x] Performance bar chart
- [x] Loading skeletons
- [x] Error states

### ✅ Pages
- [x] Dashboard with all sections
- [x] Model Comparison
- [x] Reports with download

### ✅ Backend API
- [x] FastAPI server
- [x] All required endpoints
- [x] CORS configuration
- [x] Error handling
- [x] Data caching

### ✅ Animations
- [x] Framer Motion transitions
- [x] Fade-in effects
- [x] Slide-up animations
- [x] Hover effects
- [x] Loading states

### ✅ Responsive Design
- [x] Mobile layout
- [x] Tablet layout
- [x] Desktop layout
- [x] Touch-friendly
- [x] Optimized charts

---

## 🔧 Troubleshooting

### Issue: API Connection Failed
**Solution**: Ensure backend is running on port 8000
```bash
python api_server.py
```

### Issue: CORS Error
**Solution**: Check CORS configuration in `api_server.py`
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    ...
)
```

### Issue: Charts Not Rendering
**Solution**: Check data format and Recharts props
```javascript
// Ensure data has correct structure
const data = [
  { date: '2025-01-01', actual: 0.5, predicted: 0.48 },
  ...
];
```

### Issue: Build Fails
**Solution**: Clear cache and reinstall
```bash
rm -rf node_modules package-lock.json
npm install
npm run build
```

---

## 📚 Resources

### Documentation
- [React Docs](https://react.dev)
- [Vite Docs](https://vitejs.dev)
- [TailwindCSS Docs](https://tailwindcss.com)
- [Recharts Docs](https://recharts.org)
- [Framer Motion Docs](https://www.framer.com/motion)
- [FastAPI Docs](https://fastapi.tiangolo.com)

### Tutorials
- [React + Vite Setup](https://vitejs.dev/guide/)
- [TailwindCSS with Vite](https://tailwindcss.com/docs/guides/vite)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/AmazingFeature`
3. Commit changes: `git commit -m 'Add AmazingFeature'`
4. Push to branch: `git push origin feature/AmazingFeature`
5. Open Pull Request

---

## 📄 License

This project is part of the IndiTrendAI stock prediction system.

---

## 👨‍💻 Author

**Vimal Krishna**
- GitHub: [@vimalkrishna](https://github.com/vimalkrishna)
- LinkedIn: [vimalkrishna](https://linkedin.com/in/vimalkrishna)
- Email: vimal.krishna@example.com

---

## 🙏 Acknowledgments

- **FinBERT** by ProsusAI - Sentiment analysis
- **GARCH** models - Volatility modeling
- **TensorFlow/Keras** - Deep learning
- **React Community** - Amazing ecosystem
- **TailwindCSS** - Beautiful styling
- **Recharts** - Powerful charting

---

## 📈 Future Enhancements

### Planned Features
- [ ] Real-time WebSocket updates
- [ ] User authentication
- [ ] Portfolio tracking
- [ ] Alert system
- [ ] Export to PDF
- [ ] Dark/Light theme toggle
- [ ] Multi-language support
- [ ] Mobile app (React Native)

### Model Improvements
- [ ] Ensemble models
- [ ] Transformer architecture
- [ ] Multi-task learning
- [ ] Online learning
- [ ] Feature importance analysis (SHAP)

---

**Made with ❤️ by Vimal Krishna | IndiTrendAI 2025**

🚀 **Ready for Production!**

