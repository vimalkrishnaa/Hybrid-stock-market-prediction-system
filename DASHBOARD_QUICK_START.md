# 🚀 IndiTrendAI Dashboard - Quick Start Guide

## ⚡ 5-Minute Setup

### Step 1: Install Frontend Dependencies (2 min)

```bash
cd frontend
npm install
```

### Step 2: Install Backend Dependencies (1 min)

```bash
cd ..
pip install fastapi uvicorn pandas numpy
```

### Step 3: Start Backend API (30 sec)

```bash
# Terminal 1
python api_server.py
```

✅ API running at: `http://localhost:8000`
📚 API Docs: `http://localhost:8000/docs`

### Step 4: Start Frontend Dashboard (30 sec)

```bash
# Terminal 2
cd frontend
npm run dev
```

✅ Dashboard running at: `http://localhost:3000`

---

## 🎯 What You'll See

### Dashboard Features

1. **Hero Section**
   - Total Stocks: 31
   - Avg Accuracy: 46.05%
   - Best Stock: AMZN

2. **Prediction Explorer**
   - Select any symbol (AAPL, GOOGL, etc.)
   - Choose time range (30/60/90 days)
   - View Actual vs Predicted chart

3. **Performance Metrics**
   - RMSE: 0.3084
   - MAE: 0.2759
   - R²: -0.1603
   - Dir. Acc: 46.05%

4. **Sentiment Analysis**
   - FinBERT sentiment chart
   - Color-coded: Green (positive), Yellow (neutral), Red (negative)

5. **Volatility Analysis**
   - GARCH(1,1) volatility
   - Overlaid with close price

6. **Feature Importance**
   - 8 key features with importance scores
   - Animated progress bars

---

## 🔧 Troubleshooting

### Backend Not Starting?
```bash
# Check if port 8000 is free
netstat -an | findstr 8000

# Kill process if needed (Windows)
taskkill /F /PID <process_id>

# Kill process if needed (Mac/Linux)
kill -9 <process_id>
```

### Frontend Not Starting?
```bash
# Clear cache
rm -rf node_modules package-lock.json
npm install

# Try different port
npm run dev -- --port 3001
```

### CORS Error?
Edit `api_server.py`:
```python
allow_origins=["http://localhost:3000"]  # Add your frontend URL
```

---

## 📱 Navigation

- **Dashboard** (`/`) - Main analytics page
- **Model Comparison** (`/comparison`) - Compare different models
- **Reports** (`/reports`) - View evaluation reports

---

## 🎨 Key Features

✅ **Responsive Design** - Works on mobile, tablet, desktop
✅ **Dark Mode** - Elegant dark theme
✅ **Smooth Animations** - Framer Motion powered
✅ **Interactive Charts** - Recharts visualizations
✅ **Real-Time Data** - FastAPI backend
✅ **Loading States** - Skeleton loaders
✅ **Error Handling** - Graceful error states

---

## 📊 API Endpoints

Test the API:

```bash
# Get symbols
curl http://localhost:8000/symbols

# Get metrics
curl http://localhost:8000/metrics

# Get prediction for AAPL
curl "http://localhost:8000/predict?symbol=AAPL&days=60"

# Get sentiment for AAPL
curl "http://localhost:8000/sentiment?symbol=AAPL&days=30"
```

---

## 🚀 Next Steps

1. **Customize**: Edit colors in `tailwind.config.js`
2. **Add Data**: Replace mock data with real predictions
3. **Deploy**: Build for production with `npm run build`
4. **Extend**: Add new features and components

---

## 📚 Documentation

- Full README: `frontend/README.md`
- Complete Guide: `INDITRENDAI_DASHBOARD_README.md`
- API Docs: `http://localhost:8000/docs`

---

## 💡 Tips

- Use **Ctrl+C** to stop servers
- Check **browser console** for errors
- Use **React DevTools** for debugging
- Test **API endpoints** in Postman or curl

---

**Made with ❤️ by Vimal Krishna | IndiTrendAI 2025**

🎉 **You're all set! Happy analyzing!** 🎉

