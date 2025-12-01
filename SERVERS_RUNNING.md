# 🚀 IndiTrendAI Dashboard - Servers Running

## ✅ Status: BOTH SERVERS ARE RUNNING!

---

## 🌐 Access URLs

### Frontend Dashboard
**URL:** http://localhost:3000

**Features:**
- 📊 Dashboard with predictions
- 📈 Model comparison
- 📄 Evaluation reports
- 🎨 Beautiful dark theme
- 📱 Fully responsive

### Backend API
**URL:** http://localhost:8000

**API Documentation:**
**URL:** http://localhost:8000/docs

**Available Endpoints:**
- `GET /` - API information
- `GET /health` - Health check
- `GET /symbols` - List of symbols
- `GET /metrics` - Model metrics
- `GET /predict?symbol=AAPL&days=60` - Predictions
- `GET /sentiment?symbol=AAPL&days=30` - Sentiment data
- `GET /volatility?symbol=AAPL&days=30` - Volatility data
- `GET /historical?symbol=AAPL&days=60` - Historical data
- `GET /model-comparison` - Model comparison

---

## 🎯 Quick Test

### Test Backend API
Open in browser or use curl:
```bash
# Health check
curl http://localhost:8000/health

# Get symbols
curl http://localhost:8000/symbols

# Get metrics
curl http://localhost:8000/metrics
```

### Test Frontend
1. Open http://localhost:3000 in your browser
2. You should see the IndiTrendAI dashboard
3. Navigate through:
   - Dashboard (/)
   - Model Comparison (/comparison)
   - Reports (/reports)

---

## 🛑 How to Stop Servers

### Stop Backend
1. Find the PowerShell window running `api_server.py`
2. Press `Ctrl + C`

### Stop Frontend
1. Find the PowerShell window running `npm run dev`
2. Press `Ctrl + C`

### Or Kill by Port
```powershell
# Find processes
netstat -ano | Select-String ":3000|:8000"

# Kill by PID (replace <PID> with actual process ID)
taskkill /F /PID <PID>
```

---

## 🔄 Restart Servers

### Restart Backend
```bash
python api_server.py
```

### Restart Frontend
```bash
cd frontend
npm run dev
```

---

## 📊 Server Details

### Backend (FastAPI)
- **Port:** 8000
- **Process:** Python
- **File:** api_server.py
- **Framework:** FastAPI + Uvicorn

### Frontend (React + Vite)
- **Port:** 3000
- **Process:** Node.js
- **Framework:** React 18 + Vite 5
- **Styling:** TailwindCSS

---

## 🎨 Dashboard Features

### 1. Dashboard Page (/)
- Overview cards (4 metrics)
- Prediction explorer with charts
- Model performance metrics
- Sentiment & volatility charts
- Feature importance visualization

### 2. Model Comparison (/comparison)
- Compare 4 different models
- Metric selector
- Detailed comparison table
- Best model highlight

### 3. Reports (/reports)
- 5 evaluation reports
- Download functionality
- View modal preview

---

## 🐛 Troubleshooting

### Port Already in Use
```powershell
# Check what's using the port
netstat -ano | findstr ":3000"
netstat -ano | findstr ":8000"

# Kill the process
taskkill /F /PID <PID>
```

### Frontend Not Loading
1. Check browser console for errors
2. Verify backend is running on port 8000
3. Clear browser cache
4. Try incognito/private mode

### API Not Responding
1. Check if api_server.py is running
2. Visit http://localhost:8000/docs
3. Check Python console for errors
4. Verify data files exist in `data/extended/processed/`

### CORS Errors
- Already configured in `api_server.py`
- If issues persist, check browser console
- Verify frontend is on port 3000

---

## 📱 Browser Compatibility

✅ **Tested and Working:**
- Chrome 90+
- Firefox 88+
- Edge 90+
- Safari 14+

---

## 🎯 Next Steps

1. ✅ **Explore Dashboard** - Navigate through all pages
2. ✅ **Test Predictions** - Select different symbols
3. ✅ **View Charts** - Interact with visualizations
4. ✅ **Check Reports** - Download evaluation reports
5. ✅ **Compare Models** - Analyze different architectures

---

## 📚 Documentation

- **Quick Start:** `DASHBOARD_QUICK_START.md`
- **Complete Guide:** `INDITRENDAI_DASHBOARD_README.md`
- **Features List:** `DASHBOARD_FEATURES_SUMMARY.md`
- **Frontend README:** `frontend/README.md`

---

## 🎉 Success!

Your IndiTrendAI Advanced Analytics Dashboard is now running!

**Frontend:** http://localhost:3000
**Backend:** http://localhost:8000
**API Docs:** http://localhost:8000/docs

**Made with ❤️ by Vimal Krishna | IndiTrendAI 2025**

