# IndiTrendAI - Advanced Analytics Dashboard

![IndiTrendAI](https://img.shields.io/badge/IndiTrendAI-v1.0.0-blue)
![React](https://img.shields.io/badge/React-18.2.0-61DAFB?logo=react)
![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3.3.6-38B2AC?logo=tailwind-css)
![Vite](https://img.shields.io/badge/Vite-5.0.8-646CFF?logo=vite)

A production-ready, fully responsive Advanced Analytics Dashboard for AI-driven hybrid LSTM stock prediction system combining technical indicators, FinBERT sentiment analysis, and GARCH volatility modeling.

## 🎯 Features

### Core Functionality
- **Real-Time Predictions**: Interactive stock price predictions with actual vs predicted comparisons
- **Sentiment Analysis**: FinBERT-powered sentiment tracking with visual indicators
- **Volatility Modeling**: GARCH(1,1) conditional volatility visualization
- **Performance Metrics**: Comprehensive model evaluation with RMSE, MAE, MAPE, R², and Directional Accuracy
- **Model Comparison**: Side-by-side comparison of different model architectures
- **Evaluation Reports**: Access to detailed analysis reports and downloadable metrics

### UI/UX Features
- **Dark Mode Design**: Elegant dark theme optimized for extended viewing
- **Responsive Layout**: Fully responsive from mobile to 4K displays
- **Smooth Animations**: Framer Motion powered transitions and interactions
- **Interactive Charts**: Recharts-based visualizations with custom tooltips
- **Loading States**: Skeleton loaders and animated loading indicators
- **Error Handling**: Graceful error states with user-friendly messages

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ and npm/yarn
- Python 3.8+ (for backend API)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd stock_market_2.0/frontend
```

2. **Install dependencies**
```bash
npm install
```

3. **Start the development server**
```bash
npm run dev
```

The app will be available at `http://localhost:3000`

### Backend API Setup

1. **Install Python dependencies**
```bash
cd ..
pip install fastapi uvicorn pandas numpy
```

2. **Start the API server**
```bash
python api_server.py
```

The API will be available at `http://localhost:8000`

## 📁 Project Structure

```
frontend/
├── src/
│   ├── components/          # Reusable UI components
│   │   ├── Navbar.jsx
│   │   ├── Footer.jsx
│   │   ├── MetricsCard.jsx
│   │   ├── StockChart.jsx
│   │   ├── SentimentChart.jsx
│   │   ├── VolatilityChart.jsx
│   │   └── PerformanceBarChart.jsx
│   ├── pages/               # Page components
│   │   ├── Dashboard.jsx
│   │   ├── ModelComparison.jsx
│   │   └── Reports.jsx
│   ├── utils/               # Utility functions
│   │   └── api.js
│   ├── App.jsx              # Main app component
│   ├── main.jsx             # Entry point
│   └── index.css            # Global styles
├── public/                  # Static assets
├── index.html
├── package.json
├── vite.config.js
├── tailwind.config.js
└── postcss.config.js
```

## 🎨 Tech Stack

### Frontend
- **React 18.2** - UI library
- **Vite 5.0** - Build tool and dev server
- **TailwindCSS 3.3** - Utility-first CSS framework
- **Framer Motion 10.16** - Animation library
- **Recharts 2.10** - Chart library
- **React Router 6.20** - Client-side routing
- **Axios 1.6** - HTTP client
- **Lucide React** - Icon library

### Backend
- **FastAPI** - Modern Python web framework
- **Uvicorn** - ASGI server
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing

## 📊 API Endpoints

### Available Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/symbols` | GET | List of available stock symbols |
| `/metrics` | GET | Overall and per-symbol metrics |
| `/predict?symbol=AAPL` | GET | Prediction data for a symbol |
| `/sentiment?symbol=AAPL` | GET | Sentiment analysis data |
| `/volatility?symbol=AAPL` | GET | GARCH volatility data |
| `/historical?symbol=AAPL&days=60` | GET | Historical OHLCV data |
| `/model-comparison` | GET | Model comparison data |

### Example API Call

```javascript
import { apiService } from './utils/api';

// Get prediction data
const data = await apiService.getPrediction('AAPL');

// Get metrics
const metrics = await apiService.getMetrics();

// Get sentiment
const sentiment = await apiService.getSentiment('AAPL');
```

## 🎯 Key Components

### Dashboard Page
- Hero section with overview cards
- Prediction explorer with symbol selector
- Model performance metrics grid
- Per-symbol performance bar chart
- Sentiment and volatility dual charts
- Feature importance visualization

### Model Comparison Page
- Metric selector for different comparisons
- Best model highlight card
- Detailed comparison table
- Individual model architecture cards
- Key insights panel

### Reports Page
- Summary statistics cards
- Report cards with metadata
- Download and view functionality
- Modal preview for reports

## 🎨 Design System

### Color Palette
- **Primary**: Cyan (#22d3ee)
- **Dark Background**: #0f172a, #1e293b
- **Dark Cards**: #1e293b, #334155
- **Text**: White, Gray shades

### Typography
- **Font**: Inter (Google Fonts)
- **Sizes**: Responsive with Tailwind utilities

### Spacing
- Consistent 8px grid system
- Responsive padding and margins

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the frontend directory:

```env
VITE_API_URL=http://localhost:8000
```

### Vite Configuration

The `vite.config.js` includes proxy configuration for API calls:

```javascript
server: {
  port: 3000,
  proxy: {
    '/api': {
      target: 'http://localhost:8000',
      changeOrigin: true,
    }
  }
}
```

## 📱 Responsive Design

The dashboard is fully responsive with breakpoints:
- **Mobile**: < 768px (single column)
- **Tablet**: 768px - 1024px (2 columns)
- **Desktop**: > 1024px (3-4 columns)

## 🚀 Deployment

### Build for Production

```bash
npm run build
```

This creates an optimized build in the `dist/` directory.

### Preview Production Build

```bash
npm run preview
```

### Deploy Options
- **Vercel**: `vercel deploy`
- **Netlify**: `netlify deploy`
- **GitHub Pages**: Configure in repository settings

## 🧪 Development

### Available Scripts

```bash
npm run dev      # Start development server
npm run build    # Build for production
npm run preview  # Preview production build
npm run lint     # Run ESLint
```

### Code Style
- ESLint configuration included
- Prettier recommended for formatting
- Follow React best practices

## 📈 Performance Optimization

- **Code Splitting**: React.lazy() for route-based splitting
- **Image Optimization**: WebP format with fallbacks
- **Bundle Size**: Tree-shaking with Vite
- **Caching**: API response caching
- **Lazy Loading**: Charts and heavy components

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is part of the IndiTrendAI stock prediction system.

## 👨‍💻 Author

**Vimal Krishna**
- GitHub: [@vimalkrishna](https://github.com/vimalkrishna)
- LinkedIn: [vimalkrishna](https://linkedin.com/in/vimalkrishna)

## 🙏 Acknowledgments

- **FinBERT** - Sentiment analysis model
- **GARCH** - Volatility modeling
- **TensorFlow/Keras** - Deep learning framework
- **React Community** - Amazing ecosystem

---

Made with ❤️ by Vimal Krishna | IndiTrendAI 2025

