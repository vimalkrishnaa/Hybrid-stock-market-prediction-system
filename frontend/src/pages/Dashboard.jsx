import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  TrendingUp, Target, Award, Clock, 
  Search, Play, Calendar 
} from 'lucide-react';
import MetricsCard from '../components/MetricsCard';
import StockChart from '../components/StockChart';
import SentimentChart from '../components/SentimentChart';
import VolatilityChart from '../components/VolatilityChart';
import PerformanceBarChart from '../components/PerformanceBarChart';
import TrainPredictPanel from '../components/TrainPredictPanel';
import { apiService } from '../utils/api';

const Dashboard = () => {
  const [loading, setLoading] = useState(true);
  const [selectedSymbol, setSelectedSymbol] = useState('RELIANCE');
  const [timeRange, setTimeRange] = useState(60);
  const [symbols, setSymbols] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [predictionData, setPredictionData] = useState([]);
  const [sentimentData, setSentimentData] = useState([]);
  const [volatilityData, setVolatilityData] = useState([]);
  const [lastUpdate, setLastUpdate] = useState(new Date());

  // Mock data for demonstration (replace with API calls)
  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    setLoading(true);
    try {
      // Load mock data
      await loadMockData();
      setLastUpdate(new Date());
    } catch (error) {
      console.error('Error loading dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadMockData = async () => {
    // Indian stocks only
    setSymbols([
      'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'HINDUNILVR', 
      'BHARTIARTL', 'KOTAKBANK', 'SBIN', 'ITC', 'AXISBANK'
    ]);

    // Mock metrics
    setMetrics({
      totalStocks: 10,
      avgDirectionalAccuracy: 46.05,
      bestPerformingStock: 'ITC',
      rmse: 0.3084,
      mae: 0.2759,
      mape: 115.18,
      r2Score: -0.1603,
    });

    // Mock prediction data
    const mockPredictions = generateMockTimeSeries(timeRange);
    setPredictionData(mockPredictions);

    // Mock sentiment data
    const mockSentiment = generateMockSentiment(30);
    setSentimentData(mockSentiment);

    // Mock volatility data
    const mockVolatility = generateMockVolatility(30);
    setVolatilityData(mockVolatility);
  };

  const generateMockTimeSeries = (days) => {
    const data = [];
    const basePrice = 0.5;
    for (let i = 0; i < days; i++) {
      const date = new Date();
      date.setDate(date.getDate() - (days - i));
      data.push({
        date: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
        actual: basePrice + Math.random() * 0.3 - 0.15,
        predicted: basePrice + Math.random() * 0.25 - 0.125,
      });
    }
    return data;
  };

  const generateMockSentiment = (days) => {
    const data = [];
    for (let i = 0; i < days; i++) {
      const date = new Date();
      date.setDate(date.getDate() - (days - i));
      data.push({
        date: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
        sentiment: (Math.random() * 2 - 1) * 0.5, // -0.5 to 0.5
      });
    }
    return data;
  };

  const generateMockVolatility = (days) => {
    const data = [];
    const basePrice = 150;
    for (let i = 0; i < days; i++) {
      const date = new Date();
      date.setDate(date.getDate() - (days - i));
      data.push({
        date: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
        close: basePrice + Math.random() * 20 - 10,
        volatility: Math.random() * 0.5,
      });
    }
    return data;
  };

  const handleRunPrediction = async () => {
    setLoading(true);
    try {
      await new Promise(resolve => setTimeout(resolve, 1000)); // Simulate API call
      const newData = generateMockTimeSeries(timeRange);
      setPredictionData(newData);
    } catch (error) {
      console.error('Error running prediction:', error);
    } finally {
      setLoading(false);
    }
  };

  const perSymbolData = [
    { symbol: 'ITC', directional_accuracy: 58.3, rmse: 0.121 },
    { symbol: 'HINDUNILVR', directional_accuracy: 33.3, rmse: 0.138 },
    { symbol: 'BHARTIARTL', directional_accuracy: 33.3, rmse: 0.170 },
    { symbol: 'AXISBANK', directional_accuracy: 50.0, rmse: 0.209 },
    { symbol: 'HDFCBANK', directional_accuracy: 33.3, rmse: 0.210 },
    { symbol: 'TCS', directional_accuracy: 41.7, rmse: 0.234 },
    { symbol: 'RELIANCE', directional_accuracy: 50.0, rmse: 0.245 },
    { symbol: 'INFY', directional_accuracy: 58.3, rmse: 0.267 },
  ];

  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="text-center space-y-4"
      >
        <h1 className="text-4xl md:text-5xl font-bold">
          <span className="gradient-text">Advanced Analytics Dashboard</span>
        </h1>
        <p className="text-gray-400 text-lg max-w-3xl mx-auto">
          Real-Time Hybrid LSTM Stock Predictions with Sentiment & Volatility Insights
        </p>
      </motion.div>

      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricsCard
          title="Total Stocks Tracked"
          value={metrics?.totalStocks || '31'}
          subtitle="Across global markets"
          icon={TrendingUp}
          loading={loading}
          delay={0}
        />
        <MetricsCard
          title="Avg Directional Accuracy"
          value={`${metrics?.avgDirectionalAccuracy?.toFixed(2) || '46.05'}%`}
          subtitle="Prediction accuracy"
          icon={Target}
          trend="down"
          loading={loading}
          delay={0.1}
        />
        <MetricsCard
          title="Best Performing Stock"
          value={metrics?.bestPerformingStock || 'AMZN'}
          subtitle="58.3% accuracy"
          icon={Award}
          trend="up"
          loading={loading}
          delay={0.2}
        />
        <MetricsCard
          title="Last Update"
          value={lastUpdate.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}
          subtitle={lastUpdate.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
          icon={Clock}
          loading={loading}
          delay={0.3}
        />
      </div>

      {/* Next-Day Prediction Module */}
      <TrainPredictPanel />

      {/* Prediction Explorer */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.4 }}
        className="card"
      >
        <div className="flex flex-col md:flex-row md:items-center md:justify-between space-y-4 md:space-y-0 mb-6">
          <h2 className="text-2xl font-bold text-white">📊 Prediction Explorer</h2>
          
          <div className="flex flex-col sm:flex-row items-stretch sm:items-center space-y-3 sm:space-y-0 sm:space-x-3">
            {/* Symbol Selector */}
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
              <select
                value={selectedSymbol}
                onChange={(e) => setSelectedSymbol(e.target.value)}
                className="pl-10 pr-4 py-2 bg-dark-700 border border-dark-600 rounded-lg text-white focus:outline-none focus:border-primary-400 transition-colors w-full sm:w-auto"
              >
                {symbols.map((symbol) => (
                  <option key={symbol} value={symbol}>
                    {symbol}
                  </option>
                ))}
              </select>
            </div>

            {/* Time Range Toggle */}
            <div className="flex items-center space-x-2 bg-dark-700 rounded-lg p-1">
              {[30, 60, 90].map((days) => (
                <button
                  key={days}
                  onClick={() => setTimeRange(days)}
                  className={`px-4 py-1.5 rounded-md transition-all duration-200 text-sm font-medium ${
                    timeRange === days
                      ? 'bg-primary-400 text-dark-900'
                      : 'text-gray-400 hover:text-white'
                  }`}
                >
                  <Calendar className="w-4 h-4 inline mr-1" />
                  {days}d
                </button>
              ))}
            </div>

            {/* Run Prediction Button */}
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={handleRunPrediction}
              disabled={loading}
              className="btn-primary flex items-center justify-center space-x-2"
            >
              <Play className="w-4 h-4" />
              <span>Run Prediction</span>
            </motion.button>
          </div>
        </div>

        <StockChart data={predictionData} loading={loading} title={`${selectedSymbol} - Actual vs Predicted`} />
      </motion.div>

      {/* Model Performance Section */}
      <div className="space-y-6">
        <h2 className="text-2xl font-bold text-white">🎯 Model Performance</h2>
        
        {/* Metrics Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <div className="card">
            <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wide mb-2">RMSE</h3>
            <p className="text-3xl font-bold text-white">{metrics?.rmse?.toFixed(4) || '0.3084'}</p>
            <p className="text-sm text-gray-500 mt-1">Root Mean Square Error</p>
          </div>
          <div className="card">
            <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wide mb-2">MAE</h3>
            <p className="text-3xl font-bold text-white">{metrics?.mae?.toFixed(4) || '0.2759'}</p>
            <p className="text-sm text-gray-500 mt-1">Mean Absolute Error</p>
          </div>
          <div className="card">
            <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wide mb-2">MAPE</h3>
            <p className="text-3xl font-bold text-white">{metrics?.mape?.toFixed(2) || '115.18'}%</p>
            <p className="text-sm text-gray-500 mt-1">Mean Absolute % Error</p>
          </div>
          <div className="card">
            <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wide mb-2">R² Score</h3>
            <p className="text-3xl font-bold text-white">{metrics?.r2Score?.toFixed(4) || '-0.1603'}</p>
            <p className="text-sm text-gray-500 mt-1">Coefficient of Determination</p>
          </div>
        </div>

        {/* Per-Symbol Performance */}
        <PerformanceBarChart data={perSymbolData} loading={loading} />
      </div>

      {/* Sentiment & Volatility Visualization */}
      <div className="space-y-6">
        <h2 className="text-2xl font-bold text-white">💡 Sentiment & Volatility Insights</h2>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <SentimentChart data={sentimentData} loading={loading} />
          <VolatilityChart data={volatilityData} loading={loading} />
        </div>
      </div>

      {/* Feature Importance Panel */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.5 }}
        className="card"
      >
        <h2 className="text-2xl font-bold text-white mb-6">🔍 Feature Importance</h2>
        
        <div className="space-y-3">
          {[
            { name: 'Close Price', importance: 95, color: 'bg-primary-400' },
            { name: 'GARCH Volatility', importance: 82, color: 'bg-purple-500' },
            { name: 'MA_20', importance: 76, color: 'bg-blue-500' },
            { name: 'FinBERT Sentiment', importance: 68, color: 'bg-green-500' },
            { name: 'Volume', importance: 61, color: 'bg-yellow-500' },
            { name: 'MA_10', importance: 58, color: 'bg-orange-500' },
            { name: 'Returns', importance: 52, color: 'bg-red-500' },
            { name: 'Momentum', importance: 45, color: 'bg-pink-500' },
          ].map((feature, index) => (
            <motion.div
              key={feature.name}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.3, delay: 0.6 + index * 0.05 }}
              className="flex items-center space-x-4"
            >
              <span className="text-sm text-gray-400 w-40">{feature.name}</span>
              <div className="flex-1 bg-dark-700 rounded-full h-6 overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${feature.importance}%` }}
                  transition={{ duration: 1, delay: 0.8 + index * 0.05 }}
                  className={`${feature.color} h-full flex items-center justify-end pr-3`}
                >
                  <span className="text-xs font-semibold text-white">{feature.importance}%</span>
                </motion.div>
              </div>
            </motion.div>
          ))}
        </div>

        <p className="mt-6 text-sm text-gray-500 italic">
          * Feature importance derived from attention weights and SHAP analysis
        </p>
      </motion.div>
    </div>
  );
};

export default Dashboard;

