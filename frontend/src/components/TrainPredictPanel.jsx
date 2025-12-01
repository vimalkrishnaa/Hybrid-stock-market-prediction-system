import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Play, Calendar, TrendingUp, TrendingDown, AlertCircle, CheckCircle, Newspaper, ExternalLink, Smile, Frown, Meh } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { apiService } from '../utils/api';

const TrainPredictPanel = () => {
  const [symbol, setSymbol] = useState('RELIANCE');
  const [startDate, setStartDate] = useState('2025-04-27');
  const [endDate, setEndDate] = useState('2025-10-24');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [useFullPipeline, setUseFullPipeline] = useState(true); // Default to full pipeline
  const [progress, setProgress] = useState(null);
  const [symbols, setSymbols] = useState([
    // Indian stocks only
    'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'HINDUNILVR', 'BHARTIARTL',
    'AXISBANK', 'KOTAKBANK', 'SBIN', 'ITC'
  ]);

  // Fetch available symbols from backend
  useEffect(() => {
    const fetchSymbols = async () => {
      try {
        const response = await apiService.getSymbols();
        if (response && response.symbols && response.symbols.length > 0) {
          setSymbols(response.symbols);
        }
      } catch (err) {
        console.warn('Failed to fetch symbols from API, using fallback list:', err);
        // Keep fallback list if API fails
      }
    };
    fetchSymbols();
  }, []);

  const handlePredict = async () => {
    if (!symbol || !startDate || !endDate) {
      setError('Please select symbol and date range');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);
    setProgress(null);

    try {
      let response;
      if (useFullPipeline) {
        // Use full pipeline: fetch, preprocess, train, predict
        setProgress({ step: 'collecting', progress: 10, message: 'Fetching new stock data...' });
        response = await apiService.trainAndPredict(symbol, startDate, endDate);
        if (response.status) {
          setProgress(response.status);
        }
      } else {
        // Use existing data and pre-trained model
        response = await apiService.predictNextDay(symbol, startDate, endDate);
          }
      setResult(response);
          // Debug: Log news articles
          console.log('Prediction result:', response);
          console.log('News articles received:', response.news_articles);
          console.log('News articles count:', response.news_articles?.length || 0);
          setProgress({ step: 'complete', progress: 100, message: 'Prediction completed!' });
    } catch (err) {
      setError(err.message || 'Failed to generate prediction');
      console.error('Prediction error:', err);
      setProgress({ step: 'error', progress: 0, message: 'An error occurred' });
    } finally {
      setLoading(false);
    }
  };

  const getAccuracyColor = (accuracy) => {
    if (accuracy >= 60) return 'text-green-400';
    if (accuracy >= 40) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getAccuracyBgColor = (accuracy) => {
    if (accuracy >= 60) return 'bg-green-500';
    if (accuracy >= 40) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  const getPriceChange = () => {
    if (!result) return { value: 0, isPositive: true };
    const change = result.predicted_close - result.last_close;
    return {
      value: Math.abs(change),
      percentage: ((change / result.last_close) * 100).toFixed(2),
      isPositive: change >= 0
    };
  };

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-dark-800 border border-dark-600 rounded-lg p-3 shadow-xl">
          <p className="text-gray-400 text-sm mb-1">{data.date}</p>
          <p className="text-white font-semibold">
            Close: {result?.currency || '₹'}{data.close.toFixed(2)}
          </p>
          {data.predicted && (
            <p className="text-primary-400 text-xs mt-1">Predicted</p>
          )}
        </div>
      );
    }
    return null;
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="card"
    >
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-2xl font-bold text-white">🎯 Next-Day Prediction Module</h2>
          <p className="text-gray-400 text-sm mt-1">
            Select date range and train model for custom prediction
          </p>
        </div>
      </div>

      {/* Input Section */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        {/* Symbol Selector */}
        <div>
          <label className="block text-sm font-medium text-gray-400 mb-2">
            Stock Symbol
          </label>
          <select
            value={symbol}
            onChange={(e) => setSymbol(e.target.value)}
            className="w-full px-4 py-2 bg-dark-700 border border-dark-600 rounded-lg text-white focus:outline-none focus:border-primary-400 transition-colors"
          >
            {symbols.map((sym) => (
              <option key={sym} value={sym}>
                {sym}
              </option>
            ))}
          </select>
        </div>

        {/* Start Date */}
        <div>
          <label className="block text-sm font-medium text-gray-400 mb-2">
            <Calendar className="w-4 h-4 inline mr-1" />
            Start Date
          </label>
          <input
            type="date"
            value={startDate}
            onChange={(e) => setStartDate(e.target.value)}
            className="w-full px-4 py-2 bg-dark-700 border border-dark-600 rounded-lg text-white focus:outline-none focus:border-primary-400 transition-colors"
          />
        </div>

        {/* End Date */}
        <div>
          <label className="block text-sm font-medium text-gray-400 mb-2">
            <Calendar className="w-4 h-4 inline mr-1" />
            End Date
          </label>
          <input
            type="date"
            value={endDate}
            onChange={(e) => setEndDate(e.target.value)}
            className="w-full px-4 py-2 bg-dark-700 border border-dark-600 rounded-lg text-white focus:outline-none focus:border-primary-400 transition-colors"
          />
        </div>

        {/* Predict Button */}
        <div className="flex flex-col items-end space-y-2">
          {/* Full Pipeline Toggle */}
          <label className="flex items-center space-x-2 text-sm text-gray-400 cursor-pointer">
            <input
              type="checkbox"
              checked={useFullPipeline}
              onChange={(e) => setUseFullPipeline(e.target.checked)}
              className="w-4 h-4 rounded border-dark-600 bg-dark-700 text-primary-400 focus:ring-primary-400"
            />
            <span>Fetch new data & train</span>
          </label>
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={handlePredict}
            disabled={loading || !symbol || !startDate || !endDate}
            className={`w-full flex items-center justify-center space-x-2 px-6 py-2 rounded-lg font-semibold transition-all duration-200 ${
              loading || !symbol || !startDate || !endDate
                ? 'bg-dark-600 text-gray-500 cursor-not-allowed'
                : 'bg-primary-400 hover:bg-primary-500 text-dark-900'
            }`}
          >
            {loading ? (
              <>
                <div className="w-5 h-5 border-2 border-dark-900 border-t-transparent rounded-full animate-spin"></div>
                <span>{useFullPipeline ? 'Training...' : 'Predicting...'}</span>
              </>
            ) : (
              <>
                <Play className="w-5 h-5" />
                <span>Train & Predict</span>
              </>
            )}
          </motion.button>
        </div>
      </div>

      {/* Progress Indicator */}
      {loading && progress && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-6 p-4 bg-dark-700 border border-dark-600 rounded-lg"
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-gray-300">{progress.message}</span>
            <span className="text-sm text-primary-400">{progress.progress}%</span>
          </div>
          <div className="w-full bg-dark-600 rounded-full h-2">
            <motion.div
              className="bg-primary-400 h-2 rounded-full"
              initial={{ width: 0 }}
              animate={{ width: `${progress.progress}%` }}
              transition={{ duration: 0.3 }}
            />
          </div>
          <div className="mt-2 text-xs text-gray-500">
            Step: {progress.step}
          </div>
        </motion.div>
      )}

      {/* Error Message */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="mb-6 p-4 bg-red-500/10 border border-red-500 rounded-lg flex items-start space-x-3"
          >
            <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-red-400 font-semibold">Prediction Failed</p>
              <p className="text-red-300 text-sm mt-1">{error}</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Loading State */}
      <AnimatePresence>
        {loading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="mb-6"
          >
            <div className="bg-dark-700 rounded-lg p-8">
              <div className="flex flex-col items-center space-y-4">
                <div className="w-16 h-16 border-4 border-primary-400 border-t-transparent rounded-full animate-spin"></div>
                <div className="text-center">
                  <p className="text-white font-semibold">Training Model...</p>
                  <p className="text-gray-400 text-sm mt-1">
                    Analyzing {symbol} from {startDate} to {endDate}
                  </p>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Results */}
      <AnimatePresence>
        {result && !loading && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            className="space-y-6"
          >
            {/* Success Banner */}
            <div className="p-4 bg-green-500/10 border border-green-500 rounded-lg flex items-center space-x-3">
              <CheckCircle className="w-5 h-5 text-green-400" />
              <p className="text-green-400 font-semibold">
                Prediction completed successfully!
              </p>
            </div>

            {/* Prediction Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* Predicted Price */}
              <div className="card bg-gradient-to-br from-primary-400/10 to-blue-500/10 border-primary-400">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-sm font-medium text-gray-400 uppercase">
                    Predicted Close
                  </h3>
                  {getPriceChange().isPositive ? (
                    <TrendingUp className="w-5 h-5 text-green-400" />
                  ) : (
                    <TrendingDown className="w-5 h-5 text-red-400" />
                  )}
                </div>
                <p className="text-4xl font-bold text-white mb-1">
                  {result.currency || '₹'}{result.predicted_close.toFixed(2)}
                </p>
                <p className="text-sm text-gray-400">
                  For {result.date_predicted_for}
                </p>
                <div className={`mt-2 text-sm font-semibold ${getPriceChange().isPositive ? 'text-green-400' : 'text-red-400'}`}>
                  {getPriceChange().isPositive ? '+' : '-'}{result.currency || '₹'}{getPriceChange().value.toFixed(2)} 
                  ({getPriceChange().isPositive ? '+' : ''}{getPriceChange().percentage}%)
                </div>
              </div>

              {/* Last Actual Price */}
              <div className="card">
                <h3 className="text-sm font-medium text-gray-400 uppercase mb-2">
                  Last Actual Close
                </h3>
                <p className="text-4xl font-bold text-white mb-1">
                  {result.currency || '₹'}{result.last_close.toFixed(2)}
                </p>
                <p className="text-sm text-gray-400">
                  On {endDate}
                </p>
              </div>

              {/* Directional Accuracy */}
              <div className="card">
                <h3 className="text-sm font-medium text-gray-400 uppercase mb-2">
                  Directional Accuracy
                </h3>
                <p className={`text-4xl font-bold ${getAccuracyColor(result.directional_accuracy)}`}>
                  {result.directional_accuracy.toFixed(1)}%
                </p>
                <div className="mt-3 bg-dark-700 rounded-full h-2 overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${result.directional_accuracy}%` }}
                    transition={{ duration: 1, delay: 0.3 }}
                    className={`h-full ${getAccuracyBgColor(result.directional_accuracy)}`}
                  />
                </div>
              </div>
            </div>

            {/* Metrics Grid */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="card">
                <h4 className="text-sm font-medium text-gray-400 uppercase mb-1">RMSE</h4>
                <p className="text-2xl font-bold text-white">{result.rmse.toFixed(4)}</p>
                <p className="text-xs text-gray-500 mt-1">Root Mean Square Error</p>
              </div>
              <div className="card">
                <h4 className="text-sm font-medium text-gray-400 uppercase mb-1">MAPE</h4>
                <p className="text-2xl font-bold text-white">{result.mape.toFixed(2)}%</p>
                <p className="text-xs text-gray-500 mt-1">Mean Absolute % Error</p>
              </div>
              <div className="card">
                <h4 className="text-sm font-medium text-gray-400 uppercase mb-1">R² Score</h4>
                <p className="text-2xl font-bold text-white">{result.r2.toFixed(4)}</p>
                <p className="text-xs text-gray-500 mt-1">Coefficient of Determination</p>
              </div>
            </div>

            {/* Chart */}
            <div className="card">
              <h3 className="text-lg font-semibold text-white mb-4">
                Recent Price Trend & Prediction
              </h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={result.recent_data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis 
                    dataKey="date" 
                    stroke="#94a3b8"
                    style={{ fontSize: '12px' }}
                    angle={-45}
                    textAnchor="end"
                    height={80}
                  />
                  <YAxis 
                    stroke="#94a3b8"
                    style={{ fontSize: '12px' }}
                    domain={['auto', 'auto']}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <ReferenceLine 
                    x={result.date_predicted_for} 
                    stroke="#22d3ee" 
                    strokeDasharray="3 3"
                    label={{ value: 'Prediction', fill: '#22d3ee', fontSize: 12 }}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="close" 
                    stroke="#22d3ee" 
                    strokeWidth={2}
                    dot={(props) => {
                      const { cx, cy, payload } = props;
                      if (payload.predicted) {
                        return (
                          <circle 
                            cx={cx} 
                            cy={cy} 
                            r={6} 
                            fill="#f97316" 
                            stroke="#fff" 
                            strokeWidth={2}
                          />
                        );
                      }
                      return <circle cx={cx} cy={cy} r={3} fill="#22d3ee" />;
                    }}
                    activeDot={{ r: 5 }}
                  />
                </LineChart>
              </ResponsiveContainer>
              <div className="mt-4 flex items-center justify-center space-x-6 text-sm">
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 bg-primary-400 rounded-full"></div>
                  <span className="text-gray-400">Actual Prices</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 bg-orange-500 rounded-full"></div>
                  <span className="text-gray-400">Predicted Price</span>
                </div>
              </div>
            </div>

            {/* News Articles Section */}
            {/* Debug: Always show news section info */}
            {result.news_articles !== undefined && (
              <div className="card">
                {result.news_articles && result.news_articles.length > 0 ? (
                  <>
                    <div className="flex items-center space-x-2 mb-4">
                      <Newspaper className="w-5 h-5 text-primary-400" />
                      <h3 className="text-lg font-semibold text-white">
                        News Articles Used for Prediction
                      </h3>
                      <span className="text-sm text-gray-400">({result.news_articles.length} articles)</span>
                    </div>
                    <div className="space-y-4 max-h-96 overflow-y-auto">
                      {result.news_articles.map((article, index) => (
                    <motion.div
                      key={index}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.1 }}
                      className="p-4 bg-dark-800 border border-dark-600 rounded-lg hover:border-primary-400/50 transition-colors"
                    >
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex-1">
                          <h4 className="text-white font-semibold mb-1 line-clamp-2">
                            {article.title}
                          </h4>
                          <p className="text-sm text-gray-400 mb-2 line-clamp-2">
                            {article.description}
                          </p>
                        </div>
                        <div className={`ml-4 flex items-center space-x-2 px-3 py-1 rounded-full ${
                          article.sentiment_label === 'positive' ? 'bg-green-500/20 text-green-400' :
                          article.sentiment_label === 'negative' ? 'bg-red-500/20 text-red-400' :
                          'bg-gray-500/20 text-gray-400'
                        }`}>
                          {article.sentiment_label === 'positive' ? (
                            <Smile className="w-4 h-4" />
                          ) : article.sentiment_label === 'negative' ? (
                            <Frown className="w-4 h-4" />
                          ) : (
                            <Meh className="w-4 h-4" />
                          )}
                          <span className="text-xs font-medium capitalize">
                            {article.sentiment_label}
                          </span>
                        </div>
                      </div>
                      <div className="flex items-center justify-between mt-3 pt-3 border-t border-dark-600">
                        <div className="flex items-center space-x-4 text-xs text-gray-500">
                          <span>{article.source}</span>
                          <span>•</span>
                          <span>{article.date}</span>
                          <span>•</span>
                          <span className="text-primary-400">
                            Sentiment: {article.sentiment_score > 0 ? '+' : ''}{article.sentiment_score.toFixed(3)}
                          </span>
                        </div>
                        {article.url && (
                          <a
                            href={article.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex items-center space-x-1 text-primary-400 hover:text-primary-300 text-xs transition-colors"
                          >
                            <span>Read more</span>
                            <ExternalLink className="w-3 h-3" />
                          </a>
                        )}
                      </div>
                    </motion.div>
                  ))}
                    </div>
                  </>
                ) : (
                  <div className="p-4 text-center">
                    <Newspaper className="w-8 h-8 text-gray-500 mx-auto mb-2" />
                    <p className="text-gray-400 text-sm">
                      No news articles found for this date range
                    </p>
                    <p className="text-gray-500 text-xs mt-1">
                      NewsAPI free tier only covers the last 30 days. Try using a date range with recent dates.
                    </p>
                  </div>
                )}
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

export default TrainPredictPanel;




