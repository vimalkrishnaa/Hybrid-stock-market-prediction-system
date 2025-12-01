import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => response.data,
  (error) => {
    const message = error.response?.data?.detail || error.message || 'An error occurred';
    return Promise.reject(new Error(message));
  }
);

// API endpoints
export const apiService = {
  // Get prediction for a symbol
  getPrediction: async (symbol) => {
    return api.get(`/predict`, { params: { symbol } });
  },

  // Get overall and per-symbol metrics
  getMetrics: async () => {
    return api.get('/metrics');
  },

  // Get sentiment data for a symbol
  getSentiment: async (symbol) => {
    return api.get('/sentiment', { params: { symbol } });
  },

  // Get volatility data for a symbol
  getVolatility: async (symbol) => {
    return api.get('/volatility', { params: { symbol } });
  },

  // Get list of available symbols
  getSymbols: async () => {
    return api.get('/symbols');
  },

  // Get historical data for a symbol
  getHistoricalData: async (symbol, days = 60) => {
    return api.get('/historical', { params: { symbol, days } });
  },

  // Get model comparison data
  getModelComparison: async () => {
    return api.get('/model-comparison');
  },

  // Health check
  healthCheck: async () => {
    return api.get('/health');
  },

  // Predict next day with custom date range
  predictNextDay: async (symbol, startDate, endDate) => {
    return api.post('/predict_next_day', {
      symbol,
      start_date: startDate,
      end_date: endDate,
    });
  },

  // Full pipeline: fetch, preprocess, train, and predict
  trainAndPredict: async (symbol, startDate, endDate) => {
    return api.post('/train_and_predict', {
      symbol,
      start_date: startDate,
      end_date: endDate,
    }, {
      timeout: 900000, // 15 minutes timeout for training (large datasets can take longer)
    });
  },
};

export default api;

