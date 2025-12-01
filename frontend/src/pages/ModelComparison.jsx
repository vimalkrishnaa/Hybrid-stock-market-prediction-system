import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Trophy, TrendingUp, AlertCircle } from 'lucide-react';

const ModelComparison = () => {
  const [loading, setLoading] = useState(false);
  const [selectedMetric, setSelectedMetric] = useState('rmse');

  const models = [
    {
      name: 'Hybrid LSTM',
      description: 'LSTM with FinBERT Sentiment + GARCH Volatility',
      rmse: 0.3084,
      mae: 0.2759,
      mape: 115.18,
      r2: -0.1603,
      directionalAccuracy: 46.05,
      bestSymbol: 'AMZN',
      features: 13,
      params: '122,274',
      status: 'active',
    },
    {
      name: 'Baseline LSTM',
      description: 'LSTM with Technical Indicators only',
      rmse: 0.3245,
      mae: 0.2891,
      mape: 121.45,
      r2: -0.2134,
      directionalAccuracy: 43.21,
      bestSymbol: 'TCS',
      features: 11,
      params: '98,561',
      status: 'baseline',
    },
    {
      name: 'GRU Model',
      description: 'GRU with Hybrid Features',
      rmse: 0.3156,
      mae: 0.2812,
      mape: 117.89,
      r2: -0.1845,
      directionalAccuracy: 44.67,
      bestSymbol: 'INFY',
      features: 13,
      params: '115,892',
      status: 'experimental',
    },
    {
      name: 'Transformer',
      description: 'Attention-based Transformer Model',
      rmse: 0.3389,
      mae: 0.3012,
      mape: 125.34,
      r2: -0.2567,
      directionalAccuracy: 41.89,
      bestSymbol: 'RELIANCE',
      features: 13,
      params: '245,678',
      status: 'experimental',
    },
  ];

  const getStatusBadge = (status) => {
    const styles = {
      active: 'bg-green-500/20 text-green-400 border-green-500',
      baseline: 'bg-blue-500/20 text-blue-400 border-blue-500',
      experimental: 'bg-yellow-500/20 text-yellow-400 border-yellow-500',
    };
    
    return (
      <span className={`px-3 py-1 rounded-full text-xs font-semibold border ${styles[status]}`}>
        {status.toUpperCase()}
      </span>
    );
  };

  const getBestModel = (metric) => {
    if (metric === 'directionalAccuracy') {
      return models.reduce((best, model) => 
        model[metric] > best[metric] ? model : best
      );
    }
    // For error metrics, lower is better
    return models.reduce((best, model) => 
      model[metric] < best[metric] ? model : best
    );
  };

  const bestModel = getBestModel(selectedMetric);

  return (
    <div className="space-y-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="text-center space-y-4"
      >
        <h1 className="text-4xl md:text-5xl font-bold">
          <span className="gradient-text">Model Comparison</span>
        </h1>
        <p className="text-gray-400 text-lg max-w-3xl mx-auto">
          Compare performance metrics across different model architectures
        </p>
      </motion.div>

      {/* Metric Selector */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.1 }}
        className="card"
      >
        <h3 className="text-lg font-semibold text-white mb-4">Select Comparison Metric</h3>
        <div className="flex flex-wrap gap-3">
          {[
            { key: 'rmse', label: 'RMSE', desc: 'Root Mean Square Error' },
            { key: 'mae', label: 'MAE', desc: 'Mean Absolute Error' },
            { key: 'mape', label: 'MAPE', desc: 'Mean Absolute % Error' },
            { key: 'r2', label: 'R² Score', desc: 'Coefficient of Determination' },
            { key: 'directionalAccuracy', label: 'Dir. Accuracy', desc: 'Prediction Direction' },
          ].map((metric) => (
            <button
              key={metric.key}
              onClick={() => setSelectedMetric(metric.key)}
              className={`px-4 py-3 rounded-lg transition-all duration-200 border ${
                selectedMetric === metric.key
                  ? 'bg-primary-400 text-dark-900 border-primary-400 font-semibold'
                  : 'bg-dark-700 text-gray-300 border-dark-600 hover:border-primary-400'
              }`}
            >
              <div className="text-sm font-medium">{metric.label}</div>
              <div className="text-xs opacity-75">{metric.desc}</div>
            </button>
          ))}
        </div>
      </motion.div>

      {/* Best Model Highlight */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5, delay: 0.2 }}
        className="card bg-gradient-to-br from-primary-400/10 to-blue-500/10 border-primary-400"
      >
        <div className="flex items-center space-x-3 mb-4">
          <Trophy className="w-8 h-8 text-primary-400" />
          <div>
            <h3 className="text-xl font-bold text-white">Best Performer</h3>
            <p className="text-sm text-gray-400">Based on {selectedMetric.toUpperCase()}</p>
          </div>
        </div>
        <div className="flex items-center justify-between">
          <div>
            <h4 className="text-2xl font-bold text-primary-400">{bestModel.name}</h4>
            <p className="text-gray-400">{bestModel.description}</p>
          </div>
          <div className="text-right">
            <p className="text-3xl font-bold text-white">
              {typeof bestModel[selectedMetric] === 'number' 
                ? bestModel[selectedMetric].toFixed(4) 
                : bestModel[selectedMetric]}
              {selectedMetric === 'mape' || selectedMetric === 'directionalAccuracy' ? '%' : ''}
            </p>
            <p className="text-sm text-gray-400">{selectedMetric.toUpperCase()}</p>
          </div>
        </div>
      </motion.div>

      {/* Comparison Table */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.3 }}
        className="card overflow-x-auto"
      >
        <h3 className="text-xl font-bold text-white mb-6">Detailed Comparison</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-left">
            <thead>
              <tr className="border-b border-dark-600">
                <th className="pb-4 pr-6 text-gray-400 font-semibold">Model</th>
                <th className="pb-4 pr-6 text-gray-400 font-semibold">Status</th>
                <th className="pb-4 pr-6 text-gray-400 font-semibold">RMSE</th>
                <th className="pb-4 pr-6 text-gray-400 font-semibold">MAE</th>
                <th className="pb-4 pr-6 text-gray-400 font-semibold">MAPE</th>
                <th className="pb-4 pr-6 text-gray-400 font-semibold">R²</th>
                <th className="pb-4 pr-6 text-gray-400 font-semibold">Dir. Acc.</th>
                <th className="pb-4 pr-6 text-gray-400 font-semibold">Best Symbol</th>
                <th className="pb-4 text-gray-400 font-semibold">Parameters</th>
              </tr>
            </thead>
            <tbody>
              {models.map((model, index) => (
                <motion.tr
                  key={model.name}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.3, delay: 0.4 + index * 0.1 }}
                  className="border-b border-dark-700 hover:bg-dark-700/50 transition-colors"
                >
                  <td className="py-4 pr-6">
                    <div>
                      <p className="font-semibold text-white">{model.name}</p>
                      <p className="text-xs text-gray-500">{model.description}</p>
                    </div>
                  </td>
                  <td className="py-4 pr-6">{getStatusBadge(model.status)}</td>
                  <td className="py-4 pr-6 text-white font-mono">{model.rmse.toFixed(4)}</td>
                  <td className="py-4 pr-6 text-white font-mono">{model.mae.toFixed(4)}</td>
                  <td className="py-4 pr-6 text-white font-mono">{model.mape.toFixed(2)}%</td>
                  <td className="py-4 pr-6 text-white font-mono">{model.r2.toFixed(4)}</td>
                  <td className="py-4 pr-6 text-white font-mono">{model.directionalAccuracy.toFixed(2)}%</td>
                  <td className="py-4 pr-6">
                    <span className="px-2 py-1 bg-primary-400/20 text-primary-400 rounded text-sm font-semibold">
                      {model.bestSymbol}
                    </span>
                  </td>
                  <td className="py-4 text-gray-400">{model.params}</td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        </div>
      </motion.div>

      {/* Model Architecture Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {models.map((model, index) => (
          <motion.div
            key={model.name}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.5 + index * 0.1 }}
            className="card card-hover"
          >
            <div className="flex items-start justify-between mb-4">
              <div>
                <h4 className="text-lg font-bold text-white">{model.name}</h4>
                <p className="text-sm text-gray-400">{model.description}</p>
              </div>
              {getStatusBadge(model.status)}
            </div>

            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-400">Features</span>
                <span className="text-white font-semibold">{model.features}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-400">Parameters</span>
                <span className="text-white font-semibold">{model.params}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-400">Best Symbol</span>
                <span className="text-primary-400 font-semibold">{model.bestSymbol}</span>
              </div>
            </div>

            <div className="mt-4 pt-4 border-t border-dark-700">
              <div className="grid grid-cols-2 gap-3 text-center">
                <div>
                  <p className="text-xs text-gray-400 mb-1">RMSE</p>
                  <p className="text-lg font-bold text-white">{model.rmse.toFixed(4)}</p>
                </div>
                <div>
                  <p className="text-xs text-gray-400 mb-1">Dir. Acc.</p>
                  <p className="text-lg font-bold text-white">{model.directionalAccuracy.toFixed(1)}%</p>
                </div>
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Insights */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.9 }}
        className="card bg-blue-500/10 border-blue-500"
      >
        <div className="flex items-start space-x-3">
          <AlertCircle className="w-6 h-6 text-blue-400 flex-shrink-0 mt-1" />
          <div>
            <h4 className="text-lg font-semibold text-white mb-2">Key Insights</h4>
            <ul className="space-y-2 text-gray-300">
              <li>• <strong>Hybrid LSTM</strong> shows best overall performance with sentiment and volatility features</li>
              <li>• <strong>GRU Model</strong> offers competitive performance with fewer parameters</li>
              <li>• <strong>Transformer</strong> requires more data and tuning for optimal performance</li>
              <li>• All models struggle with directional accuracy, suggesting need for ensemble approaches</li>
              <li>• Feature engineering (sentiment + volatility) provides marginal improvements</li>
            </ul>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default ModelComparison;

