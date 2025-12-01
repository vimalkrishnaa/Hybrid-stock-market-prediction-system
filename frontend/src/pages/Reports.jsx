import { useState } from 'react';
import { motion } from 'framer-motion';
import { FileText, Download, Eye, Calendar, TrendingUp, AlertTriangle, CheckCircle } from 'lucide-react';

const Reports = () => {
  const [selectedReport, setSelectedReport] = useState(null);

  const reports = [
    {
      id: 1,
      title: 'Hybrid LSTM Evaluation Report',
      description: 'Comprehensive analysis of Hybrid LSTM model performance',
      date: '2025-10-24',
      type: 'evaluation',
      status: 'completed',
      file: 'hybrid_model_evaluation_metrics.json',
      size: '45 KB',
      metrics: {
        rmse: 0.3084,
        mae: 0.2759,
        r2: -0.1603,
        directionalAccuracy: 46.05,
      },
    },
    {
      id: 2,
      title: 'Performance Analysis Report',
      description: 'Detailed statistical analysis and recommendations',
      date: '2025-10-24',
      type: 'analysis',
      status: 'completed',
      file: 'HYBRID_PERFORMANCE_ANALYSIS.md',
      size: '12 KB',
      highlights: [
        'Per-symbol performance breakdown',
        'Feature impact assessment',
        'Statistical residual analysis',
        'Improvement recommendations',
      ],
    },
    {
      id: 3,
      title: 'FinBERT Sentiment Integration',
      description: 'Sentiment analysis integration report',
      date: '2025-10-24',
      type: 'integration',
      status: 'completed',
      file: 'FINBERT_SENTIMENT_INTEGRATION_REPORT.md',
      size: '8 KB',
      stats: {
        textsProcessed: 99,
        datesCovered: 1,
        avgSentiment: 0.1301,
      },
    },
    {
      id: 4,
      title: 'GARCH Volatility Report',
      description: 'GARCH(1,1) volatility modeling results',
      date: '2025-10-24',
      type: 'volatility',
      status: 'completed',
      file: 'GARCH_VOLATILITY_REPORT.md',
      size: '10 KB',
      stats: {
        symbolsModeled: 31,
        avgVolatility: 0.0234,
        maxVolatility: 0.1567,
      },
    },
    {
      id: 5,
      title: 'Data Preprocessing Summary',
      description: 'Extended dataset preprocessing statistics',
      date: '2025-10-24',
      type: 'preprocessing',
      status: 'completed',
      file: 'preprocessing_metadata_with_sentiment_volatility.json',
      size: '6 KB',
      stats: {
        totalRecords: 3980,
        features: 13,
        trainSamples: 1689,
        testSamples: 431,
      },
    },
  ];

  const getStatusBadge = (status) => {
    const styles = {
      completed: 'bg-green-500/20 text-green-400 border-green-500',
      pending: 'bg-yellow-500/20 text-yellow-400 border-yellow-500',
      failed: 'bg-red-500/20 text-red-400 border-red-500',
    };
    
    const icons = {
      completed: CheckCircle,
      pending: AlertTriangle,
      failed: AlertTriangle,
    };

    const Icon = icons[status];
    
    return (
      <span className={`flex items-center space-x-1 px-3 py-1 rounded-full text-xs font-semibold border ${styles[status]}`}>
        <Icon className="w-3 h-3" />
        <span>{status.toUpperCase()}</span>
      </span>
    );
  };

  const getTypeIcon = (type) => {
    const icons = {
      evaluation: TrendingUp,
      analysis: FileText,
      integration: CheckCircle,
      volatility: Activity,
      preprocessing: FileText,
    };
    return icons[type] || FileText;
  };

  const handleDownload = (report) => {
    // Simulate download
    alert(`Downloading: ${report.file}`);
  };

  const handleView = (report) => {
    setSelectedReport(report);
  };

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
          <span className="gradient-text">Evaluation Reports</span>
        </h1>
        <p className="text-gray-400 text-lg max-w-3xl mx-auto">
          Access comprehensive reports, metrics, and analysis documents
        </p>
      </motion.div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="card"
        >
          <div className="flex items-center space-x-3 mb-2">
            <FileText className="w-6 h-6 text-primary-400" />
            <h3 className="text-lg font-semibold text-white">Total Reports</h3>
          </div>
          <p className="text-4xl font-bold text-white">{reports.length}</p>
          <p className="text-sm text-gray-400 mt-1">Available documents</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="card"
        >
          <div className="flex items-center space-x-3 mb-2">
            <CheckCircle className="w-6 h-6 text-green-400" />
            <h3 className="text-lg font-semibold text-white">Completed</h3>
          </div>
          <p className="text-4xl font-bold text-white">
            {reports.filter(r => r.status === 'completed').length}
          </p>
          <p className="text-sm text-gray-400 mt-1">Successfully generated</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          className="card"
        >
          <div className="flex items-center space-x-3 mb-2">
            <Calendar className="w-6 h-6 text-blue-400" />
            <h3 className="text-lg font-semibold text-white">Last Updated</h3>
          </div>
          <p className="text-2xl font-bold text-white">Oct 24, 2025</p>
          <p className="text-sm text-gray-400 mt-1">Most recent report</p>
        </motion.div>
      </div>

      {/* Reports Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {reports.map((report, index) => {
          const TypeIcon = getTypeIcon(report.type);
          
          return (
            <motion.div
              key={report.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.4 + index * 0.1 }}
              className="card card-hover"
            >
              {/* Header */}
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-start space-x-3">
                  <div className="p-3 bg-dark-700 rounded-lg">
                    <TypeIcon className="w-6 h-6 text-primary-400" />
                  </div>
                  <div>
                    <h3 className="text-lg font-bold text-white">{report.title}</h3>
                    <p className="text-sm text-gray-400">{report.description}</p>
                  </div>
                </div>
                {getStatusBadge(report.status)}
              </div>

              {/* Metadata */}
              <div className="flex items-center space-x-4 text-sm text-gray-400 mb-4">
                <div className="flex items-center space-x-1">
                  <Calendar className="w-4 h-4" />
                  <span>{report.date}</span>
                </div>
                <div className="flex items-center space-x-1">
                  <FileText className="w-4 h-4" />
                  <span>{report.file}</span>
                </div>
                <span className="text-gray-500">•</span>
                <span>{report.size}</span>
              </div>

              {/* Content Preview */}
              {report.metrics && (
                <div className="mb-4 p-3 bg-dark-700 rounded-lg">
                  <h4 className="text-sm font-semibold text-gray-400 mb-2">Key Metrics</h4>
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div>
                      <span className="text-gray-500">RMSE:</span>
                      <span className="text-white font-semibold ml-2">{report.metrics.rmse}</span>
                    </div>
                    <div>
                      <span className="text-gray-500">MAE:</span>
                      <span className="text-white font-semibold ml-2">{report.metrics.mae}</span>
                    </div>
                    <div>
                      <span className="text-gray-500">R²:</span>
                      <span className="text-white font-semibold ml-2">{report.metrics.r2}</span>
                    </div>
                    <div>
                      <span className="text-gray-500">Dir. Acc:</span>
                      <span className="text-white font-semibold ml-2">{report.metrics.directionalAccuracy}%</span>
                    </div>
                  </div>
                </div>
              )}

              {report.highlights && (
                <div className="mb-4">
                  <h4 className="text-sm font-semibold text-gray-400 mb-2">Highlights</h4>
                  <ul className="space-y-1">
                    {report.highlights.map((highlight, i) => (
                      <li key={i} className="text-sm text-gray-300 flex items-start">
                        <span className="text-primary-400 mr-2">•</span>
                        {highlight}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {report.stats && (
                <div className="mb-4 p-3 bg-dark-700 rounded-lg">
                  <h4 className="text-sm font-semibold text-gray-400 mb-2">Statistics</h4>
                  <div className="space-y-1 text-sm">
                    {Object.entries(report.stats).map(([key, value]) => (
                      <div key={key} className="flex items-center justify-between">
                        <span className="text-gray-500 capitalize">
                          {key.replace(/([A-Z])/g, ' $1').trim()}:
                        </span>
                        <span className="text-white font-semibold">{value}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Actions */}
              <div className="flex items-center space-x-3 pt-4 border-t border-dark-700">
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => handleView(report)}
                  className="flex-1 flex items-center justify-center space-x-2 bg-primary-400 hover:bg-primary-500 text-dark-900 font-semibold px-4 py-2 rounded-lg transition-all duration-200"
                >
                  <Eye className="w-4 h-4" />
                  <span>View</span>
                </motion.button>
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => handleDownload(report)}
                  className="flex-1 flex items-center justify-center space-x-2 bg-dark-700 hover:bg-dark-600 text-gray-100 font-semibold px-4 py-2 rounded-lg transition-all duration-200 border border-dark-600"
                >
                  <Download className="w-4 h-4" />
                  <span>Download</span>
                </motion.button>
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* Selected Report Modal */}
      {selectedReport && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4"
          onClick={() => setSelectedReport(null)}
        >
          <motion.div
            initial={{ scale: 0.9, y: 20 }}
            animate={{ scale: 1, y: 0 }}
            className="bg-dark-800 rounded-2xl p-6 max-w-2xl w-full max-h-[80vh] overflow-y-auto border border-dark-700"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-start justify-between mb-4">
              <h2 className="text-2xl font-bold text-white">{selectedReport.title}</h2>
              <button
                onClick={() => setSelectedReport(null)}
                className="text-gray-400 hover:text-white transition-colors"
              >
                ✕
              </button>
            </div>
            <p className="text-gray-400 mb-4">{selectedReport.description}</p>
            <div className="p-4 bg-dark-700 rounded-lg">
              <p className="text-gray-300">
                Full report content would be displayed here. This is a preview modal.
              </p>
            </div>
          </motion.div>
        </motion.div>
      )}
    </div>
  );
};

export default Reports;

