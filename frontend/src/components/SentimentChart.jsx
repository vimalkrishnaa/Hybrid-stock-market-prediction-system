import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { motion } from 'framer-motion';
import { Smile, Frown, Meh } from 'lucide-react';

const SentimentChart = ({ data, loading = false }) => {
  if (loading) {
    return (
      <div className="card">
        <div className="loading-skeleton h-8 w-48 mb-4"></div>
        <div className="loading-skeleton h-64 w-full"></div>
      </div>
    );
  }

  if (!data || data.length === 0) {
    return (
      <div className="card">
        <h3 className="text-lg font-semibold mb-4">FinBERT Sentiment Analysis</h3>
        <div className="h-64 flex items-center justify-center text-gray-500">
          No sentiment data available
        </div>
      </div>
    );
  }

  const getSentimentIcon = (value) => {
    if (value > 0.2) return <Smile className="w-5 h-5 text-green-400" />;
    if (value < -0.2) return <Frown className="w-5 h-5 text-red-400" />;
    return <Meh className="w-5 h-5 text-yellow-400" />;
  };

  const avgSentiment = data.reduce((acc, item) => acc + item.sentiment, 0) / data.length;

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const sentiment = payload[0].value;
      return (
        <div className="bg-dark-800 border border-dark-600 rounded-lg p-3 shadow-xl">
          <p className="text-gray-400 text-sm mb-2">{payload[0].payload.date}</p>
          <div className="flex items-center space-x-2">
            {getSentimentIcon(sentiment)}
            <p className={`text-sm font-medium ${
              sentiment > 0 ? 'text-green-400' : sentiment < 0 ? 'text-red-400' : 'text-yellow-400'
            }`}>
              Sentiment: {sentiment.toFixed(3)}
            </p>
          </div>
        </div>
      );
    }
    return null;
  };

  const getGradientId = () => {
    if (avgSentiment > 0.1) return 'colorPositive';
    if (avgSentiment < -0.1) return 'colorNegative';
    return 'colorNeutral';
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.2 }}
      className="card"
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">FinBERT Sentiment Analysis</h3>
        <div className="flex items-center space-x-2 px-3 py-1 bg-dark-700 rounded-lg">
          {getSentimentIcon(avgSentiment)}
          <span className={`text-sm font-medium ${
            avgSentiment > 0 ? 'text-green-400' : avgSentiment < 0 ? 'text-red-400' : 'text-yellow-400'
          }`}>
            Avg: {avgSentiment.toFixed(3)}
          </span>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={300}>
        <AreaChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <defs>
            <linearGradient id="colorPositive" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#22c55e" stopOpacity={0.8}/>
              <stop offset="95%" stopColor="#22c55e" stopOpacity={0.1}/>
            </linearGradient>
            <linearGradient id="colorNegative" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#ef4444" stopOpacity={0.8}/>
              <stop offset="95%" stopColor="#ef4444" stopOpacity={0.1}/>
            </linearGradient>
            <linearGradient id="colorNeutral" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#eab308" stopOpacity={0.8}/>
              <stop offset="95%" stopColor="#eab308" stopOpacity={0.1}/>
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis 
            dataKey="date" 
            stroke="#94a3b8"
            style={{ fontSize: '12px' }}
          />
          <YAxis 
            domain={[-1, 1]}
            stroke="#94a3b8"
            style={{ fontSize: '12px' }}
          />
          <Tooltip content={<CustomTooltip />} />
          <ReferenceLine y={0} stroke="#64748b" strokeDasharray="3 3" />
          <Area 
            type="monotone" 
            dataKey="sentiment" 
            stroke={avgSentiment > 0 ? '#22c55e' : avgSentiment < 0 ? '#ef4444' : '#eab308'}
            strokeWidth={2}
            fillOpacity={1} 
            fill={`url(#${getGradientId()})`}
          />
        </AreaChart>
      </ResponsiveContainer>

      {/* Legend */}
      <div className="flex items-center justify-center space-x-6 mt-4 text-sm">
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 bg-green-400 rounded-full"></div>
          <span className="text-gray-400">Positive (&gt; 0.2)</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 bg-yellow-400 rounded-full"></div>
          <span className="text-gray-400">Neutral (-0.2 to 0.2)</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 bg-red-400 rounded-full"></div>
          <span className="text-gray-400">Negative (&lt; -0.2)</span>
        </div>
      </div>
    </motion.div>
  );
};

export default SentimentChart;

