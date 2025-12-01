import { ComposedChart, Line, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { motion } from 'framer-motion';
import { Activity } from 'lucide-react';

const VolatilityChart = ({ data, loading = false }) => {
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
        <h3 className="text-lg font-semibold mb-4">GARCH Volatility Analysis</h3>
        <div className="h-64 flex items-center justify-center text-gray-500">
          No volatility data available
        </div>
      </div>
    );
  }

  const avgVolatility = data.reduce((acc, item) => acc + item.volatility, 0) / data.length;
  const maxVolatility = Math.max(...data.map(item => item.volatility));

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-dark-800 border border-dark-600 rounded-lg p-3 shadow-xl">
          <p className="text-gray-400 text-sm mb-2">{payload[0].payload.date}</p>
          {payload.map((entry, index) => (
            <p key={index} className="text-sm font-medium" style={{ color: entry.color }}>
              {entry.name}: {entry.value.toFixed(4)}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.3 }}
      className="card"
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">GARCH(1,1) Volatility Analysis</h3>
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2 px-3 py-1 bg-dark-700 rounded-lg">
            <Activity className="w-4 h-4 text-purple-400" />
            <span className="text-sm text-gray-400">Avg: 
              <span className="text-purple-400 font-medium ml-1">{avgVolatility.toFixed(4)}</span>
            </span>
          </div>
          <div className="flex items-center space-x-2 px-3 py-1 bg-dark-700 rounded-lg">
            <span className="text-sm text-gray-400">Max: 
              <span className="text-red-400 font-medium ml-1">{maxVolatility.toFixed(4)}</span>
            </span>
          </div>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={300}>
        <ComposedChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <defs>
            <linearGradient id="colorVolatility" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#a855f7" stopOpacity={0.6}/>
              <stop offset="95%" stopColor="#a855f7" stopOpacity={0.1}/>
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis 
            dataKey="date" 
            stroke="#94a3b8"
            style={{ fontSize: '12px' }}
          />
          <YAxis 
            yAxisId="left"
            stroke="#94a3b8"
            style={{ fontSize: '12px' }}
            label={{ value: 'Price', angle: -90, position: 'insideLeft', fill: '#94a3b8' }}
          />
          <YAxis 
            yAxisId="right"
            orientation="right"
            stroke="#a855f7"
            style={{ fontSize: '12px' }}
            label={{ value: 'Volatility', angle: 90, position: 'insideRight', fill: '#a855f7' }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend 
            wrapperStyle={{ paddingTop: '20px' }}
          />
          <Area 
            yAxisId="right"
            type="monotone" 
            dataKey="volatility" 
            fill="url(#colorVolatility)"
            stroke="#a855f7"
            strokeWidth={2}
            name="GARCH Volatility"
          />
          <Line 
            yAxisId="left"
            type="monotone" 
            dataKey="close" 
            stroke="#22d3ee" 
            strokeWidth={2}
            dot={false}
            name="Close Price"
          />
        </ComposedChart>
      </ResponsiveContainer>

      {/* Info */}
      <div className="mt-4 p-3 bg-dark-700 rounded-lg">
        <p className="text-sm text-gray-400">
          <span className="font-semibold text-purple-400">GARCH(1,1)</span> models conditional volatility, 
          capturing time-varying risk. Higher values indicate increased market uncertainty.
        </p>
      </div>
    </motion.div>
  );
};

export default VolatilityChart;

