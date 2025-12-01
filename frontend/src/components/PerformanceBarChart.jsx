import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';
import { motion } from 'framer-motion';

const PerformanceBarChart = ({ data, loading = false, title = "Per-Symbol Performance" }) => {
  if (loading) {
    return (
      <div className="card">
        <div className="loading-skeleton h-8 w-48 mb-4"></div>
        <div className="loading-skeleton h-80 w-full"></div>
      </div>
    );
  }

  if (!data || data.length === 0) {
    return (
      <div className="card">
        <h3 className="text-lg font-semibold mb-4">{title}</h3>
        <div className="h-80 flex items-center justify-center text-gray-500">
          No performance data available
        </div>
      </div>
    );
  }

  // Color bars based on performance
  const getColor = (value, metric) => {
    if (metric === 'directional_accuracy') {
      if (value >= 60) return '#22c55e'; // Green
      if (value >= 50) return '#eab308'; // Yellow
      return '#ef4444'; // Red
    }
    // For RMSE/MAE (lower is better)
    if (value < 0.2) return '#22c55e';
    if (value < 0.3) return '#eab308';
    return '#ef4444';
  };

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-dark-800 border border-dark-600 rounded-lg p-3 shadow-xl">
          <p className="text-white font-semibold mb-2">{payload[0].payload.symbol}</p>
          {payload.map((entry, index) => (
            <p key={index} className="text-sm" style={{ color: entry.fill }}>
              {entry.name}: {typeof entry.value === 'number' ? entry.value.toFixed(4) : entry.value}
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
      transition={{ duration: 0.5, delay: 0.1 }}
      className="card"
    >
      <h3 className="text-lg font-semibold mb-4 text-white">{title}</h3>
      <ResponsiveContainer width="100%" height={400}>
        <BarChart 
          data={data} 
          margin={{ top: 5, right: 30, left: 20, bottom: 80 }}
          layout="horizontal"
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis 
            dataKey="symbol" 
            stroke="#94a3b8"
            angle={-45}
            textAnchor="end"
            height={100}
            style={{ fontSize: '11px' }}
          />
          <YAxis 
            stroke="#94a3b8"
            style={{ fontSize: '12px' }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend 
            wrapperStyle={{ paddingTop: '20px' }}
            iconType="circle"
          />
          <Bar 
            dataKey="directional_accuracy" 
            name="Directional Accuracy (%)"
            radius={[8, 8, 0, 0]}
          >
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={getColor(entry.directional_accuracy, 'directional_accuracy')} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      {/* Legend */}
      <div className="flex items-center justify-center space-x-6 mt-4 text-sm">
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 bg-green-500 rounded-full"></div>
          <span className="text-gray-400">Good (≥60%)</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
          <span className="text-gray-400">Fair (50-60%)</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 bg-red-500 rounded-full"></div>
          <span className="text-gray-400">Poor (&lt;50%)</span>
        </div>
      </div>
    </motion.div>
  );
};

export default PerformanceBarChart;

