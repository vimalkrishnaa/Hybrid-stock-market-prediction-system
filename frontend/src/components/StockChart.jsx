import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { motion } from 'framer-motion';

const StockChart = ({ data, loading = false, title = "Stock Price Prediction" }) => {
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
        <h3 className="text-lg font-semibold mb-4">{title}</h3>
        <div className="h-64 flex items-center justify-center text-gray-500">
          No data available
        </div>
      </div>
    );
  }

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
      transition={{ duration: 0.5 }}
      className="card"
    >
      <h3 className="text-lg font-semibold mb-4 text-white">{title}</h3>
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis 
            dataKey="date" 
            stroke="#94a3b8"
            style={{ fontSize: '12px' }}
          />
          <YAxis 
            stroke="#94a3b8"
            style={{ fontSize: '12px' }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend 
            wrapperStyle={{ paddingTop: '20px' }}
            iconType="line"
          />
          <Line 
            type="monotone" 
            dataKey="actual" 
            stroke="#22d3ee" 
            strokeWidth={2}
            dot={{ fill: '#22d3ee', r: 3 }}
            activeDot={{ r: 5 }}
            name="Actual Price"
          />
          <Line 
            type="monotone" 
            dataKey="predicted" 
            stroke="#f97316" 
            strokeWidth={2}
            strokeDasharray="5 5"
            dot={{ fill: '#f97316', r: 3 }}
            activeDot={{ r: 5 }}
            name="Predicted Price"
          />
        </LineChart>
      </ResponsiveContainer>
    </motion.div>
  );
};

export default StockChart;

