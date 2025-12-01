import { motion } from 'framer-motion';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

const MetricsCard = ({ title, value, subtitle, icon: Icon, trend, loading = false, delay = 0 }) => {
  const getTrendIcon = () => {
    if (trend === 'up') return <TrendingUp className="w-4 h-4 text-green-400" />;
    if (trend === 'down') return <TrendingDown className="w-4 h-4 text-red-400" />;
    return <Minus className="w-4 h-4 text-gray-400" />;
  };

  const getTrendColor = () => {
    if (trend === 'up') return 'text-green-400';
    if (trend === 'down') return 'text-red-400';
    return 'text-gray-400';
  };

  if (loading) {
    return (
      <div className="metric-card">
        <div className="loading-skeleton h-10 w-10 rounded-lg mb-3"></div>
        <div className="loading-skeleton h-8 w-24 mb-2"></div>
        <div className="loading-skeleton h-4 w-32"></div>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay }}
      className="metric-card group"
    >
      {/* Icon */}
      <div className="flex items-start justify-between w-full mb-3">
        <div className="p-3 bg-dark-700 rounded-lg group-hover:bg-primary-400 transition-colors duration-300">
          <Icon className="w-6 h-6 text-primary-400 group-hover:text-dark-900 transition-colors duration-300" />
        </div>
        {trend && (
          <div className="flex items-center space-x-1">
            {getTrendIcon()}
          </div>
        )}
      </div>

      {/* Content */}
      <div className="space-y-1">
        <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wide">
          {title}
        </h3>
        <p className="text-3xl font-bold text-white">
          {value}
        </p>
        {subtitle && (
          <p className={`text-sm ${getTrendColor()}`}>
            {subtitle}
          </p>
        )}
      </div>
    </motion.div>
  );
};

export default MetricsCard;

