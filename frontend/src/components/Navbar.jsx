import { Link, useLocation } from 'react-router-dom';
import { TrendingUp, RefreshCw, BarChart3, FileText } from 'lucide-react';
import { motion } from 'framer-motion';
import { useState } from 'react';

const Navbar = () => {
  const location = useLocation();
  const [isRefreshing, setIsRefreshing] = useState(false);

  const navLinks = [
    { path: '/', label: 'Dashboard', icon: TrendingUp },
    { path: '/comparison', label: 'Model Comparison', icon: BarChart3 },
    { path: '/reports', label: 'Reports', icon: FileText },
  ];

  const handleRefresh = () => {
    setIsRefreshing(true);
    // Trigger data refresh
    window.location.reload();
    setTimeout(() => setIsRefreshing(false), 2000);
  };

  return (
    <nav className="bg-dark-800 border-b border-dark-700 sticky top-0 z-50 backdrop-blur-sm bg-opacity-95">
      <div className="container mx-auto px-4 max-w-7xl">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center space-x-3 group">
            <motion.div
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="bg-gradient-to-br from-primary-400 to-blue-500 p-2 rounded-lg shadow-lg"
            >
              <TrendingUp className="w-6 h-6 text-dark-900" />
            </motion.div>
            <div>
              <h1 className="text-xl font-bold gradient-text">IndiTrendAI</h1>
              <p className="text-xs text-gray-400">Hybrid LSTM Predictions</p>
            </div>
          </Link>

          {/* Navigation Links */}
          <div className="hidden md:flex items-center space-x-1">
            {navLinks.map((link) => {
              const Icon = link.icon;
              const isActive = location.pathname === link.path;
              
              return (
                <Link
                  key={link.path}
                  to={link.path}
                  className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all duration-200 ${
                    isActive
                      ? 'bg-primary-400 text-dark-900 font-semibold'
                      : 'text-gray-300 hover:bg-dark-700 hover:text-white'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{link.label}</span>
                </Link>
              );
            })}
          </div>

          {/* Refresh Button */}
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={handleRefresh}
            disabled={isRefreshing}
            className="flex items-center space-x-2 bg-dark-700 hover:bg-dark-600 text-gray-100 px-4 py-2 rounded-lg transition-all duration-200 border border-dark-600"
          >
            <RefreshCw className={`w-4 h-4 ${isRefreshing ? 'animate-spin' : ''}`} />
            <span className="hidden sm:inline">Refresh Data</span>
          </motion.button>
        </div>

        {/* Mobile Navigation */}
        <div className="md:hidden flex items-center justify-around pb-3 space-x-2">
          {navLinks.map((link) => {
            const Icon = link.icon;
            const isActive = location.pathname === link.path;
            
            return (
              <Link
                key={link.path}
                to={link.path}
                className={`flex flex-col items-center space-y-1 px-3 py-2 rounded-lg transition-all duration-200 flex-1 ${
                  isActive
                    ? 'bg-primary-400 text-dark-900'
                    : 'text-gray-400 hover:bg-dark-700 hover:text-white'
                }`}
              >
                <Icon className="w-5 h-5" />
                <span className="text-xs font-medium">{link.label}</span>
              </Link>
            );
          })}
        </div>
      </div>
    </nav>
  );
};

export default Navbar;

