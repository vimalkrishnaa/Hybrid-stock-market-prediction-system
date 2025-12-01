import { Heart, Github, Linkedin } from 'lucide-react';
import { motion } from 'framer-motion';

const Footer = () => {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="bg-dark-800 border-t border-dark-700 mt-auto">
      <div className="container mx-auto px-4 py-6 max-w-7xl">
        <div className="flex flex-col md:flex-row items-center justify-between space-y-4 md:space-y-0">
          {/* Copyright */}
          <div className="flex items-center space-x-2 text-gray-400">
            <span>Made with</span>
            <motion.div
              animate={{ scale: [1, 1.2, 1] }}
              transition={{ duration: 1, repeat: Infinity, repeatDelay: 1 }}
            >
              <Heart className="w-4 h-4 text-red-500 fill-current" />
            </motion.div>
            <span>by</span>
            <span className="font-semibold text-primary-400">Vimal Krishna</span>
            <span>| IndiTrendAI {currentYear}</span>
          </div>

          {/* Social Links */}
          <div className="flex items-center space-x-4">
            <motion.a
              whileHover={{ scale: 1.1, y: -2 }}
              whileTap={{ scale: 0.95 }}
              href="https://github.com/vimalkrishna"
              target="_blank"
              rel="noopener noreferrer"
              className="p-2 bg-dark-700 hover:bg-dark-600 rounded-lg transition-all duration-200 border border-dark-600"
              aria-label="GitHub"
            >
              <Github className="w-5 h-5 text-gray-300 hover:text-white" />
            </motion.a>
            
            <motion.a
              whileHover={{ scale: 1.1, y: -2 }}
              whileTap={{ scale: 0.95 }}
              href="https://linkedin.com/in/vimalkrishna"
              target="_blank"
              rel="noopener noreferrer"
              className="p-2 bg-dark-700 hover:bg-dark-600 rounded-lg transition-all duration-200 border border-dark-600"
              aria-label="LinkedIn"
            >
              <Linkedin className="w-5 h-5 text-gray-300 hover:text-blue-400" />
            </motion.a>
          </div>
        </div>

        {/* Additional Info */}
        <div className="mt-4 pt-4 border-t border-dark-700 text-center text-sm text-gray-500">
          <p>
            Advanced Analytics Dashboard powered by Hybrid LSTM | 
            <span className="text-primary-400 ml-1">FinBERT Sentiment</span> + 
            <span className="text-primary-400 ml-1">GARCH Volatility</span>
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;

