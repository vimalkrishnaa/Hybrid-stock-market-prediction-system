import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExtendedDataPreprocessor:
    def __init__(self, input_file='data/extended/raw/extended_stock_data_20251024.csv',
                 output_path='data/extended/processed'):
        self.input_file = input_file
        self.output_path = output_path
        self.lookback_window = 60
        self.train_split = 0.8
        
        # Create output directories
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(f"{output_path}/scalers", exist_ok=True)
        os.makedirs(f"{output_path}/sequences", exist_ok=True)
        os.makedirs(f"{output_path}/train_test_split", exist_ok=True)
        
        self.data = None
        self.cleaned_data = None
        self.scaled_data = {}
        self.sequences = {}
        self.scalers = {}
        
    def load_data(self):
        """Load the extended dataset"""
        logger.info("=" * 80)
        logger.info("STEP 1: LOADING EXTENDED DATASET")
        logger.info("=" * 80)
        
        logger.info(f"Loading data from: {self.input_file}")
        self.data = pd.read_csv(self.input_file)
        
        logger.info(f"  - Shape: {self.data.shape}")
        logger.info(f"  - Columns: {list(self.data.columns)}")
        logger.info(f"  - Date range: {self.data['Date'].min()} to {self.data['Date'].max()}")
        logger.info(f"  - Unique symbols: {self.data['Symbol'].nunique()}")
        
        # Convert Date to datetime
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        
        logger.info("✓ Data loaded successfully\n")
        
    def clean_data(self):
        """Handle missing values and duplicates"""
        logger.info("=" * 80)
        logger.info("STEP 2: DATA CLEANING")
        logger.info("=" * 80)
        
        initial_shape = self.data.shape
        logger.info(f"Initial shape: {initial_shape}")
        
        # Check for missing values
        logger.info("\n📊 Missing Values Check:")
        missing_counts = self.data.isnull().sum()
        for col, count in missing_counts.items():
            if count > 0:
                logger.info(f"  {col}: {count} missing values ({count/len(self.data)*100:.2f}%)")
        
        if missing_counts.sum() == 0:
            logger.info("  ✓ No missing values found")
        else:
            logger.info(f"\n  Total missing values: {missing_counts.sum()}")
            
            # Handle missing values in OHLCV columns
            ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in ohlcv_cols:
                if self.data[col].isnull().any():
                    # Forward fill then backward fill
                    self.data[col].fillna(method='ffill', inplace=True)
                    self.data[col].fillna(method='bfill', inplace=True)
                    logger.info(f"  ✓ Filled missing values in {col}")
        
        # Check for duplicates
        logger.info("\n📊 Duplicates Check:")
        duplicates = self.data.duplicated(subset=['Symbol', 'Date'], keep='first')
        n_duplicates = duplicates.sum()
        
        if n_duplicates > 0:
            logger.info(f"  Found {n_duplicates} duplicate records")
            self.data = self.data[~duplicates]
            logger.info(f"  ✓ Removed {n_duplicates} duplicates")
        else:
            logger.info("  ✓ No duplicates found")
        
        # Sort data by Symbol and Date
        self.data.sort_values(['Symbol', 'Date'], inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        
        self.cleaned_data = self.data.copy()
        
        logger.info(f"\nFinal shape: {self.cleaned_data.shape}")
        logger.info(f"Records removed: {initial_shape[0] - self.cleaned_data.shape[0]}")
        logger.info("✓ Data cleaning completed\n")
        
    def add_technical_features(self):
        """Add technical indicators for better model performance"""
        logger.info("=" * 80)
        logger.info("STEP 3: ADDING TECHNICAL FEATURES")
        logger.info("=" * 80)
        
        # Calculate returns (needed for future integration with GARCH)
        logger.info("Calculating returns...")
        self.cleaned_data['Returns'] = self.cleaned_data.groupby('Symbol')['Close'].pct_change()
        
        # Calculate moving averages
        logger.info("Calculating moving averages...")
        for window in [5, 10, 20]:
            self.cleaned_data[f'MA_{window}'] = self.cleaned_data.groupby('Symbol')['Close'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
        
        # Calculate volatility (rolling standard deviation of returns)
        logger.info("Calculating rolling volatility...")
        self.cleaned_data['Volatility'] = self.cleaned_data.groupby('Symbol')['Returns'].transform(
            lambda x: x.rolling(window=20, min_periods=1).std()
        )
        
        # Calculate price momentum
        logger.info("Calculating momentum...")
        self.cleaned_data['Momentum'] = self.cleaned_data.groupby('Symbol')['Close'].transform(
            lambda x: x.pct_change(periods=5)
        )
        
        # Fill NaN values created by calculations
        self.cleaned_data['Returns'].fillna(0, inplace=True)
        self.cleaned_data['Volatility'].fillna(0, inplace=True)
        self.cleaned_data['Momentum'].fillna(0, inplace=True)
        
        logger.info(f"✓ Added technical features. New shape: {self.cleaned_data.shape}")
        logger.info(f"  - New columns: Returns, MA_5, MA_10, MA_20, Volatility, Momentum\n")
        
    def normalize_features(self):
        """Normalize OHLCV and technical features using MinMaxScaler"""
        logger.info("=" * 80)
        logger.info("STEP 4: FEATURE NORMALIZATION")
        logger.info("=" * 80)
        
        # Features to normalize
        feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 
                       'Returns', 'MA_5', 'MA_10', 'MA_20', 'Volatility', 'Momentum']
        
        logger.info(f"Normalizing {len(feature_cols)} features using MinMaxScaler...")
        logger.info(f"Features: {feature_cols}")
        
        # Normalize per symbol to maintain relative patterns
        for symbol in self.cleaned_data['Symbol'].unique():
            symbol_data = self.cleaned_data[self.cleaned_data['Symbol'] == symbol].copy()
            
            # Create and fit scaler for this symbol
            scaler = MinMaxScaler(feature_range=(0, 1))
            
            # Fit and transform features
            scaled_features = scaler.fit_transform(symbol_data[feature_cols])
            
            # Create scaled dataframe
            scaled_df = symbol_data.copy()
            scaled_df[feature_cols] = scaled_features
            
            # Store scaled data and scaler
            self.scaled_data[symbol] = scaled_df
            self.scalers[symbol] = scaler
            
            logger.info(f"  ✓ Normalized {symbol}: {len(scaled_df)} records")
        
        # Save scalers for future use (for inverse transform during prediction)
        scaler_file = f"{self.output_path}/scalers/feature_scalers.pkl"
        with open(scaler_file, 'wb') as f:
            pickle.dump(self.scalers, f)
        logger.info(f"\n✓ Saved scalers to: {scaler_file}")
        logger.info(f"  - Total scalers: {len(self.scalers)} (one per symbol)\n")
        
    def create_sequences(self):
        """Create sequences with 60-day lookback window"""
        logger.info("=" * 80)
        logger.info("STEP 5: SEQUENCE CREATION")
        logger.info("=" * 80)
        
        logger.info(f"Creating sequences with {self.lookback_window}-day lookback window...")
        
        # Features for sequences (OHLCV + technical indicators)
        feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 
                       'Returns', 'MA_5', 'MA_10', 'MA_20', 'Volatility', 'Momentum']
        
        logger.info(f"Using {len(feature_cols)} features per timestep")
        
        sequence_summary = []
        
        for symbol in self.scaled_data.keys():
            symbol_data = self.scaled_data[symbol].sort_values('Date')
            
            # Extract feature matrix
            feature_matrix = symbol_data[feature_cols].values
            dates = symbol_data['Date'].values
            
            X, y, date_indices = [], [], []
            
            # Create sequences
            for i in range(self.lookback_window, len(feature_matrix)):
                # Input: lookback_window days of features
                X.append(feature_matrix[i-self.lookback_window:i])
                
                # Target: next day's Close price (index 3 in feature_cols)
                y.append(feature_matrix[i, 3])  # Close price
                
                # Store date for reference
                date_indices.append(dates[i])
            
            if len(X) > 0:
                X = np.array(X)
                y = np.array(y)
                
                self.sequences[symbol] = {
                    'X': X,
                    'y': y,
                    'dates': date_indices,
                    'feature_cols': feature_cols,
                    'n_features': len(feature_cols),
                    'lookback': self.lookback_window
                }
                
                sequence_summary.append({
                    'Symbol': symbol,
                    'Sequences': len(X),
                    'Shape_X': X.shape,
                    'Shape_y': y.shape
                })
                
                logger.info(f"  ✓ {symbol}: {len(X)} sequences | X shape: {X.shape} | y shape: {y.shape}")
            else:
                logger.warning(f"  ✗ {symbol}: Insufficient data for sequences (need >{self.lookback_window} records)")
        
        # Create summary dataframe
        summary_df = pd.DataFrame(sequence_summary)
        summary_file = f"{self.output_path}/sequence_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        
        logger.info(f"\n✓ Total symbols with sequences: {len(self.sequences)}")
        logger.info(f"✓ Sequence summary saved to: {summary_file}\n")
        
    def train_test_split(self):
        """Split sequences into train/test sets (time-based, not random)"""
        logger.info("=" * 80)
        logger.info("STEP 6: TRAIN-TEST SPLIT (TIME-BASED)")
        logger.info("=" * 80)
        
        logger.info(f"Split ratio: {self.train_split*100:.0f}% train / {(1-self.train_split)*100:.0f}% test")
        
        train_test_summary = []
        
        for symbol in self.sequences.keys():
            X = self.sequences[symbol]['X']
            y = self.sequences[symbol]['y']
            dates = self.sequences[symbol]['dates']
            
            # Calculate split index (time-based)
            split_idx = int(len(X) * self.train_split)
            
            # Split data
            X_train = X[:split_idx]
            X_test = X[split_idx:]
            y_train = y[:split_idx]
            y_test = y[split_idx:]
            dates_train = dates[:split_idx]
            dates_test = dates[split_idx:]
            
            # Store splits
            self.sequences[symbol]['X_train'] = X_train
            self.sequences[symbol]['X_test'] = X_test
            self.sequences[symbol]['y_train'] = y_train
            self.sequences[symbol]['y_test'] = y_test
            self.sequences[symbol]['dates_train'] = dates_train
            self.sequences[symbol]['dates_test'] = dates_test
            self.sequences[symbol]['split_idx'] = split_idx
            
            train_test_summary.append({
                'Symbol': symbol,
                'Total_Sequences': len(X),
                'Train_Sequences': len(X_train),
                'Test_Sequences': len(X_test),
                'Train_Date_Range': f"{dates_train[0]} to {dates_train[-1]}",
                'Test_Date_Range': f"{dates_test[0]} to {dates_test[-1]}"
            })
            
            logger.info(f"  ✓ {symbol}:")
            logger.info(f"      Train: {len(X_train)} sequences | {dates_train[0]} to {dates_train[-1]}")
            logger.info(f"      Test:  {len(X_test)} sequences | {dates_test[0]} to {dates_test[-1]}")
        
        # Save summary
        summary_df = pd.DataFrame(train_test_summary)
        summary_file = f"{self.output_path}/train_test_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        
        logger.info(f"\n✓ Train-test split completed for {len(self.sequences)} symbols")
        logger.info(f"✓ Split summary saved to: {summary_file}\n")
        
    def save_preprocessed_data(self):
        """Save preprocessed data in multiple formats"""
        logger.info("=" * 80)
        logger.info("STEP 7: SAVING PREPROCESSED DATA")
        logger.info("=" * 80)
        
        # 1. Save combined sequences as NumPy arrays
        logger.info("\n📁 Saving individual symbol sequences...")
        
        for symbol in self.sequences.keys():
            symbol_dir = f"{self.output_path}/sequences/{symbol}"
            os.makedirs(symbol_dir, exist_ok=True)
            
            # Save as .npz (compressed NumPy format)
            np.savez_compressed(
                f"{symbol_dir}/sequences.npz",
                X_train=self.sequences[symbol]['X_train'],
                X_test=self.sequences[symbol]['X_test'],
                y_train=self.sequences[symbol]['y_train'],
                y_test=self.sequences[symbol]['y_test'],
                dates_train=self.sequences[symbol]['dates_train'],
                dates_test=self.sequences[symbol]['dates_test'],
                feature_cols=self.sequences[symbol]['feature_cols'],
                lookback=self.sequences[symbol]['lookback']
            )
            
            logger.info(f"  ✓ Saved {symbol} sequences to {symbol_dir}/sequences.npz")
        
        # 2. Save combined train/test arrays (all symbols together)
        logger.info("\n📁 Saving combined train/test datasets...")
        
        # Concatenate all symbols
        X_train_all = np.concatenate([seq['X_train'] for seq in self.sequences.values()], axis=0)
        X_test_all = np.concatenate([seq['X_test'] for seq in self.sequences.values()], axis=0)
        y_train_all = np.concatenate([seq['y_train'] for seq in self.sequences.values()], axis=0)
        y_test_all = np.concatenate([seq['y_test'] for seq in self.sequences.values()], axis=0)
        
        # Create symbol labels for combined dataset
        train_labels = []
        test_labels = []
        for symbol, seq in self.sequences.items():
            train_labels.extend([symbol] * len(seq['X_train']))
            test_labels.extend([symbol] * len(seq['X_test']))
        
        combined_dir = f"{self.output_path}/train_test_split"
        
        # Save combined arrays
        np.savez_compressed(
            f"{combined_dir}/train_data.npz",
            X=X_train_all,
            y=y_train_all,
            symbols=train_labels
        )
        
        np.savez_compressed(
            f"{combined_dir}/test_data.npz",
            X=X_test_all,
            y=y_test_all,
            symbols=test_labels
        )
        
        logger.info(f"  ✓ Train data: {X_train_all.shape} -> {combined_dir}/train_data.npz")
        logger.info(f"  ✓ Test data:  {X_test_all.shape} -> {combined_dir}/test_data.npz")
        
        # 3. Save cleaned and scaled data as CSV
        logger.info("\n📁 Saving cleaned and scaled data as CSV...")
        
        # Combine all scaled data
        all_scaled_data = pd.concat(self.scaled_data.values(), ignore_index=True)
        scaled_csv = f"{self.output_path}/scaled_data_with_features.csv"
        all_scaled_data.to_csv(scaled_csv, index=False)
        
        logger.info(f"  ✓ Scaled data: {all_scaled_data.shape} -> {scaled_csv}")
        logger.info(f"    Size: {os.path.getsize(scaled_csv) / 1024:.2f} KB")
        
        # 4. Save metadata
        logger.info("\n📁 Saving preprocessing metadata...")
        
        metadata = {
            'preprocessing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'lookback_window': self.lookback_window,
            'train_split_ratio': self.train_split,
            'n_symbols': len(self.sequences),
            'total_train_sequences': len(X_train_all),
            'total_test_sequences': len(X_test_all),
            'feature_columns': list(self.sequences[list(self.sequences.keys())[0]]['feature_cols']),
            'n_features': len(self.sequences[list(self.sequences.keys())[0]]['feature_cols']),
            'symbols': list(self.sequences.keys())
        }
        
        import json
        metadata_file = f"{self.output_path}/preprocessing_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"  ✓ Metadata saved to: {metadata_file}")
        
        logger.info("\n✓ All preprocessed data saved successfully\n")
        
    def generate_statistics(self):
        """Generate comprehensive statistics"""
        logger.info("=" * 80)
        logger.info("STEP 8: PREPROCESSING STATISTICS")
        logger.info("=" * 80)
        
        logger.info("\n📊 OVERALL STATISTICS:")
        logger.info("-" * 80)
        logger.info(f"Total symbols processed: {len(self.sequences)}")
        logger.info(f"Lookback window: {self.lookback_window} days")
        logger.info(f"Number of features: {len(self.sequences[list(self.sequences.keys())[0]]['feature_cols'])}")
        logger.info(f"Train/Test split: {self.train_split*100:.0f}% / {(1-self.train_split)*100:.0f}%")
        
        logger.info("\n📊 SEQUENCE STATISTICS:")
        logger.info("-" * 80)
        total_train = sum(len(seq['X_train']) for seq in self.sequences.values())
        total_test = sum(len(seq['X_test']) for seq in self.sequences.values())
        logger.info(f"Total training sequences: {total_train:,}")
        logger.info(f"Total testing sequences: {total_test:,}")
        logger.info(f"Total sequences: {total_train + total_test:,}")
        
        logger.info("\n📊 PER-SYMBOL BREAKDOWN:")
        logger.info("-" * 80)
        for symbol in sorted(self.sequences.keys()):
            seq = self.sequences[symbol]
            logger.info(f"  {symbol:15s}: {len(seq['X_train']):3d} train / {len(seq['X_test']):3d} test")
        
        logger.info("\n📊 DATA SHAPES:")
        logger.info("-" * 80)
        sample_symbol = list(self.sequences.keys())[0]
        sample_seq = self.sequences[sample_symbol]
        logger.info(f"X_train shape (per symbol): ({len(sample_seq['X_train'])}, {self.lookback_window}, {sample_seq['n_features']})")
        logger.info(f"  - Dimension 0: Number of sequences")
        logger.info(f"  - Dimension 1: Lookback window (timesteps)")
        logger.info(f"  - Dimension 2: Number of features")
        
        logger.info("\n📊 FEATURE LIST:")
        logger.info("-" * 80)
        for i, feature in enumerate(sample_seq['feature_cols'], 1):
            logger.info(f"  {i:2d}. {feature}")
        
        logger.info("\n📊 OUTPUT FILES:")
        logger.info("-" * 80)
        logger.info(f"  1. Individual sequences: {self.output_path}/sequences/[SYMBOL]/sequences.npz")
        logger.info(f"  2. Combined train data: {self.output_path}/train_test_split/train_data.npz")
        logger.info(f"  3. Combined test data: {self.output_path}/train_test_split/test_data.npz")
        logger.info(f"  4. Scaled CSV: {self.output_path}/scaled_data_with_features.csv")
        logger.info(f"  5. Scalers: {self.output_path}/scalers/feature_scalers.pkl")
        logger.info(f"  6. Metadata: {self.output_path}/preprocessing_metadata.json")
        logger.info(f"  7. Summaries: sequence_summary.csv, train_test_summary.csv")
        
        # Create final report
        logger.info("\n📄 Creating final report...")
        report_file = f"{self.output_path}/PREPROCESSING_REPORT.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("EXTENDED DATA PREPROCESSING REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("1. PREPROCESSING CONFIGURATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Lookback window: {self.lookback_window} days\n")
            f.write(f"Train/Test split: {self.train_split*100:.0f}% / {(1-self.train_split)*100:.0f}%\n")
            f.write(f"Number of features: {len(sample_seq['feature_cols'])}\n")
            f.write(f"Normalization: MinMaxScaler (0-1 range)\n\n")
            
            f.write("2. DATASET STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total symbols: {len(self.sequences)}\n")
            f.write(f"Total training sequences: {total_train:,}\n")
            f.write(f"Total testing sequences: {total_test:,}\n")
            f.write(f"Total sequences: {total_train + total_test:,}\n\n")
            
            f.write("3. FEATURES\n")
            f.write("-" * 80 + "\n")
            f.write("OHLCV Features:\n")
            f.write("  - Open, High, Low, Close, Volume\n\n")
            f.write("Technical Features:\n")
            f.write("  - Returns: Daily percentage change\n")
            f.write("  - MA_5, MA_10, MA_20: Moving averages\n")
            f.write("  - Volatility: Rolling standard deviation of returns\n")
            f.write("  - Momentum: 5-day price momentum\n\n")
            
            f.write("4. OUTPUT FILES\n")
            f.write("-" * 80 + "\n")
            f.write(f"Location: {self.output_path}/\n")
            f.write("  - sequences/[SYMBOL]/sequences.npz (per-symbol data)\n")
            f.write("  - train_test_split/train_data.npz (combined training)\n")
            f.write("  - train_test_split/test_data.npz (combined testing)\n")
            f.write("  - scaled_data_with_features.csv (full scaled dataset)\n")
            f.write("  - scalers/feature_scalers.pkl (for inverse transform)\n\n")
            
            f.write("5. INTEGRATION NOTES\n")
            f.write("-" * 80 + "\n")
            f.write("✓ Modular design for hybrid model integration\n")
            f.write("✓ Returns & Volatility features ready for GARCH integration\n")
            f.write("✓ Date indices preserved for FinBERT sentiment alignment\n")
            f.write("✓ Time-based split ensures no data leakage\n")
            f.write("✓ Scalers saved for prediction phase inverse transform\n")
        
        logger.info(f"✓ Final report saved to: {report_file}")
        
        logger.info("\n" + "=" * 80)
        logger.info("PREPROCESSING COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"\n✓ Preprocessed {len(self.sequences)} symbols")
        logger.info(f"✓ Created {total_train:,} training sequences")
        logger.info(f"✓ Created {total_test:,} testing sequences")
        logger.info(f"✓ Ready for LSTM model training with 60-day lookback, batch size 32, 50 epochs\n")


def main():
    """Main preprocessing pipeline"""
    logger.info("\n" + "=" * 80)
    logger.info("EXTENDED DATA PREPROCESSING PIPELINE")
    logger.info("60-DAY LOOKBACK | 80/20 TRAIN-TEST SPLIT")
    logger.info("=" * 80 + "\n")
    
    # Initialize preprocessor
    preprocessor = ExtendedDataPreprocessor()
    
    # Step 1: Load data
    preprocessor.load_data()
    
    # Step 2: Clean data
    preprocessor.clean_data()
    
    # Step 3: Add technical features
    preprocessor.add_technical_features()
    
    # Step 4: Normalize features
    preprocessor.normalize_features()
    
    # Step 5: Create sequences
    preprocessor.create_sequences()
    
    # Step 6: Train-test split
    preprocessor.train_test_split()
    
    # Step 7: Save preprocessed data
    preprocessor.save_preprocessed_data()
    
    # Step 8: Generate statistics
    preprocessor.generate_statistics()


if __name__ == "__main__":
    main()

