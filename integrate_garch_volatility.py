"""
Prompt 10: GARCH Volatility Integration
Integrate GARCH(1,1) volatility modeling into the hybrid dataset.
"""

import pandas as pd
import numpy as np
from arch import arch_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import pickle
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class GARCHVolatilityIntegrator:
    def __init__(self, input_file='data/extended/processed/hybrid_data_with_sentiment.csv',
                 output_dir='data/extended/processed'):
        self.input_file = input_file
        self.output_dir = output_dir
        self.data = None
        self.volatility_data = {}
        self.garch_models = {}
        self.volatility_stats = []
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def load_data(self):
        """Load hybrid data with sentiment"""
        logger.info("=" * 80)
        logger.info("STEP 1: LOADING HYBRID DATA WITH SENTIMENT")
        logger.info("=" * 80)
        
        logger.info(f"Loading: {self.input_file}")
        self.data = pd.read_csv(self.input_file)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        
        logger.info(f"✓ Loaded {len(self.data)} records")
        logger.info(f"  - Shape: {self.data.shape}")
        logger.info(f"  - Symbols: {self.data['Symbol'].nunique()}")
        logger.info(f"  - Date range: {self.data['Date'].min()} to {self.data['Date'].max()}")
        logger.info(f"  - Current features: {self.data.shape[1] - 4}")  # Exclude Date, Symbol, Source, text_count
        logger.info("")
        
    def fit_garch_models(self):
        """Fit GARCH(1,1) models for each symbol"""
        logger.info("=" * 80)
        logger.info("STEP 2: FITTING GARCH(1,1) MODELS")
        logger.info("=" * 80)
        
        logger.info("Fitting GARCH(1,1) model for each symbol...")
        logger.info("Model specification: GARCH(p=1, q=1)")
        logger.info("")
        
        symbols = sorted(self.data['Symbol'].unique())
        
        for idx, symbol in enumerate(symbols, 1):
            symbol_data = self.data[self.data['Symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('Date')
            
            # Extract returns (already computed in preprocessing)
            returns = symbol_data['Returns'].values * 100  # Scale to percentage
            
            # Remove any NaN or inf values
            returns = returns[~np.isnan(returns)]
            returns = returns[~np.isinf(returns)]
            
            if len(returns) < 30:
                logger.warning(f"  ✗ {symbol}: Insufficient data ({len(returns)} points)")
                continue
            
            try:
                # Fit GARCH(1,1) model
                model = arch_model(returns, vol='Garch', p=1, q=1, rescale=False)
                
                # Suppress optimization output
                result = model.fit(disp='off', show_warning=False)
                
                # Extract conditional volatility
                conditional_vol = result.conditional_volatility
                
                # Store results
                self.garch_models[symbol] = result
                
                # Create volatility dataframe aligned with dates
                vol_df = pd.DataFrame({
                    'Date': symbol_data['Date'].iloc[:len(conditional_vol)],
                    'garch_volatility_raw': conditional_vol / 100  # Convert back to decimal
                })
                
                self.volatility_data[symbol] = vol_df
                
                # Store statistics
                self.volatility_stats.append({
                    'Symbol': symbol,
                    'Mean_Volatility': conditional_vol.mean(),
                    'Std_Volatility': conditional_vol.std(),
                    'Min_Volatility': conditional_vol.min(),
                    'Max_Volatility': conditional_vol.max(),
                    'AIC': result.aic,
                    'BIC': result.bic,
                    'Log_Likelihood': result.loglikelihood
                })
                
                if idx % 5 == 0 or idx == len(symbols):
                    logger.info(f"  Progress: {idx}/{len(symbols)} symbols processed")
                
            except Exception as e:
                logger.warning(f"  ✗ {symbol}: GARCH fitting failed - {str(e)[:50]}")
                continue
        
        logger.info(f"\n✓ Successfully fitted GARCH models for {len(self.garch_models)} symbols")
        logger.info("")
        
    def merge_volatility_with_data(self):
        """Merge GARCH volatility with main dataset"""
        logger.info("=" * 80)
        logger.info("STEP 3: MERGING GARCH VOLATILITY WITH DATASET")
        logger.info("=" * 80)
        
        logger.info("Merging conditional volatility for each symbol...")
        
        # Create a list to store merged data
        merged_parts = []
        
        for symbol in self.data['Symbol'].unique():
            symbol_data = self.data[self.data['Symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('Date')
            
            if symbol in self.volatility_data:
                # Merge with volatility data
                vol_df = self.volatility_data[symbol]
                merged = symbol_data.merge(vol_df, on='Date', how='left')
                merged_parts.append(merged)
            else:
                # No volatility data, fill with zeros
                symbol_data['garch_volatility_raw'] = 0.0
                merged_parts.append(symbol_data)
        
        # Combine all symbols
        self.data = pd.concat(merged_parts, ignore_index=True)
        
        # Fill any remaining NaN with 0
        self.data['garch_volatility_raw'].fillna(0, inplace=True)
        
        logger.info(f"✓ Merge completed")
        logger.info(f"  - Records with volatility: {(self.data['garch_volatility_raw'] > 0).sum()}")
        logger.info(f"  - Records without volatility: {(self.data['garch_volatility_raw'] == 0).sum()}")
        logger.info("")
        
    def normalize_volatility(self):
        """Normalize GARCH volatility using MinMaxScaler"""
        logger.info("=" * 80)
        logger.info("STEP 4: NORMALIZING GARCH VOLATILITY")
        logger.info("=" * 80)
        
        logger.info("Applying MinMaxScaler to garch_volatility...")
        
        # Get non-zero volatility values for fitting scaler
        non_zero_vol = self.data[self.data['garch_volatility_raw'] > 0]['garch_volatility_raw'].values.reshape(-1, 1)
        
        if len(non_zero_vol) > 0:
            # Fit scaler on non-zero values
            self.scaler.fit(non_zero_vol)
            
            # Transform all values
            all_vol = self.data['garch_volatility_raw'].values.reshape(-1, 1)
            normalized_vol = self.scaler.transform(all_vol)
            
            self.data['garch_volatility'] = normalized_vol
            
            logger.info(f"✓ Normalization completed")
            logger.info(f"  - Original range: [{self.data['garch_volatility_raw'].min():.6f}, {self.data['garch_volatility_raw'].max():.6f}]")
            logger.info(f"  - Normalized range: [{self.data['garch_volatility'].min():.6f}, {self.data['garch_volatility'].max():.6f}]")
            
            # Save scaler
            scaler_file = f'{self.output_dir}/scalers/garch_volatility_scaler.pkl'
            os.makedirs(os.path.dirname(scaler_file), exist_ok=True)
            with open(scaler_file, 'wb') as f:
                pickle.dump(self.scaler, f)
            logger.info(f"  - Scaler saved: {scaler_file}")
        else:
            logger.warning("⚠ No non-zero volatility values found, using raw values")
            self.data['garch_volatility'] = self.data['garch_volatility_raw']
        
        logger.info("")
        
    def save_updated_dataset(self):
        """Save dataset with GARCH volatility"""
        logger.info("=" * 80)
        logger.info("STEP 5: SAVING UPDATED DATASET")
        logger.info("=" * 80)
        
        # Drop raw volatility column
        output_data = self.data.drop('garch_volatility_raw', axis=1)
        
        output_file = f'{self.output_dir}/hybrid_data_with_sentiment_volatility.csv'
        output_data.to_csv(output_file, index=False)
        
        logger.info(f"✓ Saved updated dataset: {output_file}")
        logger.info(f"  - Shape: {output_data.shape}")
        logger.info(f"  - Size: {os.path.getsize(output_file) / 1024:.2f} KB")
        logger.info(f"  - New feature: garch_volatility (13th feature)")
        logger.info("")
        
    def update_metadata(self):
        """Update preprocessing metadata"""
        logger.info("=" * 80)
        logger.info("STEP 6: UPDATING METADATA")
        logger.info("=" * 80)
        
        # Load existing metadata
        metadata_file = f'{self.output_dir}/preprocessing_metadata_with_sentiment.json'
        
        import json
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Update metadata
        metadata['garch_integration_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        metadata['n_features'] = 13
        metadata['feature_columns'].append('garch_volatility')
        metadata['garch_models_fitted'] = len(self.garch_models)
        metadata['garch_specification'] = 'GARCH(1,1)'
        
        # Save updated metadata
        new_metadata_file = f'{self.output_dir}/preprocessing_metadata_with_sentiment_volatility.json'
        with open(new_metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✓ Metadata updated: {new_metadata_file}")
        logger.info(f"  - Features: 12 → 13 (added garch_volatility)")
        logger.info(f"  - Input shape: (60, 13)")
        logger.info("")
        
    def generate_statistics(self):
        """Generate summary statistics"""
        logger.info("=" * 80)
        logger.info("STEP 7: GENERATING SUMMARY STATISTICS")
        logger.info("=" * 80)
        
        # Convert stats to DataFrame
        stats_df = pd.DataFrame(self.volatility_stats)
        
        if len(stats_df) > 0:
            logger.info("\n📊 GARCH VOLATILITY STATISTICS (Top 10 by Mean)")
            logger.info("-" * 80)
            
            top_10 = stats_df.nlargest(10, 'Mean_Volatility')
            print("\n" + top_10[['Symbol', 'Mean_Volatility', 'Std_Volatility', 'Min_Volatility', 'Max_Volatility']].to_string(index=False))
            
            logger.info(f"\n📊 OVERALL STATISTICS")
            logger.info(f"  - Symbols with GARCH models: {len(stats_df)}")
            logger.info(f"  - Average volatility (all symbols): {stats_df['Mean_Volatility'].mean():.6f}")
            logger.info(f"  - Highest volatility symbol: {stats_df.loc[stats_df['Mean_Volatility'].idxmax(), 'Symbol']}")
            logger.info(f"  - Lowest volatility symbol: {stats_df.loc[stats_df['Mean_Volatility'].idxmin(), 'Symbol']}")
            
            # Save statistics
            stats_file = f'{self.output_dir}/garch_volatility_statistics.csv'
            stats_df.to_csv(stats_file, index=False)
            logger.info(f"\n✓ Statistics saved: {stats_file}")
        
        logger.info("")
        
    def create_visualizations(self):
        """Create volatility visualizations"""
        logger.info("=" * 80)
        logger.info("STEP 8: CREATING VISUALIZATIONS")
        logger.info("=" * 80)
        
        output_dir = 'sample_run_output/output/plots/garch_volatility'
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Plot volatility vs returns for RELIANCE
        if 'RELIANCE' in self.data['Symbol'].values:
            logger.info("\n📊 Plotting RELIANCE: Volatility vs Returns")
            
            reliance_data = self.data[self.data['Symbol'] == 'RELIANCE'].copy()
            reliance_data = reliance_data.sort_values('Date')
            
            fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
            
            # Plot 1: Close price
            axes[0].plot(reliance_data['Date'], reliance_data['Close'], 
                        color='steelblue', linewidth=2, label='Close Price')
            axes[0].set_ylabel('Close Price (Normalized)', fontsize=12, fontweight='bold')
            axes[0].set_title('RELIANCE: Price, Returns, and GARCH Volatility', 
                            fontsize=14, fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            axes[0].legend(loc='upper left')
            
            # Plot 2: Returns
            colors_returns = ['red' if x < 0 else 'green' for x in reliance_data['Returns']]
            axes[1].bar(reliance_data['Date'], reliance_data['Returns'], 
                       color=colors_returns, alpha=0.6, label='Returns')
            axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
            axes[1].set_ylabel('Returns', fontsize=12, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            axes[1].legend(loc='upper left')
            
            # Plot 3: GARCH Volatility
            axes[2].plot(reliance_data['Date'], reliance_data['garch_volatility'], 
                        color='orange', linewidth=2, label='GARCH Volatility')
            axes[2].fill_between(reliance_data['Date'], 0, reliance_data['garch_volatility'], 
                                alpha=0.3, color='orange')
            axes[2].set_xlabel('Date', fontsize=12, fontweight='bold')
            axes[2].set_ylabel('Conditional Volatility', fontsize=12, fontweight='bold')
            axes[2].grid(True, alpha=0.3)
            axes[2].legend(loc='upper left')
            
            plt.tight_layout()
            
            plot_file = f'{output_dir}/reliance_volatility_analysis.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            logger.info(f"  ✓ Saved: {plot_file}")
            plt.close()
        
        # 2. Top 5 symbols by volatility
        logger.info("\n📊 Plotting Top 5 Symbols by Volatility")
        
        if len(self.volatility_stats) >= 5:
            stats_df = pd.DataFrame(self.volatility_stats)
            top_5 = stats_df.nlargest(5, 'Mean_Volatility')
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            bars = ax.bar(top_5['Symbol'], top_5['Mean_Volatility'], 
                         color='coral', alpha=0.7, edgecolor='black', linewidth=1.5)
            
            # Add error bars for std
            ax.errorbar(top_5['Symbol'], top_5['Mean_Volatility'], 
                       yerr=top_5['Std_Volatility'], fmt='none', 
                       color='black', capsize=5, capthick=2)
            
            ax.set_xlabel('Symbol', fontsize=12, fontweight='bold')
            ax.set_ylabel('Mean GARCH Volatility (%)', fontsize=12, fontweight='bold')
            ax.set_title('Top 5 Symbols by Mean GARCH Volatility', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            
            plot_file = f'{output_dir}/top_5_volatility_symbols.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            logger.info(f"  ✓ Saved: {plot_file}")
            plt.close()
        
        # 3. Volatility distribution
        logger.info("\n📊 Plotting Volatility Distribution")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Histogram
        non_zero_vol = self.data[self.data['garch_volatility'] > 0]['garch_volatility']
        axes[0].hist(non_zero_vol, bins=50, color='skyblue', 
                    edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('GARCH Volatility (Normalized)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[0].set_title('Distribution of GARCH Volatility', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Box plot by symbol (top 10)
        if len(self.volatility_stats) >= 10:
            stats_df = pd.DataFrame(self.volatility_stats)
            top_10_symbols = stats_df.nlargest(10, 'Mean_Volatility')['Symbol'].tolist()
            
            vol_by_symbol = []
            labels = []
            for symbol in top_10_symbols:
                symbol_vol = self.data[self.data['Symbol'] == symbol]['garch_volatility']
                symbol_vol = symbol_vol[symbol_vol > 0]
                if len(symbol_vol) > 0:
                    vol_by_symbol.append(symbol_vol.values)
                    labels.append(symbol)
            
            if vol_by_symbol:
                bp = axes[1].boxplot(vol_by_symbol, labels=labels, patch_artist=True)
                
                for patch in bp['boxes']:
                    patch.set_facecolor('lightgreen')
                    patch.set_alpha(0.7)
                
                axes[1].set_xlabel('Symbol', fontsize=12, fontweight='bold')
                axes[1].set_ylabel('GARCH Volatility (Normalized)', fontsize=12, fontweight='bold')
                axes[1].set_title('Volatility Distribution by Symbol (Top 10)', 
                                fontsize=14, fontweight='bold')
                axes[1].grid(True, alpha=0.3, axis='y')
                axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        plot_file = f'{output_dir}/volatility_distribution.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        logger.info(f"  ✓ Saved: {plot_file}")
        plt.close()
        
        logger.info("")
        
    def generate_report(self):
        """Generate comprehensive GARCH integration report"""
        logger.info("=" * 80)
        logger.info("STEP 9: GENERATING INTEGRATION REPORT")
        logger.info("=" * 80)
        
        report_file = 'GARCH_VOLATILITY_INTEGRATION_REPORT.md'
        
        stats_df = pd.DataFrame(self.volatility_stats)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# GARCH Volatility Integration Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Model:** GARCH(1,1)\n")
            f.write(f"**Library:** arch (Python)\n\n")
            f.write("---\n\n")
            
            f.write("## ✅ INTEGRATION SUMMARY\n\n")
            f.write("### Status: **COMPLETED SUCCESSFULLY** ✓\n\n")
            f.write("GARCH(1,1) conditional volatility has been successfully integrated into the hybrid dataset.\n\n")
            
            f.write("---\n\n")
            f.write("## 📊 GARCH MODEL STATISTICS\n\n")
            f.write(f"- **Symbols processed:** {len(self.garch_models)}\n")
            f.write(f"- **Model specification:** GARCH(1,1)\n")
            f.write(f"- **Volatility measure:** Conditional volatility (σ_t)\n")
            f.write(f"- **Normalization:** MinMaxScaler (0-1 range)\n\n")
            
            if len(stats_df) > 0:
                f.write("### Top 10 Symbols by Mean Volatility\n\n")
                f.write("| Symbol | Mean Vol | Std Dev | Min | Max | AIC | BIC |\n")
                f.write("|--------|----------|---------|-----|-----|-----|-----|\n")
                
                top_10 = stats_df.nlargest(10, 'Mean_Volatility')
                for _, row in top_10.iterrows():
                    f.write(f"| {row['Symbol']} | {row['Mean_Volatility']:.6f} | {row['Std_Volatility']:.6f} | ")
                    f.write(f"{row['Min_Volatility']:.6f} | {row['Max_Volatility']:.6f} | ")
                    f.write(f"{row['AIC']:.2f} | {row['BIC']:.2f} |\n")
                
                f.write(f"\n**Average Volatility (All Symbols):** {stats_df['Mean_Volatility'].mean():.6f}\n")
                f.write(f"**Highest Volatility:** {stats_df['Mean_Volatility'].max():.6f} ")
                f.write(f"({stats_df.loc[stats_df['Mean_Volatility'].idxmax(), 'Symbol']})\n")
                f.write(f"**Lowest Volatility:** {stats_df['Mean_Volatility'].min():.6f} ")
                f.write(f"({stats_df.loc[stats_df['Mean_Volatility'].idxmin(), 'Symbol']})\n\n")
            
            f.write("---\n\n")
            f.write("## 🔗 DATASET INTEGRATION\n\n")
            f.write(f"- **Total records:** {len(self.data):,}\n")
            f.write(f"- **Records with volatility:** {(self.data['garch_volatility'] > 0).sum():,}\n")
            f.write(f"- **Records without (filled with 0):** {(self.data['garch_volatility'] == 0).sum():,}\n")
            f.write(f"- **Dataset shape:** {self.data.shape}\n\n")
            
            f.write("---\n\n")
            f.write("## 🎯 MODEL INPUT UPDATE\n\n")
            f.write("### Updated Model Input Shape\n\n")
            f.write("**Previous:** (60, 12)\n")
            f.write("- 60 timesteps (60-day lookback)\n")
            f.write("- 12 features (OHLCV + technical + sentiment)\n\n")
            f.write("**Current:** (60, 13) ✅\n")
            f.write("- 60 timesteps (60-day lookback)\n")
            f.write("- **13 features** (OHLCV + technical + sentiment + **GARCH volatility**)\n\n")
            
            f.write("### Complete Feature List\n\n")
            features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                       'Returns', 'MA_5', 'MA_10', 'MA_20', 'Volatility', 'Momentum',
                       'sentiment_score', 'garch_volatility']
            for i, feat in enumerate(features, 1):
                marker = " ✨ **NEW**" if feat == 'garch_volatility' else ""
                f.write(f"{i}. {feat}{marker}\n")
            
            f.write("\n---\n\n")
            f.write("## 📁 OUTPUT FILES\n\n")
            f.write(f"1. `{self.output_dir}/hybrid_data_with_sentiment_volatility.csv`\n")
            f.write(f"2. `{self.output_dir}/garch_volatility_statistics.csv`\n")
            f.write(f"3. `{self.output_dir}/scalers/garch_volatility_scaler.pkl`\n")
            f.write(f"4. `{self.output_dir}/preprocessing_metadata_with_sentiment_volatility.json`\n")
            f.write(f"5. `sample_run_output/output/plots/garch_volatility/*.png`\n\n")
            
            f.write("---\n\n")
            f.write("## 📊 VISUALIZATIONS\n\n")
            f.write("1. **RELIANCE Volatility Analysis** - Price, Returns, and GARCH Volatility\n")
            f.write("2. **Top 5 Symbols by Volatility** - Bar chart with error bars\n")
            f.write("3. **Volatility Distribution** - Histogram and box plots\n\n")
            
            f.write("---\n\n")
            f.write("## 🚀 NEXT STEPS\n\n")
            f.write("1. **Update Sequence Files** with 13 features (60, 13)\n")
            f.write("2. **Retrain LSTM Model** with new input shape\n")
            f.write("3. **Compare Performance** with previous models\n")
            f.write("4. **Evaluate Risk-Aware Predictions** using volatility information\n\n")
            
            f.write("---\n\n")
            f.write("## 🎓 GARCH(1,1) MODEL INTERPRETATION\n\n")
            f.write("**Model Equation:**\n")
            f.write("```\n")
            f.write("σ_t² = ω + α·ε_{t-1}² + β·σ_{t-1}²\n")
            f.write("```\n\n")
            f.write("Where:\n")
            f.write("- σ_t² = Conditional variance at time t\n")
            f.write("- ω = Constant term\n")
            f.write("- α = ARCH parameter (impact of past shocks)\n")
            f.write("- β = GARCH parameter (impact of past volatility)\n")
            f.write("- ε_t = Return innovation (shock)\n\n")
            
            f.write("**Key Properties:**\n")
            f.write("- Captures volatility clustering (high/low volatility periods)\n")
            f.write("- Models time-varying risk\n")
            f.write("- Useful for portfolio management and risk assessment\n\n")
            
            f.write("---\n\n")
            f.write(f"**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("**Status:** ✅ READY FOR FINAL MODEL TRAINING\n")
        
        logger.info(f"✓ Report generated: {report_file}")
        logger.info("")
        
    def create_updated_sequences(self):
        """Create sequences with 13 features"""
        logger.info("=" * 80)
        logger.info("STEP 10: CREATING UPDATED SEQUENCES (60, 13)")
        logger.info("=" * 80)
        
        logger.info("Creating sequences with garch_volatility as 13th feature...")
        
        # Feature columns (13 features)
        feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 
                       'Returns', 'MA_5', 'MA_10', 'MA_20', 'Volatility', 'Momentum',
                       'sentiment_score', 'garch_volatility']
        
        lookback = 60
        train_split = 0.8
        sequences = {}
        
        for symbol in sorted(self.data['Symbol'].unique()):
            symbol_data = self.data[self.data['Symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('Date')
            
            if len(symbol_data) <= lookback:
                logger.warning(f"  ✗ {symbol}: Insufficient data ({len(symbol_data)} records)")
                continue
            
            # Extract features
            feature_matrix = symbol_data[feature_cols].values
            dates = symbol_data['Date'].values
            
            # Create sequences
            X, y, date_indices = [], [], []
            
            for i in range(lookback, len(feature_matrix)):
                X.append(feature_matrix[i-lookback:i])
                y.append(feature_matrix[i, 3])  # Close price
                date_indices.append(dates[i])
            
            if len(X) == 0:
                continue
            
            X = np.array(X)
            y = np.array(y)
            
            # Train-test split
            split_idx = int(len(X) * train_split)
            
            sequences[symbol] = {
                'X_train': X[:split_idx],
                'X_test': X[split_idx:],
                'y_train': y[:split_idx],
                'y_test': y[split_idx:],
                'dates_train': date_indices[:split_idx],
                'dates_test': date_indices[split_idx:],
                'feature_cols': feature_cols,
                'n_features': len(feature_cols)
            }
            
            logger.info(f"  ✓ {symbol}: {len(X)} sequences | Shape: {X.shape}")
        
        # Save sequences
        sequences_dir = f'{self.output_dir}/sequences_with_sentiment_volatility'
        os.makedirs(sequences_dir, exist_ok=True)
        
        for symbol, seq_data in sequences.items():
            symbol_dir = f"{sequences_dir}/{symbol}"
            os.makedirs(symbol_dir, exist_ok=True)
            
            np.savez_compressed(
                f"{symbol_dir}/sequences.npz",
                X_train=seq_data['X_train'],
                X_test=seq_data['X_test'],
                y_train=seq_data['y_train'],
                y_test=seq_data['y_test'],
                dates_train=seq_data['dates_train'],
                dates_test=seq_data['dates_test'],
                feature_cols=seq_data['feature_cols']
            )
        
        # Save combined arrays
        combined_dir = f'{self.output_dir}/train_test_split_with_sentiment_volatility'
        os.makedirs(combined_dir, exist_ok=True)
        
        X_train_all = np.concatenate([seq['X_train'] for seq in sequences.values()], axis=0)
        X_test_all = np.concatenate([seq['X_test'] for seq in sequences.values()], axis=0)
        y_train_all = np.concatenate([seq['y_train'] for seq in sequences.values()], axis=0)
        y_test_all = np.concatenate([seq['y_test'] for seq in sequences.values()], axis=0)
        
        train_labels = []
        test_labels = []
        for symbol, seq in sequences.items():
            train_labels.extend([symbol] * len(seq['X_train']))
            test_labels.extend([symbol] * len(seq['X_test']))
        
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
        
        logger.info(f"\n✓ Sequences created successfully")
        logger.info(f"  - Combined train shape: {X_train_all.shape}")
        logger.info(f"  - Combined test shape: {X_test_all.shape}")
        logger.info(f"  - Input shape per sample: (60, 13) ✅")
        logger.info(f"  - Saved to: {sequences_dir}/")
        logger.info("")


def main():
    logger.info("\n" + "=" * 80)
    logger.info("PROMPT 10: GARCH VOLATILITY INTEGRATION")
    logger.info("Integrating GARCH(1,1) Conditional Volatility")
    logger.info("=" * 80 + "\n")
    
    # Initialize integrator
    integrator = GARCHVolatilityIntegrator()
    
    # Run pipeline
    integrator.load_data()
    integrator.fit_garch_models()
    integrator.merge_volatility_with_data()
    integrator.normalize_volatility()
    integrator.save_updated_dataset()
    integrator.update_metadata()
    integrator.generate_statistics()
    integrator.create_visualizations()
    integrator.generate_report()
    integrator.create_updated_sequences()
    
    # Final message
    logger.info("=" * 80)
    logger.info("✅ GARCH Volatility Integration Complete – Hybrid dataset now ready for final model training (60, 13 input shape).")
    logger.info("=" * 80)
    logger.info("")


if __name__ == "__main__":
    main()

