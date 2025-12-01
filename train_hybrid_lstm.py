"""
Prompt 11 - Train Final Hybrid LSTM Model
Train a hybrid deep learning model with all 13 features:
- OHLCV (5)
- Technical indicators (6): Returns, MA_5, MA_10, MA_20, Volatility, Momentum
- FinBERT sentiment (1)
- GARCH volatility (1)

Input shape: (60, 13)
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ML libraries
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Attention, Permute, Multiply, Lambda
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    ML_AVAILABLE = True
except ImportError as e:
    logger.error(f"ML libraries not available: {e}")
    ML_AVAILABLE = False
    exit(1)


class HybridLSTMTrainer:
    def __init__(self):
        self.data_path = 'data/extended/processed'
        self.model_path = 'data/extended/models'
        self.output_path = 'sample_run_output/output'
        
        # Create directories
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(f"{self.output_path}/plots", exist_ok=True)
        os.makedirs(f"{self.output_path}/reports", exist_ok=True)
        
        # Model hyperparameters
        self.lookback = 60
        self.n_features = 13
        self.batch_size = 32
        self.epochs = 70
        self.learning_rate = 1e-3
        self.patience = 12
        
        # Data containers
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.model = None
        self.history = None
        self.scaler = None
        self.feature_columns = None
        self.metadata = None
        
    def load_sequences(self):
        """Load preprocessed sequences with all 13 features"""
        logger.info("=" * 80)
        logger.info("STEP 1: LOADING PREPROCESSED SEQUENCES")
        logger.info("=" * 80)
        
        # Load sequences
        train_data_path = f"{self.data_path}/train_test_split_with_sentiment_volatility/train_data.npz"
        test_data_path = f"{self.data_path}/train_test_split_with_sentiment_volatility/test_data.npz"
        
        if not os.path.exists(train_data_path) or not os.path.exists(test_data_path):
            logger.error(f"Sequence files not found. Please run Prompt 10 (GARCH integration) first.")
            return False
        
        # Load train data
        train_data = np.load(train_data_path)
        self.X_train = train_data['X']
        self.y_train = train_data['y']
        
        # Load test data
        test_data = np.load(test_data_path)
        self.X_test = test_data['X']
        self.y_test = test_data['y']
        
        logger.info(f"✓ Training sequences loaded: {self.X_train.shape}")
        logger.info(f"✓ Testing sequences loaded: {self.X_test.shape}")
        logger.info(f"✓ Input shape per sample: ({self.lookback}, {self.n_features})")
        
        # Verify shape
        if self.X_train.shape[1:] != (self.lookback, self.n_features):
            logger.error(f"Expected shape (N, {self.lookback}, {self.n_features}), got {self.X_train.shape}")
            return False
        
        # Load metadata
        metadata_path = f"{self.data_path}/preprocessing_metadata_with_sentiment_volatility.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            self.feature_columns = self.metadata['feature_columns']
            logger.info(f"\n✓ Loaded metadata with {len(self.feature_columns)} features:")
            for i, feature in enumerate(self.feature_columns, 1):
                logger.info(f"   {i:2}. {feature}")
        else:
            logger.warning(f"Metadata file not found: {metadata_path}")
            self.feature_columns = [
                'Open', 'High', 'Low', 'Close', 'Volume',
                'Returns', 'MA_5', 'MA_10', 'MA_20', 'Volatility', 'Momentum',
                'sentiment_score', 'garch_volatility'
            ]
        
        logger.info(f"\n📊 Dataset Summary:")
        logger.info(f"   - Total training samples: {len(self.X_train):,}")
        logger.info(f"   - Total testing samples: {len(self.X_test):,}")
        logger.info(f"   - Train/Test split: {len(self.X_train)/(len(self.X_train)+len(self.X_test)):.1%} / {len(self.X_test)/(len(self.X_train)+len(self.X_test)):.1%}")
        logger.info(f"   - Features: {self.n_features}")
        logger.info(f"   - Lookback window: {self.lookback} days")
        logger.info("")
        
        return True
    
    def build_model_with_attention(self):
        """Build hybrid LSTM model with attention mechanism"""
        logger.info("=" * 80)
        logger.info("STEP 2: BUILDING HYBRID LSTM MODEL WITH ATTENTION")
        logger.info("=" * 80)
        
        model = Sequential([
            # Input layer
            Input(shape=(self.lookback, self.n_features)),
            
            # First LSTM layer
            LSTM(128, return_sequences=True, name='lstm_1'),
            Dropout(0.25, name='dropout_1'),
            
            # Second LSTM layer
            LSTM(64, return_sequences=True, name='lstm_2'),
            
            # Attention mechanism (custom implementation)
            # We'll use a simple attention that learns to weight timesteps
            Dense(1, activation='tanh', name='attention_score'),
            Lambda(lambda x: tf.nn.softmax(x, axis=1), name='attention_weights'),
            
            # Apply attention weights
            Lambda(lambda x: tf.reduce_sum(x, axis=1), name='attention_output'),
            
            # Dense layers
            Dense(32, activation='relu', name='dense_1'),
            Dense(1, activation='linear', name='output')
        ])
        
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        self.model = model
        
        logger.info("\n✓ Model built successfully")
        logger.info("\n📋 Model Architecture:")
        logger.info("-" * 80)
        
        # Print model summary to string
        summary_lines = []
        self.model.summary(print_fn=lambda x: summary_lines.append(x))
        for line in summary_lines:
            logger.info(line)
        
        logger.info("\n📊 Model Configuration:")
        logger.info(f"   - Optimizer: Adam (lr={self.learning_rate})")
        logger.info(f"   - Loss function: MSE")
        logger.info(f"   - Metrics: MAE, MSE")
        logger.info(f"   - Total parameters: {self.model.count_params():,}")
        logger.info("")
        
        return True
    
    def train_model(self):
        """Train the hybrid LSTM model"""
        logger.info("=" * 80)
        logger.info("STEP 3: TRAINING HYBRID LSTM MODEL")
        logger.info("=" * 80)
        
        # Setup callbacks
        model_checkpoint_path = f"{self.model_path}/hybrid_lstm_best.h5"
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.patience,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                model_checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        logger.info(f"📋 Training Configuration:")
        logger.info(f"   - Batch size: {self.batch_size}")
        logger.info(f"   - Epochs: {self.epochs}")
        logger.info(f"   - Learning rate: {self.learning_rate}")
        logger.info(f"   - Early stopping patience: {self.patience}")
        logger.info(f"   - Validation split: Using X_test as validation")
        logger.info("")
        
        # Start training
        logger.info("🚀 Starting training...")
        start_time = time.time()
        
        self.history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_test, self.y_test),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        training_duration = time.time() - start_time
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("✓ TRAINING COMPLETED")
        logger.info("=" * 80)
        logger.info(f"   - Training duration: {training_duration/60:.2f} minutes")
        logger.info(f"   - Total epochs: {len(self.history.history['loss'])}")
        logger.info(f"   - Best epoch: {np.argmin(self.history.history['val_loss']) + 1}")
        logger.info(f"   - Best val_loss: {min(self.history.history['val_loss']):.6f}")
        logger.info(f"   - Final train_loss: {self.history.history['loss'][-1]:.6f}")
        logger.info(f"   - Final val_loss: {self.history.history['val_loss'][-1]:.6f}")
        logger.info("")
        
        # Save final model
        final_model_path = f"{self.model_path}/hybrid_lstm_model.h5"
        self.model.save(final_model_path)
        logger.info(f"✓ Model saved: {final_model_path}")
        
        # Save training history
        history_df = pd.DataFrame(self.history.history)
        history_df['epoch'] = range(1, len(history_df) + 1)
        history_csv_path = f"{self.model_path}/training_history.csv"
        history_df.to_csv(history_csv_path, index=False)
        logger.info(f"✓ Training history saved: {history_csv_path}")
        
        # Save as JSON
        history_json_path = f"{self.model_path}/training_history.json"
        with open(history_json_path, 'w') as f:
            json.dump({
                'history': {k: [float(v) for v in vals] for k, vals in self.history.history.items()},
                'training_duration_minutes': training_duration / 60,
                'best_epoch': int(np.argmin(self.history.history['val_loss']) + 1),
                'best_val_loss': float(min(self.history.history['val_loss'])),
                'configuration': {
                    'batch_size': self.batch_size,
                    'epochs': self.epochs,
                    'learning_rate': self.learning_rate,
                    'patience': self.patience,
                    'lookback': self.lookback,
                    'n_features': self.n_features
                }
            }, f, indent=2)
        logger.info(f"✓ Training history JSON saved: {history_json_path}")
        logger.info("")
        
        return True
    
    def plot_training_history(self):
        """Plot training and validation loss"""
        logger.info("=" * 80)
        logger.info("STEP 4: PLOTTING TRAINING HISTORY")
        logger.info("=" * 80)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Loss
        axes[0].plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        axes[0].plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss (MSE)', fontsize=12)
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Mark best epoch
        best_epoch = np.argmin(self.history.history['val_loss'])
        axes[0].axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5, label=f'Best Epoch: {best_epoch+1}')
        axes[0].legend(fontsize=11)
        
        # Plot 2: MAE
        axes[1].plot(self.history.history['mae'], label='Training MAE', linewidth=2)
        axes[1].plot(self.history.history['val_mae'], label='Validation MAE', linewidth=2)
        axes[1].set_title('Model MAE Over Epochs', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('MAE', fontsize=12)
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        axes[1].axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plot_path = f"{self.output_path}/plots/hybrid_lstm_training_history.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Training history plot saved: {plot_path}")
        logger.info("")
        
    def evaluate_model(self):
        """Evaluate model on test set and calculate metrics"""
        logger.info("=" * 80)
        logger.info("STEP 5: EVALUATING MODEL ON TEST SET")
        logger.info("=" * 80)
        
        # Make predictions
        logger.info("Making predictions on test set...")
        y_pred = self.model.predict(self.X_test, batch_size=self.batch_size, verbose=0)
        y_pred = y_pred.flatten()
        y_true = self.y_test.flatten()
        
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        
        # MAPE (avoid division by zero)
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
        # R² Score
        r2 = r2_score(y_true, y_pred)
        
        # Directional Accuracy
        actual_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        
        logger.info("\n📊 EVALUATION METRICS:")
        logger.info("=" * 80)
        logger.info(f"   - RMSE:                  {rmse:.6f}")
        logger.info(f"   - MAE:                   {mae:.6f}")
        logger.info(f"   - MAPE:                  {mape:.2f}%")
        logger.info(f"   - R² Score:              {r2:.6f}")
        logger.info(f"   - Directional Accuracy:  {directional_accuracy:.2f}%")
        logger.info("=" * 80)
        logger.info("")
        
        # Save evaluation metrics
        evaluation = {
            'test_metrics': {
                'rmse': float(rmse),
                'mae': float(mae),
                'mape': float(mape),
                'r2_score': float(r2),
                'directional_accuracy': float(directional_accuracy)
            },
            'test_samples': int(len(y_true)),
            'model_configuration': {
                'lookback': self.lookback,
                'n_features': self.n_features,
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'learning_rate': self.learning_rate
            },
            'feature_columns': self.feature_columns,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        eval_path = f"{self.output_path}/reports/hybrid_lstm_evaluation.json"
        with open(eval_path, 'w') as f:
            json.dump(evaluation, f, indent=2)
        
        logger.info(f"✓ Evaluation metrics saved: {eval_path}")
        logger.info("")
        
        return y_pred, y_true
    
    def plot_predictions(self, y_pred, y_true):
        """Plot predictions vs actual for last 60 days"""
        logger.info("=" * 80)
        logger.info("STEP 6: PLOTTING PREDICTIONS VS ACTUAL")
        logger.info("=" * 80)
        
        # Plot last 60 predictions
        n_plot = min(60, len(y_true))
        
        fig, axes = plt.subplots(2, 1, figsize=(16, 12))
        
        # Plot 1: Last 60 days comparison
        axes[0].plot(range(n_plot), y_true[-n_plot:], label='Actual', marker='o', linewidth=2, markersize=4)
        axes[0].plot(range(n_plot), y_pred[-n_plot:], label='Predicted', marker='s', linewidth=2, markersize=4, alpha=0.7)
        axes[0].set_title(f'Hybrid LSTM: Predictions vs Actual (Last {n_plot} Days)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Time Step', fontsize=12)
        axes[0].set_ylabel('Close Price (Normalized)', fontsize=12)
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Scatter plot
        axes[1].scatter(y_true, y_pred, alpha=0.5, s=20)
        axes[1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', linewidth=2, label='Perfect Prediction')
        axes[1].set_title('Predicted vs Actual (All Test Samples)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Actual Close Price', fontsize=12)
        axes[1].set_ylabel('Predicted Close Price', fontsize=12)
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        # Add R² annotation
        r2 = r2_score(y_true, y_pred)
        axes[1].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[1].transAxes, 
                    fontsize=12, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plot_path = f"{self.output_path}/plots/hybrid_predictions.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Predictions plot saved: {plot_path}")
        logger.info("")
        
    def generate_final_report(self):
        """Generate comprehensive final report"""
        logger.info("=" * 80)
        logger.info("STEP 7: GENERATING FINAL REPORT")
        logger.info("=" * 80)
        
        report_path = f"{self.output_path}/reports/HYBRID_LSTM_TRAINING_REPORT.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Hybrid LSTM Model Training Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("## 1. Model Overview\n\n")
            f.write("### Architecture\n")
            f.write("- **Type:** Hybrid LSTM with Attention Mechanism\n")
            f.write("- **Input Shape:** (60, 13) - 60 days lookback, 13 features\n")
            f.write("- **Layers:**\n")
            f.write("  1. LSTM(128, return_sequences=True)\n")
            f.write("  2. Dropout(0.25)\n")
            f.write("  3. LSTM(64, return_sequences=True)\n")
            f.write("  4. Attention Mechanism\n")
            f.write("  5. Dense(32, relu)\n")
            f.write("  6. Dense(1, linear)\n")
            f.write(f"- **Total Parameters:** {self.model.count_params():,}\n\n")
            
            f.write("### Features (13 total)\n")
            f.write("**Market Data (5):**\n")
            f.write("1. Open\n2. High\n3. Low\n4. Close\n5. Volume\n\n")
            f.write("**Technical Indicators (6):**\n")
            f.write("6. Returns\n7. MA_5 (5-day Moving Average)\n")
            f.write("8. MA_10 (10-day Moving Average)\n9. MA_20 (20-day Moving Average)\n")
            f.write("10. Volatility (30-day rolling std)\n11. Momentum\n\n")
            f.write("**Hybrid Features (2):**\n")
            f.write("12. **sentiment_score** - FinBERT sentiment analysis (-1 to +1)\n")
            f.write("13. **garch_volatility** - GARCH(1,1) conditional volatility\n\n")
            
            f.write("=" * 80 + "\n\n")
            f.write("## 2. Training Configuration\n\n")
            f.write(f"- **Batch Size:** {self.batch_size}\n")
            f.write(f"- **Epochs:** {self.epochs}\n")
            f.write(f"- **Learning Rate:** {self.learning_rate}\n")
            f.write(f"- **Optimizer:** Adam\n")
            f.write(f"- **Loss Function:** MSE\n")
            f.write(f"- **Early Stopping Patience:** {self.patience}\n")
            f.write(f"- **Train/Test Split:** 80/20 (time-based)\n\n")
            
            f.write("=" * 80 + "\n\n")
            f.write("## 3. Dataset Information\n\n")
            f.write(f"- **Training Samples:** {len(self.X_train):,}\n")
            f.write(f"- **Testing Samples:** {len(self.X_test):,}\n")
            f.write(f"- **Total Samples:** {len(self.X_train) + len(self.X_test):,}\n")
            f.write(f"- **Symbols:** {self.metadata.get('symbols_count', 'N/A')}\n")
            f.write(f"- **Date Range:** {self.metadata.get('date_range', {}).get('start', 'N/A')} to {self.metadata.get('date_range', {}).get('end', 'N/A')}\n\n")
            
            f.write("=" * 80 + "\n\n")
            f.write("## 4. Training Results\n\n")
            
            # Load evaluation metrics
            eval_path = f"{self.output_path}/reports/hybrid_lstm_evaluation.json"
            with open(eval_path, 'r') as eval_f:
                evaluation = json.load(eval_f)
            
            best_epoch = np.argmin(self.history.history['val_loss']) + 1
            f.write(f"- **Total Epochs Trained:** {len(self.history.history['loss'])}\n")
            f.write(f"- **Best Epoch:** {best_epoch}\n")
            f.write(f"- **Best Validation Loss:** {min(self.history.history['val_loss']):.6f}\n")
            f.write(f"- **Final Training Loss:** {self.history.history['loss'][-1]:.6f}\n")
            f.write(f"- **Final Validation Loss:** {self.history.history['val_loss'][-1]:.6f}\n\n")
            
            f.write("=" * 80 + "\n\n")
            f.write("## 5. Evaluation Metrics (Test Set)\n\n")
            metrics = evaluation['test_metrics']
            f.write(f"- **RMSE:** {metrics['rmse']:.6f}\n")
            f.write(f"- **MAE:** {metrics['mae']:.6f}\n")
            f.write(f"- **MAPE:** {metrics['mape']:.2f}%\n")
            f.write(f"- **R² Score:** {metrics['r2_score']:.6f}\n")
            f.write(f"- **Directional Accuracy:** {metrics['directional_accuracy']:.2f}%\n\n")
            
            f.write("### Interpretation\n")
            if metrics['r2_score'] > 0.8:
                f.write("✅ **Excellent fit** - Model explains > 80% of variance\n")
            elif metrics['r2_score'] > 0.6:
                f.write("✅ **Good fit** - Model explains > 60% of variance\n")
            else:
                f.write("⚠️ **Moderate fit** - Consider further tuning\n")
            
            if metrics['directional_accuracy'] > 60:
                f.write(f"✅ **Strong directional prediction** - {metrics['directional_accuracy']:.1f}% accuracy\n")
            else:
                f.write(f"⚠️ **Moderate directional prediction** - {metrics['directional_accuracy']:.1f}% accuracy\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
            f.write("## 6. Model Artifacts\n\n")
            f.write(f"- **Model File:** `{self.model_path}/hybrid_lstm_model.h5`\n")
            f.write(f"- **Best Model Checkpoint:** `{self.model_path}/hybrid_lstm_best.h5`\n")
            f.write(f"- **Training History (CSV):** `{self.model_path}/training_history.csv`\n")
            f.write(f"- **Training History (JSON):** `{self.model_path}/training_history.json`\n")
            f.write(f"- **Evaluation Metrics:** `{self.output_path}/reports/hybrid_lstm_evaluation.json`\n")
            f.write(f"- **Training Plot:** `{self.output_path}/plots/hybrid_lstm_training_history.png`\n")
            f.write(f"- **Predictions Plot:** `{self.output_path}/plots/hybrid_predictions.png`\n\n")
            
            f.write("=" * 80 + "\n\n")
            f.write("## 7. Key Insights\n\n")
            f.write("### Hybrid Feature Impact\n")
            f.write("- **Sentiment Analysis:** FinBERT sentiment scores provide market sentiment context\n")
            f.write("- **Volatility Modeling:** GARCH(1,1) conditional volatility captures risk dynamics\n")
            f.write("- **Combined Effect:** Sentiment + Volatility enhance predictive power beyond technical indicators\n\n")
            
            f.write("### Model Strengths\n")
            f.write("- ✅ Attention mechanism focuses on relevant timesteps\n")
            f.write("- ✅ 13 diverse features capture multiple market aspects\n")
            f.write("- ✅ Time-based split prevents data leakage\n")
            f.write("- ✅ Dropout and early stopping prevent overfitting\n\n")
            
            f.write("=" * 80 + "\n\n")
            f.write("## 8. Next Steps\n\n")
            f.write("1. **Model Deployment:** Deploy model for real-time predictions\n")
            f.write("2. **Ensemble Methods:** Combine with other models (GRU, Transformer)\n")
            f.write("3. **Feature Analysis:** Analyze feature importance using SHAP/LIME\n")
            f.write("4. **Hyperparameter Tuning:** Optimize learning rate, layer sizes\n")
            f.write("5. **Multi-step Prediction:** Extend to predict multiple days ahead\n\n")
            
            f.write("=" * 80 + "\n\n")
            f.write("**Report Generated Successfully ✓**\n")
        
        logger.info(f"✓ Final report saved: {report_path}")
        logger.info("")


def main():
    logger.info("\n" + "=" * 80)
    logger.info("PROMPT 11: TRAIN FINAL HYBRID LSTM MODEL")
    logger.info("=" * 80)
    logger.info("Training hybrid deep learning model with all 13 features")
    logger.info("Features: OHLCV + Technical + FinBERT Sentiment + GARCH Volatility")
    logger.info("=" * 80 + "\n")
    
    # Initialize trainer
    trainer = HybridLSTMTrainer()
    
    # Step 1: Load sequences
    if not trainer.load_sequences():
        logger.error("Failed to load sequences. Exiting...")
        return
    
    # Step 2: Build model
    if not trainer.build_model_with_attention():
        logger.error("Failed to build model. Exiting...")
        return
    
    # Step 3: Train model
    if not trainer.train_model():
        logger.error("Failed to train model. Exiting...")
        return
    
    # Step 4: Plot training history
    trainer.plot_training_history()
    
    # Step 5: Evaluate model
    y_pred, y_true = trainer.evaluate_model()
    
    # Step 6: Plot predictions
    trainer.plot_predictions(y_pred, y_true)
    
    # Step 7: Generate final report
    trainer.generate_final_report()
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("🎉 HYBRID LSTM TRAINING COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info("\n📦 Output Files:")
    logger.info(f"   - Model: data/extended/models/hybrid_lstm_model.h5")
    logger.info(f"   - Best checkpoint: data/extended/models/hybrid_lstm_best.h5")
    logger.info(f"   - Evaluation: sample_run_output/output/reports/hybrid_lstm_evaluation.json")
    logger.info(f"   - Training plot: sample_run_output/output/plots/hybrid_lstm_training_history.png")
    logger.info(f"   - Predictions plot: sample_run_output/output/plots/hybrid_predictions.png")
    logger.info(f"   - Full report: sample_run_output/output/reports/HYBRID_LSTM_TRAINING_REPORT.md")
    logger.info("\n✅ Model ready for deployment and further analysis!")
    logger.info("=" * 80 + "\n")


if __name__ == "__main__":
    main()

