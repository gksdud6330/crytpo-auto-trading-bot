"""
ML Model Trainer for Crypto Trading
- Trains multiple models
- Evaluates performance
- Saves best model
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Config
DATA_DIR = '/Users/hy/Desktop/Coding/stock-market/ml_trading/data'
MODEL_DIR = '/Users/hy/Desktop/Coding/stock-market/ml_trading/models'
REPORTS_DIR = '/Users/hy/Desktop/Coding/stock-market/ml_trading/reports'

class MLModelTrainer:
    def __init__(self):
        self.data_dir = DATA_DIR
        self.model_dir = MODEL_DIR
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(REPORTS_DIR, exist_ok=True)
        
        # Feature columns
        self.feature_cols = [
            'rsi_14', 'rsi_7', 'rsi_21',
            'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
            'ema_9', 'ema_21', 'ema_50', 'ema_200', 'sma_50',
            'atr_14', 'atr_pct',
            'volume_sma_20', 'volume_ratio',
            'stoch_k', 'stoch_d', 'adx_14',
            'returns_1h', 'returns_4h', 'returns_1d', 'returns_1w',
            'volatility_1d', 'volatility_1w',
            'price_vs_ema_50', 'price_vs_ema_200'
        ]
        
        self.target_col = 'target'
    
    def load_data(self, timeframe='4h'):
        """Load and combine data from all pairs"""
        all_data = []
        
        for filename in os.listdir(f"{self.data_dir}/{timeframe}"):
            if filename.endswith('.parquet'):
                df = pd.read_parquet(f"{self.data_dir}/{timeframe}/{filename}")
                df['pair'] = filename.replace('.parquet', '').replace('_', '/')
                all_data.append(df)
        
        if not all_data:
            raise ValueError(f"No data found in {self.data_dir}/{timeframe}")
        
        combined = pd.concat(all_data, ignore_index=True)
        return combined
    
    def prepare_features(self, df):
        """Prepare features and target"""
        # Filter columns
        feature_df = df[self.feature_cols].copy()
        
        # Handle infinite values
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
        feature_df = feature_df.fillna(0)
        
        # Target
        target = df[self.target_col].copy()
        
        return feature_df, target
    
    def train_test_split(self, X, y, test_size=0.2):
        """Time series aware train/test split"""
        # Use last 20% as test set (no shuffling!)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train, X_test):
        """Scale features"""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler
    
    def train_models(self, X_train, y_train):
        """Train multiple models"""
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
                random_state=42,
                eval_metric='logloss'
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight='balanced',
                random_state=42,
                verbose=-1
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                random_state=42
            )
        }
        
        trained_models = {}
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            trained_models[name] = model
            print(f"  {name} trained successfully")
        
        return trained_models
    
    def evaluate_models(self, models, X_test, y_test):
        """Evaluate all models"""
        results = {}
        
        for name, model in models.items():
            y_pred = model.predict(X_test)
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0)
            }
            
            results[name] = metrics
            
            print(f"\n{name}:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1 Score:  {metrics['f1']:.4f}")
        
        return results
    
    def find_best_model(self, results):
        """Find best model based on F1 score"""
        best_model = max(results, key=lambda x: results[x]['f1'])
        return best_model
    
    def save_best_model(self, models, scaler, best_model_name, timeframe='4h'):
        """Save best model and scaler"""
        # Save model
        model_path = f"{self.model_dir}/crypto_predictor_{timeframe}.joblib"
        joblib.dump(models[best_model_name], model_path)
        
        # Save scaler
        scaler_path = f"{self.model_dir}/scaler_{timeframe}.joblib"
        joblib.dump(scaler, scaler_path)
        
        # Save feature columns
        feature_path = f"{self.model_dir}/features_{timeframe}.txt"
        with open(feature_path, 'w') as f:
            for col in self.feature_cols:
                f.write(f"{col}\n")
        
        print(f"\n✅ Best model ({best_model_name}) saved to {model_path}")
        print(f"✅ Scaler saved to {scaler_path}")
        
        return model_path
    
    def run(self, timeframe='4h', test_size=0.2):
        """Run full training pipeline"""
        print("="*60)
        print(f"ML Model Training Pipeline - {timeframe}")
        print("="*60)
        
        # Load data
        print("\n[1/5] Loading data...")
        df = self.load_data(timeframe)
        print(f"  Loaded {len(df)} samples from {df['pair'].nunique()} pairs")
        
        # Prepare features
        print("\n[2/5] Preparing features...")
        X, y = self.prepare_features(df)
        print(f"  Features: {X.shape[1]}, Samples: {X.shape[0]}")
        print(f"  Target distribution: {dict(y.value_counts())}")
        
        # Train/test split
        print("\n[3/5] Splitting data...")
        X_train, X_test, y_train, y_test = self.train_test_split(X, y, test_size)
        print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Scale features
        print("\n[4/5] Scaling features...")
        X_train_scaled, X_test_scaled, scaler = self.scale_features(X_train, X_test)
        
        # Train models
        print("\n[5/5] Training models...")
        models = self.train_models(X_train_scaled, y_train)
        
        # Evaluate
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        results = self.evaluate_models(models, X_test_scaled, y_test)
        
        # Find and save best
        best_model = self.find_best_model(results)
        print(f"\n🏆 Best Model: {best_model} (F1: {results[best_model]['f1']:.4f})")
        
        self.save_best_model(models, scaler, best_model, timeframe)
        
        return models, results

if __name__ == '__main__':
    trainer = MLModelTrainer()
    trainer.run(timeframe='4h')
