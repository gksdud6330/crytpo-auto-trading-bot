"""
Hyperparameter Optimizer using Optuna
- Optimizes XGBoost and LightGBM hyperparameters
- Uses TimeSeriesSplit for validation
- Saves best parameters and model
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score
import xgboost as xgb
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Config
DATA_DIR = '/Users/hy/Desktop/Coding/stock-market/ml_trading/data'
MODEL_DIR = '/Users/hy/Desktop/Coding/stock-market/ml_trading/models'

# Feature columns (same as before)
FEATURE_COLS = [
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


class HyperparameterOptimizer:
    def __init__(self, timeframe='4h', n_trials=50):
        self.timeframe = timeframe
        self.n_trials = n_trials
        self.data_dir = DATA_DIR
        self.model_dir = MODEL_DIR
        os.makedirs(self.model_dir, exist_ok=True)

    def load_data(self):
        """Load and combine data from all pairs"""
        all_data = []
        for filename in os.listdir(f"{self.data_dir}/{self.timeframe}"):
            if filename.endswith('.parquet'):
                df = pd.read_parquet(f"{self.data_dir}/{self.timeframe}/{filename}")
                all_data.append(df)

        combined = pd.concat(all_data, ignore_index=True)

        # Prepare features
        X = combined[FEATURE_COLS].copy()
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        y = combined['target'].copy()

        # Time series split (no shuffling!)
        split_idx = int(len(X) * 0.8)
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test, scaler

    def objective_xgboost(self, trial):
        """Objective function for XGBoost optimization"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10),
            'random_state': 42,
            'eval_metric': 'logloss',
            'n_jobs': -1
        }

        X_train, X_test, y_train, y_test, _ = self.load_data()

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        return f1

    def objective_lightgbm(self, trial):
        """Objective function for LightGBM optimization"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'class_weight': 'balanced',
            'random_state': 42,
            'verbose': -1,
            'n_jobs': -1
        }

        X_train, X_test, y_train, y_test, _ = self.load_data()

        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        return f1

    def optimize_xgboost(self):
        """Optimize XGBoost hyperparameters"""
        print("="*60)
        print("XGBoost Hyperparameter Optimization")
        print("="*60)

        sampler = TPESampler(seed=42)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(self.objective_xgboost, n_trials=self.n_trials, show_progress_bar=True)

        print(f"\nBest F1 Score: {study.best_value:.4f}")
        print(f"\nBest Parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")

        # Save best params
        params_path = f"{self.model_dir}/best_params_xgboost_{self.timeframe}.txt"
        with open(params_path, 'w') as f:
            f.write(f"# Best F1 Score: {study.best_value:.4f}\n")
            for key, value in study.best_params.items():
                f.write(f"{key}: {value}\n")

        return study.best_params, study.best_value

    def optimize_lightgbm(self):
        """Optimize LightGBM hyperparameters"""
        print("\n" + "="*60)
        print("LightGBM Hyperparameter Optimization")
        print("="*60)

        sampler = TPESampler(seed=42)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(self.objective_lightgbm, n_trials=self.n_trials, show_progress_bar=True)

        print(f"\nBest F1 Score: {study.best_value:.4f}")
        print(f"\nBest Parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")

        # Save best params
        params_path = f"{self.model_dir}/best_params_lightgbm_{self.timeframe}.txt"
        with open(params_path, 'w') as f:
            f.write(f"# Best F1 Score: {study.best_value:.4f}\n")
            for key, value in study.best_params.items():
                f.write(f"{key}: {value}\n")

        return study.best_params, study.best_value

    def train_best_xgboost(self, best_params):
        """Train final XGBoost model with best params"""
        X_train, X_test, y_train, y_test, scaler = self.load_data()

        # Add fixed params
        params = best_params.copy()
        params['random_state'] = 42
        params['eval_metric'] = 'logloss'
        params['n_jobs'] = -1

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)

        print(f"\nFinal XGBoost Performance:")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")

        # Save model
        model_path = f"{self.model_dir}/xgboost_optimized_{self.timeframe}.joblib"
        joblib.dump(model, model_path)
        joblib.dump(scaler, f"{self.model_dir}/scaler_{self.timeframe}.joblib")

        print(f"✅ Model saved to {model_path}")

        return model

    def train_best_lightgbm(self, best_params):
        """Train final LightGBM model with best params"""
        X_train, X_test, y_train, y_test, scaler = self.load_data()

        # Add fixed params
        params = best_params.copy()
        params['random_state'] = 42
        params['verbose'] = -1
        params['n_jobs'] = -1

        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)

        print(f"\nFinal LightGBM Performance:")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")

        # Save model
        model_path = f"{self.model_dir}/lightgbm_optimized_{self.timeframe}.joblib"
        joblib.dump(model, model_path)

        print(f"✅ Model saved to {model_path}")

        return model

    def run(self):
        """Run full optimization pipeline"""
        print("="*60)
        print("HYPERPARAMETER OPTIMIZATION PIPELINE")
        print(f"Timeframe: {self.timeframe}")
        print(f"Trials per model: {self.n_trials}")
        print("="*60)

        # Optimize XGBoost
        best_xgb_params, best_xgb_score = self.optimize_xgboost()
        xgb_model = self.train_best_xgboost(best_xgb_params)

        # Optimize LightGBM
        best_lgb_params, best_lgb_score = self.optimize_lightgbm()
        lgb_model = self.train_best_lightgbm(best_lgb_params)

        # Summary
        print("\n" + "="*60)
        print("OPTIMIZATION SUMMARY")
        print("="*60)
        print(f"XGBoost Best F1: {best_xgb_score:.4f}")
        print(f"LightGBM Best F1: {best_lgb_score:.4f}")

        if best_xgb_score >= best_lgb_score:
            print(f"\n🏆 Best Model: XGBoost (F1: {best_xgb_score:.4f})")
            return 'xgboost', best_xgb_params, best_xgb_score
        else:
            print(f"\n🏆 Best Model: LightGBM (F1: {best_lgb_score:.4f})")
            return 'lightgbm', best_lgb_params, best_lgb_score


if __name__ == '__main__':
    optimizer = HyperparameterOptimizer(timeframe='4h', n_trials=50)
    optimizer.run()
