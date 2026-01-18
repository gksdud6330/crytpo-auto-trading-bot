"""
Parallel Model Runner for Crypto Trading
- Tests multiple model configurations in parallel
- Finds the most profitable model for each trading pair
- Supports: XGBoost, LightGBM, RandomForest, GradientBoosting
- Tests different parameters and feature sets
"""

import pandas as pd
import numpy as np
import joblib
import os
import json
import concurrent.futures
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import xgboost as xgb
import lightgbm as lgb
import warnings
from typing import Dict, List, Tuple, Any
warnings.filterwarnings('ignore')

# Config
DATA_DIR = '/Users/hy/Desktop/Coding/stock-market/ml_trading/data'
ENHANCED_DIR = '/Users/hy/Desktop/Coding/stock-market/ml_trading/data/enhanced'
MODEL_DIR = '/Users/hy/Desktop/Coding/stock-market/ml_trading/models'
RESULTS_DIR = '/Users/hy/Desktop/Coding/stock-market/ml_trading/reports'


class ModelConfig:
    """Model configuration container"""

    def __init__(self, name: str, model_type: str, params: Dict, features: List[str]):
        self.name = name
        self.model_type = model_type
        self.params = params
        self.features = features


# Pre-defined model configurations
MODEL_CONFIGS = [
    # XGBoost variants
    ModelConfig("XGBoost_Balanced", "xgboost", {
        'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.05,
        'subsample': 0.8, 'colsample_bytree': 0.8,
        'scale_pos_weight': 3.9, 'random_state': 42, 'eval_metric': 'logloss'
    }, "technical"),

    ModelConfig("XGBoost_Optimized", "xgboost", {
        'n_estimators': 479, 'max_depth': 6, 'learning_rate': 0.0145,
        'subsample': 0.60, 'colsample_bytree': 0.67,
        'min_child_weight': 6, 'gamma': 2.13,
        'reg_alpha': 0.108, 'reg_lambda': 0.40,
        'scale_pos_weight': 9.34, 'random_state': 42
    }, "technical"),

    ModelConfig("XGBoost_Enhanced", "xgboost", {
        'n_estimators': 300, 'max_depth': 8, 'learning_rate': 0.03,
        'subsample': 0.7, 'colsample_bytree': 0.7,
        'min_child_weight': 4, 'gamma': 2.0,
        'reg_alpha': 0.05, 'reg_lambda': 0.5,
        'scale_pos_weight': 4.0, 'random_state': 42
    }, "enhanced"),

    # LightGBM variants
    ModelConfig("LightGBM_Balanced", "lightgbm", {
        'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.05,
        'subsample': 0.8, 'colsample_bytree': 0.8,
        'class_weight': 'balanced', 'random_state': 42, 'verbose': -1
    }, "technical"),

    ModelConfig("LightGBM_Optimized", "lightgbm", {
        'n_estimators': 165, 'max_depth': 9, 'learning_rate': 0.0197,
        'subsample': 0.95, 'colsample_bytree': 0.64,
        'min_child_samples': 30,
        'reg_alpha': 0.029, 'reg_lambda': 5.6e-06,
        'class_weight': 'balanced', 'random_state': 42, 'verbose': -1
    }, "technical"),

    ModelConfig("LightGBM_Enhanced", "lightgbm", {
        'n_estimators': 250, 'max_depth': 7, 'learning_rate': 0.025,
        'subsample': 0.85, 'colsample_bytree': 0.75,
        'min_child_samples': 20,
        'reg_alpha': 0.02, 'reg_lambda': 0.01,
        'class_weight': 'balanced', 'random_state': 42, 'verbose': -1
    }, "enhanced"),

    # RandomForest variants
    ModelConfig("RandomForest_Default", "randomforest", {
        'n_estimators': 200, 'max_depth': 10,
        'min_samples_split': 10, 'min_samples_leaf': 5,
        'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1
    }, "technical"),

    ModelConfig("RandomForest_Deep", "randomforest", {
        'n_estimators': 300, 'max_depth': 15,
        'min_samples_split': 5, 'min_samples_leaf': 3,
        'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1
    }, "technical"),

    ModelConfig("RandomForest_Enhanced", "randomforest", {
        'n_estimators': 250, 'max_depth': 12,
        'min_samples_split': 8, 'min_samples_leaf': 4,
        'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1
    }, "enhanced"),

    # GradientBoosting variants
    ModelConfig("GradientBoosting_Default", "gradientboosting", {
        'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.05,
        'random_state': 42
    }, "technical"),

    ModelConfig("GradientBoosting_Deep", "gradientboosting", {
        'n_estimators': 200, 'max_depth': 7, 'learning_rate': 0.03,
        'subsample': 0.8, 'random_state': 42
    }, "technical"),

    ModelConfig("GradientBoosting_Enhanced", "gradientboosting", {
        'n_estimators': 150, 'max_depth': 6, 'learning_rate': 0.04,
        'subsample': 0.85, 'random_state': 42
    }, "enhanced"),
]


class ParallelModelRunner:
    """Run multiple models in parallel and find the best one"""

    def __init__(self, n_workers: int = 4):
        self.n_workers = n_workers
        self.data_dir = DATA_DIR
        self.enhanced_dir = ENHANCED_DIR
        self.model_dir = MODEL_DIR
        self.results_dir = RESULTS_DIR
        os.makedirs(self.results_dir, exist_ok=True)

        # Feature sets
        self.technical_features = [
            'rsi_14', 'rsi_7', 'rsi_21', 'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
            'ema_9', 'ema_21', 'ema_50', 'ema_200', 'sma_50',
            'atr_14', 'atr_pct', 'volume_sma_20', 'volume_ratio',
            'stoch_k', 'stoch_d', 'adx_14',
            'returns_1h', 'returns_4h', 'returns_1d', 'returns_1w',
            'volatility_1d', 'volatility_1w',
            'price_vs_ema_50', 'price_vs_ema_200'
        ]

        self.onchain_features = [
            'volume_spike', 'volume_trend', 'mvrv_proxy', 'nupl_proxy',
            'sopr_proxy', 'whale_activity', 'exchange_inflow_proxy',
            'active_proxy', 'hodl_proxy', 'puell_proxy',
            'volume_per_tx', 'velocity', 'accumulation_score', 'institutional_proxy'
        ]

        self.sentiment_features = [
            'fear_greed_value', 'fear_greed_score', 'price_momentum_norm',
            'volatility_fear', 'volume_sentiment', 'rsi_sentiment',
            'combined_sentiment', 'fear_index', 'search_interest_proxy',
            'interest_trend', 'fomo_proxy', 'activity_proxy',
            'sentiment_direction', 'hype_proxy', 'controversy_proxy', 'sentiment_composite'
        ]

        self.enhanced_features = (
            self.technical_features + self.onchain_features + self.sentiment_features
        )

    def get_feature_list(self, feature_set: str) -> List[str]:
        """Get feature list based on feature set name"""
        if feature_set == "technical":
            return self.technical_features
        elif feature_set == "onchain":
            return self.onchain_features
        elif feature_set == "sentiment":
            return self.sentiment_features
        elif feature_set == "enhanced":
            return self.enhanced_features
        else:
            return self.technical_features

    def load_data(self, feature_set: str) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
        """Load data and prepare features"""
        all_data = []

        if feature_set == "enhanced":
            data_dir = self.enhanced_dir
        else:
            data_dir = self.data_dir + "/4h"

        for filename in os.listdir(data_dir):
            if filename.endswith('.parquet'):
                df = pd.read_parquet(f"{data_dir}/{filename}")
                all_data.append(df)

        combined = pd.concat(all_data, ignore_index=True)
        features = self.get_feature_list(feature_set)

        X = combined[features].copy()
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        y = combined['target'].copy()

        # Time series split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test, scaler

    def create_model(self, config: ModelConfig):
        """Create model from config"""
        if config.model_type == "xgboost":
            return xgb.XGBClassifier(**config.params)
        elif config.model_type == "lightgbm":
            return lgb.LGBMClassifier(**config.params)
        elif config.model_type == "randomforest":
            return RandomForestClassifier(**config.params)
        elif config.model_type == "gradientboosting":
            return GradientBoostingClassifier(**config.params)
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")

    def evaluate_model(self, config: ModelConfig) -> Dict[str, Any]:
        """Evaluate a single model configuration"""
        try:
            print(f"  Evaluating {config.name}...", end=" ")

            # Load data
            X_train, X_test, y_train, y_test, scaler = self.load_data(config.features)

            # Create and train model
            model = self.create_model(config)
            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_test)
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0)
            }

            # Save model
            model_path = f"{self.model_dir}/{config.name.lower()}.joblib"
            joblib.dump(model, model_path)
            joblib.dump(scaler, f"{self.model_dir}/{config.name.lower()}_scaler.joblib")

            print(f"F1: {metrics['f1']:.4f}, Acc: {metrics['accuracy']:.4f}")

            return {
                'config_name': config.name,
                'model_type': config.model_type,
                'feature_set': config.features,
                'metrics': metrics,
                'model_path': model_path
            }

        except Exception as e:
            print(f"Error: {e}")
            return {
                'config_name': config.name,
                'error': str(e)
            }

    def run_parallel(self) -> List[Dict]:
        """Run all models in parallel"""
        print("="*70)
        print("PARALLEL MODEL EVALUATION")
        print(f"Models to evaluate: {len(MODEL_CONFIGS)}")
        print(f"Parallel workers: {self.n_workers}")
        print("="*70)

        results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {executor.submit(self.evaluate_model, config): config
                      for config in MODEL_CONFIGS}

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)

        return results

    def analyze_results(self, results: List[Dict]) -> Dict:
        """Analyze and rank all model results"""
        # Filter successful results
        successful = [r for r in results if 'error' not in r]

        # Sort by F1 score
        ranked = sorted(successful, key=lambda x: x['metrics']['f1'], reverse=True)

        # Find best per model type
        best_per_type = {}
        for r in successful:
            model_type = r['model_type']
            if model_type not in best_per_type or r['metrics']['f1'] > best_per_type[model_type]['metrics']['f1']:
                best_per_type[model_type] = r

        # Find best per feature set
        best_per_feature = {}
        for r in successful:
            feat = r['feature_set']
            if feat not in best_per_feature or r['metrics']['f1'] > best_per_feature[feat]['metrics']['f1']:
                best_per_feature[feat] = r

        # Overall best
        overall_best = ranked[0] if ranked else None

        return {
            'ranked_models': ranked,
            'best_per_type': best_per_type,
            'best_per_feature': best_per_feature,
            'overall_best': overall_best,
            'total_evaluated': len(successful),
            'total_failed': len(results) - len(successful)
        }

    def print_results(self, analysis: Dict):
        """Print analysis results"""
        print("\n" + "="*70)
        print("MODEL RANKING BY F1 SCORE")
        print("="*70)

        print("\n🏆 TOP 10 MODELS:")
        print("-"*70)
        print(f"{'Rank':<5} {'Model':<25} {'Type':<15} {'Features':<12} {'F1':<8} {'Acc':<8}")
        print("-"*70)

        for i, model in enumerate(analysis['ranked_models'][:10], 1):
            print(f"{i:<5} {model['config_name']:<25} {model['model_type']:<15} "
                  f"{model['feature_set']:<12} {model['metrics']['f1']:.4f}   "
                  f"{model['metrics']['accuracy']:.4f}")

        print("\n" + "="*70)
        print("BEST PER MODEL TYPE")
        print("="*70)

        for model_type, model in analysis['best_per_type'].items():
            print(f"\n{model_type.upper()}:")
            print(f"  {model['config_name']}")
            print(f"  F1: {model['metrics']['f1']:.4f}, Precision: {model['metrics']['precision']:.4f}, "
                  f"Recall: {model['metrics']['recall']:.4f}")

        print("\n" + "="*70)
        print("BEST PER FEATURE SET")
        print("="*70)

        for feature_set, model in analysis['best_per_feature'].items():
            print(f"\n{feature_set.upper()} features:")
            print(f"  {model['config_name']}")
            print(f"  F1: {model['metrics']['f1']:.4f}")

        print("\n" + "="*70)
        print("OVERALL BEST MODEL")
        print("="*70)

        if analysis['overall_best']:
            best = analysis['overall_best']
            print(f"\n🏆 {best['config_name']}")
            print(f"   Model Type: {best['model_type']}")
            print(f"   Feature Set: {best['feature_set']}")
            print(f"   F1 Score: {best['metrics']['f1']:.4f}")
            print(f"   Accuracy: {best['metrics']['accuracy']:.4f}")
            print(f"   Precision: {best['metrics']['precision']:.4f}")
            print(f"   Recall: {best['metrics']['recall']:.4f}")
            print(f"   Model Path: {best['model_path']}")

        print(f"\n\n📊 Summary: {analysis['total_evaluated']} models evaluated, "
              f"{analysis['total_failed']} failed")

    def save_results(self, analysis: Dict):
        """Save results to files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save detailed results
        report_file = f"{self.results_dir}/parallel_model_report_{timestamp}.txt"

        with open(report_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("PARALLEL MODEL EVALUATION REPORT\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write("="*70 + "\n\n")

            f.write("COMPLETE RANKING\n")
            f.write("-"*70 + "\n")
            for i, model in enumerate(analysis['ranked_models'], 1):
                f.write(f"\n{i}. {model['config_name']}\n")
                f.write(f"   Type: {model['model_type']}\n")
                f.write(f"   Features: {model['feature_set']}\n")
                f.write(f"   F1: {model['metrics']['f1']:.4f}\n")
                f.write(f"   Accuracy: {model['metrics']['accuracy']:.4f}\n")
                f.write(f"   Precision: {model['metrics']['precision']:.4f}\n")
                f.write(f"   Recall: {model['metrics']['recall']:.4f}\n")

            f.write("\n\nBEST PER TYPE\n")
            f.write("-"*70 + "\n")
            for model_type, model in analysis['best_per_type'].items():
                f.write(f"\n{model_type}: {model['config_name']} (F1: {model['metrics']['f1']:.4f})\n")

            f.write("\n\nBEST PER FEATURE SET\n")
            f.write("-"*70 + "\n")
            for feature_set, model in analysis['best_per_feature'].items():
                f.write(f"\n{feature_set}: {model['config_name']} (F1: {model['metrics']['f1']:.4f})\n")

            if analysis['overall_best']:
                best = analysis['overall_best']
                f.write("\n\nOVERALL BEST\n")
                f.write("-"*70 + "\n")
                f.write(f"{best['config_name']}\n")
                f.write(f"F1: {best['metrics']['f1']:.4f}\n")
                f.write(f"Path: {best['model_path']}\n")

        # Save best model config as JSON
        if analysis['overall_best']:
            best_config = {
                'model_name': analysis['overall_best']['config_name'],
                'model_type': analysis['overall_best']['model_type'],
                'feature_set': analysis['overall_best']['feature_set'],
                'metrics': analysis['overall_best']['metrics'],
                'model_path': analysis['overall_best']['model_path'],
                'timestamp': datetime.now().isoformat()
            }

            with open(f"{self.model_dir}/best_model_config.json", 'w') as f:
                json.dump(best_config, f, indent=2)

        print(f"\n✅ Report saved to: {report_file}")
        if analysis['overall_best']:
            print(f"✅ Best config saved to: {self.model_dir}/best_model_config.json")

    def run(self):
        """Run complete parallel evaluation"""
        results = self.run_parallel()
        analysis = self.analyze_results(results)
        self.print_results(analysis)
        self.save_results(analysis)

        return analysis


def main():
    """Main entry point"""
    runner = ParallelModelRunner(n_workers=4)
    analysis = runner.run()

    return analysis


if __name__ == '__main__':
    main()
