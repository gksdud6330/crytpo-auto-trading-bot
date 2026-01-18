"""
Ensemble Model for Crypto Trading
- Combines predictions from multiple models
- Uses voting (hard and averaging) (soft) ensemble methods
- Generates more robust trading signals
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# Config
MODEL_DIR = '/Users/hy/Desktop/Coding/stock-market/ml_trading/models'
DATA_DIR = '/Users/hy/Desktop/Coding/stock-market/ml_trading/data'

# Feature columns
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


class EnsemblePredictor:
    def __init__(self, timeframe='4h'):
        self.timeframe = timeframe
        self.model_dir = MODEL_DIR
        self.models = {}
        self.scaler = None
        self.feature_cols = FEATURE_COLS

        # Load all models
        self.load_models()

    def load_models(self):
        """Load all available models"""
        model_files = {
            'random_forest': f'{MODEL_DIR}/crypto_predictor_{self.timeframe}.joblib',
            'xgboost': f'{MODEL_DIR}/xgboost_optimized_{self.timeframe}.joblib',
            'lightgbm': f'{MODEL_DIR}/lightgbm_optimized_{self.timeframe}.joblib',
        }

        # Try to load XGBoost optimized first (usually best performer)
        for name, path in model_files.items():
            if os.path.exists(path):
                try:
                    self.models[name] = joblib.load(path)
                    print(f"✅ Loaded {name}: {path}")
                except Exception as e:
                    print(f"❌ Failed to load {name}: {e}")

        # Load scaler
        scaler_path = f'{MODEL_DIR}/scaler_{self.timeframe}.joblib'
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print(f"✅ Loaded scaler")

    def prepare_features(self, df):
        """Prepare features from DataFrame"""
        X = df[self.feature_cols].copy()
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        return X

    def predict_hard_voting(self, X):
        """Hard voting - majority vote"""
        predictions = []
        for name, model in self.models.items():
            pred = model.predict(X)
            predictions.append(pred)

        predictions = np.array(predictions)
        # Majority vote
        result = np.round(np.mean(predictions, axis=0)).astype(int)
        return result

    def predict_soft_voting(self, X):
        """Soft voting - average probabilities"""
        probabilities = []
        for name, model in self.models.items():
            prob = model.predict_proba(X)[:, 1]  # Probability of class 1
            probabilities.append(prob)

        probabilities = np.array(probabilities)
        # Average probability
        avg_prob = np.mean(probabilities, axis=0)
        result = (avg_prob >= 0.5).astype(int)
        return result, avg_prob

    def predict_weighted_voting(self, X, weights=None):
        """Weighted voting - weighted average of probabilities"""
        if weights is None:
            # Equal weights
            weights = np.ones(len(self.models)) / len(self.models)

        probabilities = []
        for name, model in self.models.items():
            prob = model.predict_proba(X)[:, 1]
            probabilities.append(prob)

        probabilities = np.array(probabilities)
        # Weighted average
        weighted_prob = np.average(probabilities, axis=0, weights=weights)
        result = (weighted_prob >= 0.5).astype(int)
        return result, weighted_prob

    def evaluate_on_test_data(self):
        """Evaluate ensemble on test data"""
        # Load test data
        all_data = []
        for filename in os.listdir(f"{DATA_DIR}/{self.timeframe}"):
            if filename.endswith('.parquet'):
                df = pd.read_parquet(f"{DATA_DIR}/{self.timeframe}/{filename}")
                all_data.append(df)

        combined = pd.concat(all_data, ignore_index=True)

        # Split (last 20%)
        split_idx = int(len(combined) * 0.8)
        test_df = combined.iloc[split_idx:]

        X_test = self.prepare_features(test_df)
        y_test = test_df['target'].values

        # Scale
        X_test_scaled = self.scaler.transform(X_test)

        # Evaluate each model
        print("\n" + "="*60)
        print("INDIVIDUAL MODEL PERFORMANCE")
        print("="*60)

        individual_results = {}
        for name, model in self.models.items():
            y_pred = model.predict(X_test_scaled)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            accuracy = accuracy_score(y_test, y_pred)

            individual_results[name] = {
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'accuracy': accuracy
            }

            print(f"\n{name}:")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1 Score:  {f1:.4f}")

        # Evaluate ensembles
        print("\n" + "="*60)
        print("ENSEMBLE PERFORMANCE")
        print("="*60)

        # Hard voting
        y_pred_hard = self.predict_hard_voting(X_test_scaled)
        f1_hard = f1_score(y_test, y_pred_hard, zero_division=0)
        print(f"\nHard Voting:")
        print(f"  F1 Score:  {f1_hard:.4f}")

        # Soft voting
        y_pred_soft, _ = self.predict_soft_voting(X_test_scaled)
        f1_soft = f1_score(y_test, y_pred_soft, zero_division=0)
        print(f"\nSoft Voting:")
        print(f"  F1 Score:  {f1_soft:.4f}")

        # Weighted voting (weight by F1 score)
        weights = [individual_results[name]['f1'] for name in self.models.keys()]
        weights = np.array(weights) / sum(weights)
        print(f"\nWeights: {dict(zip(self.models.keys(), weights))}")

        y_pred_weighted, _ = self.predict_weighted_voting(X_test_scaled, weights)
        f1_weighted = f1_score(y_test, y_pred_weighted, zero_division=0)
        print(f"\nWeighted Voting:")
        print(f"  F1 Score:  {f1_weighted:.4f}")

        # Find best method
        methods = {
            'Hard Voting': f1_hard,
            'Soft Voting': f1_soft,
            'Weighted Voting': f1_weighted
        }
        best_method = max(methods, key=methods.get)
        print(f"\n🏆 Best Method: {best_method} (F1: {methods[best_method]:.4f})")

        return methods

    def get_prediction(self, X_scaled):
        """Get ensemble prediction for single sample"""
        _, probabilities = self.predict_soft_voting(X_scaled)
        avg_prob = probabilities[0]

        prediction = 'BUY' if avg_prob >= 0.5 else 'HOLD'
        confidence = avg_prob if prediction == 'BUY' else 1 - avg_prob

        return {
            'prediction': prediction,
            'buy_probability': avg_prob,
            'confidence': confidence
        }


def main():
    print("="*60)
    print("ENSEMBLE MODEL EVALUATION")
    print("="*60)

    ensemble = EnsemblePredictor(timeframe='4h')
    results = ensemble.evaluate_on_test_data()

    return results


if __name__ == '__main__':
    main()
