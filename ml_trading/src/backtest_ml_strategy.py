"""
ML Strategy Backtester
- Tests ML-based trading signals against historical data
- Evaluates performance metrics
- Compares with buy & hold strategy
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Config
DATA_DIR = '/Users/hy/Desktop/Coding/stock-market/ml_trading/data'
MODEL_DIR = '/Users/hy/Desktop/Coding/stock-market/ml_trading/models'
RESULTS_DIR = '/Users/hy/Desktop/Coding/stock-market/ml_trading/reports'

# Strategy parameters
ML_CONFIDENCE_THRESHOLD = 0.55
MIN_ADX = 15
MAX_RSI_OVERSOLD = 40
MIN_VOLUME_RATIO = 0.8


class MLBacktester:
    def __init__(self, timeframe='4h'):
        self.timeframe = timeframe
        self.data_dir = DATA_DIR
        self.model_dir = MODEL_DIR
        os.makedirs(RESULTS_DIR, exist_ok=True)

        # Load model and scaler
        self.load_model()

    def load_model(self):
        """Load ML model and scaler"""
        model_file = os.path.join(MODEL_DIR, 'xgboost_optimized_4h.joblib')
        scaler_file = os.path.join(MODEL_DIR, 'scaler_4h.joblib')
        feature_file = os.path.join(MODEL_DIR, 'features_4h.txt')

        if os.path.exists(model_file):
            self.model = joblib.load(model_file)
            self.scaler = joblib.load(scaler_file)

            with open(feature_file, 'r') as f:
                self.feature_cols = [line.strip() for line in f]

            print(f"✅ Model loaded: {model_file}")
        else:
            raise ValueError(f"Model not found: {model_file}")

    def load_data(self):
        """Load all data for backtesting"""
        all_data = []

        for filename in os.listdir(f"{self.data_dir}/{self.timeframe}"):
            if filename.endswith('.parquet'):
                df = pd.read_parquet(f"{self.data_dir}/{self.timeframe}/{filename}")
                df['pair'] = filename.replace('.parquet', '').replace('_', '/')
                all_data.append(df)

        combined = pd.concat(all_data, ignore_index=True)
        return combined

    def calculate_indicators(self, df):
        """Calculate technical indicators (same as in strategy)"""
        df = df.copy()

        # RSI
        df['rsi_14'] = self._rsi(df['close'], 14)
        df['rsi_7'] = self._rsi(df['close'], 7)
        df['rsi_21'] = self._rsi(df['close'], 21)

        # EMA
        df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()

        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

        # ADX
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        df['atr_14'] = tr.rolling(window=14).mean()

        plus_di = 100 * (df['high'].diff()).rolling(window=14).mean() / df['atr_14']
        minus_di = 100 * (df['low'].diff().rolling(window=14).mean().abs()) / df['atr_14']
        df['adx_14'] = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        df['adx_14'] = df['adx_14'].rolling(window=14).mean()

        # Stochastic
        lowest_low = df['low'].rolling(window=14).min()
        highest_high = df['high'].rolling(window=14).max()
        df['stoch_k'] = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low)
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()

        # Volume
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']

        # Returns and volatility
        df['returns_1h'] = df['close'].pct_change(1)
        df['returns_4h'] = df['close'].pct_change(1)
        df['returns_1d'] = df['close'].pct_change(6)
        df['returns_1w'] = df['close'].pct_change(42)
        df['volatility_1d'] = df['returns_1h'].rolling(window=6).std()
        df['volatility_1w'] = df['returns_1h'].rolling(window=42).std()

        # Price position
        df['price_vs_ema_50'] = df['close'] / df['ema_50'] - 1
        df['price_vs_ema_200'] = df['close'] / df['ema_200'] - 1

        # ATR percentage
        df['atr_pct'] = df['atr_14'] / df['close']

        return df

    def _rsi(self, series, period):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def generate_ml_signals(self, df):
        """Generate ML buy/sell signals"""
        df = df.copy()

        # Prepare features
        X = df[self.feature_cols].copy()
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Scale
        X_scaled = self.scaler.transform(X)

        # Predict
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]

        # Generate signals
        df['ml_prediction'] = predictions
        df['ml_probability'] = probabilities

        # Buy signal: ML predicts BUY with confidence >= threshold
        df['ml_buy_signal'] = ((predictions == 1) & (probabilities >= ML_CONFIDENCE_THRESHOLD)).astype(int)

        return df

    def apply_strategy_rules(self, df):
        """Apply strategy rules to generate final buy/sell signals"""
        df = df.copy()

        # BUY conditions
        buy_conditions = (
            (df['ml_buy_signal'] == 1) &  # ML signal
            (df['adx_14'] > MIN_ADX) &  # Trending
            (df['rsi_14'] < 70) &  # Not overbought
            (df['rsi_14'] > 30) &  # Not oversold
            (df['macd_hist'] > df['macd_hist'].shift(1)) &  # MACD turning up
            (df['close'] > df['ema_50']) &  # Above major EMA
            (df['close'] > df['ema_200']) &  # Above long-term EMA
            (df['volume_ratio'] > MIN_VOLUME_RATIO) &  # Volume confirmation
            (df['stoch_k'] < 80)  # Not overbought
        )

        df['buy_signal'] = buy_conditions.astype(int)

        # SELL conditions
        sell_conditions = (
            (df['close'] > df['close'].shift(1) * 1.03) |  # 3% profit target
            (df['rsi_14'] > 80) |  # Overbought
            ((df['macd_hist'] < 0) & (df['macd_hist'] < df['macd_hist'].shift(1))) |  # MACD turning down
            (df['close'] < df['ema_50'])  # Trend reversal
        )

        df['sell_signal'] = sell_conditions.astype(int)

        return df

    def simulate_trading(self, df):
        """Simulate trading and calculate returns"""
        df = df.copy()

        initial_balance = 10000
        balance = initial_balance
        position = 0  # 0 = no position, 1 = long
        entry_price = 0
        entry_date = None
        entry_idx = None
        trades = []

        for i in range(1, len(df)):
            price = df.iloc[i]['close']
            current_date = df.index[i]
            entry_date_val = df.index[entry_idx] if entry_idx is not None else None

            # Check sell signal if we have a position
            if position == 1 and df.iloc[i]['sell_signal'] == 1:
                # Close position
                pnl = (price - entry_price) / entry_price
                balance *= (1 + pnl)
                trades.append({
                    'entry_date': entry_date_val,
                    'exit_date': current_date,
                    'entry_price': entry_price,
                    'exit_price': price,
                    'return': pnl,
                    'pair': df.iloc[i]['pair']
                })
                position = 0

            # Check buy signal if we don't have a position
            if position == 0 and df.iloc[i]['buy_signal'] == 1:
                position = 1
                entry_price = price
                entry_idx = i
                entry_date = current_date

        # Calculate metrics
        total_trades = len(trades)
        if total_trades > 0:
            returns = [t['return'] for t in trades]
            winning_trades = sum(1 for r in returns if r > 0)
            losing_trades = sum(1 for r in returns if r <= 0)
            win_rate = winning_trades / total_trades
            avg_return = np.mean(returns)
            max_return = max(returns)
            min_return = min(returns)
            total_return = (balance / initial_balance) - 1
        else:
            win_rate = 0
            avg_return = 0
            max_return = 0
            min_return = 0
            total_return = 0
            returns = []

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades if total_trades > 0 else 0,
            'losing_trades': losing_trades if total_trades > 0 else 0,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'max_return': max_return,
            'min_return': min_return,
            'total_return': total_return,
            'final_balance': balance,
            'trades': trades
        }

    def run_backtest(self):
        """Run full backtest"""
        print("="*60)
        print("ML STRATEGY BACKTEST")
        print("="*60)

        # Load data
        print("\n[1/4] Loading data...")
        df = self.load_data()
        print(f"  Loaded {len(df)} samples from {df['pair'].nunique()} pairs")

        # Calculate indicators
        print("\n[2/4] Calculating indicators...")
        df = self.calculate_indicators(df)

        # Generate ML signals
        print("\n[3/4] Generating ML signals...")
        df = self.generate_ml_signals(df)

        # Apply strategy rules
        print("\n[4/4] Applying strategy rules...")
        df = self.apply_strategy_rules(df)

        # Simulate trading
        print("\n" + "="*60)
        print("TRADING SIMULATION")
        print("="*60)

        results = {}
        for pair in df['pair'].unique():
            pair_df = df[df['pair'] == pair]
            pair_result = self.simulate_trading(pair_df)
            results[pair] = pair_result

            print(f"\n{pair}:")
            print(f"  Trades: {pair_result['total_trades']}")
            print(f"  Win Rate: {pair_result['win_rate']:.1%}")
            print(f"  Total Return: {pair_result['total_return']:.2%}")

        # Aggregate results
        all_trades = []
        for pair_result in results.values():
            all_trades.extend(pair_result['trades'])

        if all_trades:
            all_returns = [t['return'] for t in all_trades]
            winning = sum(1 for r in all_returns if r > 0)
            total = len(all_returns)

            agg_results = {
                'total_trades': total,
                'winning_trades': winning,
                'losing_trades': total - winning,
                'win_rate': winning / total if total > 0 else 0,
                'avg_return': np.mean(all_returns),
                'max_return': max(all_returns),
                'min_return': min(all_returns),
            }
        else:
            agg_results = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_return': 0,
                'max_return': 0,
                'min_return': 0,
            }

        print("\n" + "="*60)
        print("AGGREGATE RESULTS")
        print("="*60)
        print(f"\nTotal Trades: {agg_results['total_trades']}")
        print(f"Win Rate: {agg_results['win_rate']:.1%}")
        print(f"Average Return: {agg_results['avg_return']:.2%}")
        print(f"Max Return: {agg_results['max_return']:.2%}")
        print(f"Min Return: {agg_results['min_return']:.2%}")

        # Compare with buy & hold
        print("\n" + "="*60)
        print("COMPARISON: ML STRATEGY vs BUY & HOLD")
        print("="*60)

        for pair in results:
            pair_df = df[df['pair'] == pair]
            bh_return = (pair_df.iloc[-1]['close'] / pair_df.iloc[0]['close']) - 1
            ml_return = results[pair]['total_return']

            print(f"\n{pair}:")
            print(f"  ML Strategy: {ml_return:.2%}")
            print(f"  Buy & Hold:  {bh_return:.2%}")
            print(f"  Difference:  {(ml_return - bh_return):.2%}")

        # Save results
        self.save_results(results, agg_results)

        return results, agg_results

    def save_results(self, results, agg_results):
        """Save backtest results to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = os.path.join(RESULTS_DIR, f'backtest_report_{timestamp}.txt')

        with open(report_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("ML STRATEGY BACKTEST REPORT\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write("="*60 + "\n\n")

            f.write("AGGREGATE RESULTS\n")
            f.write("-"*40 + "\n")
            f.write(f"Total Trades: {agg_results['total_trades']}\n")
            f.write(f"Win Rate: {agg_results['win_rate']:.1%}\n")
            f.write(f"Average Return: {agg_results['avg_return']:.2%}\n")
            f.write(f"Max Return: {agg_results['max_return']:.2%}\n")
            f.write(f"Min Return: {agg_results['min_return']:.2%}\n\n")

            f.write("PER PAIR RESULTS\n")
            f.write("-"*40 + "\n")
            for pair, result in results.items():
                f.write(f"\n{pair}:\n")
                f.write(f"  Trades: {result['total_trades']}\n")
                f.write(f"  Win Rate: {result['win_rate']:.1%}\n")
                f.write(f"  Total Return: {result['total_return']:.2%}\n")

        print(f"\n✅ Report saved to: {report_file}")


def main():
    backtester = MLBacktester(timeframe='4h')
    results, agg_results = backtester.run_backtest()


if __name__ == '__main__':
    main()
