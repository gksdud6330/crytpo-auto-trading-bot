"""
Crypto Trading Predictor
- Loads trained ML model
- Makes predictions on new data
- Generates trading signals
"""

import pandas as pd
import numpy as np
import ccxt
import joblib
from datetime import datetime, timedelta
import os
import time
import warnings
warnings.filterwarnings('ignore')

# Config
MODEL_DIR = '/Users/hy/Desktop/Coding/stock-market/ml_trading/models'
EXCHANGE = 'bitget'
PAIRS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT']
TIMEFRAMES = ['4h', '1d']

class CryptoPredictor:
    def __init__(self, timeframe='4h'):
        self.timeframe = timeframe
        self.model_dir = MODEL_DIR
        
        # Load model
        model_path = f"{MODEL_DIR}/crypto_predictor_{timeframe}.joblib"
        scaler_path = f"{MODEL_DIR}/scaler_{timeframe}.joblib"
        feature_path = f"{MODEL_DIR}/features_{timeframe}.txt"
        
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            
            with open(feature_path, 'r') as f:
                self.feature_cols = [line.strip() for line in f]
            
            print(f"✅ Model loaded: {model_path}")
        else:
            print(f"❌ Model not found: {model_path}")
            self.model = None
        
        # Initialize exchange
        self.exchange = getattr(ccxt, EXCHANGE)({
            'enableRateLimit': True,
            'rateLimit': 100,
        })
    
    def fetch_latest_candles(self, symbol, limit=100):
        """Fetch latest candles for a symbol"""
        try:
            candles = self.exchange.fetch_ohlcv(
                symbol,
                timeframe=self.timeframe,
                limit=limit
            )
            return candles
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return None
    
    def candles_to_dataframe(self, candles):
        """Convert candles to DataFrame"""
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    
    def calculate_indicators(self, df):
        """Calculate indicators (same as in data_collector.py)"""
        # RSI
        df['rsi_14'] = self._rsi(df['close'], 14)
        df['rsi_7'] = self._rsi(df['close'], 7)
        df['rsi_21'] = self._rsi(df['close'], 21)
        
        # MACD
        df['macd'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # EMA
        for period in [9, 21, 50, 200]:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        # SMA
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(window=14).mean()
        df['atr_pct'] = df['atr_14'] / df['close']
        
        # Volume
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # Stochastic
        lowest_low = df['low'].rolling(window=14).min()
        highest_high = df['high'].rolling(window=14).max()
        df['stoch_k'] = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low)
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        # ADX
        plus_di = 100 * (df['high'].diff()).rolling(window=14).mean() / df['atr_14']
        minus_di = 100 * (df['low'].diff().rolling(window=14).mean().abs()) / df['atr_14']
        df['adx_14'] = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        df['adx_14'] = df['adx_14'].rolling(window=14).mean()
        
        # Returns
        df['returns_1h'] = df['close'].pct_change(1)
        df['returns_4h'] = df['close'].pct_change(1)
        df['returns_1d'] = df['close'].pct_change(6)
        df['returns_1w'] = df['close'].pct_change(42)
        
        # Volatility
        df['volatility_1d'] = df['returns_1h'].rolling(window=6).std()
        df['volatility_1w'] = df['returns_1h'].rolling(window=42).std()
        
        # Price position
        df['price_vs_ema_50'] = df['close'] / df['ema_50'] - 1
        df['price_vs_ema_200'] = df['close'] / df['ema_200'] - 1
        
        return df
    
    def _rsi(self, series, period):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def predict(self, symbol):
        """Generate prediction for a symbol"""
        if self.model is None:
            return None
        
        # Fetch latest candles
        candles = self.fetch_latest_candles(symbol)
        if not candles:
            return None
        
        # Convert to DataFrame
        df = self.candles_to_dataframe(candles)
        
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        # Get latest row (current state)
        latest = df.iloc[-1:][self.feature_cols].copy()
        
        # Handle infinite/NaN
        latest = latest.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Scale
        latest_scaled = self.scaler.transform(latest)
        
        # Predict
        prediction = self.model.predict(latest_scaled)[0]
        probability = self.model.predict_proba(latest_scaled)[0]
        
        current_price = df.iloc[-1]['close']
        
        return {
            'symbol': symbol,
            'price': current_price,
            'prediction': 'BUY' if prediction == 1 else 'HOLD',
            'confidence': probability[1] if prediction == 1 else probability[0],
            'buy_probability': probability[1],
            'hold_probability': probability[0],
            'timestamp': datetime.now().isoformat()
        }
    
    def predict_all(self):
        """Generate predictions for all pairs"""
        results = []
        for pair in PAIRS:
            result = self.predict(pair)
            if result:
                results.append(result)
        return results
    
    def generate_signals(self, min_confidence=0.6):
        """Generate trading signals"""
        predictions = self.predict_all()
        
        signals = []
        for p in predictions:
            if p['prediction'] == 'BUY' and p['confidence'] >= min_confidence:
                signals.append({
                    'signal': 'BUY',
                    'symbol': p['symbol'],
                    'price': p['price'],
                    'confidence': p['confidence'],
                    'reason': f"ML model predicts UP with {p['confidence']:.1%} confidence"
                })
            elif p['prediction'] == 'HOLD' or p['confidence'] < min_confidence:
                signals.append({
                    'signal': 'HOLD',
                    'symbol': p['symbol'],
                    'price': p['price'],
                    'confidence': p['confidence'],
                    'reason': f"Low confidence ({p['confidence']:.1%}) or HOLD signal"
                })
        
        return signals

def main():
    predictor = CryptoPredictor(timeframe='4h')
    
    print("\n" + "="*60)
    print("CRYPTO ML TRADING SIGNALS")
    print("="*60)
    
    # Get predictions
    signals = predictor.generate_signals(min_confidence=0.55)
    
    for signal in signals:
        emoji = "🟢" if signal['signal'] == 'BUY' else "🟡"
        print(f"\n{emoji} {signal['signal']} - {signal['symbol']}")
        print(f"   Price: ${signal['price']:,.2f}")
        print(f"   Confidence: {signal['confidence']:.1%}")
        print(f"   Reason: {signal['reason']}")
    
    return signals

if __name__ == '__main__':
    main()
