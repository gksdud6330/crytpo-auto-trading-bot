"""
Crypto Trading Data Collector
- Downloads OHLCV data from exchanges
- Calculates technical indicators
- Fetches on-chain data
- Saves to structured format
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time
import warnings
warnings.filterwarnings('ignore')

# Config
DATA_DIR = '/Users/hy/Desktop/Coding/stock-market/ml_trading/data'
EXCHANGE = 'bitget'
TIMEFRAMES = ['4h', '1d']
PAIRS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT']
START_DATE = '2023-01-01'

class DataCollector:
    def __init__(self):
        self.exchange = getattr(ccxt, EXCHANGE)({
            'enableRateLimit': True,
            'rateLimit': 100,
        })
        self.data_dir = DATA_DIR
        
    def download_ohlcv(self, symbol, timeframe, since):
        """Download OHLCV data from exchange"""
        all_candles = []
        max_retries = 3
        
        while True:
            try:
                candles = self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    since=int(datetime.strptime(since, '%Y-%m-%d').timestamp() * 1000),
                    limit=1000
                )
                
                if not candles:
                    break
                    
                all_candles.extend(candles)
                
                # Update since to last candle timestamp
                since = datetime.fromtimestamp(candles[-1][0] / 1000 + 1).strftime('%Y-%m-%d')
                
                # Check if we have recent data
                if datetime.strptime(since, '%Y-%m-%d') > datetime.now() - timedelta(days=1):
                    break
                    
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                print(f"Error downloading {symbol} {timeframe}: {e}")
                max_retries -= 1
                if max_retries == 0:
                    break
                time.sleep(5)
        
        return all_candles
    
    def ohlcv_to_dataframe(self, candles):
        """Convert OHLCV candles to DataFrame"""
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    
    def calculate_indicators(self, df):
        """Calculate technical indicators"""
        # RSI
        df['rsi_14'] = self._rsi(df['close'], 14)
        df['rsi_7'] = self._rsi(df['close'], 7)
        df['rsi_21'] = self._rsi(df['close'], 21)
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = self._macd(df['close'])
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._bollinger_bands(df['close'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # EMA
        for period in [9, 21, 50, 200]:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        # SMA
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # ATR
        df['atr_14'] = self._atr(df, 14)
        df['atr_pct'] = df['atr_14'] / df['close']
        
        # Volume
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # Stochastic
        df['stoch_k'], df['stoch_d'] = self._stochastic(df)
        
        # ADX
        df['adx_14'] = self._adx(df)
        
        # Returns
        df['returns_1h'] = df['close'].pct_change(1)  # 4h timeframe = 1 candle
        df['returns_4h'] = df['close'].pct_change(1)
        df['returns_1d'] = df['close'].pct_change(6)  # 4h * 6 = 1d
        df['returns_1w'] = df['close'].pct_change(42)  # 4h * 42 = 1w
        
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
    
    def _macd(self, series, fast=12, slow=26, signal=9):
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        macd_hist = macd_line - signal_line
        return macd_line, signal_line, macd_hist
    
    def _bollinger_bands(self, series, period=20, std=2):
        middle = series.rolling(window=period).mean()
        std_dev = series.rolling(window=period).std()
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        return upper, middle, lower
    
    def _atr(self, df, period):
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    def _stochastic(self, df, period=14, smooth_k=3, smooth_d=3):
        lowest_low = df['low'].rolling(window=period).min()
        highest_high = df['high'].rolling(window=period).max()
        stoch_k = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low)
        stoch_d = stoch_k.rolling(window=smooth_d).mean()
        return stoch_k, stoch_d
    
    def _adx(self, df, period=14):
        plus_di = 100 * (df['high'].diff()).rolling(window=period).mean() / df['atr_14']
        minus_di = 100 * (df['low'].diff().rolling(window=period).mean().abs()) / df['atr_14']
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        return dx.rolling(window=period).mean()
    
    def create_target(self, df, lookahead=6, threshold=0.02):
        """
        Create target variable for ML model
        - 1: Price goes up more than threshold within lookahead candles
        - 0: Otherwise
        """
        future_returns = df['close'].shift(-lookahead) / df['close'] - 1
        df['target'] = (future_returns > threshold).astype(int)
        df['target_return'] = future_returns
        return df
    
    def save_data(self, df, symbol, timeframe):
        """Save data to parquet file"""
        os.makedirs(f"{self.data_dir}/{timeframe}", exist_ok=True)
        filename = f"{self.data_dir}/{timeframe}/{symbol.replace('/', '_')}.parquet"
        df.to_parquet(filename)
        print(f"Saved {symbol} {timeframe}: {len(df)} rows")
    
    def collect_all(self):
        """Collect all data for all pairs and timeframes"""
        for timeframe in TIMEFRAMES:
            for pair in PAIRS:
                print(f"\n{'='*50}")
                print(f"Downloading {pair} {timeframe}")
                print('='*50)
                
                # Download OHLCV
                candles = self.download_ohlcv(pair, timeframe, START_DATE)
                
                if not candles:
                    print(f"No data for {pair} {timeframe}")
                    continue
                
                # Convert to DataFrame
                df = self.ohlcv_to_dataframe(candles)
                
                # Calculate indicators
                df = self.calculate_indicators(df)
                
                # Create target
                df = self.create_target(df)
                
                # Drop NaN rows (for indicators that need warmup)
                df = df.dropna()
                
                # Save
                self.save_data(df, pair, timeframe)

if __name__ == '__main__':
    collector = DataCollector()
    collector.collect_all()
