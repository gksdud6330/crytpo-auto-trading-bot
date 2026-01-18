"""
ML Signal Strategy for Freqtrade
- Uses ML model predictions for trading signals
- Combines ML signals with technical indicators for confirmation
- Implements proper risk management
"""

import numpy as np
from pandas import DataFrame
from freqtrade.strategy import IStrategy
from freqtrade.vendor.qtpylib.indicators import (
    rsi, ema, adx, bb_upper, bb_lower, macd, stochastic
)
import joblib
import os
from datetime import datetime

# ML Model path
MODEL_PATH = '/Users/hy/Desktop/Coding/stock-market/ml_trading/models'
SCALER_PATH = '/Users/hy/Desktop/Coding/stock-market/ml_trading/models/scaler_4h.joblib'
FEATURES_PATH = '/Users/hy/Desktop/Coding/stock-market/ml_trading/models/features_4h.txt'

# Strategy parameters
ML_CONFIDENCE_THRESHOLD = 0.55  # Minimum confidence for BUY signal
USE_ML_FILTER = True  # If True, only trade when ML predicts UP
MIN_ADX = 15  # Minimum ADX for trend confirmation
MAX_RSI_OVERSOLD = 40  # RSI below this is oversold (BUY zone)
MIN_VOLUME_RATIO = 0.8  # Minimum volume relative to 20-period average


class MLStrategy(IStrategy):
    """
    ML-based trading strategy that combines:
    1. ML model predictions (primary signal)
    2. Technical indicator confirmation (filters)
    3. Risk management rules
    """

    # Strategy metadata
    minimal_roi = {
        "0": 0.03,      # 3% profit target
        "60": 0.02,     # 2% after 1 hour
        "180": 0.015,   # 1.5% after 3 hours
        "360": 0.01,    # 1% after 6 hours
        "720": 0.005,   # 0.5% after 12 hours
    }

    stoploss = -0.05  # 5% stop loss
    trailing_stop = True
    trailing_stop_positive = 0.015  # 1.5% trailing
    trailing_stop_positive_offset = 0.03  # Start trailing after 3% profit
    timeframe = '4h'

    # Process only completed candles
    process_only_new_candles = True

    # Startup candles needed for indicators
    startup_candles = 200

    # ML model and scaler (loaded once)
    _ml_model = None
    _scaler = None
    _feature_cols = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_ml_model()

    def load_ml_model(self):
        """Load ML model and scaler on strategy initialization"""
        if MLStrategy._ml_model is None:
            model_file = os.path.join(MODEL_PATH, 'xgboost_optimized_4h.joblib')

            if os.path.exists(model_file):
                try:
                    MLStrategy._ml_model = joblib.load(model_file)
                    MLStrategy._scaler = joblib.load(SCALER_PATH)

                    # Load feature columns
                    with open(FEATURES_PATH, 'r') as f:
                        MLStrategy._feature_cols = [line.strip() for line in f]

                    print(f"✅ ML Strategy: Model loaded from {model_file}")
                except Exception as e:
                    print(f"❌ ML Strategy: Failed to load model: {e}")
                    MLStrategy._ml_model = None
            else:
                print(f"❌ ML Strategy: Model file not found: {model_file}")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Calculate technical indicators"""

        # RSI
        dataframe['rsi_14'] = rsi(dataframe, timeperiod=14)
        dataframe['rsi_7'] = rsi(dataframe, timeperiod=7)
        dataframe['rsi_21'] = rsi(dataframe, timeperiod=21)

        # EMA
        dataframe['ema_9'] = ema(dataframe, timeperiod=9)
        dataframe['ema_21'] = ema(dataframe, timeperiod=21)
        dataframe['ema_50'] = ema(dataframe, timeperiod=50)
        dataframe['ema_200'] = ema(dataframe, timeperiod=200)

        # MACD
        macd_line, macd_signal, macd_hist = macd(dataframe)
        dataframe['macd'] = macd_line
        dataframe['macd_signal'] = macd_signal
        dataframe['macd_hist'] = macd_hist

        # Bollinger Bands
        dataframe['bb_upper'] = bb_upper(dataframe, timeperiod=20, nbdevup=2, nbdevdn=2)
        dataframe['bb_middle'] = ema(dataframe, timeperiod=20)
        dataframe['bb_lower'] = bb_lower(dataframe, timeperiod=20, nbdevup=2, nbdevdn=2)
        dataframe['bb_width'] = (dataframe['bb_upper'] - dataframe['bb_lower']) / dataframe['bb_middle']

        # ADX (Average Directional Index)
        dataframe['adx_14'] = adx(dataframe, timeperiod=14)

        # Stochastic
        stoch_k, stoch_d = stochastic(dataframe)
        dataframe['stoch_k'] = stoch_k
        dataframe['stoch_d'] = stoch_d

        # Volume
        dataframe['volume_sma_20'] = dataframe['volume'].rolling(window=20).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_sma_20']

        # Returns
        dataframe['returns_1h'] = dataframe['close'].pct_change(1)
        dataframe['returns_4h'] = dataframe['close'].pct_change(1)
        dataframe['returns_1d'] = dataframe['close'].pct_change(6)  # 4h * 6 = 1d
        dataframe['returns_1w'] = dataframe['close'].pct_change(42)  # 4h * 42 = 1w

        # Volatility
        dataframe['volatility_1d'] = dataframe['returns_1h'].rolling(window=6).std()
        dataframe['volatility_1w'] = dataframe['returns_1h'].rolling(window=42).std()

        # Price position
        dataframe['price_vs_ema_50'] = dataframe['close'] / dataframe['ema_50'] - 1
        dataframe['price_vs_ema_200'] = dataframe['close'] / dataframe['ema_200'] - 1

        # ATR
        high_low = dataframe['high'] - dataframe['low']
        high_close = np.abs(dataframe['high'] - dataframe['close'].shift())
        low_close = np.abs(dataframe['low'] - dataframe['close'].shift())
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        dataframe['atr_14'] = tr.rolling(window=14).mean()
        dataframe['atr_pct'] = dataframe['atr_14'] / dataframe['close']

        # ML Signal (only if model is loaded)
        if MLStrategy._ml_model is not None and MLStrategy._feature_cols is not None:
            dataframe['ml_signal'] = self.get_ml_signal(dataframe)
        else:
            dataframe['ml_signal'] = 0  # Neutral when model not loaded

        return dataframe

    def get_ml_signal(self, dataframe: DataFrame) -> np.ndarray:
        """Get ML prediction signal for the latest candle"""
        try:
            # Get latest row features
            latest = dataframe.iloc[-1:][MLStrategy._feature_cols].copy()

            # Handle infinite/NaN
            latest = latest.replace([np.inf, -np.inf], np.nan).fillna(0)

            # Scale
            latest_scaled = MLStrategy._scaler.transform(latest)

            # Predict
            prediction = MLStrategy._ml_model.predict(latest_scaled)[0]
            probability = MLStrategy._ml_model.predict_proba(latest_scaled)[0]

            # Return signal: 1 = BUY, 0 = HOLD
            if prediction == 1 and probability[1] >= ML_CONFIDENCE_THRESHOLD:
                return 1
            else:
                return 0

        except Exception as e:
            print(f"ML prediction error: {e}")
            return 0

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Define buy signals"""
        buy_conditions = []

        # ML Signal must be positive
        if USE_ML_FILTER:
            buy_conditions.append(dataframe['ml_signal'] == 1)

        # Trend confirmation: ADX above minimum
        buy_conditions.append(dataframe['adx_14'] > MIN_ADX)

        # RSI not oversold (we want upward momentum, not bottom fishing)
        buy_conditions.append(dataframe['rsi_14'] < 70)
        buy_conditions.append(dataframe['rsi_14'] > 30)

        # MACD histogram turning positive or already positive
        buy_conditions.append(dataframe['macd_hist'] > dataframe['macd_hist'].shift(1))

        # Price above EMAs (uptrend confirmation)
        buy_conditions.append(dataframe['close'] > dataframe['ema_50'])
        buy_conditions.append(dataframe['close'] > dataframe['ema_200'])

        # Volume confirmation
        buy_conditions.append(dataframe['volume_ratio'] > MIN_VOLUME_RATIO)

        # Stochastic not overbought
        buy_conditions.append(dataframe['stoch_k'] < 80)

        if buy_conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, buy_conditions),
                'buy'
            ] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Define sell signals"""
        sell_conditions = []

        # Take profit: Price increased significantly
        sell_conditions.append(dataframe['close'] > dataframe['close'].shift(1) * 1.03)

        # Or: RSI overbought
        sell_conditions.append(dataframe['rsi_14'] > 80)

        # Or: MACD turning negative
        sell_conditions.append(dataframe['macd_hist'] < 0)
        sell_conditions.append(dataframe['macd_hist'] < dataframe['macd_hist'].shift(1))

        # Or: Price below major EMAs (trend reversal)
        sell_conditions.append(dataframe['close'] < dataframe['ema_50'])

        if sell_conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, sell_conditions),
                'sell'
            ] = 1

        return dataframe

    def custom_stake_amount(self, pair: str, current_time, current_rate, current_profit, **kwargs) -> float:
        """
        Calculate stake amount based on ML confidence
        - Higher ML confidence = larger position
        - Default to 5% of account
        """
        if MLStrategy._ml_model is None:
            return self.wallets.get_total_stake_amount() * 0.05

        try:
            # This would need access to the dataframe for confidence
            # Simplified version: use base stake
            return self.wallets.get_total_stake_amount() * 0.05
        except:
            return self.wallets.get_total_stake_amount() * 0.05

    def leverage(self, pair: str, current_time, current_rate, current_profit, **kwargs) -> float:
        """
        Adjust leverage based on ML confidence
        - High confidence: 2x leverage
        - Medium confidence: 1.5x leverage
        - Low confidence: 1x leverage
        """
        return 1.0  # Default 1x leverage for safety

    def bot_loop_started(self, current_datetime) -> None:
        """Called when the bot loop starts"""
        if current_datetime is None:
            return

        # Log ML model status
        if MLStrategy._ml_model is not None:
            print(f"🧠 ML Strategy active - using ML signals")
        else:
            print(f"⚠️ ML Strategy inactive - model not loaded")


# Import reduce for combining conditions
from functools import reduce
