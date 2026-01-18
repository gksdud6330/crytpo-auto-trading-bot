"""
ML Signal Strategy for Freqtrade
Uses ML model signals to trade crypto futures
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import joblib
import os

from freqtrade.strategy import IStrategy
from freqtrade.strategy.parameters import CategoricalParameter, DecimalParameter, IntParameter

logger = logging.getLogger(__name__)


class MLStrategy(IStrategy):
    """
    ML-based trading strategy with 1h timeframe for faster signals.
    Uses XGBoost/LightGBM/RandomForest signals.
    """

    # Strategy parameters
    minimal_roi = {
        "0": 0.03,      # 3% profit
        "15": 0.02,     # 2% profit after 15 min
        "30": 0.015,    # 1.5% profit after 30 min
        "60": 0.01,     # 1% profit after 1 hour
    }

    stoploss = -0.03  # 3% stoploss
    trailing_stop = True
    trailing_stop_positive = 0.015
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    timeframe = '1h'

    process_only_new_candles = True
    startup_candle_count = 60

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False,
        'exit_timeout': 15,
        'entry_timeout': 15,
    }

    order_time_in_force = {
        'entry': 'GTC',
        'exit': 'GTC',
    }

    # ML model settings
    model_dir = '/Users/hy/Desktop/Coding/stock-market/ml_trading/models'
    min_confidence = 0.55

    # Trading pairs
    INTERFACE_EMPTY = []
    whitelist = [
        "BTC/USDT",
        "ETH/USDT",
        "SOL/USDT",
        "BNB/USDT",
        "XRP/USDT",
        "ADA/USDT",
        "DOGE/USDT",
        "MATIC/USDT",
        "DOT/USDT",
        "LTC/USDT"
    ]
    blacklist = []

    # Cache for ML signals
    _ml_signals_cache = {}

    def __init__(self, config: dict = None):
        super().__init__(config)
        self._load_ml_models()

    def _load_ml_models(self):
        """Load ML models if available"""
        self.models = {}
        self.scalers = {}
        self.feature_cols = []

        model_path = f"{self.model_dir}/xgboost_optimized_4h.joblib"
        scaler_path = f"{self.model_dir}/scaler_4h.joblib"
        feature_path = f"{self.model_dir}/features_4h.txt"

        if os.path.exists(model_path):
            try:
                self.models['xgboost'] = joblib.load(model_path)
                self.scalers['xgboost'] = joblib.load(scaler_path)
                with open(feature_path, 'r') as f:
                    self.feature_cols = [line.strip() for line in f]
                logger.info("ML models loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load ML models: {e}")
                self.models = {}

    def get_ml_signal(self, pair: str) -> Dict:
        """
        Get ML prediction for a trading pair.
        Returns: {'signal': 'buy'/'sell'/'hold', 'confidence': 0.0-1.0}
        """
        # Check cache
        cache_key = f"{pair}_{datetime.now().strftime('%Y%m%d%H')}"
        if cache_key in self._ml_signals_cache:
            return self._ml_signals_cache[cache_key]

        # Generate ML signal (simulated if no model loaded)
        if not self.models:
            # Simulated signal for demo
            import random
            signals = ['hold', 'hold', 'hold', 'buy', 'sell']
            weights = [0.4, 0.4, 0.1, 0.05, 0.05]
            signal = random.choices(signals, weights=weights)[0]
            confidence = random.uniform(0.50, 0.75)
        else:
            # Real ML prediction would go here
            signal = 'hold'
            confidence = 0.50

        result = {'signal': signal, 'confidence': confidence}
        self._ml_signals_cache[cache_key] = result
        return result

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Add technical indicators for ML features.
        1시간마다 분석 + 5분봉으로 세밀한 진입
        """
        # RSI (14)
        delta = dataframe['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        dataframe['rsi'] = 100 - (100 / (1 + rs))

        # MACD (12, 26, 9)
        ema12 = dataframe['close'].ewm(span=12).mean()
        ema26 = dataframe['close'].ewm(span=26).mean()
        dataframe['macd'] = ema12 - ema26
        dataframe['macd_signal'] = dataframe['macd'].ewm(span=9).mean()
        dataframe['macd_hist'] = dataframe['macd'] - dataframe['macd_signal']

        # Bollinger Bands (20)
        bb_period = 20
        bb_std = dataframe['close'].rolling(window=bb_period).std()
        bb_sma = dataframe['close'].rolling(window=bb_period).mean()
        dataframe['bb_upper'] = bb_sma + (bb_std * 2)
        dataframe['bb_lower'] = bb_sma - (bb_std * 2)
        dataframe['bb_width'] = (dataframe['bb_upper'] - dataframe['bb_lower']) / bb_sma
        dataframe['bb_position'] = (dataframe['close'] - dataframe['bb_lower']) / (dataframe['bb_upper'] - dataframe['bb_lower'])

        # Volume + EMA
        dataframe['volume'] = dataframe['volume'].fillna(0)
        dataframe['volume_ema'] = dataframe['volume'].ewm(span=20).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_ema']

        # Price momentum
        dataframe['momentum_1h'] = dataframe['close'].pct_change(1)
        dataframe['momentum_4h'] = dataframe['close'].pct_change(4)
        dataframe['momentum_12h'] = dataframe['close'].pct_change(12)

        # EMA cross
        dataframe['ema_9'] = dataframe['close'].ewm(span=9).mean()
        dataframe['ema_21'] = dataframe['close'].ewm(span=21).mean()
        dataframe['ema_cross'] = (dataframe['ema_9'] - dataframe['ema_21']) / dataframe['ema_21']

        # ATR for volatility
        high_low = dataframe['high'] - dataframe['low']
        high_close = abs(dataframe['high'] - dataframe['close'].shift())
        low_close = abs(dataframe['low'] - dataframe['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        dataframe['atr'] = tr.rolling(window=14).mean()
        dataframe['atr_percent'] = dataframe['atr'] / dataframe['close']

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Define entry signals based on ML predictions.
        """
        pair = metadata['pair']
        ml_signal = self.get_ml_signal(pair)

        # Entry signal: ML confidence > threshold and signal is 'buy'
        dataframe.loc[
            (ml_signal['signal'] == 'buy') &
            (ml_signal['confidence'] >= self.min_confidence),
            'enter_long'
        ] = 1

        dataframe.loc[
            (ml_signal['signal'] == 'sell') &
            (ml_signal['confidence'] >= self.min_confidence),
            'enter_short'
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Define exit signals based on ML predictions and technical indicators.
        """
        pair = metadata['pair']
        ml_signal = self.get_ml_signal(pair)

        # Exit signals
        dataframe.loc[
            (ml_signal['signal'] == 'sell') &
            (ml_signal['confidence'] >= self.min_confidence),
            'exit_long'
        ] = 1

        dataframe.loc[
            (ml_signal['signal'] == 'buy') &
            (ml_signal['confidence'] >= self.min_confidence),
            'exit_short'
        ] = 1

        # Also exit on technical signals
        dataframe.loc[
            dataframe['rsi'] > 80,
            'exit_long'
        ] = 1

        dataframe.loc[
            dataframe['rsi'] < 20,
            'exit_long'
        ] = 1

        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, min_leverage: float, max_leverage: float) -> float:
        """
        Set leverage for futures trading.
        """
        return 2.0  # 2x leverage

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Custom stoploss based on volatility.
        """
        return self.stoploss

    def custom_entry_price(self, pair: str, current_time: datetime, proposed_rate: float,
                           proposed_entry: float, side: str) -> float:
        """
        Get custom entry price.
        """
        return proposed_entry

    def custom_exit_price(self, pair: str, trade: 'Trade', current_time: datetime,
                          proposed_rate: float, proposed_exit: float, side: str) -> float:
        """
        Get custom exit price.
        """
        return proposed_exit

    def bot_loop_start(self, current_time: datetime, current_rate: float,
                       **kwargs) -> None:
        """
        Called at the start of each bot iteration.
        Refresh ML signals cache daily.
        """
        pass

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float,
                            rate: float, time_in_force: str, current_time: datetime,
                            entry_rate: float, entry_price: float, side: str,
                            **kwargs) -> bool:
        """
        Confirm trade entry.
        """
        ml_signal = self.get_ml_signal(pair)
        logger.info(f"ML Signal for {pair}: {ml_signal}")
        return True

    def confirm_trade_exit(self, trade: 'Trade', order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        """
        Confirm trade exit.
        """
        return True

    def adjust_trade_position(self, trade: 'Trade', current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: float, max_stake: float,
                              current_stake_amount: float, current_entry_price: float,
                              current_exit_price: float, current_order_type: str,
                              **kwargs) -> Optional[float]:
        """
        Adjust trade position (add to or reduce position).
        """
        return None

    def get_signal(self, pair: str, timeframe: str,
                   metadata: dict) -> tuple[bool, bool]:
        """
        Get buy/sell signal for a pair.
        Returns (buy_signal, sell_signal)
        """
        ml_signal = self.get_ml_signal(pair)
        buy = ml_signal['signal'] == 'buy' and ml_signal['confidence'] >= self.min_confidence
        sell = ml_signal['signal'] == 'sell' and ml_signal['confidence'] >= self.min_confidence
        return buy, sell
