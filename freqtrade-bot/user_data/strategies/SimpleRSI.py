from freqtrade.strategy import IStrategy
from pandas import DataFrame
import numpy as np
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class SimpleRSI(IStrategy):
    """
    Simple RSI Reversal Strategy
    - RSI < 30: Buy
    - RSI > 70: Sell
    - Simple and effective
    """
    INTERFACE_VERSION = 3
    timeframe = '5m'
    startup_candle_count = 50
    
    minimal_roi = {
        "0": 0.02,      # 2% profit target
        "30": 0.015,    # 1.5% after 30 min
        "60": 0.01,     # 1% after 1 hour
        "120": 0        # Break even
    }
    
    stoploss = -0.03
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=7)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['volume_mean'] = dataframe['volume'].rolling(window=20).mean()
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Simple RSI oversold entry
        dataframe.loc[
            (
                (dataframe['rsi'] < 35) |
                (qtpylib.crossed_above(dataframe['rsi_fast'], dataframe['rsi'])) &
                (dataframe['rsi'] < 45)
            ) &
            (dataframe['volume'] > dataframe['volume_mean'] * 0.3),
            'enter_long'
        ] = 1
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # RSI overbought or reversal
        dataframe.loc[
            (
                (dataframe['rsi'] > 65) |
                (qtpylib.crossed_below(dataframe['rsi_fast'], dataframe['rsi'])) &
                (dataframe['rsi'] > 55)
            ) &
            (dataframe['volume'] > 0),
            'exit_long'
        ] = 1
        return dataframe
