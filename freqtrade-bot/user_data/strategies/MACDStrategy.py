from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class MACDStrategy(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '5m'
    startup_candle_count = 200
    
    minimal_roi = {
        "0": 0.05,
        "60": 0.03,
        "180": 0.015,
        "360": 0
    }
    
    stoploss = -0.08
    
    trailing_stop = True
    trailing_stop_positive = 0.015
    trailing_stop_positive_offset = 0.04
    trailing_only_offset_is_reached = True
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['volume_mean'] = dataframe['volume'].rolling(window=20).mean()
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['macd'], dataframe['macdsignal'])) &
                (dataframe['macdhist'] > 0) &
                (dataframe['ema_50'] > dataframe['ema_200']) &
                (dataframe['close'] > dataframe['ema_50']) &
                (dataframe['adx'] > 20) &
                (dataframe['volume'] > dataframe['volume_mean'])
            ),
            'enter_long'] = 1
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_below(dataframe['macd'], dataframe['macdsignal'])) |
                (dataframe['close'] < dataframe['ema_50']) |
                (dataframe['adx'] < 15)
            ) &
            (dataframe['volume'] > 0),
            'exit_long'] = 1
        return dataframe
