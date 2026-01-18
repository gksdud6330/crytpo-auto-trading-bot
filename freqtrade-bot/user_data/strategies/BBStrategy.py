from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class BBStrategy(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '5m'
    startup_candle_count = 200
    
    minimal_roi = {
        "0": 0.122,
        "28": 0.089,
        "67": 0.025,
        "180": 0
    }
    
    stoploss = -0.256
    
    trailing_stop = True
    trailing_stop_positive = 0.187
    trailing_stop_positive_offset = 0.244
    trailing_only_offset_is_reached = True
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['volume_mean'] = dataframe['volume'].rolling(window=20).mean()
        
        bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe['bb_lower'] = bollinger['lowerband']
        dataframe['bb_middle'] = bollinger['middleband']
        dataframe['bb_upper'] = bollinger['upperband']
        dataframe['bb_width'] = (dataframe['bb_upper'] - dataframe['bb_lower']) / dataframe['bb_middle']
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['close'] < dataframe['bb_lower'] * 1.01) &
                (dataframe['ema_50'] > dataframe['ema_200']) &
                (dataframe['adx'] > 20) &
                (dataframe['volume'] > dataframe['volume_mean'])
            ),
            'enter_long'] = 1
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['close'] > dataframe['bb_middle']) |
                (dataframe['rsi'] > 60) |
                (dataframe['ema_50'] < dataframe['ema_200'])
            ) &
            (dataframe['volume'] > 0),
            'exit_long'] = 1
        return dataframe
