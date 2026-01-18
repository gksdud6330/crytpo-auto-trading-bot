from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class RSIStrategy(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '5m'
    startup_candle_count = 200
    
    minimal_roi = {
        "0": 0.04,
        "60": 0.025,
        "180": 0.01,
        "360": 0
    }
    
    stoploss = -0.06
    
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True
    
    buy_rsi_low = 30
    sell_rsi = 60
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=7)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['volume_mean'] = dataframe['volume'].rolling(window=20).mean()
        
        bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe['bb_lower'] = bollinger['lowerband']
        dataframe['bb_middle'] = bollinger['middleband']
        dataframe['bb_upper'] = bollinger['upperband']
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['rsi'], self.buy_rsi_low)) &
                (dataframe['rsi_fast'] > dataframe['rsi']) &
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
                (qtpylib.crossed_above(dataframe['rsi'], self.sell_rsi)) |
                (dataframe['close'] > dataframe['bb_middle']) |
                (dataframe['ema_50'] < dataframe['ema_200'])
            ) &
            (dataframe['volume'] > 0),
            'exit_long'] = 1
        return dataframe
