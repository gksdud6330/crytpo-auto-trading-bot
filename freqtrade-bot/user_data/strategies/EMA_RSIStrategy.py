from freqtrade.strategy import IStrategy, merge_informative_pair
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class EMA_RSIStrategy(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '5m'
    startup_candle_count = 200
    
    minimal_roi = {
        "0": 0.045,
        "60": 0.03,
        "180": 0.015,
        "360": 0
    }
    
    stoploss = -0.07
    
    trailing_stop = True
    trailing_stop_positive = 0.012
    trailing_stop_positive_offset = 0.035
    trailing_only_offset_is_reached = True
    
    def informative_pairs(self):
        whitelist = []
        if self.dp:
            whitelist = self.dp.current_whitelist()
        if not whitelist:
            whitelist = self.config.get('exchange', {}).get('pair_whitelist', [])
        return [(pair, '1h') for pair in whitelist]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['volume_mean'] = dataframe['volume'].rolling(window=20).mean()

        if self.dp:
            informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe='1h')
            informative['ema_50'] = ta.EMA(informative, timeperiod=50)
            informative['ema_200'] = ta.EMA(informative, timeperiod=200)
            dataframe = merge_informative_pair(dataframe, informative, self.timeframe, '1h', ffill=True)

        if 'ema_50_1h' not in dataframe:
            dataframe['ema_50_1h'] = dataframe['ema_50']
        if 'ema_200_1h' not in dataframe:
            dataframe['ema_200_1h'] = dataframe['ema_200']

        dataframe['ema_50_1h'] = dataframe['ema_50_1h'].fillna(dataframe['ema_50'])
        dataframe['ema_200_1h'] = dataframe['ema_200_1h'].fillna(dataframe['ema_200'])

        dataframe['trend_1h'] = dataframe['ema_50_1h'] >= (dataframe['ema_200_1h'] * 0.995)
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['trend_1h']) &
                (dataframe['ema_20'] > dataframe['ema_50']) &
                (dataframe['close'] > dataframe['ema_50']) &
                (dataframe['rsi'] > 40) &
                (dataframe['rsi'] < 70) &
                (dataframe['adx'] > 18) &
                (dataframe['volume'] > dataframe['volume_mean'] * 0.8)
            ),
            'enter_long'] = 1
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (~dataframe['trend_1h']) |
                (dataframe['ema_20'] < dataframe['ema_50']) |
                (dataframe['rsi'] < 35) |
                (dataframe['close'] < dataframe['ema_50'])
            ) &
            (dataframe['volume'] > 0),
            'exit_long'] = 1
        return dataframe
