from freqtrade.strategy import IStrategy, merge_informative_pair
from pandas import DataFrame
import numpy as np
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class AggressiveScalper(IStrategy):
    """
    Aggressive Scalper - PROFIT MAXIMIZATION FOCUS
    - Fast entries/exits
    - Multiple confirmation (but relaxed)
    - High trading frequency
    """
    INTERFACE_VERSION = 3
    timeframe = '5m'
    startup_candle_count = 50
    
    # Quick profit targets
    minimal_roi = {
        "0": 0.01,      # 1% immediate
        "10": 0.008,    # 0.8% after 10 min
        "30": 0.005,    # 0.5% after 30 min
        "60": 0         # Break even
    }
    
    stoploss = -0.02
    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.01
    trailing_only_offset_is_reached = True
    
    def informative_pairs(self):
        whitelist = []
        if self.dp:
            whitelist = self.dp.current_whitelist()
        if not whitelist:
            whitelist = self.config.get('exchange', {}).get('pair_whitelist', [])
        return [(pair, '1h') for pair in whitelist]
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # === RSI ===
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=7)
        
        # === EMA ===
        dataframe['ema_9'] = ta.EMA(dataframe, timeperiod=9)
        dataframe['ema_21'] = ta.EMA(dataframe, timeperiod=21)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        
        # === MACD ===
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        
        # === Bollinger Bands ===
        bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe['bb_lower'] = bollinger['lowerband']
        dataframe['bb_middle'] = bollinger['middleband']
        dataframe['bb_upper'] = bollinger['upperband']
        
        # === Stochastic ===
        stoch = ta.STOCH(dataframe)
        dataframe['stoch_k'] = stoch['slowk']
        dataframe['stoch_d'] = stoch['slowd']
        
        # === Volume ===
        dataframe['volume'] = dataframe['volume'].astype(float)
        dataframe['volume_mean'] = dataframe['volume'].rolling(window=20).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_mean']
        
        # === ADX (Trend Strength) ===
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        
        # === ATR (Volatility) ===
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        
        # === 1h Trend Confirmation ===
        if self.dp:
            informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe='1h')
            informative['ema_21'] = ta.EMA(informative, timeperiod=21)
            dataframe = merge_informative_pair(dataframe, informative, '5m', '1h', ffill=True)
        
        if 'ema_21_1h' not in dataframe:
            dataframe['ema_21_1h'] = dataframe['ema_21']
        dataframe['ema_21_1h'] = dataframe['ema_21_1h'].fillna(dataframe['ema_21'])
        
        # === Trend Signals ===
        dataframe['trend_up_5m'] = dataframe['ema_9'] > dataframe['ema_21']
        dataframe['trend_up_1h'] = dataframe['close'] > dataframe['ema_21_1h']
        dataframe['rsi_oversold'] = dataframe['rsi'] < 45
        dataframe['rsi_overbought'] = dataframe['rsi'] > 55
        dataframe['macd_bullish'] = dataframe['macd'] > dataframe['macdsignal']
        dataframe['stoch_bullish'] = dataframe['stoch_k'] > dataframe['stoch_d']
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # === ENTRY CONDITIONS (Relaxed for more trades) ===
        
        # Main entry: Multiple bullish signals
        entry_1 = (
            dataframe['trend_up_5m'] &  # 5m uptrend
            dataframe['rsi_oversold'] &  # RSI not overbought
            dataframe['macd_bullish'] &  # MACD bullish
            dataframe['stoch_bullish'] &  # Stochastic bullish
            dataframe['volume_ratio'] > 0.5  # Minimal volume
        )
        
        # RSI reversal entry
        entry_2 = (
            (dataframe['rsi'] < 50) &
            (dataframe['rsi_fast'] > dataframe['rsi']) &
            (dataframe['close'] > dataframe['bb_lower']) &
            dataframe['trend_up_5m']
        )
        
        # MACD cross entry
        entry_3 = (
            qtpylib.crossed_above(dataframe['macd'], dataframe['macdsignal']) &
            (dataframe['macdhist'] > -0.5) &
            dataframe['trend_up_5m']
        )
        
        # Stochastic reversal entry
        entry_4 = (
            qtpylib.crossed_above(dataframe['stoch_k'], dataframe['stoch_d']) &
            (dataframe['stoch_k'] < 70) &
            (dataframe['volume_ratio'] > 0.3)
        )
        
        # Combined
        dataframe.loc[(entry_1 | entry_2 | entry_3 | entry_4), 'enter_long'] = 1
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # === EXIT CONDITIONS ===
        
        # RSI overbought
        exit_1 = (dataframe['rsi'] > 70) | (dataframe['rsi_fast'] < dataframe['rsi'])
        
        # MACD bearish cross
        exit_2 = qtpylib.crossed_below(dataframe['macd'], dataframe['macdsignal'])
        
        # Trend reversal
        exit_3 = ~dataframe['trend_up_5m']
        
        # Take profit at BB upper
        exit_4 = dataframe['close'] > dataframe['bb_middle']
        
        # Stochastic overbought reversal
        exit_5 = qtpylib.crossed_below(dataframe['stoch_k'], dataframe['stoch_d']) & (dataframe['stoch_k'] > 60)
        
        dataframe.loc[(exit_1 | exit_2 | exit_3 | exit_4 | exit_5) & (dataframe['volume'] > 0), 'exit_long'] = 1
        
        return dataframe
