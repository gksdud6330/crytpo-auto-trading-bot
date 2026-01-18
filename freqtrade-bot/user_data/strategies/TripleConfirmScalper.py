from freqtrade.strategy import IStrategy, merge_informative_pair
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class TripleConfirmScalper(IStrategy):
    """
    Triple-Confirmation Scalper
    1m timeframe + 5m trend + 1h direction confirmation
    Focused on PROFIT MAXIMIZATION, not stability
    """
    INTERFACE_VERSION = 3
    timeframe = '1m'
    startup_candle_count = 200
    
    # Aggressive ROI - quick profits
    minimal_roi = {
        "0": 0.015,      # 1.5% immediate profit target
        "3": 0.01,       # 1% after 3 minutes
        "5": 0.005,      # 0.5% after 5 minutes
        "15": 0          # Break even after 15 minutes
    }
    
    # Tight stoploss - but not too tight
    stoploss = -0.015
    
    # Trailing stop for big moves
    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.008
    trailing_only_offset_is_reached = True
    
    def informative_pairs(self):
        whitelist = []
        if self.dp:
            whitelist = self.dp.current_whitelist()
        if not whitelist:
            whitelist = self.config.get('exchange', {}).get('pair_whitelist', [])
        return [
            (pair, '5m') for pair in whitelist
        ] + [
            (pair, '1h') for pair in whitelist
        ]
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # === 1m Indicators (Entry) ===
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=7)
        
        # EMA for quick trend
        dataframe['ema_9'] = ta.EMA(dataframe, timeperiod=9)
        dataframe['ema_21'] = ta.EMA(dataframe, timeperiod=21)
        
        # MACD for momentum
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        
        # Bollinger Bands for volatility
        bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe['bb_lower'] = bollinger['lowerband']
        dataframe['bb_middle'] = bollinger['middleband']
        dataframe['bb_upper'] = bollinger['upperband']
        dataframe['bb_width'] = (dataframe['bb_upper'] - dataframe['bb_lower']) / dataframe['bb_middle']
        
        # Volume
        dataframe['volume_mean'] = dataframe['volume'].rolling(window=20).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_mean']
        
        # Stochastic for oversold/overbought
        stoch = ta.STOCH(dataframe)
        dataframe['stoch_k'] = stoch['slowk']
        dataframe['stoch_d'] = stoch['slowd']
        
        # === Merge 5m and 1h data ===
        if self.dp:
            # 5m timeframe
            informative_5m = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe='5m')
            informative_5m['ema_21_5m'] = ta.EMA(informative_5m, timeperiod=21)
            dataframe = merge_informative_pair(dataframe, informative_5m, '1m', '5m', ffill=True)
            
            # 1h timeframe
            informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe='1h')
            informative_1h['ema_21_1h'] = ta.EMA(informative_1h, timeperiod=21)
            dataframe = merge_informative_pair(dataframe, informative_1h, '1m', '1h', ffill=True)
        
        # Fallback if merge failed
        if 'ema_21_5m' not in dataframe:
            dataframe['ema_21_5m'] = dataframe['ema_21']
        if 'ema_21_1h' not in dataframe:
            dataframe['ema_21_1h'] = dataframe['ema_21']
        
        # Fill NaN
        dataframe['ema_21_5m'] = dataframe['ema_21_5m'].fillna(dataframe['ema_21'])
        dataframe['ema_21_1h'] = dataframe['ema_21_1h'].fillna(dataframe['ema_21'])
        
        # === Multi-timeframe Confirmation ===
        # 5m trend: price above/below 5m EMA
        dataframe['trend_5m'] = dataframe['close'] > dataframe['ema_21_5m']
        
        # 1h trend: price above/below 1h EMA
        dataframe['trend_1h'] = dataframe['close'] > dataframe['ema_21_1h']
        
        # 1m quick trend
        dataframe['trend_1m'] = dataframe['ema_9'] > dataframe['ema_21']
        
        # Strong uptrend: all timeframes aligned
        dataframe['strong_uptrend'] = (
            dataframe['trend_1h'] & 
            dataframe['trend_5m'] & 
            dataframe['trend_1m']
        )
        
        # Strong downtrend: all timeframes aligned
        dataframe['strong_downtrend'] = (
            (~dataframe['trend_1h']) & 
            (~dataframe['trend_5m']) & 
            (~dataframe['trend_1m'])
        )
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # === Aggressive Long Entries ===
        
        # Condition 1: Strong uptrend confirmation (all timeframes)
        cond_trend = dataframe['strong_uptrend']
        
        # Condition 2: RSI oversold or recovering
        cond_rsi = (
            (dataframe['rsi'] < 40) |  # Oversold
            (qtpylib.crossed_above(dataframe['rsi_fast'], dataframe['rsi']))  # RSI cross up
        )
        
        # Condition 3: MACD turning bullish
        cond_macd = (
            (qtpylib.crossed_above(dataframe['macd'], dataframe['macdsignal'])) |  # MACD cross
            (dataframe['macdhist'] > 0) & (dataframe['macdhist'].shift(1) <= 0)  # MACD turning positive
        )
        
        # Condition 4: Price near or below lower Bollinger Band (bounce opportunity)
        cond_bb = dataframe['close'] < dataframe['bb_middle']
        
        # Condition 5: Volume confirmation
        cond_volume = dataframe['volume'] > dataframe['volume_mean'] * 0.8
        
        # Combined entry signal
        dataframe.loc[
            cond_trend & cond_rsi & cond_macd & cond_bb & cond_volume,
            'enter_long'
        ] = 1
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # === Exit Conditions ===
        
        # Exit 1: RSI overbought
        cond_rsi_overbought = (
            (dataframe['rsi'] > 70) |
            (qtpylib.crossed_above(dataframe['rsi'], dataframe['rsi_fast']))  # RSI cross down
        )
        
        # Exit 2: MACD turning bearish
        cond_macd_bearish = (
            (qtpylib.crossed_below(dataframe['macd'], dataframe['macdsignal'])) |
            (dataframe['macdhist'] < 0) & (dataframe['macdhist'].shift(1) >= 0)
        )
        
        # Exit 3: Trend reversal on any timeframe
        cond_trend_reversal = (
            (~dataframe['trend_1m']) |  # 1m trend broken
            (~dataframe['trend_5m'])    # 5m trend broken
        )
        
        # Exit 4: Price hits upper Bollinger Band
        cond_bb_upper = dataframe['close'] > dataframe['bb_upper'] * 0.99
        
        # Combined exit signal
        dataframe.loc[
            (cond_rsi_overbought | cond_macd_bearish | cond_trend_reversal | cond_bb_upper) &
            (dataframe['volume'] > 0),
            'exit_long'
        ] = 1
        
        return dataframe
