from freqtrade.strategy import IStrategy
from pandas import DataFrame
import numpy as np
import talib.abstract as ta

class BuyTheDip(IStrategy):
    """
    Buy The Dip - Simple and Profitable
    Only buy when price drops significantly from recent high
    Sell when price recovers to reasonable profit
    """
    INTERFACE_VERSION = 3
    timeframe = '1h'  # Longer timeframe for swing trading
    startup_candle_count = 200
    
    # Hold longer for bigger moves
    minimal_roi = {
        "0": 0.03,      # 3% take profit
        "24": 0.02,     # 2% after 24 hours
        "72": 0.015,    # 1.5% after 3 days
        "168": 0.01,    # 1% after 1 week
        "336": 0        # Break even after 2 weeks
    }
    
    stoploss = -0.05  # 5% stop loss
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # === Price-based indicators ===
        dataframe['close'] = dataframe['close'].astype(float)
        
        # Rolling highs and lows
        dataframe['highest_20'] = dataframe['high'].rolling(window=20).max()
        dataframe['lowest_20'] = dataframe['low'].rolling(window=20).min()
        
        # Distance from recent high (dip percentage)
        dataframe['dip_pct'] = (dataframe['highest_20'] - dataframe['close']) / dataframe['highest_20']
        
        # Distance from recent low
        dataframe['rise_pct'] = (dataframe['close'] - dataframe['lowest_20']) / dataframe['lowest_20']
        
        # === RSI ===
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        
        # === EMA ===
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        
        # === Volume confirmation ===
        dataframe['volume'] = dataframe['volume'].astype(float)
        dataframe['volume_mean'] = dataframe['volume'].rolling(window=20).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_mean']
        
        # === Trend ===
        dataframe['uptrend'] = dataframe['close'] > dataframe['ema_200']
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # === ENTRY: Buy when significant dip in uptrend ===
        
        # Condition 1: Uptrend (price above 200 EMA)
        cond_trend = dataframe['uptrend']
        
        # Condition 2: Significant dip (10-25% from high)
        cond_dip = (dataframe['dip_pct'] > 0.08) & (dataframe['dip_pct'] < 0.30)
        
        # Condition 3: RSI not too oversold (avoid bottom catching)
        cond_rsi = (dataframe['rsi'] > 30) & (dataframe['rsi'] < 55)
        
        # Condition 4: Near support (within 8% of recent low)
        cond_support = dataframe['close'] < dataframe['lowest_20'] * 1.08
        
        # Combined entry
        dataframe.loc[
            cond_trend & cond_dip & cond_rsi & cond_support,
            'enter_long'
        ] = 1
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # === EXIT: Sell when recovered or trend reversal ===
        
        # Exit 1: RSI overbought
        cond_overbought = dataframe['rsi'] > 70
        
        # Exit 2: Significant rise from low (potential top)
        cond_rise = dataframe['rise_pct'] > 0.15
        
        # Exit 3: Trend reversal (price below 50 EMA)
        cond_reversal = dataframe['close'] < dataframe['ema_50']
        
        # Exit 4: Massive volume spike (distribution)
        cond_volume = (dataframe['volume_ratio'] > 3) & (dataframe['rsi'] > 60)
        
        dataframe.loc[
            (cond_overbought | cond_rise | cond_reversal | cond_volume) &
            (dataframe['volume'] > 0),
            'exit_long'
        ] = 1
        
        return dataframe
