# Freqtrade + Bitget Auto Trading Bot Specification

## 📋 Project Overview

### Goal
Build an automated cryptocurrency trading bot using Freqtrade framework integrated with Bitget exchange to achieve **2% weekly profit** (after all fees).

### Exchange
- **Exchange**: Bitget
- **Mode**: Spot Trading (can be extended to Futures)
- **API Required**: Read + Trade permissions (no Withdraw)

### Target Performance
- Weekly profit target: 2% (after fees)
- Win rate: 50-70%
- Maximum drawdown: -15%
- Average profit per trade: 1-2%

---

## 🏗️ Phase 1: Environment Setup

### Directory Structure
```
stock-market/
└── freqtrade-bot/
    ├── user_data/
    │   ├── strategies/
    │   │   ├── RSIStrategy.py
    │   │   ├── MACDStrategy.py
    │   │   ├── BBStrategy.py
    │   │   ├── EMA_RSIStrategy.py
    │   │   └── Strategy005.py
    │   ├── data/                    # Historical data for backtesting
    │   ├── backtest_results/        # Backtesting results
    │   └── notebooks/               # Analysis notebooks
    ├── config.json                  # Freqtrade configuration
    ├── config.bitget.json          # Bitget-specific config
    ├── spec.md                    # This specification document
    └── logs/                      # Trading logs
```

### Installation Methods

#### Option A: Docker (Recommended)
```bash
cd /Users/hy/Desktop/Coding/stock-market
mkdir freqtrade-bot
cd freqtrade-bot
git clone https://github.com/freqtrade/freqtrade.git
cd freqtrade
docker-compose up
```

#### Option B: Native Installation
```bash
cd /Users/hy/Desktop/Coding/stock-market/freqtrade-bot
git clone https://github.com/freqtrade/freqtrade.git
cd freqtrade
pip install -e .
freqtrade create-userdir --userdir user_data
```

### Required Dependencies
- freqtrade (core framework)
- pandas (data analysis)
- ta-lib (technical indicators)
- numpy (calculations)
- ccxt (exchange integration - included with freqtrade)

---

## 🔐 Phase 2: Bitget API Configuration

### 2.1 API Key Creation Steps
1. Login to Bitget account
2. Navigate to: Account → API Management → Create API
3. Set permissions:
   - ✅ **Read** (account balance queries)
   - ✅ **Trade** (order placement)
   - ❌ **Withdraw** (NOT required)

### 2.2 API Credentials (NEVER SHARE)
```
API Key: [Generated key]
Secret Key: [Only shown once - save immediately]
Passphrase: [Password set during API creation]
```

### 2.3 Configuration File Template

**config.bitget.json**
```json
{
  "exchange": {
    "name": "bitget",
    "key": "your_api_key",
    "secret": "your_api_secret",
    "password": "your_api_passphrase",
    "ccxt_config": {},
    "ccxt_async_config": {},
    "pair_whitelist": [
      "BTC/USDT",
      "ETH/USDT",
      "BNB/USDT",
      "SOL/USDT"
    ],
    "pair_blacklist": [
      ".*/_BTC",
      ".*/_ETH"
    ]
  },
  "stake_currency": "USDT",
  "stake_amount": 100,
  "dry_run": true,
  "dry_run_wallet": 1000,
  "db_url": "sqlite:///tradesv3.dryrun.sqlite",
  "max_open_trades": 3,
  "timeframe": "5m"
}
```

---

## 📊 Phase 3: Strategy Implementations

### Strategy Comparison Approach
**IMPORTANT**: The goal is NOT to use all strategies simultaneously. We will:
1. Implement 5 different strategies
2. Backtest all of them on the same period
3. Compare performance metrics
4. Select the best-performing strategy
5. Optimize the selected strategy using Hyperopt

---

### Strategy 1: RSI-Based Strategy

**File**: `user_data/strategies/RSIStrategy.py`

**Logic**: RSI overbought/oversold mean reversion

```python
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class RSIStrategy(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '5m'
    startup_candle_count = 30
    
    minimal_roi = {
        "0": 0.02,
        "30": 0.015,
        "60": 0.01,
        "120": 0.008
    }
    
    stoploss = -0.05
    
    buy_rsi_low = 30
    buy_rsi_high = 35
    sell_rsi = 70
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=7)
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['rsi'], self.buy_rsi_low)) &
                (dataframe['rsi_fast'] > dataframe['rsi']) &
                (dataframe['volume'] > 0)
            ),
            'enter_long'] = 1
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['rsi'], self.sell_rsi)) &
                (dataframe['volume'] > 0)
            ),
            'exit_long'] = 1
        return dataframe
```

---

### Strategy 2: MACD-Based Strategy

**File**: `user_data/strategies/MACDStrategy.py`

**Logic**: MACD momentum following

```python
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class MACDStrategy(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '5m'
    startup_candle_count = 30
    
    minimal_roi = {
        "0": 0.02,
        "30": 0.015,
        "60": 0.01,
        "120": 0.008
    }
    
    stoploss = -0.05
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['macd'], dataframe['macdsignal'])) &
                (dataframe['macdhist'] > 0) &
                (dataframe['close'] > dataframe['ema_50']) &
                (dataframe['volume'] > 0)
            ),
            'enter_long'] = 1
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_below(dataframe['macd'], dataframe['macdsignal'])) &
                (dataframe['volume'] > 0)
            ),
            'exit_long'] = 1
        return dataframe
```

---

### Strategy 3: Bollinger Bands Strategy

**File**: `user_data/strategies/BBStrategy.py`

**Logic**: Volatility-based mean reversion

```python
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class BBStrategy(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '5m'
    startup_candle_count = 30
    
    minimal_roi = {
        "0": 0.02,
        "30": 0.015,
        "60": 0.01,
        "120": 0.008
    }
    
    stoploss = -0.05
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2, nbdevdn=2)
        dataframe['bb_lower'] = bollinger['lowerband']
        dataframe['bb_middle'] = bollinger['middleband']
        dataframe['bb_upper'] = bollinger['upperband']
        
        dataframe['bb_width'] = (dataframe['bb_upper'] - dataframe['bb_lower']) / dataframe['bb_middle']
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['close'] < dataframe['bb_lower']) &
                (dataframe['close'] > dataframe['bb_lower'].shift(1)) &
                (dataframe['volume'] > 0)
            ),
            'enter_long'] = 1
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['close'] > dataframe['bb_upper']) &
                (dataframe['volume'] > 0)
            ),
            'exit_long'] = 1
        return dataframe
```

---

### Strategy 4: EMA + RSI Combined Strategy

**File**: `user_data/strategies/EMA_RSIStrategy.py`

**Logic**: Trend following with oscillator confirmation

```python
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class EMA_RSIStrategy(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '5m'
    startup_candle_count = 30
    
    minimal_roi = {
        "0": 0.02,
        "30": 0.015,
        "60": 0.01,
        "120": 0.008
    }
    
    stoploss = -0.05
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['ema_20'], dataframe['ema_50'])) &
                (dataframe['rsi'] < 40) &
                (dataframe['rsi'] > 30) &
                (dataframe['volume'] > 0)
            ),
            'enter_long'] = 1
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_below(dataframe['ema_20'], dataframe['ema_50'])) &
                (dataframe['rsi'] > 60) &
                (dataframe['volume'] > 0)
            ),
            'exit_long'] = 1
        return dataframe
```

---

### Strategy 5: Official Strategy005

**File**: `user_data/strategies/Strategy005.py`

**Source**: Download from Freqtrade official strategies repository

**Download Command**:
```bash
cd /Users/hy/Desktop/Coding/stock-market/freqtrade-bot/user_data/strategies
curl -O https://raw.githubusercontent.com/freqtrade/freqtrade-strategies/main/user_data/strategies/Strategy005.py
```

**Features**: Multi-indicator strategy with Hyperopt parameters
- RSI, MACD, Fisher RSI, Stochastic, SAR, SMA
- Configurable buy/sell parameters

---

## 🧪 Phase 4: Data Download

### Download Historical Data

```bash
cd /Users/hy/Desktop/Coding/stock-market/freqtrade-bot

# Download 90 days of historical data for backtesting
freqtrade download-data \
  --exchange bitget \
  --pairs BTC/USDT ETH/USDT BNB/USDT SOL/USDT \
  --days 90 \
  --timeframes 5m \
  --timerange 20241001-
```

### Verify Downloaded Data

```bash
freqtrade list-data --exchange bitget
```

---

## 📈 Phase 5: Backtesting All Strategies

### 5.1 Create Configuration File

**config.json**
```json
{
  "exchange": {
    "name": "bitget",
    "pair_whitelist": [
      "BTC/USDT",
      "ETH/USDT",
      "BNB/USDT",
      "SOL/USDT"
    ]
  },
  "stake_currency": "USDT",
  "stake_amount": 100,
  "dry_run_wallet": 1000,
  "max_open_trades": 3,
  "timeframe": "5m"
}
```

### 5.2 Run Comparative Backtesting

```bash
cd /Users/hy/Desktop/Coding/stock-market/freqtrade-bot

# Backtest all strategies simultaneously
freqtrade backtesting \
  --config config.json \
  --strategy-list RSIStrategy MACDStrategy BBStrategy EMA_RSIStrategy Strategy005 \
  --timeframe 5m \
  --timerange 20241001- \
  --export trades \
  --breakdown month
```

### 5.3 Expected Output Format

```
====================================================
Strategy Comparison Results (2024-10-01 ~ 2025-01-18)
====================================================

┌─────────────────┬──────────┬───────────┬─────────┬─────────────┬──────────────┐
│ Strategy        │ Trades   │ Win Rate  │ Avg Pft │ Total Profit│ Max Drawdown │
├─────────────────┼──────────┼───────────┼─────────┼─────────────┼──────────────┤
│ RSIStrategy     │   145    │  52.4%    │  1.24%  │  179.8 USDT │  -12.3%     │
│ MACDStrategy    │    98    │  58.2%    │  1.67%  │  163.7 USDT │  -10.5%     │
│ BBStrategy      │   187    │  48.7%    │  0.98%  │  183.3 USDT │  -14.8%     │
│ EMA_RSIStrategy │   112    │  61.3%    │  1.85%  │  207.2 USDT │   -8.9%     │
│ Strategy005     │   234    │  55.1%    │  1.12%  │  262.1 USDT │  -11.2%     │
└─────────────────┴──────────┴───────────┴─────────┴─────────────┴──────────────┘
```

---

### 5.4 Performance Evaluation Criteria

**Scoring System** (weighted for optimal performance):
1. **Total Profit (40%)** - Weekly 2% goal achievement
2. **Win Rate (25%)** - Stability and consistency
3. **Max Drawdown (15%)** - Risk management
4. **Average Profit (10%)** - Trade efficiency
5. **Trade Count (10%)** - Balance between too few/many

**Example Scoring**:
```
┌─────────────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
│ Strategy        │ Total P  │ Win R   │ Max DD  │ Avg P   │ Trades  │  Score  │
│                 │ (40)    │ (25)    │ (15)    │ (10)    │ (10)    │         │
├─────────────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
│ RSIStrategy     │   32    │   26     │   10     │    8     │    9     │  85     │
│ MACDStrategy    │   29    │   30     │   12     │    9     │    7     │  87     │
│ BBStrategy      │   33    │   24     │    8     │    6     │    7     │  78     │
│ EMA_RSIStrategy │   38    │   33     │   14     │   10     │    8     │ 103     │
│ Strategy005     │   48    │   28     │   11     │    7     │    6     │ 100     │
└─────────────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘

🏆 1st Place: EMA_RSIStrategy (103 points)
🥈 2nd Place: Strategy005 (100 points)
🥉 3rd Place: MACDStrategy (87 points)
```

---

## ⚙️ Phase 6: Strategy Optimization (Hyperopt)

### 6.1 Add Hyperopt Parameters to Strategy

Add to selected strategy (e.g., EMA_RSIStrategy.py):

```python
from freqtrade.strategy import IntParameter, CategoricalParameter

class EMA_RSIStrategy(IStrategy):
    # ... existing code ...
    
    # Hyperopt parameters
    buy_rsi = IntParameter(low=20, high=40, default=30, space='buy', optimize=True)
    sell_rsi = IntParameter(low=60, high=80, default=70, space='sell', optimize=True)
```

### 6.2 Run Hyperopt Optimization

```bash
# Optimize ROI and stoploss only
freqtrade hyperopt \
  --config config.json \
  --strategy EMA_RSIStrategy \
  --hyperopt-loss SharpeHyperOptLossDaily \
  --spaces roi stoploss trailing \
  -e 100

# Optimize buy/sell signals
freqtrade hyperopt \
  --config config.json \
  --strategy EMA_RSIStrategy \
  --hyperopt-loss SharpeHyperOptLossDaily \
  --spaces buy sell \
  -e 500

# Optimize all parameters
freqtrade hyperopt \
  --config config.json \
  --strategy EMA_RSIStrategy \
  --hyperopt-loss SharpeHyperOptLossDaily \
  --spaces all \
  -e 1000 \
  --jobs 4
```

### 6.3 Apply Optimized Parameters

Copy best parameters from hyperopt results to strategy file:

```python
# Example optimized parameters
buy_params = {
    'buy_rsi': 27,
    'buy_ema_threshold': 0.98
}

sell_params = {
    'sell_rsi': 73,
    'sell_ema_threshold': 1.02
}
```

---

## 🎯 Phase 7: Dry-Run Testing

### 7.1 Enable Dry-Run Mode

Update config.json:
```json
{
  "dry_run": true,
  "dry_run_wallet": 1000,
  "db_url": "sqlite:///tradesv3.dryrun.sqlite",
  "api_server": {
    "enabled": true,
    "listen_ip_address": "127.0.0.1",
    "listen_port": 8080,
    "verbosity": "error"
  }
}
```

### 7.2 Start Dry-Run

```bash
freqtrade trade \
  --config config.bitget.json \
  --strategy EMA_RSIStrategy
```

### 7.3 Monitoring

- **FreqUI**: http://localhost:8080
- **Logs**: `tail -f logs/freqtrade.log`
- **Telegram**: Enable notifications (optional)

### 7.4 Recommended Dry-Run Duration

**Minimum: 1-2 weeks**
- Verify strategy stability
- Check expected profit rate
- Identify any bugs or errors

---

## 💰 Phase 8: Live Trading

### 8.1 Enable Live Trading

Update config.bitget.json:
```json
{
  "dry_run": false,
  "db_url": "sqlite:///tradesv3.sqlite",
  "max_open_trades": 3,
  "stake_amount": "unlimited",
  "tradable_balance_ratio": 0.99,
  "telegram": {
    "enabled": true,
    "token": "your_telegram_bot_token",
    "chat_id": "your_telegram_chat_id"
  }
}
```

### 8.2 Risk Management Settings

```json
{
  "force_entry_enable": false,
  "position_stacking": false,
  "max_open_trades": 3,
  "stake_amount": 100,
  "use_max_hold_positions": false
}
```

### 8.3 Start Live Trading

```bash
freqtrade trade \
  --config config.bitget.json \
  --strategy EMA_RSIStrategy
```

---

## 📊 Phase 9: Performance Monitoring

### 9.1 Key Performance Metrics

**Daily/Monthly Profit Rate**
- Goal: 2% weekly profit
- Monitor: Check regularly

**Win Rate**
- Target: 50-70%

**Maximum Drawdown**
- Acceptable: -15% or less

**Average Profit per Trade**
- Target: 1-2%

### 9.2 Monitoring Commands

```bash
# View trade history
freqtrade show-trades --config config.bitget.json

# Performance report
freqtrade profit --config config.bitget.json

# Daily performance
freqtrade daily --config config.bitget.json

# Weekly performance
freqtrade profit --config config.bitget.json --days 7
```

### 9.3 Strategy Re-optimization

```bash
# Download new data monthly
freqtrade download-data --days 30 --prepend

# Re-optimize with new data
freqtrade hyperopt --config config.bitget.json --strategy EMA_RSIStrategy --spaces all -e 500
```

---

## ⚠️ Phase 10: Risk Management & Best Practices

### 10.1 Trading Fees

**Bitget Fees**:
- Maker: 0.1%
- Taker: 0.1%
- **Total Fee: 0.2% per round-trip**

**Important**: Include 0.2% fee in all profit calculations!

```bash
# Backtesting with fees
freqtrade backtesting \
  --config config.json \
  --strategy EMA_RSIStrategy \
  --fee 0.001
```

### 10.2 Risk Management Principles

```
✅ Maximum Exposure: Trade only 20-30% of total portfolio
✅ Single Coin Limit: No more than 10% per coin
✅ Stop Loss: Always respect -5% stop loss
✅ Market Monitoring: Pause bot during extreme volatility
✅ Diversification: Trade multiple pairs (3-5 coins)
```

### 10.3 Security Best Practices

- Never share API keys
- Never commit config.json to GitHub (add to .gitignore)
- Always enable 2FA on exchange
- Rotate API keys periodically
- Use read-only API keys for testing

---

## 🚀 Execution Summary

### Step-by-Step Commands

```bash
# 1. Setup
cd /Users/hy/Desktop/Coding/stock-market
mkdir freqtrade-bot
cd freqtrade-bot
git clone https://github.com/freqtrade/freqtrade.git
cd freqtrade
pip install -e .
freqtrade create-userdir --userdir user_data

# 2. Create strategies (5 files in user_data/strategies/)
# - RSIStrategy.py
# - MACDStrategy.py
# - BBStrategy.py
# - EMA_RSIStrategy.py
# - Strategy005.py (download)

# 3. Create config.json with Bitget API keys

# 4. Download data
freqtrade download-data \
  --exchange bitget \
  --pairs BTC/USDT ETH/USDT BNB/USDT SOL/USDT \
  --days 90 \
  --timeframes 5m

# 5. Backtest all strategies
freqtrade backtesting \
  --config config.json \
  --strategy-list RSIStrategy MACDStrategy BBStrategy EMA_RSIStrategy Strategy005 \
  --export trades

# 6. Analyze results and select best strategy

# 7. Hyperopt optimization
freqtrade hyperopt \
  --config config.json \
  --strategy <BEST_STRATEGY> \
  --spaces all \
  -e 500

# 8. Dry-run (1-2 weeks)
freqtrade trade \
  --config config.bitget.json \
  --strategy <BEST_STRATEGY> \
  --dry-run

# 9. Live trading
freqtrade trade \
  --config config.bitget.json \
  --strategy <BEST_STRATEGY>
```

---

## 📝 Additional Notes

### Expected Timeframes

- **Setup**: 1-2 hours
- **Data Download**: 30-60 minutes
- **Strategy Implementation**: 2-3 hours
- **Backtesting**: 30-60 minutes
- **Hyperopt**: 2-6 hours (depending on epochs)
- **Dry-Run**: 1-2 weeks
- **Live Trading**: Ongoing

### Success Indicators

✅ Consistent weekly profit > 2%
✅ Win rate 50-70%
✅ Max drawdown < -15%
✅ Smooth equity curve (no sharp drops)

### Failure Indicators

❌ Weekly profit < 0%
❌ Win rate < 40%
❌ Max drawdown > -20%
❌ Sharp equity drops

### Troubleshooting

**No trades executing**:
- Check whitelist pairs
- Verify timeframe has sufficient data
- Review strategy entry conditions

**All trades losing**:
- Review stop loss settings
- Check market conditions (trending vs ranging)
- Re-evaluate strategy parameters

**High drawdown**:
- Reduce stake amount
- Tighten stop loss
- Lower max open trades

---

## 📚 References

- [Freqtrade Official Docs](https://www.freqtrade.io)
- [Freqtrade GitHub](https://github.com/freqtrade/freqtrade)
- [Freqtrade Strategies](https://github.com/freqtrade/freqtrade-strategies)
- [Bitget API Docs](https://www.bitget.com/api-doc)
- [Bitget Exchange Notes](https://www.freqtrade.io/en/stable/exchanges)

---

## 📞 Support & Next Steps

For implementation assistance:
1. Review this spec.md
2. Execute Phase 1-10 sequentially
3. Monitor results at each phase
4. Adjust parameters based on performance
5. Seek help if stuck

**Ready to implement? Start with Phase 1: Environment Setup!** 🚀
