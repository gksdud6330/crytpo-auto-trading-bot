# Freqtrade + Bitget Auto Trading Bot

Automated cryptocurrency trading bot using Freqtrade framework integrated with Bitget exchange.

## 🎯 Goal

Achieve **2% weekly profit** (after all fees) through automated trading.

## 📚 Documentation

- **[spec.md](./spec.md)** - Complete project specification
- **[strategies/](./user_data/strategies/)** - Trading strategy implementations
- **[Freqtrade Docs](https://www.freqtrade.io)** - Official documentation

## 🚀 Quick Start

```bash
# 1. Clone repository
cd /Users/hy/Desktop/Coding/stock-market/freqtrade-bot
git clone https://github.com/freqtrade/freqtrade.git

# 2. Install Freqtrade
cd freqtrade
pip install -e .
freqtrade create-userdir --userdir user_data

# 3. Configure Bitget API
# Edit config.bitget.json with your API credentials

# 4. Download historical data
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

# 6. Select best strategy and optimize
freqtrade hyperopt \
  --config config.json \
  --strategy <BEST_STRATEGY> \
  --spaces all \
  -e 500

# 7. Dry-run test (1-2 weeks)
freqtrade trade \
  --config config.bitget.json \
  --strategy <BEST_STRATEGY> \
  --dry-run

# 8. Live trading
freqtrade trade \
  --config config.bitget.json \
  --strategy <BEST_STRATEGY>
```

## 📊 Strategies

| Strategy | Description | Indicators |
|----------|-------------|-------------|
| **RSIStrategy** | RSI mean reversion | RSI (14, 7) |
| **MACDStrategy** | MACD momentum | MACD, EMA (50, 200) |
| **BBStrategy** | Bollinger Bands volatility | BB (20, 2) |
| **EMA_RSIStrategy** | Trend + oscillator combo | EMA (20, 50), RSI (14) |
| **Strategy005** | Multi-indicator official | RSI, MACD, Fisher, Stoch, SAR, SMA |

## 📁 Project Structure

```
freqtrade-bot/
├── user_data/
│   ├── strategies/           # Trading strategies (5 files)
│   ├── data/                # Historical OHLCV data
│   ├── backtest_results/    # Backtesting results
│   └── notebooks/          # Analysis notebooks
├── config.json              # General configuration
├── config.bitget.json      # Bitget-specific config
├── spec.md                # Complete specification
└── README.md              # This file
```

## 🎯 Performance Targets

- **Weekly Profit**: 2% (after fees)
- **Win Rate**: 50-70%
- **Max Drawdown**: < -15%
- **Avg Profit/Trade**: 1-2%

## ⚠️ Risk Management

- Trade only 20-30% of total portfolio
- Max 10% exposure per coin
- Always respect stop-loss (-5%)
- Monitor daily performance
- Pause during extreme volatility

## 🔐 Security

- Never share API keys
- Never commit config files to git
- Always enable 2FA on Bitget
- Use read-only keys for testing
- Rotate API keys periodically

## 📖 Workflow

1. **Setup** → Install Freqtrade and configure
2. **Data** → Download historical data
3. **Backtest** → Compare all 5 strategies
4. **Optimize** → Hyperopt best strategy
5. **Dry-Run** → Test for 1-2 weeks
6. **Live** → Deploy with real capital

## 📚 Reference Documents

- [spec.md](./spec.md) - Detailed implementation guide
- [Freqtrade Documentation](https://www.freqtrade.io)
- [Bitget API Docs](https://www.bitget.com/api-doc)
- [Freqtrade Strategies](https://github.com/freqtrade/freqtrade-strategies)

---

**Ready to start? See [spec.md](./spec.md) for complete instructions!** 🚀
