# Freqtrade + Bitget Auto Trading Bot

Automated cryptocurrency trading bot using Freqtrade framework integrated with Bitget exchange.

## 🎯 Goal

Achieve **2% weekly profit** (after all fees) through automated trading.

## 📚 Documentation

- **[spec.md](./spec.md)** - Complete project specification (22KB)
- **[Freqtrade Docs](https://www.freqtrade.io)** - Official documentation

## 🚀 Quick Start

```bash
# 1. Clone and install Freqtrade
git clone https://github.com/freqtrade/freqtrade.git
cd freqtrade
pip install -e .
freqtrade create-userdir --userdir user_data

# 2. Configure Bitget API (edit config.bitget.json)

# 3. Download data
freqtrade download-data --exchange bitget --pairs BTC/USDT ETH/USDT --days 90 --timeframes 5m

# 4. Backtest all strategies
freqtrade backtesting --config config.json --strategy-list RSIStrategy MACDStrategy BBStrategy EMA_RSIStrategy Strategy005

# 5. Select best and optimize
freqtrade hyperopt --config config.json --strategy <BEST> --spaces all -e 500

# 6. Dry-run (1-2 weeks)
freqtrade trade --config config.bitget.json --strategy <BEST> --dry-run

# 7. Live trading
freqtrade trade --config config.bitget.json --strategy <BEST>
```

## 📊 Strategies

| Strategy | Description | Indicators |
|----------|-------------|-------------|
| **RSIStrategy** | RSI mean reversion | RSI (14, 7) |
| **MACDStrategy** | MACD momentum | MACD, EMA (50, 200) |
| **BBStrategy** | Bollinger Bands volatility | BB (20, 2) |
| **EMA_RSIStrategy** | Trend + oscillator combo | EMA (20, 50), RSI (14) |
| **Strategy005** | Multi-indicator official | RSI, MACD, Fisher, Stoch, SAR, SMA |

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

## 🔐 Security

- Never share API keys
- Never commit config files to git
- Always enable 2FA on Bitget
- Use read-only keys for testing

## 📖 Complete Guide

See **[spec.md](./spec.md)** for complete implementation details!
