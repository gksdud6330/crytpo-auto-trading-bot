# Freqtrade + Bitget Auto Trading Bot (ML Enhanced)

Automated cryptocurrency trading bot using Freqtrade framework integrated with Bitget exchange, enhanced with ML/AI predictions.

## 🎯 Goal

Achieve **2% weekly profit** (after all fees) through automated trading with ML signals.

## 🚀 Quick Start

### ML Trading (Recommended)

```bash
# 1. Install ML dependencies
cd ml_trading
pip install -r requirements.txt

# 2. Collect data
python3 src/data_collector.py

# 3. Train models
python3 src/parallel_model_runner.py

# 4. Get signals
python3 src/predictor.py
```

### Freqtrade

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
```

## 📁 Project Structure

```
stock-market/
├── ml_trading/                    # ML Trading Module
│   ├── src/
│   │   ├── data_collector.py      # OHLCV data collection
│   │   ├── predictor.py           # Signal generation
│   │   ├── model_trainer.py       # Model training
│   │   ├── parallel_model_runner.py  # Multi-model evaluation
│   │   ├── hyperparameter_optimizer.py  # Optuna tuning
│   │   ├── backtest_ml_strategy.py  # ML strategy backtesting
│   │   ├── telegram_bot.py        # Telegram notifications
│   │   ├── onchain_data.py        # On-chain metrics
│   │   ├── sentiment_data.py      # Sentiment analysis
│   │   └── enhanced_data_pipeline.py  # Full pipeline (60 features)
│   ├── models/                    # Trained models (excluded from git)
│   ├── data/                      # Market data (excluded from git)
│   └── scripts/
│       └── setup_telegram.sh      # Telegram setup script
│
├── freqtrade-bot/                 # Freqtrade Bot
│   └── user_data/strategies/
│       ├── MLStrategy.py          # ML-based trading strategy
│       ├── RSIStrategy.py
│       ├── MACDStrategy.py
│       ├── BBStrategy.py
│       ├── Strategy005.py         # Best performing
│       └── ...
│
├── DEPLOYMENT.md                  # Deployment guide (English)
└── SETUP_KOREAN.md                # Setup guide (Korean)
```

## 📊 ML Strategy Performance (Backtest)

| Metric | Value |
|--------|-------|
| Total Trades | 45 |
| Win Rate | 51.1% |
| Avg Return | +2.39% per trade |
| Best Pair | ETH/USDT (+84.08%) |

## 🧠 ML Models

| Model | F1 Score | Recall | Use Case |
|-------|----------|--------|----------|
| XGBoost (Optimized) | 0.3007 | 58.85% | Primary |
| LightGBM (Optimized) | 0.3133 | 17.22% | Ensemble |
| RandomForest | 0.2770 | 47.85% | Ensemble |

## 📱 Telegram Bot

Commands:
- `/start` - Start bot
- `/signals` - ML trading signals
- `/status` - Current positions
- `/profit` - Profit summary
- `/models` - ML model status

Setup:
```bash
cd ml_trading/scripts
./setup_telegram.sh
```

## 📖 Documentation

- **[DEPLOYMENT.md](./DEPLOYMENT.md)** - Full deployment guide
- **[SETUP_KOREAN.md](./SETUP_KOREAN.md)** - Korean setup guide
- **[spec.md](./freqtrade-bot/spec.md)** - Complete specification

## ⚠️ Risk Management

- Trade only 20-30% of total portfolio
- Max 10% exposure per coin
- Always respect stop-loss (-5%)
- Monitor daily performance
- Use dry-run before live trading

## 🔐 Security

- Never share API keys
- Never commit config files to git (use .gitignore)
- Always enable 2FA on Bitget
- Use read-only keys for testing

## 📝 Files Excluded from Git

- `config.json`, `config.bitget.json` - API keys
- `.env` - Environment variables
- `ml_trading/data/` - Market data (large files)
- `ml_trading/models/` - Trained models (regeneratable)
- `*.log` - Log files
- `*.sqlite` - Database files
