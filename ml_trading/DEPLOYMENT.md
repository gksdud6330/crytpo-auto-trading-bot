# 🚀 ML Trading Bot - Deployment Guide

Automated cryptocurrency trading system using ML predictions with Freqtrade.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Running the System](#running-the-system)
6. [Monitoring](#monitoring)
7. [Troubleshooting](#troubleshooting)
8. [Production Checklist](#production-checklist)

---

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    ML TRADING BOT ARCHITECTURE                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│   │   Bitget    │────▶│  Data       │────▶│     ML      │       │
│   │   Exchange  │     │  Collector  │     │   Models    │       │
│   └─────────────┘     └─────────────┘     └──────┬──────┘       │
│                                                  │               │
│                                                  ▼               │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│   │   Freqtrade │◀────│   Signal    │◀────│  Predictor  │       │
│ │  Bot         │     │   Generator  │     │             │       │
│   └──────┬──────┘     └─────────────┘     └─────────────┘       │
│          │                                                       │
│          ▼                                                       │
│   ┌─────────────┐                                                │
│   │   Execute   │     📊 Telegram Alerts                        │
│   │   Trades    │     📈 Performance Reports                    │
│   └─────────────┘     🔔 Stop-Loss Alerts                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Strategy Performance

| Metric | Value |
|--------|-------|
| Total Trades | 45 |
| Win Rate | 51.1% |
| Avg Return | 2.39% per trade |
| Best Pair | ETH/USDT (+84.08%) |
| Max Drawdown | ~5% |

---

## Prerequisites

### System Requirements

```bash
# Minimum
- macOS / Linux / Ubuntu 20.04+
- Python 3.10+
- 8GB RAM
- 10GB Disk Space

# Recommended
- 16GB+ RAM
- 4+ CPU cores
- SSD Storage
```

### Required Accounts

1. **Bitget Account**
   - Create API key with trade permissions
   - Enable 2FA

2. **Optional: Telegram Bot** (for alerts)
   - Create bot via @BotFather
   - Get bot token and chat ID

---

## Installation

### Step 1: Clone and Setup

```bash
# Navigate to project directory
cd /Users/hy/Desktop/Coding/stock-market

# Clone Freqtrade (if not already done)
git clone https://github.com/freqtrade/freqtrade.git
cd freqtrade

# Install Freqtrade
pip install -e .

# Create user directory
freqtrade create-userdir --userdir user_data
```

### Step 2: Install ML Dependencies

```bash
# Navigate to ML trading module
cd ../ml_trading

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Install Optuna for hyperparameter tuning
pip install optuna
```

### Step 3: Verify Installation

```bash
# Test ML modules
cd src
python3 -c "
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
print('✅ All ML dependencies installed successfully')
"
```

---

## Configuration

### 1. Exchange API Configuration

Edit `freqtrade-bot/config.bitget.json`:

```json
{
    "max_open_trades": 3,
    "stake_amount": 100,
    "stake_currency": "USDT",
    "tradable_asset_pairs": {
        "BTC/USDT": {"stop_loss": -0.05, "timeframe": "4h"},
        "ETH/USDT": {"stop_loss": -0.05, "timeframe": "4h"},
        "SOL/USDT": {"stop_loss": -0.05, "timeframe": "4h"},
        "BNB/USDT": {"stop_loss": -0.05, "timeframe": "4h"}
    },
    "exchange": {
        "name": "bitget",
        "key": "YOUR_API_KEY_HERE",
        "secret": "YOUR_API_SECRET_HERE",
        "password": "",
        "ccxt_config": {
            "enableRateLimit": true
        }
    },
    "telegram": {
        "enabled": true,
        "token": "YOUR_TELEGRAM_BOT_TOKEN",
        "chat_id": "YOUR_CHAT_ID"
    },
    "dry_run": true,
    "api_server": {
        "enabled": true,
        "listen_ip": "127.0.0.1",
        "listen_port": 8080
    }
}
```

### 2. ML Model Configuration

Edit `src/predictor.py`:

```python
# Trading parameters
ML_CONFIDENCE_THRESHOLD = 0.55  # Minimum confidence for BUY signal
USE_ML_FILTER = True  # Only trade when ML predicts UP
MIN_ADX = 15  # Minimum ADX for trend confirmation
```

### 3. Strategy Parameters

Edit `freqtrade-bot/user_data/strategies/MLStrategy.py`:

```python
class MLStrategy(IStrategy):
    minimal_roi = {
        "0": 0.03,      # 3% profit target
        "60": 0.02,     # 2% after 1 hour
        "180": 0.015,   # 1.5% after 3 hours
    }
    
    stoploss = -0.05   # 5% stop loss
    timeframe = '4h'
```

---

## Running the System

### Option 1: Dry Run (Recommended First)

```bash
# Terminal 1: Start ML prediction service
cd ml_trading
source venv/bin/activate
python3 src/predictor.py

# Terminal 2: Start Freqtrade in dry-run mode
cd freqtrade
freqtrade trade \
    --config config.bitget.json \
    --strategy MLStrategy \
    --dry-run
```

### Option 2: Paper Trading

```bash
# Same as dry run but with realistic order execution
freqtrade trade \
    --config config.bitget.json \
    --strategy MLStrategy \
    --dry-run \
    --force_entry
```

### Option 3: Live Trading

```bash
# ⚠️ WARNING: Real money at risk
freqtrade trade \
    --config config.bitget.json \
    --strategy MLStrategy
```

---

## Monitoring

### 1. Check Predictions

```bash
# Run predictor manually
cd ml_trading
python3 src/predictor.py
```

Expected output:
```
============================================================
CRYPTO ML TRADING SIGNALS
============================================================

🟢 BUY - SOL/USDT @ $144.29 (70.5% confidence)
🟡 HOLD - BTC/USDT @ $95,287 (65.1% confidence)
🟡 HOLD - ETH/USDT @ $3,316 (54.8% confidence)
```

### 2. Check Freqtrade Status

```bash
# API endpoint (when server is running)
curl http://localhost:8080/api/v1/status

# Or use Telegram bot
/status
```

### 3. View Logs

```bash
# Freqtrade logs
tail -f freqtrade.log

# ML predictor logs
tail -f ml_trading/predictor.log
```

### 4. Telegram Alerts

Send these commands to your Telegram bot:
- `/status` - Current trades
- `/profit` - Profit summary
`/balance` - Account balance
`/daily` - Daily performance

---

## Retraining Models

### Weekly Retraining (Recommended)

```bash
cd ml_trading

# 1. Collect latest data
python3 src/data_collector.py

# 2. Run parallel model evaluation
python3 src/parallel_model_runner.py

# 3. Check best model
cat models/best_model_config.json

# 4. Copy best model to Freqtrade
cp models/xgboost_optimized_4h.joblib ../freqtrade-bot/user_data/strategies/
```

### Monthly Full Retraining

```bash
# 1. Run enhanced pipeline
python3 src/enhanced_data_pipeline.py

# 2. Hyperparameter tuning
python3 src/hyperparameter_optimizer.py

# 3. Full parallel evaluation
python3 src/parallel_model_runner.py
```

---

## Troubleshooting

### Common Issues

#### 1. "Model not found" Error

```bash
# Solution: Ensure model files exist
ls -la ml_trading/models/

# If missing, retrain
python3 src/parallel_model_runner.py
```

#### 2. API Rate Limiting

```bash
# Solution: Increase rate limit delay
# In predictor.py
EXCHANGE_RATE_LIMIT = 1000  # ms
```

#### 3. Memory Issues

```bash
# Solution: Reduce batch size or use smaller model
# In parallel_model_runner.py
n_workers = 2  # Instead of 4
```

#### 4. Wrong Predictions

```bash
# Solution: Retrain with fresh data
python3 src/data_collector.py
python3 src/parallel_model_runner.py
```

### Log Locations

| Component | Log File |
|-----------|----------|
| Freqtrade | `./freqtrade.log` |
| ML Predictor | `./ml_trading/predictor.log` |
| Backtests | `./ml_trading/reports/` |

---

## Production Checklist

Before going live with real money:

- [ ] **Security**
  - [ ] API keys have trade permissions only (no withdrawal)
  - [ ] 2FA enabled on exchange account
  - [ ] Strong password for server
  - [ ] Firewall configured (only needed ports)

- [ ] **Testing**
  - [ ] Dry run completed for 1+ week
  - [ ] Paper trading completed for 1+ week
  - [ ] All strategy parameters tested
  - [ ] Stop-loss tested

- [ ] **Monitoring**
  - [ ] Telegram alerts configured
  - [ ] Email notifications set up
  - [ ] Log rotation configured
  - [ ] Disk space monitoring

- [ ] **Backup**
  - [ ] Model files backed up
  - [ ] Configuration files backed up
  - [ ] Recovery procedure documented

---

## File Structure

```
stock-market/
├── ml_trading/
│   ├── src/
│   │   ├── data_collector.py         # Fetch OHLCV data
│   │   ├── predictor.py              # Generate signals
│   │   ├── model_trainer.py          # Train models
│   │   ├── hyperparameter_optimizer.py  # Optuna tuning
│   │   ├── parallel_model_runner.py  # Test multiple models
│   │   ├── ensemble_predictor.py     # Ensemble methods
│   │   ├── backtest_ml_strategy.py   # Backtesting
│   │   ├── onchain_data.py           # On-chain metrics
│   │   ├── sentiment_data.py         # Sentiment data
│   │   └── enhanced_data_pipeline.py # Full pipeline
│   ├── models/                       # Trained models
│   ├── data/                         # Market data
│   └── reports/                      # Backtest reports
│
├── freqtrade-bot/
│   ├── config.bitget.json            # Exchange config
│   └── user_data/
│       └── strategies/
│           └── MLStrategy.py         # Freqtrade strategy
│
└── README.md
```

---

## Telegram Bot Integration

Get real-time trading signals and notifications on Telegram.

### Setup

```bash
cd ml_trading/scripts

# Run setup script
./setup_telegram.sh

# Or manually:
export TELEGRAM_BOT_TOKEN="your_bot_token"
export TELEGRAM_CHAT_ID="your_chat_id"

# Test connection
python3 ../src/telegram_bot.py --test
```

### Bot Commands

| Command | Description |
|---------|-------------|
| `/start` | Start the bot |
| `/help` | Show help |
| `/status` | Current positions |
| `/signals` | ML trading signals |
| `/profit` | Profit summary |
| `/balance` | Account balance |
| `/daily` | Daily performance |
| `/models` | ML model status |
| `/test` | Test connection |

### Running the Bot

```bash
# Terminal 1: Start Telegram bot
cd ml_trading
source .env  # Load credentials
python3 src/telegram_bot.py

# Or run in background
nohup python3 src/telegram_bot.py > bot.log 2>&1 &

# Check if running
ps aux | grep telegram_bot
```

### Auto-start with Systemd

```bash
# Copy service file
sudo cp scripts/ml-trading-bot.service /etc/systemd/system/

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable ml-trading-bot
sudo systemctl start ml-trading-bot

# Check status
sudo systemctl status ml-trading-bot

# View logs
sudo journalctl -u ml-trading-bot -f
```

### Telegram Bot Features

```
┌─────────────────────────────────────┐
│     🤖 ML Trading Bot               │
├─────────────────────────────────────┤
│                                     │
│  🎯 Real-time Signals               │
│     - BUY/SELL recommendations      │
│     - Confidence scores             │
│     - Price alerts                  │
│                                     │
│  📊 Performance Updates             │
│     - Daily P&L summary             │
│     - Win rate statistics           │
│     - Trade history                 │
│                                     │
│  🔔 Critical Alerts                 │
│     - Stop-loss triggered           │
│     - Profit targets hit            │
│     - Model errors                  │
│                                     │
│  📈 Interactive Commands            │
│     - /status - Positions           │
│     - /profit - Performance         │
│     - /signals - ML predictions     │
│                                     │
└─────────────────────────────────────┘
```

### Creating a Telegram Bot

1. **Open Telegram** and search for @BotFather
2. **Send /newbot** to create a new bot
3. **Follow the instructions**:
   - Bot name: "ML Trading Bot"
   - Bot username: something ending with "bot"
4. **Copy the token** (format: `123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11`)
5. **Get your Chat ID**:
   - Search for @userinfobot
   - Send /start
   - Copy your ID number

### Sample Notifications

**Buy Signal:**
```
🟢 BUY SIGNAL

Pair: ETH/USDT
Price: $3,316.25
Confidence: 70.5%

Reason: ML model predicts UP with 70.5% confidence
```

**Daily Summary:**
```
📊 Daily Summary - 2026-01-18

Trades: 2
Win Rate: 100%
P&L: +$42.50
ROI: +0.42%

Best Performer: ETH/USDT
```

---

## Quick Commands Reference

```bash
# Daily prediction check
cd ml_trading && python3 src/predictor.py

# Weekly model retrain
cd ml_trading && python3 src/parallel_model_runner.py

# Check open trades
curl http://localhost:8080/api/v1/status

# Force reload strategy
freqtrade trade --config config.json --strategy MLStrategy --reload

# View profit
freqtrade profit --config config.json
```

---

## Support

- **Issues**: Check `reports/` for backtest reports
- **Logs**: Check log files for errors
- **Model Performance**: Check `reports/parallel_model_report_*.txt`

---

*Last Updated: 2026-01-18*
*Version: 1.0.0*
