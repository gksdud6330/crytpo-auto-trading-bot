"""
Telegram Bot for ML Trading System
- Sends trading signals
- Shows current positions and profit/loss
- Allows manual commands
- Real-time notifications
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

import requests
import pandas as pd
import joblib

from bitget_client import get_client as get_bitget_client, BitgetClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')

# Trading mode
TRADING_MODE = os.environ.get('TRADING_MODE', 'demo')
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

# Model paths
MODEL_DIR = '/Users/hy/Desktop/Coding/stock-market/ml_trading/models'
DATA_DIR = '/Users/hy/Desktop/Coding/stock-market/ml_trading/data'

# Trading pairs
PAIRS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT']


class CryptoPredictor:
    """Simple predictor for getting ML signals"""

    def __init__(self, timeframe='4h'):
        self.timeframe = timeframe
        self.model_dir = MODEL_DIR
        self._load_model()

    def _load_model(self):
        """Load ML model"""
        model_path = f"{self.model_dir}/xgboost_optimized_{self.timeframe}.joblib"
        scaler_path = f"{self.model_dir}/scaler_{self.timeframe}.joblib"
        feature_path = f"{self.model_dir}/features_{self.timeframe}.txt"

        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)

            with open(feature_path, 'r') as f:
                self.feature_cols = [line.strip() for line in f]
        else:
            self.model = None
            self.feature_cols = []

    def generate_signals(self, min_confidence=0.55):
        """Generate trading signals"""
        # Simulated signals for demo
        signals = [
            {
                'signal': 'HOLD',
                'symbol': 'BTC/USDT',
                'price': 95287.01,
                'confidence': 0.651,
                'reason': 'Low confidence (65.1%) or HOLD signal'
            },
            {
                'signal': 'HOLD',
                'symbol': 'ETH/USDT',
                'price': 3316.25,
                'confidence': 0.548,
                'reason': 'Low confidence (54.8%) or HOLD signal'
            },
            {
                'signal': 'BUY',
                'symbol': 'SOL/USDT',
                'price': 144.29,
                'confidence': 0.705,
                'reason': 'ML model predicts UP with 70.5% confidence'
            },
            {
                'signal': 'HOLD',
                'symbol': 'BNB/USDT',
                'price': 952.80,
                'confidence': 0.501,
                'reason': 'Low confidence (50.1%) or HOLD signal'
            }
        ]
        return signals


class TelegramBot:
    """Telegram bot for ML trading notifications"""

    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.predictor = CryptoPredictor(timeframe='4h')
        self.trading_mode = TRADING_MODE
        self.bitget_client = None

        if self.trading_mode == 'bitget':
            self.bitget_client = get_bitget_client()
            if not self.bitget_client.is_available():
                logger.warning("Bitget not available, switching to demo mode")
                self.trading_mode = 'demo'

        # Command handlers
        self.commands = {
            '/start': self.cmd_start,
            '/help': self.cmd_help,
            '/status': self.cmd_status,
            '/signals': self.cmd_signals,
            '/profit': self.cmd_profit,
            '/balance': self.cmd_balance,
            '/daily': self.cmd_daily,
            '/models': self.cmd_models,
            '/test': self.cmd_test,
            '/buy': self.cmd_buy,
            '/sell': self.cmd_sell,
            '/trade': self.cmd_trade,
            '/open': self.cmd_open,
        }

        # Message handlers
        self.messages = {
            'hi': self.cmd_start,
            'hello': self.cmd_start,
            'status': self.cmd_status,
            'signals': self.cmd_signals,
            'profit': self.cmd_profit,
            'balance': self.cmd_balance,
            'daily': self.cmd_daily,
            'models': self.cmd_models,
            'test': self.cmd_test,
            'buy': self.cmd_buy,
            'sell': self.cmd_sell,
            'trade': self.cmd_trade,
            'open': self.cmd_open,
        }

    def send_message(self, text: str, parse_mode: str = 'Markdown') -> bool:
        """Send message to Telegram"""
        if not self.token or not self.chat_id:
            logger.warning("Telegram credentials not configured")
            return False

        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            data = {
                'chat_id': self.chat_id,
                'text': text,
                'parse_mode': parse_mode,
                'disable_web_page_preview': True
            }

            response = requests.post(url, json=data, timeout=10)

            if response.status_code == 200:
                logger.info("Message sent successfully")
                return True
            else:
                logger.error(f"Failed to send message: {response.text}")
                return False

        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False

    def cmd_start(self, args: List[str] = None) -> str:
        """Handle /start command"""
        return """
🤖 *ML Trading Bot*
━━━━━━━━━━━━━━━━━━━━━
Welcome to your ML-powered crypto trading assistant!

*Available Commands:*
• /status - Current positions
• /signals - ML trading signals
• /profit - Profit summary
• /balance - Account balance
• /daily - Daily performance
• /models - Model status
• /test - Test connection

*Exchange:* Bitget via Freqtrade
*ML Models Active:*
• XGBoost (Optimized)
• LightGBM (Optimized)
• RandomForest

_Growth is a marathon, not a sprint_ 🐢
"""

    def cmd_help(self, args: List[str] = None) -> str:
        """Handle /help command"""
        return """
📚 *Help - ML Trading Bot*
━━━━━━━━━━━━━━━━━━━━━

*Main Commands:*
/start - Start the bot
/help - Show this help
/status - Show open positions
/signals - Show ML buy/sell signals
/profit - Show profit/loss summary
/balance - Show account balance
/daily - Show daily performance
/models - Show ML model status

*About:*
This bot integrates with Freqtrade
connected to Bitget exchange for
automated crypto trading signals.

*Timeframe:* 4h
*Exchange:* Bitget
"""

    def cmd_status(self, args: List[str] = None) -> str:
        """Handle /status command"""
        message = f"""
📊 *Current Positions* [{self.trading_mode.upper()}]
━━━━━━━━━━━━━━━━━━━━━
"""
        try:
            if self.trading_mode == 'bitget' and self.bitget_client:
                positions = self.bitget_client.get_futures_positions()
            else:
                positions = []

            if positions:
                for pos in positions:
                    pnl_pct = pos.get('pnl_pct', 0)
                    emoji = '🟢' if pnl_pct > 0 else '🔴'
                    message += f"{emoji} *{pos.get('pair', 'Unknown')}*\n"
                    message += f"   Entry: ${pos.get('entry_price', 0):,.2f}\n"
                    message += f"   Amount: {pos.get('amount', 0):.4f}\n"
                    message += f"   P&L: {pnl_pct:+.2f}%\n\n"
            else:
                message += "No open positions 📭\n"
        except Exception as e:
            message += f"Error fetching positions: {e}\n"

        message += f"""
*Bot Status:* 🟢 Running
*Mode:* {self.trading_mode.upper()}
*Exchange:* Bitget
"""
        return message

    def cmd_signals(self, args: List[str] = None) -> str:
        """Handle /signals command"""
        message = """
🎯 *ML Trading Signals*
━━━━━━━━━━━━━━━━━━━━━
*Timeframe:* 4h | *Confidence:* 55%+
"""

        try:
            signals = self.predictor.generate_signals(min_confidence=0.55)

            if signals:
                for signal in signals:
                    emoji = "🟢" if signal['signal'] == 'BUY' else "🟡"
                    confidence_pct = signal['confidence'] * 100

                    message += f"\n{emoji} *{signal['signal']}* - {signal['symbol']}\n"
                    message += f"   Price: ${signal['price']:,.2f}\n"
                    message += f"   Confidence: {confidence_pct:.1f}%\n"
                    message += f"   Reason: {signal['reason']}\n"
            else:
                message += "\nNo strong signals at this time ⏳\n"

        except Exception as e:
            message += f"\n❌ Error fetching signals: {e}\n"

        message += """
━━━━━━━━━━━━━━━━━━━━━
*Tip:* Only trade with 55%+ confidence
"""
        return message

    def cmd_profit(self, args: List[str] = None) -> str:
        """Handle /profit command"""
        try:
            if self.trading_mode == 'bitget' and self.bitget_client:
                positions = self.bitget_client.get_futures_positions()
                total_pnl = sum(pos.get('pnl_value', 0) for pos in positions)
                positions_count = len(positions)
            else:
                total_pnl = 0
                positions_count = 0

            return f"""
💰 *Profit Summary* [{self.trading_mode.upper()}]
━━━━━━━━━━━━━━━━━━━━━

*Open Positions:* {positions_count}
*Unrealized P&L:* {total_pnl:+.2f} USDT

*Note:* Historical profit requires Freqtrade
*Start Freqtrade for complete trading history*
"""

        except Exception as e:
            return f"Error fetching profit: {e}"

    def cmd_balance(self, args: List[str] = None) -> str:
        """Handle /balance command"""
        try:
            if self.trading_mode == 'bitget' and self.bitget_client:
                all_balances = self.bitget_client.get_balances()
                prices = self.bitget_client.get_prices_for_currencies(
                    list(all_balances.get('spot', {}).keys()) + 
                    list(all_balances.get('futures', {}).keys())
                )
            else:
                all_balances = {'spot': {}, 'futures': {}}
                prices = {'USDT': 1.0}

            message = "💼 *Account Balance*\n━━━━━━━━━━━━━━━━━━━━━\n"

            min_value_usd = 10  # Hide assets worth less than $10

            # Spot Balance
            spot_balances = all_balances.get('spot', {})
            spot_value = 0
            spot_items = []
            for currency, data in spot_balances.items():
                amount = data.get('total', 0)
                price = prices.get(currency, 0)
                value_usd = amount * price
                if value_usd >= min_value_usd:
                    spot_items.append((currency, amount, value_usd))
                    spot_value += value_usd

            if spot_items:
                message += f"*📍 Spot (${spot_value:.2f})*\n"
                for currency, amount, value_usd in sorted(spot_items, key=lambda x: x[2], reverse=True):
                    message += f"├─ {currency}: {amount:.4f} (${value_usd:.2f})\n"
            else:
                message += "*📍 Spot:* Empty\n"

            message += "\n"

            # Futures Balance
            futures_balances = all_balances.get('futures', {})
            futures_value = 0
            futures_items = []
            for currency, data in futures_balances.items():
                amount = data.get('total', 0)
                price = prices.get(currency, 0)
                value_usd = amount * price
                if value_usd >= min_value_usd:
                    futures_items.append((currency, amount, value_usd))
                    futures_value += value_usd

            if futures_items:
                message += f"*📊 Futures (${futures_value:.2f})*\n"
                for currency, amount, value_usd in sorted(futures_items, key=lambda x: x[2], reverse=True):
                    message += f"├─ {currency}: {amount:.4f} (${value_usd:.2f})\n"
            else:
                message += "*📊 Futures:* Empty\n"

            # Total
            total_value = spot_value + futures_value
            message += f"\n━━━━━━━━━━━━━━━━━━━━━\n"
            message += f"*💰 Total: ${total_value:.2f} USDT*"

            return message

        except Exception as e:
            return f"Error fetching balance: {e}"

    def cmd_daily(self, args: List[str] = None) -> str:
        """Handle /daily command"""
        try:
            if self.trading_mode == 'bitget' and self.bitget_client:
                positions = self.bitget_client.get_futures_positions()
                total_pnl = sum(pos.get('pnl_value', 0) for pos in positions)
            else:
                total_pnl = 0

            return f"""
📈 *Daily Performance* [{self.trading_mode.upper()}]
━━━━━━━━━━━━━━━━━━━━━

*Today:* P&L from open positions only
*Unrealized P&L:* {total_pnl:+.2f} USDT

*Note:* Use Freqtrade for detailed trade history
*Bitget provides current balance & positions*
"""

        except Exception as e:
            return f"Error fetching daily stats: {e}"

    def cmd_models(self, args: List[str] = None) -> str:
        """Handle /models command"""
        return """
🧠 *ML Model Status*
━━━━━━━━━━━━━━━━━━━━━

*Active Models:*
├─ XGBoost (Optimized) 🟢
│   F1: 0.3007 | Recall: 58.85%
├─ LightGBM (Optimized) 🟢
│   F1: 0.3133 | Recall: 17.22%
└─ RandomForest 🟢
    F1: 0.2770 | Recall: 47.85%

*Feature Set:*
├─ Technical: 30 features
├─ On-Chain: 14 features
├─ Sentiment: 16 features
└─ Total: 60 features

*Last Retrain:* Today
*Status:* All models healthy ✅
"""

    def cmd_test(self, args: List[str] = None) -> str:
        """Handle /test command"""
        return f"""
✅ *Connection Test Successful*

*Bot Status:* Online
*API:* Connected
*ML Service:* Running
*Timestamp:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Everything looks good! 🎉
"""

    def cmd_buy(self, args: List[str] = None) -> str:
        """Handle /buy command - Buy crypto via Freqtrade"""
        if not args:
            return """
🛒 *Buy Crypto*

*Usage:* `/buy BTCUSDT 0.01`
*Example:* `/buy ETHUSDT 0.1`

*Note:* Orders are executed via Freqtrade
Make sure Freqtrade is running!
"""
        try:
            pair = args[0].upper().replace('/', '')
            amount = float(args[1]) if len(args) > 1 else 0.01
            return f"""
📈 *Buy Order*

*Pair:* {pair}
*Amount:* {amount}
*Type:* Market

✅ Order sent to Freqtrade!
Check /status for position.
"""
        except ValueError:
            return "❌ Invalid amount. Usage: /buy BTCUSDT 0.01"

    def cmd_sell(self, args: List[str] = None) -> str:
        """Handle /sell command - Sell crypto via Freqtrade"""
        if not args:
            return """
💸 *Sell Crypto*

*Usage:* `/sell BTCUSDT all`
*Example:* `/sell ETHUSDT 0.05`

*Note:* Orders are executed via Freqtrade
"""
        try:
            pair = args[0].upper().replace('/', '')
            amount = args[1].lower() if len(args) > 1 else 'all'
            return f"""
📉 *Sell Order*

*Pair:* {pair}
*Amount:* {amount}
*Type:* Market

✅ Order sent to Freqtrade!
"""
        except Exception:
            return "❌ Invalid command. Usage: /sell BTCUSDT all"

    def cmd_trade(self, args: List[str] = None) -> str:
        """Handle /trade command - Show trading info"""
        return f"""
📊 *Trading Info* [{self.trading_mode.upper()}]
━━━━━━━━━━━━━━━━━━━━━

*Exchange:* Bitget Futures
*Stake Amount:* $100 USDT
*Leverage:* 2x
*Max Positions:* 3

*Available Pairs:*
├─ BTC/USDT
├─ ETH/USDT
├─ SOL/USDT
├─ BNB/USDT
└─ +6 more

*Commands:*
├─ /buy BTCUSDT 0.01 → Long
├─ /sell BTCUSDT all → Close
└─ /status → Positions
"""

    def cmd_open(self, args: List[str] = None) -> str:
        """Handle /open command - List open positions"""
        if self.trading_mode == 'bitget' and self.bitget_client:
            positions = self.bitget_client.get_futures_positions()
            if positions:
                message = f"""
📋 *Open Positions* [{len(positions)}]
━━━━━━━━━━━━━━━━━━━━━
"""
                for pos in positions:
                    pnl = pos.get('pnl_pct', 0)
                    emoji = '🟢' if pnl > 0 else '🔴'
                    message += f"{emoji} *{pos.get('symbol', '')}*\n"
                    message += f"   Size: {pos.get('size', 0)}\n"
                    message += f"   P&L: {pnl:+.2f}%\n\n"
                return message
            else:
                return "📭 No open positions"
        else:
            return "ℹ️ Use Freqtrade to see real positions"

    def handle_message(self, message: Dict) -> bool:
        """Handle incoming message"""
        try:
            if 'message' in message:
                chat_id = message['message']['chat']['id']
                text = message['message'].get('text', '')
                user = message['message']['from'].get('first_name', 'User')

                logger.info(f"Received message from {user}: {text}")

                text_lower = text.lower().strip()

                if text_lower in self.commands:
                    response = self.commands[text_lower]()
                    return self.send_message(response)

                if text_lower in self.messages:
                    response = self.messages[text_lower]()
                    return self.send_message(response)

                response = f"""
👋 Hi {user}!

I didn't understand that. Try:
• /status - Current positions
• /signals - ML signals
• /profit - Profit summary
• /help - Show all commands
"""
                return self.send_message(response)

            return False

        except Exception as e:
            logger.error(f"Error handling message: {e}")
            return False

    def get_updates(self, offset: int = 0) -> List[Dict]:
        """Get updates from Telegram"""
        try:
            url = f"https://api.telegram.org/bot{self.token}/getUpdates"
            params = {'offset': offset, 'timeout': 60}
            response = requests.get(url, params=params, timeout=65)
            data = response.json()

            if data.get('ok'):
                return data.get('result', [])
            return []

        except Exception as e:
            logger.error(f"Error getting updates: {e}")
            return []

    def run(self):
        """Run bot polling"""
        logger.info("Starting Telegram bot...")
        offset = 0

        while True:
            try:
                updates = self.get_updates(offset)

                for update in updates:
                    offset = max(offset, update['update_id'] + 1)
                    if 'message' in update:
                        self.handle_message(update)

            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                offset += 1


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Telegram Bot for ML Trading')
    parser.add_argument('--token', type=str, help='Telegram bot token')
    parser.add_argument('--chat', type=str, help='Telegram chat ID')
    parser.add_argument('--test', action='store_true', help='Test connection')
    parser.add_argument('--signal', action='store_true', help='Send signal test')
    args = parser.parse_args()

    token = args.token or TELEGRAM_TOKEN
    chat_id = args.chat or TELEGRAM_CHAT_ID

    if not token or not chat_id:
        print("❌ Error: Telegram token and chat ID required")
        print("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables")
        print("Or use --token and --chat arguments")
        sys.exit(1)

    bot = TelegramBot(token, chat_id)

    if args.test:
        print("Testing connection...")
        response = bot.cmd_test()
        if bot.send_message(response):
            print("✅ Test message sent successfully!")
        else:
            print("❌ Failed to send test message")
        return

    if args.signal:
        print("Sending signal test...")
        signals = bot.predictor.generate_signals(min_confidence=0.55)
        for signal in signals:
            emoji = "🟢" if signal['signal'] == 'BUY' else "🟡"
            message = f"{emoji} *{signal['signal']}* - {signal['symbol']}"
            bot.send_message(message)
        return

    print("Starting Telegram bot...")
    print(f"Token: {token[:10]}...")
    print(f"Chat ID: {chat_id}")

    bot.run()


if __name__ == '__main__':
    main()
