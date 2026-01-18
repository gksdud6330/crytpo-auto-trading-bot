#!/bin/bash

# Telegram Bot Setup Script for ML Trading System
# Run this to configure Telegram bot

set -e

echo "======================================"
echo "  Telegram Bot Setup for ML Trading"
echo "======================================"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python
echo "Checking Python..."
python3 --version || { echo -e "${RED}Python 3 not found${NC}"; exit 1; }
echo -e "${GREEN}✓ Python 3 found${NC}"

# Install requests if not installed
echo "Installing dependencies..."
pip3 install requests -q 2>/dev/null || pip install requests -q
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Get Telegram Token
echo ""
echo -e "${YELLOW}Step 1: Telegram Bot Token${NC}"
echo "You need to create a Telegram bot and get the token."
echo ""
echo "1. Open Telegram and search for @BotFather"
echo "2. Send /newbot to create a new bot"
echo "3. Follow the instructions"
echo "4. Copy the bot token"
echo ""

read -p "Enter your Telegram Bot Token: " TELEGRAM_TOKEN

if [ -z "$TELEGRAM_TOKEN" ]; then
    echo -e "${RED}Error: Token is required${NC}"
    exit 1
fi

# Get Chat ID
echo ""
echo -e "${YELLOW}Step 2: Get Your Chat ID${NC}"
echo "You need to find your Telegram chat ID."
echo ""
echo "1. Search for @userinfobot on Telegram"
echo "2. Send /start"
echo "3. Copy your ID number"
echo ""

read -p "Enter your Chat ID (numbers only): " TELEGRAM_CHAT_ID

if [ -z "$TELEGRAM_CHAT_ID" ]; then
    echo -e "${RED}Error: Chat ID is required${NC}"
    exit 1
fi

# Save to .env file
echo ""
echo -e "${YELLOW}Step 3: Saving Configuration${NC}"

cat > .env << EOF
# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN=$TELEGRAM_TOKEN
TELEGRAM_CHAT_ID=$TELEGRAM_CHAT_ID

# Freqtrade Configuration (optional)
# FREQTRADE_API_URL=http://localhost:8080
EOF

echo -e "${GREEN}✓ Configuration saved to .env${NC}"

# Test connection
echo ""
echo -e "${YELLOW}Step 4: Testing Connection${NC}"

# Create test script
cat > test_telegram.py << EOF
import requests
import sys

TOKEN = "$TELEGRAM_TOKEN"
CHAT_ID = "$TELEGRAM_CHAT_ID"
API_URL = f"https://api.telegram.org/bot{TOKEN}"

# Test bot
try:
    # Get bot info
    resp = requests.get(f"{API_URL}/getMe", timeout=10)
    if resp.status_code == 200:
        bot_name = resp.json()['result']['first_name']
        print(f"✓ Bot '{bot_name}' connected successfully")
    else:
        print("✗ Failed to connect bot")
        sys.exit(1)

    # Test sending message
    test_msg = "✅ ML Trading Bot connected successfully!"
    msg_resp = requests.post(f"{API_URL}/sendMessage", json={
        'chat_id': CHAT_ID,
        'text': test_msg
    }, timeout=10)

    if msg_resp.status_code == 200:
        print("✓ Test message sent successfully!")
    else:
        print("✗ Failed to send test message")
        print(f"Error: {msg_resp.json()}")
        sys.exit(1)

    print("\n🎉 Telegram bot is ready!")
    print("Now start the bot with: python3 src/telegram_bot.py")

except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)
EOF

python3 test_telegram.py
rm -f test_telegram.py

# Create systemd service file
echo ""
echo -e "${YELLOW}Step 5: Creating Service File (Optional)$NC"

read -p "Create systemd service for auto-start? (y/n): " create_service

if [ "$create_service" = "y" ] || [ "$create_service" = "Y" ]; then

    # Get current directory
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

    cat > ml-trading-bot.service << EOF
[Unit]
Description=ML Trading Telegram Bot
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$SCRIPT_DIR
ExecStart=$(which python3) $SCRIPT_DIR/src/telegram_bot.py
Restart=on-failure
RestartSec=10
Environment=TELEGRAM_BOT_TOKEN=$TELEGRAM_TOKEN
Environment=TELEGRAM_CHAT_ID=$TELEGRAM_CHAT_ID

[Install]
WantedBy=multi-user.target
EOF

    echo -e "${GREEN}✓ Service file created: ml-trading-bot.service${NC}"
    echo ""
    echo "To install:"
    echo "  sudo cp ml-trading-bot.service /etc/systemd/system/"
    echo "  sudo systemctl daemon-reload"
    echo "  sudo systemctl enable ml-trading-bot"
    echo "  sudo systemctl start ml-trading-bot"
    echo ""
    echo "To check status:"
    echo "  sudo systemctl status ml-trading-bot"
    echo ""
    echo "To view logs:"
    echo "  sudo journalctl -u ml-trading-bot -f"
fi

echo ""
echo "======================================"
echo "  Setup Complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. Source the environment: source .env"
echo "2. Run the bot: python3 src/telegram_bot.py"
echo ""
echo "Or run in background:"
echo "  nohup python3 src/telegram_bot.py > bot.log 2>&1 &"
echo ""
