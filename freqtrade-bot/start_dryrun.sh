#!/bin/bash
export PATH="$PATH:/Users/hy/Library/Python/3.12/bin"
export FREQTRADE_CONFIG="/Users/hy/Desktop/Coding/stock-market/freqtrade-bot/config.bitget.json"

# 최고 전략 지정 (백테스팅 후 하이퍼옵트 후 수정 필요!)
BEST_STRATEGY="${1:-EMA_RSIStrategy}"

echo "===================================================="
echo "Freqtrade Dry-Run 트레이딩 시작"
echo "===================================================="
echo ""
echo "⚠️  중요: config.bitget.json에 실제 API 키를 입력했는지 확인하세요!"
echo ""
echo "🎯 사용 전략: $BEST_STRATEGY"
echo "💰 Dry-Run 지갑: 1000 USDT (시뮬레이션)"
echo "⏱️  타임프레임: 5m"
echo "📊  최대 오픈 트레이드: 3"
echo ""
echo "📡 WebUI 접속: http://localhost:8080"
echo "📱  Telegram 알림: 설정 필요시 config.bitget.json 수정"
echo ""
echo "⏳ Dry-Run 트레이딩 시작 중..."
echo ""

python3 -m freqtrade trade \
  --config "$FREQTRADE_CONFIG" \
  --strategy "$BEST_STRATEGY"

