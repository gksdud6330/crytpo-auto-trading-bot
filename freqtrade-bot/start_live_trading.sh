#!/bin/bash
export PATH="$PATH:/Users/hy/Library/Python/3.12/bin"
export FREQTRADE_CONFIG="/Users/hy/Desktop/Coding/stock-market/freqtrade-bot/config.bitget.json"

# 최고 전략 지정
BEST_STRATEGY="${1:-EMA_RSIStrategy}"

echo "===================================================="
echo "⚠️  ⚠️  실전 트레이딩 시작 - 리스크 주의 ⚠️  ⚠️"
echo "===================================================="
echo ""
echo "❗  실제 자금으로 트레이딩합니다!"
echo "❗  손실 가능성이 있으니 주의하세요!"
echo ""
echo "🔐 필수 사항:"
echo "  1. config.bitget.json에 실제 Bitget API 키 입력"
echo "  2. 'dry_run': false로 변경"
echo "  3. Telegram 알림 설정 (선택)"
echo ""
echo "🎯 사용 전략: $BEST_STRATEGY"
echo "⏱️  타임프레임: 5m"
echo "📊  최대 오픈 트레이드: 3"
echo ""
echo "⏳ 실전 트레이딩 시작 중..."
echo ""

read -p "실전 트레이딩을 시작하시겠습니까? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
  echo "❌ 취소되었습니다."
  exit 1
fi

python3 -m freqtrade trade \
  --config "$FREQTRADE_CONFIG" \
  --strategy "$BEST_STRATEGY"

