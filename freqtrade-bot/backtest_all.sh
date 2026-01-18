#!/bin/bash
export PATH="$PATH:/Users/hy/Library/Python/3.12/bin"
export FREQTRADE_CONFIG="/Users/hy/Desktop/Coding/stock-market/freqtrade-bot/config.json"

echo "===================================================="
echo "Freqtrade 전략 비교 백테스팅 시작"
echo "===================================================="
echo ""

echo "📊 백테스팅할 전략 목록:"
echo "  1. RSIStrategy"
echo "  2. MACDStrategy"
echo "  3. BBStrategy"
echo "  4. EMA_RSIStrategy"
echo "  5. Strategy005"
echo ""

echo "⏱️  백테스팅 기간: 2024-10-01 ~ 현재"
echo "⏱️  타임프레임: 5m"
echo "💰  시드 자본: 1000 USDT (dry_run_wallet)"
echo ""
echo "⏳ 백테스팅 시작 중..."
echo ""

python3 -m freqtrade backtesting \
  --config "$FREQTRADE_CONFIG" \
  --strategy-list RSIStrategy MACDStrategy BBStrategy EMA_RSIStrategy Strategy005 \
  --timeframe 5m \
  --timerange 20241001- \
  --export trades \
  --breakdown month \
  --fee 0.001

echo ""
echo "===================================================="
echo "백테스팅 완료!"
echo "===================================================="
echo ""
echo "📁 결과 위치:"
echo "  - user_data/backtest_results/"
echo "  - user_data/backtest_results/backtest-result-*.json"
echo ""
