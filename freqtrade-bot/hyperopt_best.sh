#!/bin/bash
export PATH="$PATH:/Users/hy/Library/Python/3.12/bin"
export FREQTRADE_CONFIG="/Users/hy/Desktop/Coding/stock-market/freqtrade-bot/config.bitget.json"

# 최고 전략 지정 (백테스팅 후 수정 필요!)
BEST_STRATEGY="${1:-EMA_RSIStrategy}"

echo "===================================================="
echo "Freqtrade 하이퍼옵트 최적화 시작"
echo "===================================================="
echo ""
echo "🎯 최적화할 전략: $BEST_STRATEGY"
echo "🔢 Epochs: 500"
echo "📊 Loss Function: SharpeHyperOptLossDaily"
echo "⚙️  최적화 공간: all (roi, stoploss, trailing, buy, sell)"
echo ""
echo "⏳ 하이퍼옵트 시작 중..."
echo ""

python3 -m freqtrade hyperopt \
  --config "$FREQTRADE_CONFIG" \
  --strategy "$BEST_STRATEGY" \
  --hyperopt-loss SharpeHyperOptLossDaily \
  --spaces all \
  -e 500 \
  --jobs 4

echo ""
echo "===================================================="
echo "하이퍼옵트 완료!"
echo "===================================================="
echo ""
echo "📁 결과 위치:"
echo "  - user_data/hyperopt_results/"
echo "  - user_data/hyperopts/hyperopt_results_*.pickle"
echo ""
echo "💡 최적 파라미터 적용 방법:"
echo "  1. user_data/hyperopt_results/hyperopt_results_*.pickle 파일 열기"
echo "  2. 'Best result:' 섹션에서 최적 파라미터 확인"
echo "  3. 해당 파라미터를 전략 파일의 buy_params, sell_params에 복사"
echo ""
