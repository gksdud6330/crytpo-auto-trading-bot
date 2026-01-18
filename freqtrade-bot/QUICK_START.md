# 🚀 Freqtrade + Bitget Quick Start Guide

## 📊 진행 상태

### ✅ 완료 (Phase 1-3):
- ✅ Freqtrade 설치 완료
- ✅ 5가지 전략 파일 생성 완료
- ✅ 설정 파일 생성 완료
- ✅ 자동화 스크립트 준비 완료

### 🔄 진행 중 (Phase 4):
- 🔄 과거 데이터 다운로드 중 (백그라운드)
  - 코인: BTC/USDT, ETH/USDT, BNB/USDT, SOL/USDT
  - 기간: 2024-10-01 ~ 현재
  - 타임프레임: 5m
  - 디렉토리: `user_data/data/`

---

## 📁 생성된 파일

### 📝 문서
- ✅ `spec.md` - 상세 명세서 (22KB)
- ✅ `README.md` - 빠른 참조 가이드
- ✅ `.gitignore` - 보안 설정

### 🎯 전략 파일 (user_data/strategies/)
- ✅ `RSIStrategy.py` (1.4KB) - RSI 기반 전략
- ✅ `MACDStrategy.py` (1.6KB) - MACD 기반 전략
- ✅ `BBStrategy.py` (1.5KB) - Bollinger Bands 기반 전략
- ✅ `EMA_RSIStrategy.py` (1.5KB) - EMA + RSI 결합 전략
- ✅ `Strategy005.py` (6.5KB) - Freqtrade 공식 전략

### ⚙️  설정 파일
- ✅ `config.json` - 백테스팅용 설정
- ✅ `config.bitget.json` - Bitget용 설정 (API 키 입력 필요!)

### 🚀 자동화 스크립트
- ✅ `backtest_all.sh` - 모든 전략 백테스팅
- ✅ `hyperopt_best.sh` - 최고 전략 하이퍼옵트 최적화
- ✅ `start_dryrun.sh` - Dry-Run 트레이딩 시작
- ✅ `start_live_trading.sh` - 실전 트레이딩 시작

---

## 🔄 다음 단계

### Step 1: 데이터 다운로드 완료 대기
```bash
# 데이터 다운로드 상태 확인
ls -lh user_data/data/

# 백그라운드 데이터 다운로드가 완료될 때까지 기다림
```

### Step 2: 모든 전략 백테스팅
```bash
# 모든 전략 한 번에 백테스팅 (데이터 완료 후)
./backtest_all.sh
```

### Step 3: 최고 전략 선택
백테스팅 결과를 확인하고 가장 좋은 전략 선택:
- 총 수익
- 승률
- 최대 손실
- 평균 수익

### Step 4: 하이퍼옵트 최적화
```bash
# 선택된 전략 최적화 (예: EMA_RSIStrategy)
./hyperopt_best.sh EMA_RSIStrategy
```

### Step 5: Dry-Run 테스트 (1-2주)
```bash
# API 키 설정 후 Dry-Run 실행
./start_dryrun.sh EMA_RSIStrategy
```

### Step 6: 실전 트레이딩
```bash
# Dry-Run으로 안정성 확인 후 실전 트레이딩
./start_live_trading.sh EMA_RSIStrategy
```

---

## 🔐 중요: config.bitget.json API 키 설정

**백테스팅**은 API 키 없이 가능하지만, **Dry-Run**과 **실전 트레이딩**에는 필수!

```bash
# config.bitget.json 편집
vim config.bitget.json  # 또는 사용하는 에디터
```

**변경할 항목:**
```json
{
  "exchange": {
    "key": "YOUR_BITGET_API_KEY_HERE",      ← 실제 API 키
    "secret": "YOUR_BITGET_API_SECRET_HERE", ← 실제 Secret 키
    "password": "YOUR_BITGET_API_PASSPHRASE_HERE" ← 실제 Passphrase
  }
}
```

**Bitget API 키 생성:**
1. https://www.bitget.com/ 접속
2. 로그인 → Account → API Management → Create API
3. 권한: ✅ Read + ✅ Trade (❌ Withdraw 비필요)
4. 생성된 키/시크릿/패스프레이즈 저장

---

## 📊 백테스팅 결과 확인

백테스팅 완료 후:

```bash
# 결과 파일 확인
ls -lh user_data/backtest_results/

# 최신 결과 보기
cat user_data/backtest_results/backtest-result-*.json | head -50
```

**기대 결과 포맷:**
```
========================================
Strategy Comparison Results
========================================

┌───────────────┬─────────┬──────────┬───────────┬──────────┬───────────┐
│ Strategy      │ Trades  │ Win Rate  │ Avg Profit │ Total %   │ Max DD    │
├───────────────┼─────────┼──────────┼───────────┼──────────┼───────────┤
│ RSIStrategy   │   145   │  52.4%   │   1.24%   │  17.98%   │  -12.3%   │
│ MACDStrategy  │    98   │  58.2%   │   1.67%   │  16.37%   │  -10.5%   │
│ BBStrategy    │   187   │  48.7%   │   0.98%   │  18.33%   │  -14.8%   │
│ EMA_RSIStrategy│   112   │  61.3%   │   1.85%   │  20.72%   │   -8.9%   │
│ Strategy005   │   234   │  55.1%   │   1.12%   │  26.21%   │  -11.2%   │
└───────────────┴─────────┴──────────┴───────────┴──────────┴───────────┘
```

---

## 🎯 성공 지표

### ✅ 목표 달성
- ✅ 총 수익률 > 20% (주 2% × 10주)
- ✅ 승률 50-70%
- ✅ 최대 손실 < -15%
- ✅ 평균 수익/트레이드 1-2%

### ⚠️  개선 필요
- ⚠️  승률 < 40% → 전략 재설계
- ⚠️  최대 손실 > -20% → 리스크 관리 강화
- ⚠️  총 수익 < 0% → 시장 상황 확인, 전략 조정

---

## 📖 참조 문서

- **[spec.md](./spec.md)** - 상세 명세서
- **[README.md](./README.md)** - 프로젝트 개요
- **[Freqtrade Docs](https://www.freqtrade.io)** - 공식 문서

---

**준비 완료! 데이터 다운로드 완료 후 백테스팅을 실행하세요!** 🚀
