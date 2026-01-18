# 📱 Telegram 봇 설정 완벽 가이드

이 문서는 텔레그램 봇을 설정하는 모든 단계를 상세히 설명합니다.

---

## 1단계: Telegram Bot 생성

### 1.1 BotFather에서 봇 생성

```
1. Telegram 앱을 열고 검색창에 "BotFather" 입력
2. @BotFather 선택 (파란색 체크표시가 있음)
3. /newbot 입력하여 새 봇 생성 시작

 BotsFather에게:

/newbot

4. 봇 이름 입력 (예시: "ML Trading Bot")
   → ML Trading Bot

5. 봇 username 입력 (끝에 "bot" 포함, 예시: "mltrading2024bot")
   → mltrading2024bot

6. ✅ 완료! 토큰이 표시됩니다
   예시: 123456789:ABCdefGHIjklMNOpqrsTUVwxyz

⚠️ 중요: 이 토큰을 복사해서 안전한 곳에 저장하세요!
```

### 1.2 Chat ID 확인

```
1. Telegram에서 "userinfobot" 검색
2. @userinfobot 선택
3. /start 입력
4. 숫자로 된 "id"를 복사 (예시: 123456789)
```

---

## 2단계: API 키 설정

### 2.1 환경변수로 설정 (임시)

터미널에서 다음 명령어 실행:

```bash
cd /Users/hy/Desktop/Coding/stock-market/ml_trading

# Telegram 토큰 입력 (1.1에서 복사한 것)
export TELEGRAM_BOT_TOKEN="여기에_토큰_粘贴"

# Chat ID 입력 (1.2에서 복사한 것)
export TELEGRAM_CHAT_ID="여기에_ID_粘贴"
```

### 2.2 영구적으로 설정 (.env 파일 생성)

```bash
cd /Users/hy/Desktop/Coding/stock-market/ml_trading

# .env 파일 생성
nano .env
```

파일 내용:
```
TELEGRAM_BOT_TOKEN=여기에_실제_토큰_粘贴
TELEGRAM_CHAT_ID=여기에_실제_ID_粘贴
```

저장: `Ctrl + O` → `Enter` → `Ctrl + X`

```bash
# 적용
source .env

# 확인 (토큰 앞 10글자만 표시됨)
echo ${TELEGRAM_BOT_TOKEN:0:10}
echo $TELEGRAM_CHAT_ID
```

---

## 3단계: 봇 실행 테스트

### 3.1 연결 테스트

```bash
cd /Users/hy/Desktop/Coding/stock-market/ml_trading

python3 src/telegram_bot.py --test --token $TELEGRAM_BOT_TOKEN --chat $TELEGRAM_CHAT_ID
```

성공 시:
```
Testing connection...
Token: 1234567890...
Chat ID: 123456789
✅ Test message sent successfully!
```

 Telegram에서 메시지를 확인하세요!

### 3.2 실시간 봇 실행

```bash
cd /Users/hy/Desktop/Coding/stock-market/ml_trading

python3 src/telegram_bot.py --token $TELEGRAM_BOT_TOKEN --chat $TELEGRAM_CHAT_ID
```

출력:
```
Starting Telegram bot...
Token: 1234567890...
Chat ID: 123456789
```

이 상태에서 Telegram으로 명령어 입력:
- `/start`
- `/signals`
- `/status`
- `/profit`

### 3.3 백그라운드에서 실행 (서버처럼)

```bash
# 방법 1: nohup 사용
cd /Users/hy/Desktop/Coding/stock-market/ml_trading
nohup python3 src/telegram_bot.py > bot.log 2>&1 &

# 방법 2: screen 사용
screen -S telegram_bot
cd /Users/hy/Desktop/Coding/stock-market/ml_trading
python3 src/telegram_bot.py
# (나중에 나올려면: Ctrl+A, D)

# 실행 확인
ps aux | grep telegram_bot
```

---

## 4단계: 자동 시작 설정 (Mac)

### 4.1 launchd 사용

```bash
# 파일 생성
nano ~/Library/LaunchAgents/com.mltrading.telegram.plist
```

파일 내용 (토큰과 ID만 변경):
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.mltrading.telegram</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>/Users/hy/Desktop/Coding/stock-market/ml_trading/src/telegram_bot.py</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>TELEGRAM_BOT_TOKEN</key>
        <string>여기에_토큰_粘贴</string>
        <key>TELEGRAM_CHAT_ID</key>
        <string>여기에_ID_粘贴</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>WorkingDirectory</key>
    <string>/Users/hy/Desktop/Coding/stock-market/ml_trading</string>
</dict>
</plist>
```

등록 및 실행:
```bash
# 등록
launchctl load ~/Library/LaunchAgents/com.mltrading.telegram.plist

# 실행
launchctl start com.mltrading.telegram

# 상태 확인
launchctl list | grep mltrading

# 로그 확인
tail -f ~/Library/Logs/com.mltrading.telegram.log
```

중지:
```bash
launchctl stop com.mltrading.telegram
launchctl unload ~/Library/LaunchAgents/com.mltrading.telegram.plist
```

---

## 5단계: Linux 서버에서 자동 시작

```bash
# 서비스 파일 생성
sudo nano /etc/systemd/system/ml-trading-bot.service
```

파일 내용:
```ini
[Unit]
Description=ML Trading Telegram Bot
After=network.target

[Service]
Type=simple
User=hy
WorkingDirectory=/Users/hy/Desktop/Coding/stock-market/ml_trading
ExecStart=/usr/bin/python3 /Users/hy/Desktop/Coding/stock-market/ml_trading/src/telegram_bot.py
Restart=on-failure
RestartSec=10
Environment=TELEGRAM_BOT_TOKEN=여기에_토큰_粘贴
Environment=TELEGRAM_CHAT_ID=여기에_ID_粘贴

[Install]
WantedBy=multi-user.target
```

설치:
```bash
# 등록
sudo systemctl daemon-reload
sudo systemctl enable ml-trading-bot

# 실행
sudo systemctl start ml-trading-bot

# 상태 확인
sudo systemctl status ml-trading-bot

# 로그 확인
sudo journalctl -u ml-trading-bot -f
```

---

## 명령어 체크리스트

| 순서 | 명령어 | 설명 |
|------|--------|------|
| 1 | `cd /Users/hy/Desktop/Coding/stock-market/ml_trading` | 디렉토리 이동 |
| 2 | `export TELEGRAM_BOT_TOKEN="..."` | 토큰 설정 |
| 3 | `export TELEGRAM_CHAT_ID="..."` | Chat ID 설정 |
| 4 | `python3 src/telegram_bot.py --test` | 연결 테스트 |
| 5 | `python3 src/telegram_bot.py` | 봇 실행 |
| 6 | `ps aux \| grep telegram` | 실행 확인 |

---

## 자주 발생하는 문제

### 문제 1: "Token is required" 에러

```
→ .env 파일이 없거나 비어있음
→ TELEGRAM_BOT_TOKEN 환경변수 확인
```

해결:
```bash
cd /Users/hy/Desktop/Coding/stock-market/ml_trading
cat .env
# 내용이 없으면 다시 생성
echo 'TELEGRAM_BOT_TOKEN="your_token"' > .env
echo 'TELEGRAM_CHAT_ID="your_id"' >> .env
source .env
```

### 문제 2: "Failed to send message" 에러

```
→ Chat ID가 잘못됨
→ Bot이 채팅에 추가되지 않음
```

해결:
```
1. Telegram에서 직접 Bot에게 메시지 전송
2. Bot을 그룹에 추가
3. Chat ID가 숫자만 인지 확인
```

### 문제 3: 봇이 응답하지 않음

```
→ 파이썬이 실행 중이 아님
```

해결:
```bash
# 프로세스 확인
ps aux | grep python

# 재시작
pkill -f telegram_bot
cd /Users/hy/Desktop/Coding/stock-market/ml_trading
nohup python3 src/telegram_bot.py > bot.log 2>&1 &
```

---

## 전체 흐름 요약

```
┌─────────────────────────────────────────────────────────┐
│ 1. Telegram에서 BotFather로 토큰 생성                    │
│    ↓                                                    │
│ 2. userinfobot로 Chat ID 확인                           │
│    ↓                                                    │
│ 3. .env 파일에 토큰/ID 저장                             │
│    ↓                                                    │
│ 4. python3 src/telegram_bot.py --test 로 테스트         │
│    ↓                                                    │
│ 5. 백그라운드에서 실행                                   │
│    ↓                                                    │
│ 6. Telegram에서 /signals 입력 → 신호 확인               │
└─────────────────────────────────────────────────────────┘
```

---

##快速参考 (Quick Reference)

```bash
# 설정
cd /Users/hy/Desktop/Coding/stock-market/ml_trading
nano .env
# → TELEGRAM_BOT_TOKEN=xxx
# → TELEGRAM_CHAT_ID=xxx
source .env

# 테스트
python3 src/telegram_bot.py --test --token $TELEGRAM_BOT_TOKEN --chat $TELEGRAM_CHAT_ID

# 실행
python3 src/telegram_bot.py --token $TELEGRAM_BOT_TOKEN --chat $TELEGRAM_CHAT_ID

# 백그라운드
nohup python3 src/telegram_bot.py > bot.log 2>&1 &
```

질문이 있으면 언제든지 물어보세요! 😊
