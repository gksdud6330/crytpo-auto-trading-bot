"""
Bitget API Client for ML Trading System
Supports Spot and Futures trading
"""

import os
import hmac
import hashlib
import time
import json
import logging
import base64
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import urlencode

import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BitgetClient:
    """Bitget API Client"""

    def __init__(self, api_key: str = None, api_secret: str = None, passphrase: str = None):
        self.api_key = api_key or os.environ.get('BITGET_API_KEY', '')
        self.api_secret = api_secret or os.environ.get('BITGET_API_SECRET', '')
        self.passphrase = passphrase or os.environ.get('BITGET_PASSPHRASE', '')

        self.base_url = 'https://api.bitget.com'
        self.session = requests.Session()
        self.session.timeout = 10

    def _sign(self, timestamp: str, method: str, path: str, body: str = '') -> str:
        """Generate signature for API request"""
        message = f"{timestamp}{method.upper()}{path}{body}"
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).digest()
        return base64.b64encode(signature).decode('utf-8')

    def _request(self, method: str, path: str, body: str = '') -> Optional[Dict]:
        """Make signed request to Bitget API"""
        if not self.api_key or not self.api_secret:
            logger.warning("Bitget credentials not configured")
            return None

        try:
            timestamp = str(int(time.time() * 1000))
            headers = {
                'ACCESS-KEY': self.api_key,
                'ACCESS-SIGN': self._sign(timestamp, method, path, body),
                'ACCESS-TIMESTAMP': timestamp,
                'ACCESS-PASSPHRASE': self.passphrase,
                'Content-Type': 'application/json'
            }

            url = f"{self.base_url}{path}"

            if method.upper() == 'GET':
                response = self.session.get(url, headers=headers, timeout=10)
            else:
                response = self.session.request(method.upper(), url, headers=headers, data=body, timeout=10)

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"Bitget API error: {e}")
            return None

    def get_spot_balance(self) -> Dict[str, float]:
        """Get spot account balance"""
        path = '/api/v2/spot/account/assets'
        data = self._request('GET', path)

        if data and data.get('data'):
            balance = {}
            for item in data['data']:
                currency = item.get('coin', '')
                available = float(item.get('available', 0))
                frozen = float(item.get('frozen', 0))
                if available > 0 or frozen > 0:
                    balance[currency] = {
                        'available': available,
                        'locked': frozen,
                        'total': available + frozen
                    }
            return balance
        return {}

    def get_futures_balance(self) -> Dict[str, float]:
        """Get futures account balance"""
        path = '/api/v2/mix/account/accounts?productType=USDT-FUTURES'
        data = self._request('GET', path)

        if data and data.get('data'):
            balance = {}
            for item in data['data']:
                currency = item.get('marginCoin', '')
                available = float(item.get('available', 0))
                locked = float(item.get('frozen', 0))
                if available > 0 or locked > 0:
                    balance[currency] = {
                        'available': available,
                        'locked': locked,
                        'total': available + locked
                    }
            return balance
        return {}

    def get_futures_positions(self) -> List[Dict]:
        """Get futures positions"""
        path = '/api/v2/mix/position/all-position?productType=USDT-FUTURES'
        data = self._request('GET', path)

        if data and data.get('data'):
            positions = []
            for item in data['data']:
                size = float(item.get('totalVolume', 0))
                if size > 0:
                    positions.append({
                        'symbol': item.get('symbol', ''),
                        'side': item.get('side', ''),
                        'size': size,
                        'entry_price': float(item.get('averageOpenPrice', 0)),
                        'mark_price': float(item.get('markPrice', 0)),
                        'pnl_pct': float(item.get('closePnL', 0)),
                        'pnl_value': float(item.get('unrealizedPnl', 0)),
                    })
            return positions
        return []

    def get_ticker_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        path = f'/api/v2/spot/market/tickers?symbol={symbol}'
        data = self._request('GET', path)

        if data and data.get('data'):
            return float(data['data'][0]['lastPr'])
        return None

    def get_prices_for_currencies(self, currencies: List[str]) -> Dict[str, float]:
        """Get prices for multiple currencies in USDT"""
        prices = {}
        for currency in currencies:
            if currency == 'USDT':
                prices[currency] = 1.0
            else:
                symbol = f"{currency}USDT"
                price = self.get_ticker_price(symbol)
                if price:
                    prices[currency] = price
        return prices

    def get_balances(self) -> Dict:
        """Get all balances (spot + futures)"""
        spot = self.get_spot_balance()
        futures = self.get_futures_balance()
        return {
            'spot': spot,
            'futures': futures
        }

    def is_available(self) -> bool:
        """Check if Bitget API is available"""
        return self.get_spot_balance() is not None


# Singleton instance
_client = None


def get_client() -> BitgetClient:
    """Get or create Bitget client singleton"""
    global _client
    if _client is None:
        _client = BitgetClient()
    return _client


if __name__ == '__main__':
    client = BitgetClient()
    print("Bitget API Test")
    print("=" * 50)

    if client.is_available():
        print("✓ Connected to Bitget")

        # Get spot balance
        spot = client.get_spot_balance()
        print(f"\nSpot balances: {len(spot)} currencies")
        for currency, data in list(spot.items())[:5]:
            print(f"  {currency}: {data['total']}")

        # Get futures balance
        futures = client.get_futures_balance()
        print(f"\nFutures balances: {len(futures)} currencies")
        for currency, data in list(futures.items())[:5]:
            print(f"  {currency}: {data['total']}")

        # Get positions
        positions = client.get_futures_positions()
        print(f"\nOpen positions: {len(positions)}")
        for pos in positions:
            print(f"  {pos['symbol']}: {pos['side']} {pos['size']}")
    else:
        print("✗ Cannot connect to Bitget")
        print("\nAdd credentials to .env:")
        print("BITGET_API_KEY=your_key")
        print("BITGET_API_SECRET=your_secret")
        print("BITGET_PASSPHRASE=your_passphrase")
