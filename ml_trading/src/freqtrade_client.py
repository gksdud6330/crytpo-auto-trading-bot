"""
Freqtrade API Client for ML Trading System
Connects to Freqtrade running on Bitget exchange
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from urllib.parse import urljoin

import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FreqtradeClient:
    """Client for Freqtrade REST API"""

    def __init__(self, api_url: str = None):
        self.api_url = api_url or os.environ.get('FREQTRADE_API_URL', 'http://127.0.0.1:8080/api/v1')
        self.session = requests.Session()
        self.session.timeout = 10

    def _request(self, endpoint: str) -> Optional[Dict]:
        """Make GET request to Freqtrade API"""
        try:
            url = urljoin(self.api_url, endpoint)
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Freqtrade API error ({endpoint}): {e}")
            return None

    def get_status(self) -> List[Dict]:
        """Get open trades"""
        data = self._request('/status')
        if data and 'open_trades' in data:
            return data['open_trades']
        return []

    def get_trades(self, limit: int = 500) -> List[Dict]:
        """Get recent trades"""
        data = self._request('/trades')
        if data and 'trades' in data:
            return data['trades'][:limit]
        return []

    def get_profit(self) -> Dict:
        """Get profit summary"""
        data = self._request('/profit')
        if data:
            return data
        return {}

    def get_balance(self) -> Dict:
        """Get account balance"""
        data = self._request('/balance')
        if data:
            # Handle different response formats
            if 'balance' in data:
                return data['balance']
            elif 'balances' in data:
                return data['balances']
        return {}

    def get_daily_profit(self, days: int = 7) -> List[Dict]:
        """Get profit for last N days"""
        trades = self.get_trades(limit=500)
        daily_stats = {}

        for trade in trades:
            if trade.get('status') != 'closed':
                continue

            close_date = trade.get('close_date')
            if not close_date:
                continue

            try:
                date = datetime.fromisoformat(close_date.replace('Z', '+00:00')).date()
                if (datetime.now().date() - date).days <= days:
                    if date not in daily_stats:
                        daily_stats[date] = {'trades': 0, 'profit': 0.0, 'wins': 0}

                    daily_stats[date]['trades'] += 1
                    profit = float(trade.get('profit', 0))
                    daily_stats[date]['profit'] += profit
                    if profit > 0:
                        daily_stats[date]['wins'] += 1
            except (ValueError, TypeError):
                continue

        return [
            {
                'date': str(date),
                'trades': stats['trades'],
                'profit': stats['profit'],
                'win_rate': stats['wins'] / stats['trades'] if stats['trades'] > 0 else 0
            }
            for date, stats in sorted(daily_stats.items())
        ]

    def get_open_trades_with_pnl(self) -> List[Dict]:
        """Get open trades with current P&L"""
        open_trades = self.get_status()
        result = []

        for trade in open_trades:
            # Calculate P&L if we have current price info
            entry_price = trade.get('entry_price', 0)
            amount = trade.get('amount', 0)
            current_price = trade.get('current_price', 0)

            if current_price and entry_price:
                # For long positions
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
                trade['pnl_pct'] = round(pnl_pct, 2)
                trade['pnl_value'] = round((current_price - entry_price) * amount, 2)
            else:
                trade['pnl_pct'] = 0
                trade['pnl_value'] = 0

            result.append(trade)

        return result

    def is_available(self) -> bool:
        """Check if Freqtrade API is available"""
        return self._request('/status') is not None


# Singleton instance
_client = None


def get_client() -> FreqtradeClient:
    """Get or create Freqtrade client singleton"""
    global _client
    if _client is None:
        _client = FreqtradeClient()
    return _client


if __name__ == '__main__':
    # Test connection
    client = FreqtradeClient()
    print("Freqtrade API Test")
    print("=" * 50)

    if client.is_available():
        print("✓ Connected to Freqtrade")

        # Test status
        status = client.get_status()
        print(f"Open trades: {len(status)}")

        # Test profit
        profit = client.get_profit()
        print(f"Total profit: {profit}")

        # Test balance
        balance = client.get_balance()
        print(f"Balance currencies: {list(balance.keys())}")
    else:
        print("✗ Cannot connect to Freqtrade")
        print(f"API URL: {client.api_url}")
        print("Make sure Freqtrade is running with API enabled")
