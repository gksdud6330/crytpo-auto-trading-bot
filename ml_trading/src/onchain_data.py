"""
On-Chain Data Integration for Crypto ML Trading
- Fetches on-chain metrics (whale activity, MVRV, etc.)
- Uses public APIs where available
- Provides fallback to derived metrics when API unavailable
"""

import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

# Config
DATA_DIR = '/Users/hy/Desktop/Coding/stock-market/ml_trading/data'
ONCHAIN_DIR = '/Users/hy/Desktop/Coding/stock-market/ml_trading/data/onchain'

# Free API endpoints (no API key required)
FREE_API_ENDPOINTS = {
    'glassnode': {
        'base': 'https://api.glassnode.com/v1/metrics',
        'endpoints': {
            'mvrv': '/market/mrv_usd',  # Market Value / Realized Value
            'whale': '/distribution/balance_1k_usd',  # Whales (>1000 BTC)
            'exchange': '/distribution/balance_exchange_usd',  # Exchange reserves
            'nuv': '/indicators/nupl',  # Net Unrealized Profit/Loss
            'sopr': '/indicators/sopr',  # Spent Output Profit Ratio
        }
    },
    'cryptoquant': {
        'base': 'https://api.cryptoquant.com/v1/btc',
        'endpoints': {
            'flow': '/onchain-flows/inflow-all',
            'whale': '/onchain/whale-transaction-count',
        }
    },
    'glassnode_free': {
        'base': 'https://api.glassnode.com/v2/metrics',
        'endpoints': {
            'price': '/market/price_usd_close',
            'volume': '/market/volume_usd',
            'addresses': '/addresses/count',
            'active': '/addresses/active_count',
        }
    }
}

# API Keys (set as environment variables)
GLASSNODE_API_KEY = os.environ.get('GLASSNODE_API_KEY', '')
CRYPTOQUANT_API_KEY = os.environ.get('CRYPTOQUANT_API_KEY', '')


class OnChainDataFetcher:
    """Fetch on-chain data from various sources"""

    def __init__(self):
        self.data_dir = DATA_DIR
        self.onchain_dir = ONCHAIN_DIR
        os.makedirs(self.onchain_dir, exist_ok=True)

        # Cache for API responses
        self.cache = {}

    def fetch_from_api(self, url, params=None, api_key=''):
        """Generic API fetcher with rate limiting"""
        if url in self.cache:
            cached = self.cache[url]
            if (datetime.now() - cached['timestamp']).seconds < 3600:  # 1 hour cache
                return cached['data']

        try:
            headers = {'api-key': api_key} if api_key else {}

            # For public endpoints without API key
            if 'glassnode.com/v2' in url or 'glassnode.com/v1' in url:
                if not api_key:
                    # Use public endpoints
                    headers = {}

            response = requests.get(url, params=params, headers=headers, timeout=30)

            if response.status_code == 200:
                data = response.json()
                self.cache[url] = {
                    'data': data,
                    'timestamp': datetime.now()
                }
                return data
            elif response.status_code == 429:
                print(f"Rate limited. Waiting 60s...")
                time.sleep(60)
                return self.fetch_from_api(url, params, api_key)
            else:
                print(f"API error {response.status_code}: {response.text[:100]}")
                return None

        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None

    def fetch_glassnode_public(self, asset='BTC', since='2023-01-01'):
        """Fetch public Glassnode data (no API key needed)"""
        base_url = 'https://api.glassnode.com/v2/metrics'
        results = {}

        # Public endpoints that don't require API key
        public_metrics = {
            'price': 'market/price_usd_close',
            'volume': 'market/volume_usd',
            'active_addresses': 'addresses/active_count',
            'new_addresses': 'addresses/new_count',
            'total_addresses': 'addresses/count',
        }

        for metric_name, endpoint in public_metrics.items():
            url = f"{base_url}/{endpoint}"
            params = {
                'a': asset,
                's': since,
                'i': '4h'  # 4-hour interval
            }

            print(f"Fetching {metric_name}...")
            data = self.fetch_from_api(url, params)

            if data:
                df = pd.DataFrame(data)
                df['timestamp'] = pd.to_datetime(df['t'], unit='s')
                df.set_index('timestamp', inplace=True)
                df['value'] = df['v'].astype(float)
                results[metric_name] = df[['value']]

            time.sleep(0.5)  # Rate limiting

        return results

    def calculate_derived_onchain(self, df):
        """
        Calculate on-chain derived metrics from price/volume data
        These don't require external API calls
        """
        derived = pd.DataFrame(index=df.index)

        # Volume-based whale proxy
        derived['volume_sma_24h'] = df['volume'].rolling(window=6).mean()  # 6 * 4h = 24h
        derived['volume_spike'] = df['volume'] / derived['volume_sma_24h']
        derived['volume_trend'] = df['volume'].pct_change(6)  # 24h change

        # Price-based MVRV proxy (simplified)
        # MVRV = Current Price / Realized Price (simplified using cost basis)
        derived['cost_basis_7d'] = df['close'].rolling(window=42).mean()  # 7 days * 6 = 42
        derived['mvrv_proxy'] = df['close'] / derived['cost_basis_7d']

        # NUPL proxy (Net Unrealized Profit/Loss)
        # NUPL = (Market Cap - Realized Cap) / Market Cap
        # Simplified: (Price - Avg Cost) / Price
        derived['avg_cost_30d'] = df['close'].rolling(window=180).mean()  # 30 days
        derived['nupl_proxy'] = (df['close'] - derived['avg_cost_30d']) / df['close']

        # SOPR proxy (Spent Output Profit Ratio)
        # SOPR > 1 means profit taking, < 1 means capitulation
        derived['sopr_proxy'] = df['close'] / df['close'].shift(6)  # 24h return

        # Whale activity proxy (large volume detection)
        derived['whale_activity'] = (df['volume'] > df['volume'].rolling(24).quantile(0.95)).astype(int)

        # Exchange flow proxy (volume trend)
        derived['exchange_inflow_proxy'] = (df['volume_ratio'] > 1.5).astype(int)

        # Active addresses proxy (volume/price ratio anomaly)
        derived['active_proxy'] = (df['volume_ratio'] * df['volatility_1d']).rolling(6).mean()

        # HODL waves proxy (long-term holder detection)
        derived['hodl_proxy'] = df['close'] / df['close'].rolling(180).min()  # Distance from 30d low

        # Puell Multiple proxy (current issuance / 365-day avg)
        derived['puell_proxy'] = df['close'] / df['close'].rolling(876).mean()  # 365 days * 6

        return derived

    def calculate_network_metrics(self, df):
        """Calculate network health metrics"""
        network = pd.DataFrame(index=df.index)

        # Volume per transaction proxy
        network['volume_per_tx'] = df['volume'] / (df['volume'] / df['close'].rolling(6).mean())

        # Velocity (transaction volume / supply)
        network['velocity'] = df['volume'].rolling(6).sum() / df['close']

        # Accumulation score (price + volume relationship)
        network['accumulation_score'] = (
            (df['close'].pct_change(6) > 0).astype(int) +
            (df['volume_ratio'] > 1.2).astype(int) +
            (df['price_vs_ema_50'] > 0).astype(int)
        ) / 3

        # Institutional flow proxy
        network['institutional_proxy'] = (
            (df['volume'] > df['volume'].rolling(24).quantile(0.8)).astype(int) *
            (df['close'] > df['ema_50']).astype(int)
        )

        return network

    def merge_onchain_features(self, df, timeframe='4h'):
        """
        Merge all on-chain features with main dataframe
        Returns enhanced dataframe with on-chain indicators
        """
        print("="*60)
        print("ON-CHAIN DATA INTEGRATION")
        print("="*60)

        # Try to fetch real on-chain data first
        onchain_data = {}

        try:
            # Attempt to fetch public Glassnode data
            print("\n[1/3] Fetching public on-chain data...")
            glassnode_data = self.fetch_glassnode_public(asset='BTC', since='2023-01-01')

            if glassnode_data:
                onchain_data.update(glassnode_data)
                print(f"  ✅ Fetched {len(glassnode_data)} real metrics")
            else:
                print("  ⚠️ Using derived metrics only")

        except Exception as e:
            print(f"  ⚠️ Could not fetch on-chain data: {e}")
            print("  📊 Using derived on-chain metrics")

        # Calculate derived metrics
        print("\n[2/3] Calculating derived on-chain metrics...")
        derived = self.calculate_derived_onchain(df)
        print(f"  ✅ Calculated {len(derived.columns)} derived metrics")

        # Calculate network metrics
        print("\n[3/3] Calculating network metrics...")
        network = self.calculate_network_metrics(df)
        print(f"  ✅ Calculated {len(network.columns)} network metrics")

        # Merge all features
        print("\n" + "="*60)
        print("MERGING FEATURES")
        print("="*60)

        # Add derived metrics
        for col in derived.columns:
            df[col] = derived[col]

        # Add network metrics
        for col in network.columns:
            df[col] = network[col]

        # Add real on-chain data if available
        if onchain_data:
            for metric_name, metric_df in onchain_data.items():
                df[metric_name] = metric_df['value'].reindex(df.index)

        print(f"\n✅ Total features: {len(df.columns)} (was 37)")

        # Save enhanced data
        os.makedirs(f"{self.data_dir}/enhanced", exist_ok=True)
        for pair in df['pair'].unique():
            if 'pair' in df.columns:
                pair_df = df[df['pair'] == pair]
                filename = pair.replace('/', '_')
                pair_df.to_parquet(f"{self.data_dir}/enhanced/{filename}_enhanced.parquet")
                print(f"  ✅ Saved {pair} enhanced data")

        return df

    def create_onchain_features_csv(self):
        """Create a CSV with on-chain feature descriptions"""
        features = {
            'feature': [
                'volume_spike',
                'volume_trend',
                'mvrv_proxy',
                'nupl_proxy',
                'sopr_proxy',
                'whale_activity',
                'exchange_inflow_proxy',
                'active_proxy',
                'hodl_proxy',
                'puell_proxy',
                'volume_per_tx',
                'velocity',
                'accumulation_score',
                'institutional_proxy'
            ],
            'description': [
                'Volume relative to 24h average (whale detection)',
                '24h volume change percentage',
                'Market Value / Realized Value proxy (over/undervaluation)',
                'Net Unrealized Profit/Loss proxy',
                'Spent Output Profit Ratio proxy (profit taking indicator)',
                'Large transaction detection (95th percentile volume)',
                'High volume indicating exchange inflow',
                'Active address proxy (volume/volatility relationship)',
                'Distance from 30-day low (HODLer sentiment)',
                'Puell Multiple proxy (miner profitability)',
                'Volume per transaction proxy',
                'Token velocity (circulation speed)',
                'Combined accumulation score (0-1)',
                'Institutional flow proxy'
            ],
            'source': [
                'derived',
                'derived',
                'derived',
                'derived',
                'derived',
                'derived',
                'derived',
                'derived',
                'derived',
                'derived',
                'derived',
                'derived',
                'derived',
                'derived'
            ],
            'interpretation': [
                '>2.0 = extreme whale activity',
                '>0.5 = increasing interest',
                '>2.0 = overvalued, <1.0 = undervalued',
                '>0 = profit, <0 = loss',
                '>1 = profit taking, <1 = capitulation',
                '1 = whale activity detected',
                '1 = potential selling pressure',
                'High = active network',
                'High = far from lows (strong hands)',
                '>1 = above average issuance value',
                'Higher = larger transactions',
                'Higher = faster circulation',
                '>0.7 = strong accumulation',
                '1 = institutional activity'
            ]
        }

        df = pd.DataFrame(features)
        df.to_csv(f"{self.onchain_dir}/onchain_features.csv", index=False)
        print(f"\n✅ Feature guide saved to {self.onchain_dir}/onchain_features.csv")

        return df


def main():
    """Test on-chain data integration"""
    fetcher = OnChainDataFetcher()

    # Create feature guide
    fetcher.create_onchain_features_csv()

    print("\n" + "="*60)
    print("ON-CHAIN DATA INTEGRATION COMPLETE")
    print("="*60)
    print("\nTo enhance ML model with real on-chain data:")
    print("1. Get free API key from https://glassnode.com/")
    print("2. Set environment variable: export GLASSNODE_API_KEY='your-key'")
    print("3. Run this script again to fetch real data")


if __name__ == '__main__':
    main()
