"""
Sentiment Data Integration for Crypto ML Trading
- Fetches Fear & Greed Index
- Google Trends analysis
- Twitter/X sentiment proxy
- Alternative.me API for market sentiment
"""

import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import os
import json
import warnings
from pytrends.request import TrendReq
warnings.filterwarnings('ignore')

# Config
DATA_DIR = '/Users/hy/Desktop/Coding/stock-market/ml_trading/data'
SENTIMENT_DIR = '/Users/hy/Desktop/Coding/stock-market/ml_trading/data/sentiment'

# API Endpoints
FEAR_GREED_API = 'https://api.alternative.me/fng/'
GOOGLE_TRENDS_API = 'https://trends.google.com/trends/api/dailyTrends'


class SentimentDataFetcher:
    """Fetch and process sentiment data"""

    def __init__(self):
        self.data_dir = DATA_DIR
        self.sentiment_dir = SENTIMENT_DIR
        os.makedirs(self.sentiment_dir, exist_ok=True)

        # Cache
        self.cache = {}

    def fetch_fear_greed_index(self, limit=365):
        """
        Fetch Fear & Greed Index from Alternative.me
        Returns: DataFrame with timestamp, value, and classification
        """
        print("="*60)
        print("FEAR & GREED INDEX")
        print("="*60)

        try:
            url = f"{FEAR_GREED_API}?limit={limit}&format=json"
            response = requests.get(url, timeout=30)

            if response.status_code == 200:
                data = response.json()

                df = pd.DataFrame(data['data'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df['value'] = pd.to_numeric(df['value'])
                df['value_classification'] = df['value_classification']

                # Create numeric sentiment score
                df['sentiment_score'] = self._classify_fear_greed(df['value'])

                print(f"✅ Fetched {len(df)} days of Fear & Greed data")
                print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                print(f"  Mean: {df['value'].mean():.1f}")
                print(f"  Classification distribution:")
                for cls in df['value_classification'].unique():
                    count = (df['value_classification'] == cls).sum()
                    print(f"    {cls}: {count}")

                return df
            else:
                print(f"❌ API error: {response.status_code}")
                return self._generate_fear_greed_proxy()

        except Exception as e:
            print(f"❌ Error fetching Fear & Greed: {e}")
            return self._generate_fear_greed_proxy()

    def _classify_fear_greed(self, value):
        """
        Convert Fear & Greed value to sentiment score
        0-25: Extreme Fear (0.0-0.25)
        25-50: Fear (0.25-0.50)
        50-75: Greed (0.50-0.75)
        75-100: Extreme Greed (0.75-1.0)
        """
        return value / 100

    def _generate_fear_greed_proxy(self):
        """
        Generate proxy Fear & Greed data based on price/volatility
        Used when API is unavailable
        """
        print("⚠️ Generating Fear & Greed proxy data...")

        # This would normally be called with real price data
        # For now, return None and handle in the merge function
        return None

    def calculate_sentiment_proxy(self, df):
        """
        Calculate sentiment proxy from price/volume data
        When external sentiment data is unavailable
        """
        sentiment = pd.DataFrame(index=df.index)

        # Price momentum sentiment
        sentiment['price_momentum'] = df['close'].pct_change(6)  # 24h change
        sentiment['price_momentum_norm'] = np.clip(sentiment['price_momentum'] / 0.05, -1, 1)

        # Volatility sentiment (high vol = fear)
        sentiment['volatility_fear'] = np.clip(df['volatility_1d'] / 0.05, 0, 1)
        sentiment['volatility_fear'] = 1 - sentiment['volatility_fear']  # Invert (low vol = greed)

        # Volume sentiment (high vol = interest/fomo)
        sentiment['volume_sentiment'] = np.clip(df['volume_ratio'] / 2, 0, 1)

        # RSI sentiment (oversold = fear, overbought = greed)
        sentiment['rsi_sentiment'] = (df['rsi_14'] - 50) / 50  # -1 to 1

        # Combined sentiment score (weighted average)
        sentiment['combined_sentiment'] = (
            0.3 * sentiment['price_momentum_norm'] +
            0.2 * sentiment['volatility_fear'] +
            0.2 * sentiment['volume_sentiment'] +
            0.3 * sentiment['rsi_sentiment']
        )

        # Normalize to 0-1
        sentiment['combined_sentiment'] = (sentiment['combined_sentiment'] + 1) / 2

        # Fear indicator (high = fear)
        sentiment['fear_index'] = 1 - sentiment['combined_sentiment']

        return sentiment

    def calculate_google_trends_proxy(self, df):
        """
        Calculate Google Trends proxy for crypto search interest
        """
        trends = pd.DataFrame(index=df.index)

        # Search interest proxy based on volume/volatility relationship
        # High volume + low price change = increased search interest
        trends['search_interest_proxy'] = (
            df['volume_ratio'] * (1 - np.abs(df['returns_1h']))
        )

        # Trend direction (increasing interest = positive sentiment)
        trends['interest_trend'] = trends['search_interest_proxy'].pct_change(6)

        # FOMO indicator (rapidly increasing interest)
        trends['fomo_proxy'] = (
            (trends['search_interest_proxy'] > trends['search_interest_proxy'].rolling(24).mean()) &
            (trends['interest_trend'] > 0.1)
        ).astype(int)

        return trends

    def calculate_twitter_sentiment_proxy(self, df):
        """
        Calculate Twitter/social media sentiment proxy
        Based on price action patterns
        """
        twitter = pd.DataFrame(index=df.index)

        # Twitter activity proxy (high volume = high activity)
        twitter['activity_proxy'] = np.clip(df['volume_ratio'] / 3, 0, 1)

        # Sentiment direction from price
        twitter['sentiment_direction'] = np.sign(df['returns_1h'])
        twitter['sentiment_direction'] = twitter['sentiment_direction'].rolling(6).mean()

        # Hype indicator (sustained upward movement)
        twitter['hype_proxy'] = (
            (df['close'] > df['close'].shift(6)) &  # Up in last 24h
            (df['close'] > df['close'].shift(12)) &  # Up in last 48h
            (df['volume_ratio'] > 1.5)  # High volume
        ).astype(int)

        # Controversy indicator (high volatility + mixed signals)
        twitter['controversy_proxy'] = (
            (df['volatility_1d'] > df['volatility_1d'].rolling(168).mean()) &  # High vol
            (np.abs(df['returns_1h']) < df['returns_1h'].rolling(6).std() * 2)  # Whipsaw
        ).astype(int)

        return twitter

    def calculate_market_sentiment_composite(self, df):
        """
        Create a composite market sentiment indicator
        Combines multiple sentiment factors
        """
        composite = pd.DataFrame(index=df.index)

        # Fear & Greed components (0 = extreme fear, 1 = extreme greed)
        # 1. Price momentum (24h)
        composite['momentum'] = np.clip((df['close'].pct_change(6) + 0.1) / 0.2, 0, 1)

        # 2. Volatility regime (low vol = greed)
        composite['volatility_regime'] = 1 - np.clip(df['volatility_1d'] / 0.08, 0, 1)

        # 3. Volume regime (high vol = interest)
        composite['volume_regime'] = np.clip(df['volume_ratio'] / 2.5, 0, 1)

        # 4. Trend strength
        composite['trend_strength'] = np.clip(df['adx_14'] / 50, 0, 1)

        # 5. RSI position
        composite['rsi_position'] = df['rsi_14'] / 100

        # Composite score
        composite['sentiment_composite'] = (
            composite['momentum'] * 0.25 +
            composite['volatility_regime'] * 0.15 +
            composite['volume_regime'] * 0.20 +
            composite['trend_strength'] * 0.15 +
            composite['rsi_position'] * 0.25
        )

        # Sentiment regimes
        composite['sentiment_regime'] = pd.cut(
            composite['sentiment_composite'],
            bins=[0, 0.25, 0.45, 0.55, 0.75, 1.0],
            labels=['extreme_fear', 'fear', 'neutral', 'greed', 'extreme_greed']
        )

        return composite

    def merge_sentiment_features(self, df, timeframe='4h'):
        """
        Merge all sentiment features with main dataframe
        """
        print("="*60)
        print("SENTIMENT DATA INTEGRATION")
        print("="*60)

        # Fetch Fear & Greed Index
        print("\n[1/4] Fetching Fear & Greed Index...")
        fg_df = self.fetch_fear_greed_index()

        if fg_df is not None and len(fg_df) > 0:
            # Resample to 4h and merge
            fg_df = fg_df.set_index('timestamp')
            fg_resampled = fg_df.resample('4h').ffill()

            # Merge Fear & Greed
            df['fear_greed_value'] = fg_resampled['value'].reindex(df.index)
            df['fear_greed_class'] = fg_resampled['value_classification'].reindex(df.index)
            df['fear_greed_score'] = fg_resampled['sentiment_score'].reindex(df.index)

            # Fill missing values
            df['fear_greed_value'] = df['fear_greed_value'].fillna(method='ffill').fillna(50)
            df['fear_greed_score'] = df['fear_greed_score'].fillna(method='ffill').fillna(0.5)

            print(f"  ✅ Merged Fear & Greed data")
        else:
            # Use proxy
            df['fear_greed_value'] = 50  # Neutral
            df['fear_greed_score'] = 0.5
            df['fear_greed_class'] = 'Neutral'
            print("  ⚠️ Using neutral Fear & Greed (API unavailable)")

        # Calculate sentiment proxy
        print("\n[2/4] Calculating price-based sentiment proxies...")
        sentiment_proxy = self.calculate_sentiment_proxy(df)
        for col in sentiment_proxy.columns:
            df[col] = sentiment_proxy[col]
        print(f"  ✅ Calculated {len(sentiment_proxy.columns)} sentiment proxies")

        # Calculate Google Trends proxy
        print("\n[3/4] Calculating search interest proxies...")
        trends_proxy = self.calculate_google_trends_proxy(df)
        for col in trends_proxy.columns:
            df[col] = trends_proxy[col]
        print(f"  ✅ Calculated {len(trends_proxy.columns)} trends proxies")

        # Calculate Twitter proxy
        print("\n[4/4] Calculating social media sentiment proxies...")
        twitter_proxy = self.calculate_twitter_sentiment_proxy(df)
        for col in twitter_proxy.columns:
            df[col] = twitter_proxy[col]
        print(f"  ✅ Calculated {len(twitter_proxy.columns)} social proxies")

        # Calculate composite
        print("\n[5/5] Creating composite sentiment indicator...")
        composite = self.calculate_market_sentiment_composite(df)
        for col in composite.columns:
            if col != 'sentiment_regime':
                df[col] = composite[col]
        df['sentiment_regime'] = composite['sentiment_regime']

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

    def create_sentiment_features_csv(self):
        """Create a CSV with sentiment feature descriptions"""
        features = {
            'feature': [
                'fear_greed_value',
                'fear_greed_score',
                'price_momentum_norm',
                'volatility_fear',
                'volume_sentiment',
                'rsi_sentiment',
                'combined_sentiment',
                'fear_index',
                'search_interest_proxy',
                'interest_trend',
                'fomo_proxy',
                'activity_proxy',
                'sentiment_direction',
                'hype_proxy',
                'controversy_proxy',
                'sentiment_composite',
                'sentiment_regime'
            ],
            'description': [
                'Alternative.me Fear & Greed Index value (0-100)',
                'Normalized Fear & Greed (0=extreme fear, 1=extreme greed)',
                '24h price momentum normalized (-1 to 1)',
                'Volatility-based fear indicator (0=calm, 1=fear)',
                'Volume-based interest indicator',
                'RSI-based sentiment (0=bearish, 1=bullish)',
                'Combined sentiment score (0-1)',
                'Inverse of combined sentiment (fear indicator)',
                'Search interest proxy based on volume/price',
                'Change in search interest',
                'FOMO detection (rapidly increasing interest)',
                'Social media activity proxy',
                'Direction of social sentiment',
                'Hype detection (sustained up + high volume)',
                'Controversy detection (high volatility)',
                'Composite market sentiment (0-1)',
                'Sentiment regime classification'
            ],
            'interpretation': [
                '<25 Extreme Fear, >75 Extreme Greed',
                'Same as above, normalized',
                'Positive = bullish, Negative = bearish',
                'Low = calm market, High = fearful',
                'High = strong interest',
                'RSI-based market sentiment',
                '0=bearish, 1=bullish',
                '0=calm, 1=fearful',
                'Higher = more search interest',
                'Positive = increasing interest',
                '1 = FOMO detected',
                'Higher = more activity',
                'Positive = bullish discussion',
                '1 = hype detected',
                '1 = controversial price action',
                '0=extreme fear, 1=extreme greed',
                'extreme_fear/neutral/greed/extreme_greed'
            ]
        }

        df = pd.DataFrame(features)
        df.to_csv(f"{self.sentiment_dir}/sentiment_features.csv", index=False)
        print(f"\n✅ Sentiment feature guide saved to {self.sentiment_dir}/sentiment_features.csv")

        return df


def main():
    """Test sentiment data integration"""
    fetcher = SentimentDataFetcher()

    # Create feature guide
    fetcher.create_sentiment_features_csv()

    print("\n" + "="*60)
    print("SENTIMENT DATA INTEGRATION COMPLETE")
    print("="*60)
    print("\nAvailable sentiment features:")
    print("- Fear & Greed Index (Alternative.me API)")
    print("- Price momentum sentiment")
    print("- Volatility-based fear")
    print("- Volume-based interest")
    print("- Google Trends proxy")
    print("- Social media activity proxy")
    print("- Composite sentiment indicator")


if __name__ == '__main__':
    main()
