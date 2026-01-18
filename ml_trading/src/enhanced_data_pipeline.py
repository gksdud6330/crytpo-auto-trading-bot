"""
Enhanced Data Pipeline
- Combines technical, on-chain, and sentiment features
- Creates comprehensive feature set for ML model
- Saves enhanced dataset for training
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# Config
DATA_DIR = '/Users/hy/Desktop/Coding/stock-market/ml_trading/data'
ENHANCED_DIR = '/Users/hy/Desktop/Coding/stock-market/ml_trading/data/enhanced'

# Import our data modules
import sys
sys.path.append('/Users/hy/Desktop/Coding/stock-market/ml_trading/src')


class EnhancedDataPipeline:
    """
    Complete data pipeline that combines:
    1. Technical indicators (price-based)
    2. On-chain metrics (blockchain data)
    3. Sentiment data (market psychology)
    """

    def __init__(self):
        self.data_dir = DATA_DIR
        self.enhanced_dir = ENHANCED_DIR
        os.makedirs(self.enhanced_dir, exist_ok=True)

        # All feature columns
        self.technical_features = [
            'rsi_14', 'rsi_7', 'rsi_21',
            'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
            'ema_9', 'ema_21', 'ema_50', 'ema_200', 'sma_50',
            'atr_14', 'atr_pct',
            'volume_sma_20', 'volume_ratio',
            'stoch_k', 'stoch_d', 'adx_14',
            'returns_1h', 'returns_4h', 'returns_1d', 'returns_1w',
            'volatility_1d', 'volatility_1w',
            'price_vs_ema_50', 'price_vs_ema_200'
        ]

        self.onchain_features = [
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
        ]

        self.sentiment_features = [
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
            'sentiment_composite'
        ]

        self.all_features = (
            self.technical_features +
            self.onchain_features +
            self.sentiment_features
        )

    def load_base_data(self, timeframe='4h'):
        """Load base OHLCV data with technical indicators"""
        print("="*60)
        print("ENHANCED DATA PIPELINE")
        print("="*60)

        all_data = []
        for filename in os.listdir(f"{self.data_dir}/{timeframe}"):
            if filename.endswith('.parquet'):
                df = pd.read_parquet(f"{self.data_dir}/{timeframe}/{filename}")
                pair = filename.replace('.parquet', '').replace('_', '/')
                df['pair'] = pair
                all_data.append(df)

        combined = pd.concat(all_data, ignore_index=True)
        print(f"\n✅ Loaded {len(combined)} samples from {len(all_data)} pairs")

        return combined

    def add_onchain_features(self, df):
        """Add on-chain derived features"""
        print("\n[1/3] Adding on-chain features...")

        # On-chain derived metrics
        df = df.copy()

        # Volume-based whale proxy
        df['volume_sma_24h'] = df['volume'].rolling(window=6).mean()
        df['volume_spike'] = df['volume'] / df['volume_sma_24h']
        df['volume_trend'] = df['volume'].pct_change(6)

        # MVRV proxy
        df['cost_basis_7d'] = df['close'].rolling(window=42).mean()
        df['mvrv_proxy'] = df['close'] / df['cost_basis_7d']

        # NUPL proxy
        df['avg_cost_30d'] = df['close'].rolling(window=180).mean()
        df['nupl_proxy'] = (df['close'] - df['avg_cost_30d']) / df['close']

        # SOPR proxy
        df['sopr_proxy'] = df['close'] / df['close'].shift(6)

        # Whale activity
        df['whale_activity'] = (df['volume'] > df['volume'].rolling(24).quantile(0.95)).astype(int)

        # Exchange inflow proxy
        df['exchange_inflow_proxy'] = (df['volume_ratio'] > 1.5).astype(int)

        # Active proxy
        df['active_proxy'] = (df['volume_ratio'] * df['volatility_1d']).rolling(6).mean()

        # HODL proxy
        df['hodl_proxy'] = df['close'] / df['close'].rolling(180).min()

        # Puell proxy
        df['puell_proxy'] = df['close'] / df['close'].rolling(876).mean()

        # Network metrics
        df['volume_per_tx'] = df['volume'] / (df['volume'] / df['close'].rolling(6).mean())
        df['velocity'] = df['volume'].rolling(6).sum() / df['close']

        # Accumulation score
        df['accumulation_score'] = (
            (df['close'].pct_change(6) > 0).astype(int) +
            (df['volume_ratio'] > 1.2).astype(int) +
            (df['price_vs_ema_50'] > 0).astype(int)
        ) / 3

        # Institutional proxy
        df['institutional_proxy'] = (
            (df['volume'] > df['volume'].rolling(24).quantile(0.8)).astype(int) *
            (df['close'] > df['ema_50']).astype(int)
        )

        print(f"  ✅ Added {len(self.onchain_features)} on-chain features")

        return df

    def add_sentiment_features(self, df):
        """Add sentiment features"""
        print("\n[2/3] Adding sentiment features...")

        df = df.copy()

        # Fear & Greed proxy (when API unavailable)
        df['fear_greed_value'] = 50  # Neutral default
        df['fear_greed_score'] = 0.5

        # Price momentum sentiment
        df['price_momentum_norm'] = np.clip(df['returns_1h'] / 0.05, -1, 1)

        # Volatility fear
        df['volatility_fear'] = 1 - np.clip(df['volatility_1d'] / 0.05, 0, 1)

        # Volume sentiment
        df['volume_sentiment'] = np.clip(df['volume_ratio'] / 2, 0, 1)

        # RSI sentiment
        df['rsi_sentiment'] = (df['rsi_14'] - 50) / 50

        # Combined sentiment
        df['combined_sentiment'] = (
            0.3 * df['price_momentum_norm'] +
            0.2 * df['volatility_fear'] +
            0.2 * df['volume_sentiment'] +
            0.3 * df['rsi_sentiment']
        )
        df['combined_sentiment'] = (df['combined_sentiment'] + 1) / 2

        # Fear index
        df['fear_index'] = 1 - df['combined_sentiment']

        # Google Trends proxy
        df['search_interest_proxy'] = df['volume_ratio'] * (1 - np.abs(df['returns_1h']))
        df['interest_trend'] = df['search_interest_proxy'].pct_change(6)
        df['fomo_proxy'] = (
            (df['search_interest_proxy'] > df['search_interest_proxy'].rolling(24).mean()) &
            (df['interest_trend'] > 0.1)
        ).astype(int)

        # Twitter proxy
        df['activity_proxy'] = np.clip(df['volume_ratio'] / 3, 0, 1)
        df['sentiment_direction'] = np.sign(df['returns_1h']).rolling(6).mean()
        df['hype_proxy'] = (
            (df['close'] > df['close'].shift(6)) &
            (df['close'] > df['close'].shift(12)) &
            (df['volume_ratio'] > 1.5)
        ).astype(int)
        df['controversy_proxy'] = (
            (df['volatility_1d'] > df['volatility_1d'].rolling(168).mean()) &
            (np.abs(df['returns_1h']) < df['returns_1h'].rolling(6).std() * 2)
        ).astype(int)

        # Composite sentiment
        df['momentum'] = np.clip((df['returns_1h'] + 0.1) / 0.2, 0, 1)
        df['volatility_regime'] = 1 - np.clip(df['volatility_1d'] / 0.08, 0, 1)
        df['volume_regime'] = np.clip(df['volume_ratio'] / 2.5, 0, 1)
        df['trend_strength'] = np.clip(df['adx_14'] / 50, 0, 1)
        df['rsi_position'] = df['rsi_14'] / 100
        df['sentiment_composite'] = (
            df['momentum'] * 0.25 +
            df['volatility_regime'] * 0.15 +
            df['volume_regime'] * 0.20 +
            df['trend_strength'] * 0.15 +
            df['rsi_position'] * 0.25
        )

        print(f"  ✅ Added {len(self.sentiment_features)} sentiment features")

        return df

    def clean_features(self, df):
        """Clean and prepare features for ML"""
        print("\n[3/3] Cleaning features...")

        df = df.copy()

        # Handle infinite values
        for col in self.all_features:
            if col in df.columns:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)

        # Fill NaN with 0
        for col in self.all_features:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # Drop rows with NaN target
        df = df.dropna(subset=['target'])

        print(f"  ✅ Final dataset: {len(df)} samples, {len(self.all_features)} features")

        return df

    def run_pipeline(self, timeframe='4h'):
        """Run complete data pipeline"""
        # Load base data
        df = self.load_base_data(timeframe)

        # Add on-chain features
        df = self.add_onchain_features(df)

        # Add sentiment features
        df = self.add_sentiment_features(df)

        # Clean features
        df = self.clean_features(df)

        # Summary
        print("\n" + "="*60)
        print("FEATURE SUMMARY")
        print("="*60)

        print(f"\n📊 Technical Features: {len(self.technical_features)}")
        for i, feat in enumerate(self.technical_features[:5], 1):
            print(f"  {i}. {feat}")
        print(f"  ... and {len(self.technical_features) - 5} more")

        print(f"\n📊 On-Chain Features: {len(self.onchain_features)}")
        for i, feat in enumerate(self.onchain_features, 1):
            print(f"  {i}. {feat}")

        print(f"\n📊 Sentiment Features: {len(self.sentiment_features)}")
        for i, feat in enumerate(self.sentiment_features, 1):
            print(f"  {i}. {feat}")

        print(f"\n📊 Total Features: {len(self.all_features)}")

        # Save enhanced data
        print("\n" + "="*60)
        print("SAVING ENHANCED DATA")
        print("="*60)

        for pair in df['pair'].unique():
            pair_df = df[df['pair'] == pair]
            filename = pair.replace('/', '_')
            pair_df.to_parquet(f"{self.enhanced_dir}/{filename}_enhanced.parquet")
            print(f"  ✅ Saved {pair}: {len(pair_df)} rows")

        # Create feature list file
        with open(f"{self.enhanced_dir}/enhanced_features.txt", 'w') as f:
            f.write("ENHANCED FEATURE LIST\n")
            f.write("="*60 + "\n\n")
            f.write("TECHNICAL FEATURES (price-based)\n")
            f.write("-"*40 + "\n")
            for feat in self.technical_features:
                f.write(f"  {feat}\n")
            f.write("\nON-CHAIN FEATURES (blockchain metrics)\n")
            f.write("-"*40 + "\n")
            for feat in self.onchain_features:
                f.write(f"  {feat}\n")
            f.write("\nSENTIMENT FEATURES (market psychology)\n")
            f.write("-"*40 + "\n")
            for feat in self.sentiment_features:
                f.write(f"  {feat}\n")

        print(f"\n✅ Feature list saved to {self.enhanced_dir}/enhanced_features.txt")

        # Target distribution
        print("\n" + "="*60)
        print("TARGET DISTRIBUTION")
        print("="*60)
        print(f"  Class 0 (HOLD): {(df['target'] == 0).sum()} ({(df['target'] == 0).mean():.1%})")
        print(f"  Class 1 (BUY):  {(df['target'] == 1).sum()} ({(df['target'] == 1).mean():.1%})")

        return df

    def get_feature_importance_sample(self):
        """Generate sample feature importance calculation"""
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*60)

        # Load enhanced data
        all_data = []
        for filename in os.listdir(self.enhanced_dir):
            if filename.endswith('.parquet'):
                df = pd.read_parquet(f"{self.enhanced_dir}/{filename}")
                all_data.append(df)

        combined = pd.concat(all_data, ignore_index=True)

        # Calculate correlation with target
        correlations = {}
        for feat in self.all_features:
            if feat in combined.columns:
                corr = combined[feat].corr(combined['target'])
                if not np.isnan(corr):
                    correlations[feat] = abs(corr)

        # Sort by correlation
        sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)

        print("\n🔝 Top 15 Features by Correlation with Target:")
        print("-"*50)
        for i, (feat, corr) in enumerate(sorted_features[:15], 1):
            bar = "█" * int(corr * 50)
            print(f"  {i:2d}. {feat:30s} {corr:.4f} {bar}")

        return sorted_features


def main():
    pipeline = EnhancedDataPipeline()
    df = pipeline.run_pipeline(timeframe='4h')
    pipeline.get_feature_importance_sample()


if __name__ == '__main__':
    main()
