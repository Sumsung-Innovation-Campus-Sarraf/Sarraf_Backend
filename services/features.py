"""
Gold Feature Calculator - FIXED FOR MODEL EXPECTATIONS
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import requests
import warnings
from typing import Dict, List, Optional, Tuple, Any
import json
import os
from pathlib import Path
from dotenv import load_dotenv
import logging
from supabase import create_client, Client

warnings.filterwarnings('ignore')
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GoldFeatureCalculator:
    """Gold feature calculator - MATCHING MODEL EXPECTATIONS"""
    
    def __init__(self, model_loader=None):
        self.model_loader = model_loader
        
        # Initialize Supabase client
        self.supabase_url = os.getenv("SUPABASE_URL", "")
        self.supabase_key = os.getenv("SUPABASE_KEY", "")
        
        if not self.supabase_url or not self.supabase_key:
            logger.error("Supabase URL or Key not found in environment variables")
            raise ValueError("Please set SUPABASE_URL and SUPABASE_KEY in your .env file")
        
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
        
        # Gold-API configuration
        self.gold_api_key = os.getenv("GOLD_API_KEY", "")
        
        # Market indicators tickers
        self.TICKERS = {
            'vix': '^VIX',
            'dxy': 'DX-Y.NYB',
            'sp500': '^GSPC',
            'nasdaq': '^IXIC',
            'treasury_10y': '^TNX',
            'treasury_2y': '^FVX',
            'crude_oil': 'CL=F',
            'copper': 'HG=F',
        }
        
        # Define the EXACT features your model expects
        # CORRECT MODEL_FEATURES list - MATCHING WHAT YOU RENAMED IN DATABASE
        self.MODEL_FEATURES = [
            'silver_price_usd', 'vix', 'dxy', 'sp500', 'nasdaq', 
            'treasury_10y', 'treasury_2y', 'crude_oil', 'copper', 
            'gdp_growth', 'inflation_rate', 'fed_funds_rate', 
            'year', 'month', 'quarter', 'day_of_week', 'day_of_year', 
            'month_sin', 'month_cos', 'season', 'gold_silver_ratio', 
            'vix_spike', 'vix_surge', 'real_yield_10y', 'yield_curve', 
            'yield_curve_inverted', 'high_inflation', 'recession', 
            'fed_funds_change', 'rate_hike', 'rate_cut', 
            'hist_vol_5d', 'hist_vol_10d', 'hist_vol_20d', 'hist_vol_30d', 'hist_vol_60d', 
            'vol_ratio_5_20', 'vol_ratio_10_30', 
            'return_10d', 'return_20d',  # ✅ CORRECT - after renaming
            'price_to_ma_20', 'price_to_ma_50', 'price_to_ma_200',  # ✅ CORRECT - after renaming
            'vix_lag1', 'vix_lag7', 'vix_change_5d', 'vix_high', 
            'dxy_lag1', 'dxy_return_5d', 'dxy_vol_20d',  # ✅ CORRECT - after renaming
            'treasury_10y_dec', 'inflation_rate_dec', 
            'real_yield', 'real_yield_change', 
            'sp500_vol_20d', 'sp500_return_5d',  # ✅ CORRECT - after renaming
            'large_move_count_5d', 'max_drawdown_20d'  # ✅ ADD THESE!
            ]
        
        logger.info("Gold Feature Calculator initialized for model expectations")
    
    # Keep all the fetch methods the same as before...
    def fetch_gold_price(self) -> Optional[Dict]:
        """Fetch gold price from Gold-API"""
        if not self.gold_api_key:
            return self._fetch_from_yahoo("GC=F")
        
        try:
            url = "https://api.gold-api.com/price/XAU"
            headers = {"x-api-key": self.gold_api_key}
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'symbol': 'XAU',
                    'price': float(data.get('price', 0)),
                    'timestamp': data.get('timestamp'),
                    'source': 'gold-api'
                }
        except Exception as e:
            logger.error(f"Error fetching from Gold-API: {e}")
        
        return self._fetch_from_yahoo("GC=F")
    
    def fetch_silver_price(self) -> Optional[Dict]:
        """Fetch silver price from Gold-API"""
        if not self.gold_api_key:
            return self._fetch_from_yahoo("SI=F")
        
        try:
            url = "https://api.gold-api.com/price/XAG"
            headers = {"x-api-key": self.gold_api_key}
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'symbol': 'XAG',
                    'price': float(data.get('price', 0)),
                    'timestamp': data.get('timestamp'),
                    'source': 'gold-api'
                }
        except Exception as e:
            logger.error(f"Error fetching silver from Gold-API: {e}")
        
        return self._fetch_from_yahoo("SI=F")
    
    def _fetch_from_yahoo(self, symbol: str) -> Optional[Dict]:
        """Fetch data from Yahoo Finance"""
        try:
            data = yf.download(symbol, period="1d", progress=False)
            
            if not data.empty:
                return {
                    'symbol': symbol,
                    'price': float(data['Close'].iloc[-1]),
                    'timestamp': datetime.now().isoformat(),
                    'source': 'yfinance'
                }
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
        
        return None
    
    def fetch_market_indicators(self) -> Dict:
        """Fetch market indicators from Yahoo Finance"""
        indicators = {}
        
        logger.info(f"\nDownloading {len(self.TICKERS)} market indicators...")
        
        for name, ticker in self.TICKERS.items():
            try:
                logger.info(f"  Downloading {ticker}...")
                
                data = yf.download(ticker, period="5d", progress=False)
                
                if data.empty:
                    logger.warning(f"    ⚠️ No data for {ticker}")
                    indicators[name] = None
                    continue
                
                indicators[name] = float(data['Close'].iloc[-1])
                logger.info(f"    ✓ {ticker}: ${indicators[name]:.2f}")
                
            except Exception as e:
                logger.error(f"    ✗ Error downloading {ticker}: {str(e)[:100]}")
                indicators[name] = None
        
        # Adjust treasury yields from percentage to decimal
        if indicators.get('treasury_10y'):
            indicators['treasury_10y'] = indicators['treasury_10y'] / 100
        if indicators.get('treasury_2y'):
            indicators['treasury_2y'] = indicators['treasury_2y'] / 100
        
        return indicators
    
    def fetch_economic_indicators(self) -> Dict:
        """Fetch economic indicators from IMF API"""
        indicators = {}
        
        try:
            # GDP from IMF
            logger.info("Fetching GDP data from IMF API...")
            gdp_response = requests.get(
                "https://www.imf.org/external/datamapper/api/v1/NGDP_RPCH/WEOWORLD",
                timeout=30
            )
            
            if gdp_response.status_code == 200:
                gdp_data = gdp_response.json()
                if 'values' in gdp_data and 'NGDP_RPCH' in gdp_data['values']:
                    gdp_dict = gdp_data['values']['NGDP_RPCH']['WEOWORLD']
                    latest_year = max([int(k) for k in gdp_dict.keys() if k.isdigit()])
                    indicators['gdp_growth'] = float(gdp_dict[str(latest_year)])
                    logger.info(f"✓ GDP growth ({latest_year}): {indicators['gdp_growth']:.1f}%")
                else:
                    indicators['gdp_growth'] = 3.2
                    logger.warning("⚠️ Using default GDP growth: 3.2%")
            else:
                indicators['gdp_growth'] = 3.2
                logger.warning("⚠️ Using default GDP growth: 3.2%")
            
            # Inflation from IMF
            logger.info("Fetching inflation data from IMF API...")
            inflation_response = requests.get(
                "https://www.imf.org/external/datamapper/api/v1/PCPIEPCH/USA",
                timeout=30
            )
            
            if inflation_response.status_code == 200:
                inflation_data = inflation_response.json()
                if 'values' in inflation_data and 'PCPIEPCH' in inflation_data['values']:
                    inflation_dict = inflation_data['values']['PCPIEPCH']['USA']
                    latest_year = max([int(k) for k in inflation_dict.keys() if k.isdigit()])
                    indicators['inflation_rate'] = float(inflation_dict[str(latest_year)])
                    logger.info(f"✓ Inflation rate ({latest_year}): {indicators['inflation_rate']:.1f}%")
                else:
                    indicators['inflation_rate'] = 2.7
                    logger.warning("⚠️ Using default inflation rate: 2.7%")
            else:
                indicators['inflation_rate'] = 2.7
                logger.warning("⚠️ Using default inflation rate: 2.7%")
            
            # Fed funds rate from Yahoo
            logger.info("Fetching Fed funds rate...")
            fed_data = yf.download("^IRX", period="1d", progress=False)
            if not fed_data.empty:
                indicators['fed_funds_rate'] = float(fed_data['Close'].iloc[-1]) / 100
                logger.info(f"✓ Fed funds rate: {indicators['fed_funds_rate']:.2%}")
            else:
                indicators['fed_funds_rate'] = 0.0533
                logger.warning("⚠️ Using default Fed funds rate: 5.33%")
            
            return indicators
            
        except Exception as e:
            logger.error(f"❌ Error fetching economic indicators: {e}")
            return {
                'gdp_growth': 3.2,
                'inflation_rate': 2.7,
                'fed_funds_rate': 0.0533
            }
    
    def check_supabase_data(self, date: str) -> bool:
        """Check if data exists in Supabase for a date"""
        try:
            response = self.supabase.table("gold_silver_dataset") \
                .select("date") \
                .eq("date", date) \
                .execute()
            
            return len(response.data) > 0
            
        except Exception as e:
            logger.error(f"Error checking Supabase: {e}")
            return False
    
    def get_historical_data(self, days: int = 200) -> pd.DataFrame:
        """Get historical data from Supabase for calculations"""
        try:
            start_date = (datetime.now() - timedelta(days=days)).date().isoformat()
            
            response = self.supabase.table("gold_silver_dataset") \
                .select("*") \
                .gte("date", start_date) \
                .order("date") \
                .execute()
            
            if not response.data:
                logger.warning("No historical data found in Supabase")
                return pd.DataFrame()
            
            df = pd.DataFrame(response.data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            
            logger.info(f"📈 Loaded {len(df)} days of historical data from Supabase")
            return df
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            return pd.DataFrame()
    
  

    def calculate_time_series_features(self, historical_df: pd.DataFrame, current_prices: Dict = None) -> Dict:
        """Calculate time-series features - NOW MATCHING MODEL EXPECTATIONS"""
        features = {}
        
        if historical_df.empty:
            logger.warning("No historical data for time-series calculations")
            return self._get_default_time_series_features()
        
        df = historical_df.copy()
        
        # Add current price if available
        if current_prices and 'gold_price_usd' in current_prices:
            current_gold = current_prices['gold_price_usd']
            if 'gold_price_usd' in df.columns:
                last_date = df.index[-1]
                new_date = last_date + pd.Timedelta(days=1)
                df.loc[new_date, 'gold_price_usd'] = current_gold
        
        # 1. Calculate returns for volatility
        if 'gold_price_usd' in df.columns:
            gold_returns = df['gold_price_usd'].pct_change().dropna()
            
            # Historical volatility
            for window in [5, 10, 20, 30, 60]:
                if len(gold_returns) >= max(3, window//2):
                    vol = gold_returns.rolling(window, min_periods=max(3, window//2)).std() * np.sqrt(252)
                    features[f'hist_vol_{window}d'] = float(vol.iloc[-1]) if not pd.isna(vol.iloc[-1]) else 0.15
                else:
                    features[f'hist_vol_{window}d'] = 0.15
            
            # Volatility ratios
            vol_5d = features.get('hist_vol_5d', 0.15)
            vol_20d = features.get('hist_vol_20d', 0.15)
            vol_10d = features.get('hist_vol_10d', 0.15)
            vol_30d = features.get('hist_vol_30d', 0.15)
            
            features['vol_ratio_5_20'] = float(vol_5d / vol_20d) if vol_20d > 0 else 1.0
            features['vol_ratio_10_30'] = float(vol_10d / vol_30d) if vol_30d > 0 else 1.0
            
            # Returns (10d and 20d for model)
            for days in [10, 20]:  # Changed from [30, 60]
                if len(df) >= days + 1:
                    return_val = df['gold_price_usd'].pct_change(days).iloc[-1]
                    features[f'return_{days}d'] = float(return_val) if not pd.isna(return_val) else 0.01
                else:
                    features[f'return_{days}d'] = 0.01
            
            # Price to moving averages (20d, 50d, 200d for model)
            for ma_days in [20, 50, 200]:  # Changed from [30, 60]
                if len(df) >= ma_days:
                    ma = df['gold_price_usd'].rolling(ma_days, min_periods=max(5, ma_days//4)).mean().iloc[-1]
                    current_price = df['gold_price_usd'].iloc[-1]
                    if ma > 0:
                        features[f'price_to_ma_{ma_days}'] = float(current_price / ma)
                    else:
                        features[f'price_to_ma_{ma_days}'] = 1.0
                else:
                    features[f'price_to_ma_{ma_days}'] = 1.0
            
            # Calculate max drawdown for last 20 days
            if len(df) >= 20:
                rolling_max = df['gold_price_usd'].rolling(20, min_periods=1).max()
                drawdown = (df['gold_price_usd'] - rolling_max) / rolling_max
                features['max_drawdown_20d'] = float(drawdown.iloc[-1]) if not pd.isna(drawdown.iloc[-1]) else 0.0
            else:
                features['max_drawdown_20d'] = 0.0
            
            # Calculate large move count (absolute return > 2% in last 5 days)
            if len(df) >= 6:
                recent_returns = df['gold_price_usd'].pct_change().iloc[-5:].abs()
                features['large_move_count_5d'] = int((recent_returns > 0.02).sum())
            else:
                features['large_move_count_5d'] = 0
        
        # 2. DXY features (5d return, 20d vol for model)
        if 'dxy' in df.columns and len(df) > 1:
            # DXY 5-day return
            if len(df) >= 6:
                dxy_return = df['dxy'].pct_change(5).iloc[-1]
                features['dxy_return_5d'] = float(dxy_return) if not pd.isna(dxy_return) else 0.0
            else:
                features['dxy_return_5d'] = 0.0
            
            # DXY 20-day volatility
            if 'dxy' in df.columns:
                dxy_returns = df['dxy'].pct_change().dropna()
                if len(dxy_returns) >= 20:
                    vol = dxy_returns.rolling(20, min_periods=10).std().iloc[-1]
                    features['dxy_vol_20d'] = float(vol) if not pd.isna(vol) else 0.0
                else:
                    features['dxy_vol_20d'] = 0.0
        
        # 3. S&P 500 features (5d return, 20d vol for model)
        if 'sp500' in df.columns:
            # S&P 500 5-day return
            if len(df) >= 6:
                sp_return = df['sp500'].pct_change(5).iloc[-1]
                features['sp500_return_5d'] = float(sp_return) if not pd.isna(sp_return) else 0.01
            else:
                features['sp500_return_5d'] = 0.01
            
            # S&P 500 20-day volatility
            if 'sp500' in df.columns:
                sp_returns = df['sp500'].pct_change().dropna()
                if len(sp_returns) >= 20:
                    vol = sp_returns.rolling(20, min_periods=10).std().iloc[-1]
                    features['sp500_vol_20d'] = float(vol) if not pd.isna(vol) else 0.15
                else:
                    features['sp500_vol_20d'] = 0.15
        
        # ... rest of the method stays the same for VIX, etc.
        
        logger.info(f"📊 Calculated {len(features)} time-series features for model")
        return features
    
    def _get_default_time_series_features(self) -> Dict:
        """Get default time-series features for model"""
        return {
                'hist_vol_5d': 0.15,
                'hist_vol_10d': 0.15,
                'hist_vol_20d': 0.15,
                'hist_vol_30d': 0.15,
                'hist_vol_60d': 0.15,
                'vol_ratio_5_20': 1.0,
                'vol_ratio_10_30': 1.0,
                'return_10d': 0.01,  # Changed from return_30d
                'return_20d': 0.01,  # Changed from return_60d
                'price_to_ma_20': 1.0,  # Changed from price_to_ma_30
                'price_to_ma_50': 1.0,  # Changed from price_to_ma_60
                'price_to_ma_200': 1.0,  # New
                'vix_lag1': 16.5,
                'vix_lag7': 16.5,
                'vix_change_5d': 0.0,
                'vix_high': False,
                'dxy_lag1': 103.5,
                'dxy_return_5d': 0.0,  # Changed from dxy_return_30d
                'dxy_vol_20d': 0.0,   # Changed from dxy_vol_60d
                'real_yield_change': 0.0,
                'sp500_vol_20d': 0.15,  # Changed from sp500_vol_60d
                'sp500_return_5d': 0.01, # Changed from sp500_return_30d
                'large_move_count_5d': 0,  # New
                'max_drawdown_20d': 0.0,   # New
                }
    
    def prepare_features_for_model(self, features: Dict) -> Dict:
        """Prepare features exactly as model expects"""
        # Start with base features
        model_features = {}
        
        # Add gold price (required by model even if not in feature list)
        if 'gold_price_usd' in features:
            model_features['gold_price_usd'] = float(features.get('gold_price_usd', 2100.0))
        
        # Add silver price (first feature in model list)
        model_features['silver_price_usd'] = float(features.get('silver_price_usd', 26.25))
        
        # Add all other features in the EXACT order/model expects
        for feature in self.MODEL_FEATURES:
            if feature in features:
                value = features[feature]
                if value is None:
                    # Provide reasonable defaults
                    if 'vol' in feature or 'hist_vol' in feature:
                        model_features[feature] = 0.15
                    elif 'return' in feature:
                        model_features[feature] = 0.01
                    elif 'price_to_ma' in feature:
                        model_features[feature] = 1.0
                    elif 'vix' in feature and 'lag' not in feature and 'change' not in feature:
                        model_features[feature] = 16.5
                    elif 'dxy' in feature and 'lag' not in feature and 'return' not in feature and 'vol' not in feature:
                        model_features[feature] = 103.5
                    elif 'sp500' in feature and 'vol' not in feature and 'return' not in feature:
                        model_features[feature] = 4950.0
                    elif 'treasury' in feature:
                        model_features[feature] = 0.0425
                    elif 'inflation' in feature:
                        model_features[feature] = 2.7
                    elif 'fed_funds' in feature:
                        model_features[feature] = 0.0533
                    elif 'gdp' in feature:
                        model_features[feature] = 3.2
                    elif 'gold_silver_ratio' in feature:
                        model_features[feature] = 80.0
                    elif 'real_yield' in feature:
                        model_features[feature] = 2.0
                    elif 'yield_curve' in feature:
                        model_features[feature] = 0.0
                    elif isinstance(feature, bool) or feature.endswith('_inverted') or feature.endswith('_spike') or feature.endswith('_surge') or feature.endswith('_high') or feature.endswith('_hike') or feature.endswith('_cut'):
                        model_features[feature] = False
                    else:
                        model_features[feature] = 0.0
                else:
                    # Convert to proper type
                    if isinstance(value, bool):
                        model_features[feature] = bool(value)
                    elif isinstance(value, (int, float)):
                        if 'dec' in feature:
                            model_features[feature] = float(value)
                        elif 'year' in feature or 'month' in feature or 'quarter' in feature or 'day_' in feature or 'season' in feature:
                            model_features[feature] = int(value)
                        else:
                            model_features[feature] = float(value)
                    else:
                        model_features[feature] = value
            else:
                # Feature not in dictionary, use default
                model_features[feature] = self._get_feature_default(feature)
        
        # Ensure all derived features are calculated
        model_features = self._calculate_derived_features(model_features)
        
        logger.info(f"✅ Prepared {len(model_features)} features for model")
        return model_features
    
    def _get_feature_default(self, feature: str) -> Any:
        """Get default value for a feature"""
        defaults = {
        'vix': 16.5,
        'dxy': 103.5,
        'sp500': 4950.0,
        'nasdaq': 17500.0,
        'treasury_10y': 0.0425,
        'treasury_2y': 0.0455,
        'crude_oil': 75.0,
        'copper': 4.2,
        'gdp_growth': 3.2,
        'inflation_rate': 2.7,
        'fed_funds_rate': 0.0533,
        'year': datetime.now().year,
        'month': datetime.now().month,
        'quarter': (datetime.now().month - 1) // 3 + 1,
        'day_of_week': datetime.now().weekday(),
        'day_of_year': datetime.now().timetuple().tm_yday,
        'month_sin': float(np.sin(2 * np.pi * datetime.now().month / 12)),
        'month_cos': float(np.cos(2 * np.pi * datetime.now().month / 12)),
        'season': self._get_season(datetime.now().month),
        'gold_silver_ratio': 80.0,
        'vix_spike': False,
        'vix_surge': False,
        'real_yield_10y': 2.0,
        'yield_curve': 0.0,
        'yield_curve_inverted': False,
        'high_inflation': False,
        'recession': False,
        'fed_funds_change': 0.0,
        'rate_hike': False,
        'rate_cut': False,
        'hist_vol_5d': 0.15,
        'hist_vol_10d': 0.15,
        'hist_vol_20d': 0.15,
        'hist_vol_30d': 0.15,
        'hist_vol_60d': 0.15,
        'vol_ratio_5_20': 1.0,
        'vol_ratio_10_30': 1.0,
        # ✅ UPDATED FOR NEW NAMES:
        'return_10d': 0.01,  # Was return_30d
        'return_20d': 0.01,  # Was return_60d
        'price_to_ma_20': 1.0,  # Was price_to_ma_30
        'price_to_ma_50': 1.0,  # Was price_to_ma_60
        'price_to_ma_200': 1.0,  # New
        'vix_lag1': 16.5,
        'vix_lag7': 16.5,
        'vix_change_5d': 0.0,
        'vix_high': False,
        'dxy_lag1': 103.5,
        'dxy_return_5d': 0.0,  # Was dxy_return_30d
        'dxy_vol_20d': 0.0,   # Was dxy_vol_60d
        'treasury_10y_dec': 0.0425,
        'inflation_rate_dec': 0.027,
        'real_yield': 2.0,
        'real_yield_change': 0.0,
        'sp500_vol_20d': 0.15,  # Was sp500_vol_60d
        'sp500_return_5d': 0.01,  # Was sp500_return_30d
        'large_move_count_5d': 0,  # New
        'max_drawdown_20d': 0.0,   # New
    }
        return defaults.get(feature, 0.0)
    
    def _get_season(self, month: int) -> int:
        """Get season from month"""
        if month in [12, 1, 2]:
            return 0
        elif month in [3, 4, 5]:
            return 1
        elif month in [6, 7, 8]:
            return 2
        else:
            return 3
    
    def _calculate_derived_features(self, features: Dict) -> Dict:
        """Calculate derived features"""
        result = features.copy()
        
        # Gold-silver ratio
        if result.get('gold_price_usd') and result.get('silver_price_usd'):
            if result['silver_price_usd'] > 0:
                result['gold_silver_ratio'] = float(result['gold_price_usd'] / result['silver_price_usd'])
        
        # Real yield
        if result.get('treasury_10y') is not None and result.get('inflation_rate') is not None:
            result['real_yield_10y'] = float(result['treasury_10y'] - (result['inflation_rate'] / 100))
            result['real_yield'] = result['real_yield_10y']
        
        # Yield curve
        if result.get('treasury_10y') is not None and result.get('treasury_2y') is not None:
            result['yield_curve'] = float(result['treasury_10y'] - result['treasury_2y'])
            result['yield_curve_inverted'] = bool(result['yield_curve'] < 0)
        
        # Boolean features
        vix = result.get('vix', 16.5)
        result['vix_spike'] = bool(vix > 20)
        result['vix_surge'] = bool(vix > 30)
        result['vix_high'] = bool(vix > 25)
        
        inflation = result.get('inflation_rate', 2.7)
        result['high_inflation'] = bool(inflation > 3)
        
        gdp = result.get('gdp_growth', 3.2)
        result['recession'] = bool(gdp < 0)
        
        # Decimal versions
        if result.get('treasury_10y') is not None:
            result['treasury_10y_dec'] = float(result['treasury_10y'])
        
        if result.get('inflation_rate') is not None:
            result['inflation_rate_dec'] = float(result['inflation_rate'] / 100)
        
        return result
    
    def save_to_supabase(self, date: str, features: Dict) -> bool:
        """Save features to Supabase - WITH CORRECT COLUMN NAMES"""
        try:
            # Prepare data for Supabase - WITH CORRECT NAMES AFTER RENAMING
            supabase_data = {
                'date': date,
                'gold_price_usd': float(features.get('gold_price_usd', 2100.0)),
                'silver_price_usd': float(features.get('silver_price_usd', 26.25)),
                'vix': float(features.get('vix', 16.5)),
                'dxy': float(features.get('dxy', 103.5)),
                'sp500': float(features.get('sp500', 4950.0)),
                'nasdaq': float(features.get('nasdaq', 17500.0)),
                'treasury_10y': float(features.get('treasury_10y', 0.0425)),
                'treasury_2y': float(features.get('treasury_2y', 0.0455)),
                'crude_oil': float(features.get('crude_oil', 75.0)),
                'copper': float(features.get('copper', 4.2)),
                'gdp_growth': float(features.get('gdp_growth', 3.2)),
                'inflation_rate': float(features.get('inflation_rate', 2.7)),
                'fed_funds_rate': float(features.get('fed_funds_rate', 0.0533)),
                'year': int(features.get('year', datetime.now().year)),
                'month': int(features.get('month', datetime.now().month)),
                'quarter': int(features.get('quarter', (datetime.now().month - 1) // 3 + 1)),
                'day_of_week': int(features.get('day_of_week', datetime.now().weekday())),
                'day_of_year': int(features.get('day_of_year', datetime.now().timetuple().tm_yday)),
                'month_sin': float(features.get('month_sin', 0.0)),
                'month_cos': float(features.get('month_cos', 0.0)),
                'season': int(features.get('season', 0)),
                'gold_silver_ratio': float(features.get('gold_silver_ratio', 80.0)),
                'vix_spike': bool(features.get('vix_spike', False)),
                'vix_surge': bool(features.get('vix_surge', False)),
                'real_yield_10y': float(features.get('real_yield_10y', 2.0)),
                'yield_curve': float(features.get('yield_curve', 0.0)),
                'yield_curve_inverted': bool(features.get('yield_curve_inverted', False)),
                'high_inflation': bool(features.get('high_inflation', False)),
                'recession': bool(features.get('recession', False)),
                'fed_funds_change': float(features.get('fed_funds_change', 0.0)),
                'rate_hike': bool(features.get('rate_hike', False)),
                'rate_cut': bool(features.get('rate_cut', False)),
                'hist_vol_5d': float(features.get('hist_vol_5d', 0.15)),
                'hist_vol_10d': float(features.get('hist_vol_10d', 0.15)),
                'hist_vol_20d': float(features.get('hist_vol_20d', 0.15)),
                'hist_vol_30d': float(features.get('hist_vol_30d', 0.15)),
                'hist_vol_60d': float(features.get('hist_vol_60d', 0.15)),
                'vol_ratio_5_20': float(features.get('vol_ratio_5_20', 1.0)),
                'vol_ratio_10_30': float(features.get('vol_ratio_10_30', 1.0)),
                # ✅ CORRECT NAMES AFTER RENAMING:
                'return_10d': float(features.get('return_10d', 0.01)),
                'return_20d': float(features.get('return_20d', 0.01)),
                'price_to_ma_20': float(features.get('price_to_ma_20', 1.0)),
                'price_to_ma_50': float(features.get('price_to_ma_50', 1.0)),
                'price_to_ma_200': float(features.get('price_to_ma_200', 1.0)),
                'vix_lag1': float(features.get('vix_lag1', 16.5)),
                'vix_lag7': float(features.get('vix_lag7', 16.5)),
                'vix_change_5d': float(features.get('vix_change_5d', 0.0)),
                'vix_high': bool(features.get('vix_high', False)),
                'dxy_lag1': float(features.get('dxy_lag1', 103.5)),
                'dxy_return_5d': float(features.get('dxy_return_5d', 0.0)),
                'dxy_vol_20d': float(features.get('dxy_vol_20d', 0.0)),
                'treasury_10y_dec': float(features.get('treasury_10y_dec', 0.0425)),
                'inflation_rate_dec': float(features.get('inflation_rate_dec', 0.027)),
                'real_yield': float(features.get('real_yield', 2.0)),
                'real_yield_change': float(features.get('real_yield_change', 0.0)),
                'sp500_vol_20d': float(features.get('sp500_vol_20d', 0.15)),
                'sp500_return_5d': float(features.get('sp500_return_5d', 0.01)),
                'large_move_count_5d': int(features.get('large_move_count_5d', 0)),
                'max_drawdown_20d': float(features.get('max_drawdown_20d', 0.0)),
            }
            
            # Insert or update in Supabase
            self.supabase.table("gold_silver_dataset").upsert(supabase_data).execute()
            logger.info(f"✅ Saved features to Supabase for {date}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving to Supabase: {e}")
            return False
    def get_features_for_prediction(self) -> Dict:
        """MAIN METHOD: Get features for model prediction"""
        try:
            today = datetime.now().date().isoformat()
            
            # 1. Check if data exists in Supabase
            if self.check_supabase_data(today):
                logger.info(f"✅ Loading features from Supabase for {today}")
                
                # Get data from Supabase
                response = self.supabase.table("gold_silver_dataset") \
                    .select("*") \
                    .eq("date", today) \
                    .execute()
                
                if response.data:
                    # Get base features from database
                    base_features = response.data[0]
                    
                    # Convert to proper types
                    features = {}
                    for key, value in base_features.items():
                        if value is None:
                            features[key] = None
                        elif isinstance(value, bool):
                            features[key] = bool(value)
                        elif isinstance(value, (int, float)):
                            if isinstance(value, float):
                                features[key] = float(value)
                            else:
                                features[key] = int(value)
                        else:
                            features[key] = value
                    
                    # Prepare features for model
                    model_features = self.prepare_features_for_model(features)
                    
                    logger.info(f"✅ Loaded and prepared {len(model_features)} features for model")
                    return model_features
            
            # 2. If not in Supabase, fetch new data
            logger.info(f"🔄 No data in Supabase for {today}, fetching new...")
            
            # Fetch gold price
            logger.info("1. Fetching gold price...")
            gold_data = self.fetch_gold_price()
            gold_price = gold_data['price'] if gold_data else 2100.0
            
            # Fetch silver price
            logger.info("2. Fetching silver price...")
            silver_data = self.fetch_silver_price()
            silver_price = silver_data['price'] if silver_data else gold_price / 80
            
            # Fetch market indicators
            logger.info("3. Fetching market indicators...")
            market_data = self.fetch_market_indicators()
            
            # Fetch economic indicators
            logger.info("4. Fetching economic indicators...")
            econ_data = self.fetch_economic_indicators()
            
            # Create basic features
            features = self._create_basic_features(gold_price, silver_price, market_data, econ_data)
            
            # Get historical data for time-series calculations
            historical_df = self.get_historical_data(days=200)
            
            # Add current prices for time-series calculations
            current_prices = {
                'gold_price_usd': gold_price,
                'silver_price_usd': silver_price
            }
            
            # Calculate time-series features
            if not historical_df.empty:
                ts_features = self.calculate_time_series_features(historical_df, current_prices)
                features.update(ts_features)
            else:
                features.update(self._get_default_time_series_features())
            
            # Prepare features for model
            model_features = self.prepare_features_for_model(features)
            
            # Save to Supabase
            logger.info("6. Saving to Supabase...")
            self.save_to_supabase(today, features)
            
            logger.info(f"✅ Calculated and saved {len(model_features)} features for model")

                # DEBUG: Show what we're returning
            print(f"\n🔍 FEATURE CALCULATOR DEBUG:")
            print(f"Returning {len(model_features)} features")
            print("First 20 features being returned:")
            for i, (key, value) in enumerate(list(model_features.items())[:20]):
                print(f"  {i+1:2d}. {key}: {value} (type: {type(value).__name__})")
        
            return model_features
        

            
        except Exception as e:
            logger.error(f"❌ Error in get_features_for_prediction: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Return default features in case of error
            return self._create_default_features()
    
    def _create_basic_features(self, gold_price: float, silver_price: float, 
                              market_data: Dict, econ_data: Dict) -> Dict:
        """Create basic feature dictionary"""
        today = datetime.now()
        features = {}
        
        # Basic prices
        features['gold_price_usd'] = float(gold_price)
        features['silver_price_usd'] = float(silver_price)
        
        # Market indicators
        features['vix'] = float(market_data.get('vix', 16.5))
        features['dxy'] = float(market_data.get('dxy', 103.5))
        features['sp500'] = float(market_data.get('sp500', 4950.0))
        features['nasdaq'] = float(market_data.get('nasdaq', 17500.0))
        features['treasury_10y'] = float(market_data.get('treasury_10y', 0.0425))
        features['treasury_2y'] = float(market_data.get('treasury_2y', 0.0455))
        features['crude_oil'] = float(market_data.get('crude_oil', 75.0))
        features['copper'] = float(market_data.get('copper', 4.2))
        
        # Economic indicators
        features['gdp_growth'] = float(econ_data.get('gdp_growth', 3.2))
        features['inflation_rate'] = float(econ_data.get('inflation_rate', 2.7))
        features['fed_funds_rate'] = float(econ_data.get('fed_funds_rate', 0.0533))
        
        # Time-based features
        features['year'] = int(today.year)
        features['month'] = int(today.month)
        features['quarter'] = int((today.month - 1) // 3 + 1)
        features['day_of_week'] = int(today.weekday())
        features['day_of_year'] = int(today.timetuple().tm_yday)
        
        features['date'] = today.date().isoformat()
        
        return features
    
    def _create_default_features(self) -> Dict:
        """Create default features in case of error"""
        today = datetime.now()
        features = {}
        
        # Use defaults for all features
        for feature in self.MODEL_FEATURES:
            features[feature] = self._get_feature_default(feature)
        
        # Add gold price
        features['gold_price_usd'] = 2100.0
        
        # Add date
        features['date'] = today.date().isoformat()
        
        # Calculate derived features
        features = self._calculate_derived_features(features)
        
        return features

# Singleton instance
_feature_calculator = None

def get_feature_calculator(model_loader=None):
    """Get or create feature calculator instance"""
    global _feature_calculator
    if _feature_calculator is None:
        _feature_calculator = GoldFeatureCalculator(model_loader)
    return _feature_calculator
