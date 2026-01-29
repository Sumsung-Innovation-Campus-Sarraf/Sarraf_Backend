import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from supabase import Client
import yfinance as yf
import logging

logger = logging.getLogger(__name__)

class USDDataFetcher:
    """Fetch USD/DZD features for prediction"""

    def __init__(self, supabase_client: Client):
        self.supabase = supabase_client

    def fetch_from_yfinance(self, symbol: str, date: datetime) -> Optional[float]:
        """Fetch data from Yahoo Finance"""
        try:
            start = (date - timedelta(days=10)).strftime('%Y-%m-%d')
            end = (date + timedelta(days=1)).strftime('%Y-%m-%d')
            
            logger.info(f"🔍 Fetching {symbol} from Yahoo Finance for {date.date()}")
            hist = yf.download(symbol, start=start, end=end, progress=False)
            
            if hist.empty:
                logger.warning(f"⚠️ No data for {symbol}")
                return None
            
            # Get the closest date to target
            hist['Date'] = pd.to_datetime(hist.index).date
            target = date.date()
            hist = hist.sort_index(ascending=False)
            
            for idx, row in hist.iterrows():
                if row['Date'] <= target:
                    value = float(row['Close'])
                    logger.info(f"✅ {symbol}: {value:.4f}")
                    return value
            
            return None
        except Exception as e:
            logger.warning(f"❌ Error fetching {symbol} from yfinance: {e}")
            return None

    async def fetch_or_calculate_features(self, target_date: datetime) -> Dict:
        """Fetch or calculate features for the target date"""
        target_str = target_date.strftime('%Y-%m-%d')
        features = {}

        logger.info(f"📅 Fetching features for {target_str}")

        # Step 1: Check if data already exists in Supabase
        try:
            resp = self.supabase.table("usd_dzd_dataset").select("*").eq("date", target_str).execute()
            if resp.data and len(resp.data) > 0:
                logger.info(f"✅ Features for {target_str} found in database")
                data = resp.data[0]
                
                # Map database columns to model feature names
                features = {
                    'usd_dzd_parallel': data.get('usd_dzd_parallel'),
                    'usd_dzd_official': data.get('usd_dzd_official'),
                    'eur_usd': data.get('eur_usd'),
                    'brent_oil': data.get('brent_oil'),
                    'dxy': data.get('dxy'),
                    'lag1': data.get('usd_dzd_parallel_lag1'),  # Map lag columns
                    'lag7': data.get('usd_dzd_parallel_lag7'),
                    'lag30': data.get('usd_dzd_parallel_lag30'),
                }
                
                # Return if we have all required features
                if all(features.get(k) is not None for k in ['eur_usd', 'brent_oil', 'dxy', 'lag1', 'lag7', 'lag30', 'usd_dzd_official']):
                    logger.info(f"✅ All features available from database")
                    return features
                else:
                    logger.info(f"⚠️ Some features missing, will fetch from external sources")
        except Exception as e:
            logger.warning(f"⚠️ Error checking usd_dzd_dataset: {e}")

        # Step 2: Try EUR dataset (alternative source)
        try:
            resp = self.supabase.table("euro_dzd_dataset").select("*").eq("date", target_str).execute()
            if resp.data and len(resp.data) > 0:
                euro = resp.data[0]
                eur_dzd = euro.get('eur_dzd_parallel')
                eur_usd = euro.get('eur_usd')
                
                if eur_dzd and eur_usd and eur_usd > 0:
                    features['usd_dzd_parallel'] = eur_dzd / eur_usd
                    logger.info(f"📊 Calculated USD/DZD from EUR data: {features['usd_dzd_parallel']:.4f}")
                
                if not features.get('usd_dzd_official'):
                    features['usd_dzd_official'] = euro.get('eur_dzd_official')
                if not features.get('brent_oil'):
                    features['brent_oil'] = euro.get('brent_oil')
                if not features.get('dxy'):
                    features['dxy'] = euro.get('dxy')
                if not features.get('eur_usd'):
                    features['eur_usd'] = eur_usd
        except Exception as e:
            logger.warning(f"⚠️ Error fetching euro_dzd_dataset: {e}")

        # Step 3: Fetch missing data from external sources (Yahoo Finance)
        if 'usd_dzd_official' not in features or features['usd_dzd_official'] is None:
            features['usd_dzd_official'] = self.fetch_from_yfinance('DZD=X', target_date)
        
        if 'eur_usd' not in features or features['eur_usd'] is None:
            features['eur_usd'] = self.fetch_from_yfinance('EURUSD=X', target_date)
        
        if 'brent_oil' not in features or features['brent_oil'] is None:
            features['brent_oil'] = self.fetch_from_yfinance('BZ=F', target_date)
        
        if 'dxy' not in features or features['dxy'] is None:
            features['dxy'] = self.fetch_from_yfinance('DX-Y.NYB', target_date)

        # Step 4: Get historical USD/DZD for lag features
        logger.info("📈 Fetching historical data for lag calculation...")
        hist_df = self.get_historical_data(target_date, days_back=60)
        
        if not hist_df.empty:
            hist_df = hist_df.sort_values('date').reset_index(drop=True)
            
            # Calculate lag features
            if len(hist_df) >= 2:
                features['lag1'] = hist_df['usd_dzd_parallel'].iloc[-2] if len(hist_df) >= 2 else None
                logger.info(f"  lag1: {features.get('lag1')}")
            
            if len(hist_df) >= 8:
                features['lag7'] = hist_df['usd_dzd_parallel'].iloc[-8] if len(hist_df) >= 8 else None
                logger.info(f"  lag7: {features.get('lag7')}")
            
            if len(hist_df) >= 31:
                features['lag30'] = hist_df['usd_dzd_parallel'].iloc[-31] if len(hist_df) >= 31 else None
                logger.info(f"  lag30: {features.get('lag30')}")
        else:
            logger.warning("⚠️ No historical data available for lag calculation")

        features['date'] = target_str

        # Step 5: Store features in Supabase for future use
        try:
            insert_data = {
                'date': target_str,
                'usd_dzd_parallel': features.get('usd_dzd_parallel'),
                'usd_dzd_official': features.get('usd_dzd_official'),
                'eur_usd': features.get('eur_usd'),
                'brent_oil': features.get('brent_oil'),
                'dxy': features.get('dxy'),
                'usd_dzd_parallel_lag1': features.get('lag1'),
                'usd_dzd_parallel_lag7': features.get('lag7'),
                'usd_dzd_parallel_lag30': features.get('lag30'),
            }
            
            # Remove None values
            insert_data = {k: v for k, v in insert_data.items() if v is not None}
            
            if len(insert_data) > 1:  # More than just date
                self.supabase.table("usd_dzd_dataset").upsert(insert_data).execute()
                logger.info(f"💾 Stored {len(insert_data)-1} features in database")
        except Exception as e:
            logger.warning(f"⚠️ Could not store features: {e}")

        logger.info(f"✅ Feature collection complete: {list(features.keys())}")
        return features

    def get_historical_data(self, target_date: datetime, days_back: int = 60) -> pd.DataFrame:
        """Fetch historical USD/DZD parallel for lag calculation"""
        start = (target_date - timedelta(days=days_back)).strftime('%Y-%m-%d')
        end = target_date.strftime('%Y-%m-%d')
        
        try:
            resp = self.supabase.table("usd_dzd_dataset")\
                .select("date, usd_dzd_parallel")\
                .gte("date", start)\
                .lt("date", end)\
                .order("date", desc=False)\
                .execute()
            
            if resp.data and len(resp.data) > 0:
                df = pd.DataFrame(resp.data)
                df['date'] = pd.to_datetime(df['date'])
                logger.info(f"📊 Loaded {len(df)} historical records")
                return df
            else:
                logger.warning(f"⚠️ No historical data found between {start} and {end}")
        except Exception as e:
            logger.warning(f"❌ Error fetching historical USD/DZD: {e}")
        
        return pd.DataFrame()

    def get_usd_history(self, start_date: datetime, end_date: datetime, rate_type: str = "both") -> List[Dict]:
        """Get historical USD/DZD data"""
        try:
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # Select columns based on rate_type
            if rate_type == "parallel":
                select_cols = "date, usd_dzd_parallel"
            elif rate_type == "official":
                select_cols = "date, usd_dzd_official"
            else:
                select_cols = "date, usd_dzd_parallel, usd_dzd_official"
            
            resp = self.supabase.table("usd_dzd_dataset")\
                .select(select_cols)\
                .gte("date", start_str)\
                .lte("date", end_str)\
                .order("date", desc=False)\
                .execute()
            
            return resp.data if resp.data else []
        except Exception as e:
            logger.error(f"❌ Error fetching history: {e}")
            return []

    def get_history_statistics(self, history: List[Dict]) -> Dict:
        """Calculate statistics from history"""
        if not history:
            return {}
        
        try:
            df = pd.DataFrame(history)
            stats = {}
            
            if 'usd_dzd_parallel' in df.columns:
                parallel = df['usd_dzd_parallel'].dropna()
                if len(parallel) > 0:
                    stats['parallel'] = {
                        'current': float(parallel.iloc[-1]),
                        'min': float(parallel.min()),
                        'max': float(parallel.max()),
                        'mean': float(parallel.mean()),
                        'std': float(parallel.std())
                    }
            
            if 'usd_dzd_official' in df.columns:
                official = df['usd_dzd_official'].dropna()
                if len(official) > 0:
                    stats['official'] = {
                        'current': float(official.iloc[-1]),
                        'min': float(official.min()),
                        'max': float(official.max()),
                        'mean': float(official.mean()),
                        'std': float(official.std())
                    }
            
            return stats
        except Exception as e:
            logger.error(f"❌ Error calculating statistics: {e}")
            return {}