"""
USD/DZD Data Fetcher Service
Handles data fetching from Supabase and Yahoo Finance
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from supabase import create_client, Client
import yfinance as yf
import os
import logging

logger = logging.getLogger(__name__)

class USDDataFetcher:
    """Service for fetching USD/DZD data"""
    
    def __init__(self, supabase_client: Client):
        self.supabase = supabase_client
    
    def fetch_from_yfinance(self, symbol: str, date: datetime) -> Optional[float]:
        """Fetch data from Yahoo Finance"""
        try:
            # Format date for yfinance
            start_date = (date - timedelta(days=10)).strftime('%Y-%m-%d')
            end_date = (date + timedelta(days=1)).strftime('%Y-%m-%d')
            
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date)
            
            if not hist.empty:
                # Get the closest date to requested date
                hist['Date'] = hist.index
                hist['Date'] = pd.to_datetime(hist['Date']).dt.date
                target_date = date.date()
                
                # Find closest date (could be same day or previous trading day)
                hist = hist.sort_index(ascending=False)
                for idx, row in hist.iterrows():
                    if row['Date'] <= target_date:
                        return float(row['Close'])
            
            return None
        except Exception as e:
            logger.error(f"Error fetching {symbol} from yfinance: {e}")
            return None
    
    async def fetch_or_calculate_features(self, target_date: datetime) -> Dict:
        """
        Fetch or calculate USD/DZD features
        Strategy:
        1. Check if features already in usd_dzd_dataset
        2. If not, fetch from euro_dzd_dataset and calculate
        3. If still missing, fetch from external sources (yfinance)
        """
        features = {}
        target_date_str = target_date.strftime('%Y-%m-%d')
        
        # Step 1: Check if already in usd_dzd_dataset
        try:
            response = self.supabase.table("usd_dzd_dataset")\
                .select("*")\
                .eq("date", target_date_str)\
                .execute()
            
            if response.data:
                logger.info(f"Found features for {target_date_str} in usd_dzd_dataset")
                return response.data[0]
        except Exception as e:
            logger.warning(f"Error checking usd_dzd_dataset: {e}")
        
        # Step 2: Fetch from euro_dzd_dataset and calculate USD features
        try:
            response = self.supabase.table("euro_dzd_dataset")\
                .select("*")\
                .eq("date", target_date_str)\
                .execute()
            
            if response.data:
                euro_data = response.data[0]
                logger.info(f"Found EUR data for {target_date_str} in euro_dzd_dataset")
                
                # Calculate USD/DZD parallel
                eur_dzd_parallel = euro_data.get('eur_dzd_parallel')
                eur_usd = euro_data.get('eur_usd')
                
                if eur_dzd_parallel and eur_usd:
                    features['usd_dzd_parallel'] = eur_dzd_parallel / eur_usd
                
                # Copy other features
                feature_mapping = {
                    'usd_dzd_official': 'eur_dzd_official',
                    'brent_oil': 'brent_oil',
                    'dxy': 'dxy',
                }
                
                for usd_feature, eur_feature in feature_mapping.items():
                    if eur_feature in euro_data:
                        features[usd_feature] = euro_data[eur_feature]
                
                features['date'] = target_date_str
                features['eur_usd'] = eur_usd
                
                # Store in usd_dzd_dataset for future use
                try:
                    self.supabase.table("usd_dzd_dataset").insert(features).execute()
                    logger.info(f"Stored calculated features for {target_date_str}")
                except Exception as e:
                    logger.warning(f"Could not store features: {e}")
                
                return features
        except Exception as e:
            logger.warning(f"Error fetching from euro_dzd_dataset: {e}")
        
        # Step 3: Fetch from external sources
        logger.info(f"Fetching data from external sources for {target_date_str}")
        
        # Fetch USD/DZD official rate
        features['usd_dzd_official'] = self.fetch_from_yfinance('DZD=X', target_date)
        
        # Fetch EUR/USD
        features['eur_usd'] = self.fetch_from_yfinance('EURUSD=X', target_date)
        
        # Fetch Brent Oil
        features['brent_oil'] = self.fetch_from_yfinance('BZ=F', target_date)
        
        # Fetch DXY (US Dollar Index)
        features['dxy'] = self.fetch_from_yfinance('DX-Y.NYB', target_date)
        
        # Try to get EUR/DZD parallel from supabase for calculation
        try:
            eur_response = self.supabase.table("euro_dzd_dataset")\
                .select("eur_dzd_parallel")\
                .eq("date", target_date_str)\
                .execute()
            
            if eur_response.data and eur_response.data[0].get('eur_dzd_parallel'):
                eur_dzd_parallel = eur_response.data[0]['eur_dzd_parallel']
                if features['eur_usd']:
                    features['usd_dzd_parallel'] = eur_dzd_parallel / features['eur_usd']
        except Exception as e:
            logger.warning(f"Error fetching EUR/DZD parallel: {e}")
        
        features['date'] = target_date_str
        
        # Store what we have
        try:
            self.supabase.table("usd_dzd_dataset").insert(features).execute()
        except Exception as e:
            logger.warning(f"Could not store features: {e}")
        
        return features
    
    def calculate_lag_features(self, features_df: pd.DataFrame, target_date: datetime) -> pd.DataFrame:
        """Calculate lag features needed by the USD/DZD model"""
        # Create a copy for calculations
        df = features_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # Calculate lag features for target variable
        target_col = 'usd_dzd_parallel'
        if target_col in df.columns:
            for lag in [1, 2, 3, 5, 7, 10, 14, 30]:
                df[f'lag_{lag}'] = df[target_col].shift(lag)
        
        # Calculate lag features for external variables
        external_cols = ['eur_usd', 'brent_oil', 'dxy']
        for col in external_cols:
            if col in df.columns:
                for lag in [1, 2, 3, 5]:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        # Calculate moving averages
        if target_col in df.columns:
            for window in [5, 10, 20, 50]:
                df[f'ma_{window}'] = df[target_col].rolling(window=window).mean()
                df[f'std_{window}'] = df[target_col].rolling(window=window).std()
        
        # Add date features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        
        return df
    
    def get_usd_history(
        self, 
        start_date: datetime, 
        end_date: datetime,
        rate_type: str = "both"  # "parallel", "official", or "both"
    ) -> List[Dict]:
        """
        Get historical USD/DZD data
        Returns list of {date, usd_dzd_parallel, usd_dzd_official}
        """
        try:
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            
            # Select columns based on rate_type
            if rate_type == "parallel":
                select_cols = "date, usd_dzd_parallel"
            elif rate_type == "official":
                select_cols = "date, usd_dzd_official"
            else:  # "both"
                select_cols = "date, usd_dzd_parallel, usd_dzd_official"
            
            response = self.supabase.table("usd_dzd_dataset") \
                .select(select_cols) \
                .gte("date", start_date_str) \
                .lte("date", end_date_str) \
                .order("date", desc=False) \
                .execute()
            
            history = []
            if response.data:
                for record in response.data:
                    history.append({
                        "date": record["date"],
                        "usd_dzd_parallel": float(record["usd_dzd_parallel"]) if record.get("usd_dzd_parallel") else None,
                        "usd_dzd_official": float(record["usd_dzd_official"]) if record.get("usd_dzd_official") else None
                    })
            
            return history
            
        except Exception as e:
            logger.error(f"Error fetching USD history: {e}")
            return []
    
    def get_history_statistics(self, history: List[Dict]) -> Dict:
        """Calculate statistics from historical data"""
        stats = {}
        
        # Parallel rate statistics
        parallel_prices = [h["usd_dzd_parallel"] for h in history if h["usd_dzd_parallel"] is not None]
        if parallel_prices:
            stats["parallel"] = {
                "current": parallel_prices[-1] if parallel_prices else None,
                "min": min(parallel_prices),
                "max": max(parallel_prices),
                "avg": sum(parallel_prices) / len(parallel_prices),
                "change": parallel_prices[-1] - parallel_prices[0] if len(parallel_prices) > 1 else 0,
                "percent_change": ((parallel_prices[-1] - parallel_prices[0]) / parallel_prices[0] * 100) if parallel_prices[0] > 0 else 0
            }
        
        # Official rate statistics
        official_prices = [h["usd_dzd_official"] for h in history if h["usd_dzd_official"] is not None]
        if official_prices:
            stats["official"] = {
                "current": official_prices[-1] if official_prices else None,
                "min": min(official_prices),
                "max": max(official_prices),
                "avg": sum(official_prices) / len(official_prices),
                "change": official_prices[-1] - official_prices[0] if len(official_prices) > 1 else 0,
                "percent_change": ((official_prices[-1] - official_prices[0]) / official_prices[0] * 100) if official_prices[0] > 0 else 0
            }
        
        # Spread statistics
        spreads = []
        for h in history:
            if h["usd_dzd_parallel"] is not None and h["usd_dzd_official"] is not None:
                spread = h["usd_dzd_parallel"] - h["usd_dzd_official"]
                spreads.append(spread)
        
        if spreads:
            stats["spread"] = {
                "current": spreads[-1] if spreads else 0,
                "min": min(spreads),
                "max": max(spreads),
                "avg": sum(spreads) / len(spreads),
                "avg_percent": (sum(spreads) / len(spreads)) / stats["official"]["avg"] * 100 if stats.get("official") and stats["official"]["avg"] > 0 else 0
            }
        
        return stats