"""
USD/DZD Forecaster Service
Handles model loading and prediction
"""

import joblib
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from pathlib import Path
import os
import logging
from .usd_data_fetcher import USDDataFetcher

logger = logging.getLogger(__name__)

class USDForecaster:
    """Service for USD/DZD forecasting"""
    
    def __init__(self, data_fetcher: USDDataFetcher):
        self.data_fetcher = data_fetcher
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.metadata = None
    
    def load_models(self) -> bool:
        """Load USD/DZD model components"""
        try:
            # Get paths from environment variables
            USD_MODEL_PATH = os.getenv("USD_MODEL_PATH", "./models/usd_dzd_model.pkl")
            USD_SCALER_PATH = os.getenv("USD_SCALER_PATH", "./models/usd_dzd_scaler.pkl")
            USD_FEATURES_PATH = os.getenv("USD_FEATURES_PATH", "./models/feature_columns.pkl")
            
            logger.info(f"Loading USD/DZD models from:")
            logger.info(f"  Model: {USD_MODEL_PATH}")
            logger.info(f"  Scaler: {USD_SCALER_PATH}")
            logger.info(f"  Features: {USD_FEATURES_PATH}")
            
            # Load model
            if os.path.exists(USD_MODEL_PATH):
                self.model = joblib.load(USD_MODEL_PATH)
                logger.info(f"✅ USD/DZD model loaded from {USD_MODEL_PATH}")
            else:
                logger.error(f"❌ USD/DZD model not found at {USD_MODEL_PATH}")
                return False
            
            # Load scaler
            if os.path.exists(USD_SCALER_PATH):
                self.scaler = joblib.load(USD_SCALER_PATH)
                logger.info(f"✅ USD/DZD scaler loaded from {USD_SCALER_PATH}")
            else:
                logger.error(f"❌ USD/DZD scaler not found at {USD_SCALER_PATH}")
                return False
            
            # Load feature columns
            if os.path.exists(USD_FEATURES_PATH):
                with open(USD_FEATURES_PATH, 'rb') as f:
                    self.feature_cols = pickle.load(f)
                logger.info(f"✅ USD/DZD feature columns loaded: {len(self.feature_cols)} features")
            else:
                logger.error(f"❌ USD/DZD feature columns not found at {USD_FEATURES_PATH}")
                return False
            
            # Load metadata if available
            metadata_path = USD_FEATURES_PATH.replace('feature_columns.pkl', 'model_metadata.json')
            if os.path.exists(metadata_path):
                import json
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"✅ USD/DZD metadata loaded")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading USD/DZD models: {e}")
            return False
    
    def is_loaded(self) -> bool:
        """Check if model components are loaded"""
        return all([
            self.model is not None,
            self.scaler is not None,
            self.feature_cols is not None
        ])
    
    async def prepare_model_input(self, target_date) -> Dict:
        """Prepare complete model input for USD/DZD prediction"""
        # Step 1: Fetch historical data (last 60 days)
        required_days = 60
        start_date = target_date - pd.Timedelta(days=required_days)
        
        historical_features = []
        
        # Fetch or calculate features for each historical date
        current_date = start_date
        while current_date <= target_date:
            features = await self.data_fetcher.fetch_or_calculate_features(current_date)
            historical_features.append(features)
            current_date += pd.Timedelta(days=1)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(historical_features)
        
        # Step 2: Calculate lag features
        features_df = self.data_fetcher.calculate_lag_features(features_df, target_date)
        
        # Step 3: Get the row for target date
        target_date_str = target_date.strftime('%Y-%m-%d')
        target_row = features_df[features_df['date'] == target_date_str]
        
        if target_row.empty:
            raise ValueError(f"No data found for {target_date_str}")
        
        # Step 4: Prepare features in exact order expected by model
        if not self.feature_cols:
            raise ValueError("USD/DZD model feature columns not loaded")
        
        model_input = {}
        missing_features = []
        
        for feature in self.feature_cols:
            if feature in target_row.columns:
                value = target_row[feature].iloc[0]
                if pd.isna(value):
                    missing_features.append(feature)
                else:
                    model_input[feature] = value
            else:
                missing_features.append(feature)
        
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            # Try to fill with reasonable defaults
            for feature in missing_features:
                if feature.startswith('lag_'):
                    # For lag features, use current value as approximation
                    model_input[feature] = target_row['usd_dzd_parallel'].iloc[0] if 'usd_dzd_parallel' in target_row.columns else 240.0
                elif feature.startswith('ma_') or feature.startswith('std_'):
                    # For moving stats, use current value
                    model_input[feature] = target_row['usd_dzd_parallel'].iloc[0] if 'usd_dzd_parallel' in target_row.columns else 240.0
                elif 'eur_usd' in feature:
                    model_input[feature] = 1.08  # Default EUR/USD
                elif 'brent_oil' in feature:
                    model_input[feature] = 80.0  # Default Brent
                elif 'dxy' in feature:
                    model_input[feature] = 100.0  # Default DXY
                elif feature in ['day_of_week', 'month', 'quarter']:
                    # Use actual date features
                    if feature == 'day_of_week':
                        model_input[feature] = target_date.weekday()
                    elif feature == 'month':
                        model_input[feature] = target_date.month
                    elif feature == 'quarter':
                        model_input[feature] = (target_date.month - 1) // 3 + 1
                else:
                    model_input[feature] = 0.0
        
        return model_input
    
    async def forecast(self, target_date, use_cached: bool = True) -> Dict:
        """Make a USD/DZD forecast"""
        if not self.is_loaded():
            raise ValueError("USD/DZD model components not loaded")
        
        # Prepare model input
        model_input = await self.prepare_model_input(target_date)
        
        # Create feature array in correct order
        feature_values = []
        for feature in self.feature_cols:
            if feature in model_input:
                feature_values.append(model_input[feature])
            else:
                feature_values.append(0.0)
        
        # Convert to numpy array and reshape
        X = np.array(feature_values).reshape(1, -1)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        prediction = self.model.predict(X_scaled)[0]
        
        # Get current rate
        current_rate = model_input.get('usd_dzd_parallel', 240.0)
        expected_change = prediction - current_rate
        
        # Calculate confidence interval
        rmse = self.metadata.get('performance', {}).get('rmse', 0.6140) if self.metadata else 0.6140
        confidence_interval = {
            "lower": float(prediction - 1.96 * rmse),
            "upper": float(prediction + 1.96 * rmse),
            "confidence_level": 0.95
        }
        
        return {
            "forecast": float(prediction),
            "current_rate": float(current_rate),
            "expected_change": float(expected_change),
            "confidence_interval": confidence_interval,
            "model_input": model_input,
            "data_source": "cached" if use_cached else "calculated"
        }
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        if not self.is_loaded():
            return {}
        
        info = {
            "model_type": type(self.model).__name__,
            "feature_count": len(self.feature_cols),
            "feature_columns": self.feature_cols,
            "is_loaded": True
        }
        
        # Add metadata if available
        if self.metadata:
            info.update({
                "training_date": self.metadata.get('training_date'),
                "performance": self.metadata.get('performance', {}),
                "data_info": self.metadata.get('data_info', {})
            })
        
        # Add coefficients if available
        if hasattr(self.model, 'coef_'):
            info["has_coefficients"] = True
            if len(self.model.coef_) == len(self.feature_cols):
                coefs = list(zip(self.feature_cols, self.model.coef_))
                coefs.sort(key=lambda x: abs(x[1]), reverse=True)
                info["top_coefficients"] = [
                    {"feature": feature, "coefficient": float(coef)} 
                    for feature, coef in coefs[:10]
                ]
        
        return info