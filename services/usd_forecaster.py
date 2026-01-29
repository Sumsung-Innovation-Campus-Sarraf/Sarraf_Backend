import numpy as np
from datetime import datetime
from pathlib import Path
import joblib
import pickle
import json
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class USDForecaster:
    """USD/DZD forecaster using USDDataFetcher"""
    
    def __init__(self, data_fetcher=None):
        self.data_fetcher = data_fetcher
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.feature_cols = [
            'eur_usd', 'brent_oil', 'dxy', 'lag1', 'lag7', 'lag30',
            'usd_dzd_official'
        ]
        self.metadata = {
            "model_type": "Ridge",
            "feature_count": 7,
            "performance": {"rmse": 0.614, "mae": 0.496, "r2": 0.9996}
        }
        self.target_col = 'usd_dzd_parallel'

    def load_models(self, model_path, scaler_X_path, scaler_y_path, metadata_path=None):
        """Load model and scalers"""
        try:
            # Load model
            if Path(model_path).exists():
                self.model = joblib.load(model_path)
                logger.info(f"✅ Model loaded from {model_path}")
            else:
                raise FileNotFoundError(f"Model file not found: {model_path}")

            # Load scaler X
            if Path(scaler_X_path).exists():
                try:
                    with open(scaler_X_path, 'rb') as f:
                        self.scaler_X = pickle.load(f)
                except Exception:
                    self.scaler_X = joblib.load(scaler_X_path)
                logger.info(f"✅ Scaler X loaded from {scaler_X_path}")
            else:
                raise FileNotFoundError(f"Scaler X file not found: {scaler_X_path}")

            # Load scaler Y
            if Path(scaler_y_path).exists():
                try:
                    with open(scaler_y_path, 'rb') as f:
                        self.scaler_y = pickle.load(f)
                except Exception:
                    self.scaler_y = joblib.load(scaler_y_path)
                logger.info(f"✅ Scaler Y loaded from {scaler_y_path}")
            else:
                raise FileNotFoundError(f"Scaler Y file not found: {scaler_y_path}")

            # Load metadata if provided
            if metadata_path and Path(metadata_path).exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"✅ Metadata loaded from {metadata_path}")

            return self.is_loaded()
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            raise

    def is_loaded(self):
        """Check if all components are loaded"""
        return all([self.model is not None, self.scaler_X is not None, self.scaler_y is not None])

    def get_model_info(self):
        """Get model information"""
        if not self.is_loaded():
            return {"error": "Model not loaded"}
        
        return {
            "model_type": self.metadata.get("model_type", "Ridge"),
            "feature_count": len(self.feature_cols),
            "features": self.feature_cols,
            "performance": self.metadata.get("performance", {}),
            "target": self.target_col
        }

    async def forecast(self, target_date_str: str) -> Dict:
        """
        Fetch features via fetcher and make a prediction
        
        Args:
            target_date_str: Date string in format 'YYYY-MM-DD'
            
        Returns:
            Dict with prediction results
        """
        if not self.is_loaded():
            raise ValueError("Model not loaded. Call load_models() first.")

        if not self.data_fetcher:
            raise ValueError("Data fetcher not initialized")

        # Convert string to datetime
        target_date = datetime.strptime(target_date_str, '%Y-%m-%d')
        
        logger.info(f"📊 Forecasting for {target_date_str}")

        # Fetch features (fetcher handles lag calculation + external fallback)
        features = await self.data_fetcher.fetch_or_calculate_features(target_date)
        
        logger.info(f"📋 Features fetched: {list(features.keys())}")

        # Prepare feature array in correct order
        X = []
        missing_features = []
        
        for feature in self.feature_cols:
            value = features.get(feature)
            if value is None or (isinstance(value, float) and np.isnan(value)):
                missing_features.append(feature)
                # Use defaults for missing features
                defaults = {
                    'eur_usd': 1.08,
                    'brent_oil': 80.0,
                    'dxy': 100.0,
                    'lag1': 150.0,
                    'lag7': 150.0,
                    'lag30': 150.0,
                    'usd_dzd_official': 150.0
                }
                value = defaults.get(feature, 0.0)
            X.append(float(value))
        
        if missing_features:
            logger.warning(f"⚠️ Missing features (using defaults): {missing_features}")
        
        X = np.array([X])
        logger.info(f"🔢 Feature vector shape: {X.shape}, values: {X[0]}")

        # Scale features
        X_scaled = self.scaler_X.transform(X)
        logger.info(f"📏 Scaled features: {X_scaled[0]}")

        # Predict and inverse scale
        y_pred_scaled = self.model.predict(X_scaled)[0]
        y_pred = self.scaler_y.inverse_transform([[y_pred_scaled]])[0][0]
        
        logger.info(f"🎯 Prediction: {y_pred:.4f} (scaled: {y_pred_scaled:.4f})")

        # Build result
        current_rate = features.get('usd_dzd_parallel')
        rmse = self.metadata.get('performance', {}).get('rmse', 0.614)
        
        ci = {
            "lower": float(y_pred - 1.96 * rmse),
            "upper": float(y_pred + 1.96 * rmse),
            "confidence_level": 0.95
        }

        result = {
            "predicted_rate": float(y_pred),
            "confidence_interval": ci,
            "features_used": self.feature_cols,
            "rmse": float(rmse)
        }

        if current_rate:
            result["current_rate"] = float(current_rate)
            result["expected_change"] = float(y_pred - current_rate)
            result["percent_change"] = float((y_pred - current_rate) / current_rate * 100)

        return result