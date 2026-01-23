"""
USD/DZD Model Loader Service
"""

import joblib
import pickle
import os
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

class USDModelLoader:
    def __init__(self):
        """Initialize USD/DZD model loader"""
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.metadata = None
        
        # Load models on initialization
        self.load_models()
    
    def load_models(self) -> bool:
        """Load USD/DZD model components"""
        try:
            # Use environment variables
            model_path = os.getenv("USD_MODEL_PATH", "./models/usd_dzd_model.pkl")
            scaler_path = os.getenv("USD_SCALER_PATH", "./models/usd_dzd_scaler.pkl")
            features_path = os.getenv("USD_FEATURES_PATH", "./models/feature_columns.pkl")
            
            print(f"Loading USD/DZD models from:")
            print(f"  Model: {model_path}")
            print(f"  Scaler: {scaler_path}")
            print(f"  Features: {features_path}")
            
            # Load model
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                print(f"✅ USD/DZD model loaded")
            else:
                print(f"❌ USD/DZD model not found at {model_path}")
                return False
            
            # Load scaler
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                print(f"✅ USD/DZD scaler loaded")
            else:
                print(f"❌ USD/DZD scaler not found at {scaler_path}")
                return False
            
            # Load feature columns
            if os.path.exists(features_path):
                with open(features_path, 'rb') as f:
                    self.feature_cols = pickle.load(f)
                print(f"✅ USD/DZD feature columns loaded: {len(self.feature_cols)} features")
            else:
                print(f"❌ USD/DZD feature columns not found at {features_path}")
                return False
            
            # Load metadata if available
            metadata_path = os.getenv("USD_METADATA_PATH", "./models/model_metadata.json")
            if os.path.exists(metadata_path):
                import json
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                print(f"✅ USD/DZD metadata loaded")
            
            # Print model info
            print(f"📊 USD/DZD Model Info:")
            print(f"  - Model type: {type(self.model).__name__}")
            print(f"  - Feature count: {len(self.feature_cols)}")
            print(f"  - Top 10 features: {self.feature_cols[:10]}")
            
            if hasattr(self.model, 'coef_'):
                print(f"  - Model has coefficients")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading USD/DZD models: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def is_loaded(self) -> bool:
        """Check if models are loaded"""
        return all([
            self.model is not None,
            self.scaler is not None,
            self.feature_cols is not None
        ])
    
    def get_required_features(self) -> List[str]:
        """Get list of required features"""
        return self.feature_cols if self.feature_cols else []
    
    def predict(self, features: Dict) -> float:
        """Make prediction using USD/DZD model"""
        if not self.is_loaded():
            raise ValueError("USD/DZD models not loaded")
        
        # Create feature array in correct order
        feature_values = []
        for feature in self.feature_cols:
            if feature in features:
                feature_values.append(features[feature])
            else:
                feature_values.append(0.0)
        
        # Convert to numpy array and reshape
        X = np.array(feature_values).reshape(1, -1)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        prediction = self.model.predict(X_scaled)[0]
        
        return float(prediction)
    
    def get_confidence_interval(self, prediction: float) -> Dict[str, float]:
        """Get confidence interval for prediction"""
        # Use RMSE from training or metadata
        rmse = 0.6140  # Default from your training
        if self.metadata and 'performance' in self.metadata:
            rmse = self.metadata['performance'].get('rmse', 0.6140)
        
        return {
            "lower": float(prediction - 1.96 * rmse),
            "upper": float(prediction + 1.96 * rmse),
            "confidence_level": 0.95,
            "rmse": rmse
        }

# Singleton instance
_usd_model_loader = None

def get_usd_model_loader() -> USDModelLoader:
    """Get USD model loader instance"""
    global _usd_model_loader
    if _usd_model_loader is None:
        _usd_model_loader = USDModelLoader()
    return _usd_model_loader