"""
Model loader service - Improved for both Gold and Silver models
"""

import joblib
import pickle
import os
from pathlib import Path
from typing import Any, Optional, Tuple, Dict
import numpy as np
import pandas as pd
import warnings
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class ModelLoader:
    """Handles loading models from single pickle files"""
    
    def __init__(self, models_dir: str = "."):
        self.models_dir = Path(models_dir)
        self.models = {}  # Format: {'monthly_direction': {'model': ..., 'scaler': ..., 'feature_cols': ...}}
        self.model_info = {}
        
        logger.info(f"📂 Model loader initialized with directory: {self.models_dir.absolute()}")
    
    def get_model_metrics(self, model_name: str = "monthly_direction"):
        """Get model metrics like accuracy if saved with the model"""
        if model_name in self.models:
            model_data = self.models[model_name]
            
            if 'metrics' in model_data:
                return model_data['metrics']
            elif 'accuracy' in model_data:
                return {'accuracy': model_data['accuracy']}
        
        model_obj = self.models.get(model_name, {}).get('model')
        if hasattr(model_obj, 'best_score_'):
            return {'best_score': model_obj.best_score_}
        elif hasattr(model_obj, 'score'):
            return {'has_score_method': True}
        
        return {'error': 'No metrics found in model file'}

    def load_all_models(self):
        """Load all available model files from directory"""
        logger.info("🔍 Scanning for model files...")
        
        model_files = []
        
        # Search patterns for model files
        search_patterns = [
            "*model*.pkl",
            "*_model.pkl",
            "*.pkl"
        ]
        
        for pattern in search_patterns:
            for file in self.models_dir.glob(pattern):
                if file not in model_files and file.is_file():
                    model_files.append(file)
        
        logger.info(f"📦 Found {len(model_files)} potential model files")
        
        if not model_files:
            logger.warning(f"⚠️ No model files found in {self.models_dir.absolute()}")
            logger.warning("   Please ensure your model files are in the correct location")
            return self.models
        
        # Load each model
        loaded_count = 0
        for model_path in model_files:
            try:
                logger.info(f"   Loading {model_path.name}...")
                model_data = self._load_model_file(model_path)
                
                if model_data:
                    # Determine model name from filename
                    model_name = self._extract_model_name(model_path)
                    
                    self.models[model_name] = model_data
                    
                    # Extract model info
                    self.model_info[model_name] = {
                        'path': str(model_path),
                        'loaded': True,
                        'features': model_data.get('feature_cols', []),
                        'has_scaler': model_data.get('scaler') is not None,
                        'has_model': model_data.get('model') is not None,
                        'model_type': type(model_data.get('model')).__name__
                    }
                    
                    feature_count = len(model_data.get('feature_cols', []))
                    logger.info(f"      ✅ Loaded as '{model_name}' ({feature_count} features)")
                    loaded_count += 1
                else:
                    logger.warning(f"      ⚠️ Could not load {model_path.name}")
                    
            except Exception as e:
                logger.error(f"      ❌ Failed to load {model_path.name}: {e}")
        
        logger.info(f"📊 Successfully loaded {loaded_count}/{len(model_files)} models")
        
        if loaded_count > 0:
            logger.info("📋 Available models:")
            for name, info in self.model_info.items():
                logger.info(f"   • {name}: {len(info.get('features', []))} features, "
                          f"type={info.get('model_type', 'Unknown')}")
        
        return self.models
    
    def _extract_model_name(self, model_path: Path) -> str:
        """Extract a clean model name from the file path"""
        name = model_path.stem
        
        # Remove common suffixes
        name = name.replace('_model', '')
        name = name.replace('model_', '')
        name = name.replace('_pkl', '')
        
        # If the name is too generic, use the full stem
        if name in ['model', 'silver', 'gold', 'usd']:
            name = model_path.stem
        
        return name
    
    def _load_model_file(self, model_path: Path) -> Optional[Dict]:
        """Load a model file that contains model, scaler, and features"""
        try:
            # Load the pickle file
            model_data = joblib.load(model_path)
            
            # Check the structure
            if isinstance(model_data, dict):
                # Check if it has the expected structure
                if 'model' in model_data:
                    # Validate it has a predict method
                    if hasattr(model_data['model'], 'predict'):
                        return model_data
                    else:
                        logger.warning(f"      ⚠️ 'model' object doesn't have predict method")
                        return None
                else:
                    # Try to reconstruct the expected structure
                    logger.info(f"      🔧 Attempting to reconstruct model structure")
                    logger.info(f"         Keys found: {list(model_data.keys())}")
                    
                    result = {}
                    
                    # Look for model object
                    for key, value in model_data.items():
                        if hasattr(value, 'predict'):
                            result['model'] = value
                            logger.info(f"         Found model in key: '{key}'")
                        elif hasattr(value, 'transform'):
                            result['scaler'] = value
                            logger.info(f"         Found scaler in key: '{key}'")
                        elif isinstance(value, list) and len(value) > 0:
                            if isinstance(value[0], str):
                                result['feature_cols'] = value
                                logger.info(f"         Found features in key: '{key}' ({len(value)} features)")
                    
                    if 'model' in result:
                        return result
                    else:
                        logger.warning(f"      ⚠️ No model object found in dictionary")
                        return None
                    
            elif hasattr(model_data, 'predict'):
                # It's just the model object
                logger.info(f"      🔧 File contains only model object")
                return {
                    'model': model_data,
                    'scaler': None,
                    'feature_cols': []
                }
            else:
                logger.warning(f"      ⚠️ Unknown model file format")
                return None
            
        except Exception as e:
            logger.error(f"      ❌ Error loading {model_path.name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def get_model(self, model_type: str = "direction", horizon: str = "monthly"):
        """Get a specific model by type and horizon"""
        model_key = f"{horizon}_{model_type}"
        
        logger.info(f"🔍 Looking for model: {model_key}")
        logger.info(f"📋 Available models: {list(self.models.keys())}")
        
        # Try exact match
        if model_key in self.models:
            logger.info(f"✅ Found exact match: {model_key}")
            return self.models[model_key]
        
        # Try variations
        variations = [
            f"{horizon}_{model_type}",
            f"{model_type}_{horizon}",
            f"{horizon}_direction",
            f"{horizon}_volatility",
            "monthly_direction",
            "weekly_direction",
            model_type,
            horizon
        ]
        
        for variation in variations:
            if variation in self.models:
                logger.info(f"✅ Found variation: {variation}")
                return self.models[variation]
        
        # Try partial match
        for key in self.models.keys():
            if model_type in key.lower() and horizon in key.lower():
                logger.info(f"✅ Found partial match: {key}")
                return self.models[key]
        
        # Return first available model if any
        if self.models:
            first_key = list(self.models.keys())[0]
            logger.warning(f"⚠️ No match found, using first available model: {first_key}")
            return self.models[first_key]
        
        logger.error(f"❌ No models available!")
        return None
    
    def predict(self, features_dict: Dict, model_type: str = "direction", horizon: str = "monthly"):
        """Make a prediction using the specified model"""
        model_data = self.get_model(model_type, horizon)
        
        if not model_data:
            return {'error': 'No model available'}
        
        model = model_data.get('model')
        scaler = model_data.get('scaler')
        feature_cols = model_data.get('feature_cols', [])
        
        if model is None:
            return {'error': 'Model not found in loaded data'}
        
        try:
            # Prepare features in the correct order
            features_list = []
            missing_features = []
            
            logger.info(f"🔧 Preparing features for prediction...")
            logger.info(f"   Model expects {len(feature_cols)} features")
            logger.info(f"   Received {len(features_dict)} features in dict")
            
            for feature_name in feature_cols:
                if feature_name in features_dict:
                    value = features_dict[feature_name]
                    
                    # Handle None and NaN values
                    if value is None:
                        features_list.append(0.0)
                        missing_features.append(feature_name + " (None)")
                    elif hasattr(value, '__float__') and pd.isna(value):
                        features_list.append(0.0)
                        missing_features.append(feature_name + " (NaN)")
                    else:
                        try:
                            features_list.append(float(value))
                        except (ValueError, TypeError):
                            features_list.append(0.0)
                            missing_features.append(feature_name + " (invalid type)")
                else:
                    features_list.append(0.0)
                    missing_features.append(feature_name + " (not in dict)")
            
            # Log feature preparation results
            if missing_features:
                logger.warning(f"⚠️ {len(missing_features)} features missing or invalid:")
                for i, feat in enumerate(missing_features[:5]):
                    logger.warning(f"   {i+1}. {feat}")
                if len(missing_features) > 5:
                    logger.warning(f"   ... and {len(missing_features) - 5} more")
            else:
                logger.info(f"✅ All {len(feature_cols)} features prepared successfully")
            
            # Convert to numpy array
            features_array = np.array([features_list])
            logger.info(f"📊 Feature array shape: {features_array.shape}")
            
            # Scale features if scaler exists
            if scaler is not None:
                features_scaled = scaler.transform(features_array)
                logger.info(f"📏 Features scaled using {type(scaler).__name__}")
            else:
                features_scaled = features_array
                logger.info(f"📏 No scaler applied")
            
            # Make prediction
            if hasattr(model, 'predict_proba'):  # Classification model
                prediction = model.predict(features_scaled)[0]
                probability = model.predict_proba(features_scaled)[0]
                
                result = {
                    'direction': 'UP' if prediction == 1 else 'DOWN',
                    'prediction': int(prediction),
                    'probability': float(probability[int(prediction)]),
                    'up_probability': float(probability[1]) if len(probability) > 1 else 0.0,
                    'down_probability': float(probability[0]),
                    'model_type': 'classification'
                }
                logger.info(f"🎯 Prediction: {result['direction']} (probability: {result['probability']:.2%})")
            else:  # Regression model (volatility)
                prediction = model.predict(features_scaled)[0]
                result = {
                    'prediction': float(prediction),
                    'model_type': 'regression'
                }
                logger.info(f"🎯 Prediction: {prediction:.4f}")
            
            # Add metadata
            result['features_used'] = len(features_list) - len(missing_features)
            result['features_missing'] = len(missing_features)
            result['model_name'] = f"{horizon}_{model_type}"
            
            return result
            
        except Exception as e:
            import traceback
            logger.error(f"❌ Prediction error: {e}")
            logger.error(traceback.format_exc())
            return {'error': f'Prediction failed: {str(e)}'}
    
    def get_available_models(self):
        """Get list of available models"""
        available = []
        for name, data in self.models.items():
            model_type = 'direction' if 'direction' in name else ('volatility' if 'volatility' in name else 'unknown')
            horizon = 'monthly' if 'monthly' in name else ('weekly' if 'weekly' in name else 'unknown')
            
            available.append({
                'name': name,
                'type': model_type,
                'horizon': horizon,
                'has_scaler': data.get('scaler') is not None,
                'feature_count': len(data.get('feature_cols', [])),
                'model_class': type(data.get('model')).__name__ if data.get('model') else 'Unknown'
            })
        
        return available
    
    def get_required_features(self, model_name: str = "monthly_direction"):
        """Get list of features required by a specific model"""
        if model_name in self.models:
            return self.models[model_name].get('feature_cols', [])
        
        # Try to find any model
        for name, data in self.models.items():
            if 'feature_cols' in data:
                return data['feature_cols']
        
        return []


# Singleton instance
_model_loader = None

def get_model_loader():
    """Get or create the singleton model loader"""
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader()
        _model_loader.load_all_models()
    return _model_loader