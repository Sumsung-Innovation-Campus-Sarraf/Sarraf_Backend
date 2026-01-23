"""
Model loader service - Updated for single pickle file format
FIXED: Added missing pandas import
"""

import joblib
import pickle
import os
from pathlib import Path
from typing import Any, Optional, Tuple, Dict
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

class ModelLoader:
    """Handles loading models from single pickle files"""
    
    def __init__(self, models_dir: str = "."):  # Changed from "./models" to current directory
        self.models_dir = Path(models_dir)
        
        self.models = {}  # Format: {'monthly_direction': {'model': ..., 'scaler': ..., 'features': ...}}
        self.model_info = {}
    def get_model_metrics(self, model_name: str = "monthly_direction"):
        """Get model metrics like accuracy if saved with the model"""
        if model_name in self.models:
            model_data = self.models[model_name]
            
            # Check if metrics were saved with the model
            if 'metrics' in model_data:
                return model_data['metrics']
            elif 'accuracy' in model_data:
                return {'accuracy': model_data['accuracy']}
        
        # Try to extract from model attributes
        model_obj = self.models.get(model_name, {}).get('model')
        if hasattr(model_obj, 'best_score_'):
            return {'best_score': model_obj.best_score_}
        elif hasattr(model_obj, 'score'):
            # This would require test data to calculate
            return {'has_score_method': True}
        
        return {'error': 'No metrics found in model file'}

    def load_all_models(self):
        """Load all available model files from directory"""
        print("📦 Loading models...")
        
        # Look for model files
        model_files = []
        
        # Check current directory first
        for file in self.models_dir.glob("*model*.pkl"):
            model_files.append(file)
        
        # Also check for specific model names
        model_names = [
            "monthly_direction_model.pkl",
            "weekly_direction_model.pkl",
            "monthly_volatility_model.pkl",
            "weekly_volatility_model.pkl"
        ]
        
        for model_name in model_names:
            model_path = self.models_dir / model_name
            if model_path.exists() and model_path not in model_files:
                model_files.append(model_path)
        
        # Load each model
        for model_path in model_files:
            try:
                model_data = self._load_model_file(model_path)
                if model_data:
                    model_name = model_path.stem.replace("_model", "")
                    self.models[model_name] = model_data
                    
                    # Extract model info
                    self.model_info[model_name] = {
                        'path': str(model_path),
                        'loaded': True,
                        'features': model_data.get('feature_cols', []),
                        'has_scaler': model_data.get('scaler') is not None,
                        'has_model': model_data.get('model') is not None
                    }
                    
                    print(f"  ✅ Loaded {model_name} from {model_path.name}")
                    
            except Exception as e:
                print(f"  ❌ Failed to load {model_path.name}: {e}")
        
        print(f"📊 Total models loaded: {len(self.models)}")
        
        # Print model details
        for name, info in self.model_info.items():
            print(f"    {name}: {len(info.get('features', []))} features")
        
        return self.models
    
    def _load_model_file(self, model_path: Path) -> Optional[Dict]:
        """Load a model file that contains model, scaler, and features"""
        try:
            # Load the pickle file
            model_data = joblib.load(model_path)
            
            # Check the structure
            if isinstance(model_data, dict):
                # Check if it has the expected structure
                if 'model' in model_data:
                    return model_data
                else:
                    # Might be a different format, try to adapt
                    print(f"  ⚠️  Model file {model_path.name} has unexpected format")
                    print(f"     Keys: {list(model_data.keys())}")
                    
                    # Try to find model, scaler, features in the dict
                    result = {}
                    
                    # Look for a model object
                    for key, value in model_data.items():
                        if hasattr(value, 'predict'):
                            result['model'] = value
                        elif hasattr(value, 'transform'):
                            result['scaler'] = value
                        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], str):
                            result['feature_cols'] = value
                    
                    if 'model' in result:
                        return result
                    
            elif hasattr(model_data, 'predict'):
                # It's just the model object
                return {
                    'model': model_data,
                    'scaler': None,
                    'feature_cols': []
                }
            
            return None
            
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            return None
    
    def get_model(self, model_type: str = "direction", horizon: str = "monthly"):
        """Get a specific model by type and horizon"""
        model_key = f"{horizon}_{model_type}"
        
        # Try exact match
        if model_key in self.models:
            return self.models[model_key]
        
        # Try variations
        variations = [
            f"{horizon}_direction",  # monthly_direction
            f"{horizon}_volatility", # monthly_volatility
            "monthly_direction",     # fallback
            "weekly_direction",      # fallback
        ]
        
        for variation in variations:
            if variation in self.models:
                return self.models[variation]
        
        # Return first available model if any
        if self.models:
            first_key = list(self.models.keys())[0]
            return self.models[first_key]
        
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
                
                # Debug: Show what's actually in the features_dict vs what the model expects
                if missing_features:
                    print(f"\n🔍 DEBUG - Feature mismatch:")
                    print(f"Model expects {len(feature_cols)} features")
                    print(f"Received {len(features_dict)} features")
                    
                    # Check first few features
                    print("\nFirst 10 features in dict:")
                    for i, (key, value) in enumerate(list(features_dict.items())[:10]):
                        print(f"  {key}: {value} (type: {type(value).__name__})")
                    
                    print(f"\n⚠️  Missing or invalid features: {len(missing_features)}")
                    for feature in missing_features[:10]:
                        print(f"  {feature}")
                    if len(missing_features) > 10:
                        print(f"  ... and {len(missing_features) - 10} more")
                
                # Convert to numpy array
                features_array = np.array([features_list])
                
                # Scale features if scaler exists
                if scaler is not None:
                    features_scaled = scaler.transform(features_array)
                else:
                    features_scaled = features_array
                
                # Make prediction
                if hasattr(model, 'predict_proba'):  # Classification model
                    prediction = model.predict(features_scaled)[0]
                    probability = model.predict_proba(features_scaled)[0]
                    
                    result = {
                        'direction': 'UP' if prediction == 1 else 'DOWN',
                        'prediction': int(prediction),
                        'probability': float(probability[int(prediction)]),
                        'up_probability': float(probability[1]),
                        'down_probability': float(probability[0]),
                        'model_type': 'classification'
                    }
                else:  # Regression model (volatility)
                    prediction = model.predict(features_scaled)[0]
                    result = {
                        'prediction': float(prediction),
                        'model_type': 'regression'
                    }
                
                # Add metadata
                result['features_used'] = len(features_list) - len(missing_features)
                result['features_missing'] = missing_features
                result['model_name'] = f"{horizon}_{model_type}"
                
                return result
                
            except Exception as e:
                import traceback
                print(f"❌ Prediction error: {e}")
                print(traceback.format_exc())
                return {'error': f'Prediction failed: {str(e)}'}
    
    def get_available_models(self):
        """Get list of available models"""
        available = []
        for name, data in self.models.items():
            model_type = 'direction' if 'direction' in name else 'volatility'
            horizon = 'monthly' if 'monthly' in name else 'weekly'
            
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