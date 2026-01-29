"""
FastAPI Backend for Gold, Silver and USD/DZD Forecasting
"""
from fastapi import HTTPException, Depends, Query, Response
from datetime import date as dt, datetime, timedelta
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import os
from dotenv import load_dotenv
from supabase import create_client, Client
import io
import csv

# Import services
from services.features import GoldFeatureCalculator, get_feature_calculator, SilverFeatureCalculator, get_silver_feature_calculator
from services.model_loader import get_model_loader, ModelLoader
from services.usd_data_fetcher import USDDataFetcher
from services.usd_forecaster import USDForecaster

# Load environment variables
load_dotenv()

from fastapi import FastAPI
from routes import routes
from core.middlewares import setup_middlewares
from apscheduler.schedulers.background import BackgroundScheduler
from services.daily_updates import insert_daily_rates
import logging
import atexit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Gold, Silver and USD/DZD Forecast API",
    description="API for forecasting gold/silver price direction and USD/DZD exchange rates",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.include_router(routes.router)
setup_middlewares(app)

# Scheduler for daily updates
def daily_job():
    try:
        data = insert_daily_rates()
        logger.info(f"Données journalières insérées : {data}")
    except Exception as e:
        logger.error(f"Erreur lors de l'insertion quotidienne : {e}")

scheduler = BackgroundScheduler()
scheduler.add_job(daily_job, 'cron', hour=9, minute=0)
scheduler.start()
atexit.register(lambda: scheduler.shutdown())

# Add CORS middleware
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== PYDANTIC MODELS ====================

# Gold Models
class GoldPredictionRequest(BaseModel):
    model_type: str = "direction"
    horizon: str = "monthly"
    force_refresh: bool = False

class GoldPredictionResponse(BaseModel):
    success: bool
    timestamp: str
    model_used: str
    prediction: Dict[str, Any]
    features: Dict[str, Any]
    gold_price: Optional[float] = None
    message: Optional[str] = None

class GoldHealthResponse(BaseModel):
    status: str
    timestamp: str
    models_loaded: int
    available_models: List[str]
    feature_calculator_ready: bool

# Silver Models
class SilverPredictionRequest(BaseModel):
    model_type: str = "direction"
    horizon: str = "monthly"
    force_refresh: bool = False

class SilverPredictionResponse(BaseModel):
    success: bool
    timestamp: str
    model_used: str
    prediction: Dict[str, Any]
    features: Dict[str, Any]
    silver_price: Optional[float] = None
    message: Optional[str] = None

class SilverHealthResponse(BaseModel):
    status: str
    timestamp: str
    models_loaded: int
    available_models: List[str]
    feature_calculator_ready: bool

# USD/DZD Models
class USDForecastRequest(BaseModel):
    use_cached: bool = True
    model_type: str = "level"

class USDForecastResponse(BaseModel):
    success: bool
    timestamp: str
    date: str
    forecast: Optional[float] = None
    current_rate: Optional[float] = None
    expected_change: Optional[float] = None
    change_probability: Optional[float] = None
    confidence_interval: Optional[Dict[str, float]] = None
    features_used: List[str]
    data_source: str
    message: Optional[str] = None

class USDHealthResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool
    scaler_loaded: bool
    features_loaded: bool
    supabase_connected: bool
    feature_count: int
    model_info: Optional[Dict] = None

class USDHistoryResponse(BaseModel):
    success: bool
    timestamp: str
    date_range: Dict[str, str]
    total_records: int
    statistics: Dict[str, Any]
    data: List[Dict[str, Any]]
    rate_type: str

# ==================== GLOBAL COMPONENTS ====================

GOLD_MODEL_LOADER = None
GOLD_FEATURE_CALCULATOR = None
SILVER_MODEL_LOADER = None
SILVER_FEATURE_CALCULATOR = None
USD_DATA_FETCHER = None
USD_FORECASTER = None

# ==================== DEPENDENCIES ====================

def get_supabase_client() -> Client:
    """Get Supabase client"""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    
    if not url or not key:
        raise HTTPException(status_code=500, detail="Supabase credentials not configured")
    
    return create_client(url, key)

# Gold Dependencies
def get_gold_model_loader():
    """Get gold model loader"""
    global GOLD_MODEL_LOADER
    if GOLD_MODEL_LOADER is None:
        GOLD_MODEL_LOADER = get_model_loader()
    return GOLD_MODEL_LOADER

def get_gold_feature_calculator():
    """Get gold feature calculator"""
    global GOLD_FEATURE_CALCULATOR
    if GOLD_FEATURE_CALCULATOR is None:
        model_loader = get_gold_model_loader()
        GOLD_FEATURE_CALCULATOR = get_feature_calculator(model_loader)
    return GOLD_FEATURE_CALCULATOR

# Silver Dependencies
def get_silver_model_loader():
    """Get silver model loader - separate from gold"""
    global SILVER_MODEL_LOADER
    if SILVER_MODEL_LOADER is None:
        from services.model_loader import ModelLoader
        
        # Silver models should be in a DIFFERENT directory or have DIFFERENT names
        # Option 1: Separate directory
        SILVER_MODEL_LOADER = ModelLoader(models_dir="models/silver")
        
        # Option 2: Use naming convention in same directory
        # SILVER_MODEL_LOADER = ModelLoader(models_dir=".")
        
        SILVER_MODEL_LOADER.load_all_models()
        
        if SILVER_MODEL_LOADER.models:
            logger.info(f"✅ Silver models loaded: {list(SILVER_MODEL_LOADER.models.keys())}")
        else:
            logger.warning("⚠️ No silver models found!")
            logger.warning("   Expected silver model files like: silver_monthly_direction_model.pkl")
    
    return SILVER_MODEL_LOADER

def get_silver_feature_calculator():
    """Get silver feature calculator"""
    global SILVER_FEATURE_CALCULATOR
    if SILVER_FEATURE_CALCULATOR is None:
        supabase = get_supabase_client()
        SILVER_FEATURE_CALCULATOR = get_silver_feature_calculator(supabase)
        logger.info("✅ Silver feature calculator initialized")
    return SILVER_FEATURE_CALCULATOR

# USD/DZD Dependencies
def get_usd_data_fetcher() -> USDDataFetcher:
    """Get USD data fetcher"""
    global USD_DATA_FETCHER
    
    if USD_DATA_FETCHER is None:
        supabase = get_supabase_client()
        USD_DATA_FETCHER = USDDataFetcher(supabase)
    
    return USD_DATA_FETCHER

def get_usd_forecaster() -> USDForecaster:
    """Get USD forecaster - Option 1: usd_model.pkl"""
    global USD_FORECASTER
    
    if USD_FORECASTER is None:
        try:
            data_fetcher = get_usd_data_fetcher()
            USD_FORECASTER = USDForecaster(data_fetcher=data_fetcher)
            
            model_path = "models/usd_model.pkl"
            scaler_X_path = "models/scaler_X.pkl"
            scaler_y_path = "models/scaler_y.pkl"
            
            USD_FORECASTER.load_models(model_path, scaler_X_path, scaler_y_path)
            logger.info("✅ USD/DZD models loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing USD forecaster: {e}")
            raise
    
    return USD_FORECASTER
# ==================== STARTUP EVENT ====================

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    print("🚀 Starting Gold, Silver and USD/DZD Forecast API...")
    
    # Initialize gold components
    try:
        model_loader = get_gold_model_loader()
        feature_calculator = get_gold_feature_calculator()
        print("✅ Gold components initialized successfully")
    except Exception as e:
        print(f"⚠️ Gold components initialization warning: {e}")
    
    # Initialize silver components
    try:
        silver_loader = get_silver_model_loader()
        silver_calculator = get_silver_feature_calculator()
        print("✅ Silver components initialized successfully")
        if silver_loader and silver_loader.models:
            print(f"📊 Silver Models: {list(silver_loader.models.keys())}")
    except Exception as e:
        print(f"⚠️ Silver components initialization warning: {e}")
    
    # Initialize USD/DZD components
    try:
        forecaster = get_usd_forecaster()
        if forecaster.is_loaded():
            print("✅ USD/DZD models loaded successfully")
            info = forecaster.get_model_info()
            print(f"📊 USD/DZD Model Info:")
            print(f"  - Model type: {info.get('model_type')}")
            print(f"  - Feature count: {info.get('feature_count')}")
        else:
            print("⚠️ USD/DZD models failed to load")
    except Exception as e:
        print(f"❌ Error initializing USD/DZD components: {e}")
    
    print("✅ API initialized successfully")

# ==================== ROOT ENDPOINT ====================

@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "Gold and USD/DZD Forecast API",
        "version": "2.0.0",
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc"
        },
        "endpoints": {
            "gold": {
                "health": "/health",
                "predict": "/predict",
                "models": "/models",
                "features": "/features/latest",
                "gold_price_history": "/gold-price-history",
                "test": "/test"
            },
            "usd": {
                "health": "/usd/health",
                "forecast": "/usd/forecast",
                "features": "/usd/features",
                "history": "/usd/history",
                "model_info": "/usd/model/info",
                "test": "/usd/test"
            }
        }
    }

# ==================== GOLD ENDPOINTS ====================

@app.get("/health", response_model=GoldHealthResponse)
async def gold_health_check():
    """Check Gold API health status"""
    try:
        model_loader = get_gold_model_loader()
        models_loaded = len(model_loader.models) if model_loader else 0
        available_models = list(model_loader.models.keys()) if model_loader else []
        
        return GoldHealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            models_loaded=models_loaded,
            available_models=available_models,
            feature_calculator_ready=GOLD_FEATURE_CALCULATOR is not None
        )
    except Exception as e:
        return GoldHealthResponse(
            status=f"error: {str(e)[:100]}",
            timestamp=datetime.now().isoformat(),
            models_loaded=0,
            available_models=[],
            feature_calculator_ready=False
        )

@app.get("/models")
async def list_gold_models():
    """List all available gold models"""
    try:
        model_loader = get_gold_model_loader()
        if not model_loader:
            raise HTTPException(status_code=503, detail="Model loader not initialized")
        
        models = model_loader.get_available_models()
        
        return {
            "success": True,
            "count": len(models),
            "models": models,
            "default_model": "monthly_direction"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/features/latest")
async def get_latest_gold_features():
    """Get the latest calculated gold features"""
    try:
        feature_calculator = get_gold_feature_calculator()
        if not feature_calculator:
            raise HTTPException(status_code=503, detail="Feature calculator not initialized")
        
        features = feature_calculator.get_features_for_prediction()

        if not features:
            raise HTTPException(status_code=404, detail="Could not calculate features")
        
        # Filter out only the features needed by the model
        model_loader = get_gold_model_loader()
        model_features = {}
        if model_loader:
            required_features = model_loader.get_required_features("monthly_direction")
            for feat in required_features:
                if feat in features:
                    model_features[feat] = features[feat]
        
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "features_count": len(features),
            "model_features_count": len(model_features),
            "all_features": features,
            "model_features": model_features,
            "gold_price": features.get('gold_price_usd')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/gold-price-history")
async def get_gold_price_history(
    range: str = Query("1y", description="Time range: '1m', '2m', '6m', '1y', '2y', '5y', 'all'"),
    format: str = Query("json", description="Output format: 'json' or 'csv'")
):
    """
    Get historical gold prices from database
    """
    try:
        # Calculate start date based on range
        end_date = datetime.now()
        
        if range == "1m":
            start_date = end_date - timedelta(days=30)
        elif range == "2m":
            start_date = end_date - timedelta(days=60)
        elif range == "6m":
            start_date = end_date - timedelta(days=180)
        elif range == "1y":
            start_date = end_date - timedelta(days=365)
        elif range == "2y":
            start_date = end_date - timedelta(days=730)
        elif range == "5y":
            start_date = end_date - timedelta(days=1825)
        elif range == "all":
            start_date = datetime(2000, 1, 1)  # Far back date
        else:
            return {
                "success": False,
                "error": f"Invalid range: {range}. Use: 1m, 2m, 6m, 1y, 2y, 5y, all"
            }
        
        # Fetch data from Supabase using feature calculator
        feature_calculator = get_gold_feature_calculator()
        if not feature_calculator:
            raise HTTPException(status_code=503, detail="Feature calculator not initialized")
        
        start_date_str = start_date.date().isoformat()
        end_date_str = end_date.date().isoformat()
        
        response = feature_calculator.supabase.table("gold_silver_dataset") \
            .select("date, gold_price_usd") \
            .gte("date", start_date_str) \
            .lte("date", end_date_str) \
            .order("date", desc=False) \
            .execute()
        
        if not response.data:
            return {
                "success": False,
                "error": "No historical data found",
                "range": range,
                "start_date": start_date_str,
                "end_date": end_date_str
            }
        
        # Format the data
        history = []
        for record in response.data:
            history.append({
                "date": record["date"],
                "price": float(record["gold_price_usd"]) if record["gold_price_usd"] else 0.0
            })
        
        # Calculate some statistics
        prices = [h["price"] for h in history if h["price"] > 0]
        stats = {}
        if prices:
            stats = {
                "current_price": prices[-1] if prices else 0.0,
                "min_price": min(prices),
                "max_price": max(prices),
                "avg_price": sum(prices) / len(prices),
                "price_change": prices[-1] - prices[0] if len(prices) > 1 else 0,
                "percent_change": ((prices[-1] - prices[0]) / prices[0] * 100) if prices[0] > 0 else 0
            }
        
        # Return in requested format
        result = {
            "success": True,
            "range": range,
            "start_date": start_date_str,
            "end_date": end_date_str,
            "total_records": len(history),
            "statistics": stats,
            "data": history
        }
        
        # Return CSV if requested
        if format.lower() == "csv":
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=["date", "price"])
            writer.writeheader()
            writer.writerows(history)
            
            return Response(
                content=output.getvalue(),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=gold_prices_{range}_{end_date_str}.csv"}
            )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_gold", response_model=GoldPredictionResponse)
async def make_gold_prediction(request: GoldPredictionRequest = None):
    """Make a gold direction prediction"""
    try:
        if request is None:
            request = GoldPredictionRequest()
        
        model_loader = get_gold_model_loader()
        feature_calculator = get_gold_feature_calculator()
        
        if not model_loader or not feature_calculator:
            raise HTTPException(status_code=503, detail="Components not initialized")
        
        # Get features
        features = feature_calculator.get_features_for_prediction()
        if not features:
            raise HTTPException(status_code=404, detail="Could not calculate features")
        
        # Extract only the features needed for prediction
        prediction_features = {}
        required_features = model_loader.get_required_features(f"{request.horizon}_{request.model_type}")
        
        for feat in required_features:
            if feat in features:
                prediction_features[feat] = features[feat]
            else:
                # Use default value for missing features
                prediction_features[feat] = 0.0
        
        # Make prediction
        prediction = model_loader.predict(
            prediction_features,
            model_type=request.model_type,
            horizon=request.horizon
        )
        
        if 'error' in prediction:
            raise HTTPException(status_code=500, detail=prediction['error'])
        
        # Prepare response
        response = GoldPredictionResponse(
            success=True,
            timestamp=datetime.now().isoformat(),
            model_used=f"{request.horizon}_{request.model_type}",
            prediction=prediction,
            features=prediction_features,
            gold_price=features.get('gold_price_usd'),
            message="Prediction successful"
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/test")
async def gold_test_endpoint():
    """Test endpoint for quick verification"""
    try:
        # Test feature calculation
        feature_calculator = get_gold_feature_calculator()
        model_loader = get_gold_model_loader()
        
        features = feature_calculator.get_features_for_prediction() if feature_calculator else {}
        
        # Test model loading
        models_available = len(model_loader.models) if model_loader else 0
        
        # Make a test prediction
        test_features = {
            'hist_vol_20d': 0.15,
            'return_5d': 0.02,
            'vix_lag1': 18.5,
            'gold_price_usd': 2100.50
        }
        
        test_prediction = None
        if model_loader:
            test_prediction = model_loader.predict(
                test_features,
                model_type="direction",
                horizon="monthly"
            )
        
        return {
            "status": "operational",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "feature_calculator": "ready" if feature_calculator else "not ready",
                "model_loader": "ready" if model_loader else "not ready",
                "models_loaded": models_available
            },
            "test": {
                "features_calculated": len(features),
                "test_prediction": test_prediction
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
# Replace your Silver endpoint section with this:

# ==================== SILVER DEPENDENCIES ====================

SILVER_MODEL_LOADER = None
SILVER_FEATURE_CALCULATOR = None

def get_silver_model_loader():
    """Get silver model loader"""
    global SILVER_MODEL_LOADER
    if SILVER_MODEL_LOADER is None:
        from services.model_loader import ModelLoader
        # Use current directory "." not "./models"
        SILVER_MODEL_LOADER = ModelLoader(models_dir=".")
        SILVER_MODEL_LOADER.load_all_models()
        
        if SILVER_MODEL_LOADER.models:
            logger.info(f"✅ Silver models loaded: {list(SILVER_MODEL_LOADER.models.keys())}")
        else:
            logger.warning("⚠️ No silver models found!")
            
    return SILVER_MODEL_LOADER

def get_silver_feature_calculator():
    """Get silver feature calculator"""
    global SILVER_FEATURE_CALCULATOR
    if SILVER_FEATURE_CALCULATOR is None:
        from services.features import SilverFeatureCalculator  # Import the class
        supabase = get_supabase_client()
        SILVER_FEATURE_CALCULATOR = SilverFeatureCalculator(supabase)  # ✅ CORRECT
        logger.info("✅ Silver feature calculator initialized")
    return SILVER_FEATURE_CALCULATOR

# ==================== SILVER ENDPOINTS ====================

class SilverHealthResponse(BaseModel):
    status: str
    timestamp: str
    models_loaded: int
    available_models: List[str]
    feature_calculator_ready: bool

@app.get("/silver/health", response_model=SilverHealthResponse)
async def silver_health_check():
    """Check Silver API health status"""
    try:
        model_loader = get_silver_model_loader()
        feature_calc = get_silver_feature_calculator()
        
        models_loaded = len(model_loader.models) if model_loader else 0
        available_models = list(model_loader.models.keys()) if model_loader else []
        
        return SilverHealthResponse(
            status="healthy" if models_loaded > 0 else "no models loaded",
            timestamp=datetime.now().isoformat(),
            models_loaded=models_loaded,
            available_models=available_models,
            feature_calculator_ready=feature_calc is not None
        )
    except Exception as e:
        logger.error(f"Silver health check error: {e}")
        return SilverHealthResponse(
            status=f"error: {str(e)[:100]}",
            timestamp=datetime.now().isoformat(),
            models_loaded=0,
            available_models=[],
            feature_calculator_ready=False
        )

@app.get("/silver/features/latest")
async def get_latest_silver_features():
    """Get latest calculated silver features"""
    try:
        feature_calculator = get_silver_feature_calculator()
        if not feature_calculator:
            raise HTTPException(
                status_code=503, 
                detail="Silver feature calculator not initialized"
            )
        
        logger.info("📊 Calculating silver features...")
        features = feature_calculator.get_features_for_prediction()
        
        if not features:
            raise HTTPException(
                status_code=404, 
                detail="Could not calculate silver features"
            )
        
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "features_count": len(features),
            "all_features": features,
            "silver_price": features.get('silver_price_usd'),
            "model_features": {
                k: features.get(k) 
                for k in ['silver_return_5d', 'silver_return_10d', 'silver_return_30d',
                         'gold_silver_ratio_lag', 'dxy_return_30d', 'dxy_vol_30d',
                         'sp500_return_30d', 'sp500_vol_30d', 'vix_lag1', 'vix_lag7',
                         'crude_oil_return_30d']
                if k in features
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting silver features: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_silver", response_model=SilverPredictionResponse)
async def predict_silver(
    request: SilverPredictionRequest = None
):
    """Make a silver direction prediction"""
    try:
        if request is None:
            request = SilverPredictionRequest()
        
        logger.info(f"🎯 Making silver prediction: {request.model_type}/{request.horizon}")
        
        # Get feature calculator (this is correct)
        feature_calculator = get_silver_feature_calculator()
        
        if not feature_calculator:
            raise HTTPException(
                status_code=503, 
                detail="Silver feature calculator not initialized"
            )
        
        # Get features (11 silver-specific features)
        logger.info("📊 Calculating silver features...")
        features = feature_calculator.get_features_for_prediction()
        
        if not features:
            raise HTTPException(
                status_code=404, 
                detail="Could not calculate silver features"
            )
        
        logger.info(f"✅ Calculated {len(features)} features")
        logger.info(f"📋 Features: {list(features.keys())}")
        
        
        
        # ✅ SOLUTION: Load silver model directly
        from pathlib import Path
        import joblib
        
        # Try to find silver model
        silver_model_path = None
        possible_paths = [
            Path("models/silver") / f"{request.horizon}_{request.model_type}_model.pkl",
            Path("models") / f"silver_{request.horizon}_{request.model_type}_model.pkl",
            Path(".") / f"silver_{request.horizon}_{request.model_type}_model.pkl",
            Path("models/silver") / "monthly_direction_model.pkl",  # Default
            Path("models") / "silver_model.pkl",  # Generic
        ]
        
        for path in possible_paths:
            if path.exists():
                silver_model_path = path
                logger.info(f"✅ Found silver model: {path}")
                break
        
        if not silver_model_path:
            raise HTTPException(
                status_code=404,
                detail=f"Silver model not found. Tried: {[str(p) for p in possible_paths]}"
            )
        
        # Load the model
        try:
            model_data = joblib.load(silver_model_path)
            logger.info(f"📦 Loaded silver model from {silver_model_path}")
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error loading silver model: {e}"
            )
        
        # Extract model components
        if isinstance(model_data, dict):
            model = model_data.get('model')
            scaler = model_data.get('scaler')
            feature_cols = model_data.get('feature_cols', [])
        else:
            model = model_data
            scaler = None
            feature_cols = list(features.keys())
        
        if not model:
            raise HTTPException(
                status_code=500,
                detail="Model object not found in model file"
            )
        
        logger.info(f"🔧 Model expects {len(feature_cols)} features")
        logger.info(f"📋 Required features: {feature_cols}")
        
        # Prepare features in correct order
        import numpy as np
        features_array = []
        missing_features = []
        
        for feat in feature_cols:
            if feat in features and features[feat] is not None:
                features_array.append(float(features[feat]))
            else:
                features_array.append(0.0)
                missing_features.append(feat)
        
        if missing_features:
            logger.warning(f"⚠️ Missing features: {missing_features}")
        
        X = np.array([features_array])
        
        # Scale if scaler exists
        if scaler:
            X = scaler.transform(X)
        
        # Make prediction
        if hasattr(model, 'predict_proba'):
            prediction = model.predict(X)[0]
            probability = model.predict_proba(X)[0]
            
            prediction_result = {
                'direction': 'UP' if prediction == 1 else 'DOWN',
                'prediction': int(prediction),
                'probability': float(probability[int(prediction)]),
                'up_probability': float(probability[1]) if len(probability) > 1 else 0.0,
                'down_probability': float(probability[0]),
                'model_type': 'classification'
            }
        else:
            prediction_value = model.predict(X)[0]
            prediction_result = {
                'prediction': float(prediction_value),
                'model_type': 'regression'
            }
        
        logger.info(f"✅ Prediction complete: {prediction_result}")
        
        # Get silver price
        silver_price = features.get('silver_price_usd', 0.0)
        
        # Prepare response
        response = SilverPredictionResponse(
            success=True,
            timestamp=datetime.now().isoformat(),
            model_used=f"{request.horizon}_{request.model_type}",
            prediction=prediction_result,
            features=features,  # Return the 11 silver features, not gold features
            silver_price=silver_price,
            message="Silver prediction successful"
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Silver prediction failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, 
            detail=f"Silver prediction failed: {str(e)}"
        )
    
@app.get("/silver-price-history")
async def get_silver_price_history(
    range: str = Query("1y", description="Time range: '1m', '2m', '6m', '1y', '2y', '5y', 'all'"),
    format: str = Query("json", description="Output format: 'json' or 'csv'")
):
    """Get historical silver prices from Supabase"""
    try:
        # Calculate start date based on range
        end_date = datetime.now()
        
        range_days = {
            "1m": 30,
            "2m": 60,
            "6m": 180,
            "1y": 365,
            "2y": 730,
            "5y": 1825,
            "all": 10000
        }
        
        if range not in range_days:
            return {
                "success": False, 
                "error": f"Invalid range: {range}. Use: {', '.join(range_days.keys())}"
            }
        
        if range == "all":
            start_date = datetime(2000, 1, 1)
        else:
            start_date = end_date - timedelta(days=range_days[range])

        # Fetch data from Supabase
        feature_calculator = get_silver_feature_calculator()
        start_date_str = start_date.date().isoformat()
        end_date_str = end_date.date().isoformat()
        
        logger.info(f"📊 Fetching silver price history from {start_date_str} to {end_date_str}")
        
        response = feature_calculator.supabase.table("gold_silver_dataset") \
            .select("date, silver_price_usd") \
            .gte("date", start_date_str) \
            .lte("date", end_date_str) \
            .order("date", desc=False) \
            .execute()
        
        if not response.data:
            return {
                "success": False,
                "error": "No historical data found",
                "range": range,
                "start_date": start_date_str,
                "end_date": end_date_str
            }
        
        # Format data
        history = []
        for r in response.data:
            if r.get("silver_price_usd"):
                history.append({
                    "date": r["date"], 
                    "price": float(r["silver_price_usd"])
                })
        
        # Calculate statistics
        prices = [h["price"] for h in history if h["price"] > 0]
        stats = {}
        
        if prices:
            stats = {
                "current_price": prices[-1],
                "min_price": min(prices),
                "max_price": max(prices),
                "avg_price": sum(prices) / len(prices),
                "price_change": prices[-1] - prices[0] if len(prices) > 1 else 0,
                "percent_change": ((prices[-1] - prices[0]) / prices[0] * 100) if prices[0] > 0 else 0
            }
        
        result = {
            "success": True,
            "range": range,
            "start_date": start_date_str,
            "end_date": end_date_str,
            "total_records": len(history),
            "statistics": stats,
            "data": history
        }

        # Return CSV if requested
        if format.lower() == "csv":
            import io
            import csv
            
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=["date", "price"])
            writer.writeheader()
            writer.writerows(history)
            
            return Response(
                content=output.getvalue(),
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename=silver_prices_{range}_{end_date_str}.csv"
                }
            )
        
        return result

    except Exception as e:
        logger.error(f"Error fetching silver price history: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
# In main.py, replace your USD endpoints section with this:

# ==================== USD/DZD Dependencies ====================

def get_usd_data_fetcher() -> USDDataFetcher:
    """Get USD data fetcher"""
    global USD_DATA_FETCHER
    
    if USD_DATA_FETCHER is None:
        supabase = get_supabase_client()
        USD_DATA_FETCHER = USDDataFetcher(supabase)
    
    return USD_DATA_FETCHER

def get_usd_forecaster() -> USDForecaster:
    """Get USD forecaster with better error handling"""
    global USD_FORECASTER
    
    if USD_FORECASTER is None:
        try:
            from pathlib import Path
            
            logger.info("🔧 Initializing USD forecaster...")
            
            # Initialize with data fetcher
            data_fetcher = get_usd_data_fetcher()
            USD_FORECASTER = USDForecaster(data_fetcher=data_fetcher)
            
            # Find model files
            models_dir = Path("models")
            
            # Auto-detect model file
            possible_model_files = [
                models_dir / "usd_model.pkl",
            ]
            
            model_path = None
            for path in possible_model_files:
                if path.exists():
                    model_path = str(path)
                    logger.info(f"✅ Found USD model: {path}")
                    break
            
            if not model_path:
                # List what files ARE in models/
                if models_dir.exists():
                    available = list(models_dir.glob("*.pkl"))
                    logger.error(f"❌ USD model not found. Available files: {[f.name for f in available]}")
                else:
                    logger.error(f"❌ models/ directory not found at {models_dir.absolute()}")
                
                raise FileNotFoundError(
                    f"USD model not found. Tried: {[str(p) for p in possible_model_files]}"
                )
            
            # Check for scalers
            scaler_X_path = str(models_dir / "scaler_X.pkl")
            scaler_y_path = str(models_dir / "scaler_y.pkl")
            
            if not Path(scaler_X_path).exists():
                raise FileNotFoundError(f"Scaler X not found: {scaler_X_path}")
            
            if not Path(scaler_y_path).exists():
                raise FileNotFoundError(f"Scaler Y not found: {scaler_y_path}")
            
            logger.info(f"📦 Loading USD models:")
            logger.info(f"   Model: {model_path}")
            logger.info(f"   Scaler X: {scaler_X_path}")
            logger.info(f"   Scaler Y: {scaler_y_path}")
            
            # Load models
            success = USD_FORECASTER.load_models(model_path, scaler_X_path, scaler_y_path)
            
            if not success:
                raise RuntimeError("USD model loading failed")
            
            if not USD_FORECASTER.is_loaded():
                raise RuntimeError("USD model components not properly loaded")
            
            logger.info("✅ USD/DZD models loaded successfully")
            
            # Log model info
            info = USD_FORECASTER.get_model_info()
            logger.info(f"📊 USD Model Info: {info}")
            
        except Exception as e:
            logger.error(f"❌ Error initializing USD forecaster: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Don't raise - let the endpoint handle it
            USD_FORECASTER = None
    
    return USD_FORECASTER

# ==================== USD/DZD ENDPOINTS ====================

@app.get("/usd/health", response_model=USDHealthResponse)
async def usd_health_check():
    """Check USD/DZD API health status"""
    try:
        forecaster = get_usd_forecaster()
        
        if forecaster is None:
            return USDHealthResponse(
                status="error: forecaster is None",
                timestamp=datetime.now().isoformat(),
                model_loaded=False,
                scaler_loaded=False,
                features_loaded=False,
                supabase_connected=False,
                feature_count=0,
                model_info=None
            )
        
        is_loaded = forecaster.is_loaded()
        
        return USDHealthResponse(
            status="healthy" if is_loaded else "models not loaded",
            timestamp=datetime.now().isoformat(),
            model_loaded=forecaster.model is not None,
            scaler_loaded=forecaster.scaler_X is not None and forecaster.scaler_y is not None,
            features_loaded=len(forecaster.feature_cols) > 0,
            supabase_connected=forecaster.data_fetcher is not None,
            feature_count=len(forecaster.feature_cols),
            model_info=forecaster.get_model_info() if is_loaded else None
        )
    except Exception as e:
        logger.error(f"USD health check error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        return USDHealthResponse(
            status=f"error: {str(e)[:100]}",
            timestamp=datetime.now().isoformat(),
            model_loaded=False,
            scaler_loaded=False,
            features_loaded=False,
            supabase_connected=False,
            feature_count=0,
            model_info=None
        )

@app.post("/usd/forecast", response_model=USDForecastResponse)
async def make_usd_forecast(
    request: USDForecastRequest = None
):
    """Make a USD/DZD forecast for today"""
    try:
        if request is None:
            request = USDForecastRequest()
        
        # Use today's date as target
        target_date = datetime.now().date().isoformat()

        logger.info(f"🎯 Making USD/DZD forecast for {target_date}")

        # Get forecaster
        forecaster = get_usd_forecaster()
        
        # Check if forecaster was initialized
        if forecaster is None:
            raise HTTPException(
                status_code=503, 
                detail="USD/DZD forecaster failed to initialize. Check logs for details."
            )
        
        # Check if components are loaded
        if not forecaster.is_loaded():
            # Try to provide more details about what's missing
            details = []
            if forecaster.model is None:
                details.append("model is None")
            if forecaster.scaler_X is None:
                details.append("scaler_X is None")
            if forecaster.scaler_y is None:
                details.append("scaler_y is None")
            
            raise HTTPException(
                status_code=503, 
                detail=f"USD/DZD model components not loaded: {', '.join(details)}"
            )

        # Make forecast
        logger.info("🔮 Generating forecast...")
        forecast_result = await forecaster.forecast(target_date)

        logger.info(f"✅ Forecast complete: {forecast_result.get('predicted_rate', 'N/A')}")

        return USDForecastResponse(
            success=True,
            timestamp=datetime.now().isoformat(),
            date=target_date,
            forecast=forecast_result.get("predicted_rate"),
            current_rate=forecast_result.get("current_rate"),
            expected_change=forecast_result.get("expected_change"),
            change_probability=forecast_result.get("percent_change"),
            confidence_interval=forecast_result.get("confidence_interval"),
            features_used=forecast_result.get("features_used", []),
            data_source="Supabase/YahooFinance",
            message="USD/DZD forecast successful"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ USD/DZD forecast failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, 
            detail=f"USD/DZD forecast failed: {str(e)}"
        )

@app.get("/usd/history")
async def get_usd_history(
    days: int = Query(30, description="Number of days of history to retrieve"),
    rate_type: str = Query("both", description="Rate type: 'parallel', 'official', 'both'"),
    format: str = Query("json", description="Output format: 'json' or 'csv'"),
    data_fetcher: USDDataFetcher = Depends(get_usd_data_fetcher)
):
    """Get historical USD/DZD data"""
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        logger.info(f"📊 Fetching USD/DZD history from {start_date.date()} to {end_date.date()}")
        
        # Get historical data
        history = data_fetcher.get_usd_history(start_date, end_date, rate_type)
        
        if not history:
            return USDHistoryResponse(
                success=False,
                timestamp=datetime.now().isoformat(),
                date_range={
                    "start": start_date.strftime('%Y-%m-%d'),
                    "end": end_date.strftime('%Y-%m-%d')
                },
                total_records=0,
                statistics={},
                data=[],
                rate_type=rate_type,
                message="No historical data found"
            )
        
        # Calculate statistics
        statistics = data_fetcher.get_history_statistics(history)
        
        response_data = USDHistoryResponse(
            success=True,
            timestamp=datetime.now().isoformat(),
            date_range={
                "start": start_date.strftime('%Y-%m-%d'),
                "end": end_date.strftime('%Y-%m-%d')
            },
            total_records=len(history),
            statistics=statistics,
            data=history,
            rate_type=rate_type
        )
        
        # Return CSV if requested
        if format.lower() == "csv":
            import io
            import csv
            from fastapi.responses import Response
            
            output = io.StringIO()
            fieldnames = ["date"]
            if rate_type in ["parallel", "both"]:
                fieldnames.append("usd_dzd_parallel")
            if rate_type in ["official", "both"]:
                fieldnames.append("usd_dzd_official")
            
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            
            for record in history:
                row = {k: record.get(k) for k in fieldnames if k in record}
                writer.writerow(row)
            
            return Response(
                content=output.getvalue(),
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename=usd_dzd_{days}d_{rate_type}_{end_date.strftime('%Y-%m-%d')}.csv"
                }
            )
        
        return response_data

    except Exception as e:
        logger.error(f"Error fetching USD/DZD history: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/usd/features")
async def get_usd_features(
    date: str = Query(None, description="Date in YYYY-MM-DD format (default: today)"),
    data_fetcher: USDDataFetcher = Depends(get_usd_data_fetcher)
):
    """Get USD/DZD features for a specific date"""
    try:
        if not date:
            date = datetime.now().strftime('%Y-%m-%d')
        
        # Validate date format
        try:
            target_date = datetime.strptime(date, '%Y-%m-%d')
        except ValueError:
            raise HTTPException(
                status_code=400, 
                detail="Invalid date format. Use YYYY-MM-DD"
            )
        
        logger.info(f"📋 Fetching features for {date}")
        
        features = await data_fetcher.fetch_or_calculate_features(target_date)
        
        return {
            "success": True,
            "date": date,
            "features": features,
            "features_count": len(features),
            "model_features": {
                "eur_usd": features.get('eur_usd'),
                "brent_oil": features.get('brent_oil'),
                "dxy": features.get('dxy'),
                "lag1": features.get('lag1'),
                "lag7": features.get('lag7'),
                "lag30": features.get('lag30'),
                "usd_dzd_official": features.get('usd_dzd_official')
            },
            "current_rates": {
                "usd_dzd_parallel": features.get('usd_dzd_parallel'),
                "usd_dzd_official": features.get('usd_dzd_official')
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching features: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/usd/model/info")
async def get_usd_model_info(
    forecaster: USDForecaster = Depends(get_usd_forecaster)
):
    """Get detailed information about the USD/DZD model"""
    try:
        if not forecaster.is_loaded():
            raise HTTPException(
                status_code=503, 
                detail="USD/DZD model not loaded"
            )
        
        info = forecaster.get_model_info()
        
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "model_info": info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/usd/test")
async def usd_test_endpoint(
    forecaster: USDForecaster = Depends(get_usd_forecaster)
):
    """Test USD/DZD endpoint"""
    try:
        components_status = {
            "forecaster_initialized": forecaster is not None,
            "model_loaded": forecaster.model is not None if forecaster else False,
            "scaler_X_loaded": forecaster.scaler_X is not None if forecaster else False,
            "scaler_y_loaded": forecaster.scaler_y is not None if forecaster else False,
            "data_fetcher_initialized": forecaster.data_fetcher is not None if forecaster else False,
            "all_loaded": forecaster.is_loaded() if forecaster else False
        }
        
        # Try a simple forecast
        test_forecast = None
        test_error = None
        
        if forecaster and forecaster.is_loaded():
            try:
                today = datetime.now().date().isoformat()
                test_forecast = await forecaster.forecast(today)
            except Exception as e:
                test_error = str(e)
        
        return {
            "status": "operational" if components_status["all_loaded"] else "incomplete",
            "timestamp": datetime.now().isoformat(),
            "components": components_status,
            "model_info": forecaster.get_model_info() if forecaster and forecaster.is_loaded() else None,
            "test_forecast": {
                "attempted": test_forecast is not None or test_error is not None,
                "successful": test_forecast is not None,
                "result": test_forecast,
                "error": test_error
            }
        }
        
    except Exception as e:
        logger.error(f"Test endpoint error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }



# ==================== RUN APPLICATION ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)