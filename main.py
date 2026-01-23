"""
FastAPI Backend for Gold and USD/DZD Forecasting
"""
from datetime import date
from fastapi import FastAPI, HTTPException, Depends, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from supabase import create_client, Client
import io
import csv

# Import gold services
from services.features import GoldFeatureCalculator, get_feature_calculator
from services.model_loader import get_model_loader

# Import USD services
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

app = FastAPI()
app.include_router(routes.router)
setup_middlewares(app)

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
# Initialize FastAPI app
app = FastAPI(
    title="Gold and USD/DZD Forecast API",
    description="API for forecasting gold price direction and USD/DZD exchange rates",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global components
GOLD_MODEL_LOADER = None
GOLD_FEATURE_CALCULATOR = None
USD_DATA_FETCHER = None
USD_FORECASTER = None

# ==================== PYDANTIC MODELS ====================

# Gold Models
class GoldPredictionRequest(BaseModel):
    model_type: str = "direction"  # "direction" or "volatility"
    horizon: str = "monthly"  # "monthly" or "weekly"
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

# USD/DZD Models
class USDForecastRequest(BaseModel):
    use_cached: bool = True
    model_type: str = "level"  # "level" or "change"

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

# ==================== DEPENDENCIES ====================

# Supabase Client
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

# USD/DZD Dependencies
def get_usd_data_fetcher() -> USDDataFetcher:
    """Get USD data fetcher"""
    global USD_DATA_FETCHER
    
    if USD_DATA_FETCHER is None:
        supabase = get_supabase_client()
        USD_DATA_FETCHER = USDDataFetcher(supabase)
    
    return USD_DATA_FETCHER

def get_usd_forecaster() -> USDForecaster:
    """Get USD forecaster"""
    global USD_FORECASTER
    
    if USD_FORECASTER is None:
        data_fetcher = get_usd_data_fetcher()
        USD_FORECASTER = USDForecaster(data_fetcher)
        USD_FORECASTER.load_models()
    
    return USD_FORECASTER

# ==================== STARTUP EVENT ====================

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    print("🚀 Starting Gold and USD/DZD Forecast API...")
    
    # Initialize gold components
    try:
        model_loader = get_gold_model_loader()
        feature_calculator = get_gold_feature_calculator()
        print("✅ Gold components initialized successfully")
    except Exception as e:
        print(f"⚠️  Gold components initialization warning: {e}")
    
    # Initialize USD/DZD components
    try:
        forecaster = get_usd_forecaster()
        if forecaster.is_loaded():
            print("✅ USD/DZD models loaded successfully")
            
            # Print model info
            info = forecaster.get_model_info()
            print(f"📊 USD/DZD Model Info:")
            print(f"  - Model type: {info.get('model_type')}")
            print(f"  - Feature count: {info.get('feature_count')}")
            print(f"  - Training date: {info.get('training_date')}")
            
            if info.get('performance'):
                perf = info['performance']
                print(f"  - Performance: R²={perf.get('r2', 0):.4f}, RMSE={perf.get('rmse', 0):.4f}")
        else:
            print("⚠️  USD/DZD models failed to load")
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
#=================== Silver ENDPOINTS ====================
# ==================== SILVER ENDPOINTS ====================

# Silver Dependencies
SILVER_FEATURE_CALCULATOR = None

def get_silver_feature_calculator_dep():
    global SILVER_FEATURE_CALCULATOR
    if SILVER_FEATURE_CALCULATOR is None:
        SILVER_FEATURE_CALCULATOR = get_silver_feature_calculator()
    return SILVER_FEATURE_CALCULATOR


@app.get("/silver/features/latest")
async def get_latest_silver_features():
    """Get latest calculated silver features"""
    try:
        feature_calculator = get_silver_feature_calculator_dep()
        if not feature_calculator:
            raise HTTPException(status_code=503, detail="Silver feature calculator not initialized")
        
        features = feature_calculator.get_features_for_prediction()
        if not features:
            raise HTTPException(status_code=404, detail="Could not calculate features")
        
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "features_count": len(features),
            "all_features": features,
            "silver_price": features.get('silver_price_usd')
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/silver-price-history")
async def get_silver_price_history(
    range: str = Query("1y", description="Time range: '1m', '2m', '6m', '1y', '2y', '5y', 'all'"),
    format: str = Query("json", description="Output format: 'json' or 'csv'")
):
    """Get historical silver prices from Supabase"""
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
            start_date = datetime(2000, 1, 1)
        else:
            return {"success": False, "error": f"Invalid range: {range}"}

        # Fetch data from Supabase
        feature_calculator = get_silver_feature_calculator_dep()
        start_date_str = start_date.date().isoformat()
        end_date_str = end_date.date().isoformat()
        
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
        
        history = [{"date": r["date"], "price": float(r["silver_price_usd"])} for r in response.data]
        prices = [h["price"] for h in history if h["price"] > 0]

        stats = {}
        if prices:
            stats = {
                "current_price": prices[-1],
                "min_price": min(prices),
                "max_price": max(prices),
                "avg_price": sum(prices)/len(prices),
                "price_change": prices[-1]-prices[0],
                "percent_change": ((prices[-1]-prices[0])/prices[0]*100) if prices[0]>0 else 0
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
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=["date", "price"])
            writer.writeheader()
            writer.writerows(history)
            
            return Response(
                content=output.getvalue(),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=silver_prices_{range}_{end_date_str}.csv"}
            )
        
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


from services.features import SilverFeatureCalculator

SILVER_FEATURE_CALCULATOR = None

def get_silver_feature_calculator():
    global SILVER_FEATURE_CALCULATOR
    if SILVER_FEATURE_CALCULATOR is None:
        SILVER_FEATURE_CALCULATOR = SilverFeatureCalculator(get_supabase_client())
    return SILVER_FEATURE_CALCULATOR


@app.get("/features/latest")
async def get_latest_silver_features():
    """Get the latest calculated silver features"""
    try:
        feature_calculator = get_silver_feature_calculator()
        features = feature_calculator.get_features_for_prediction()
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "features": features
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_silver")
async def predict_silver():
    feature_calculator = get_silver_feature_calculator()
    model_loader = get_silver_model_loader()

    features = feature_calculator.get_features_for_prediction()
    model_features = {
        k: features[k] for k in model_loader.feature_cols
    }

    prediction = model_loader.predict(model_features)

    return {
        "success": True,
        "timestamp": datetime.now().isoformat(),
        "features": model_features,
        "prediction": prediction
    }

@app.get("/usd/health", response_model=USDHealthResponse)
async def usd_health_check(
    forecaster: USDForecaster = Depends(get_usd_forecaster),
    data_fetcher: USDDataFetcher = Depends(get_usd_data_fetcher)
):
    """Check USD/DZD API health status"""
    try:
        supabase = get_supabase_client()
        supabase_connected = supabase is not None
        
        model_info = forecaster.get_model_info() if forecaster.is_loaded() else {}
        
        return USDHealthResponse(
            status="healthy" if forecaster.is_loaded() else "partial",
            timestamp=datetime.now().isoformat(),
            model_loaded=forecaster.model is not None,
            scaler_loaded=forecaster.scaler is not None,
            features_loaded=forecaster.feature_cols is not None,
            supabase_connected=supabase_connected,
            feature_count=len(forecaster.feature_cols) if forecaster.feature_cols else 0,
            model_info=model_info
        )
    except Exception as e:
        return USDHealthResponse(
            status=f"error: {str(e)[:100]}",
            timestamp=datetime.now().isoformat(),
            model_loaded=False,
            scaler_loaded=False,
            features_loaded=False,
            supabase_connected=False,
            feature_count=0
        )

@app.post("/usd/forecast", response_model=USDForecastResponse)
async def make_usd_forecast(
    request: USDForecastRequest,
    forecaster: USDForecaster = Depends(get_usd_forecaster)
):
    """Make a USD/DZD forecast"""
    try:
        # Validate date
        target_date = datetime.strptime(request.date, '%Y-%m-%d')
        
        if target_date.date() > datetime.now().date():
            raise HTTPException(status_code=400, detail="Cannot forecast future dates without historical data")
        
        # Check if components are loaded
        if not forecaster.is_loaded():
            raise HTTPException(status_code=503, detail="USD/DZD model components not loaded")
        
        # Make forecast
        forecast_result = await forecaster.forecast(target_date, request.use_cached)
        
        return USDForecastResponse(
            success=True,
            timestamp=datetime.now().isoformat(),
            date=request.date,
            forecast=forecast_result["forecast"],
            current_rate=forecast_result["current_rate"],
            expected_change=forecast_result["expected_change"],
            change_probability=None,  # Could add change prediction model later
            confidence_interval=forecast_result["confidence_interval"],
            features_used=forecaster.feature_cols,
            data_source=forecast_result["data_source"],
            message="USD/DZD forecast successful"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"USD/DZD forecast failed: {str(e)}")

@app.get("/usd/history")
async def get_usd_history(
    range: str = Query("1y", description="Time range: '1m', '2m', '6m', '1y', '2y', '5y', 'all'"),
    rate_type: str = Query("both", description="Rate type: 'parallel', 'official', 'both'"),
    format: str = Query("json", description="Output format: 'json' or 'csv'"),
    data_fetcher: USDDataFetcher = Depends(get_usd_data_fetcher)
):
    """
    Get historical USD/DZD data
    
    Available ranges:
    - '1m': Last 1 month
    - '2m': Last 2 months  
    - '6m': Last 6 months
    - '1y': Last 1 year (default)
    - '2y': Last 2 years
    - '5y': Last 5 years
    - 'all': All available data
    
    Rate types:
    - 'parallel': Only parallel rate
    - 'official': Only official rate  
    - 'both': Both rates (default)
    
    Returns: List of {date, usd_dzd_parallel, usd_dzd_official} objects
    """
    try:
        # Calculate date range
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
            # For "all", get earliest date from database
            try:
                response = data_fetcher.supabase.table("usd_dzd_dataset") \
                    .select("date") \
                    .order("date", desc=False) \
                    .limit(1) \
                    .execute()
                
                if response.data and response.data[0].get("date"):
                    start_date = datetime.strptime(response.data[0]["date"], '%Y-%m-%d')
                else:
                    start_date = datetime(2000, 1, 1)  # Default far back date
            except:
                start_date = datetime(2000, 1, 1)
        else:
            return {
                "success": False,
                "error": f"Invalid range: {range}. Use: 1m, 2m, 6m, 1y, 2y, 5y, all"
            }
        
        # Get historical data
        history = data_fetcher.get_usd_history(start_date, end_date, rate_type)
        
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
            output = io.StringIO()
            fieldnames = ["date"]
            if rate_type in ["parallel", "both"]:
                fieldnames.append("usd_dzd_parallel")
            if rate_type in ["official", "both"]:
                fieldnames.append("usd_dzd_official")
            
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            
            for record in history:
                writer.writerow(record)
            
            return Response(
                content=output.getvalue(),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=usd_dzd_{range}_{rate_type}_{end_date.strftime('%Y-%m-%d')}.csv"}
            )
        
        return response_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/usd/features")
async def get_usd_features(
    date: str = None,
    data_fetcher: USDDataFetcher = Depends(get_usd_data_fetcher)
):
    """Get USD/DZD features for a specific date"""
    try:
        if not date:
            date = datetime.now().strftime('%Y-%m-%d')
        
        target_date = datetime.strptime(date, '%Y-%m-%d')
        
        features = await data_fetcher.fetch_or_calculate_features(target_date)
        
        return {
            "success": True,
            "date": date,
            "features": features,
            "features_count": len(features),
            "usd_dzd_parallel": features.get('usd_dzd_parallel'),
            "usd_dzd_official": features.get('usd_dzd_official'),
            "eur_usd": features.get('eur_usd'),
            "brent_oil": features.get('brent_oil'),
            "dxy": features.get('dxy')
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/usd/model/info")
async def get_usd_model_info(
    forecaster: USDForecaster = Depends(get_usd_forecaster)
):
    """Get detailed information about the USD/DZD model"""
    try:
        if not forecaster.is_loaded():
            raise HTTPException(status_code=503, detail="USD/DZD model not loaded")
        
        info = forecaster.get_model_info()
        
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "model_info": info
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/usd/test")
async def usd_test_endpoint(
    forecaster: USDForecaster = Depends(get_usd_forecaster)
):
    """Test USD/DZD endpoint"""
    try:
        if not forecaster or not forecaster.is_loaded():
            return {
                "status": "error",
                "error": "USD/DZD forecaster not initialized or loaded",
                "timestamp": datetime.now().isoformat()
            }
        
        # Make a simple test forecast for yesterday
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        test_request = USDForecastRequest(date=yesterday, use_cached=True, model_type="level")
        
        try:
            forecast = await make_usd_forecast(test_request, forecaster)
        except:
            forecast = None
        
        return {
            "status": "operational",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "forecaster": "ready" if forecaster else "not ready",
                "model_loaded": forecaster.is_loaded() if forecaster else False,
                "feature_count": len(forecaster.feature_cols) if forecaster else 0
            },
            "test": {
                "test_date": yesterday,
                "forecast_attempted": forecast is not None
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }



# ==================== RUN APPLICATION ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)