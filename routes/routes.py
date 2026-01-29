# routers/rates.py
from fastapi import APIRouter
from services.official import get_rate
from services.parallel import get_parallel_rates
from services.financial import get_today_financial_data
from fastapi import APIRouter, Query
from core.db import supabase
import pandas as pd
from services.daily_updates import insert_daily_rates   
from services.historical import get_historical_euro
from fastapi import APIRouter
from services.future import forecast_next_days
from fastapi import APIRouter, Query
from datetime import datetime

from services.usd import usd_forecast_logic, usd_history_logic

router = APIRouter(prefix="", tags=["Rates"])


@router.get("/")
def get_forecast(currency: str = Query("EUR", description="Devise à convertir: EUR ou USD")):

    today_str = datetime.today().strftime("%Y-%m-%d")
    parallel_rates = get_parallel_rates()
    if currency.upper() == "EUR":
        today_rate = parallel_rates.get("eur_dzd_parallel")
    elif currency.upper() == "USD":
        today_rate = parallel_rates.get("usd_dzd_parallel")
    else:
        return {"status": "error", "message": "Currency must be EUR or USD."}

    history = get_historical_euro()  
  

    forecast = forecast_next_days()  # Déjà gère EUR/DZD Parallel

    # forecaster = get_usd_forecaster()
    # data_fetcher = get_usd_data_fetcher()
    
    # history = usd_history_logic(data_fetcher, rate_type="both")
    # forecast = usd_forecast_logic(forecaster, datetime.today().strftime("%Y-%m-%d"))

    return {
        "status": "success",
        "date": today_str,
        "currency": currency.upper(),
        "today_parallel": today_rate,
        "history": history,       # liste des dict {date, eur_dzd_parallel, ...}
        "forecast_30_days": forecast  # liste des dict {date, eur_dzd_parallel_forecast}
    }



@router.get("/forecast")
def get_eur_dzd_forecast():
  
    try:
        forecasts = forecast_next_days()
        if isinstance(forecasts, dict) and "error" in forecasts:
            return {"status": "error", "message": forecasts["error"]}
        
        return {
            "status": "success",
            "horizon_days": len(forecasts),
            "forecasts": forecasts
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}



@router.get("/update")
def update_daily_rates():
    """
    Manually trigger the insertion of daily EUR/DZD rates into the database.
    """
    try:
        data = insert_daily_rates()
        return {
            "status": "success",
            "message": "Daily rates inserted successfully.",
            "data": data
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
    

@router.get("/official")
def get_rates():
    return {
        "EUR_to_DZD": get_rate("EUR"),
        "USD_to_DZD": get_rate("USD")
    }

@router.get("/historical_euro")
def historical_euro():
    return get_historical_euro()


@router.get("/parallel")
def parallel():
    return get_parallel_rates()


@router.get("/financial")
def today_rates():
    return get_today_financial_data()


@router.get("/eur_dzd")
def forecast_eur_dzd():
  
    try:
        response = supabase.table("eur_dzd_dataset").select("*").execute()
        data = response.data

        if not data:
            return {"status": "empty", "message": "Aucune donnée historique."}

        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        required_cols = ['eur_usd', 'brent_oil', 'dxy', 'eur_dzd_official', 'eur_dzd_parallel']
        for col in required_cols:
            if col not in df.columns:
                return {"status": "error", "message": f"Colonne manquante : {col}"}

        df_future = forecast_next_30_days(df)

        return {
            "status": "success",
            "forecast": df_future.to_dict(orient="records")
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}