# routers/rates.py
from fastapi import APIRouter
from services.official import get_rate
from services.parallel import get_parallel_rates
from services.financial import get_today_financial_data
from services.daily_updates import update_daily_parallel_rates   
from services.historical import get_historical_euro

router = APIRouter(prefix="", tags=["Rates"])

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

@router.get("/update")
def daily_parallel_update():
    return update_daily_parallel_rates()