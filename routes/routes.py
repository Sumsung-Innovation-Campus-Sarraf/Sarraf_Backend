# routers/rates.py
from fastapi import APIRouter
from services.official import get_rate
from services.parallel import get_parallel_rates
from services.financial import get_today_financial_data

router = APIRouter(prefix="", tags=["Rates"])

@router.get("/official")
def get_rates():
    return {
        "EUR_to_DZD": get_rate("EUR"),
        "USD_to_DZD": get_rate("USD")
    }


@router.get("/parallel")
def parallel():
    return get_parallel_rates()


@router.get("/financial")
def today_rates():
    return get_today_financial_data()