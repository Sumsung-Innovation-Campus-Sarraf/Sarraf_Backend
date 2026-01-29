# services/usd_logic.py
from datetime import datetime

async def usd_forecast_logic(forecaster, target_date_str, use_cached=False):
    target_date = datetime.strptime(target_date_str, "%Y-%m-%d")
    if target_date.date() > datetime.now().date():
        raise ValueError("Cannot forecast future dates")
    if not forecaster.is_loaded():
        raise ValueError("USD/DZD model not loaded")
    result = await forecaster.forecast(target_date, use_cached)
    return result

def usd_history_logic(data_fetcher, rate_type="both"):
    response = data_fetcher.supabase.table("usd_dzd_dataset") \
        .select("date") \
        .order("date", desc=False) \
        .limit(1) \
        .execute()
    
    start_date = datetime.strptime(response.data[0]["date"], "%Y-%m-%d") \
        if response.data else datetime(2000,1,1)
    end_date = datetime.now()
    
    history = data_fetcher.get_usd_history(start_date, end_date, rate_type)
    statistics = data_fetcher.get_history_statistics(history)
    return {"history": history, "statistics": statistics}
