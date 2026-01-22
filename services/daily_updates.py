from datetime import date
from services.parallel import get_parallel_rates
from core.db import supabase  

def update_daily_parallel_rates():
    try:
        rates = get_parallel_rates()

        if not rates.get("eur_dzd_parallel") or not rates.get("usd_dzd_parallel"):
            return {
                "status": "Erreur",
                "message": "Taux parallèles non disponibles"
            }

        today = date.today().isoformat()

        data = {
            "date": today,
            "eur_dzd_parallel": rates["eur_dzd_parallel"],
            "usd_dzd_parallel": rates["usd_dzd_parallel"]
        }

        response = (
            supabase
            .table("dzd_rates")
            .upsert(data, on_conflict="date")
            .execute()
        )

        return {
            "status": "Mise à jour quotidienne réussie",
            "data": response.data
        }

    except Exception as e:
        return {
            "status": "Erreur Supabase",
            "message": str(e)
        }
