from core.db import supabase
from datetime import datetime, timedelta

def get_historical_euro():
    try:
        one_year_ago = datetime.now() - timedelta(days=365)
        one_year_ago_iso = one_year_ago.strftime("%Y-%m-%dT00:00:00Z")  

        response = (
            supabase.table("eur_dzd_dataset")
            .select("date, eur_dzd_parallel")
            .gte("date", one_year_ago_iso)
            .order("date", desc=False)
            .execute()
        )

        print("DEBUG response:", response.data)  

        if response.data:
            historical_data = [
                {"date": record["date"], "eur_dzd_parallel": record["eur_dzd_parallel"]}
                for record in response.data
            ]
            return historical_data
        else:
            return []

    except Exception as e:
        print("Erreur récupération historique EUR:", e)
        return []
