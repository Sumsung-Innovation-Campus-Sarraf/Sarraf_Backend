from datetime import datetime
from core.db import supabase
from services.parallel import get_parallel_rates
from services.official import get_rate

LAGS = [1, 2, 3, 7, 14, 30]

def insert_daily_rates():
    today_str = datetime.today().strftime("%Y-%m-%d")

    # Vérifier si les données du jour existent déjà
    existing = supabase.table("eur_dzd_dataset").select("date").eq("date", today_str).execute()
    if existing.data:
        print(f"Données du {today_str} déjà présentes.")
        return existing.data

    # Récupérer taux officiel et parallèle
    eur_dzd_official = get_rate("EUR", "DZD")
    parallel_rates = get_parallel_rates()
    eur_dzd_parallel = parallel_rates.get("eur_dzd_parallel")

    # Moyennes ou valeurs par défaut
    eur_usd = 1.17
    brent_oil = 65.23
    dxy = 98.76

    # Récupérer les dernières valeurs de la table pour calculer les lags
    last_rows = supabase.table("eur_dzd_dataset").select("date,eur_dzd_parallel").order("date", desc=True).limit(max(LAGS)).execute()
    last_data = last_rows.data if last_rows.data else []

    # Calcul des lags
    lag_values = {}
    for lag in LAGS:
        if len(last_data) >= lag:
            lag_values[f"eur_dzd_parallel_lag{lag}"] = last_data[lag-1]["eur_dzd_parallel"]
        else:
            lag_values[f"eur_dzd_parallel_lag{lag}"] = None  # Pas assez de données

    # Préparer le dictionnaire d'insertion
    data = {
        "date": today_str,
        "eur_dzd_official": eur_dzd_official,
        "eur_dzd_parallel": eur_dzd_parallel,
        "eur_usd": eur_usd,
        "brent_oil": brent_oil,
        "dxy": dxy,
        **lag_values
    }

    # Insertion
    response = supabase.table("eur_dzd_dataset").insert(data).execute()
    if response.status_code in [200, 201]:
        print(f"Données du {today_str} insérées avec succès.")
    else:
        print("Erreur insertion :", response.data)

    return data
