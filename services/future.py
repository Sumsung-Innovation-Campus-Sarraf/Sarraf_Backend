import joblib
import pandas as pd
from datetime import datetime, timedelta
from core.db import supabase

MODEL_PATH = "models/euro/euro_model.pkl"
SCALER_X_PATH = "models/euro/scaler_X.pkl"
SCALER_Y_PATH = "models/euro/scaler_y.pkl"

# Features utilisées par le modèle (seulement lag1, lag7, lag30)
FEATURE_COLS = [
    'eur_usd', 'brent_oil', 'dxy',
    'eur_dzd_official',
    'eur_dzd_parallel_lag1', 'eur_dzd_parallel_lag7', 'eur_dzd_parallel_lag30'
]

HORIZON = 30  # Nombre de jours à prédire

# Charger modèle et scalers
ridge = joblib.load(MODEL_PATH)
scaler_X = joblib.load(SCALER_X_PATH)
scaler_y = joblib.load(SCALER_Y_PATH)

def forecast_next_days():
    """
    Forecast du EUR/DZD Parallel pour les 30 prochains jours
    en utilisant les colonnes de lags existantes (lag1, lag7, lag30).
    """
    # Récupérer les dernières données
    last_rows = supabase.table("eur_dzd_dataset").select("*").order("date", desc=True).limit(31).execute()
    df = pd.DataFrame(last_rows.data)
    df = df.sort_values("date").reset_index(drop=True)

    if df.empty:
        return {"error": "Pas assez de données pour le forecasting."}

    # Commencer avec la dernière ligne
    last_row = df.iloc[-1].copy()
    forecasts = []

    for i in range(HORIZON):
        # Préparer les features pour le modèle
        X = pd.DataFrame([last_row[FEATURE_COLS]])
        X_scaled = scaler_X.transform(X)
        y_scaled = ridge.predict(X_scaled)
        y_pred = scaler_y.inverse_transform(y_scaled.reshape(-1,1))[0][0]

        # Ajouter la prédiction à la liste
        next_date = datetime.strptime(last_row['date'], "%Y-%m-%d") + timedelta(days=1)
        forecasts.append({
            "date": next_date.strftime("%Y-%m-%d"),
            "eur_dzd_parallel_forecast": round(y_pred, 4)
        })

        # Mettre à jour les lags pour le prochain tour
        last_row['eur_dzd_parallel_lag30'] = last_row['eur_dzd_parallel_lag7']
        last_row['eur_dzd_parallel_lag7'] = last_row['eur_dzd_parallel_lag1']
        last_row['eur_dzd_parallel_lag1'] = y_pred

        # Optionnel : mettre à jour eur_dzd_parallel pour référence
        last_row['eur_dzd_parallel'] = y_pred
        last_row['date'] = next_date.strftime("%Y-%m-%d")

    return forecasts
