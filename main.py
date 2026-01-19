from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Charger le modèle et les scalers
ridge = joblib.load('euro_model.pkl')
scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')

# Remplacer par tes vraies colonnes
feature_cols = ["feature1", "feature2", "feature3"]

app = FastAPI()

class PredictRequest(BaseModel):
    data: list  # valeurs envoyées par le frontend

@app.post("/predict")
def predict(request: PredictRequest):
    X_new = np.array(request.data).reshape(1, -1)
    X_new_scaled = scaler_X.transform(X_new)
    y_pred_scaled = ridge.predict(X_new_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1,1)).ravel()[0]
    return {"prediction": y_pred}
