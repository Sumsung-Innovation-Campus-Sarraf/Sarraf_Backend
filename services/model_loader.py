import joblib
from pathlib import Path

MODEL_DIR = Path("models/euro")

ridge_model = joblib.load(MODEL_DIR / "euro_model.pkl")
scaler_X = joblib.load(MODEL_DIR / "scaler_X.pkl")
scaler_y = joblib.load(MODEL_DIR / "scaler_y.pkl")
