import os

EXCHANGE_API_KEY = os.getenv("EXCHANGE_API_KEY")
EXCHANGE_API_BASE_URL = "https://v6.exchangerate-api.com/v6"

if not EXCHANGE_API_KEY:
    raise RuntimeError("EXCHANGE_API_KEY n'est pas défini !")
