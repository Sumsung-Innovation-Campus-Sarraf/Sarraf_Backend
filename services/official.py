import requests
from core.config import EXCHANGE_API_KEY, EXCHANGE_API_BASE_URL

def get_rate(base: str, target: str = "DZD"):
    url = f"{EXCHANGE_API_BASE_URL}/{EXCHANGE_API_KEY}/latest/{base}"
    print("URL appelée :", url)  # <- Debug

    try:
        response = requests.get(url, timeout=5)
        print("Status code :", response.status_code)  # <- Debug
        data = response.json()
        print("Réponse API :", data)  # <- Debug

        if data.get("result") == "success":
            rate = data["conversion_rates"].get(target)
            print(f"Taux {base} -> {target} :", rate)  # <- Debug
            return rate
        else:
            print("Erreur API :", data)  # <- Debug
    except requests.RequestException as e:
        print("Exception HTTP :", e)

    return None