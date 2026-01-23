# services/parallel_rates.py
import requests
from bs4 import BeautifulSoup

EURODZ_URL = "https://eurodz.com/"

def get_parallel_rates():
    try:
        response = requests.get(EURODZ_URL, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        rates = {
            "eur_dzd_parallel": None,
            "usd_dzd_parallel": None
        }

        table_rows = soup.select("table tr")

        for row in table_rows:
            cols = row.find_all("td")
            if len(cols) >= 3:
                currency_name = cols[0].get_text(strip=True)
                achat = cols[1].get_text(strip=True).split()[0]  # buy

                if "Euro" in currency_name:
                    rates["eur_dzd_parallel"] = float(achat)

                elif "Dollar US" in currency_name:
                    rates["usd_dzd_parallel"] = float(achat)

        return rates

    except Exception as e:
        print("Error scraping eurodz:", e)
        return {
            "eur_dzd_parallel": None,
            "usd_dzd_parallel": None
        }
