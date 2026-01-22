# services/parallel_rates.py
import requests
from bs4 import BeautifulSoup

EURODZ_URL = "https://eurodz.com/"

def get_parallel_rates():
    try:
        response = requests.get(EURODZ_URL, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        rates = {}
        # Find the table rows
        table_rows = soup.select("table tr")
        for row in table_rows:
            cols = row.find_all("td")
            if len(cols) >= 4:
                currency_name = cols[0].get_text(strip=True)
                achat = cols[1].get_text(strip=True).split()[0]  # buy
                vente = cols[2].get_text(strip=True).split()[0]  # sell

                if "Euro" in currency_name:
                    rates["EUR"] = {"buy": float(achat), "sell": float(vente)}
                elif "Dollar US" in currency_name:
                    rates["USD"] = {"buy": float(achat), "sell": float(vente)}

        return rates
    except Exception as e:
        print("Error scraping eurodz:", e)
        return {"EUR": None, "USD": None}
