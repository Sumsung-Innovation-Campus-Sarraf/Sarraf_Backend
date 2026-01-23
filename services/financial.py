import yfinance as yf
from datetime import datetime, timedelta
import ssl
import certifi
import os

ssl._create_default_https_context = ssl._create_default_https_context
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
os.environ['SSL_CERT_FILE'] = certifi.where()

def get_today_financial_data():
    start_date = (datetime.today() - timedelta(days=2)).strftime('%Y-%m-%d')
    end_date = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')

    results = {}
    try:
        data = yf.download('EURUSD=X', start=start_date, end=end_date,
                           interval='1d', progress=False, auto_adjust=True)
        if not data.empty:
            results['eur_usd'] = float(data['Close'].values[-1])
        else:
            results['eur_usd'] = None
    except Exception as e:
        print("EUR/USD error:", e)
        results['eur_usd'] = None

    try:
        data = yf.download('BZ=F', start=start_date, end=end_date,
                           interval='1d', progress=False, auto_adjust=True)
        if not data.empty:
            results['brent_oil'] = float(data['Close'].values[-1])
        else:
            results['brent_oil'] = None
    except Exception as e:
        print("Brent Oil error:", e)
        results['brent_oil'] = None

    try:
        data = yf.download('DX-Y.NYB', start=start_date, end=end_date,
                           interval='1d', progress=False, auto_adjust=True)
        if not data.empty:
            results['dxy'] = float(data['Close'].values[-1])
        else:
            results['dxy'] = None
    except Exception as e:
        print("DXY error:", e)
        results['dxy'] = None

    return results

