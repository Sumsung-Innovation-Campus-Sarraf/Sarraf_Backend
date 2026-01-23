from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from routes import routes
from core.middlewares import setup_middlewares
from apscheduler.schedulers.background import BackgroundScheduler
from services.daily_updates import insert_daily_rates
import logging
import atexit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.include_router(routes.router)
setup_middlewares(app)

def daily_job():
    try:
        data = insert_daily_rates()
        logger.info(f"Données journalières insérées : {data}")
    except Exception as e:
        logger.error(f"Erreur lors de l'insertion quotidienne : {e}")

scheduler = BackgroundScheduler()
scheduler.add_job(daily_job, 'cron', hour=9, minute=0)
scheduler.start()

atexit.register(lambda: scheduler.shutdown())
