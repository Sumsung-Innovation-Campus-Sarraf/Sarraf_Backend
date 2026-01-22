from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from routes import routes
from core.middlewares import setup_middlewares

app = FastAPI()
app.include_router(routes.router)


setup_middlewares(app)

