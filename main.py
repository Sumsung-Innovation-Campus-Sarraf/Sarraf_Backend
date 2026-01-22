from dotenv import load_dotenv
load_dotenv()  

from fastapi import FastAPI
from routes import routes 

app = FastAPI()
app.include_router(routes.router)
