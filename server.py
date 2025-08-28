from contextlib import asynccontextmanager
import logging
import os
import warnings
from fastapi import FastAPI, staticfiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from utils.socket import ConnectionManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
manager = ConnectionManager(logger)

warnings.filterwarnings("ignore")


def start_models():
    return True

@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL_LOADED
    MODEL_LOADED = start_models()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.exists("static"):
    app.mount("/static", staticfiles.StaticFiles(directory="static"), name="static")

