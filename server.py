from contextlib import asynccontextmanager
from fastapi import FastAPI


def start_models():
    return True

@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL_LOADED
    MODEL_LOADED = start_models()

app = FastAPI(lifespan=lifespan)


