import asyncio
import base64
import json
import logging
import os
import warnings
from contextlib import asynccontextmanager
from datetime import datetime

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, staticfiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from utils.socket import ConnectionManager
from utils.speech_client import (
    analyze_text_intent,
    check_model_health,
    merge_transcription_texts,
    transcribe_audio_bytes,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
manager = ConnectionManager(logger)

warnings.filterwarnings("ignore")

# Global variables for client state management
client_languages = {}
NLU_PROCESSOR = None
MODEL_LOADED = False


async def transcribe_segment(audio_np: np.ndarray, language: str) -> str:
    """Transcribe audio segment using gRPC service."""
    try:
        # Convert numpy array to bytes
        audio_bytes = (audio_np * 32768.0).astype(np.int16).tobytes()
        transcription = await transcribe_audio_bytes(audio_bytes, language)
        return transcription or ""
    except Exception as e:
        logger.exception(f"Error transcribing segment: {e}")
        return ""


async def merge_transcriptions(transcriptions: list) -> str:
    """Merge transcriptions using gRPC service."""
    try:
        merged = await merge_transcription_texts(transcriptions)
        return merged or ""
    except Exception as e:
        logger.exception(f"Error merging transcriptions: {e}")
        return ""


async def process_audio_for_client(client_id: str, language: str):
    """Process audio chunks for a client using gRPC service."""
    try:
        while manager.is_recording.get(client_id, False):
            if manager.audio_buffers[client_id]:
                # Get audio chunks
                audio_chunks = manager.audio_buffers[client_id].copy()
                manager.audio_buffers[client_id] = []

                if audio_chunks:
                    # Concatenate audio chunks
                    audio_np = np.concatenate(audio_chunks)

                    # Transcribe the audio
                    transcription = await transcribe_segment(audio_np, language)

                    if transcription:
                        # Store transcription
                        if client_id not in manager.transcribed_segments:
                            manager.transcribed_segments[client_id] = []
                        manager.transcribed_segments[client_id].append(transcription)

                        # Send interim transcription
                        await manager.send_personal_message(
                            json.dumps(
                                {
                                    "type": "transcription",
                                    "text": transcription,
                                    "is_final": False,
                                    "intent": None,
                                }
                            ),
                            client_id,
                        )

            # Wait before processing next batch
            await asyncio.sleep(0.1)

    except Exception as e:
        logger.exception(f"Error processing audio for client {client_id}: {e}")


def start_models():
    """Initialize model service connection."""
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global MODEL_LOADED
    MODEL_LOADED = start_models()

    # Test connection to model service
    try:
        health = await check_model_health()
        if health and health.get("healthy"):
            logger.info("✅ Model service is healthy")
        else:
            logger.warning("⚠️ Model service health check failed")
    except Exception as e:
        logger.error(f"Failed to connect to model service: {e}")

    yield


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


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    interim_task: asyncio.Task = None

    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)

            if data["type"] == "ping":
                await manager.send_personal_message(
                    json.dumps(
                        {"type": "pong", "timestamp": datetime.now().isoformat()}
                    ),
                    client_id,
                )
                continue

            if data["type"] == "start_recording":
                lang = data.get("language", "en")
                supported = ["en", "ms", "my", "zh", "id", "ne", "ta", "bn"]
                if lang not in supported:
                    logger.warning(
                        f"Unsupported language {lang} from {client_id}, defaulting to en"
                    )
                    lang = "en"

                client_languages[client_id] = lang
                logger.info(f"[{client_id}] Started recording; language={lang}")

                manager.audio_buffers[client_id] = []
                manager.is_recording[client_id] = True

                if interim_task and not interim_task.done():
                    interim_task.cancel()
                    try:
                        await interim_task
                    except asyncio.CancelledError:
                        pass

                interim_task = asyncio.create_task(
                    process_audio_for_client(client_id, lang)
                )

                await manager.send_personal_message(
                    json.dumps(
                        {
                            "type": "status",
                            "message": "Recording started.",
                            "language": lang,
                        }
                    ),
                    client_id,
                )
                continue

            if data["type"] == "stop_recording":
                manager.is_recording[client_id] = False

                if interim_task and not interim_task.done():
                    interim_task.cancel()
                    try:
                        await interim_task
                    except asyncio.CancelledError:
                        pass
                    interim_task = None

                # Process any remaining audio in buffer
                final_audio = np.array([], dtype=np.float32)
                if manager.audio_buffers[client_id]:
                    final_audio = np.concatenate(manager.audio_buffers[client_id])
                    logger.info(
                        f"Processing remaining audio buffer for {client_id}, size: {len(final_audio)} samples"
                    )
                    manager.audio_buffers[client_id] = []

                final_segments = manager.transcribed_segments.get(client_id, [])
                if final_audio.size > 0:
                    transcription = await transcribe_segment(
                        final_audio, client_languages.get(client_id, "en")
                    )
                    if transcription:
                        final_segments.append(transcription)
                        logger.info(
                            f"Appended final transcription for {client_id}: {transcription}"
                        )

                final_text = (
                    await merge_transcriptions(final_segments) or "No speech detected."
                )
                logger.info(f"Final transcription for {client_id}: {final_text}")

                # Initialize result to avoid UnboundLocalError
                result = {
                    "action": "error",
                    "message": "No transcription or Rasa processor not initialized.",
                    "form_data": {},
                }

                # Process with gRPC model service
                if final_text != "No speech detected.":
                    try:
                        result = await analyze_text_intent(final_text)
                        if result:
                            logger.info(
                                f"Intent analysis result for {client_id}: {result}"
                            )
                        else:
                            logger.warning(f"Intent analysis failed for {client_id}")
                            result = {
                                "action": "error",
                                "message": "Intent analysis failed",
                                "form_data": {},
                            }
                    except Exception as e:
                        logger.exception(
                            f"Intent processing error for {client_id}: {e}"
                        )
                        result = {
                            "action": "error",
                            "message": f"Intent processing failed: {str(e)}",
                            "form_data": {},
                        }
                else:
                    result = {
                        "action": "no_speech",
                        "message": "No speech detected",
                        "form_data": {},
                    }

                await manager.send_personal_message(
                    json.dumps(
                        {
                            "type": "transcription",
                            "text": final_text,
                            "is_final": True,
                            "intent": result,
                        }
                    ),
                    client_id,
                )

                # Navigate and prefill topup page for topup_wallet intent
                if result.get("action") == "topup_wallet" and result.get(
                    "form_data", {}
                ).get("amount"):
                    amount = result["form_data"]["amount"]
                    _ = result["form_data"].get("name", "")
                    navigate_message = {
                        "type": "navigate",
                        "route": "/topup.html",  # Changed from /static/topup.html
                        "prefill": {
                            "amount": amount,
                            # "name": name
                        },
                    }
                    logger.info(
                        f"Sending navigate message to {client_id}: {navigate_message}"
                    )
                    await manager.send_personal_message(
                        json.dumps(navigate_message), client_id
                    )

                # Navigate to multicurrency wallet
                elif result.get("action") == "multicurrency_wallet" and result.get(
                    "form_data", {}
                ).get("amount"):
                    amount = result["form_data"]["amount"]
                    navigate_message = {
                        "type": "navigate",
                        "route": "/multicurrency.html",
                        "prefill": {
                            "amount": amount,
                        },
                    }
                    logger.info(
                        f"Sending navigate message to {client_id}: {navigate_message}"
                    )
                    await manager.send_personal_message(
                        json.dumps(navigate_message), client_id
                    )

                manager.audio_buffers[client_id] = []
                manager.transcribed_segments[client_id] = []

                await manager.send_personal_message(
                    json.dumps({"type": "status", "message": "Recording stopped."}),
                    client_id,
                )
                continue

            if data["type"] == "audio_chunk":
                if not manager.is_recording.get(client_id, False):
                    logger.info(
                        f"Received audio_chunk for {client_id}, but not recording, ignoring."
                    )
                    continue

                try:
                    audio_bytes = base64.b64decode(data["audio"])
                    audio_np = (
                        np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                        / 32768.0
                    )
                    logger.info(
                        f"Received audio chunk for {client_id}, size: {len(audio_np)} samples"
                    )
                    manager.audio_buffers[client_id].append(audio_np)
                except Exception:
                    logger.exception(f"Audio chunk processing error for {client_id}")
                    await manager.send_personal_message(
                        json.dumps(
                            {"type": "error", "message": "Audio chunk processing error"}
                        ),
                        client_id,
                    )

    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected.")
        manager.is_recording[client_id] = False
        if interim_task and not interim_task.done():
            interim_task.cancel()
            try:
                await interim_task
            except asyncio.CancelledError:
                pass
        manager.disconnect(client_id)
    except Exception as e:
        logger.exception(f"Unhandled server error for {client_id}: {e}")
        manager.is_recording[client_id] = False
        if interim_task and not interim_task.done():
            interim_task.cancel()
            try:
                await interim_task
            except asyncio.CancelledError:
                pass
        manager.disconnect(client_id)


@app.get("/topup.html", response_class=HTMLResponse)
async def get_topup():
    try:
        with open("static/topup.html") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        logger.error("topup.html not found in static directory")
        return HTMLResponse(content="Error: topup.html not found", status_code=404)


@app.get("/multicurrency.html", response_class=HTMLResponse)
async def get_multicurrency():
    try:
        with open("static/multicurrency.html") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        logger.error("multicurrency.html not found in static directory")
        return HTMLResponse(
            content="Error: multicurrency.html not found", status_code=404
        )


@app.get("/", response_class=HTMLResponse)
async def get_root():
    return HTMLResponse(content=open("static/index.html").read())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
