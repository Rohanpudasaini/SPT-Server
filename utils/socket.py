# --- CONNECTION MANAGER FOR WEBSOCKET ---
import asyncio
import json
from typing import Dict, List
from fastapi.websockets import WebSocket
from numpy import ndarray



class ConnectionManager:
    def __init__(self, logger):
        self.active_connections: Dict[str, WebSocket] = {}
        self.audio_buffers: Dict[str, List[ndarray]] = {}
        self.processing_tasks: Dict[str, asyncio.Task] = {}
        self.is_recording: Dict[str, bool] = {}
        self.transcribed_segments: Dict[str, List[str]] = {}
        self.logger = logger

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        if client_id in self.active_connections:
            self.logger.warning(f"Client {client_id} reconnected, closing old connection.")
            await self.active_connections[client_id].close()

        self.active_connections[client_id] = websocket
        self.audio_buffers[client_id] = []
        self.is_recording[client_id] = False
        self.transcribed_segments[client_id] = []
        self.logger.info(f"Client {client_id} connected.")
        await self.send_personal_message(json.dumps({"type": "status", "message": "Connected"}), client_id)

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.audio_buffers:
            del self.audio_buffers[client_id]
        if client_id in self.processing_tasks and not self.processing_tasks[client_id].done():
            self.processing_tasks[client_id].cancel()
            del self.processing_tasks[client_id]
        if client_id in self.is_recording:
            del self.is_recording[client_id]
        if client_id in self.transcribed_segments:
            del self.transcribed_segments[client_id]
        self.logger.info(f"Client {client_id} disconnected.")

    async def send_personal_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(message)
                self.logger.info(f"Sent message to {client_id}: {message}")
            except Exception as e:
                self.logger.error(f"Failed sending to {client_id}: {e} — disconnecting.")
                self.disconnect(client_id)