import asyncio
import logging
from typing import Dict, List, Optional

import grpc
import protos.speech_service_pb2 as speech_pb2
import protos.speech_service_pb2_grpc as speech_pb2_grpc

logger = logging.getLogger(__name__)


class SpeechServiceClient:
    """gRPC client for the SpeechService."""

    def __init__(self, host: str = "localhost", port: int = 50051):
        self.host = host
        self.port = port
        self.channel = None
        self.stub = None

    async def connect(self):
        """Establish connection to the gRPC server."""
        try:
            self.channel = grpc.aio.insecure_channel(f"{self.host}:{self.port}")
            self.stub = speech_pb2_grpc.SpeechServiceStub(self.channel)
            logger.info(f"Connected to SpeechService at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to SpeechService: {e}")
            raise

    async def disconnect(self):
        """Close the connection."""
        if self.channel:
            await self.channel.close()
            logger.info("Disconnected from SpeechService")

    async def transcribe_audio_segment(
        self, audio_data: bytes, language: str, sample_rate: int = 16000
    ) -> Optional[str]:
        """Transcribe an audio segment."""
        try:
            if not self.stub:
                await self.connect()

            request = speech_pb2.TranscribeAudioRequest(
                audio_data=audio_data, language=language, sample_rate=sample_rate
            )

            response = await self.stub.TranscribeAudioSegment(request)

            if response.success:
                logger.info(
                    f"Transcription successful: {response.transcription[:50]}..."
                )
                return response.transcription
            else:
                logger.error(f"Transcription failed: {response.error_message}")
                return None

        except Exception as e:
            logger.exception(f"Error transcribing audio: {e}")
            return None

    async def merge_transcriptions(
        self, transcriptions: List[str], language: str = "en"
    ) -> Optional[str]:
        """Merge multiple transcriptions into a single text."""
        try:
            if not self.stub:
                await self.connect()

            request = speech_pb2.MergeTranscriptionsRequest(
                transcriptions=transcriptions, language=language
            )

            response = await self.stub.MergeTranscriptions(request)

            if response.success:
                logger.info(
                    f"Transcription merging successful: {response.merged_text[:50]}..."
                )
                return response.merged_text
            else:
                logger.error(f"Transcription merging failed: {response.error_message}")
                return None

        except Exception as e:
            logger.exception(f"Error merging transcriptions: {e}")
            return None

    async def process_intent(
        self, text: str, confidence_threshold: float = 0.80
    ) -> Optional[Dict]:
        """Process text for intent analysis and entity extraction."""
        try:
            if not self.stub:
                await self.connect()

            request = speech_pb2.ProcessIntentRequest(
                text=text, confidence_threshold=confidence_threshold
            )

            response = await self.stub.ProcessIntent(request)

            if response.success:
                result = {
                    "action": response.action,
                    "message": response.message,
                    "form_data": dict(response.form_data),
                    "confidence": response.confidence,
                    "entities": [
                        {
                            "entity": entity.entity,
                            "value": entity.value,
                            "confidence": entity.confidence,
                            "start": entity.start,
                            "end": entity.end,
                        }
                        for entity in response.entities
                    ],
                }
                logger.info(f"Intent processing successful: {result['action']}")
                return result
            else:
                logger.error(f"Intent processing failed: {response.error_message}")
                return None

        except Exception as e:
            logger.exception(f"Error processing intent: {e}")
            return None

    async def health_check(self) -> Optional[Dict]:
        """Perform a health check on the model service."""
        try:
            if not self.stub:
                await self.connect()

            request = speech_pb2.HealthCheckRequest()
            response = await self.stub.HealthCheck(request)

            result = {
                "healthy": response.healthy,
                "status": response.status,
                "version": response.version,
                "loaded_models": list(response.loaded_models),
            }
            logger.info(f"Health check: {result['status']}")
            return result

        except Exception as e:
            logger.exception(f"Error performing health check: {e}")
            return None


# Global client instance
_speech_client = None


async def get_speech_client() -> SpeechServiceClient:
    """Get or create the global speech client instance."""
    global _speech_client
    if _speech_client is None:
        _speech_client = SpeechServiceClient()
        await _speech_client.connect()
    return _speech_client


async def transcribe_audio_bytes(audio_bytes: bytes, language: str) -> Optional[str]:
    """Convenience function to transcribe audio bytes."""
    client = await get_speech_client()
    return await client.transcribe_audio_segment(audio_bytes, language)


async def merge_transcription_texts(transcriptions: List[str]) -> Optional[str]:
    """Convenience function to merge transcription texts."""
    client = await get_speech_client()
    return await client.merge_transcriptions(transcriptions)


async def analyze_text_intent(text: str) -> Optional[Dict]:
    """Convenience function to analyze text for intent."""
    client = await get_speech_client()
    return await client.process_intent(text)


async def check_model_health() -> Optional[Dict]:
    """Convenience function to check model service health."""
    client = await get_speech_client()
    return await client.health_check()
