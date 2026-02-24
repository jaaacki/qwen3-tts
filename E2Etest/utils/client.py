"""HTTP and WebSocket client wrappers for Qwen3-TTS E2E tests."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import AsyncGenerator, Optional

import httpx
import websockets


DEFAULT_BASE_URL = "http://localhost:8101"
DEFAULT_WS_URL   = "ws://localhost:8101/v1/audio/speech/ws"


class TTSHTTPClient:
    """Synchronous HTTP client for Qwen3-TTS API."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: float = 300):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(base_url=self.base_url, timeout=timeout)

    # ------------------------------------------------------------------
    # Health / cache
    # ------------------------------------------------------------------

    def health(self) -> dict:
        """GET /health"""
        response = self.client.get("/health")
        response.raise_for_status()
        return response.json()

    def clear_cache(self) -> dict:
        """POST /cache/clear"""
        response = self.client.post("/cache/clear")
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def synthesize(
        self,
        input: str,
        voice: str = "alloy",
        response_format: str = "wav",
        speed: float = 1.0,
        language: Optional[str] = None,
        instruct: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> bytes:
        """POST /v1/audio/speech — returns raw audio bytes."""
        body: dict = {
            "model": "qwen3-tts",
            "input": input,
            "voice": voice,
            "response_format": response_format,
            "speed": speed,
        }
        if language is not None:
            body["language"] = language
        if instruct is not None:
            body["instruct"] = instruct
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p

        response = self.client.post("/v1/audio/speech", json=body)
        response.raise_for_status()
        return response.content

    def synthesize_stream(
        self,
        input: str,
        voice: str = "alloy",
        language: Optional[str] = None,
    ) -> list[dict]:
        """POST /v1/audio/speech/stream — returns parsed SSE events."""
        body: dict = {"model": "qwen3-tts", "input": input, "voice": voice}
        if language is not None:
            body["language"] = language

        response = self.client.post(
            "/v1/audio/speech/stream",
            json=body,
            headers={"Accept": "text/event-stream"},
        )
        response.raise_for_status()

        events = []
        for line in response.text.strip().split("\n"):
            if line.startswith("data: "):
                payload = line[6:].strip()
                if payload:
                    try:
                        events.append(json.loads(payload))
                    except json.JSONDecodeError:
                        events.append({"raw": payload})
        return events

    def synthesize_stream_pcm(
        self,
        input: str,
        voice: str = "alloy",
        language: Optional[str] = None,
    ) -> tuple[bytes, dict]:
        """POST /v1/audio/speech/stream/pcm — returns (pcm_bytes, response_headers)."""
        body: dict = {"model": "qwen3-tts", "input": input, "voice": voice}
        if language is not None:
            body["language"] = language

        response = self.client.post("/v1/audio/speech/stream/pcm", json=body)
        response.raise_for_status()
        return response.content, dict(response.headers)

    def clone(
        self,
        ref_audio_path: Path,
        input: str,
        ref_text: Optional[str] = None,
        language: Optional[str] = None,
        response_format: str = "wav",
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> bytes:
        """POST /v1/audio/speech/clone — multipart upload, returns audio bytes."""
        with open(ref_audio_path, "rb") as f:
            files = {"file": (ref_audio_path.name, f, "audio/wav")}
            data: dict = {"input": input, "response_format": response_format}
            if ref_text is not None:
                data["ref_text"] = ref_text
            if language is not None:
                data["language"] = language
            if temperature is not None:
                data["temperature"] = str(temperature)
            if top_p is not None:
                data["top_p"] = str(top_p)

            response = self.client.post(
                "/v1/audio/speech/clone", files=files, data=data
            )
            response.raise_for_status()
            return response.content

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class TTSAsyncHTTPClient:
    """Asynchronous HTTP client for Qwen3-TTS API."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: float = 300):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)
        return self

    async def __aexit__(self, *_):
        if self.client:
            await self.client.aclose()

    async def health(self) -> dict:
        response = await self.client.get("/health")
        response.raise_for_status()
        return response.json()

    async def synthesize(
        self,
        input: str,
        voice: str = "alloy",
        response_format: str = "wav",
    ) -> bytes:
        body = {"model": "qwen3-tts", "input": input, "voice": voice, "response_format": response_format}
        response = await self.client.post("/v1/audio/speech", json=body)
        response.raise_for_status()
        return response.content


class TTSWebSocketClient:
    """WebSocket client for Qwen3-TTS streaming endpoint."""

    def __init__(self, ws_url: str = DEFAULT_WS_URL):
        self.ws_url = ws_url
        self.websocket = None

    async def connect(self):
        self.websocket = await websockets.connect(self.ws_url)

    async def synthesize(
        self,
        input: str,
        voice: str = "alloy",
        language: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> tuple[list[bytes], dict]:
        """Send a synthesis request; collect PCM chunks and the done event.

        Returns (pcm_chunks, done_event).
        """
        if not self.websocket:
            raise RuntimeError("Not connected")

        msg: dict = {"input": input, "voice": voice}
        if language is not None:
            msg["language"] = language
        if temperature is not None:
            msg["temperature"] = temperature
        if top_p is not None:
            msg["top_p"] = top_p

        await self.websocket.send(json.dumps(msg))

        pcm_chunks: list[bytes] = []
        done_event: dict = {}

        while True:
            raw = await asyncio.wait_for(self.websocket.recv(), timeout=120)
            if isinstance(raw, bytes):
                pcm_chunks.append(raw)
            else:
                event = json.loads(raw)
                if event.get("event") == "done":
                    done_event = event
                    break
                elif event.get("event") == "error":
                    raise RuntimeError(f"Server error: {event.get('detail')}")

        return pcm_chunks, done_event

    async def send_raw(self, payload: dict):
        """Send a raw JSON message."""
        await self.websocket.send(json.dumps(payload))

    async def recv(self, timeout: float = 10):
        """Receive one message."""
        return await asyncio.wait_for(self.websocket.recv(), timeout=timeout)

    async def close(self):
        if self.websocket:
            await self.websocket.close()
            self.websocket = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *_):
        await self.close()
