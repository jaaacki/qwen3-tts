"""WebSocket streaming tests for Qwen3-TTS server.

Tests WS /v1/audio/speech/ws:
- Send JSON with 'input' field
- Receive binary PCM chunks (one per sentence)
- Receive {"event": "done"} after all chunks
"""

import asyncio
import json

import pytest
import websockets

from utils.client import TTSWebSocketClient


TEXT_SHORT = "Hello world."
TEXT_MULTI = (
    "The quick brown fox. Pack my box with five dozen liquor jugs. "
    "How razorback-jumping frogs can level six piqued gymnasts."
)


@pytest.mark.websocket
class TestWebSocketSpeech:
    """Core WebSocket synthesis tests."""

    @pytest.mark.asyncio
    async def test_ws_synthesize_returns_pcm(self, ensure_server, ensure_model_loaded, ws_url: str):
        """WebSocket synthesis returns at least one binary PCM chunk."""
        async with TTSWebSocketClient(ws_url) as client:
            pcm_chunks, done_event = await client.synthesize(TEXT_SHORT)

        assert len(pcm_chunks) > 0, "No PCM chunks received"
        for chunk in pcm_chunks:
            assert isinstance(chunk, bytes)
            assert len(chunk) > 0

    @pytest.mark.asyncio
    async def test_ws_receives_done_event(self, ensure_server, ensure_model_loaded, ws_url: str):
        """WebSocket synthesis ends with {"event": "done"}."""
        async with TTSWebSocketClient(ws_url) as client:
            pcm_chunks, done_event = await client.synthesize(TEXT_SHORT)

        assert done_event.get("event") == "done", f"Unexpected done event: {done_event}"

    @pytest.mark.asyncio
    async def test_ws_pcm_chunks_are_int16(self, ensure_server, ensure_model_loaded, ws_url: str):
        """Each PCM chunk has even byte length (int16 format)."""
        async with TTSWebSocketClient(ws_url) as client:
            pcm_chunks, _ = await client.synthesize(TEXT_SHORT)

        for i, chunk in enumerate(pcm_chunks):
            assert len(chunk) % 2 == 0, f"Chunk {i} has odd byte length {len(chunk)}"

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_ws_multi_sentence_multiple_chunks(
        self, ensure_server, ensure_model_loaded, ws_url: str
    ):
        """Multi-sentence input produces multiple PCM chunks (one per sentence)."""
        async with TTSWebSocketClient(ws_url) as client:
            pcm_chunks, _ = await client.synthesize(TEXT_MULTI)

        # 3-sentence text should produce >= 2 chunks
        assert len(pcm_chunks) >= 2, (
            f"Expected multiple chunks for multi-sentence input, got {len(pcm_chunks)}"
        )

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_ws_multiple_requests(self, ensure_server, ensure_model_loaded, ws_url: str):
        """WebSocket connection can handle multiple sequential synthesis requests."""
        async with TTSWebSocketClient(ws_url) as client:
            chunks1, done1 = await client.synthesize(TEXT_SHORT)
            chunks2, done2 = await client.synthesize("Second request.")

        assert len(chunks1) > 0
        assert len(chunks2) > 0
        assert done1.get("event") == "done"
        assert done2.get("event") == "done"


@pytest.mark.websocket
class TestWebSocketErrors:
    """Error handling for malformed WebSocket messages."""

    @pytest.mark.asyncio
    async def test_ws_empty_input_returns_error(self, ensure_server, ws_url: str):
        """Empty input field returns an error event (not a crash)."""
        async with TTSWebSocketClient(ws_url) as client:
            await client.send_raw({"input": ""})
            raw = await client.recv(timeout=10)
            if isinstance(raw, bytes):
                pytest.fail("Expected JSON error event, got binary data")
            event = json.loads(raw)
            assert event.get("event") == "error", f"Expected error event, got: {event}"

    @pytest.mark.asyncio
    async def test_ws_voice_parameter_accepted(
        self, ensure_server, ensure_model_loaded, ws_url: str
    ):
        """voice parameter is accepted and applied."""
        async with TTSWebSocketClient(ws_url) as client:
            chunks, done = await client.synthesize(TEXT_SHORT, voice="vivian")
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_ws_temperature_param_accepted(
        self, ensure_server, ensure_model_loaded, ws_url: str
    ):
        """temperature and top_p parameters are accepted without error."""
        async with TTSWebSocketClient(ws_url) as client:
            chunks, done = await client.synthesize(
                TEXT_SHORT, temperature=0.8, top_p=0.95
            )
        assert len(chunks) > 0
        assert done.get("event") == "done"
