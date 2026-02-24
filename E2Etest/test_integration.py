"""Integration tests for Qwen3-TTS multi-feature scenarios.

Tests feature combinations:
- Audio LRU cache: same request → cache hit (audio_cache_size increases)
- Cache clear: clears audio + voice caches
- Speed parameter: affects audio duration
- Generation params: temperature and top_p are accepted
- Priority queue: WS requests are not blocked by concurrent REST requests
- Queue depth: health shows queue_depth field
"""

import asyncio
import time

import pytest

from utils.client import TTSHTTPClient, TTSAsyncHTTPClient, TTSWebSocketClient
from utils.audio import is_valid_wav, audio_duration_from_wav

TEXT = "Integration test sentence."


# =============================================================================
# Audio Cache Tests
# =============================================================================

@pytest.mark.integration
class TestAudioCache:
    """Audio LRU cache warms on first call; hits on repeat calls."""

    def test_audio_cache_populated(self, http_client, ensure_server, ensure_model_loaded):
        """audio_cache_size in /health increases after a synthesis request."""
        # Clear caches for clean measurement
        http_client.post("/cache/clear")
        before = http_client.get("/health").json()["audio_cache_size"]

        # Synthesize once
        http_client.post(
            "/v1/audio/speech",
            json={"model": "qwen3-tts", "input": "Cache population test.", "voice": "alloy"},
        )

        after = http_client.get("/health").json()["audio_cache_size"]
        assert after > before, f"Cache did not grow: before={before} after={after}"

    def test_cache_cleared_on_request(self, http_client, ensure_server, ensure_model_loaded):
        """POST /cache/clear empties audio cache and returns cleared counts."""
        # Put something in cache
        http_client.post(
            "/v1/audio/speech",
            json={"model": "qwen3-tts", "input": "Cache clear test.", "voice": "alloy"},
        )

        before = http_client.get("/health").json()["audio_cache_size"]
        result = http_client.post("/cache/clear").json()

        assert result["audio_cleared"] >= 0
        assert result["voice_cleared"] >= 0
        after = http_client.get("/health").json()["audio_cache_size"]
        assert after == 0, f"Cache not cleared: audio_cache_size={after}"
        if before > 0:
            assert result["audio_cleared"] == before


# =============================================================================
# Speed Parameter
# =============================================================================

@pytest.mark.integration
@pytest.mark.slow
class TestSpeedParameter:
    """Speed parameter modifies audio duration."""

    def test_speed_parameter_affects_duration(self, ensure_server, ensure_model_loaded):
        """Audio at speed=2.0 is shorter than at speed=1.0."""
        long_text = (
            "The speed parameter should change the duration of the synthesized audio output. "
            "This sentence is long enough to produce a measurable difference."
        )
        with TTSHTTPClient() as client:
            audio_normal = client.synthesize(long_text, speed=1.0, response_format="wav")
            audio_fast   = client.synthesize(long_text, speed=2.0, response_format="wav")

        dur_normal = audio_duration_from_wav(audio_normal)
        dur_fast   = audio_duration_from_wav(audio_fast)

        assert dur_normal is not None
        assert dur_fast   is not None
        assert dur_fast < dur_normal, (
            f"Fast audio ({dur_fast:.2f}s) not shorter than normal ({dur_normal:.2f}s)"
        )


# =============================================================================
# Generation Parameters
# =============================================================================

@pytest.mark.integration
class TestGenerationParams:
    """temperature and top_p parameters are accepted without error."""

    def test_temperature_param_accepted(self, ensure_server, ensure_model_loaded):
        """temperature=0.8 is accepted and returns valid audio."""
        with TTSHTTPClient() as client:
            audio = client.synthesize(TEXT, temperature=0.8, response_format="wav")
        assert is_valid_wav(audio)
        print(f"Audio bytes: {len(audio)}")
        print(f"Voice: alloy")
        print(f"Format: wav")

    def test_top_p_param_accepted(self, ensure_server, ensure_model_loaded):
        """top_p=0.9 is accepted and returns valid audio."""
        with TTSHTTPClient() as client:
            audio = client.synthesize(TEXT, top_p=0.9, response_format="wav")
        assert is_valid_wav(audio)

    def test_temperature_and_top_p_together(self, ensure_server, ensure_model_loaded):
        """temperature and top_p can be combined."""
        with TTSHTTPClient() as client:
            audio = client.synthesize(TEXT, temperature=0.7, top_p=0.95, response_format="wav")
        assert is_valid_wav(audio)


# =============================================================================
# Priority Queue
# =============================================================================

@pytest.mark.integration
@pytest.mark.slow
class TestPriorityQueue:
    """WebSocket requests are not blocked by concurrent REST synthesis."""

    @pytest.mark.asyncio
    async def test_websocket_not_blocked_by_rest(
        self, ensure_server, ensure_model_loaded, base_url: str, ws_url: str
    ):
        """A WebSocket synthesis completes while a REST request is running."""
        long_text = " ".join(["Sentence number {}.".format(i) for i in range(5)])

        async with TTSAsyncHTTPClient(base_url) as http:
            # Fire off a REST request (runs in background)
            rest_task = asyncio.create_task(
                http.synthesize(long_text, voice="alloy")
            )

            # Give REST a head start
            await asyncio.sleep(0.2)

            # WebSocket should complete independently
            ws_start = time.time()
            async with TTSWebSocketClient(ws_url) as ws:
                chunks, done = await ws.synthesize("Quick WS test.", voice="alloy")
            ws_elapsed = time.time() - ws_start

            assert len(chunks) > 0
            assert done.get("event") == "done"
            # WS should not wait for the REST request to finish (priority queue)
            assert ws_elapsed < 60.0, f"WS took {ws_elapsed:.1f}s (possibly blocked by REST)"

            # Wait for REST too
            rest_audio = await rest_task
            assert len(rest_audio) > 0


# =============================================================================
# Queue Depth
# =============================================================================

@pytest.mark.integration
class TestQueueDepth:
    """queue_depth and max_queue_depth in health endpoint."""

    def test_queue_depth_in_health(self, http_client, ensure_server):
        """Health includes queue_depth ≥ 0 and max_queue_depth ≥ 0."""
        data = http_client.get("/health").json()
        assert data["queue_depth"] >= 0
        assert data["max_queue_depth"] >= 0
