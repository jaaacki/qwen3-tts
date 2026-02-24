"""HTTP API tests for Qwen3-TTS server.

Tests:
- GET /health
- POST /v1/audio/speech
- POST /cache/clear
- Error handling
"""

import time
from pathlib import Path

import httpx
import pytest

from utils.client import TTSHTTPClient
from utils.audio import is_valid_wav, audio_duration_from_wav


# =============================================================================
# Smoke Tests — Health
# =============================================================================

@pytest.mark.smoke
class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, http_client: httpx.Client, ensure_server):
        response = http_client.get("/health")
        assert response.status_code == 200

    def test_health_has_expected_fields(self, http_client: httpx.Client, ensure_server):
        data = http_client.get("/health").json()
        assert "status" in data
        assert "model_loaded" in data
        assert "model_id" in data
        assert "cuda" in data
        assert "voices" in data
        assert "queue_depth" in data
        assert "max_queue_depth" in data

    def test_health_status_is_ok(self, http_client: httpx.Client, ensure_server):
        data = http_client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_lists_voices(self, http_client: httpx.Client, ensure_server):
        data = http_client.get("/health").json()
        voices = data["voices"]
        assert isinstance(voices, list)
        assert len(voices) > 0
        # Known native Qwen voices
        assert "vivian" in voices
        assert "alloy" in voices  # OpenAI alias

    def test_health_gpu_info_when_available(self, http_client: httpx.Client, ensure_server):
        data = http_client.get("/health").json()
        if data.get("cuda"):
            assert "gpu_name" in data
            assert "gpu_allocated_mb" in data
            assert isinstance(data["gpu_allocated_mb"], (int, float))

    def test_health_cache_fields(self, http_client: httpx.Client, ensure_server):
        data = http_client.get("/health").json()
        assert "audio_cache_size" in data
        assert "audio_cache_max" in data
        assert "voice_cache_size" in data
        assert "voice_cache_max" in data
        assert isinstance(data["audio_cache_size"], int)
        assert isinstance(data["audio_cache_max"], int)

    def test_queue_depth_in_health(self, http_client: httpx.Client, ensure_server):
        data = http_client.get("/health").json()
        assert isinstance(data["queue_depth"], int)
        assert data["queue_depth"] >= 0
        assert data["max_queue_depth"] >= 0


# =============================================================================
# Smoke Tests — Basic Synthesis
# =============================================================================

@pytest.mark.smoke
class TestBasicSynthesis:
    """Basic POST /v1/audio/speech tests."""

    def test_basic_synthesis_returns_audio(self, ensure_server):
        """Synthesis returns non-empty bytes."""
        with TTSHTTPClient() as client:
            audio = client.synthesize("Hello, world!")
        assert isinstance(audio, bytes)
        assert len(audio) > 0

    def test_response_is_valid_wav(self, ensure_server):
        """Default response_format=wav produces valid WAV bytes."""
        with TTSHTTPClient() as client:
            audio = client.synthesize("Hello.", response_format="wav")
        assert is_valid_wav(audio), "Response is not a valid WAV file"
        print(f"Audio bytes: {len(audio)}")
        print(f"Format: wav")

    def test_audio_has_nonzero_duration(self, ensure_server):
        """Synthesized WAV has a positive duration."""
        with TTSHTTPClient() as client:
            audio = client.synthesize("The quick brown fox.", response_format="wav")
        duration = audio_duration_from_wav(audio)
        assert duration is not None
        assert duration > 0.1, f"Audio duration too short: {duration:.3f}s"
        print(f"Audio duration: {duration:.2f}s")
        print(f"Audio bytes: {len(audio)}")
        print(f"Voice: alloy")
        print(f"Format: wav")

    @pytest.mark.slow
    def test_synthesis_deterministic(self, ensure_server, ensure_model_loaded):
        """Same input produces byte-identical output (cache hit guarantees this)."""
        with TTSHTTPClient() as client:
            audio1 = client.synthesize("Determinism test.", voice="alloy")
            audio2 = client.synthesize("Determinism test.", voice="alloy")
        # Exact match is only guaranteed after the cache warms on the first call
        assert audio1 == audio2, "Same input produced different audio (cache miss on repeat?)"

    def test_model_loads_on_first_request(self, http_client: httpx.Client, ensure_server):
        """After synthesis, model_loaded is True in health."""
        # Trigger synthesis (may already be loaded)
        http_client.post(
            "/v1/audio/speech",
            json={"model": "qwen3-tts", "input": "Load test.", "voice": "alloy"},
        )
        data = http_client.get("/health").json()
        assert data["model_loaded"] is True


# =============================================================================
# Cache Tests
# =============================================================================

class TestCacheEndpoint:
    """Tests for POST /cache/clear."""

    def test_clear_cache_returns_counts(self, http_client: httpx.Client, ensure_server):
        """Cache clear returns audio_cleared and voice_cleared counts."""
        response = http_client.post("/cache/clear")
        assert response.status_code == 200
        data = response.json()
        assert "audio_cleared" in data
        assert "voice_cleared" in data
        assert isinstance(data["audio_cleared"], int)
        assert isinstance(data["voice_cleared"], int)


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error responses."""

    def test_empty_input_returns_400(self, http_client: httpx.Client, ensure_server):
        """Empty input string returns 400 Bad Request."""
        response = http_client.post(
            "/v1/audio/speech",
            json={"model": "qwen3-tts", "input": "", "voice": "alloy"},
        )
        assert response.status_code == 400

    def test_whitespace_input_returns_400(self, http_client: httpx.Client, ensure_server):
        """Whitespace-only input returns 400."""
        response = http_client.post(
            "/v1/audio/speech",
            json={"model": "qwen3-tts", "input": "   ", "voice": "alloy"},
        )
        assert response.status_code == 400

    def test_missing_input_returns_4xx(self, http_client: httpx.Client, ensure_server):
        """Request missing 'input' field returns 4xx."""
        response = http_client.post(
            "/v1/audio/speech",
            json={"model": "qwen3-tts", "voice": "alloy"},
        )
        assert response.status_code in (400, 422)

    def test_health_latency(self, ensure_server):
        """Health check responds in under 1 second on average."""
        with TTSHTTPClient() as client:
            times = []
            for _ in range(5):
                start = time.time()
                client.health()
                times.append(time.time() - start)
        avg = sum(times) / len(times)
        assert avg < 1.0, f"Health avg latency {avg:.3f}s"
        print(f"Health latency: {avg:.3f}s")
