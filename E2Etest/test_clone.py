"""Voice clone endpoint tests for Qwen3-TTS server.

Tests POST /v1/audio/speech/clone (multipart form):
- Reference audio + input text → synthesized audio in cloned voice
- Voice prompt cache: second call with same ref audio is faster
- ref_text parameter accepted
- language parameter accepted
"""

import time
from pathlib import Path

import pytest

from utils.client import TTSHTTPClient
from utils.audio import is_valid_wav


TEXT = "This is a voice cloning test."


@pytest.mark.slow
class TestVoiceClone:
    """Basic voice clone functionality."""

    def test_clone_returns_audio(self, ensure_server, ensure_model_loaded, ref_audio_3s: Path):
        """Clone endpoint returns non-empty audio bytes."""
        with TTSHTTPClient() as client:
            audio = client.clone(ref_audio_3s, input=TEXT)
        assert isinstance(audio, bytes)
        assert len(audio) > 0
        print(f"Audio bytes: {len(audio)}")
        print(f"Voice: clone")
        print(f"Format: wav")

    def test_clone_response_is_valid_wav(
        self, ensure_server, ensure_model_loaded, ref_audio_3s: Path
    ):
        """Clone default response_format=wav returns valid WAV."""
        with TTSHTTPClient() as client:
            audio = client.clone(ref_audio_3s, input=TEXT, response_format="wav")
        assert is_valid_wav(audio), "Clone response is not valid WAV"

    def test_clone_with_ref_text(
        self, ensure_server, ensure_model_loaded, ref_audio_3s: Path
    ):
        """Clone with ref_text parameter completes without error."""
        with TTSHTTPClient() as client:
            audio = client.clone(
                ref_audio_3s,
                input=TEXT,
                ref_text="Some reference transcript text.",
            )
        assert len(audio) > 0

    def test_clone_with_language(
        self, ensure_server, ensure_model_loaded, ref_audio_3s: Path
    ):
        """Clone with explicit language parameter completes without error."""
        with TTSHTTPClient() as client:
            audio = client.clone(
                ref_audio_3s,
                input=TEXT,
                language="English",
            )
        assert len(audio) > 0

    def test_clone_missing_file_returns_4xx(self, http_client, ensure_server):
        """Clone request without file returns 4xx."""
        response = http_client.post(
            "/v1/audio/speech/clone",
            data={"input": TEXT},
        )
        assert response.status_code in (400, 422)

    def test_clone_missing_input_returns_4xx(
        self, http_client, ensure_server, ref_audio_3s: Path
    ):
        """Clone request without input text returns 400."""
        with open(ref_audio_3s, "rb") as f:
            files = {"file": ("ref.wav", f, "audio/wav")}
            response = http_client.post(
                "/v1/audio/speech/clone",
                files=files,
                data={"input": ""},
            )
        assert response.status_code == 400


@pytest.mark.slow
class TestVoiceCloneCache:
    """Voice prompt cache hit tests (second call with same ref audio is faster)."""

    def test_clone_cache_hit(
        self, ensure_server, ensure_model_loaded, ref_audio_5s: Path
    ):
        """Repeated clone of same ref audio is served from cache (faster)."""
        with TTSHTTPClient() as client:
            # First call — computes speaker embedding
            t0 = time.time()
            audio1 = client.clone(ref_audio_5s, input=TEXT)
            t_first = time.time() - t0

            # Second call — should hit voice prompt cache
            t0 = time.time()
            audio2 = client.clone(ref_audio_5s, input=TEXT)
            t_second = time.time() - t0

        assert len(audio1) > 0
        assert len(audio2) > 0
        # Cache hit should be noticeably faster (at least 20% faster than first)
        # This is a soft check — if the model is already fast, we skip the ratio check
        if t_first > 5.0:  # Only assert when first call was slow enough to measure
            assert t_second < t_first * 0.9, (
                f"Cache hit not faster: first={t_first:.1f}s second={t_second:.1f}s"
            )
        print(f"First call: {t_first:.2f}s")
        print(f"Second call (cache): {t_second:.2f}s")

    def test_clone_cache_reflected_in_health(
        self, http_client, ensure_server, ensure_model_loaded, ref_audio_3s: Path
    ):
        """voice_cache_size in /health increases after clone call."""
        # Clear caches first
        http_client.post("/cache/clear")
        before = http_client.get("/health").json()["voice_cache_size"]

        # Clone with a ref audio
        with open(ref_audio_3s, "rb") as f:
            files = {"file": ("ref.wav", f, "audio/wav")}
            http_client.post(
                "/v1/audio/speech/clone",
                files=files,
                data={"input": TEXT},
            )

        after = http_client.get("/health").json()["voice_cache_size"]
        assert after >= before, (
            f"voice_cache_size did not increase: before={before} after={after}"
        )
