"""Output format tests for Qwen3-TTS server.

Tests that each response_format returns audio with the expected
Content-Type header and non-empty body.
"""

import pytest
import httpx

from utils.client import TTSHTTPClient
from utils.audio import is_valid_wav

TEXT = "Audio format test."

# (format, expected content-type prefix)
FORMATS = [
    ("wav",  "audio/wav"),
    ("mp3",  "audio/mpeg"),
    ("flac", "audio/flac"),
    ("ogg",  "audio/ogg"),
    ("opus", "audio/ogg"),
]


@pytest.mark.parametrize("fmt,expected_ct", FORMATS)
class TestOutputFormats:
    """Each format returns non-empty audio with the correct Content-Type."""

    def test_format_returns_bytes(self, ensure_server, ensure_model_loaded, fmt, expected_ct, http_client):
        """Response body is non-empty for format '{fmt}'."""
        response = http_client.post(
            "/v1/audio/speech",
            json={"model": "qwen3-tts", "input": TEXT, "voice": "alloy", "response_format": fmt},
        )
        assert response.status_code == 200, f"Format {fmt}: status {response.status_code}"
        assert len(response.content) > 0, f"Format {fmt}: empty response body"
        print(f"Format: {fmt}")
        print(f"Audio bytes: {len(response.content)}")

    def test_format_content_type(self, ensure_server, ensure_model_loaded, fmt, expected_ct, http_client):
        """Content-Type header matches expected type for format '{fmt}'."""
        response = http_client.post(
            "/v1/audio/speech",
            json={"model": "qwen3-tts", "input": TEXT, "voice": "alloy", "response_format": fmt},
        )
        assert response.status_code == 200
        ct = response.headers.get("content-type", "")
        assert ct.startswith(expected_ct), (
            f"Format {fmt}: expected Content-Type starting with '{expected_ct}', got '{ct}'"
        )


class TestWavFormat:
    """Extra validation for WAV output specifically."""

    def test_wav_format(self, ensure_server, ensure_model_loaded):
        """WAV output has a valid RIFF/WAVE header."""
        with TTSHTTPClient() as client:
            audio = client.synthesize(TEXT, response_format="wav")
        assert is_valid_wav(audio)
        print(f"Format: wav")
        print(f"Audio bytes: {len(audio)}")
        print(f"Voice: alloy")
