"""Voice mapping tests for Qwen3-TTS server.

Tests that all built-in voices and OpenAI aliases produce valid audio.
"""

import pytest

from utils.client import TTSHTTPClient
from utils.audio import is_valid_wav

TEXT = "Testing voice output."

NATIVE_VOICES = [
    "vivian", "serena", "uncle_fu", "dylan",
    "eric", "ryan", "aiden", "ono_anna", "sohee",
]

OPENAI_ALIASES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]


@pytest.mark.smoke
class TestVoiceMapping:
    """Each voice name/alias must return valid audio."""

    @pytest.mark.parametrize("voice", NATIVE_VOICES)
    def test_native_voice_names(self, ensure_server, ensure_model_loaded, voice: str):
        """Native Qwen speaker names produce valid WAV audio."""
        with TTSHTTPClient() as client:
            audio = client.synthesize(TEXT, voice=voice, response_format="wav")
        assert len(audio) > 0
        assert is_valid_wav(audio), f"Voice '{voice}' returned invalid WAV"
        print(f"Voice: {voice}")
        print(f"Audio bytes: {len(audio)}")
        print(f"Format: wav")

    @pytest.mark.parametrize("voice", OPENAI_ALIASES)
    def test_openai_voice_aliases(self, ensure_server, ensure_model_loaded, voice: str):
        """OpenAI-compatible voice aliases produce valid WAV audio."""
        with TTSHTTPClient() as client:
            audio = client.synthesize(TEXT, voice=voice, response_format="wav")
        assert len(audio) > 0
        assert is_valid_wav(audio), f"Alias '{voice}' returned invalid WAV"
        print(f"Voice: {voice}")
        print(f"Audio bytes: {len(audio)}")
        print(f"Format: wav")

    def test_unknown_voice_returns_400(self, ensure_server):
        """Unknown voice name returns 400 with valid voice list (#99)."""
        with TTSHTTPClient() as client:
            response = client.client.post(
                f"{client.base_url}/v1/audio/speech",
                json={"input": TEXT, "voice": "custom_voice_xyz", "response_format": "wav"},
            )
        assert response.status_code == 400
        assert "voice" in response.json().get("detail", "").lower()


class TestVoiceOutput:
    """Different voices should produce different audio bytes."""

    @pytest.mark.slow
    def test_two_voices_differ(self, ensure_server, ensure_model_loaded):
        """vivian and eric produce different audio for the same text."""
        with TTSHTTPClient() as client:
            a1 = client.synthesize(TEXT, voice="vivian", response_format="wav")
            a2 = client.synthesize(TEXT, voice="eric", response_format="wav")
        assert a1 != a2, "Different voices produced identical audio"
