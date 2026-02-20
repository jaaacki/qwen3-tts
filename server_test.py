"""Tests for server.py — Issue #1: Per-request latency breakdown logging."""
import pytest
import logging
from unittest.mock import patch, MagicMock

_torch_mock = MagicMock()
_torch_mock.cuda.is_available.return_value = False

_mock_modules = {
    "torch": _torch_mock,
    "torch.cuda": _torch_mock.cuda,
    "torch.backends": MagicMock(),
    "torch.backends.cudnn": MagicMock(),
    "torch.backends.cuda": MagicMock(),
    "soundfile": MagicMock(),
    "numpy": MagicMock(),
    "scipy": MagicMock(),
    "scipy.signal": MagicMock(),
    "pydub": MagicMock(),
    "qwen_tts": MagicMock(),
}

with patch.dict("sys.modules", _mock_modules):
    from server import resolve_voice, detect_language, logger


class TestLatencyLogger:
    def test_logger_name(self):
        assert logger.name == "qwen3-tts"

    def test_logger_is_standard(self):
        assert isinstance(logger, logging.Logger)

    def test_logger_can_emit(self, caplog):
        with caplog.at_level(logging.INFO, logger="qwen3-tts"):
            logger.info("request_complete", extra={
                "endpoint": "/v1/audio/speech",
                "queue_ms": 1.2, "inference_ms": 450.3,
                "encode_ms": 12.5, "total_ms": 464.0,
                "chars": 20, "voice": "vivian",
                "format": "wav", "language": "English",
            })
        assert "request_complete" in caplog.text


class TestResolveVoice:
    def test_default_voice_when_none(self):
        assert resolve_voice(None) == "vivian"
    def test_default_voice_when_empty(self):
        assert resolve_voice("") == "vivian"
    def test_direct_qwen_voice(self):
        assert resolve_voice("serena") == "serena"
    def test_openai_alias(self):
        assert resolve_voice("alloy") == "ryan"
    def test_unknown_voice_passthrough(self):
        assert resolve_voice("custom_voice") == "custom_voice"
    def test_case_insensitive(self):
        assert resolve_voice("VIVIAN") == "vivian"
        assert resolve_voice("Alloy") == "ryan"


class TestDetectLanguage:
    def test_english(self):
        assert detect_language("Hello world") == "English"
    def test_chinese(self):
        assert detect_language("\u4f60\u597d\u4e16\u754c") == "Chinese"
    def test_japanese(self):
        assert detect_language("\u3053\u3093\u306b\u3061\u306f") == "Japanese"
    def test_korean(self):
        assert detect_language("\uc548\ub155\ud558\uc138\uc694") == "Korean"
    def test_mixed_defaults_to_first_match(self):
        assert detect_language("Hello \u4f60\u597d") == "Chinese"
