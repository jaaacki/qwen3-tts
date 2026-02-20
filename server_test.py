"""Tests for server.py — Phase 1 Real-Time features."""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock


# Mock heavy imports before importing server
_mock_modules = {
    "qwen_tts": MagicMock(),
    "torch": MagicMock(),
    "torch.cuda": MagicMock(),
    "torch.backends": MagicMock(),
    "torch.backends.cudnn": MagicMock(),
    "torch.backends.cuda": MagicMock(),
}

_mock_torch = _mock_modules["torch"]
_mock_torch.cuda.is_available.return_value = False
_mock_torch.inference_mode.return_value.__enter__ = MagicMock()
_mock_torch.inference_mode.return_value.__exit__ = MagicMock()

with patch.dict("sys.modules", _mock_modules):
    from server import (
        app, resolve_voice, detect_language, convert_audio_format,
        _adaptive_max_tokens,
    )

from fastapi.testclient import TestClient

client = TestClient(app)


# --- Issue #2: Adaptive max_new_tokens tests ---


class TestAdaptiveMaxTokens:
    """Tests for _adaptive_max_tokens (Issue #2)."""

    def test_short_text_clamps_to_minimum(self):
        """Very short text should clamp to 128 (minimum)."""
        assert _adaptive_max_tokens("hello world") == 128

    def test_single_word_clamps_to_minimum(self):
        assert _adaptive_max_tokens("hello") == 128

    def test_medium_text_scales_linearly(self):
        """20 words * 8 = 160 tokens."""
        text = " ".join(["word"] * 20)
        assert _adaptive_max_tokens(text) == 160

    def test_long_text_clamps_to_maximum(self):
        """300 words * 8 = 2400, clamped to 2048."""
        text = " ".join(["word"] * 300)
        assert _adaptive_max_tokens(text) == 2048

    def test_exact_boundary_128(self):
        """16 words * 8 = 128, exactly at minimum."""
        text = " ".join(["word"] * 16)
        assert _adaptive_max_tokens(text) == 128

    def test_just_above_minimum(self):
        """17 words * 8 = 136, just above minimum."""
        text = " ".join(["word"] * 17)
        assert _adaptive_max_tokens(text) == 136

    def test_exact_boundary_2048(self):
        """256 words * 8 = 2048, exactly at maximum."""
        text = " ".join(["word"] * 256)
        assert _adaptive_max_tokens(text) == 2048

    def test_empty_string(self):
        """Empty string should return 128 (minimum)."""
        assert _adaptive_max_tokens("") == 128


# --- Existing utility function tests ---


class TestResolveVoice:
    """Tests for voice resolution."""

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
    """Tests for language detection."""

    def test_english(self):
        assert detect_language("Hello world") == "English"

    def test_chinese(self):
        assert detect_language("你好世界") == "Chinese"

    def test_japanese(self):
        assert detect_language("こんにちは") == "Japanese"

    def test_korean(self):
        assert detect_language("안녕하세요") == "Korean"

    def test_mixed_defaults_to_first_match(self):
        assert detect_language("Hello 你好") == "Chinese"
