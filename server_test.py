"""Tests for server.py — GPU optimization flags and inference configuration."""
import pytest
import torch
from unittest.mock import patch, MagicMock

# Mock heavy imports before importing server
_mock_modules = {
    "qwen_tts": MagicMock(),
}

with patch.dict("sys.modules", _mock_modules):
    from server import resolve_voice, detect_language


# --- Issue #5: TF32 matmul mode tests ---


class TestTF32Flags:
    """Issue #5: TF32 matmul and cudnn flags should be enabled on CUDA hardware."""

    def test_tf32_matmul_enabled(self):
        """torch.backends.cuda.matmul.allow_tf32 should be True after server import."""
        if torch.cuda.is_available():
            assert torch.backends.cuda.matmul.allow_tf32 is True

    def test_tf32_cudnn_enabled(self):
        """torch.backends.cudnn.allow_tf32 should be True after server import."""
        if torch.cuda.is_available():
            assert torch.backends.cudnn.allow_tf32 is True

    def test_cudnn_benchmark_enabled(self):
        """torch.backends.cudnn.benchmark should be True after server import."""
        if torch.cuda.is_available():
            assert torch.backends.cudnn.benchmark is True


# --- Baseline utility tests ---


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
