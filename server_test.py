"""Tests for server.py — GPU optimization flags and inference configuration."""
import importlib
import sys
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
    """Issue #5: TF32 matmul and cudnn flags should be enabled on CUDA hardware.

    Uses mock-based reimport so tests work on non-CUDA CI machines:
    reset flags to False, mock cuda.is_available -> True, reimport server,
    assert flags became True.
    """

    def _reimport_server_with_cuda(self):
        """Reset GPU flags, mock CUDA as available, reimport server module."""
        saved = (
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
            torch.backends.cudnn.benchmark,
        )
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        sys.modules.pop("server", None)
        with patch("torch.cuda.is_available", return_value=True), \
             patch.dict("sys.modules", _mock_modules):
            importlib.import_module("server")
        return saved

    def _restore_flags(self, saved):
        """Restore original GPU flags after test."""
        torch.backends.cuda.matmul.allow_tf32 = saved[0]
        torch.backends.cudnn.allow_tf32 = saved[1]
        torch.backends.cudnn.benchmark = saved[2]

    def test_tf32_matmul_enabled(self):
        """torch.backends.cuda.matmul.allow_tf32 should be True after server import."""
        saved = self._reimport_server_with_cuda()
        try:
            assert torch.backends.cuda.matmul.allow_tf32 is True
        finally:
            self._restore_flags(saved)

    def test_tf32_cudnn_enabled(self):
        """torch.backends.cudnn.allow_tf32 should be True after server import."""
        saved = self._reimport_server_with_cuda()
        try:
            assert torch.backends.cudnn.allow_tf32 is True
        finally:
            self._restore_flags(saved)

    def test_cudnn_benchmark_enabled(self):
        """torch.backends.cudnn.benchmark should be True after server import."""
        saved = self._reimport_server_with_cuda()
        try:
            assert torch.backends.cudnn.benchmark is True
        finally:
            self._restore_flags(saved)


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
        assert detect_language("\u4f60\u597d\u4e16\u754c") == "Chinese"

    def test_japanese(self):
        assert detect_language("\u3053\u3093\u306b\u3061\u306f") == "Japanese"

    def test_korean(self):
        assert detect_language("\uc548\ub155\ud558\uc138\uc694") == "Korean"

    def test_mixed_defaults_to_first_match(self):
        assert detect_language("Hello \u4f60\u597d") == "Chinese"
