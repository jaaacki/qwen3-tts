"""Tests for server.py - Phase 2 Speed & Quality features."""
import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

# Mock heavy imports before importing server
_mock_modules = {
    "qwen_tts": MagicMock(),
}

with patch.dict("sys.modules", _mock_modules):
    from server import _trim_silence, _normalize_text, _expand_currency, resolve_voice, detect_language
    import server


# --- Issue #11: VAD silence trimming tests ---

class TestTrimSilence:
    def test_trim_silence_removes_leading_silence(self):
        sr = 24000
        silence = np.zeros(sr)
        speech = np.random.randn(sr).astype(np.float32) * 0.5
        audio = np.concatenate([silence, speech])
        with patch.object(server, "VAD_TRIM", True):
            result = _trim_silence(audio, sr)
        assert len(result) < len(audio)
        assert len(result) >= len(speech)

    def test_trim_silence_removes_trailing_silence(self):
        sr = 24000
        speech = np.random.randn(sr).astype(np.float32) * 0.5
        silence = np.zeros(sr)
        audio = np.concatenate([speech, silence])
        with patch.object(server, "VAD_TRIM", True):
            result = _trim_silence(audio, sr)
        assert len(result) < len(audio)
        assert len(result) >= len(speech)

    def test_trim_silence_preserves_content(self):
        sr = 24000
        speech = np.random.randn(sr).astype(np.float32) * 0.5
        with patch.object(server, "VAD_TRIM", True):
            result = _trim_silence(speech.copy(), sr)
        assert len(result) >= len(speech) - int(0.05 * sr)

    def test_trim_silence_all_silence_returns_original(self):
        sr = 24000
        audio = np.zeros(sr, dtype=np.float32)
        with patch.object(server, "VAD_TRIM", True):
            result = _trim_silence(audio, sr)
        np.testing.assert_array_equal(result, audio)

    def test_trim_silence_disabled_returns_original(self):
        sr = 24000
        silence = np.zeros(sr)
        speech = np.random.randn(sr).astype(np.float32) * 0.5
        audio = np.concatenate([silence, speech, silence])
        with patch.object(server, "VAD_TRIM", False):
            result = _trim_silence(audio, sr)
        np.testing.assert_array_equal(result, audio)

    def test_trim_silence_adds_padding(self):
        sr = 24000
        pad_samples = int(0.05 * sr)
        silence = np.zeros(sr * 2)
        spike = np.zeros(100, dtype=np.float32)
        spike[50] = 1.0
        audio = np.concatenate([silence, spike, silence])
        with patch.object(server, "VAD_TRIM", True):
            result = _trim_silence(audio, sr)
        assert len(result) >= len(spike)
        assert len(result) <= len(spike) + 2 * pad_samples


# --- Issue #12: Text normalization tests ---

class TestNormalizeTextCurrency:
    def test_dollar_amount(self):
        with patch.object(server, "TEXT_NORMALIZE", True):
            assert _normalize_text("Price is $5.00") == "Price is 5 dollars"

    def test_dollar_with_cents(self):
        with patch.object(server, "TEXT_NORMALIZE", True):
            assert _normalize_text("Cost $10.50") == "Cost 10 dollars and 50 cents"

    def test_dollar_no_cents(self):
        with patch.object(server, "TEXT_NORMALIZE", True):
            assert _normalize_text("$100") == "100 dollars"

    def test_euro(self):
        with patch.object(server, "TEXT_NORMALIZE", True):
            result = _normalize_text("€50")
            assert result == "50 euros"

    def test_pound(self):
        with patch.object(server, "TEXT_NORMALIZE", True):
            result = _normalize_text("£20")
            assert result == "20 pounds"


class TestNormalizeTextAbbreviations:
    def test_doctor(self):
        with patch.object(server, "TEXT_NORMALIZE", True):
            assert _normalize_text("Dr. Smith") == "Doctor Smith"

    def test_mister(self):
        with patch.object(server, "TEXT_NORMALIZE", True):
            assert _normalize_text("Mr. Jones") == "Mister Jones"

    def test_professor(self):
        with patch.object(server, "TEXT_NORMALIZE", True):
            assert _normalize_text("Prof. Lee") == "Professor Lee"


class TestNormalizeTextCommas:
    def test_comma_in_number(self):
        with patch.object(server, "TEXT_NORMALIZE", True):
            assert _normalize_text("1,000 items") == "1000 items"

    def test_large_number(self):
        with patch.object(server, "TEXT_NORMALIZE", True):
            result = _normalize_text("1,000,000")
            assert "," not in result


class TestNormalizeTextDisabled:
    def test_disabled_passthrough(self):
        with patch.object(server, "TEXT_NORMALIZE", False):
            assert _normalize_text("$5.00 Dr. Smith") == "$5.00 Dr. Smith"


class TestExpandCurrency:
    def test_whole_dollars(self):
        assert _expand_currency("5", "dollars") == "5 dollars"

    def test_dollars_with_cents(self):
        assert _expand_currency("5.50", "dollars") == "5 dollars and 50 cents"

    def test_dollars_zero_cents(self):
        assert _expand_currency("5.00", "dollars") == "5 dollars"


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
    def test_default_voice_when_none(self):
        assert resolve_voice(None) == "vivian"
    def test_openai_alias(self):
        assert resolve_voice("alloy") == "ryan"
    def test_case_insensitive(self):
        assert resolve_voice("VIVIAN") == "vivian"


class TestDetectLanguage:
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
