"""Tests for server.py - Phase 2 Speed & Quality features."""
import sys
import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

# Mock heavy imports before importing server
_mock_modules = {
    "qwen_tts": MagicMock(),
}

with patch.dict("sys.modules", _mock_modules):
    from server import (
        _trim_silence, _normalize_text, _expand_currency,
        _detect_language_unicode, _get_langdetect, detect_language,
        resolve_voice, _LANG_MAP,
    )
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


# --- Issue #13: fasttext language detection tests ---

class TestDetectLanguageUnicode:
    """Test the Unicode fallback detector (_detect_language_unicode)."""

    def test_chinese_characters(self):
        assert _detect_language_unicode("你好世界") == "Chinese"

    def test_japanese_hiragana(self):
        assert _detect_language_unicode("こんにちは") == "Japanese"

    def test_japanese_katakana(self):
        assert _detect_language_unicode("カタカナ") == "Japanese"

    def test_korean(self):
        assert _detect_language_unicode("안녕하세요") == "Korean"

    def test_english_default(self):
        assert _detect_language_unicode("Hello world") == "English"

    def test_empty_string(self):
        assert _detect_language_unicode("") == "English"

    def test_numbers_only(self):
        assert _detect_language_unicode("12345") == "English"

    def test_mixed_starts_with_chinese(self):
        assert _detect_language_unicode("你好 hello") == "Chinese"


class TestLangMap:
    """Test the ISO code to Qwen language name mapping."""

    def test_known_codes(self):
        assert _LANG_MAP["zh"] == "Chinese"
        assert _LANG_MAP["en"] == "English"
        assert _LANG_MAP["ja"] == "Japanese"
        assert _LANG_MAP["ko"] == "Korean"
        assert _LANG_MAP["fr"] == "French"
        assert _LANG_MAP["de"] == "German"
        assert _LANG_MAP["es"] == "Spanish"

    def test_all_ten_languages(self):
        assert len(_LANG_MAP) == 10


class TestGetLangdetect:
    """Test lazy-loading of the fasttext detector."""

    def test_returns_false_when_import_fails(self):
        server._langdetect_model = None
        with patch.dict(sys.modules, {"fasttext_langdetect": None}):
            with patch("builtins.__import__", side_effect=ImportError("no module")):
                result = _get_langdetect()
        assert result is False
        server._langdetect_model = None

    def test_caches_result(self):
        server._langdetect_model = "cached_value"
        result = _get_langdetect()
        assert result == "cached_value"
        server._langdetect_model = None


class TestDetectLanguageWithFasttext:
    """Test detect_language with mocked fasttext detector."""

    def test_uses_fasttext_when_available(self):
        mock_detector = MagicMock(return_value={"lang": "fr", "score": 0.99})
        server._langdetect_model = None
        with patch.object(server, "_get_langdetect", return_value=mock_detector):
            result = detect_language("Bonjour le monde")
        assert result == "French"
        mock_detector.assert_called_once_with("Bonjour le monde", low_memory=False)

    def test_maps_zh_to_chinese(self):
        mock_detector = MagicMock(return_value={"lang": "zh", "score": 0.95})
        with patch.object(server, "_get_langdetect", return_value=mock_detector):
            result = detect_language("你好")
        assert result == "Chinese"

    def test_maps_ja_to_japanese(self):
        mock_detector = MagicMock(return_value={"lang": "ja", "score": 0.95})
        with patch.object(server, "_get_langdetect", return_value=mock_detector):
            result = detect_language("こんにちは")
        assert result == "Japanese"

    def test_maps_en_to_english(self):
        mock_detector = MagicMock(return_value={"lang": "en", "score": 0.99})
        with patch.object(server, "_get_langdetect", return_value=mock_detector):
            result = detect_language("Hello world")
        assert result == "English"

    def test_unknown_lang_defaults_to_english(self):
        mock_detector = MagicMock(return_value={"lang": "xx", "score": 0.5})
        with patch.object(server, "_get_langdetect", return_value=mock_detector):
            result = detect_language("something")
        assert result == "English"

    def test_falls_back_on_exception(self):
        mock_detector = MagicMock(side_effect=RuntimeError("model error"))
        with patch.object(server, "_get_langdetect", return_value=mock_detector):
            result = detect_language("Hello world")
        assert result == "English"

    def test_falls_back_when_fasttext_unavailable(self):
        with patch.object(server, "_get_langdetect", return_value=False):
            result = detect_language("你好世界")
        assert result == "Chinese"

    def test_falls_back_unicode_korean(self):
        with patch.object(server, "_get_langdetect", return_value=False):
            result = detect_language("안녕하세요")
        assert result == "Korean"

    def test_maps_all_supported_languages(self):
        for iso, name in _LANG_MAP.items():
            mock_detector = MagicMock(return_value={"lang": iso, "score": 0.9})
            with patch.object(server, "_get_langdetect", return_value=mock_detector):
                result = detect_language("test")
            assert result == name, f"Failed for {iso} -> {name}"


# --- Issue #5: TF32 matmul mode tests ---

class TestTF32Flags:
    """Issue #5: TF32 matmul and cudnn flags should be enabled on CUDA hardware."""

    def test_tf32_matmul_enabled(self):
        if torch.cuda.is_available():
            assert torch.backends.cuda.matmul.allow_tf32 is True

    def test_tf32_cudnn_enabled(self):
        if torch.cuda.is_available():
            assert torch.backends.cudnn.allow_tf32 is True

    def test_cudnn_benchmark_enabled(self):
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
