"""Tests for server.py - Phase 2 Speed & Quality features."""
import sys
import io
import hashlib
import pytest
import torch
import numpy as np
import soundfile as sf
from unittest.mock import patch, MagicMock

# Mock heavy imports before importing server
_mock_modules = {
    "qwen_tts": MagicMock(),
}

with patch.dict("sys.modules", _mock_modules):
    from server import (
        _trim_silence, _normalize_text, _expand_currency,
        _detect_language_unicode, _get_langdetect, detect_language,
        _adjust_speed, resolve_voice, _LANG_MAP,
        _get_cached_ref_audio,
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

    def test_mixed_starts_with_chinese(self):
        assert _detect_language_unicode("你好 hello") == "Chinese"


class TestLangMap:
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

    def test_maps_all_supported_languages(self):
        for iso, name in _LANG_MAP.items():
            mock_detector = MagicMock(return_value={"lang": iso, "score": 0.9})
            with patch.object(server, "_get_langdetect", return_value=mock_detector):
                result = detect_language("test")
            assert result == name, f"Failed for {iso} -> {name}"


# --- Issue #14: pyrubberband speed adjustment tests ---

class TestAdjustSpeedWithPyrubberband:
    def test_speed_1_returns_unchanged(self):
        audio = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        result = _adjust_speed(audio, 24000, 1.0)
        np.testing.assert_array_equal(result, audio)

    def test_speed_faster_calls_pyrubberband(self):
        audio = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        expected = np.array([0.1, 0.3, 0.5], dtype=np.float32)
        mock_rb = MagicMock()
        mock_rb.time_stretch.return_value = expected
        with patch.object(server, "_pyrubberband", mock_rb):
            result = _adjust_speed(audio, 24000, 1.5)
        mock_rb.time_stretch.assert_called_once_with(audio, 24000, 1.5)
        np.testing.assert_array_equal(result, expected)

    def test_speed_slower_calls_pyrubberband(self):
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        expected = np.array([0.1, 0.15, 0.2, 0.25, 0.3], dtype=np.float32)
        mock_rb = MagicMock()
        mock_rb.time_stretch.return_value = expected
        with patch.object(server, "_pyrubberband", mock_rb):
            result = _adjust_speed(audio, 24000, 0.75)
        mock_rb.time_stretch.assert_called_once_with(audio, 24000, 0.75)
        np.testing.assert_array_equal(result, expected)

    def test_preserves_sample_rate(self):
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        mock_rb = MagicMock()
        mock_rb.time_stretch.return_value = audio
        with patch.object(server, "_pyrubberband", mock_rb):
            _adjust_speed(audio, 48000, 2.0)
        mock_rb.time_stretch.assert_called_once_with(audio, 48000, 2.0)


class TestAdjustSpeedFallback:
    def test_speed_1_returns_unchanged(self):
        audio = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        with patch.object(server, "_pyrubberband", None):
            result = _adjust_speed(audio, 24000, 1.0)
        np.testing.assert_array_equal(result, audio)

    def test_speed_faster_calls_scipy_resample(self):
        audio = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
        expected = np.array([0.1, 0.3, 0.5, 0.6], dtype=np.float32)
        mock_scipy = MagicMock()
        mock_scipy.resample.return_value = expected
        with patch.object(server, "_pyrubberband", None), \
             patch.object(server, "scipy_signal", mock_scipy):
            result = _adjust_speed(audio, 24000, 1.5)
        mock_scipy.resample.assert_called_once_with(audio, 4)
        np.testing.assert_array_equal(result, expected)

    def test_zero_length_returns_original(self):
        audio = np.array([0.1], dtype=np.float32)
        with patch.object(server, "_pyrubberband", None):
            result = _adjust_speed(audio, 24000, 100.0)
        np.testing.assert_array_equal(result, audio)


# --- Issue #5: TF32 matmul mode tests ---

class TestTF32Flags:
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


# --- Issue #15: Voice prompt cache tests ---

def _make_wav_bytes(samples=None, sr=24000, channels=1):
    """Helper to create valid WAV bytes for testing."""
    if samples is None:
        samples = np.random.randn(sr).astype(np.float32) * 0.1
    if channels > 1:
        samples = np.column_stack([samples] * channels)
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV")
    return buf.getvalue(), samples, sr


class TestVoiceCacheBasics:
    def setup_method(self):
        server._voice_cache.clear()
        server._voice_cache_hits = 0

    def test_first_call_populates_cache(self):
        wav_bytes, _, sr = _make_wav_bytes()
        with patch.object(server, "VOICE_CACHE_MAX", 32):
            result_data, result_sr = _get_cached_ref_audio(wav_bytes)
        assert result_sr == sr
        assert len(server._voice_cache) == 1
        assert isinstance(result_data, np.ndarray)

    def test_second_call_returns_cached(self):
        wav_bytes, _, _ = _make_wav_bytes()
        with patch.object(server, "VOICE_CACHE_MAX", 32):
            first_data, first_sr = _get_cached_ref_audio(wav_bytes)
            hits_before = server._voice_cache_hits
            second_data, second_sr = _get_cached_ref_audio(wav_bytes)
        assert server._voice_cache_hits == hits_before + 1
        np.testing.assert_array_equal(first_data, second_data)
        assert first_sr == second_sr

    def test_different_audio_gets_different_cache_entries(self):
        wav1, _, _ = _make_wav_bytes(np.ones(1000, dtype=np.float32) * 0.1)
        wav2, _, _ = _make_wav_bytes(np.ones(1000, dtype=np.float32) * 0.2)
        with patch.object(server, "VOICE_CACHE_MAX", 32):
            _get_cached_ref_audio(wav1)
            _get_cached_ref_audio(wav2)
        assert len(server._voice_cache) == 2


class TestVoiceCacheLRU:
    def setup_method(self):
        server._voice_cache.clear()
        server._voice_cache_hits = 0

    def test_evicts_oldest_when_full(self):
        wavs = []
        for i in range(4):
            w, _, _ = _make_wav_bytes(np.ones(1000, dtype=np.float32) * (i + 1) * 0.1)
            wavs.append(w)
        with patch.object(server, "VOICE_CACHE_MAX", 2):
            for w in wavs:
                _get_cached_ref_audio(w)
        assert len(server._voice_cache) == 2
        # First two should have been evicted
        key0 = hashlib.sha256(wavs[0]).hexdigest()
        key1 = hashlib.sha256(wavs[1]).hexdigest()
        assert key0 not in server._voice_cache
        assert key1 not in server._voice_cache

    def test_access_promotes_entry(self):
        wavs = []
        for i in range(3):
            w, _, _ = _make_wav_bytes(np.ones(1000, dtype=np.float32) * (i + 1) * 0.1)
            wavs.append(w)
        with patch.object(server, "VOICE_CACHE_MAX", 2):
            _get_cached_ref_audio(wavs[0])  # cache: [0]
            _get_cached_ref_audio(wavs[1])  # cache: [0, 1]
            _get_cached_ref_audio(wavs[0])  # hit, promotes 0 -> cache: [1, 0]
            _get_cached_ref_audio(wavs[2])  # evicts 1 -> cache: [0, 2]
        key0 = hashlib.sha256(wavs[0]).hexdigest()
        key1 = hashlib.sha256(wavs[1]).hexdigest()
        key2 = hashlib.sha256(wavs[2]).hexdigest()
        assert key0 in server._voice_cache
        assert key1 not in server._voice_cache
        assert key2 in server._voice_cache


class TestVoiceCacheDisabled:
    def setup_method(self):
        server._voice_cache.clear()
        server._voice_cache_hits = 0

    def test_cache_disabled_skips_caching(self):
        wav_bytes, _, _ = _make_wav_bytes()
        with patch.object(server, "VOICE_CACHE_MAX", 0):
            _get_cached_ref_audio(wav_bytes)
        assert len(server._voice_cache) == 0

    def test_cache_disabled_still_returns_valid_audio(self):
        wav_bytes, _, sr = _make_wav_bytes()
        with patch.object(server, "VOICE_CACHE_MAX", 0):
            result_data, result_sr = _get_cached_ref_audio(wav_bytes)
        assert result_sr == sr
        assert isinstance(result_data, np.ndarray)
        assert len(result_data) > 0


class TestStereoToMono:
    def setup_method(self):
        server._voice_cache.clear()
        server._voice_cache_hits = 0

    def test_stereo_converted_to_mono(self):
        mono_samples = np.random.randn(1000).astype(np.float32) * 0.1
        wav_bytes, _, sr = _make_wav_bytes(mono_samples, channels=2)
        with patch.object(server, "VOICE_CACHE_MAX", 32):
            result_data, result_sr = _get_cached_ref_audio(wav_bytes)
        assert result_data.ndim == 1
        assert result_sr == sr

    def test_mono_stays_mono(self):
        wav_bytes, _, sr = _make_wav_bytes()
        with patch.object(server, "VOICE_CACHE_MAX", 32):
            result_data, result_sr = _get_cached_ref_audio(wav_bytes)
        assert result_data.ndim == 1


class TestCacheKeyHashing:
    def setup_method(self):
        server._voice_cache.clear()
        server._voice_cache_hits = 0

    def test_same_content_same_key(self):
        samples = np.ones(1000, dtype=np.float32) * 0.5
        wav1, _, _ = _make_wav_bytes(samples)
        wav2, _, _ = _make_wav_bytes(samples)
        assert wav1 == wav2  # same input -> same bytes
        with patch.object(server, "VOICE_CACHE_MAX", 32):
            _get_cached_ref_audio(wav1)
            _get_cached_ref_audio(wav2)
        assert len(server._voice_cache) == 1
        assert server._voice_cache_hits == 1


# --- Issue #16: GPU memory pool pre-allocation tests ---

class TestGpuPoolPreAllocation:
    """Verify GPU memory pool pre-warming code exists in _load_model_sync."""

    def test_load_model_contains_pool_prewarm(self):
        import inspect
        source = inspect.getsource(server._load_model_sync)
        assert "Pre-warming CUDA memory pool" in source
        assert "torch.empty" in source
        assert "dtype=torch.bfloat16" in source

    def test_dummy_tensor_size_is_128mb(self):
        import inspect
        source = inspect.getsource(server._load_model_sync)
        assert "64 * 1024 * 1024" in source

    def test_pool_prewarm_has_exception_handling(self):
        import inspect
        source = inspect.getsource(server._load_model_sync)
        idx = source.find("Pre-warming CUDA memory pool")
        assert idx > 0
        section = source[idx - 200:idx + 500]
        assert "try:" in section
        assert "except Exception" in section

    def test_pool_prewarm_after_warmup(self):
        import inspect
        source = inspect.getsource(server._load_model_sync)
        warmup_idx = source.find("Warming up GPU with multi-length synthesis")
        pool_idx = source.find("Pre-warming CUDA memory pool")
        assert warmup_idx > 0
        assert pool_idx > 0
        assert pool_idx > warmup_idx

    def test_dummy_tensor_is_deleted(self):
        import inspect
        source = inspect.getsource(server._load_model_sync)
        pool_section = source[source.find("Pre-warming CUDA memory pool"):]
        assert "del dummy" in pool_section
