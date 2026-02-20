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
        app, resolve_voice, detect_language, _split_sentences,
    )

from fastapi.testclient import TestClient

client = TestClient(app)


# --- Issue #3: Sentence splitting tests ---


class TestSplitSentences:
    """Tests for _split_sentences (Issue #3)."""

    def test_single_sentence(self):
        assert _split_sentences("Hello world.") == ["Hello world."]

    def test_multiple_sentences(self):
        result = _split_sentences("First sentence. Second sentence. Third one.")
        assert len(result) == 3
        assert result[0] == "First sentence."
        assert result[1] == "Second sentence."
        assert result[2] == "Third one."

    def test_question_and_exclamation(self):
        result = _split_sentences("Is it working? Yes it is! Great.")
        assert len(result) == 3

    def test_abbreviation_awareness(self):
        """Should not split on Dr. or Mr. abbreviations."""
        result = _split_sentences("Dr. Smith called Mr. Jones today.")
        assert len(result) == 1

    def test_empty_string(self):
        assert _split_sentences("") == []

    def test_whitespace_only(self):
        assert _split_sentences("   ") == []

    def test_no_sentence_ending(self):
        """Text without sentence-ending punctuation returns as single item."""
        result = _split_sentences("Hello world")
        assert result == ["Hello world"]

    def test_strips_whitespace(self):
        result = _split_sentences("  First.   Second.  ")
        assert all(s == s.strip() for s in result)


# --- Existing utility function tests ---


class TestResolveVoice:
    def test_default_voice_when_none(self):
        assert resolve_voice(None) == "vivian"

    def test_openai_alias(self):
        assert resolve_voice("alloy") == "ryan"

    def test_unknown_voice_passthrough(self):
        assert resolve_voice("custom_voice") == "custom_voice"


class TestDetectLanguage:
    def test_english(self):
        assert detect_language("Hello world") == "English"

    def test_chinese(self):
        assert detect_language("\u4f60\u597d\u4e16\u754c") == "Chinese"

    def test_japanese(self):
        assert detect_language("\u3053\u3093\u306b\u3061\u306f") == "Japanese"

    def test_korean(self):
        assert detect_language("\uc548\ub155\ud558\uc138\uc694") == "Korean"
