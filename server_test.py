"""Tests for audio output LRU cache (Issue #17)."""
import pytest
from collections import OrderedDict
from unittest.mock import patch
import hashlib

from server import (
    _audio_cache_key,
    _get_audio_cache,
    _set_audio_cache,
    _audio_cache,
    _AUDIO_CACHE_MAX,
)


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear the audio cache before and after each test."""
    _audio_cache.clear()
    yield
    _audio_cache.clear()


class TestAudioCacheKey:
    def test_deterministic(self):
        """Same inputs produce the same cache key."""
        k1 = _audio_cache_key("hello", "vivian", 1.0, "wav", "English", "")
        k2 = _audio_cache_key("hello", "vivian", 1.0, "wav", "English", "")
        assert k1 == k2

    def test_different_text_produces_different_key(self):
        k1 = _audio_cache_key("hello", "vivian", 1.0, "wav", "English", "")
        k2 = _audio_cache_key("world", "vivian", 1.0, "wav", "English", "")
        assert k1 != k2

    def test_different_voice_produces_different_key(self):
        k1 = _audio_cache_key("hello", "vivian", 1.0, "wav", "English", "")
        k2 = _audio_cache_key("hello", "ryan", 1.0, "wav", "English", "")
        assert k1 != k2

    def test_different_speed_produces_different_key(self):
        k1 = _audio_cache_key("hello", "vivian", 1.0, "wav", "English", "")
        k2 = _audio_cache_key("hello", "vivian", 1.5, "wav", "English", "")
        assert k1 != k2

    def test_different_format_produces_different_key(self):
        k1 = _audio_cache_key("hello", "vivian", 1.0, "wav", "English", "")
        k2 = _audio_cache_key("hello", "vivian", 1.0, "mp3", "English", "")
        assert k1 != k2

    def test_different_language_produces_different_key(self):
        k1 = _audio_cache_key("hello", "vivian", 1.0, "wav", "English", "")
        k2 = _audio_cache_key("hello", "vivian", 1.0, "wav", "Chinese", "")
        assert k1 != k2

    def test_different_instruct_produces_different_key(self):
        k1 = _audio_cache_key("hello", "vivian", 1.0, "wav", "English", "")
        k2 = _audio_cache_key("hello", "vivian", 1.0, "wav", "English", "speak slowly")
        assert k1 != k2

    def test_key_is_sha256_hex(self):
        key = _audio_cache_key("hello", "vivian", 1.0, "wav", "English", "")
        assert len(key) == 64
        int(key, 16)  # valid hex


class TestAudioCacheGetSet:
    def test_cache_miss_returns_none(self):
        assert _get_audio_cache("nonexistent") is None

    def test_cache_hit_returns_stored_data(self):
        _set_audio_cache("key1", b"audio_data", "audio/wav")
        result = _get_audio_cache("key1")
        assert result is not None
        assert result[0] == b"audio_data"
        assert result[1] == "audio/wav"

    def test_cache_hit_moves_to_end(self):
        _set_audio_cache("key1", b"data1", "audio/wav")
        _set_audio_cache("key2", b"data2", "audio/wav")
        _get_audio_cache("key1")
        keys = list(_audio_cache.keys())
        assert keys[-1] == "key1"

    def test_cache_eviction_when_full(self):
        """Verify LRU eviction when cache reaches max capacity."""
        for i in range(_AUDIO_CACHE_MAX):
            _set_audio_cache(f"key{i}", f"data{i}".encode(), "audio/wav")
        assert len(_audio_cache) == _AUDIO_CACHE_MAX

        _set_audio_cache("new_key", b"new_data", "audio/wav")
        assert len(_audio_cache) == _AUDIO_CACHE_MAX
        assert "key0" not in _audio_cache
        assert "new_key" in _audio_cache

    def test_cache_disabled_when_max_zero(self):
        """Verify AUDIO_CACHE_MAX=0 disables caching."""
        with patch("server._AUDIO_CACHE_MAX", 0):
            _set_audio_cache("key1", b"data", "audio/wav")
            assert len(_audio_cache) == 0
            assert _get_audio_cache("key1") is None


class TestCacheClearEndpoint:
    @pytest.mark.asyncio
    async def test_clear_cache_returns_count(self):
        from server import clear_cache as _clear_cache_endpoint
        _set_audio_cache("k1", b"d1", "audio/wav")
        _set_audio_cache("k2", b"d2", "audio/wav")
        result = await _clear_cache_endpoint()
        assert result == {"cleared": 2}
        assert len(_audio_cache) == 0

    @pytest.mark.asyncio
    async def test_clear_empty_cache(self):
        from server import clear_cache as _clear_cache_endpoint
        result = await _clear_cache_endpoint()
        assert result == {"cleared": 0}


class TestHealthEndpointCacheInfo:
    def test_health_includes_cache_fields(self):
        from fastapi.testclient import TestClient
        from server import app
        client = TestClient(app)
        resp = client.get("/health")
        data = resp.json()
        assert "audio_cache_size" in data
        assert "audio_cache_max" in data
        assert data["audio_cache_max"] == _AUDIO_CACHE_MAX
