"""Streaming endpoint tests for Qwen3-TTS server.

Tests:
- POST /v1/audio/speech/stream  — Server-Sent Events (base64 PCM per sentence)
- POST /v1/audio/speech/stream/pcm — raw int16 PCM stream
"""

import base64

import pytest

from utils.client import TTSHTTPClient

TEXT = "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs."


class TestSSEStream:
    """Tests for POST /v1/audio/speech/stream (SSE)."""

    @pytest.mark.slow
    def test_sse_stream_returns_events(self, ensure_server, ensure_model_loaded):
        """Streaming endpoint returns at least one SSE event."""
        with TTSHTTPClient() as client:
            events = client.synthesize_stream(TEXT)
        assert len(events) > 0

    @pytest.mark.slow
    def test_sse_stream_has_done_marker(self, ensure_server, ensure_model_loaded):
        """SSE stream ends with a [DONE] or done event."""
        with TTSHTTPClient() as client:
            events = client.synthesize_stream(TEXT)
        # The done marker is sent as data: [DONE]  (stored as {"raw": "[DONE]"})
        raw_strs = [e.get("raw", "") for e in events]
        done_events = [e for e in events if e.get("event") == "done" or e.get("raw") == "[DONE]"]
        assert len(done_events) > 0, f"No done marker found in events: {events[:5]}"

    @pytest.mark.slow
    def test_sse_events_contain_audio(self, ensure_server, ensure_model_loaded):
        """SSE events contain non-empty audio data (base64-encoded PCM)."""
        with TTSHTTPClient() as client:
            events = client.synthesize_stream(TEXT)

        audio_events = [e for e in events if "raw" in e and e["raw"] != "[DONE]"]
        assert len(audio_events) > 0, "No audio events in SSE stream"

        for ev in audio_events:
            raw_b64 = ev["raw"]
            assert isinstance(raw_b64, str)
            decoded = base64.b64decode(raw_b64)
            assert len(decoded) > 0


class TestPCMStream:
    """Tests for POST /v1/audio/speech/stream/pcm."""

    @pytest.mark.slow
    def test_pcm_stream_returns_bytes(self, ensure_server, ensure_model_loaded):
        """PCM stream endpoint returns non-empty bytes."""
        with TTSHTTPClient() as client:
            pcm, _ = client.synthesize_stream_pcm(TEXT)
        assert isinstance(pcm, bytes)
        assert len(pcm) > 0

    @pytest.mark.slow
    def test_pcm_stream_headers(self, ensure_server, ensure_model_loaded):
        """PCM stream response includes sample-rate and bit-depth headers."""
        with TTSHTTPClient() as client:
            _, headers = client.synthesize_stream_pcm(TEXT)

        # Headers may be lower-cased by httpx
        header_keys = {k.lower() for k in headers}
        assert "x-pcm-sample-rate" in header_keys, f"Missing X-PCM-Sample-Rate; headers: {list(headers)}"
        assert "x-pcm-bit-depth" in header_keys,   f"Missing X-PCM-Bit-Depth; headers: {list(headers)}"
        assert "x-pcm-channels" in header_keys,    f"Missing X-PCM-Channels; headers: {list(headers)}"

    @pytest.mark.slow
    def test_pcm_data_length_is_even(self, ensure_server, ensure_model_loaded):
        """PCM bytes are int16 — total length must be even."""
        with TTSHTTPClient() as client:
            pcm, _ = client.synthesize_stream_pcm("Short sentence.")
        assert len(pcm) % 2 == 0, f"PCM byte length {len(pcm)} is not even (not int16)"
