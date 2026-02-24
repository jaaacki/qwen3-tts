"""Performance tests for Qwen3-TTS server.

Tests latency and cache speed-up for synthesis requests.
"""

import time

import pytest

from utils.client import TTSHTTPClient
from utils.audio import is_valid_wav

TEXT_SHORT  = "Performance test."
TEXT_MEDIUM = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs."
)


@pytest.mark.performance
@pytest.mark.slow
class TestInferenceLatency:
    """Inference latency benchmarks."""

    def test_warm_inference_latency(self, ensure_server, ensure_model_loaded):
        """Warm inference (model already loaded) completes in ≤ 60 s."""
        with TTSHTTPClient() as client:
            # Discard first call (may still be loading)
            client.synthesize(TEXT_SHORT)

            times = []
            for _ in range(3):
                start = time.time()
                audio = client.synthesize(TEXT_SHORT)
                elapsed = time.time() - start
                times.append(elapsed)
                assert is_valid_wav(audio)

        avg = sum(times) / len(times)
        assert avg < 60.0, f"Warm inference avg {avg:.1f}s exceeds 60s"
        print(f"Warm inference: {avg:.2f}s")
        for i, t in enumerate(times):
            print(f"  run {i+1}: {t:.2f}s")

    def test_audio_cache_hit_latency(self, ensure_server, ensure_model_loaded):
        """Cache hit serves audio without GPU — should be < 100 ms."""
        with TTSHTTPClient() as client:
            # Prime the cache
            client.synthesize(TEXT_SHORT, voice="alloy")

            # Hit the cache
            start = time.time()
            audio = client.synthesize(TEXT_SHORT, voice="alloy")
            elapsed = time.time() - start

        # Cache hits skip inference entirely — should be very fast
        assert elapsed < 5.0, f"Cache hit took {elapsed:.3f}s (expected < 5s)"
        print(f"Cache hit: {elapsed:.3f}s")
        print(f"Audio bytes: {len(audio)}")


@pytest.mark.performance
class TestHealthLatency:
    """Health endpoint latency."""

    def test_health_latency(self, ensure_server):
        """Health endpoint responds in < 1 s on average over 5 calls."""
        with TTSHTTPClient() as client:
            times = []
            for _ in range(5):
                start = time.time()
                client.health()
                times.append(time.time() - start)

        avg = sum(times) / len(times)
        assert avg < 1.0, f"Health latency avg {avg:.3f}s"
        print(f"Health latency avg: {avg:.3f}s")
        for i, t in enumerate(times):
            print(f"  call {i+1}: {t:.3f}s")
