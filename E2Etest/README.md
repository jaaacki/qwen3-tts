# E2E Test Suite for Qwen3-TTS

End-to-end tests for the Qwen3-TTS server.

## Quick Start

```bash
# Install test dependencies
pip install -r E2Etest/requirements.txt

# Run all tests (requires server running on localhost:8101)
pytest E2Etest/ -v

# Or use the helper script
./E2Etest/run_tests.sh
```

## Test Structure

```
E2Etest/
├── conftest.py              # Pytest fixtures and markdown report generator
├── requirements.txt         # Test dependencies
├── pytest.ini               # Pytest configuration and markers
├── run_tests.sh             # Helper script with server management
├── test_api_http.py         # /health, /v1/audio/speech, /cache/clear
├── test_voices.py           # Built-in voices and OpenAI aliases
├── test_formats.py          # Output formats (wav/mp3/flac/ogg/opus)
├── test_streaming.py        # SSE /stream and PCM /stream/pcm
├── test_websocket.py        # WS /v1/audio/speech/ws
├── test_clone.py            # /v1/audio/speech/clone (voice cloning)
├── test_performance.py      # Latency benchmarks
├── test_integration.py      # Cache, speed, gen params, priority queue
├── utils/
│   ├── client.py            # TTSHTTPClient, TTSWebSocketClient
│   └── audio.py             # WAV generation and validation helpers
└── data/
    └── audio/               # Auto-generated ref audio for clone tests
```

## Running Tests

### With Server Running

```bash
# Start server
docker compose up -d

# Smoke tests only (fast — no model inference needed for health/error tests)
pytest E2Etest/ -v -m smoke

# Skip slow tests
pytest E2Etest/ -v -m "not slow"

# Specific test files
pytest E2Etest/test_api_http.py -v
pytest E2Etest/test_voices.py -v
pytest E2Etest/test_formats.py -v
pytest E2Etest/test_streaming.py -v
pytest E2Etest/test_websocket.py -v
pytest E2Etest/test_clone.py -v
pytest E2Etest/test_performance.py -v
pytest E2Etest/test_integration.py -v

# Full run with server auto-start/stop
./E2Etest/run_tests.sh --with-server
```

### Environment Variables

```bash
# Custom server URL (default: http://localhost:8101)
E2E_BASE_URL=http://myhost:8101 pytest E2Etest/

# Custom WebSocket URL
E2E_WS_URL=ws://myhost:8101/v1/audio/speech/ws pytest E2Etest/
```

## Test Categories

### Smoke Tests (`-m smoke`)
- Health endpoint returns 200 with expected fields
- Basic synthesis returns valid WAV
- Voice names and aliases are listed

### HTTP Tests (`test_api_http.py`)
- Health endpoint fields and cache info
- Basic synthesis response validation
- Empty/whitespace input → 400
- Cache clear endpoint

### Voice Tests (`test_voices.py`)
- All 9 native Qwen speaker names (`vivian`, `serena`, `uncle_fu`, `dylan`, `eric`, `ryan`, `aiden`, `ono_anna`, `sohee`)
- All 6 OpenAI aliases (`alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`)
- Unknown voice passthrough (no 4xx)
- Two different voices produce different audio

### Format Tests (`test_formats.py`)
- wav / mp3 / flac / ogg / opus each return non-empty bytes
- Content-Type header matches format
- WAV output has valid RIFF/WAVE header

### Streaming Tests (`test_streaming.py`)
- SSE stream returns events with base64 audio data
- SSE stream ends with `[DONE]` marker
- PCM stream returns binary bytes
- PCM stream has `X-PCM-Sample-Rate` / `X-PCM-Bit-Depth` / `X-PCM-Channels` headers
- PCM bytes have even length (int16)

### WebSocket Tests (`test_websocket.py`)
- Send JSON → receive binary PCM chunks + `{"event": "done"}`
- Multi-sentence input → multiple chunks
- Empty input → `{"event": "error"}`
- Multiple sequential requests on same connection
- `temperature` and `top_p` accepted

### Clone Tests (`test_clone.py`)
- Multipart upload returns valid WAV
- `ref_text` and `language` params accepted
- Missing file or empty input → 4xx
- Repeat call with same ref audio is faster (voice prompt cache)
- `voice_cache_size` in /health increases after clone call

### Performance Tests (`test_performance.py`)
- Warm inference avg ≤ 60 s
- Audio cache hit < 5 s
- Health endpoint avg < 1 s

### Integration Tests (`test_integration.py`)
- `audio_cache_size` increases after synthesis
- Cache clear empties audio cache to 0
- `speed=2.0` produces shorter audio than `speed=1.0`
- `temperature` and `top_p` accepted without error
- WebSocket completes while REST request is running (priority queue)

## Reference Audio

Reference WAV files for clone tests are **auto-generated** on first run via a
session-scoped `autouse` fixture. No manual setup needed.

Files generated:
- `data/audio/ref_3s.wav` — 3-second 220 Hz sine wave, 24 kHz
- `data/audio/ref_5s.wav` — 5-second 220 Hz sine wave, 24 kHz

## Troubleshooting

### Server not responding
- Check health: `curl http://localhost:8101/health`
- Verify port mapping in `compose.yaml` (container 8000 → host 8101)

### Model not loading
- First synthesis request triggers model load (~10-30s, longer on first run with download)
- Tests with `ensure_model_loaded` fixture wait for load automatically

### Clone tests failing
- Clone requires GPU inference — ensure CUDA is available
- Ref audio is generated automatically; if missing: `python -c "import sys; sys.path.insert(0, 'E2Etest'); from utils.audio import generate_ref_audio; from pathlib import Path; generate_ref_audio(Path('E2Etest/data/audio'), force=True)"`

### Format tests skipped
- `opus` format requires `pydub`/`ffmpeg` in the server container
- `mp3` requires `pydub`/`ffmpeg`; if not installed the server returns 500
