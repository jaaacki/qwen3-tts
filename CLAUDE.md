# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenAI-compatible TTS API server wrapping [Qwen3-TTS-0.6B](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice). Single-file FastAPI server (`server.py`) containerized with Docker and NVIDIA GPU acceleration. Designed for shared GPU environments with on-demand model loading/unloading.

## Build & Run

```bash
docker compose up -d --build        # Build image and start service
docker compose logs -f qwen3-tts    # View logs
docker compose down                 # Stop service
curl http://localhost:8101/health    # Health check
```

The service maps container port 8000 to host port 8101.

## Testing

`server_test.py` is the primary test suite — pure unit tests using pytest and mocks (no running server needed):

```bash
pytest server_test.py               # Run all unit tests
pytest server_test.py -k "trim"     # Run specific tests matching a pattern
pytest server_test.py -v            # Verbose output
```

Integration tests hit a live HTTP server:

```bash
python test_tts.py                  # Full integration suite
bash test_quick.sh                  # Quick smoke test, outputs test_output.wav
TTS_URL=http://host:port python test_tts.py  # Test against different host
```

`test_tts.py` uses a custom pass/fail reporter, not pytest. `server_test.py` mocks `qwen_tts` at import time to avoid loading the actual model.

E2E tests (require running server + GPU):

```bash
pytest E2Etest/ -v                  # Full E2E suite
pytest E2Etest/ -v -m smoke         # Smoke tests only (no inference)
pytest E2Etest/ -v -m "not slow"    # Skip slow tests
./E2Etest/run_tests.sh --with-server  # Auto-start/stop server
```

Note: Clone tests are skipped — the CustomVoice model does not support voice cloning.

## Architecture

**Single-file server** (`server.py`): All API logic, model management, and audio processing in one file. No separate modules.

**GPU lifecycle**: Model loads on first request (or at startup if `PRELOAD_MODEL=true`). An `_idle_watchdog` background task unloads the model after `IDLE_TIMEOUT` seconds of inactivity to free VRAM. `_model_lock` prevents concurrent load/unload races. `_infer_semaphore(1)` serializes GPU inference. Two thread pools: `_infer_executor` (1 thread, GPU work) and `_encode_executor` (2 threads, CPU audio encoding, runs in parallel with GPU).

**Request pipeline** for `/v1/audio/speech`:
1. Queue depth check → 503 early rejection if full
2. LRU audio cache lookup → skip GPU entirely on hit
3. `_ensure_model_loaded()` — load model if needed
4. `detect_language()` — fasttext if installed, else Unicode heuristic
5. `_normalize_text()` — expand numbers, currency, abbreviations
6. `_do_synthesize()` in `_infer_executor` under `_infer_semaphore`
7. `_adjust_speed()` — pyrubberband (pitch-preserving) or scipy fallback
8. `_encode_audio_async()` — format conversion in `_encode_executor`
9. Cache result and return

**Endpoints**:
- `POST /v1/audio/speech` — JSON body, OpenAI-compatible TTS (buffered, full audio)
- `POST /v1/audio/speech/stream` — SSE streaming, sentence-chunked, base64 PCM frames
- `POST /v1/audio/speech/stream/pcm` — raw binary streaming, sentence-chunked, int16 PCM
- `WS /v1/audio/speech/ws` — WebSocket, send JSON receive binary PCM per sentence
- `POST /v1/audio/speech/clone` — multipart form, voice cloning from reference audio
- `GET /health` — status, GPU memory, queue depth, cache stats, voice list
- `POST /cache/clear` — evict audio output cache
- `GET /metrics` — Prometheus metrics (when `PROMETHEUS_ENABLED=true`)

**Caching**:
- `_audio_cache` — LRU `OrderedDict` keyed by SHA-256 of `(text, voice, speed, format, language, instruct)`. Size controlled by `AUDIO_CACHE_MAX` (default 256). Skips GPU entirely on hit.
- `_voice_cache` — LRU cache of decoded reference audio (numpy arrays) keyed by SHA-256 of raw audio bytes. Size controlled by `VOICE_CACHE_MAX` (default 32).

**Voice mapping**: `VOICE_MAP` maps both native Qwen speaker names (`vivian`, `serena`, `uncle_fu`, `dylan`, `eric`, `ryan`, `aiden`, `ono_anna`, `sohee`) and OpenAI-style aliases (`alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`) to Qwen speakers. Unknown voice names return 400 with a list of valid voices.

**Model loading**: Uses `flash_attention_2` if `flash-attn` is installed, falls back to `sdpa`. After loading, runs multi-length GPU warmup and pre-warms CUDA memory pool. `torch.compile(mode="reduce-overhead")` is applied unless `TORCH_COMPILE=false`.

**Adaptive token budget**: `_adaptive_max_tokens()` scales `max_new_tokens` (128–2048) based on text length and CJK character ratio, avoiding fixed 2048 allocation for short inputs.

## Key Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | HuggingFace model ID |
| `IDLE_TIMEOUT` | `120` | Seconds before idle GPU unload (0 = disabled) |
| `REQUEST_TIMEOUT` | `300` | Max seconds per inference request |
| `PRELOAD_MODEL` | `false` | Load model at startup instead of first request |
| `TORCH_COMPILE` | `true` | Enable `torch.compile` on the model |
| `PROMETHEUS_ENABLED` | `true` | Expose `/metrics` endpoint |
| `LOG_FORMAT` | `json` | `json` or `text` log format |
| `LOG_LEVEL` | `INFO` | Minimum log level (DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL) |
| `TEXT_NORMALIZE` | `true` | Expand numbers, currency, abbreviations |
| `MAX_QUEUE_DEPTH` | `5` | 503 early rejection threshold (0 = unlimited) |
| `AUDIO_CACHE_MAX` | `256` | Max cached audio outputs (0 = disabled) |
| `VOICE_CACHE_MAX` | `32` | Max cached reference audio arrays (0 = disabled) |
| `INFERENCE_CPU_CORES` | `` | CPU affinity spec (e.g. `0-3,6`) for GPU-adjacent cores |
| `UNIX_SOCKET_PATH` | `` | Run on UDS instead of TCP (bypasses TCP stack) |
| `SSL_KEYFILE` / `SSL_CERTFILE` | `` | Enable HTTP/2 via TLS |

## Important Details

- Model weights download to `./models` (mounted as HuggingFace cache volume) on first run (~2.4 GB)
- The `qwen-tts` pip package provides `Qwen3TTSModel` — this is the upstream model library
- Audio format: WAV/FLAC/OGG use `soundfile`; MP3/Opus use `pydub`; WAV can use `torchaudio` if available
- Speed adjustment: `pyrubberband` (pitch-preserving, preferred) or `scipy.signal.resample` (fallback)
- Language detection: `fasttext-langdetect` if installed, else Unicode character range heuristic (`_detect_language_unicode`)
- Streaming endpoints (`/stream`, `/stream/pcm`, `/ws`) split text into sentences via `_split_sentences()` and synthesize each independently — no cross-sentence context
- `docker-entrypoint.sh` handles GPU persistence mode, clock locking, THP, jemalloc, and UDS/TLS startup branching
