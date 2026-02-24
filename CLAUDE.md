# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenAI-compatible TTS API server wrapping [Qwen3-TTS-0.6B](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base). Single-file FastAPI server (`server.py`) containerized with Docker and NVIDIA GPU acceleration. Designed for shared GPU environments with on-demand model loading/unloading. Supports both named voices (via pre-generated reference audio) and voice cloning from user-provided audio.

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

Clone tests exercise the `/v1/audio/speech/clone` endpoint with real voice cloning inference.

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
- `_voice_prompt_cache` — LRU cache of pre-computed voice clone prompts (speaker embeddings) keyed by SHA-256 of raw audio bytes. Used by `/clone` endpoint. Size controlled by `VOICE_CACHE_MAX` (default 32).

**Voice mapping**: `VOICE_MAP` maps voice names to reference audio WAV filenames in `voices/`. Both native Qwen speaker names (`vivian`, `serena`, `uncle_fu`, `dylan`, `eric`, `ryan`, `aiden`, `ono_anna`, `sohee`) and OpenAI-style aliases (`alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`) resolve to WAV files. Named voices use `generate_voice_clone()` with pre-computed voice prompts from these reference files. Unknown voice names return 400 with a list of valid voices.

**Voice prompts**: `_voice_prompts` dict maps WAV filename → pre-computed `VoiceClonePromptItem` list. Populated during `_load_model_sync()` via `create_voice_clone_prompt(x_vector_only_mode=True)`. Cleared on model unload. All synthesis (named voices and streaming) goes through `generate_voice_clone()`.

**instruct parameter**: Accepted for API backwards compatibility but silently ignored — the Base model does not support instruction-controlled synthesis. A warning is logged when `instruct` is set.

**Model loading**: Uses `flash_attention_2` if `flash-attn` is installed, falls back to `sdpa`. After loading, pre-computes voice prompts from reference WAVs, runs multi-length GPU warmup via `generate_voice_clone()`, and pre-warms CUDA memory pool. `torch.compile(mode="reduce-overhead")` is applied unless `TORCH_COMPILE=false`.

**Adaptive token budget**: `_adaptive_max_tokens()` scales `max_new_tokens` (128–2048) based on text length and CJK character ratio, avoiding fixed 2048 allocation for short inputs.

**Logging**: loguru with `_InterceptHandler` routing all uvicorn/FastAPI stdlib logs. JSON (`LOG_FORMAT=json`) or human-readable (`LOG_FORMAT=text`). Structured context via `logger.bind()` on every log point. Coverage: startup config dump, cache hit/miss/eviction (DEBUG), all request errors/timeouts (ERROR with traceback), validation rejections (WARNING), streaming completion events, WebSocket lifecycle, idle watchdog triggers, inference queue dispatch decisions (DEBUG).

## Key Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | HuggingFace model ID |
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
- `voices/` directory contains 9 pre-generated reference WAV files (one per speaker), baked into the Docker image. These were bootstrapped from the CustomVoice model and serve as voice identity for named speakers via `generate_voice_clone()`
- The `qwen-tts` pip package provides `Qwen3TTSModel` — this is the upstream model library. The Base model variant supports `generate_voice_clone()` and `create_voice_clone_prompt()` but not `generate_custom_voice()`
- Audio format: WAV/FLAC/OGG use `soundfile`; MP3/Opus use `pydub`; WAV can use `torchaudio` if available
- Speed adjustment: `pyrubberband` (pitch-preserving, preferred) or `scipy.signal.resample` (fallback)
- Language detection: `fasttext-langdetect` if installed, else Unicode character range heuristic (`_detect_language_unicode`)
- Streaming endpoints (`/stream`, `/stream/pcm`, `/ws`) split text into sentences via `_split_sentences()` and synthesize each independently — no cross-sentence context
- `docker-entrypoint.sh` handles GPU persistence mode, clock locking, THP, jemalloc, and UDS/TLS startup branching
