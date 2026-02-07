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

Tests require a running server instance (they hit the HTTP API):

```bash
python test_tts.py                  # Full test suite (health, synthesis, voices, languages, formats, errors)
bash test_quick.sh                  # Quick smoke test, outputs test_output.wav
TTS_URL=http://host:port python test_tts.py  # Test against different host
```

Tests are not pytest-based; `test_tts.py` uses a custom pass/fail reporter with global counters.

## Architecture

**Single-file server** (`server.py`): All API logic, model management, and audio processing in one file. No separate modules.

**GPU lifecycle**: Model loads on first request, not at startup. An `_idle_watchdog` background task unloads the model after `IDLE_TIMEOUT` seconds (default 120s) of inactivity to free VRAM. `_model_lock` prevents concurrent load/unload races. `_infer_semaphore(1)` serializes GPU inference to prevent OOM.

**Endpoints**:
- `POST /v1/audio/speech` — JSON body, OpenAI-compatible TTS (uses `generate_custom_voice`)
- `POST /v1/audio/speech/clone` — multipart form, voice cloning from reference audio (uses `generate_voice_clone`)
- `GET /health` — status, GPU memory, loaded model info

**Voice mapping**: `VOICE_MAP` dict maps both native Qwen speaker names and OpenAI-style aliases (alloy, echo, etc.) to Qwen speakers. Unknown voice names pass through as-is.

**Inference pattern**: All inference runs in a thread pool executor (`run_in_executor`) wrapped with `asyncio.wait_for` for timeout. `_do_synthesize` and `_do_voice_clone` handle the actual model calls within `torch.inference_mode()` and always call `release_gpu_memory()` in `finally`.

## Key Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | HuggingFace model ID |
| `IDLE_TIMEOUT` | `120` | Seconds before idle GPU unload (0 = disabled) |
| `REQUEST_TIMEOUT` | `300` | Max seconds per inference request |

## Important Details

- Model weights download to `./models` (mounted as HuggingFace cache volume) on first run (~2.4 GB)
- The `qwen-tts` pip package provides `Qwen3TTSModel` — this is the upstream model library
- Audio format conversion uses `soundfile` for WAV/FLAC/OGG and `pydub` for MP3
- Speed adjustment is done via `scipy.signal.resample`, not at the model level
- Language auto-detection (`detect_language`) uses Unicode character ranges — simple heuristic, not a library
