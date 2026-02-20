# Changelog

## [Unreleased — Issue #27: Always-on mode] — 2026-02-20
### Added
- Always-on mode documentation — `IDLE_TIMEOUT=0` disables idle unload for dedicated GPU servers (#27)

## [Unreleased — Issue #33: Migrate @app.on_event to FastAPI lifespan] — 2026-02-20
### Changed
- Migrated from deprecated `@app.on_event("startup")` to FastAPI lifespan context manager (#33)
- Server now performs graceful model unload on shutdown via lifespan teardown

## [Unreleased — Issue #17: Audio output LRU cache] — 2026-02-20
### Added
- Audio output LRU cache — in-memory cache keyed by SHA-256 of (text, voice, speed, format, language, instruct); cache hit returns bytes in ~1ms, skipping GPU entirely (#17)
- `POST /cache/clear` endpoint — clears audio cache, returns count of evicted entries (#17)
- `AUDIO_CACHE_MAX` env var — max cache entries (default 256, set to 0 to disable) (#17)
- Cache info in `/health` — `audio_cache_size` and `audio_cache_max` fields (#17)
- Voice clone endpoint is intentionally not cached — clone inputs (ref audio) are unlikely to repeat

## [Unreleased — Issue #36: remove dead VoiceCloneRequest] — 2026-02-20
### Changed
- Remove unused `VoiceCloneRequest` Pydantic model from `server.py` — the `/clone` endpoint uses `Form()` parameters directly; model was dead code (#36)
## [Unreleased — Issue #16: Pre-allocate GPU memory pool] — 2026-02-20
### Added
- GPU memory pool pre-warming after model warmup — allocates and frees a 128 MB dummy tensor to pre-reserve a contiguous CUDA memory block, reducing first-request allocation jitter (#16)
- `max_split_size_mb:512` added to `PYTORCH_CUDA_ALLOC_CONF` in Dockerfile to reduce memory fragmentation from large allocations

## [Unreleased — Issue #15: Add voice prompt cache for /clone endpoint] — 2026-02-20
### Added
- Voice prompt cache for `/clone` endpoint — caches processed reference audio by SHA-256 content hash (#15)
- `VOICE_CACHE_MAX` env var (default: 32) — controls LRU cache capacity; set to 0 to disable
- Cache stats exposed in `/health` endpoint: `voice_cache_size`, `voice_cache_max`, `voice_cache_hits`

## [Unreleased — Issue #14: Replace scipy speed adjustment with pyrubberband] — 2026-02-20
### Changed
- `_adjust_speed()` now uses `pyrubberband.time_stretch()` for pitch-preserving speed changes, falling back to `scipy.signal.resample` when pyrubberband is unavailable (#14)
- Added `pyrubberband` to Dockerfile pip dependencies and `rubberband-cli` to apt dependencies

## [Unreleased — Issue #13: Replace Unicode language heuristic with fasttext detection] — 2026-02-20
### Changed
- `detect_language()` now uses `fasttext-langdetect` for accurate multi-language detection across 10 languages, falling back to Unicode character-range heuristic when fasttext is unavailable (#13)
- Added `fasttext-langdetect` to Dockerfile dependencies
- Added `_get_langdetect()` lazy-loader, `_detect_language_unicode()` fallback, and `_LANG_MAP` ISO-to-Qwen mapping

## [Unreleased — Issue #10: Multi-length GPU warmup] — 2026-02-20
### Changed
- GPU warmup now runs 3 synthesis calls at different text lengths (5, 30, 90 chars) to pre-cache more CUDA kernel paths (#10)

## [Unreleased — Issue #9: Enable torch.compile] — 2026-02-20
### Added
- `torch.compile` on model forward pass with `reduce-overhead` mode for faster inference (#9)
- `TORCH_COMPILE` env var (default: true) to opt-in/out of compilation

## [Unreleased — Issue #8: Switch to flash_attention_2] — 2026-02-20
### Changed
- Switch attention implementation from `sdpa` to `flash_attention_2` with graceful fallback (#8)
- Added `flash-attn` to Dockerfile dependencies

## [Unreleased — Issue #7: Lock GPU clocks to max boost] — 2026-02-20
### Changed
- `docker-entrypoint.sh` — lock GPU clocks to max boost frequency at container startup for consistent inference latency (#7)

## [Unreleased — Issue #6: Enable GPU persistence mode] — 2026-02-20
### Added
- `docker-entrypoint.sh` — GPU persistence mode (`nvidia-smi -pm 1`) runs at container startup, eliminating 200-500ms GPU cold-start penalty (#6)

### Changed
- Dockerfile now uses ENTRYPOINT for GPU tuning before uvicorn starts

## [Unreleased — Issue #5: Enable TF32 matmul mode] — 2026-02-20
### Changed
- Enable TF32 matmul and cuDNN TF32 on Ampere+ GPUs for ~3x faster matrix operations (#5)

## [Unreleased — Issue #4: Add raw PCM streaming endpoint] — 2026-02-20
### Added
- `POST /v1/audio/speech/stream/pcm` — raw PCM streaming endpoint; splits text into sentences, streams each as raw int16 PCM bytes with `X-PCM-Sample-Rate`, `X-PCM-Bit-Depth`, `X-PCM-Channels` headers (#4)

## [Unreleased — Issue #3: Add sentence-chunked SSE streaming] — 2026-02-20
### Added
- Sentence-chunked SSE streaming endpoint `POST /v1/audio/speech/stream` (#3)
  - Splits input into sentences with abbreviation-aware regex
  - Streams base64-encoded raw PCM (int16, 24kHz) via Server-Sent Events
  - Sends `data: [DONE]` on completion, `data: [ERROR] message` on failure
  - Updates `_last_used` per chunk to prevent idle unload during streaming

## [Unreleased — Issue #2: Add adaptive max_new_tokens scaling] — 2026-02-20
### Changed
- Replace hardcoded `max_new_tokens: 2048` with adaptive scaling based on input text length (#2)
  - Short inputs (<=16 words) get minimum budget of 128 tokens
  - Budget scales at 8 tokens/word with a cap at 2048
  - Reduces KV-cache allocation overhead by up to 40x for short texts

## [Unreleased — Issue #1: Add per-request latency breakdown logging] — 2026-02-20
### Added
- Per-request latency breakdown logging via `logging.getLogger("qwen3-tts")` with `time.perf_counter()` timing in both `/v1/audio/speech` and `/v1/audio/speech/clone` endpoints (#1)
- Logged fields: `queue_ms`, `inference_ms`, `encode_ms`, `total_ms`, `chars`, `voice`, `format`, `language`

## [Docs] 2026-02-20 — Improvement roadmap and project documentation

### Added
- `ROADMAP.md` — three-phase improvement plan (v0.1.0 Real-Time, v0.2.0 Speed & Quality, v0.3.0 Production Grade) with 36 linked GitHub issues
- `LEARNING_LOG.md` — 7 narrative entries covering architecture baseline, streaming rationale, max_new_tokens discovery, phase ordering logic, streaming risks, GPU tuning, and caching hierarchy
- `improvements.md` — full catalogue of 40 optimizations with performance estimates, implementation sketches, and execution order
- GitHub milestones: Phase 1 (#1), Phase 2 (#2), Phase 3 (#3)
- GitHub issues: #1–#36 covering all roadmap items with What/Why/Expectations for each
- GitHub labels: `phase-1`, `phase-2`, `phase-3`, `enhancement`, `refactor`, `chore`

## v0.3.2 — 2026-02-07

### Fixed
- **Audio cutoff at end of speech** — reverted `np.asarray` (zero-copy view) back to `np.array` with explicit copy; the model's underlying buffer could be freed before audio encoding completes, truncating the tail of the audio

## v0.3.1 — 2026-02-07

### Added
- **uvloop** — high-performance async event loop replacing default asyncio loop
- **httptools** — C-based HTTP parser for uvicorn replacing pure-Python h11
- **orjson** — fast JSON serialization for FastAPI request/response handling
- **`shm_size: 1g`** in compose.yaml for PyTorch shared memory

### Changed
- Uvicorn CMD now uses `--loop uvloop --http httptools --no-access-log --timeout-keep-alive 65`
- `OMP_NUM_THREADS=2` / `MKL_NUM_THREADS=2` — limits CPU thread spawning (GPU does the heavy work)
- `PYTHONUNBUFFERED=1` / `PYTHONDONTWRITEBYTECODE=1` — immediate log output, skip .pyc generation
- Healthcheck `start_period` reduced from 120s to 15s (model loads on-demand, server starts in seconds)
- `IDLE_TIMEOUT` now explicit in compose.yaml environment

## v0.3.0 — 2026-02-07

### Added
- **`instruct` parameter** on `/v1/audio/speech` for style/instruction control
- **Dedicated inference executor** — single-thread `ThreadPoolExecutor` replaces default pool, reducing thread management overhead
- **`cudnn.benchmark` enabled** — CUDA autotuner selects fastest convolution algorithms for the GPU

### Changed
- **Removed per-request `gc.collect()` + `torch.cuda.empty_cache()`** — eliminates ~50-150ms latency penalty per request; CUDA memory cache is now reused across requests instead of thrashed
- **Full GPU cleanup (`gc.collect` + `empty_cache` + `ipc_collect`) only runs during model unload**, not on every inference
- **Module-level imports** for `scipy.signal` and `pydub` — no more per-request import overhead
- **`asyncio.get_running_loop()`** replaces deprecated `asyncio.get_event_loop()` (3 occurrences)
- **`np.asarray`** replaces `np.array` for zero-copy when model output is already float32
- **Warmup runs inside `torch.inference_mode()`** with longer text (64 tokens) for better CUDA kernel coverage
- Removed `release_gpu_memory()` calls from error handlers — local tensors are freed by Python refcounting on stack unwind

## v0.2.0 — 2026-02-06

### Added
- **On-demand model loading** — model loads on first request instead of at startup (0 VRAM when idle)
- **Idle auto-unload** — model automatically unloads after `IDLE_TIMEOUT` seconds of inactivity (default: 120s), freeing GPU VRAM for other services
- **GPU inference semaphore** — serializes concurrent requests to prevent OOM on shared GPU
- **Request timeout** — configurable via `REQUEST_TIMEOUT` env var (default: 300s)
- **GPU warmup** — runs a short inference on first load to pre-cache CUDA kernels
- **Health endpoint improvements** — reports GPU memory usage, device name
- **Docker healthcheck** in compose.yaml
- **Timeout and error handling** — 504 on timeout, proper GPU memory cleanup on errors

### Changed
- SDPA attention implementation (`attn_implementation="sdpa"`) for better memory efficiency
- `low_cpu_mem_usage=True` for reduced peak memory during model loading
- `torch.inference_mode()` context for all inference calls
- GPU memory explicitly released after every inference via `torch.cuda.empty_cache()`
- Thread pool execution for inference (non-blocking async server)
- Dockerfile: added `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `--no-install-recommends`, apt cache cleanup, sox dependency

## v0.1.0 — 2026-02-06

- Initial release with Qwen3-TTS-0.6B-CustomVoice model
- OpenAI-compatible `/v1/audio/speech` endpoint
- Voice cloning via `/v1/audio/speech/clone`
- 9 built-in voices + OpenAI voice aliases
- Multi-language support with auto-detection
- Multiple output formats (WAV, MP3, FLAC, OGG)
