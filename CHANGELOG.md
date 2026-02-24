# Changelog

## [Unreleased — Issue #83: expose temperature and top_p] — 2026-02-24

### Added
- Optional `temperature` and `top_p` fields on `TTSRequest` — passed through to `model.generate()` kwargs for controlling generation diversity
- `temperature` and `top_p` Form parameters on `/v1/audio/speech/clone` endpoint
- `temperature` and `top_p` JSON fields accepted on WebSocket `/v1/audio/speech/ws` endpoint
- `_build_gen_kwargs()` helper to DRY up gen_kwargs construction across all synthesis endpoints

### Changed
- Replaced 4x repeated inline `gen_kwargs` dict construction with `_build_gen_kwargs()` calls in `/v1/audio/speech`, `/v1/audio/speech/stream`, `/v1/audio/speech/stream/pcm`
- Fixed variable ordering bug in `/v1/audio/speech/clone` where `_adaptive_max_tokens(text)` was called before `text` was assigned

---

## v0.6.0 — 2026-02-20

Phase 3 Production Grade complete. All 36 roadmap issues implemented.

### Added
- Audio output LRU cache — `AUDIO_CACHE_MAX`, `POST /cache/clear` (#17)
- Opus codec support via pydub/ffmpeg (#18)
- GPU-accelerated audio processing with torchaudio (#19)
- Async audio encode pipeline — overlap encode N with synthesis N+1 (#20)
- jemalloc memory allocator via `LD_PRELOAD` (#21)
- CPU affinity for inference thread — `INFERENCE_CPU_CORES` env var (#22)
- Transparent huge pages for model weights via docker-entrypoint.sh (#23)
- WebSocket streaming endpoint `WS /v1/audio/speech/ws` (#24)
- HTTP/2 support with conditional TLS (`SSL_KEYFILE`/`SSL_CERTFILE`) (#25)
- Unix domain socket support — `UNIX_SOCKET_PATH` env var (#26)
- Always-on mode documentation — `IDLE_TIMEOUT=0` (#27)
- Eager model preload — `PRELOAD_MODEL` env var (#28)
- `ipc: host` in Docker compose for CUDA IPC (#29)
- Prometheus metrics endpoint `GET /metrics` with custom TTS gauges (#30)
- Structured JSON logging with per-request fields and `LOG_FORMAT` env var (#31)
- Request queue depth limit with 503 early rejection — `MAX_QUEUE_DEPTH` (#32)

### Changed
- Migrated from `@app.on_event` to FastAPI lifespan context manager (#33)
- Pinned all dependency versions in `requirements.txt` (#34)
- Converted to multi-stage Docker build — runtime image ships no build tools (#35)
- Removed dead `VoiceCloneRequest` Pydantic model (#36)

### Detail

**#32 Request queue depth limit**
- `MAX_QUEUE_DEPTH` env var (default 5, 0 = unlimited)
- `Retry-After: 5` header on 503 responses
- `queue_depth` and `max_queue_depth` fields in `/health` response

**#31 Structured JSON logging**
- `LOG_FORMAT` env var: `json` (default) for structured output, `text` for human-readable
- Per-request `request_id`, latency breakdown (queue_ms, infer_ms, encode_ms, total_ms), voice, language, chars, format

**#26 Unix domain socket support**
- `UNIX_SOCKET_PATH` env var enables UDS mode, bypassing TCP stack for same-host clients
- UDS mode disables TCP binding — use `curl --unix-socket` syntax

**#25 HTTP/2 support**
- `h2>=4.0.0` package installed; requires TLS certificates (h2c cleartext not widely supported)
- `docker-entrypoint.sh` appends `--ssl-keyfile`/`--ssl-certfile` to uvicorn args only when env vars are set

**#24 WebSocket streaming endpoint**
- `WS /v1/audio/speech/ws` accepts JSON, streams binary PCM per sentence chunk, sends `{"event": "done"}` on completion
- Abbreviation-aware sentence splitting handles Dr., U.S.A., CJK full-width punctuation
- Add `np.clip(-1.0, 1.0)` before int16 conversion to prevent audio distortion

**#23 Transparent huge pages**
- `docker-entrypoint.sh` enables THP madvise mode at startup; reduces TLB pressure for 2.4 GB model weights
- Dockerfile now uses ENTRYPOINT with shell script instead of CMD array

**#22 CPU affinity** — Uses `os.sched_setaffinity()` (not `os.system(taskset)`) to prevent command injection via env var

**#21 jemalloc** — `MALLOC_CONF` tuning: background thread, 1s dirty decay, immediate muzzy decay. LD_PRELOAD path assumes x86_64; adjust for aarch64.

**#20 Async audio encode** — dedicated `_encode_executor` (2 CPU threads) for format conversion; `_encode_audio_async` runs in CPU thread pool alongside inference

**#19 GPU-accelerated audio** — torchaudio WAV encoding + GPU speed adjustment via `torchaudio.functional.resample()` with CUDA tensor; falls back to soundfile/scipy on CPU-only hosts

**#18 Opus codec** — `response_format=opus` via pydub/ffmpeg libopus at 32kbps; ~2.5ms encode latency vs ~50ms for MP3

**#17 Audio output LRU cache** — SHA-256 key over (text, voice, speed, format, language, instruct); ~1ms cache hit vs 500ms+ GPU inference. Voice clone not cached — ref audio inputs are unlikely to repeat.

---

## v0.5.0 — 2026-02-20

Phase 2 Speed & Quality complete. Issues #5–#16.

### Added
- `torch.compile` with `reduce-overhead` mode — `TORCH_COMPILE` env var (#9)
- Multi-length GPU warmup — 3 synthesis calls at 5/30/90 chars to pre-cache CUDA kernel paths (#10)
- VAD silence trimming — strips leading/trailing silence, `VAD_TRIM` env var (#11)
- Text normalization — expands numbers, currency, abbreviations, `TEXT_NORMALIZE` env var (#12)
- fasttext language detection — `fasttext-langdetect` with Unicode heuristic fallback (#13)
- Voice prompt cache for `/clone` — SHA-256 content hash, `VOICE_CACHE_MAX` env var (#15)
- GPU memory pool pre-warming — allocates/frees 128 MB dummy tensor after model load to pre-reserve contiguous CUDA block (#16)

### Changed
- Enable TF32 matmul and cuDNN TF32 on Ampere+ GPUs for ~3x faster matrix operations (#5)
- GPU persistence mode (`nvidia-smi -pm 1`) at container startup — eliminates 200–500ms cold-start penalty (#6)
- Lock GPU clocks to max boost for consistent inference latency (#7)
- Switch attention from `sdpa` to `flash_attention_2` with graceful fallback (#8)
- `_adjust_speed()` uses `pyrubberband.time_stretch()` for pitch-preserving speed changes, falling back to `scipy.signal.resample` (#14)

---

## v0.4.0 — 2026-02-20

Phase 1 Real-Time complete. Issues #1–#4.

### Added
- Per-request latency breakdown logging — `queue_ms`, `inference_ms`, `encode_ms`, `total_ms`, `chars`, `voice`, `format`, `language` (#1)
- Sentence-chunked SSE streaming endpoint `POST /v1/audio/speech/stream` — abbreviation-aware regex, base64 PCM via Server-Sent Events, `data: [DONE]` on completion (#3)
- Raw PCM streaming endpoint `POST /v1/audio/speech/stream/pcm` — int16 bytes with `X-PCM-Sample-Rate`/`X-PCM-Bit-Depth`/`X-PCM-Channels` headers (#4)

### Changed
- Replace hardcoded `max_new_tokens: 2048` with adaptive scaling — 8 tokens/word, min 128, cap 2048; up to 40x reduction in KV-cache for short texts (#2)

---

## [Docs] 2026-02-20 — Improvement roadmap and project documentation

### Added
- `ROADMAP.md` — three-phase improvement plan with 36 linked GitHub issues
- `LEARNING_LOG.md` — narrative entries covering architecture decisions and tradeoffs
- `improvements.md` — full catalogue of 40 optimizations with performance estimates
- GitHub milestones: Phase 1 (#1), Phase 2 (#2), Phase 3 (#3)
- GitHub issues: #1–#36 covering all roadmap items with What/Why/Expectations for each
- GitHub labels: `phase-1`, `phase-2`, `phase-3`, `enhancement`, `refactor`, `chore`

---

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
