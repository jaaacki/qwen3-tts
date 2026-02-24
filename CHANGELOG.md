# Changelog

## v0.8.1 — 2026-02-24

E2E test suite fixes — first successful full run.

### Fixed
- Docker build: `accelerate==1.1.1` → `1.12.0` to match qwen-tts dependency (#94)
- Dockerfile: multi-stage COPY to `/opt/conda/` instead of `/usr/local/` for conda-based base image (#94)
- Dockerfile: removed `torchao` from main deps — incompatible with both torch 2.5.x and transformers 4.57.3 (#95)
- `_load_model_sync()`: removed `import torch._dynamo` that shadowed global `torch` binding, causing `UnboundLocalError` on every model load (#96)
- Queue depth counter: wrapped cache-hit early-return path in `try/finally` so `_queue_depth` always decrements (#97)
- Unknown voice names now return 400 with valid voice list instead of passing through to model and crashing with 500 (#99)
- Opus format content-type changed from `audio/opus` to `audio/ogg` to match OpenAI API convention (#102)

### Removed
- `_trim_silence()` / `VAD_TRIM` env var — VAD is wrong tool for TTS output; synthesized audio doesn't need voice activity detection (#100)

### Changed (E2E tests)
- WAV duration parser walks RIFF chunks instead of assuming 44-byte header (#98)
- SSE audio event test matches server's raw base64 format (#101)
- Clone inference tests skipped when model doesn't support voice cloning (#103)

---

## v0.8.0 — 2026-02-24

Phase 5 Scale complete. Issues #84–#86 implemented.

### Added
- `PriorityInferQueue.submit_batch()` — queues a batchable synthesis job with `batch_key="synthesis"` (#84)
- `_do_synthesize_batch()` — dispatches multiple texts in a single `model.generate_custom_voice(text=[...])` call (#84)
- `MAX_BATCH_SIZE` env var (default 4) — controls max jobs per GPU dispatch (#84)
- `gateway.py` — lightweight FastAPI proxy that manages inference worker subprocess (#85)
- `worker.py` — worker subprocess entry point; preloads model, disables idle timeout (#85)
- `GATEWAY_MODE` env var in compose.yaml — set `true` for ~30 MB idle footprint vs ~1 GB (#85)
- `WORKER_HOST`, `WORKER_PORT` env vars for gateway → worker routing (#85)
- `QUANTIZE` env var — `int8` (bitsandbytes, ~50% VRAM reduction) or `fp8` (torchao, ~67% VRAM reduction) (#86)
- `_resolve_quant_kwargs()` helper — returns `(dtype, load_kwargs)` for model loading (#86)
- `bitsandbytes>=0.43.0` and `torchao>=0.5.0` added to `requirements.txt` (optional, install only when needed) (#86)

### Changed
- `PriorityInferQueue._worker()` now drains all pending synthesis jobs up to `MAX_BATCH_SIZE` and dispatches in one GPU call (#84)
- `synthesize_speech` falls back to single-job path when `instruct` param is set (#84)
- Dockerfile CMD now branches on `GATEWAY_MODE`: gateway or full server (#85)
- `_load_model_sync()` uses `_resolve_quant_kwargs()` instead of hardcoded `dtype=torch.bfloat16` (#86)

### Detail

**#84 Batch inference**
- Synthesis jobs tagged with `batch_key="synthesis"` are drained atomically from the heap in one lock pass; `_do_synthesize_batch()` pads and runs them as a single forward pass
- `instruct` and voice-clone requests fall back to single-job path automatically (model batch API does not support mixed modes)
- Setting `MAX_BATCH_SIZE=1` effectively disables batching while keeping code path consistent

**#85 Gateway/Worker mode**
- `gateway.py` (~30 MB RSS idle) spawns `worker.py` on first request; double-checked lock prevents concurrent spawns; idle watchdog kills worker after `IDLE_TIMEOUT` seconds
- Known limitations: SSE and WebSocket endpoints buffered (not streamed) through the proxy — documented for follow-up

**#86 Quantization**
- `int8` → bitsandbytes `load_in_8bit=True`, `float16` dtype (~50% VRAM); `fp8` → torchao `TorchAoConfig`, `bfloat16` dtype (~67% VRAM, Hopper+ only)
- `_resolve_quant_kwargs()` validates at model load time; misconfigured `QUANTIZE` fails fast with a clear error

---

## v0.7.0 — 2026-02-24

Phase 4 Intelligence complete. Issues #81–#83 implemented.

### Added
- Optional `temperature` and `top_p` fields on `TTSRequest` — passed through to `model.generate()` kwargs for controlling generation diversity (#83)
- `temperature` and `top_p` Form parameters on `/v1/audio/speech/clone` endpoint (#83)
- `temperature` and `top_p` JSON fields accepted on WebSocket `/v1/audio/speech/ws` endpoint (#83)
- `_build_gen_kwargs()` helper to DRY up gen_kwargs construction across all synthesis endpoints (#83)

### Changed
- Replaced `asyncio.Semaphore(1)` inference serialization with `PriorityInferQueue` min-heap (#81)
- WebSocket, SSE, and raw PCM streaming endpoints now run at `PRIORITY_REALTIME=0` (#81)
- REST `/v1/audio/speech` and `/v1/audio/speech/clone` run at `PRIORITY_BATCH=1` (#81)
- Voice clone cache now stores pre-computed speaker embeddings via `model.create_voice_clone_prompt()` instead of raw decoded audio arrays (#82)
- `_voice_cache` renamed to `_voice_prompt_cache`; `_get_cached_ref_audio()` replaced by `_get_cached_voice_prompt()` (#82)
- `_do_voice_clone()` accepts a pre-computed prompt object instead of raw `(audio, sr)` tuple (#82)
- `POST /cache/clear` now also clears the voice prompt cache, returning `{"audio_cleared": N, "voice_cleared": M}` (#82)
- Replaced 4x repeated inline `gen_kwargs` dict construction with `_build_gen_kwargs()` calls (#83)
- Fixed variable ordering bug in `/v1/audio/speech/clone` where `_adaptive_max_tokens(text)` was called before `text` was assigned (#83)

### Detail

**#81 Priority inference queue**
- `PriorityInferQueue` backed by `heapq` stdlib min-heap; `@dataclass(order=True) _InferJob(priority, submit_time)` sort key
- `PRIORITY_REALTIME=0` for WS/SSE/PCM, `PRIORITY_BATCH=1` for REST — under mixed load real-time streams always run first
- Queue singleton started from `lifespan()`; asyncio.Lock + asyncio.Event internals; no busy-wait

**#82 Voice clone prompt cache**
- `create_voice_clone_prompt()` runs speaker encoder once per unique ref audio; repeat callers pay near-zero overhead
- Cache key: SHA-256 of raw audio bytes; eviction: LRU via OrderedDict, `VOICE_CACHE_MAX` limit

**#83 Generation parameters**
- `None` means omit from kwargs entirely — model uses defaults; passing `None` explicitly is different from not passing it
- `_build_gen_kwargs()` centralizes max_new_tokens + conditional temperature/top_p construction

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
