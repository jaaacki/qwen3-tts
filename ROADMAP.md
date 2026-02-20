# Roadmap

Three-phase plan to take Qwen3-TTS from a working prototype to a production-grade, real-time TTS server. Each phase targets a minor version and has a clear milestone vision.

---

## Phase 1 — Real-Time (v0.1.0)

**Vision**: First audio reaches the client in under 500 ms.

Before optimizing anything, we instrument. Then we remove the biggest bottleneck (fixed token budget), and finally deliver streaming so clients hear audio while synthesis continues.

- [x] #1 Add per-request latency breakdown logging
- [x] #2 Add adaptive `max_new_tokens` scaling with text length
- [x] #3 Add sentence-chunked SSE streaming endpoint
- [x] #4 Add raw PCM streaming endpoint

---

## Phase 2 — Speed & Quality (v0.2.0)

**Vision**: Every synthesis is measurably faster and the audio quality is noticeably better.

GPU tuning flags come first because they are low-risk and establish a faster baseline for benchmarking everything after. Inference optimizations build on that baseline. Audio quality improvements come last in the phase because they depend on a stable, fast inference path.

- [x] #5 Enable TF32 matmul mode
- [x] #6 Enable GPU persistence mode
- [x] #7 Lock GPU clocks to max boost
- [x] #8 Switch `attn_implementation` to `flash_attention_2`
- [x] #9 Enable `torch.compile` on model forward pass
- [x] #10 Deepen GPU warmup with multi-length synthesis calls
- [x] #11 Add VAD silence trimming (strip leading/trailing silence)
- [x] #12 Add text normalization for numbers, currency, abbreviations
- [x] #13 Replace Unicode language heuristic with fasttext detection
- [x] #14 Replace scipy speed adjustment with pitch-preserving pyrubberband
- [x] #15 Add voice prompt cache for `/clone` endpoint
- [x] #16 Pre-allocate GPU memory pool to reduce allocation jitter

---

## Phase 3 — Production Grade (v0.3.0)

**Vision**: The server is observable, resilient, and efficient under real production load.

Caching and codec work unlocks efficient streaming. System-level tuning reduces jitter. Protocol upgrades widen the deployment surface. Lifecycle and observability close the loop for production operations. Housekeeping at the end cleans up tech debt while the server is stable.

- [x] #17 Add full audio output LRU cache (text+voice to cached bytes)
- [x] #18 Add Opus codec support for streaming
- [x] #19 Add GPU-accelerated audio processing with torchaudio
- [x] #20 Add async audio encode pipeline (overlap encode N with synthesis N+1)
- [x] #21 Add jemalloc memory allocator via `LD_PRELOAD`
- [x] #22 Set CPU affinity for inference thread to GPU-adjacent cores
- [x] #23 Enable transparent huge pages for model weights
- [x] #24 Add WebSocket streaming endpoint
- [x] #25 Enable HTTP/2 support
- [x] #26 Add Unix domain socket support for same-host clients
- [x] #27 Add always-on mode (`IDLE_TIMEOUT=0` option, documented)
- [x] #28 Add eager model preload on startup (`PRELOAD_MODEL` env var)
- [x] #29 Add `ipc:host` to Docker compose for CUDA IPC
- [x] #30 Add Prometheus metrics endpoint
- [ ] #31 Add structured JSON logging with per-request fields
- [ ] #32 Add request queue depth limit with 503 early rejection
- [x] #33 Migrate `@app.on_event` to FastAPI lifespan context manager
- [x] #34 Pin all dependency versions in `requirements.txt`
- [x] #35 Convert to multi-stage Docker build
- [x] #36 Remove dead `VoiceCloneRequest` model

---

## Backlog

No unplaced items. All improvements from the initial analysis have been assigned to a phase.

---

## Current Status

**v0.0.1** — Pre-roadmap. The server is functional with OpenAI-compatible TTS and voice cloning endpoints, GPU lifecycle management, and Docker deployment. No streaming, no caching, no observability beyond the `/health` endpoint.
