# Qwen3-TTS Server — Research Analysis

## Date: 2026-02-08

## Table of Contents

1. [Current Architecture Review](#1-current-architecture-review)
2. [Community Alternatives Analyzed](#2-community-alternatives-analyzed)
3. [Head-to-Head Comparison](#3-head-to-head-comparison)
4. [Model Landscape](#4-model-landscape)
5. [Gap Analysis](#5-gap-analysis)
6. [RAM Idle Problem](#6-ram-idle-problem)
7. [Recommendations (Prioritized)](#7-recommendations-prioritized)

---

## 1. Current Architecture Review

### What We Have

A single-file FastAPI server (`server.py`, ~410 lines) providing:
- OpenAI-compatible `/v1/audio/speech` endpoint (CustomVoice generation)
- Voice cloning endpoint `/v1/audio/speech/clone`
- GPU idle unloading with configurable timeout
- Health check endpoint
- Multi-format audio output (WAV, MP3, FLAC, OGG)
- OpenAI voice name mapping (alloy, echo, etc.)

### Strengths of Current Implementation

| Feature | Implementation | Assessment |
|---------|---------------|------------|
| Concurrency safety | `asyncio.Semaphore(1)` + `run_in_executor` with dedicated `ThreadPoolExecutor` | Correct. Event loop never blocked. Only server that gets this right. |
| Idle VRAM unload | Background watchdog task, double-checked locking pattern | Correct. Critical for shared GPU on Synology NAS. |
| GPU memory cleanup | `gc.collect()` + `torch.cuda.empty_cache()` + `ipc_collect()` on unload | Correct. Frees VRAM back to system. |
| Request timeout | `asyncio.wait_for()` with configurable `REQUEST_TIMEOUT` (300s default) | Correct. Prevents hung requests. |
| Model load/unload locking | `asyncio.Lock()` with double-checked locking | Correct. Prevents race between load and unload. |
| `torch.inference_mode()` | Applied during synthesis and voice clone | Correct. Disables gradient tracking. |
| Audio format conversion | WAV/MP3/FLAC/OGG via soundfile + pydub fallback | Good. Graceful fallback if pydub not installed. |
| Docker config | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, uvloop, httptools | Good. Reduces thread sprawl and memory fragmentation. |
| Language detection | Character-range heuristic for CJK + English default | Adequate for primary use case. |
| Speed adjustment | `scipy.signal.resample` | Functional. No external dependency on librosa. |

### Weaknesses Identified

| Issue | Severity | Details |
|-------|----------|---------|
| No WebSocket streaming | **Critical** | For phone calls, caller must wait for full audio generation (~2-3s) before hearing anything. No progressive audio delivery. |
| No `torch.compile` | Medium | Missing 20-30% inference speedup from `torch.compile(mode="reduce-overhead")` after warmup. |
| `sdpa` instead of `flash_attention_2` | Medium | Flash Attention 2 is ~20% faster on long sequences. Requires `flash-attn` package. |
| No voice prompt caching | Medium | Clone endpoint reprocesses reference audio on every request. Repeated calls with same voice waste ~1s each. |
| No text normalization | Low-Medium | "$100" passed raw to model instead of "one hundred dollars". Numbers, URLs, emails not expanded. |
| 0.6B model | Medium | 1.7B-CustomVoice has measurably better quality (WER 1.24 vs higher on 0.6B). |
| Python process RAM when idle | **High** | ~1.9GB resident RAM even with model unloaded, due to PyTorch + CUDA runtime. |
| No `lifespan` context manager | Low | Uses deprecated `@app.on_event("startup")`. Should migrate to FastAPI `lifespan`. |
| Per-request GC in `_do_transcribe` but not `_do_synthesize` | Low | Inconsistency — TTS skips per-request GC (comment says "let CUDA reuse cached allocations") but design intent should be documented. |

---

## 2. Community Alternatives Analyzed

### 2a. twolven/Qwen3-TTS-Openai-Fastapi

**Architecture**: Multi-file FastAPI app with abstract backend system supporting Official and vLLM-Omni backends. OpenAI-compatible API at `/v1/audio/speech`. Static web UI included.

**Key Features**:
- Backend abstraction (Official Qwen3-TTS vs vLLM-Omni)
- `torch.compile(mode="reduce-overhead")` + `flash_attention_2`
- Text normalization engine (English numbers, money, URLs, emails, phone numbers, units, abbreviations)
- Multi-stage Dockerfile (GPU, vLLM, CPU variants)
- 9 voice presets with OpenAI-compatible naming
- Language extraction from model name (`tts-1-es`, `tts-1-hd-fr`)

**Critical Bugs Found**:

| Bug | Severity | Details |
|-----|----------|---------|
| Event loop blocking | **Critical** | `generate_speech()` is `async` but calls synchronous `model.generate_custom_voice()` directly — no `run_in_executor()`. Blocks entire event loop during inference. Health checks unresponsive during generation. |
| No concurrency control | **Critical** | No semaphore, no lock, no queue. Concurrent requests contend for GPU with no coordination. |
| Streaming not implemented | **High** | Schema accepts `stream=True`, code ignores it completely. `encode_audio_streaming()` utility exists but is never wired up. |
| No model unloading | **High** | Model loaded once, stays in GPU memory forever. No idle timeout. |
| No GPU memory management | **High** | No `torch.cuda.empty_cache()`, no OOM handling, no memory monitoring beyond health endpoint. |
| No request timeout | **High** | Requests can hang indefinitely. |
| No voice cloning endpoint | Medium | Despite README claims, no API for uploading reference audio. |
| Global singleton race | Low | Backend factory creates singleton without locking. |
| English-only text normalization | Low | `inflect` library only works for English. CJK numbers/money not handled. |

### 2b. ValyrianTech/Qwen3-TTS_server

**Architecture**: Single-file FastAPI server (`server.py`) focused on voice cloning. Uses 1.7B-Base model. Bundles Whisper for auto-transcribing reference audio.

**Key Features**:
- Voice prompt caching — pre-computes `create_voice_clone_prompt()` on first use, reuses on subsequent requests
- Reference audio processing pipeline — intelligent silence splitting, 15s clip limit, edge silence trimming
- Built-in Whisper model for auto-transcribing reference audio text
- Voice conversion endpoint (transcribe input audio, re-synthesize with target voice)
- File upload + voice management endpoints

**Critical Bugs Found**:

| Bug | Severity | Details |
|-----|----------|---------|
| Shared temp file race condition | **Critical** | All requests write to same paths: `outputs/output_synthesized.wav`, `outputs/input_audio.wav`. Concurrent requests corrupt each other's audio. |
| Event loop blocking | **Critical** | All endpoints `async def` but GPU inference is synchronous with no `run_in_executor()`. |
| No concurrency control | **Critical** | No semaphore, no lock. Concurrent requests hit GPU unserialized. |
| No model unloading | **High** | Both Qwen3-TTS and Whisper permanently resident. |
| No GPU memory management | **High** | No `empty_cache()`, no `inference_mode()`, no OOM protection. |
| No request timeout | **High** | Requests can hang indefinitely. |
| WAV only output | Medium | No MP3/FLAC/OGG options. |
| File handle leak | Medium | `StreamingResponse(open(save_path, 'rb'), ...)` — file descriptor not explicitly managed. |
| Dead code | Low | `generate_speech()` and `audio_to_wav_bytes()` defined but never called. |
| No health check | Low | No `/health` endpoint. |
| Inconsistent error handling | Low | `/upload_audio/` returns 200 with `{"error": ...}` instead of proper HTTP error codes. |

### 2c. Official QwenLM/Qwen3-ASR Server (For Reference)

The official ASR server is a 45-line wrapper around `vllm serve`. It registers the custom model architecture into vLLM's model registry and delegates all serving to vLLM (PagedAttention, continuous batching, OpenAI-compatible API, streaming, GPU memory management). Zero custom HTTP routing code.

**Philosophy**: Don't build a server — build a vLLM plugin and let vLLM handle everything.

**Note**: There is no official Qwen3-TTS server equivalent yet. The TTS model has vLLM-Omni support for offline inference, but online serving with streaming is still under development.

---

## 3. Head-to-Head Comparison

| Feature | Our Server | twolven | ValyrianTech |
|---------|-----------|---------|--------------|
| **Event loop safety** | Correct (`run_in_executor`) | **BROKEN** (blocks event loop) | **BROKEN** (blocks event loop) |
| **Concurrency control** | Correct (`Semaphore(1)`) | **NONE** | **NONE** |
| **Idle VRAM unload** | Yes (watchdog) | No | No |
| **GPU memory cleanup** | Yes (`empty_cache` + `ipc_collect`) | No | No |
| **Request timeout** | Yes (300s) | No | No |
| **Model load locking** | Yes (double-checked `asyncio.Lock`) | No | No |
| **Thread safety** | Yes (dedicated executor) | No | **BROKEN** (shared temp files) |
| **torch.compile** | No | Yes (`reduce-overhead`) | No |
| **Flash Attention** | `sdpa` | `flash_attention_2` | None |
| **Voice cloning** | Yes (API endpoint) | No | Yes (with caching) |
| **Voice prompt caching** | No | N/A | Yes |
| **Text normalization** | No | Yes (English) | No |
| **WebSocket streaming** | No | No | No |
| **Audio formats** | WAV/MP3/FLAC/OGG | WAV/MP3/Opus/AAC/FLAC | WAV only |
| **Speed control** | scipy.signal.resample | librosa.time_stretch | librosa.time_stretch |
| **Docker optimization** | Good (thread limits, uvloop) | Multi-stage (3 variants) | Basic |
| **Web UI** | No | Yes | No |
| **OpenAI API compat** | Yes | Yes | No (custom endpoints) |
| **Health check** | Yes | Yes | No |

### Verdict

**Our server is the most production-ready** of the three on fundamentals: concurrency safety, GPU resource management, timeout handling, and thread safety. The community alternatives have critical bugs that would cause failures under concurrent load.

**What the community alternatives have that we don't**: `torch.compile`, Flash Attention 2, voice prompt caching (ValyrianTech), and text normalization (twolven). These are additive features we can adopt.

---

## 4. Model Landscape

### Qwen3-TTS Family (Released 2026-01-22, Apache 2.0)

| Model | Params | Purpose | Downloads |
|-------|--------|---------|-----------|
| Qwen3-TTS-12Hz-1.7B-Base | ~2B | Voice cloning from reference audio, fine-tuning base | 428k |
| Qwen3-TTS-12Hz-1.7B-CustomVoice | ~2B | Speaker identity + optional instruction | 375k |
| Qwen3-TTS-12Hz-1.7B-VoiceDesign | ~2B | Natural-language voice description | 177k |
| **Qwen3-TTS-12Hz-0.6B-CustomVoice** | ~0.9B | **Currently deployed** | 104k |
| Qwen3-TTS-12Hz-0.6B-Base | ~0.6B | Lightweight cloning base | 162k |
| Qwen3-TTS-Tokenizer-12Hz | -- | Audio tokenizer (codec) | 46.6k |

### 0.6B vs 1.7B Performance

| Metric | 0.6B | 1.7B |
|--------|------|------|
| WER (English) | Higher | **1.24** (SOTA) |
| WER (Chinese) | Higher | **0.77** (SOTA) |
| MOS quality | Lower | **4.53** |
| First-packet latency | N/A | **97ms** |
| Speaker similarity | Lower | **0.789** (highest across 10 languages) |

**The 1.7B model is strictly better across all metrics.**

### Competing Open-Source TTS Models (2026)

| Model | Size | Strengths | Weaknesses |
|-------|------|-----------|------------|
| **Fish Speech V1.5** | varies | Highest TTS Arena ELO (1339), DualAR | Larger compute |
| **CosyVoice 3.0** | 1.5B | 1M hours training, 9 langs + 18 Chinese dialects | Alibaba ecosystem |
| **Higgs Audio V2** | 3.6B+2.2B | 75.7% win rate over gpt-4o-mini-tts on emotions | Very large |
| **Chatterbox Turbo** | 350M | MIT license, 23 langs, emotion control | English-focused |
| **IndexTTS-2** | -- | Precise duration control, best for dubbing | Narrow use case |
| **Dia 2** | 1B/2B | Multi-speaker dialogue, nonverbal sounds | Dialogue niche |

### Recommendation

Upgrade to **Qwen3-TTS-12Hz-1.7B-CustomVoice** for quality improvement. For phone call use case, Qwen3-TTS remains the best choice due to low latency, multilingual support, and manageable model size.

---

## 5. Gap Analysis

### Critical Gaps (For Phone Call Use Case)

#### Gap 1: No WebSocket Streaming for TTS

**Impact**: Callers wait 2-3s for full audio generation before hearing anything. For 2-way phone calls, this creates unnatural pauses.

**What's needed**: A WebSocket endpoint that sends audio chunks (e.g., 100-200ms of audio) as they are generated by the model. The model generates tokens autoregressively — each token can be decoded to audio and sent immediately.

**Complexity**: High. Requires hooking into the model's token generation loop or using vLLM-Omni's streaming capabilities (currently under development).

**Workaround**: Sentence splitting — break long text into short phrases and generate/send each one immediately. First audio arrives in ~200-500ms instead of waiting for full paragraph.

#### Gap 2: Python Process RAM When Idle

**Impact**: ~1.9GB resident RAM per container even with model unloaded. With 3 containers (ASR + TTS + Whisper), that's ~5.7GB wasted on a NAS.

**Root cause**: `import torch` + CUDA context initialization permanently allocates ~1-1.2GB CPU RAM that cannot be freed without killing the process. The Python interpreter, loaded libraries, and CUDA runtime are resident.

**What's needed**: Process-level idle — kill the heavy inference worker process entirely after timeout, keep only a lightweight gateway alive (~30MB). Respawn worker on next request.

**Complexity**: Medium. Requires restructuring into gateway + subprocess architecture.

### Medium Gaps

#### Gap 3: No `torch.compile`

**Impact**: Missing 20-30% inference speedup on repeated requests after initial warmup compilation.

**Fix**: Add `torch.compile(model.model, mode="reduce-overhead", fullgraph=False)` after model loading. Requires testing — some model architectures don't compile cleanly.

**Complexity**: Low (3 lines), but needs validation.

#### Gap 4: `sdpa` Instead of `flash_attention_2`

**Impact**: ~20% slower attention computation on long sequences.

**Fix**: Change `attn_implementation="sdpa"` to `attn_implementation="flash_attention_2"`. Requires installing `flash-attn` package in Docker image.

**Complexity**: Low (1 line + Dockerfile change), but `flash-attn` compilation can be slow. Use pre-built wheels.

#### Gap 5: No Voice Prompt Caching

**Impact**: Voice clone endpoint reprocesses reference audio on every request (~1s overhead). If using a consistent "agent voice" for phone calls, this is wasted computation.

**Fix**: Cache `model.create_voice_clone_prompt()` result keyed by a voice identifier. Return cached prompt on subsequent requests.

**Complexity**: Low (~20 lines).

### Low Gaps

#### Gap 6: No Text Normalization

**Impact**: "$100" → model receives "$100" literally instead of "one hundred dollars". Less critical for non-English use cases.

**Fix**: Port text normalization from twolven (English-centric) or implement CJK-aware normalization.

**Complexity**: Medium (port existing module + adapt for multilingual).

#### Gap 7: Model Size

**Impact**: 0.6B produces lower quality audio than 1.7B.

**Fix**: Change `MODEL_ID` env var to `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`. Requires more VRAM (~1GB additional).

**Complexity**: Config change only. Verify VRAM fits.

---

## 6. RAM Idle Problem

### Current State

```
Container running, model loaded:    ~1.9GB RAM + ~1.5GB VRAM
Container running, model unloaded:  ~1.9GB RAM + ~0GB VRAM   ← RAM not freed
Container stopped:                  ~0GB RAM   + ~0GB VRAM
```

### Breakdown of ~1.9GB Idle RAM

| Component | Approx RAM | Can Free? |
|-----------|-----------|-----------|
| CUDA runtime + driver context | 800MB-1.2GB | Only by killing process |
| PyTorch library (libtorch, etc.) | 300-400MB | Only by killing process |
| Python interpreter + all imports | 100-200MB | Only by killing process |
| uvicorn + FastAPI framework | 20-30MB | Keep this |

### Proposed Solution: Gateway + Worker Architecture

```
                    ┌─────────────────────────────────┐
                    │  Lightweight Gateway (30MB RAM)  │
                    │  - FastAPI + uvicorn             │
                    │  - Health check endpoint         │
                    │  - WebSocket proxy               │
                    │  - Spawns/kills worker process   │
                    └───────────────┬──────────────────┘
                                    │
                          (spawn on first request,
                           kill after idle timeout)
                                    │
                    ┌───────────────▼──────────────────┐
                    │  Heavy Worker (1.9GB RAM)         │
                    │  - import torch, load model      │
                    │  - GPU inference                  │
                    │  - Communicates via IPC/socket    │
                    └──────────────────────────────────┘
```

**Idle state**: Only gateway running → ~30MB RAM, ~0 VRAM
**Active state**: Gateway + worker → ~1.9GB RAM, ~1.5GB VRAM
**Cold start penalty**: ~5-10s to spawn worker + load model

This reduces idle RAM from ~1.9GB to ~30MB per service. Across 3 services, that's ~5.6GB saved on the NAS when idle.

---

## 7. Recommendations (Prioritized)

Priority is ordered by impact for the primary use case: **2-way phone calls via WebSocket + subtitle transcription**.

### P0 — Critical

| # | Improvement | Impact | Effort | Details |
|---|------------|--------|--------|---------|
| 1 | **Add TTS WebSocket streaming** | Reduces first-audio latency from ~2-3s to ~200-500ms for phone calls | High | Sentence splitting approach as immediate win. True token-level streaming requires model hook or vLLM-Omni. |
| 2 | **Gateway + Worker architecture** | Reclaim ~5.6GB idle RAM across services on Synology NAS | Medium | Split into lightweight gateway process + heavy worker subprocess. Kill worker after idle, respawn on demand. |

### P1 — High

| # | Improvement | Impact | Effort | Details |
|---|------------|--------|--------|---------|
| 3 | **Upgrade to 1.7B-CustomVoice** | Better audio quality for phone calls | Config change | Change `MODEL_ID` env var. Verify VRAM. |
| 4 | **Add `torch.compile`** | 20-30% faster inference | Low (3 lines) | `torch.compile(model.model, mode="reduce-overhead")` after load. Validate with test suite. |
| 5 | **Switch to `flash_attention_2`** | ~20% faster attention | Low (1 line + Docker) | Requires `flash-attn` package. Use pre-built wheel. |

### P2 — Medium

| # | Improvement | Impact | Effort | Details |
|---|------------|--------|--------|---------|
| 6 | **Voice prompt caching** | ~1s faster per clone call with same voice | Low (~20 lines) | Cache `create_voice_clone_prompt()` keyed by voice ID. Add cache management endpoint. |
| 7 | **Migrate to `lifespan` context manager** | Future-proofs against FastAPI deprecation | Low | Replace `@app.on_event("startup")` with `@asynccontextmanager async def lifespan(app)`. |

### P3 — Low

| # | Improvement | Impact | Effort | Details |
|---|------------|--------|--------|---------|
| 8 | **Text normalization** | Better pronunciation of numbers, URLs, money | Medium | Port from twolven or build CJK-aware version. |
| 9 | **Add process title** | Easier identification in htop | Trivial | `import setproctitle; setproctitle.setproctitle("qwen3-tts")` |

---

## Appendix: Community Server Reference

### twolven/Qwen3-TTS-Openai-Fastapi
- GitHub: https://github.com/twolven/Qwen3-TTS-Openai-Fastapi
- License: Not specified
- Worth stealing: `torch.compile`, `flash_attention_2`, text normalization module, multi-backend architecture

### ValyrianTech/Qwen3-TTS_server
- GitHub: https://github.com/ValyrianTech/Qwen3-TTS_server
- License: MIT
- Worth stealing: Voice prompt caching pattern, reference audio processing pipeline (silence splitting, 15s clip, edge trim)

### Official QwenLM/Qwen3-TTS
- GitHub: https://github.com/QwenLM/Qwen3-TTS
- License: Apache 2.0
- Note: No production server implementation yet. vLLM-Omni offline inference supported; online serving under development.
