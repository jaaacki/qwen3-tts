# TTS Performance Optimizations Design

**Date:** 2026-03-01
**Status:** Approved
**Goal:** Reduce inference latency and improve streaming throughput for real-time voice conversations.

## Context

The TTS server is the highest-latency component in the voice call pipeline (FreePBX → ASR → LLM → **TTS** → caller). The server already has flash_attention_2, torch.compile (reduce-overhead), TF32, bfloat16, adaptive token budget, voice prompt pre-computation, and audio caching. This design covers the remaining optimizations.

**Hardware:** RTX 4060 8GB VRAM (Ada Lovelace, sm_89), shared with ASR (~3.5GB).
**Deployment:** Always-on (`IDLE_TIMEOUT=0`, `PRELOAD_MODEL=true`).

## Changes

### 1. torch.compile max-autotune

**File:** `server.py` (model loading section)

Switch `torch.compile(model.model, mode="reduce-overhead")` to `mode="max-autotune"`. This mode runs autotuning on CUDA kernels at warmup time, selecting the fastest kernel for each operation on the specific GPU.

- Warmup time increases ~30-60s (acceptable for always-on deployment)
- Steady-state inference faster due to optimized kernel selection
- Env var `TORCH_COMPILE_MODE` to override (default `max-autotune`)

### 2. INT8 Quantization

**File:** `server.py` (model loading section), `compose.yaml`

Enable bitsandbytes INT8 weight-only quantization by default. bitsandbytes is already installed in the Docker image.

- Default `QUANTIZE=int8` in compose.yaml
- `QUANTIZE=""` or `QUANTIZE=none` to disable
- ~50% VRAM reduction (~1.2GB instead of ~2.4GB)
- Ada architecture (sm_89) has native INT8 tensor core support
- Existing `quant_kwargs` plumbing in `_load_model_sync()` should already handle this — verify and wire up

### 3. Sentence Pipelining in Streaming

**Files:** `server.py` (streaming endpoints: `/stream/pcm`, `/stream`, `/ws`)

Currently each streaming endpoint processes sentences sequentially:
```
synthesize sentence 1 → yield → synthesize sentence 2 → yield → ...
```

Change to pipelined:
```
synthesize sentence 1 → yield sentence 1 + submit sentence 2 → yield sentence 2 + submit sentence 3 → ...
```

Implementation:
- After submitting sentence N to `_infer_queue`, immediately submit sentence N+1 as a pre-fetch
- Store the pre-fetched future
- When yielding sentence N's audio, await the pre-fetched future for sentence N+1
- Use `asyncio.ensure_future()` to avoid blocking the yield loop
- Maintain sentence ordering (futures resolve in submission order since GPU is single-threaded)

### 4. Server-Side Sample Rate Conversion

**File:** `server.py` (all speech endpoints)

Add optional `sample_rate` parameter to request bodies. When set, server resamples audio before returning/yielding.

- Accepted on `/v1/audio/speech`, `/v1/audio/speech/stream`, `/v1/audio/speech/stream/pcm`, `/v1/audio/speech/ws`, `/v1/audio/speech/clone`
- Default: `null` (return native 24kHz)
- Common value: `8000` (telephony)
- Resampling via `scipy.signal.resample()` (already imported)
- For streaming endpoints, resample each chunk before yielding
- Report actual sample rate in SSE metadata and streaming response headers

### 5. CUDA Graphs

**File:** `server.py` (inference section)

Capture CUDA graphs during warmup for common input shapes. Replay on subsequent calls to eliminate kernel launch overhead.

- Capture graphs for 2-3 representative input lengths during warmup (short/medium/long)
- On inference, check if input shape matches a captured graph
- If match: replay graph (fast path)
- If no match: fall back to eager execution (current behavior)
- Env var `CUDA_GRAPHS=true` (default true) to enable/disable
- Only applicable when `torch.compile` is also enabled
- Graph capture happens after torch.compile warmup

### 6. Pinned Memory

**File:** `server.py` (inference and audio output sections)

Pre-allocate pinned (page-locked) CPU memory buffers for audio output transfer.

- Allocate pinned buffer pool during model load (e.g., 4 buffers of max expected audio size)
- Use `torch.cuda.HostAllocator` or `torch.empty(..., pin_memory=True)`
- Copy GPU inference output to pinned buffer, then to numpy
- Enables async DMA transfers via CUDA streams
- Fall back to regular memory if pinned allocation fails

### 7. CUDA Streams

**File:** `server.py` (inference section)

Use dedicated CUDA streams to overlap compute with data transfer.

- Create inference stream and transfer stream during model load
- Run model forward pass on inference stream
- Copy results to CPU on transfer stream
- Synchronize only when audio data is needed for encoding
- Combined with pinned memory, enables true async pipeline

## Environment Variables (new/changed)

| Variable | Default | Description |
|----------|---------|-------------|
| `TORCH_COMPILE_MODE` | `max-autotune` | torch.compile mode (`max-autotune`, `reduce-overhead`, `default`) |
| `QUANTIZE` | `int8` | Quantization (`int8`, `fp8`, or empty to disable) |
| `CUDA_GRAPHS` | `true` | Enable CUDA graph capture and replay |

## What We're NOT Doing

- **vLLM backend** — Major architecture change, separate effort
- **KV cache reuse** — Requires upstream qwen-tts library changes
- **ONNX/TensorRT export** — Model uses custom architecture, export compatibility uncertain
- **Speculative decoding** — Requires two models loaded, VRAM constraint

## Testing

- All changes must pass existing `server_test.py` unit tests
- Latency comparison: measure time-to-first-byte and total synthesis time before/after
- VRAM usage comparison: check `/health` endpoint GPU memory reporting
- Streaming quality: verify no audio gaps or ordering issues with sentence pipelining
- INT8 quality: subjective listening test for degradation

## Risk

- CUDA Graphs may not work with dynamic-shape TTS generation — fall back to eager if capture fails
- INT8 may affect voice quality — reversible via env var
- Sentence pipelining adds complexity to streaming error handling — ensure abort/barge-in still works correctly
