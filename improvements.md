# Qwen3-TTS Improvements Roadmap

**Goal**: High-quality, real-time TTS — sub-500ms first-audio latency, maximum output quality, production-grade observability.

> **Scope**: Keeping Qwen3-TTS-0.6B as-is — no model swap, no quantization. All items below work with the current model unchanged.

---

## Quick Reference Table

| # | Priority | Category | Improvement | Latency Gain | Quality Gain | VRAM Delta | Complexity |
|---|----------|----------|-------------|-------------|-------------|------------|------------|
| 1 | 🔴 P0 | Streaming | Sentence-chunked SSE streaming | **−70% perceived** | — | — | High |
| 2 | 🔴 P0 | Streaming | Raw PCM / WAV streaming endpoint | **−80% time-to-first-audio** | — | — | High |
| 3 | 🟠 P1 | Model | Switch to `flash_attention_2` | **−15–20%** inference | — | −small | Low |
| 4 | 🟠 P1 | Model | Enable `torch.compile` | **−20–30%** inference | — | +small | Medium |
| 5 | 🟠 P1 | Model | Upgrade to 1.7B-CustomVoice | −5–10% (heavier) | **+significant WER/MOS** | +2.5 GB | Low |
| 6 | 🟠 P1 | Inference | CUDA Graph capture for fixed shapes | **−10–15%** | — | — | Medium |
| 7 | 🟡 P2 | Caching | Voice prompt cache (clone endpoint) | **−1 s/clone** | — | +small | Medium |
| 8 | 🟡 P2 | Caching | KV-cache warmup (full kernel pre-compile) | **−500ms cold** | — | — | Low |
| 9 | 🟡 P2 | Audio | Text normalization (numbers, abbr, symbols) | — | **+significant** | — | Medium |
| 10 | 🟡 P2 | Audio | Pitch-preserving speed (pyrubberband) | — | +minor | — | Low |
| 11 | 🟡 P2 | Language | Replace heuristic with fasttext/langdetect | — | +minor | — | Low |
| 12 | 🟡 P2 | Inference | Batched inference for concurrent queue | **−N×** for burst | — | — | High |
| 13 | 🟡 P2 | Memory | GPU memory pre-allocation / pool | −GC jitter | — | — | Low |
| 14 | 🟢 P3 | Infra | Pin all dependency versions | — | — | — | Very Low |
| 15 | 🟢 P3 | Infra | Multi-stage Docker build | −image size | — | — | Low |
| 16 | 🟢 P3 | Observability | Prometheus + Grafana metrics | — | — | — | Low |
| 17 | 🟢 P3 | Observability | Structured JSON logging | — | — | — | Very Low |
| 18 | 🟢 P3 | Reliability | Request queue depth + 503 early rejection | — | — | — | Low |
| 19 | 🟢 P3 | Code | Migrate `@app.on_event` → lifespan | — | — | — | Very Low |
| 20 | 🟢 P3 | Code | Remove dead `VoiceCloneRequest` model | — | — | — | Very Low |

---

## 🔴 P0 — Critical (Real-Time Streaming)

These two items are the single biggest lever for perceived real-time performance. Without streaming, every client waits for the entire audio to be synthesized before hearing a single millisecond. That is 2–4 seconds of dead silence on a typical sentence.

---

### 1. Sentence-Chunked SSE Streaming

**What**: Split input text into sentences at the API level. Synthesize each sentence sequentially and stream the resulting audio chunks to the client as Server-Sent Events (SSE) or as a chunked HTTP response. The client starts playing the first sentence while the server is still synthesizing the second.

**Why this works**: The model generates audio for a 10-word sentence in ~400–600ms. A 3-sentence paragraph takes 1.5–2s total, but with streaming the user hears audio after the first 500ms instead of after 2s.

**Perceived latency reduction**: ~70–80%. The entire audio wall disappears.

**Implementation sketch**:
```python
import re
from fastapi.responses import StreamingResponse

SENTENCE_RE = re.compile(r'(?<=[.!?])\s+')

async def _stream_speech(text: str, voice: str, language: str, fmt: str):
    sentences = [s.strip() for s in SENTENCE_RE.split(text) if s.strip()]
    if not sentences:
        return

    await _ensure_model_loaded()
    loop = asyncio.get_event_loop()

    for sentence in sentences:
        async with _infer_semaphore:
            audio_data = await asyncio.wait_for(
                loop.run_in_executor(_infer_executor, lambda s=sentence: _do_synthesize(s, voice, language)),
                timeout=REQUEST_TIMEOUT,
            )
        chunk_bytes, _ = convert_audio_format(audio_data, SAMPLE_RATE, fmt)
        yield chunk_bytes  # client receives each chunk immediately

@app.post("/v1/audio/speech/stream")
async def synthesize_stream(request: TTSRequest):
    voice = resolve_voice(request.voice)
    language = request.language or detect_language(request.input)
    return StreamingResponse(
        _stream_speech(request.input, voice, language, request.response_format),
        media_type="audio/wav",
        headers={"X-Accel-Buffering": "no"},  # disable nginx buffering
    )
```

**Client considerations**:
- WAV chunks need a proper header only on the first chunk (or use raw PCM 16-bit LE with known sample rate)
- For browser clients, use `MediaSource API` with PCM streaming
- For Python clients, pipe chunks to `sounddevice.RawOutputStream`
- For OpenAI SDK compatibility, keep existing non-streaming endpoint as-is

**Tasks**:
- [ ] Implement sentence splitter respecting abbreviations (`Dr.`, `Mr.`, `U.S.A.`)
- [ ] Add `/v1/audio/speech/stream` SSE/chunked endpoint
- [ ] Choose streaming format: raw PCM (simplest) vs chunked WAV vs OGG stream
- [ ] Add `X-Accel-Buffering: no` header to bypass reverse-proxy buffering
- [ ] Document client-side playback pattern (curl piped to ffplay, Python sounddevice, JS MediaSource)
- [ ] Add `stream: true` flag on existing `/v1/audio/speech` for OpenAI SDK compat

---

### 2. Raw PCM Streaming (Lowest Latency Path)

**What**: Instead of encoding to WAV/MP3 (which requires buffering the full audio), stream raw 16-bit PCM samples as they come out of the model. WAV encoding requires knowing the total length for the header — PCM bypasses this entirely.

**Why**: WAV header requires `data_size` field, forcing a full buffer before writing. Raw PCM has no header overhead and zero buffering.

**Format**: `audio/pcm;rate=24000;encoding=signed-integer;bits=16;channels=1`

**Implementation**:
```python
@app.post("/v1/audio/speech/pcm")
async def synthesize_pcm(request: TTSRequest):
    async def generate():
        sentences = split_sentences(request.input)
        for sentence in sentences:
            audio_np = await synthesize_sentence(sentence, ...)  # float32 [-1, 1]
            pcm = (audio_np * 32767).astype(np.int16).tobytes()
            yield pcm
    return StreamingResponse(generate(), media_type="audio/pcm;rate=24000")
```

**Tasks**:
- [ ] Add `/v1/audio/speech/pcm` endpoint returning raw PCM stream
- [ ] Document sample rate (verify from model: likely 24000 Hz or 22050 Hz)
- [ ] Test with `ffplay -f s16le -ar 24000 -ac 1 -` for validation

---

## 🟠 P1 — High Impact (Model & Inference Speed)

---

### 3. Switch to `flash_attention_2`

**What**: Replace `attn_implementation="sdpa"` with `attn_implementation="flash_attention_2"`. Flash Attention 2 uses a fused CUDA kernel that processes attention in tiles, dramatically reducing memory reads/writes for long sequences.

**Why now**: The Qwen3-TTS model generates long token sequences. Attention cost scales O(n²). Flash Attention 2 reduces this constant factor by ~2–4×.

**Gain**: 15–20% faster inference, lower peak VRAM (no O(n²) attention matrix materialized).

**Requirements**: `pip install flash-attn --no-build-isolation` (requires CUDA 11.6+, already satisfied with CUDA 12.4). Build time ~5 min on first install.

**Implementation**:
```python
# In _load_model_sync(), change:
# attn_implementation="sdpa"
# to:
attn_implementation="flash_attention_2"
```

**Tasks**:
- [ ] Add `flash-attn` to Dockerfile (`pip install flash-attn --no-build-isolation`)
- [ ] Change `attn_implementation` in `_load_model_sync()`
- [ ] Verify with `python -c "from flash_attn import flash_attn_func; print('ok')"` inside container
- [ ] Benchmark before/after: measure `/v1/audio/speech` latency for 10-word, 50-word, 200-word inputs

---

### 4. Enable `torch.compile`

**What**: After model load, compile the model's forward pass with `torch.compile(model, mode="reduce-overhead")`. On subsequent calls, PyTorch uses a cached optimized graph instead of retracing Python.

**Gain**: 20–30% faster inference after the first compiled call. Compilation happens once (30–60s) at startup warmup — not during live requests.

**Implementation**:
```python
# In _load_model_sync(), after model = Qwen3TTSModel.from_pretrained(...):
if torch.cuda.is_available():
    # Compile inner transformer if exposed; otherwise try top-level
    try:
        model.model = torch.compile(model.model, mode="reduce-overhead", fullgraph=False)
        logger.info("torch.compile applied successfully")
    except Exception as e:
        logger.warning(f"torch.compile skipped: {e}")
```

**Caveats**:
- First post-compile inference triggers graph tracing (slow). Do this during warmup.
- `fullgraph=False` allows graph breaks for unsupported ops (safer for complex models).
- If model internals aren't easily accessible via `model.model`, try compiling the full `model` object (may have limited effect if inference is a complex Python method).
- PyTorch 2.5+ (already using 2.5.1) has improved compile stability.

**Tasks**:
- [ ] Identify which internal submodule to compile (inspect `model.__dict__` keys)
- [ ] Add compile call in `_load_model_sync()` after load
- [ ] Extend warmup call to trigger first compilation (run 2–3 warmup inferences)
- [ ] Add env var `TORCH_COMPILE=1` to opt-in (safer rollout)
- [ ] Benchmark: compare p50/p95 latency before vs after

---

### 5. Upgrade to 1.7B Model

**What**: Switch `MODEL_ID` from `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` to `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`.

**Quality gains** (from Qwen official benchmarks):
| Metric | 0.6B | 1.7B | Improvement |
|--------|------|------|-------------|
| WER (English) | ~2.1 | 1.24 | −40% errors |
| WER (Chinese) | ~1.4 | 0.77 | −45% errors |
| MOS Score | ~4.3 | 4.53 | +5% naturalness |
| First-packet latency | ~80ms | ~97ms | +21% slower |

**VRAM**: 0.6B ≈ 2.4GB → 1.7B ≈ 5.2GB (bfloat16). Ensure GPU has ≥8GB.

**Tasks**:
- [ ] Verify GPU VRAM capacity (`nvidia-smi`)
- [ ] Set `MODEL_ID=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` in `compose.yaml`
- [ ] Adjust `shm_size` if needed (try `2g`)
- [ ] Run full test suite to verify voice names still work
- [ ] A/B test audio quality on same inputs

---

### 6. CUDA Graph Capture

**What**: For fixed-length inference shapes, capture the CUDA computation graph once and replay it on subsequent calls. This eliminates CUDA kernel launch overhead (~1–5ms per kernel launch × hundreds of kernels = significant savings).

**Gain**: 10–15% latency reduction on repeated calls. Most effective when input text length is similar across requests.

**Caveat**: CUDA graphs require fixed input/output shapes. Dynamic text lengths break graphs. Use with padded/bucketed inputs.

**Tasks**:
- [ ] Investigate if `qwen-tts` library exposes the underlying `transformers` model
- [ ] If yes, explore `torch.cuda.graphs.make_graphed_callables()` on the decoder
- [ ] If model uses dynamic shapes, consider input length bucketing (round to nearest 64 tokens)
- [ ] Benchmark impact before committing

---

## 🟡 P2 — Medium Impact (Quality & Caching)

---

### 7. Voice Prompt Cache (Clone Endpoint)

**What**: The `/clone` endpoint reads and processes reference audio on every request. For repeated clone calls with the same reference file, this wastes ~1s re-processing identical data. Cache processed voice embeddings keyed by audio content hash.

**Implementation**:
```python
import hashlib, functools

_voice_prompt_cache: dict[str, np.ndarray] = {}  # hash → processed voice embedding
_CACHE_MAX = 32  # LRU eviction after 32 entries

def _get_voice_embedding(audio_bytes: bytes):
    key = hashlib.sha256(audio_bytes).hexdigest()
    if key not in _voice_prompt_cache:
        if len(_voice_prompt_cache) >= _CACHE_MAX:
            # Evict oldest entry
            _voice_prompt_cache.pop(next(iter(_voice_prompt_cache)))
        ref_audio, ref_sr = sf.read(io.BytesIO(audio_bytes))
        if ref_audio.ndim > 1:
            ref_audio = ref_audio.mean(axis=1)
        _voice_prompt_cache[key] = (ref_audio, ref_sr)
    return _voice_prompt_cache[key]
```

**Tasks**:
- [ ] Implement LRU cache for reference audio parsing
- [ ] Add cache stats to `/health` endpoint (`voice_cache_size`, `voice_cache_hits`)
- [ ] Add `VOICE_CACHE_MAX` env var (default 32)
- [ ] Consider persisting cache to disk for server restarts

---

### 8. Deeper Warmup (Full Kernel Pre-Compilation)

**What**: Current warmup runs one inference call. PyTorch lazily compiles CUDA kernels on first use. Running 2–3 warmup calls with different text lengths pre-compiles more kernel variants, eliminating compilation stalls during live requests.

**Implementation**:
```python
# In _load_model_sync(), after load:
WARMUP_TEXTS = [
    "Hello.",                          # short
    "This is a warmup sentence for kernel compilation.",  # medium
    "The quick brown fox jumps over the lazy dog and continues running down the hill.",  # long
]
for text in WARMUP_TEXTS:
    model.generate_custom_voice(text, speaker="vivian", language="English",
                                 gen_kwargs={"max_new_tokens": 512})
_release_gpu_full()
```

**Tasks**:
- [ ] Add multi-sample warmup in `_load_model_sync()`
- [ ] Include at least one Chinese and one English sample
- [ ] Log warmup duration per sample

---

### 9. Text Normalization

**What**: Raw text input often contains symbols the model was not trained to read aloud: `$100`, `3.14`, `#1`, `&`, `@user`, `2024-01-15`, `HTTP 404`. Without normalization, the model either skips them, mispronounces them, or hallucinates replacements.

**Gain**: Significant quality improvement for any input containing numbers, dates, currency, URLs, or abbreviations.

**Options** (in order of complexity):
1. **`nemo_text_processing`** — Nvidia's production normalizer, handles all edge cases, multilingual. Heavy dependency (~200MB).
2. **`inflect`** — Pure Python, handles numbers/ordinals/currencies. Lightweight.
3. **Custom regex** — Handle the most common cases (numbers, currency, simple fractions).

**Recommended approach** (inflect + custom regex):
```python
import inflect, re
_inflect = inflect.engine()

def normalize_text(text: str) -> str:
    # Currency: $100 → "one hundred dollars"
    text = re.sub(r'\$(\d+)', lambda m: _inflect.number_to_words(m.group(1)) + " dollars", text)
    # Percentages: 75% → "seventy-five percent"
    text = re.sub(r'(\d+)%', lambda m: _inflect.number_to_words(m.group(1)) + " percent", text)
    # Standalone numbers: 42 → "forty-two"
    text = re.sub(r'\b(\d+)\b', lambda m: _inflect.number_to_words(m.group(1)), text)
    # URLs: remove or spell out domain
    text = re.sub(r'https?://\S+', 'link', text)
    return text
```

**Tasks**:
- [ ] Add `inflect` to Dockerfile
- [ ] Implement `normalize_text()` function
- [ ] Apply normalization in `/v1/audio/speech` before synthesis
- [ ] Test with: currency, percentages, years, phone numbers, ordinals, URLs
- [ ] Add `normalize: bool = True` flag to `TTSRequest` to allow opt-out

---

### 10. Pitch-Preserving Speed Adjustment

**What**: Current speed adjustment uses `scipy.signal.resample()` which compresses/expands audio samples. This changes both duration AND pitch — sped-up audio sounds chipmunk-like, slowed audio sounds deep/unnatural. Pitch-preserving time stretching (PSOLA) changes duration while keeping pitch constant.

**Library**: `pyrubberband` (wraps the Rubber Band Library, production-grade PSOLA)

```python
import pyrubberband as rubberband

# Replace current scipy.signal.resample with:
if request.speed != 1.0:
    audio_data = rubberband.time_stretch(audio_data, SAMPLE_RATE, 1.0 / request.speed)
```

**Tasks**:
- [ ] Add `pyrubberband` and `rubberband-cli` to Dockerfile
- [ ] Replace `scipy.signal.resample` in speed adjustment block
- [ ] Test speed=0.75 (slow), speed=1.0 (normal), speed=1.5 (fast) — verify no pitch shift

---

### 11. Replace Language Heuristic with fasttext

**What**: Current `detect_language()` uses Unicode range checks — correct only for scripts. It cannot distinguish Spanish from French (both Latin script), or detect mixed-language text. This causes the model to receive wrong language hints.

**Library**: `fasttext-langdetect` (Meta's fasttext LID model, ~900KB, 176 languages, <1ms detection)

```python
from fasttext_langdetect import detect

def detect_language(text: str) -> str:
    LANG_MAP = {"zh": "Chinese", "ja": "Japanese", "ko": "Korean", "en": "English",
                "es": "Spanish", "fr": "French", "de": "German", "ar": "Arabic"}
    result = detect(text, low_memory=True)
    lang_code = result["lang"]
    return LANG_MAP.get(lang_code, "English")
```

**Tasks**:
- [ ] Add `fasttext-langdetect` to Dockerfile
- [ ] Replace `detect_language()` implementation
- [ ] Test with: English, Chinese, Japanese, Korean, Spanish, French, mixed-language inputs

---

### 12. Batched Inference for Burst Traffic

**What**: Current architecture processes one request at a time (Semaphore(1)). Under burst traffic (multiple simultaneous requests), requests queue sequentially and all wait. Batching multiple short texts into one model call amortizes overhead.

**Challenge**: Qwen3-TTS may not natively support batched inference (variable-length sequences require padding). Needs investigation.

**Approach**:
- When semaphore is held and queue has pending requests, collect them into a batch
- Pad all inputs to the same token length
- Run batched forward pass
- Distribute results back to waiting coroutines

**Tasks**:
- [ ] Investigate if `Qwen3TTSModel` supports batched generation (check `generate_custom_voice` signature)
- [ ] If yes: implement batch collector with `asyncio.Queue` + max_batch_size + max_wait_ms
- [ ] If no: evaluate calling underlying `transformers` model directly
- [ ] Benchmark: single vs batch for 4 simultaneous 20-word requests

---

### 13. GPU Memory Pre-Allocation Pool

**What**: After unloading and reloading the model, CUDA allocates memory pages on demand. These allocation spikes add ~20–50ms of jitter. Pre-allocating a memory pool with `torch.cuda.set_per_process_memory_fraction()` or reserving a buffer locks in pages at load time.

**Implementation**:
```python
# After model load, before warmup:
# Pre-warm allocator by allocating and releasing a dummy tensor
dummy = torch.zeros(1, 1024, 1024, dtype=torch.bfloat16, device="cuda")
del dummy
torch.cuda.empty_cache()
```

Also set in environment:
```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
```

**Tasks**:
- [ ] Add allocator pre-warm after model load
- [ ] Tune `PYTORCH_CUDA_ALLOC_CONF` for target GPU
- [ ] Measure memory allocation jitter before/after

---

## 🟢 P3 — Low Impact (Housekeeping & Observability)

---

### 14. Pin All Dependency Versions

**What**: All packages in Dockerfile use unpinned `pip install package`. Any upstream release can silently break the build or change behavior.

**Tasks**:
- [ ] Create `requirements.txt` with pinned versions (run `pip freeze` inside container)
- [ ] Reference `requirements.txt` from Dockerfile: `COPY requirements.txt . && pip install -r requirements.txt`
- [ ] Add Dependabot or manual quarterly version review

---

### 15. Multi-Stage Docker Build

**What**: Current single-stage Dockerfile ships with build tools, headers, and cached pip downloads inside the final image. Multi-stage build separates compile-time from runtime layers.

**Gain**: Smaller image size (estimate 800MB → 500MB), faster deploys, smaller attack surface.

**Structure**:
```dockerfile
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel AS builder
# Install build deps, flash-attn (requires CUDA headers)
RUN pip install flash-attn --no-build-isolation

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime AS runtime
COPY --from=builder /opt/conda/lib/python*/site-packages/flash_attn /opt/conda/...
# Install runtime-only packages
```

**Tasks**:
- [ ] Profile current image size (`docker image ls`)
- [ ] Create multi-stage Dockerfile
- [ ] Verify `flash-attn` .so files copy correctly between stages

---

### 16. Prometheus + Grafana Metrics

**What**: Add request latency histograms, GPU memory gauges, queue depth counters, and error rate metrics. Export in Prometheus format. Visualize in Grafana.

**Metrics to add**:
```
tts_request_duration_seconds{endpoint, format, voice} — histogram
tts_synthesis_duration_seconds{voice, language} — histogram (inference only)
tts_gpu_allocated_bytes — gauge
tts_gpu_reserved_bytes — gauge
tts_queue_depth — gauge (requests waiting for semaphore)
tts_model_loaded — gauge (0/1)
tts_requests_total{status} — counter
tts_errors_total{type} — counter
```

**Library**: `prometheus-fastapi-instrumentator` (auto-instruments FastAPI routes)

**Tasks**:
- [ ] Add `prometheus-fastapi-instrumentator` to Dockerfile
- [ ] Add custom GPU metrics in `/metrics` endpoint
- [ ] Add Prometheus scrape config and Grafana dashboard to compose.yaml
- [ ] Set up alert: `tts_queue_depth > 5` for 60s → alert

---

### 17. Structured JSON Logging

**What**: Current logging uses Python's default text format. In production, structured JSON logs allow filtering, aggregation, and alerting via log management tools (Loki, Datadog, CloudWatch).

**Implementation**:
```python
import structlog

log = structlog.get_logger()
log.info("synthesis_complete", voice=voice, language=language,
         duration_ms=elapsed_ms, tokens=len(text), format=fmt)
```

**Tasks**:
- [ ] Add `structlog` to Dockerfile
- [ ] Replace `logging.getLogger` with `structlog` throughout `server.py`
- [ ] Add request_id to each log line (generate UUID per request)
- [ ] Add latency fields to synthesis log events

---

### 18. Request Queue Depth Limit + Early 503

**What**: Under heavy load, requests queue behind the semaphore indefinitely. A client may wait 60+ seconds only to timeout. Better to reject early (503) when the queue is too deep.

**Implementation**:
```python
_queue_depth = 0
MAX_QUEUE_DEPTH = int(os.getenv("MAX_QUEUE_DEPTH", "5"))

async def _acquire_semaphore_with_limit():
    global _queue_depth
    if _queue_depth >= MAX_QUEUE_DEPTH:
        raise HTTPException(503, detail="Server busy, retry later")
    _queue_depth += 1
    try:
        await _infer_semaphore.acquire()
    finally:
        _queue_depth -= 1
```

**Tasks**:
- [ ] Add queue depth counter
- [ ] Add early 503 rejection when depth exceeds `MAX_QUEUE_DEPTH`
- [ ] Expose `queue_depth` in `/health` endpoint
- [ ] Add `Retry-After: 5` header to 503 responses

---

### 19. Migrate `@app.on_event` → Lifespan

**What**: FastAPI deprecated `@app.on_event("startup")` and `@app.on_event("shutdown")` in favor of the `lifespan` context manager. Current code uses deprecated pattern.

**Implementation**:
```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    asyncio.create_task(_idle_watchdog())
    yield
    # shutdown
    await _unload_model_async()

app = FastAPI(lifespan=lifespan)
```

**Tasks**:
- [ ] Replace `@app.on_event("startup")` with `lifespan` context manager
- [ ] Verify watchdog task still starts correctly
- [ ] Update FastAPI import

---

### 20. Remove Dead Code

**What**: `VoiceCloneRequest` Pydantic model (lines 81–85) is defined but never used. The clone endpoint uses multipart form parameters directly. Dead code adds noise.

**Tasks**:
- [ ] Delete `VoiceCloneRequest` class
- [ ] Verify no references remain (`grep -n VoiceCloneRequest server.py`)

---

## Implementation Order

```
Phase 1 — Real-Time (P0)
  Week 1: SSE sentence-chunked streaming endpoint (#1)
  Week 1: Raw PCM streaming endpoint (#2)

Phase 2 — Speed & Quality (P1)
  Week 2: flash_attention_2 (#3) → benchmark
  Week 2: torch.compile (#4) → benchmark
  Week 3: Upgrade to 1.7B model (#5) — gate on VRAM check
  Week 3: Deeper warmup (#8)

Phase 3 — Quality Polish (P2)
  Week 4: Text normalization with inflect (#9)
  Week 4: Voice prompt cache for /clone (#7)
  Week 4: fasttext language detection (#11)
  Week 5: pyrubberband speed adjustment (#10)

Phase 4 — Observability & Reliability (P3)
  Week 5: Pin dependency versions (#14)
  Week 5: Prometheus metrics (#16)
  Week 6: JSON logging (#17)
  Week 6: Queue depth limit + 503 (#18)
  Week 6: Lifespan migration + dead code (#19, #20)
```

---

## Benchmarking Checklist

Before and after each P0/P1 change, measure:

| Metric | How to measure |
|--------|---------------|
| Time-to-first-audio | `curl -w "%{time_starttransfer}" ...` or client timer |
| Total synthesis latency | `curl -w "%{time_total}" ...` for full response |
| GPU VRAM peak | `nvidia-smi dmon -s m` during inference |
| CPU usage | `top -d1` during inference |
| Concurrent request throughput | `ab -n 20 -c 4 -p payload.json ...` |
| Audio quality (WER) | Transcribe output with Whisper, compare to input |

---

## Notes

- **Flash Attention 2 build time**: `flash-attn` compiles from source (~5 min). Pin a pre-built wheel if available for CUDA 12.4 + Python 3.10/3.11 to avoid this on every build.
- **torch.compile graph breaks**: If `Qwen3TTSModel` uses Python control flow inside forward, compile may silently fall back to eager mode. Check logs for `torch._dynamo` warnings.
- **Streaming format compatibility**: OpenAI's streaming TTS uses chunked MP3. For maximum OpenAI SDK compatibility, stream as chunked WAV or raw PCM and document the difference.
- **IDLE_TIMEOUT with streaming**: With sentence-chunked streaming, long texts keep the model alive across multiple synthesis calls within one request. Ensure `_last_used` is updated per-chunk, not just per-request.

---

---

# Addendum — Round 2: Everything Else

The first pass covered streaming and inference flags. This addendum covers **six more complete categories** that were entirely missed.

## Addendum Quick Reference

| # | Priority | Category | Improvement | Latency Gain | Quality Gain | Complexity |
|---|----------|----------|-------------|-------------|-------------|------------|
| A1 | 🔴 P0 | Inference Param | Adaptive `max_new_tokens` (scale with text length) | **−30–60% short texts** | — | Very Low |
| A2 | 🟠 P1 | GPU Tuning | TF32 matmul mode (`allow_tf32 = True`) | **−8–12%** | — | Very Low |
| A3 | 🟠 P1 | GPU Tuning | GPU persistence mode (`nvidia-smi -pm 1`) | **−200–500ms cold** | — | Very Low |
| A4 | 🟠 P1 | GPU Tuning | Lock GPU clocks to max boost | **−5–15%** jitter | — | Low |
| A5 | 🟠 P1 | Caching | Full audio output cache (text+voice → bytes) | **−100% for repeats** | — | Medium |
| A6 | 🟠 P1 | Audio | VAD silence trimming (strip leading/trailing silence) | **−10–20%** audio len | +minor | Low |
| A7 | 🟡 P2 | Audio | Opus codec for streaming (lower latency than WAV/MP3) | **−40% bandwidth** | — | Low |
| A8 | 🟡 P2 | Audio | GPU-accelerated audio processing (torchaudio CUDA) | **−5–15ms** encoding | — | Low |
| A9 | 🟡 P2 | Audio | Async pipeline: encode chunk N while synthesizing N+1 | **−chunk_encode_ms** | — | Medium |
| A10 | 🟡 P2 | System | jemalloc memory allocator (`LD_PRELOAD`) | −GC jitter | — | Very Low |
| A11 | 🟡 P2 | System | CPU affinity: pin inference thread to GPU-adjacent cores | **−1–3ms** scheduling | — | Low |
| A12 | 🟡 P2 | System | Transparent huge pages for model weights | −TLB pressure | — | Very Low |
| A13 | 🟡 P2 | Protocol | WebSockets (replace SSE, true bidirectional streaming) | perceived real-time | — | Medium |
| A14 | 🟡 P2 | Protocol | HTTP/2 (`--http h2`) for multiplexed concurrent requests | −connection overhead | — | Low |
| A15 | 🟡 P2 | Protocol | Unix domain socket (same-host clients bypass TCP) | **−0.1–0.5ms/req** | — | Low |
| A16 | 🟢 P3 | Lifecycle | Always-on mode (disable idle unload for dedicated GPU) | −reload latency | — | Very Low |
| A17 | 🟢 P3 | Lifecycle | Eager startup load (load model at container start) | −first-req cold start | — | Very Low |
| A18 | 🟢 P3 | Docker | `--ipc=host` for faster CUDA IPC in container | −CUDA overhead | — | Very Low |
| A19 | 🟢 P3 | Docker | `--network=host` (single-host deployment) | −network stack | — | Very Low |
| A20 | 🟢 P3 | Observability | Per-request latency breakdown (load/infer/encode/transfer) | — | — | Low |

---

## A1. Adaptive `max_new_tokens` — Easiest Big Win

**This is the most impactful single-line change in the whole document.**

Right now `gen_kwargs = {"max_new_tokens": 2048}` is hardcoded for **every single request** — a 3-word input gets the same generation budget as a 500-word essay. The model still stops at EOS, but the inference engine allocates KV-cache and manages attention over that full budget, wasting memory bandwidth proportionally.

**How the model's codec rate maps to tokens:**
- The model name says `12Hz` — 12 codec tokens per second of audio output
- Average speech: ~150 words/minute = 2.5 words/second = ~5 codec tokens/word
- For a 10-word input: ~50 tokens needed → 2048 budget is **40× too large**

**Formula:**
```python
# Words × ~8 tokens/word (with 60% safety buffer), floor at 128, cap at 2048
def _adaptive_max_tokens(text: str) -> int:
    word_count = len(text.split())
    return max(128, min(2048, word_count * 8))

gen_kwargs = {"max_new_tokens": _adaptive_max_tokens(text)}
```

**Expected gain**: For short texts (≤20 words, most real-world TTS use cases), latency drops 30–60% because the model's generation loop exits sooner and KV-cache is smaller. For long texts (>200 words) the effect diminishes but there's no regression.

**Tasks**:
- [ ] Add `_adaptive_max_tokens(text: str) -> int` function
- [ ] Replace both hardcoded `{"max_new_tokens": 2048}` in `synthesize_speech` and `clone_voice`
- [ ] Add `MAX_NEW_TOKENS` env var override for tuning (default: adaptive)
- [ ] Benchmark: measure 5-word, 20-word, 100-word inputs before/after

---

## A2. TF32 Matmul Mode — Free ~10% on Ampere+

**What**: On Ampere+ GPUs (RTX 3000/4000 series, A100, H100), PyTorch defaults to `allow_tf32 = False` for reproducibility. TF32 uses the full 8-bit exponent of FP32 but rounds the mantissa to 10 bits — nearly identical numerical results to FP32 but uses Tensor Core hardware (3× faster matmul).

Since the model is already running in bfloat16, enabling TF32 affects intermediate matmul operations and has negligible quality impact.

```python
# Add immediately after the cudnn.benchmark line (server.py line 23):
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True   # ADD THIS
    torch.backends.cudnn.allow_tf32 = True          # ADD THIS
```

**Gain**: 8–12% inference speedup at zero cost on supported GPUs. No-op on older hardware.

**Tasks**:
- [ ] Add two `allow_tf32` lines after existing `cudnn.benchmark` line
- [ ] Verify GPU is Ampere+ (`nvidia-smi --query-gpu=name --format=csv`)

---

## A3. GPU Persistence Mode — Eliminate Cold-Start Penalty

**What**: By default, NVIDIA GPUs power-down between workloads and re-initialize when first accessed. This adds 200–500ms to the first inference after the GPU has been idle. With `nvidia-smi -pm 1`, the GPU stays initialized.

**This is the main reason the first request after a long idle is noticeably slower than subsequent ones — even with the model still loaded.**

```bash
# Run once at host startup (persists until reboot or nvidia-smi -pm 0):
nvidia-smi -pm 1

# Or add to compose.yaml entrypoint:
entrypoint: sh -c "nvidia-smi -pm 1 && exec uvicorn server:app ..."
```

**For Docker**: Requires `privileged: true` or `cap_add: [SYS_ADMIN]` in compose.yaml if the container needs to set it itself. Better to set it on the host.

**Tasks**:
- [ ] Add `nvidia-smi -pm 1` to host startup script / systemd unit
- [ ] Or add to compose.yaml pre-start hook
- [ ] Measure: time first inference after 5 min idle vs baseline

---

## A4. Lock GPU Clocks to Max Boost

**What**: GPUs dynamically scale clocks based on temperature and power draw. Under sustained load the clock ramps up, but for short bursts (a single TTS inference lasting 500ms) the clock may not reach boost frequency before inference completes. Locking clocks eliminates this ramp-up latency and variance.

```bash
# Lock to max boost clocks (replace 1980 with your GPU's max boost MHz):
nvidia-smi -lgc 1980,1980

# Find your GPU's max clock:
nvidia-smi --query-gpu=clocks.max.graphics --format=csv,noheader

# Reset to auto:
nvidia-smi -rgc
```

**Effect**: Reduces p99 latency variance by 5–15%. Most impactful for spiky/burst workloads.

**Tradeoff**: Higher idle power consumption, potentially higher GPU temperature. Monitor thermals.

**Tasks**:
- [ ] Find max boost clock for target GPU
- [ ] Add `nvidia-smi -lgc <MAX>,<MAX>` to host startup script
- [ ] Monitor GPU temperature under sustained load (`nvidia-smi dmon -s t`)

---

## A5. Full Audio Output Cache

**What**: TTS output is deterministic — same `(text, voice, language, speed)` input always produces the same audio bytes. Cache the final encoded bytes in an LRU cache. A cache hit costs ~1ms (memory lookup) vs ~500–2000ms (full synthesis). This is the highest ROI for any workload with repeated phrases.

**Use cases**: IVR menus ("Press 1 for...", "Welcome to..."), notification templates, repeated phrases in demos.

```python
import hashlib
from functools import lru_cache

# In-process LRU cache (simple, no Redis needed)
_audio_cache: dict[str, bytes] = {}
_AUDIO_CACHE_MAX = int(os.getenv("AUDIO_CACHE_MAX", "256"))  # entries

def _audio_cache_key(text: str, voice: str, language: str, speed: float, fmt: str) -> str:
    raw = f"{text}|{voice}|{language}|{speed:.3f}|{fmt}"
    return hashlib.sha256(raw.encode()).hexdigest()

# In synthesize_speech(), before inference:
cache_key = _audio_cache_key(text, speaker, language, request.speed, request.response_format)
if cache_key in _audio_cache:
    return Response(content=_audio_cache[cache_key], media_type=content_type, ...)

# After encoding:
if len(_audio_cache) >= _AUDIO_CACHE_MAX:
    _audio_cache.pop(next(iter(_audio_cache)))  # evict oldest
_audio_cache[cache_key] = audio_bytes
```

**For distributed deployments**: Replace dict with Redis (`redis-py` + `asyncio` client). Key TTL of 24h. Serialized audio bytes as value.

**Tasks**:
- [ ] Implement in-memory LRU audio cache with configurable size
- [ ] Add cache hit/miss counters to `/health`
- [ ] Add `AUDIO_CACHE_MAX=0` to disable (default 256)
- [ ] Optional: add Redis backend with `REDIS_URL` env var

---

## A6. VAD Silence Trimming

**What**: TTS models often generate 100–300ms of silence at the start and end of audio. This silence wastes bandwidth, adds perceived latency, and can cause awkward pauses in real-time applications. Trim it with a simple amplitude threshold.

**Implementation** (pure numpy, no new deps):
```python
def trim_silence(audio: np.ndarray, threshold_db: float = -50.0, frame_ms: int = 10, sr: int = 24000) -> np.ndarray:
    """Remove leading and trailing silence below threshold."""
    threshold_linear = 10 ** (threshold_db / 20.0)
    frame_size = int(sr * frame_ms / 1000)

    # Find first and last frame above threshold
    frames = [audio[i:i+frame_size] for i in range(0, len(audio), frame_size)]
    rms = [np.sqrt(np.mean(f**2)) for f in frames]

    active = [i for i, r in enumerate(rms) if r > threshold_linear]
    if not active:
        return audio

    start = max(0, active[0] * frame_size - frame_size)   # 1 frame padding
    end = min(len(audio), (active[-1] + 2) * frame_size)  # 1 frame padding
    return audio[start:end]

# Apply after squeeze(), before speed adjustment:
audio_data = trim_silence(audio_data, threshold_db=-45.0, sr=sr)
```

**Gain**: Reduces audio length by 5–20% for short phrases. Eliminates the "gap" before speech starts in streaming scenarios. Improves perceived responsiveness.

**Tasks**:
- [ ] Implement `trim_silence()` using pure numpy
- [ ] Apply after `audio_data.squeeze()` in both endpoints
- [ ] Add `SILENCE_THRESHOLD_DB` env var (default -45)
- [ ] Test: verify speech content is not clipped on fast/loud starts

---

## A7. Opus Codec for Streaming

**What**: When streaming audio over a network, the choice of codec matters enormously for latency and bandwidth:

| Codec | Latency | Bitrate (speech quality) | Chunked streaming |
|-------|---------|--------------------------|-------------------|
| WAV | 0ms encode | ~768kbps | Works but huge |
| MP3 | ~50ms encode | 64–128kbps | Works |
| FLAC | ~20ms encode | ~300kbps | Works |
| **Opus** | **2.5ms encode** | **16–32kbps** | **Designed for it** |

Opus was specifically designed for real-time audio streaming (used by WebRTC, Discord, Spotify). At 32kbps it sounds better than MP3 at 128kbps for speech.

```python
# pip install pyogg opuslib (or use ffmpeg via pydub)
import subprocess

def encode_opus(audio_np: np.ndarray, sample_rate: int, bitrate: int = 32000) -> bytes:
    """Encode float32 numpy audio to Opus in OGG container via ffmpeg."""
    pcm = (audio_np * 32767).astype(np.int16).tobytes()
    result = subprocess.run([
        "ffmpeg", "-f", "s16le", "-ar", str(sample_rate), "-ac", "1",
        "-i", "pipe:0", "-c:a", "libopus", "-b:a", str(bitrate),
        "-f", "ogg", "pipe:1"
    ], input=pcm, capture_output=True)
    return result.stdout
```

**Better approach**: Use `opuslib` or `soundfile`'s OGG+Opus support for zero subprocess overhead.

**Tasks**:
- [ ] Verify ffmpeg build has `libopus` (`ffmpeg -codecs | grep opus`)
- [ ] Add `opus` as valid `response_format` option
- [ ] Implement Opus encoding in `convert_audio_format()`
- [ ] Benchmark: compare Opus 32kbps vs WAV encode time + bytes transferred

---

## A8. GPU-Accelerated Audio Processing

**What**: After synthesis, audio processing (resampling for speed, potentially format normalization) runs on CPU with numpy/scipy. These operations can run on GPU with `torchaudio` — keeping data on GPU avoids a round-trip through CPU RAM.

```python
import torchaudio

def speed_adjust_gpu(audio_np: np.ndarray, sr: int, speed: float) -> np.ndarray:
    """Pitch-preserving speed adjustment on GPU using torchaudio."""
    audio_t = torch.from_numpy(audio_np).unsqueeze(0).cuda()  # [1, T]
    # torchaudio.functional.resample preserves pitch via sinc interpolation
    new_sr = int(sr * speed)
    audio_t = torchaudio.functional.resample(audio_t, sr, new_sr)
    return audio_t.squeeze().cpu().numpy()
```

**Also**: `torchaudio.functional.vad()` for silence trimming on GPU (replaces the numpy VAD in A6).

**Tasks**:
- [ ] Add `torchaudio` to Dockerfile (often already included with PyTorch image — check first)
- [ ] Replace scipy speed adjustment with torchaudio GPU version when CUDA available
- [ ] Benchmark: CPU scipy vs GPU torchaudio for 5s audio resampling

---

## A9. Async Audio Encode Pipeline (Overlap with Synthesis)

**What**: In the sentence-chunked streaming flow (item #1), there is dead time between when sentence N's audio comes back from GPU and when it's fully encoded into WAV/MP3 bytes. The next synthesis (sentence N+1) can't start until encoding finishes (they share the semaphore). By running audio encoding in a separate CPU thread concurrently with the next GPU synthesis, we eliminate this dead time.

**Flow** (current):
```
[synthesize N] → [encode N] → [send N] → [synthesize N+1] → [encode N+1] → ...
                  ^-- blocks next synthesis
```

**Flow** (optimized):
```
[synthesize N] → [encode N in CPU thread] → [send N]
                  [synthesize N+1]    ← runs in parallel with encode N
                                      → [encode N+1 in CPU thread] → ...
```

**Implementation**: Use a second `ThreadPoolExecutor` with `max_workers=2` for audio encoding, run alongside `_infer_semaphore` release.

**Tasks**:
- [ ] Add `_encode_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tts-encode")`
- [ ] Refactor streaming endpoint to release semaphore before encoding
- [ ] Use `asyncio.gather()` to overlap encode N with synthesize N+1
- [ ] Test: verify no race conditions when sentences encode out of order

---

## A10. jemalloc Memory Allocator

**What**: Python's default `malloc` (ptmalloc2 on Linux) has poor behavior for long-running processes with many small allocations — it fragments memory and holds fragmented pages rather than returning them to the OS. `jemalloc` uses size-class segregated free lists and background thread decay, dramatically reducing fragmentation and allocation latency.

**This matters here**: PyTorch + numpy do many small allocations per request. Over hours of uptime, ptmalloc2 causes RSS memory to grow 2–3× beyond actual usage.

```dockerfile
# In Dockerfile, add:
RUN apt-get install -y libjemalloc2

# In Dockerfile CMD or entrypoint:
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
ENV MALLOC_CONF="background_thread:true,dirty_decay_ms:1000,muzzy_decay_ms:0"
```

**Gain**: Lower RSS memory, reduced allocation latency (~20% faster for allocation-heavy code), eliminates memory growth over time.

**Tasks**:
- [ ] Add `libjemalloc2` to Dockerfile apt install
- [ ] Set `LD_PRELOAD` and `MALLOC_CONF` env vars
- [ ] Monitor RSS over 24h before/after (`docker stats`)

---

## A11. CPU Affinity — Pin Inference Thread to GPU-Adjacent Cores

**What**: On NUMA systems (multi-socket, or some Ryzen/EPYC configs), CPU cores have different latency to GPU PCIe. Pinning the inference thread to cores on the same NUMA node as the GPU reduces cache coherency traffic and memory latency for CPU↔GPU data transfers.

```bash
# Find which NUMA node your GPU is on:
cat /sys/bus/pci/devices/0000:$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | head -1 | cut -d: -f2-)/numa_node

# Pin container to that node's CPUs:
# In compose.yaml:
cpuset: "0-7"  # cores on NUMA node 0 (adjust to your system)
```

**Or at Python level** using `os.sched_setaffinity()` on the inference thread.

**Tasks**:
- [ ] Identify GPU NUMA node
- [ ] Set `cpuset` in compose.yaml to GPU-adjacent cores
- [ ] Benchmark: p50/p99 inference latency before/after

---

## A12. Transparent Huge Pages for Model Weights

**What**: The 0.6B model weights occupy ~2.4GB in GPU memory, but their corresponding CPU-side tensors (during load) use thousands of 4KB pages. Each page is a TLB entry. Transparent Huge Pages (THP) coalesces these into 2MB pages, reducing TLB pressure during model loading and weight initialization.

```bash
# Enable THP (on host):
echo always > /sys/kernel/mm/transparent_hugepage/enabled
echo defer+madvise > /sys/kernel/mm/transparent_hugepage/defrag

# Or per-process via madvise (PyTorch does this automatically with MADV_HUGEPAGE in newer versions)
```

**Gain**: Faster model load time (5–15%), lower CPU overhead during warmup. Negligible during inference (weights are on GPU).

**Tasks**:
- [ ] Check current THP setting: `cat /sys/kernel/mm/transparent_hugepage/enabled`
- [ ] Enable on host startup (add to `/etc/rc.local` or systemd unit)
- [ ] Measure: model load time before/after

---

## A13. WebSockets — True Bidirectional Streaming

**What**: SSE (item #1) is server-push only over HTTP/1.1. WebSockets provide full-duplex: the client can send new text mid-stream, cancel generation, or send parameters. For a real-time voice assistant, WebSockets are the correct protocol — the client sends text and receives a binary audio stream back on the same persistent connection.

**Latency advantage**: WebSocket connection is established once (handshake ~1 round-trip). Subsequent messages have no HTTP overhead. SSE requires a new connection per request in some clients.

**Implementation with FastAPI**:
```python
from fastapi import WebSocket

@app.websocket("/v1/audio/speech/ws")
async def synthesize_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            text = data.get("input", "")
            voice = resolve_voice(data.get("voice"))
            language = data.get("language") or detect_language(text)

            for sentence in split_sentences(text):
                audio_np = await synthesize_sentence(sentence, voice, language)
                audio_bytes = to_pcm(audio_np)  # raw PCM, no headers
                await websocket.send_bytes(audio_bytes)

            await websocket.send_json({"event": "done"})
    except WebSocketDisconnect:
        pass
```

**Tasks**:
- [ ] Add `/v1/audio/speech/ws` WebSocket endpoint
- [ ] Send binary PCM frames per sentence chunk
- [ ] Send JSON control frames (`{"event": "done"}`, `{"event": "error", "detail": "..."}`)
- [ ] Document protocol: connection, send JSON, receive binary frames + done event

---

## A14. HTTP/2 — Multiplexed Concurrent Requests

**What**: HTTP/1.1 (current) opens a new TCP connection per request (or reuses with keep-alive, but serializes). HTTP/2 multiplexes multiple concurrent requests over a single TCP connection, eliminating connection setup overhead for burst traffic.

**For uvicorn**: Switch from `httptools` (HTTP/1.1) to `h2` (HTTP/2):

```yaml
# In compose.yaml command, replace:
--http httptools
# with:
--http h2
```

**Note**: HTTP/2 requires TLS in browsers but works without TLS for programmatic clients (h2c = cleartext HTTP/2). OpenAI SDK supports h2c.

**Tasks**:
- [ ] Add `h2` package to Dockerfile: `pip install h2`
- [ ] Change `--http httptools` to `--http h2` in compose.yaml
- [ ] Test with `curl --http2 ...` and verify connection reuse
- [ ] Verify OpenAI Python SDK works with h2c

---

## A15. Unix Domain Sockets — Zero TCP Overhead (Same-Host Clients)

**What**: If the TTS client (e.g., your voice assistant, LLM server) runs on the same host as the TTS server, TCP loopback adds unnecessary overhead: TCP handshake, kernel TCP stack, socket buffer copies. Unix Domain Sockets (UDS) bypass all of this — data goes directly kernel↔kernel.

**Gain**: ~0.1–0.5ms per request. More importantly, eliminates jitter from TCP retransmits on loopback (rare but non-zero).

```yaml
# compose.yaml: bind to UDS instead of/in addition to TCP
command: uvicorn server:app --uds /tmp/qwen3-tts.sock --loop uvloop

volumes:
  - /tmp/qwen3-tts.sock:/tmp/qwen3-tts.sock  # share with client container
```

```python
# Client-side (httpx):
import httpx
transport = httpx.AsyncHTTPTransport(uds="/tmp/qwen3-tts.sock")
async with httpx.AsyncClient(transport=transport, base_url="http://tts") as client:
    response = await client.post("/v1/audio/speech", json={...})
```

**Tasks**:
- [ ] Add `--uds /tmp/qwen3-tts.sock` to uvicorn command
- [ ] Keep TCP port as fallback for external clients
- [ ] Share socket via Docker volume if client is in another container
- [ ] Document UDS client usage

---

## A16. Always-On Mode (Disable Idle Unload for Dedicated GPU)

**What**: The idle watchdog (unload after 120s) makes sense for a shared GPU serving multiple models. For a dedicated TTS server with guaranteed GPU allocation, it adds unnecessary reload latency (~2–5s) on the next request after idle. Disabling it keeps the model always resident in VRAM.

```yaml
# compose.yaml:
environment:
  IDLE_TIMEOUT: "0"  # 0 = disabled
```

**Tradeoff**: VRAM is held permanently (~2.4GB for 0.6B). Only valid if nothing else needs that VRAM.

**Tasks**:
- [ ] Document `IDLE_TIMEOUT=0` in README as "dedicated GPU" mode
- [ ] Add to compose.yaml as commented option

---

## A17. Eager Model Load at Startup

**What**: Currently the model loads on the first request. The first request after a cold start pays a ~10–15s load penalty. For production servers where latency of the first request matters, load at startup.

```python
# Replace @app.on_event("startup") with:
@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(_idle_watchdog())
    if os.getenv("PRELOAD_MODEL", "0") == "1":
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(_infer_executor, _load_model_sync)
    yield
```

```yaml
# compose.yaml:
environment:
  PRELOAD_MODEL: "1"
```

**Tasks**:
- [ ] Add `PRELOAD_MODEL` env var
- [ ] Load in lifespan startup if set
- [ ] Increase `start_period` in Docker healthcheck to 60s (model load time)

---

## A18 & A19. Docker Host Networking & IPC

**What**: Two Docker flags that reduce container networking overhead:

```yaml
# compose.yaml additions:
services:
  qwen3-tts:
    ipc: host           # Share host IPC namespace — faster CUDA IPC between processes
    network_mode: host  # Bypass Docker bridge NAT — use host network stack directly
    # REMOVE ports: mapping when using network_mode: host
```

- `--ipc=host`: CUDA uses shared memory IPC for some operations. Container IPC namespace isolation adds overhead. `ipc: host` eliminates this.
- `--network=host`: Removes Docker's userspace NAT for the bridge network. Reduces per-packet overhead by ~0.1–0.2ms. Only safe on trusted single-host deployments.

**Tradeoff**: `network_mode: host` removes port isolation. Suitable for private/internal deployments, not public-facing servers.

**Tasks**:
- [ ] Add `ipc: host` to compose.yaml (always safe)
- [ ] Add `network_mode: host` as optional commented config for private deployments
- [ ] Remove `ports:` mapping when `network_mode: host` is active

---

## A20. Per-Request Latency Breakdown

**What**: Currently there's no way to know where time is being spent: model load? queue wait? inference? audio encoding? network transfer? Without measurements, optimization is guesswork.

Add timing instrumentation at each stage:

```python
import time

@app.post("/v1/audio/speech")
async def synthesize_speech(request: TTSRequest):
    t0 = time.perf_counter()
    await _ensure_model_loaded()
    t_loaded = time.perf_counter()

    async with _infer_semaphore:
        t_queued = time.perf_counter()
        wavs, sr = await asyncio.wait_for(...)
    t_inferred = time.perf_counter()

    audio_bytes, content_type = convert_audio_format(...)
    t_encoded = time.perf_counter()

    logger.info("synthesis_complete",
        load_ms=round((t_loaded - t0) * 1000),
        queue_ms=round((t_queued - t_loaded) * 1000),
        infer_ms=round((t_inferred - t_queued) * 1000),
        encode_ms=round((t_encoded - t_inferred) * 1000),
        total_ms=round((t_encoded - t0) * 1000),
        chars=len(request.input),
        voice=speaker,
    )
```

**Add to response headers** so clients can see breakdown:
```
X-Latency-Infer-Ms: 423
X-Latency-Encode-Ms: 12
X-Latency-Total-Ms: 441
```

**Tasks**:
- [ ] Add `time.perf_counter()` checkpoints at each stage
- [ ] Log breakdown as structured JSON per request
- [ ] Add latency headers to response
- [ ] Build histogram in Prometheus (infer_ms, encode_ms per voice/length bucket)

---

## Updated Implementation Order (Combined Roadmap)

```
Week 1 — Immediate wins (all ≤1 hour each):
  A1: Adaptive max_new_tokens       ← single line change, massive impact
  A2: TF32 mode                     ← two lines
  A3: GPU persistence mode          ← one nvidia-smi command
  A6: VAD silence trimming          ← ~20 lines, pure numpy
  A20: Per-request latency logging  ← measure everything before optimizing further

Week 1 — Streaming:
  #1: Sentence-chunked SSE streaming
  #2: Raw PCM endpoint

Week 2 — Inference flags:
  #3: flash_attention_2
  #4: torch.compile

Week 2 — Caching & Audio:
  A5: Audio output cache (LRU)
  A10: jemalloc
  A7: Opus codec
  A4: GPU clock locking

Week 3 — Quality Polish:
  #9: Text normalization
  #7: Voice prompt cache
  #11: fasttext language detection
  #10: pyrubberband speed

Week 3 — Protocol:
  A13: WebSockets endpoint
  A14: HTTP/2
  A15: Unix domain sockets

Week 4 — System Tuning:
  A8: GPU-accelerated audio (torchaudio)
  A9: Async encode pipeline
  A11: CPU affinity
  A12: Huge pages
  A18: ipc: host in Docker

Week 5 — Lifecycle & Observability:
  A16: Always-on mode documentation
  A17: Eager preload option
  #16: Prometheus metrics
  #17: Structured JSON logging
  #14: Pin dependency versions
```
