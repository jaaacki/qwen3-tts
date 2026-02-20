# Qwen3-TTS Improvements Roadmap

**Goal**: High-quality, real-time TTS — sub-500ms first-audio latency, maximum output quality, production-grade observability.

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
- **1.7B model**: Verify the exact HuggingFace model ID before deploying — Qwen publishes multiple 1.7B variants (base, instruct, CustomVoice, CustomDesign).
- **Streaming format compatibility**: OpenAI's streaming TTS uses chunked MP3. For maximum OpenAI SDK compatibility, stream as chunked WAV or raw PCM and document the difference.
- **IDLE_TIMEOUT with streaming**: With sentence-chunked streaming, long texts keep the model alive across multiple synthesis calls within one request. Ensure `_last_used` is updated per-chunk, not just per-request.
