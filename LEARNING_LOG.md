# Learning Log

Decisions, patterns, and lessons from building the Qwen3-TTS server. Each entry is written so someone new to the project can understand the reasoning without digging through commit history.

---

## Entry 0012 — GPU memory pool pre-warming and CUDA allocator tuning
**Date**: 2026-02-20
**Type**: Why this design
**Related**: Issue #16 — Pre-allocate GPU memory pool to reduce allocation jitter

CUDA memory allocation is lazy by default. The first time a tensor of a given size is allocated, the CUDA allocator calls `cudaMalloc`, which involves a kernel-mode transition and device synchronization. This takes 1-5ms per allocation. For a TTS server handling its first request after model load, there are multiple novel allocation sizes (KV-cache, attention intermediates, audio output buffers), each paying this penalty. The cumulative cost can add 10-30ms to the first inference.

The fix is a dummy allocation: after warmup, allocate a 128 MB tensor (`torch.empty(64*1024*1024, dtype=bfloat16, device="cuda")`) and immediately delete it. This forces the CUDA allocator to reserve a contiguous 128 MB block in its free pool. Subsequent allocations that fit within this block are served from the pool without `cudaMalloc` calls.

The `max_split_size_mb:512` addition to `PYTORCH_CUDA_ALLOC_CONF` prevents the allocator from splitting large cached blocks into small fragments. Without this, the allocator might split a 128 MB cached block into many small pieces to serve a 1 MB request, then not be able to recombine them when a 64 MB request arrives.

---

## Entry 0008 — Why pitch-preserving time stretch matters for TTS
**Date**: 2026-02-20
**Type**: Why this design
**Related**: Issue #14 — Replace scipy speed adjustment with pitch-preserving pyrubberband

The original speed adjustment used `scipy.signal.resample()` to change the number of audio samples. This is a frequency-domain resampling that compresses or expands the waveform uniformly. The problem: when you compress samples to make audio play faster, the fundamental frequency of the voice shifts upward proportionally. At speed=1.5, the voice pitch rises by 50%, creating the classic "chipmunk" effect. At speed=0.75, the pitch drops, making the voice sound artificially deep.

For a TTS server, this is unacceptable. Speed adjustment is used for accessibility (slower speech for comprehension), time fitting (faster speech for constrained UIs), and prosody matching (adjusting pace to context). In all cases, the user expects the voice to sound like the same person speaking at a different pace, not a pitch-shifted version.

pyrubberband wraps the Rubber Band Library, which implements PSOLA (Pitch-Synchronous Overlap-Add) time stretching. PSOLA works by identifying pitch periods in the audio, duplicating or removing complete pitch cycles, and crossfading at zero-crossing points. The result is audio that plays faster or slower without any pitch change. The algorithm is well-established in audio processing and adds negligible latency (< 10ms for typical TTS output lengths).

The implementation uses the same graceful-fallback pattern as other optional dependencies: if `pyrubberband` is not importable, `_pyrubberband` is set to `None` and the function falls back to `scipy.signal.resample`. This keeps the server functional on systems without the rubberband-cli binary installed, at the cost of the pitch shift artifact.

The Dockerfile installs both `pyrubberband` (Python bindings) and `rubberband-cli` (the C++ binary that pyrubberband calls). The binary is available in Ubuntu's package manager, so no compilation is needed.

---

## Entry 0001 — Project baseline: current architecture
**Date**: 2026-02-20
**Type**: Why this design
**Related**: Planning — pre-issue

The server is a single-file FastAPI application that wraps the Qwen3-TTS-0.6B model behind an OpenAI-compatible API. There are four design choices that define the baseline, and all four exist for the same reason: the event loop must never block.

First, model loading is lazy. The model does not load at startup. It loads on the first request and stays resident until an idle timeout fires. This matters because the container shares a GPU with other services on a Synology NAS. Holding 2.4 GB of VRAM permanently when the service might go hours without a request is wasteful. The `_idle_watchdog` background task checks every 30 seconds whether `_last_used` has exceeded `IDLE_TIMEOUT` (default 120 seconds), and if so, unloads the model and calls `torch.cuda.empty_cache()` plus `ipc_collect()` to return VRAM to the system.

Second, all GPU work runs through a dedicated single-thread `ThreadPoolExecutor` via `run_in_executor`. This is the single most important architectural decision. The Qwen3-TTS model's `generate_custom_voice` is a blocking synchronous call that holds the GIL and runs CUDA kernels for 400-2000ms. If this ran directly in an async handler, the entire event loop would freeze — health checks would hang, the idle watchdog would stall, and concurrent HTTP connections would time out. Offloading to a thread executor lets the event loop continue servicing other coroutines while GPU inference runs in a background thread.

Third, an `asyncio.Semaphore(1)` serializes GPU inference. Even though the executor has only one thread, the semaphore is still necessary — it prevents a second request from queuing inside the executor while a first is running. Without it, two requests could both enter `run_in_executor`, and the second would block its event loop coroutine waiting for the executor thread, which is functionally identical to blocking the loop. The semaphore makes the queueing explicit and visible to the async scheduler.

Fourth, an `asyncio.Lock` with double-checked locking protects model load and unload. Two requests arriving simultaneously on a cold server would both see `model is None` and both try to load. The lock ensures only one load happens. The double-check pattern (check before acquiring, check again after acquiring) avoids holding the lock on the hot path when the model is already loaded.

The critical baseline insight: every community alternative we analyzed (twolven, ValyrianTech) gets the event loop blocking wrong. They call synchronous model inference directly inside async handlers. Our server is the only one that correctly keeps the event loop responsive during inference. This is not an optimization — it is a correctness requirement.

---

## Entry 0002 — Why the bottleneck is not what you'd expect
**Date**: 2026-02-20
**Type**: Why this design
**Related**: Planning — pre-issue

The intuition when looking at a TTS server is that the bottleneck is model inference speed. The model takes 400-600ms to generate a sentence of audio — that feels like the thing to optimize. But the actual user experience problem is different: the server returns zero bytes until the entire synthesis is complete.

Consider a three-sentence paragraph. Total inference time is roughly 1.5 seconds. But the client does not receive the first byte of audio until all three sentences are done. The user hears 1.5 seconds of silence, then the full audio plays. The perceived latency is the total time, not the per-sentence time.

Now consider what happens with streaming. If the server splits the text into sentences and sends each sentence's audio as soon as it is ready, the first audio arrives after roughly 500ms (one sentence of inference). The user starts hearing speech while the server is still synthesizing the remaining sentences. The total wall-clock time is the same, but the perceived latency drops by 60-70%.

This is why the improvement plan puts streaming (Phase 1) before any inference optimization (Phase 2). A 20% inference speedup on a non-streaming server saves maybe 300ms on a 1.5s total. Streaming saves 1000ms of perceived silence on the same input. The streaming architecture creates a low-latency shell — once it exists, inference speedups compound on top of it because each sentence chunk gets individually faster. But doing inference optimization without streaming means the speed gains are invisible to the user. They still wait for everything to finish before hearing anything.

The ordering is not about difficulty or risk. It is about which layer of the stack the user actually perceives.

---

## Entry 0003 — The max_new_tokens blind spot
**Date**: 2026-02-20
**Type**: What just happened (planning discovery)
**Related**: Planning — pre-issue

During the second pass over the improvement plan, we found something that the first review entirely missed: `max_new_tokens` is hardcoded to 2048 for every request. This line appears in both `synthesize_speech` and `clone_voice`:

```python
gen_kwargs = {"max_new_tokens": 2048}
```

The model's name includes "12Hz" — it generates 12 codec tokens per second of audio output. Average speech runs at about 150 words per minute, which works out to roughly 5 codec tokens per word. A 10-word sentence like "Please hold while I transfer your call" needs approximately 50 tokens. The server is allocating a budget of 2048 tokens — about 40 times what is actually needed.

The model still stops at the EOS token, so the output audio is correct. But the inference engine pre-allocates a KV-cache sized for 2048 tokens and manages attention over that full budget. For short texts, this wastes memory bandwidth and adds overhead to every attention computation. The fix is a simple function that scales the token budget with input length:

```python
def _adaptive_max_tokens(text: str) -> int:
    word_count = len(text.split())
    return max(128, min(2048, word_count * 8))
```

For short inputs (the vast majority of real-world TTS — greetings, IVR prompts, single sentences), the expected latency improvement is 30-60%. For long inputs the budget stays at 2048 and nothing changes.

The aha moment: when reviewing a model server, always read the `gen_kwargs`. The architecture diagram, the async patterns, the concurrency controls — those are what draw your attention during a code review. The generation parameters are one line buried in a handler, and they are easy to gloss over. But they directly control how hard the GPU works per request.

---

## Entry 0004 — Why three phases and in this order
**Date**: 2026-02-20
**Type**: Why this design
**Related**: Planning — pre-issue

The improvement plan is organized into three phases, and the ordering is deliberate.

Phase 1 targets perceived latency. It starts with latency instrumentation (measure first — you cannot improve what you cannot see), then fixes the hardcoded `max_new_tokens` budget that wastes GPU cycles on short inputs, and finally delivers streaming — sentence-chunked SSE and raw PCM endpoints. The goal is to change what the user experiences. Even if the model runs at exactly the same speed, the user hears audio sooner because chunks arrive incrementally. The metric that matters here is time-to-first-audio, not total synthesis time. For the primary use case (phone calls), this is the difference between a natural conversation flow and an awkward 2-3 second pause.

Phase 2 targets actual latency. This is where inference gets faster: `flash_attention_2`, `torch.compile`, TF32 matmul, GPU clock locking. Each of these makes the model produce audio faster in wall-clock time. But notice — these gains only feel impactful because streaming is already in place. A 20% speedup on a 500ms sentence chunk means the user hears audio 100ms sooner. Without streaming, the same 20% speedup on a 1.5s total synthesis saves 300ms of wait time that the user still experiences as a single block of silence. The streaming shell amplifies the perceived impact of every inference optimization that follows.

Phase 3 targets operational correctness. Prometheus metrics, structured JSON logging, dependency pinning, lifecycle improvements. Phase 3 comes last not because it is least important, but because you need a working, optimized system to instrument meaningfully. Basic latency logging exists from Phase 1, but the production-grade observability stack (Prometheus, structured logs with per-request fields, request queue depth limits) needs a stable system to measure against. Instrumenting after streaming gives you time-to-first-chunk, per-sentence inference time, encode overhead, and queue wait — the metrics you actually need to find the next bottleneck.

The phases form a dependency chain: streaming creates the architecture that makes speed gains visible, and speed gains create the system worth measuring.

---

## Entry 0005 — What could go wrong with the streaming approach
**Date**: 2026-02-20
**Type**: What could go wrong
**Related**: Planning — pre-issue

Sentence-chunked streaming is the highest-leverage change in the plan, but it has at least five failure modes that are not obvious from the implementation sketch.

**Sentence splitting on abbreviations.** A naive regex split on `.!?` will break on "Dr. Smith called at 3 p.m. to confirm." — producing four fragments instead of one sentence. "U.S.A." becomes three splits. The sentence splitter needs an abbreviation-aware tokenizer, not a regex. This is solvable (libraries like `pysbd` handle it), but if you ship the naive version first and discover the bug in production, the audio will have unnatural micro-pauses between "Dr" and "Smith" that sound worse than no streaming at all.

**WAV format and the data_size header.** A WAV file begins with a RIFF header that includes the total data size. You cannot write a valid WAV header until you know how many bytes of audio follow. This means chunked WAV streaming requires either: (a) writing a placeholder header with size 0xFFFFFFFF and hoping the client tolerates it, (b) using raw PCM with no header and documenting the sample format separately, or (c) using a container format designed for streaming like OGG/Opus. The plan includes a raw PCM endpoint for this reason, but any client expecting a standard WAV file will reject a chunked stream.

**Idle timeout and streaming sessions.** The `_last_used` timestamp currently updates once per request. With streaming, a long text might take 10+ seconds to stream all sentence chunks. If `_last_used` is set at the start and the idle timeout is 120 seconds, this is fine. But if `_last_used` is only set once and the next request comes 115 seconds after the stream started, the watchdog sees 115 seconds of idle time and unloads the model. The fix is simple — update `_last_used` after each chunk — but forgetting this will cause the model to unload mid-stream, which is a request failure that only appears under specific timing conditions.

**Reverse proxy buffering.** Nginx, Cloudflare, and most load balancers buffer response bodies by default. A chunked HTTP response that the server sends in 500ms increments will arrive at the client as a single burst after the full response is buffered. The `X-Accel-Buffering: no` header disables this for Nginx, but other proxies need their own configuration. If the deployment sits behind any proxy layer, streaming will appear to not work even though the server is sending chunks correctly. This is an infrastructure problem, not a code problem, and it is invisible during local testing.

**Semaphore serialization between chunks.** The current `Semaphore(1)` serializes all GPU inference. In the streaming flow, each sentence is a separate inference call. Sentence N must complete and release the semaphore before sentence N+1 can acquire it. For a single streaming request this is correct — sentences must be sequential anyway. But if two clients are streaming simultaneously, their sentences interleave: client A sentence 1, client B sentence 1, client A sentence 2, and so on. This doubles the time-to-completion for both clients. The async pipeline optimization (encoding chunk N on CPU while synthesizing chunk N+1 on GPU) can partially hide this, but the fundamental issue is that `Semaphore(1)` means the GPU can only work on one sentence at a time regardless of how many clients are waiting.

---

## Entry 0006 — GPU system tuning: the invisible ms
**Date**: 2026-02-20
**Type**: Why this design
**Related**: Planning — pre-issue

There is a category of optimization that does not appear in any Python code review because it lives in the GPU driver layer. Three settings in particular have outsized impact on latency consistency, and all three are single-command changes.

**GPU persistence mode** (`nvidia-smi -pm 1`). By default, the NVIDIA driver powers down the GPU between workloads. When the next CUDA call arrives, the driver reinitializes the GPU context — this takes 200-500ms. With persistence mode enabled, the GPU stays initialized even when idle. The effect: the first inference after a long idle period is 200-500ms faster. Without it, there is a "double cold start" — the model loads into VRAM (5-10 seconds), then the first inference stalls while the GPU context reinitializes. Users report this as "the first request is always slow" and often attribute it to model loading, but the GPU context initialization is a separate penalty on top of model load time.

**GPU clock locking** (`nvidia-smi -lgc <max>,<max>`). GPUs dynamically scale their clock frequency based on temperature and power draw. For sustained workloads (training, video rendering) this is fine — the clock ramps up within milliseconds and stays there. But TTS inference is bursty: a single request runs for 400-600ms and then the GPU goes idle. The clock may not reach boost frequency before inference completes, meaning every request runs at a slower-than-maximum clock speed. Locking the clocks to the maximum boost frequency eliminates this ramp-up latency and, more importantly, eliminates variance. Without clock locking, two identical requests can have different latencies depending on whether the GPU was already boosted from a recent request. The tradeoff is higher idle power consumption and slightly higher GPU temperature.

**TF32 matmul** (`torch.backends.cuda.matmul.allow_tf32 = True`). On Ampere and newer GPUs (RTX 3000/4000 series, A100, H100), PyTorch defaults to full FP32 for matrix multiplication. TF32 uses the same 8-bit exponent but rounds the mantissa from 23 bits to 10 bits, allowing the operation to use Tensor Core hardware that runs 3x faster. Since the model already runs in bfloat16 (which has the same 8-bit exponent and only 7 bits of mantissa), enabling TF32 for intermediate operations has negligible quality impact. This is two lines of Python, but the effect is an 8-12% inference speedup on supported hardware. It is a no-op on older GPUs.

None of these changes appear in a typical code review. They are infrastructure-level settings. But together, they can reduce p99 latency by 20-30% and virtually eliminate tail latency variance. For a real-time application like phone calls, consistency matters as much as raw speed — a system that is usually fast but occasionally slow feels worse than one that is always moderately fast.

---

## Entry 0008 — Why fasttext over Unicode heuristic for language detection
**Date**: 2026-02-20
**Type**: Why this design
**Related**: Issue #13 — Replace Unicode language heuristic with fasttext detection

The original `detect_language()` function used a character-range heuristic: scan the input text for CJK, Hiragana/Katakana, or Hangul characters, and default to English for anything else. This works for scripts with unique Unicode ranges but fails completely for languages that share the Latin alphabet. French, German, Spanish, Italian, Portuguese, and Russian are all detected as "English" because their characters fall within ASCII or basic Latin ranges.

The Qwen3-TTS model supports all of these languages. A French user sending "Bonjour le monde" gets English prosody applied because the server cannot tell the difference. This is not a theoretical problem — it affects every European language user.

The fix uses `fasttext-langdetect`, which wraps Facebook's fasttext language identification model. It returns ISO 639-1 codes (e.g., "fr", "de", "es") with confidence scores. A mapping dict (`_LANG_MAP`) converts these to the Qwen-expected language names. The implementation is a graceful upgrade: if `fasttext-langdetect` is not installed, the function falls back to the original Unicode heuristic. This means the server works identically without the dependency — it just cannot detect Latin-script languages beyond English.

Key design decisions:
- **Lazy loading**: The fasttext model loads on first call to `detect_language()`, not at import time. This avoids slowing down startup for a model that might not be needed if every request provides an explicit `language` parameter.
- **False sentinel**: `_langdetect_model` uses `False` (not `None`) to distinguish "tried to import and failed" from "haven't tried yet". This prevents retrying the import on every request when the package is genuinely missing.
- **low_memory=False**: The fasttext model is small (~1MB). Loading it fully into memory is faster than the compressed low-memory mode, and the memory cost is negligible compared to the TTS model.
- **Default to English for unknown ISO codes**: If fasttext returns a language code not in `_LANG_MAP` (e.g., "tl" for Tagalog), the function returns "English" rather than passing through the raw code. This is because Qwen3-TTS has a fixed set of supported languages, and an unsupported language name would cause an inference error.

---

## Entry 0007 — The caching hierarchy
**Date**: 2026-02-20
**Type**: Why this design
**Related**: Planning — pre-issue

The improvement plan includes three layers of caching, and the ordering from highest to lowest leverage is the opposite of what you might expect.

**Layer 1: Audio output cache** (text + voice + language + speed + format -> final audio bytes). This is an in-memory LRU dict keyed by a SHA-256 hash of the request parameters. A cache hit costs about 1ms (memory lookup plus hash computation). A cache miss costs 500-2000ms (full model inference plus audio encoding). The hit-to-miss ratio depends entirely on the workload. For an IVR system where a menu says "Press 1 for billing, press 2 for support" on every call, the hit rate is effectively 100% after the first call. For a voice assistant generating unique responses, the hit rate is near 0%. The plan defaults to 256 entries and allows disabling via `AUDIO_CACHE_MAX=0`.

**Layer 2: Voice prompt cache** (reference audio bytes -> processed voice embedding). This applies only to the `/clone` endpoint. Voice cloning requires processing the reference audio file on every request: reading the audio, converting to mono if stereo, and passing it to the model's voice cloning pipeline. If the same reference audio is used repeatedly (which is the common case — you pick a voice and use it for all requests), this processing is redundant. Caching the processed audio keyed by a content hash saves roughly 1 second per clone request. This is lower leverage than the output cache because it only saves the preprocessing step, not the inference itself.

**Layer 3: KV prefix cache** (future). If the model's KV-cache for common text prefixes (e.g., "Thank you for calling") could be pre-computed and reused, inference for texts sharing that prefix would skip the prefill phase. This is the most technically complex cache and depends on the model's internals exposing KV-cache manipulation. It is listed as future because it requires deeper integration with the `qwen-tts` library than the other two layers.

The ordering matters for implementation priority. The output cache collapses the entire pipeline for repeated requests — inference, audio encoding, everything. One dict lookup replaces all of it. The voice prompt cache only saves preprocessing. The KV cache only saves part of inference. In terms of implementation effort, the output cache is roughly 20 lines of code. The voice prompt cache is similar. The KV cache is an open research question.

For the phone call use case, the realistic expectation is that the output cache provides the majority of the benefit. IVR menus, hold messages, greeting phrases, and common system responses repeat constantly. A deployment serving 1000 calls per day with 20 unique system phrases would see cache hit rates above 90% after the first few calls. The per-request cost drops from 500ms of GPU inference to 1ms of memory lookup.

---

## Entry 0008 — Audio cache: key design and LRU eviction
**Date**: 2026-02-20
**Type**: What just happened
**Related**: #17

The audio output cache is an OrderedDict keyed by SHA-256 of `text|voice|speed|format|instruct`. The key includes every parameter that affects the output — if any parameter changes, the cache key changes, and a new synthesis runs. The pipe delimiter prevents ambiguity between parameters (e.g., "hello|vivian" vs "hello|" + "vivian").

The cache stores the final encoded bytes and content type, not the raw audio array. This means a cache hit returns the exact HTTP response body — no format conversion, no speed adjustment, no GPU work at all. The cost of a cache hit is one SHA-256 hash (~1 microsecond) plus one OrderedDict lookup (~1 microsecond).

The cache check happens before `_ensure_model_loaded()`. This is deliberate: if every request for the next hour hits the cache, the model never loads, and VRAM stays free. The idle watchdog continues running, but since the model was never loaded, it has nothing to unload. This makes the cache especially valuable in shared GPU environments where VRAM is contended.

LRU eviction uses `OrderedDict.move_to_end()` on hit and `popitem(last=False)` when full. This is O(1) for both operations. The default capacity of 256 entries is sized for a typical IVR deployment where 20-50 unique system phrases repeat across thousands of calls. At roughly 100KB per WAV entry (1 second of 24kHz 16-bit audio), 256 entries consume about 25MB of RAM — negligible compared to the 2.4GB model.

Setting `AUDIO_CACHE_MAX=0` disables all cache operations: `_get_audio_cache` returns None immediately, `_set_audio_cache` is a no-op. This is the safe default for testing or debugging where deterministic behavior is needed.

---

## Entry 0009 — GPU persistence mode and the entrypoint pattern
**Date**: 2026-02-20
**Type**: What just happened
**Related**: #6

We introduced `docker-entrypoint.sh` as the container's ENTRYPOINT, running GPU tuning commands before exec-ing into uvicorn. GPU settings like `nvidia-smi -pm 1` cannot be baked into the image at build time (no GPU during build). The entrypoint runs at container start when the GPU is available via NVIDIA runtime. The `|| echo` pattern ensures the service starts even without sufficient permissions.

---

## Entry 0009 — flash_attention_2: hardware requirements and fallback
**Date**: 2026-02-20
**Type**: What just happened
**Related**: #8

Switched the model's attention implementation from PyTorch's native SDPA to Flash Attention 2. Flash Attention 2 uses fused CUDA kernels that are 15-20% faster and use less memory by avoiding materialization of the full attention matrix.

Hardware requirement: Flash Attention 2 requires Ampere or newer GPUs (compute capability >= 8.0). This means RTX 3000/4000 series, A100, H100. On older hardware (V100, RTX 2000), the `flash-attn` package either won't install or won't work at runtime.

The fallback pattern is a simple try/except on `import flash_attn`. If the import fails, we fall back to `sdpa` (PyTorch's built-in scaled dot product attention). This means the code works on any GPU — it just runs faster on newer ones. The check happens at model load time, not at import time, so the server starts correctly even without flash-attn installed.

---

## Entry 0011 — Voice prompt cache: hash bytes not filenames
**Date**: 2026-02-20
**Type**: Why this design
**Related**: Issue #15 — Add voice prompt cache for /clone endpoint

The voice cloning endpoint processes reference audio on every request: read bytes, decode with soundfile, convert stereo to mono. When the same reference audio is reused across requests (the common case — a user picks a voice and uses it repeatedly), this processing is redundant.

The cache key is a SHA-256 hash of the raw audio bytes, not the filename. Filenames are unreliable — the same file can be uploaded with different names, and different files can share a name. Content hashing guarantees that identical audio produces the same key regardless of how it was uploaded.

The cache uses `collections.OrderedDict` as an LRU. On hit, `move_to_end()` promotes the entry; on insert, `popitem(last=False)` evicts the oldest entry when capacity exceeds `VOICE_CACHE_MAX`. This is simpler than `functools.lru_cache` because the cache key is a hash string (not the raw bytes), and we need manual control over cache size via an env var that can be set to 0 to disable caching entirely.

The health endpoint exposes `voice_cache_size`, `voice_cache_max`, and `voice_cache_hits` so operators can monitor hit rates and tune capacity.

---

## Entry 0010 — torch.compile: the first-inference cost
**Date**: 2026-02-20
**Type**: What just happened
**Related**: #9

Enabled `torch.compile(model.model, mode="reduce-overhead", fullgraph=False)` after model loading. This tells PyTorch to trace the model's forward pass and generate optimized CUDA kernels, eliminating Python overhead on subsequent calls.

The trade-off is first-inference latency. The first call after compilation triggers the tracing/compiling step, which can take 10-30 seconds depending on model size and GPU. After that, every subsequent inference is faster. For a TTS server that loads the model once and runs many requests, this is a clear win -- the compilation cost is amortized across all requests.

`mode="reduce-overhead"` uses CUDA graphs which are ideal for repeated inference with similar-shaped inputs (exactly the TTS use case). `fullgraph=False` allows partial compilation if some operations aren't compilable, avoiding hard failures.

The `TORCH_COMPILE` env var (default true) provides an escape hatch for environments where compilation causes issues (older PyTorch versions, unsupported ops).

---

## Entry 0012 — SSE streaming: base64 PCM over text/event-stream
**Date**: 2026-02-20
**Type**: What just happened
**Related**: Issue #3

The streaming endpoint sends audio as base64-encoded raw PCM inside SSE events. This was chosen over chunked WAV (which requires a RIFF header with total data size, impossible for streaming) and raw binary HTTP chunks (no framing protocol, client must guess byte boundaries). SSE gives us text-based framing with `data:` prefix and `\n\n` delimiters, plus built-in reconnection semantics. Base64 adds ~33% overhead but keeps the protocol clean — for zero-overhead binary streaming, issue #4 adds a separate raw PCM endpoint. The `_last_used` update per chunk prevents the idle watchdog from unloading the model mid-stream.

---

## Entry 0011 — TF32: why it is safe for this model
**Date**: 2026-02-20
**Type**: Why this design
**Related**: #5

TF32 (TensorFloat-32) is a numeric format available on Ampere and newer NVIDIA GPUs. It uses the same 8-bit exponent as float32 but truncates the mantissa from 23 bits to 10 bits, allowing matrix multiplications to run on Tensor Core hardware at roughly 3x the throughput.

The safety argument for enabling TF32 on this model is straightforward: the model already runs in bfloat16, which has only 7 bits of mantissa. TF32 intermediate operations have 10 bits of mantissa — strictly more precision than the model's own weight format. Enabling TF32 cannot lose information that bfloat16 already discards.

Two separate flags are needed: `torch.backends.cuda.matmul.allow_tf32` controls general matrix multiplication, and `torch.backends.cudnn.allow_tf32` controls cuDNN convolution operations. Both default to False in PyTorch. On pre-Ampere GPUs these flags are no-ops — the hardware simply ignores them.

The test strategy uses mock-based reimport rather than `if torch.cuda.is_available()` guards. This ensures tests actually assert on non-CUDA CI machines instead of silently passing. The pattern: reset flags to False, mock `cuda.is_available` to return True, reimport the server module, verify flags became True.

---

## Entry 0013 — Why lifespan over @app.on_event
**Date**: 2026-02-20
**Type**: Why this design
**Related**: #33

FastAPI deprecated `@app.on_event("startup")` and `@app.on_event("shutdown")` in version 0.93.0. The replacement is the lifespan context manager pattern: `@asynccontextmanager async def lifespan(app)`. The yield point separates startup from shutdown — everything before yield runs at startup, everything after runs at shutdown.

The practical advantage is not just suppressing deprecation warnings. The old pattern had separate startup and shutdown functions with no shared scope. If startup allocated a resource (like a background task handle), the shutdown function needed that handle stored in a global or on the app object. With lifespan, variables from the startup section are naturally in scope during teardown:

```python
@asynccontextmanager
async def lifespan(app):
    watchdog_task = asyncio.create_task(_idle_watchdog())  # startup
    yield
    watchdog_task.cancel()  # shutdown — same scope, no global needed
```

For our server, the immediate benefit is that model unload now runs on graceful shutdown. Previously, if the container was stopped with SIGTERM, the model was not explicitly unloaded — the process just died and the GPU driver reclaimed VRAM. With the lifespan teardown, we run `_unload_model_sync()` which calls `gc.collect()` and `torch.cuda.empty_cache()` before exit. This ensures clean VRAM release in shared GPU environments where another container might be waiting for memory.

---

## Entry 0009 — Prometheus instrumentator vs manual metrics
**Date**: 2026-02-20
**Type**: Why this design
**Related**: #30

We use two layers of metrics: automatic HTTP instrumentation via `prometheus-fastapi-instrumentator` and custom TTS-specific metrics via `prometheus-client` directly.

The instrumentator auto-adds request count, latency histograms, and response size metrics for every endpoint. This covers the "web server" dimension -- you can alert on 5xx rate, p99 latency, and throughput without writing any code.

The custom metrics cover the "TTS engine" dimension that the instrumentator cannot see: `tts_inference_duration_seconds` measures only the model inference time (excluding queue wait and audio encoding), `tts_requests_total` breaks down by voice and format, and `tts_model_loaded` tracks whether the model is in VRAM. These are the metrics you actually need for capacity planning -- if inference duration is climbing, the GPU is under pressure; if model_loaded flaps between 0 and 1, the idle timeout is too aggressive.

The implementation is gated behind `PROMETHEUS_ENABLED` and falls back gracefully if the packages are not installed. This keeps Prometheus as a soft dependency -- the server works without it.

---

## Entry 0014 — jemalloc: why the default allocator causes RSS bloat
**Date**: 2026-02-20
**Type**: Why this design
**Related**: #21

Python's default memory allocator (glibc ptmalloc2) uses per-thread arenas to reduce lock contention. Each arena independently allocates and frees memory from the OS. The problem: when a thread frees a block, the arena may not return the underlying pages to the OS if adjacent blocks are still allocated. Over hours of operation with many small allocations (tokenizer strings, audio buffers, numpy intermediates), the RSS grows 2-3x beyond actual usage.

jemalloc uses a different strategy: size-class segregated regions with background thread compaction. The `dirty_decay_ms=1000` setting tells jemalloc to return freed pages to the OS within 1 second. `muzzy_decay_ms=0` tells it to immediately decommit pages rather than keeping them as "muzzy" (mapped but uncommitted). `background_thread:true` enables a dedicated thread that handles the decay without blocking application threads.

The LD_PRELOAD approach is the least invasive: the application code is unchanged, the allocator swap happens at process startup, and removing the env var reverts to ptmalloc2. No server.py changes needed.

---

## Entry 0015 — CPU affinity: sched_setaffinity over taskset
**Date**: 2026-02-20
**Type**: What could go wrong
**Related**: Issue #22

The original implementation used `os.system(f"taskset -p -c {cores} {pid}")` which has two problems. First, it is a command injection vector — the `INFERENCE_CPU_CORES` env var is interpolated directly into a shell command. Setting it to `0-7; rm -rf /` would execute the destructive command. Second, `taskset -p` changes the affinity for the entire process (all threads), including the uvicorn event loop, which defeats the purpose of pinning only the inference thread. The fix uses `os.sched_setaffinity(0, cores)` which: (a) takes a set of integers, eliminating shell injection, and (b) is a direct syscall wrapper with no shell involved. Note that `os.sched_setaffinity(0, ...)` still sets affinity for the calling process (PID 0 = current), not just the calling thread. True per-thread affinity would require `pthread_setaffinity_np` via ctypes, which is too fragile. The process-level approach is acceptable because the inference thread pool has only one thread and the event loop is lightweight.
