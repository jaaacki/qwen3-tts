# Learning Log

Decisions, patterns, and lessons from building the Qwen3-TTS server. Each entry is written so someone new to the project can understand the reasoning without digging through commit history.

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

## Entry 0008 — TF32: why it is safe for this model
**Date**: 2026-02-20
**Type**: Why this design
**Related**: #5

TF32 (TensorFloat-32) is a numeric format available on Ampere and newer NVIDIA GPUs. It uses the same 8-bit exponent as float32 but truncates the mantissa from 23 bits to 10 bits, allowing matrix multiplications to run on Tensor Core hardware at roughly 3x the throughput.

The safety argument for enabling TF32 on this model is straightforward: the model already runs in bfloat16, which has only 7 bits of mantissa. TF32 intermediate operations have 10 bits of mantissa — strictly more precision than the model's own weight format. Enabling TF32 cannot lose information that bfloat16 already discards.

Two separate flags are needed: `torch.backends.cuda.matmul.allow_tf32` controls general matrix multiplication, and `torch.backends.cudnn.allow_tf32` controls cuDNN convolution operations. Both default to False in PyTorch. On pre-Ampere GPUs these flags are no-ops — the hardware simply ignores them.

The test strategy uses mock-based reimport rather than `if torch.cuda.is_available()` guards. This ensures tests actually assert on non-CUDA CI machines instead of silently passing. The pattern: reset flags to False, mock `cuda.is_available` to return True, reimport the server module, verify flags became True.
