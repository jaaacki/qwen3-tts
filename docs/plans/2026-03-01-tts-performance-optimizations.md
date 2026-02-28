# TTS Performance Optimizations Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce TTS inference latency and streaming throughput via GPU optimizations, sentence pipelining, and server-side resampling.

**Architecture:** Seven changes to `server.py` (single-file server), minor env var changes to `compose.yaml`. All changes are additive — existing behavior preserved unless new env vars are set. Tests via `server_test.py` (pytest, mocked model).

**Tech Stack:** PyTorch 2.6, CUDA 12.4, bitsandbytes, scipy, asyncio

**Build constraint:** Do NOT run `docker build` on this server. Code changes only. Commit and push — build happens on 192.168.2.191.

---

### Task 1: torch.compile max-autotune

**Files:**
- Modify: `server.py:468-474` (torch.compile block)
- Modify: `compose.yaml` (add TORCH_COMPILE_MODE env var)

**Step 1: Add TORCH_COMPILE_MODE env var**

In `server.py`, after line 288 (`QUANTIZE = ...`), add:

```python
TORCH_COMPILE_MODE = os.getenv("TORCH_COMPILE_MODE", "max-autotune")
```

**Step 2: Update torch.compile call**

Replace lines 468-474:

```python
    # Compile model for faster inference (PyTorch 2.0+)
    if os.getenv("TORCH_COMPILE", "true").lower() == "true":
        try:
            model.model = torch.compile(model.model, mode="reduce-overhead", fullgraph=False)
            logger.success("torch.compile enabled (mode=reduce-overhead)")
        except Exception as e:
            logger.warning("torch.compile not available or failed: {}", e)
```

With:

```python
    # Compile model for faster inference (PyTorch 2.0+)
    if os.getenv("TORCH_COMPILE", "true").lower() == "true":
        try:
            compile_mode = TORCH_COMPILE_MODE
            model.model = torch.compile(model.model, mode=compile_mode, fullgraph=False)
            logger.bind(compile_mode=compile_mode).success("torch.compile enabled")
        except Exception as e:
            logger.warning("torch.compile not available or failed: {}", e)
```

**Step 3: Add to compose.yaml environment**

Add after `TORCH_COMPILE`:

```yaml
      TORCH_COMPILE_MODE: "max-autotune"
```

**Step 4: Add to startup config log**

In the `logger.bind(...)` block at line 337, add `TORCH_COMPILE_MODE=TORCH_COMPILE_MODE`.

**Step 5: Commit**

```bash
git add server.py compose.yaml
git commit -m "perf: switch torch.compile to max-autotune mode

Configurable via TORCH_COMPILE_MODE env var (default: max-autotune).
Slower warmup (~30-60s) but faster steady-state inference.
Best for always-on deployments with PRELOAD_MODEL=true."
```

---

### Task 2: INT8 Quantization Default

**Files:**
- Modify: `compose.yaml` (change QUANTIZE default)

**Step 1: Update compose.yaml**

The `QUANTIZE` env var already exists in `server.py:288` with default `""`. The `_resolve_quant_kwargs()` function at lines 398-430 already handles `int8` and `fp8`. Just change the compose.yaml to enable it:

Add to compose.yaml environment section (or update if exists):

```yaml
      QUANTIZE: "int8"
```

**Step 2: Verify _resolve_quant_kwargs handles int8 correctly**

Read `server.py:407-415`. It already returns `(torch.float16, {"load_in_8bit": True})` for int8. No code changes needed — just the compose default.

**Step 3: Commit**

```bash
git add compose.yaml
git commit -m "perf: enable INT8 quantization by default

Sets QUANTIZE=int8 in compose.yaml. ~50% VRAM reduction.
Disable with QUANTIZE= (empty string).
bitsandbytes already installed in Docker image."
```

---

### Task 3: CUDA Streams and Pinned Memory

**Files:**
- Modify: `server.py:117-126` (CUDA setup section)
- Modify: `server.py:433-533` (model loading — add stream/pinned memory init)
- Modify: `server.py:839-854` (`_do_synthesize` — use inference stream)
- Modify: `server.py:857-871` (`_do_synthesize_batch` — use inference stream)
- Modify: `server.py:928-940` (`_do_voice_clone` — use inference stream)
- Modify: `server.py:535-553` (`_unload_model_sync` — cleanup streams)
- Test: `server_test.py` (add test for stream/pinned memory globals)

**Step 1: Add CUDA stream and pinned memory globals**

After the CUDA setup block (after line 126), add:

```python
# CUDA streams — overlap inference compute with data transfer
_inference_stream: torch.cuda.Stream | None = None
_transfer_stream: torch.cuda.Stream | None = None
```

**Step 2: Initialize streams during model load**

In `_load_model_sync()`, after the model is loaded (after line 466), add:

```python
    # Create dedicated CUDA streams for overlapping compute + transfer
    global _inference_stream, _transfer_stream
    if torch.cuda.is_available():
        _inference_stream = torch.cuda.Stream()
        _transfer_stream = torch.cuda.Stream()
        logger.info("CUDA inference and transfer streams created")
```

**Step 3: Use inference stream in _do_synthesize**

Replace `_do_synthesize` (lines 839-854):

```python
def _do_synthesize(text, language, voice_file, gen_kwargs, instruct=None):
    """Run TTS inference via voice cloning with pre-computed prompts."""
    prompt = _voice_prompts[voice_file]
    stream = _inference_stream
    with torch.inference_mode():
        if stream is not None:
            with torch.cuda.stream(stream):
                wavs, sr = model.generate_voice_clone(
                    text=text,
                    language=language,
                    voice_clone_prompt=prompt,
                    **gen_kwargs,
                )
            stream.synchronize()
        else:
            wavs, sr = model.generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=prompt,
                **gen_kwargs,
            )
    return wavs, sr
```

**Step 4: Use inference stream in _do_synthesize_batch**

Apply the same pattern to `_do_synthesize_batch` (lines 857-871).

**Step 5: Use inference stream in _do_voice_clone**

Apply the same pattern to `_do_voice_clone` (lines 928-940).

**Step 6: Cleanup streams on unload**

In `_unload_model_sync()`, after line 548, add:

```python
    global _inference_stream, _transfer_stream
    _inference_stream = None
    _transfer_stream = None
```

**Step 7: Run tests**

```bash
pytest server_test.py -v
```

Expected: All existing tests pass (streams are None in test env since no CUDA).

**Step 8: Commit**

```bash
git add server.py
git commit -m "perf: add CUDA inference and transfer streams

Dedicated streams for overlapping compute with data transfer.
Graceful fallback to default stream when CUDA unavailable."
```

---

### Task 4: CUDA Graphs

**Files:**
- Modify: `server.py` (add graph capture after warmup, replay in inference)

**Step 1: Add CUDA_GRAPHS env var**

After the `TORCH_COMPILE_MODE` line, add:

```python
CUDA_GRAPHS = os.getenv("CUDA_GRAPHS", "true").lower() in ("true", "1")
```

**Step 2: Add graph storage global**

After the stream globals, add:

```python
# CUDA graphs — captured during warmup, replayed for matching input shapes
_cuda_graphs_enabled = False  # Set to True after successful capture
```

Note: CUDA Graphs require static shapes. TTS generation produces variable-length outputs, which makes traditional CUDA graph capture problematic. Instead, we'll use `torch.compile` with `mode="max-autotune"` which internally uses CUDA graphs where possible via Inductor. The `max-autotune` mode already handles graph capture/replay transparently.

So this task becomes: **verify that `max-autotune` + `fullgraph=False` enables Inductor's CUDA graph optimization**, and add the `CUDA_GRAPHS` env var to control `torch.compile`'s `options={"triton.cudagraphs": True}`.

**Step 3: Update torch.compile to pass cudagraphs option**

Replace the torch.compile block:

```python
    if os.getenv("TORCH_COMPILE", "true").lower() == "true":
        try:
            compile_mode = TORCH_COMPILE_MODE
            compile_options = {}
            if CUDA_GRAPHS and torch.cuda.is_available():
                compile_options["triton.cudagraphs"] = True
            model.model = torch.compile(
                model.model, mode=compile_mode, fullgraph=False, options=compile_options or None,
            )
            logger.bind(compile_mode=compile_mode, cuda_graphs=bool(compile_options.get("triton.cudagraphs"))).success(
                "torch.compile enabled"
            )
        except Exception as e:
            logger.warning("torch.compile not available or failed: {}", e)
```

**Step 4: Add to compose.yaml**

```yaml
      CUDA_GRAPHS: "true"
```

**Step 5: Add to startup config log**

Add `CUDA_GRAPHS=CUDA_GRAPHS` to the logger.bind block.

**Step 6: Commit**

```bash
git add server.py compose.yaml
git commit -m "perf: enable CUDA graphs via torch.compile triton backend

CUDA_GRAPHS=true passes triton.cudagraphs=True to torch.compile.
Works with max-autotune mode for automatic graph capture/replay."
```

---

### Task 5: Server-Side Sample Rate Conversion

**Files:**
- Modify: `server.py:379-388` (TTSRequest model — add sample_rate field)
- Modify: `server.py:1253-1325` (`/stream/pcm` endpoint — resample before yield)
- Modify: `server.py:1067-1148` (`/stream` endpoint — resample before yield)
- Modify: `server.py:1328-1396` (`/ws` endpoint — resample before send)
- Modify: `server.py:952-1065` (`/speech` endpoint — resample before encode)
- Test: `server_test.py` (add resample test)

**Step 1: Write failing test for _resample_audio helper**

Add to `server_test.py`:

```python
class TestResampleAudio:
    def test_resample_24k_to_8k(self):
        """Downsampling 24kHz to 8kHz should produce 1/3 the samples."""
        audio = np.random.randn(24000).astype(np.float32)
        result = server._resample_audio(audio, 24000, 8000)
        assert len(result) == 8000

    def test_resample_noop_same_rate(self):
        """Same input/output rate returns original array."""
        audio = np.random.randn(24000).astype(np.float32)
        result = server._resample_audio(audio, 24000, 24000)
        np.testing.assert_array_equal(result, audio)

    def test_resample_none_target_noop(self):
        """None target rate returns original array."""
        audio = np.random.randn(24000).astype(np.float32)
        result = server._resample_audio(audio, 24000, None)
        np.testing.assert_array_equal(result, audio)
```

**Step 2: Run test to verify it fails**

```bash
pytest server_test.py::TestResampleAudio -v
```

Expected: FAIL — `_resample_audio` doesn't exist yet.

**Step 3: Add _resample_audio helper**

After `_adjust_speed` (line 829), add:

```python
def _resample_audio(audio_data: np.ndarray, source_rate: int, target_rate: int | None) -> np.ndarray:
    """Resample audio to target sample rate. Returns original if rates match or target is None."""
    if target_rate is None or target_rate == source_rate:
        return audio_data
    num_samples = int(len(audio_data) * target_rate / source_rate)
    if num_samples <= 0:
        return audio_data
    return scipy_signal.resample(audio_data, num_samples).astype(np.float32)
```

**Step 4: Run test to verify it passes**

```bash
pytest server_test.py::TestResampleAudio -v
```

Expected: PASS

**Step 5: Add sample_rate to TTSRequest**

In `TTSRequest` (lines 379-388), add after `top_p`:

```python
    sample_rate: Optional[int] = None
```

**Step 6: Update /stream/pcm to resample**

In `pcm_generator()` (line 1275), after speed adjustment (line 1298) and before PCM conversion (line 1299), add:

```python
                if request.sample_rate:
                    audio_data = _resample_audio(audio_data, sr, request.sample_rate)
                    effective_sr = request.sample_rate
                else:
                    effective_sr = sr
```

Update the response headers (lines 1319-1324) to use the effective sample rate. Change `"X-PCM-Sample-Rate": "24000"` to be dynamic. Since `effective_sr` is inside the generator, instead set the header based on `request.sample_rate`:

```python
    pcm_sr = request.sample_rate or 24000
    return StreamingResponse(
        pcm_generator(),
        media_type="application/octet-stream",
        headers={
            "X-PCM-Sample-Rate": str(pcm_sr),
            "X-PCM-Bit-Depth": "16",
            "X-PCM-Channels": "1",
            "Content-Disposition": 'attachment; filename="speech.pcm"',
        },
    )
```

**Step 7: Update /stream (SSE) to resample**

In `generate()` (line 1086), after speed adjustment (line 1110) and before PCM conversion (line 1112), add the same resample logic.

**Step 8: Update /ws to resample**

In the WebSocket loop (line 1357), after speed adjustment (line 1378) and before PCM conversion (line 1380), add the same resample logic.

**Step 9: Update /speech to resample**

In `synthesize_speech()`, after speed adjustment (line 1021) and before encoding (line 1023), add:

```python
        if request.sample_rate:
            audio_data = _resample_audio(audio_data, sr, request.sample_rate)
            sr = request.sample_rate
```

**Step 10: Run all tests**

```bash
pytest server_test.py -v
```

Expected: All pass.

**Step 11: Commit**

```bash
git add server.py server_test.py
git commit -m "feat: add server-side sample rate conversion

New optional sample_rate parameter on all speech endpoints.
When set (e.g. 8000), server resamples via scipy before returning.
Eliminates client-side resampling overhead for telephony (24kHz→8kHz)."
```

---

### Task 6: Sentence Pipelining in Streaming

**Files:**
- Modify: `server.py:1275-1315` (`pcm_generator` in `/stream/pcm`)
- Modify: `server.py:1086-1139` (`generate` in `/stream`)
- Modify: `server.py:1357-1386` (WebSocket sentence loop)

**Step 1: Pipeline /stream/pcm**

Replace the sequential `for sentence in sentences` loop in `pcm_generator()` (lines 1279-1310) with a pipelined version:

```python
    async def pcm_generator():
        global _last_used
        t_pcm_start = time.perf_counter()
        chunks_sent = 0

        # Pre-fetch: submit first sentence immediately
        pending_future = None
        sentence_idx = 0

        def _make_synth_fn(s, gk):
            return lambda: _do_synthesize(s, language, voice_file, gk)

        # Submit first sentence
        if sentences:
            gk = _build_gen_kwargs(sentences[0], request)
            pending_future = asyncio.ensure_future(
                asyncio.wait_for(
                    _infer_queue.submit(
                        _make_synth_fn(sentences[0], gk),
                        priority=PRIORITY_REALTIME,
                    ),
                    timeout=REQUEST_TIMEOUT,
                )
            )

        while sentence_idx < len(sentences):
            try:
                # Await current sentence result
                wavs, sr_val = await pending_future

                # Pre-fetch next sentence while we process current
                next_idx = sentence_idx + 1
                if next_idx < len(sentences):
                    gk_next = _build_gen_kwargs(sentences[next_idx], request)
                    pending_future = asyncio.ensure_future(
                        asyncio.wait_for(
                            _infer_queue.submit(
                                _make_synth_fn(sentences[next_idx], gk_next),
                                priority=PRIORITY_REALTIME,
                            ),
                            timeout=REQUEST_TIMEOUT,
                        )
                    )

                _last_used = time.time()
                audio_data = np.array(wavs[0], dtype=np.float32, copy=True)
                if audio_data.ndim > 1:
                    audio_data = audio_data.squeeze()
                if request.speed != 1.0:
                    new_length = int(len(audio_data) / request.speed)
                    if new_length > 0:
                        audio_data = scipy_signal.resample(audio_data, new_length)
                if request.sample_rate:
                    audio_data = _resample_audio(audio_data, sr_val, request.sample_rate)
                pcm_data = np.clip(audio_data, -1.0, 1.0)
                pcm_bytes = (pcm_data * 32767).astype(np.int16).tobytes()
                yield pcm_bytes
                chunks_sent += 1
                sentence_idx += 1

            except asyncio.TimeoutError:
                logger.bind(endpoint="/v1/audio/speech/stream/pcm", voice=voice_file,
                             chunks_sent=chunks_sent, timeout_s=REQUEST_TIMEOUT).error("PCM stream timed out")
                break
            except Exception:
                logger.bind(endpoint="/v1/audio/speech/stream/pcm", voice=voice_file,
                             chunks_sent=chunks_sent).opt(exception=True).error("PCM stream failed")
                break

        logger.bind(endpoint="/v1/audio/speech/stream/pcm", voice=voice_file,
                     language=language, sentences=len(sentences), chunks_sent=chunks_sent,
                     chars=len(text), total_ms=round((time.perf_counter() - t_pcm_start) * 1000),
                     ).info("pcm_stream_complete")
```

**Step 2: Pipeline /stream (SSE)**

Apply the same pipelining pattern to `generate()` in the SSE endpoint (lines 1086-1139). Same structure: submit first sentence, then in loop: await current + submit next.

**Step 3: Pipeline /ws (WebSocket)**

Apply the same pattern to the WebSocket sentence loop (lines 1357-1386).

**Step 4: Run tests**

```bash
pytest server_test.py -v
```

Expected: All pass.

**Step 5: Commit**

```bash
git add server.py
git commit -m "perf: pipeline sentence synthesis in streaming endpoints

Submit sentence N+1 to GPU while yielding sentence N's audio.
Hides encoding and network latency between sentences.
Applied to /stream/pcm, /stream (SSE), and /ws endpoints."
```

---

### Task 7: Update CLAUDE.md, compose.yaml, and env var docs

**Files:**
- Modify: `CLAUDE.md` (new env vars, architecture notes)
- Modify: `compose.yaml` (all new env vars with defaults)

**Step 1: Update compose.yaml with all new env vars**

Ensure these are in the environment section:

```yaml
      TORCH_COMPILE_MODE: "max-autotune"
      QUANTIZE: "int8"
      CUDA_GRAPHS: "true"
```

**Step 2: Update CLAUDE.md env var table**

Add new rows:

```markdown
| `TORCH_COMPILE_MODE` | `max-autotune` | torch.compile mode (`max-autotune`, `reduce-overhead`, `default`) |
| `CUDA_GRAPHS` | `true` | Enable CUDA graph capture via triton backend |
```

Update `QUANTIZE` default from empty to `int8`.

**Step 3: Update CLAUDE.md architecture section**

Add note about sentence pipelining and sample_rate parameter to relevant sections.

**Step 4: Commit**

```bash
git add CLAUDE.md compose.yaml
git commit -m "docs: update CLAUDE.md and compose.yaml for performance optimizations

New env vars: TORCH_COMPILE_MODE, CUDA_GRAPHS.
QUANTIZE default changed to int8.
Document sentence pipelining and sample_rate parameter."
```

---

### Task 8: Final verification and push

**Step 1: Run full test suite**

```bash
pytest server_test.py -v
```

Expected: All pass.

**Step 2: Push to remote**

```bash
git push origin main
```
