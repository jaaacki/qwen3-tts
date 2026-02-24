from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from typing import Optional
import torch
import soundfile as sf
import hashlib
import io
import json
import os
import gc
import asyncio
import time
import re
import uuid
import numpy as np
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import scipy.signal as scipy_signal
import logging
import base64

logger = logging.getLogger("qwen3-tts")

try:
    import pyrubberband as _pyrubberband
except ImportError:
    _pyrubberband = None

# Prometheus metrics (optional — enabled by default)
PROMETHEUS_ENABLED = os.getenv("PROMETHEUS_ENABLED", "true").lower() in ("true", "1")

if PROMETHEUS_ENABLED:
    try:
        from prometheus_client import Counter, Histogram, Gauge
        from prometheus_fastapi_instrumentator import Instrumentator

        tts_requests_total = Counter(
            "tts_requests_total", "Total TTS requests", ["voice", "format"]
        )
        tts_inference_duration = Histogram(
            "tts_inference_duration_seconds", "Inference duration in seconds"
        )
        tts_model_loaded = Gauge(
            "tts_model_loaded", "Whether model is currently loaded (1=yes, 0=no)"
        )
        _prometheus_available = True
    except ImportError:
        _prometheus_available = False
else:
    _prometheus_available = False

# Structured JSON logging
LOG_FORMAT = os.getenv("LOG_FORMAT", "json")


class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        # Include extra fields passed via extra= kwarg
        if hasattr(record, "extra_fields"):
            log_obj.update(record.extra_fields)
        return json.dumps(log_obj)


def _setup_logging():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    if LOG_FORMAT == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s"
        ))
    root.handlers = [handler]


_setup_logging()
logger = logging.getLogger("qwen3-tts")

try:
    from pydub import AudioSegment as _PydubAudioSegment
except ImportError:
    _PydubAudioSegment = None

try:
    import torchaudio
    import torchaudio.functional as torchaudio_F
    _TORCHAUDIO = True
except ImportError:
    _TORCHAUDIO = False

# Enable cudnn autotuner — finds fastest convolution algorithms for the GPU
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True   # 3x faster matmul on Ampere+ GPUs
    torch.backends.cudnn.allow_tf32 = True           # enable TF32 for cuDNN ops

# Eager model preload on startup (default: false)
PRELOAD_MODEL = os.getenv("PRELOAD_MODEL", "false").lower() in ("true", "1")

@asynccontextmanager
async def lifespan(app):
    # Startup
    if _prometheus_available:
        Instrumentator().instrument(app).expose(app)
    _set_cpu_affinity()
    asyncio.create_task(_idle_watchdog())
    if PRELOAD_MODEL:
        print("PRELOAD_MODEL=true: loading model at startup")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(_infer_executor, _load_model_sync)
    print("Server started")
    yield
    # Shutdown
    print("Server shutting down")
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(_infer_executor, _unload_model_sync)

app = FastAPI(title="Qwen3-TTS API", lifespan=lifespan)

model = None
loaded_model_id = None

# Single-thread executor for GPU inference — avoids default pool overhead
_infer_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts-infer")

# CPU executor for audio encoding — runs in parallel with GPU inference
_encode_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tts-encode")

# Semaphore to serialize GPU inference — prevents OOM with concurrent requests
_infer_semaphore = asyncio.Semaphore(1)

# Lock to prevent concurrent load/unload
_model_lock = asyncio.Lock()

# Request timeout in seconds
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "300"))

# Idle unload timeout in seconds (0 = disabled)
IDLE_TIMEOUT = int(os.getenv("IDLE_TIMEOUT", "120"))

# VAD silence trimming (strip leading/trailing silence from audio)
VAD_TRIM = os.getenv("VAD_TRIM", "true").lower() in ("true", "1", "yes")

# Text normalization (expand numbers, currency, abbreviations)
TEXT_NORMALIZE = os.getenv("TEXT_NORMALIZE", "true").lower() in ("true", "1", "yes")

# Queue depth limit — 503 when exceeded (0 = unlimited)
MAX_QUEUE_DEPTH = int(os.getenv("MAX_QUEUE_DEPTH", "5"))
_queue_depth = 0

# Track last request time
_last_used = 0.0

# Audio output LRU cache — skips GPU entirely on cache hit
_AUDIO_CACHE_MAX = int(os.getenv("AUDIO_CACHE_MAX", "256"))
_audio_cache: OrderedDict[str, tuple[bytes, str]] = OrderedDict()


def _audio_cache_key(text: str, voice: str, speed: float, fmt: str, language: str = "", instruct: str = "") -> str:
    raw = f"{text}|{voice}|{speed}|{fmt}|{language}|{instruct}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _get_audio_cache(key: str) -> tuple[bytes, str] | None:
    if _AUDIO_CACHE_MAX <= 0:
        return None
    if key in _audio_cache:
        _audio_cache.move_to_end(key)
        return _audio_cache[key]
    return None


def _set_audio_cache(key: str, data: bytes, content_type: str) -> None:
    if _AUDIO_CACHE_MAX <= 0:
        return
    if len(_audio_cache) >= _AUDIO_CACHE_MAX:
        _audio_cache.popitem(last=False)
    _audio_cache[key] = (data, content_type)

# Voice prompt cache — caches speaker embeddings by reference audio content hash
# Uses model.create_voice_clone_prompt() to precompute the embedding once,
# so repeat clone requests skip the encoder pass entirely.
VOICE_CACHE_MAX = int(os.getenv("VOICE_CACHE_MAX", "32"))
_voice_prompt_cache: OrderedDict = OrderedDict()
_voice_cache_hits = 0

# Map OpenAI-style voice names to Qwen3-TTS speakers
VOICE_MAP = {
    # Direct Qwen speaker names
    "vivian": "vivian",
    "serena": "serena",
    "uncle_fu": "uncle_fu",
    "dylan": "dylan",
    "eric": "eric",
    "ryan": "ryan",
    "aiden": "aiden",
    "ono_anna": "ono_anna",
    "sohee": "sohee",
    # OpenAI-compatible aliases
    "alloy": "ryan",
    "echo": "aiden",
    "fable": "dylan",
    "onyx": "uncle_fu",
    "nova": "vivian",
    "shimmer": "serena",
}

DEFAULT_VOICE = "vivian"


class TTSRequest(BaseModel):
    input: str
    voice: Optional[str] = None
    response_format: str = "wav"
    speed: float = 1.0
    language: Optional[str] = None
    instruct: Optional[str] = None


def _release_gpu_full():
    """Full GPU memory release — only used during model unload."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _load_model_sync():
    """Load model into GPU (blocking). Called from async context via lock."""
    global model, loaded_model_id, _last_used
    from qwen_tts import Qwen3TTSModel

    if model is not None:
        return

    model_id = os.getenv("MODEL_ID", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    loaded_model_id = model_id

    # Prefer flash_attention_2 on Ampere+ GPUs; fall back to sdpa
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"
        print("flash_attention_2 not available, falling back to sdpa")

    print(f"Loading {model_id} (attn={attn_impl})...")
    model = Qwen3TTSModel.from_pretrained(
        model_id,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation=attn_impl,
    )

    # Compile model for faster inference (PyTorch 2.0+)
    if os.getenv("TORCH_COMPILE", "true").lower() == "true":
        try:
            import torch._dynamo  # noqa: F401
            model.model = torch.compile(model.model, mode="reduce-overhead", fullgraph=False)
            print("torch.compile enabled on model (mode=reduce-overhead)")
        except Exception as e:
            print(f"torch.compile not available or failed: {e}")

    # Multi-length warmup to pre-cache CUDA kernels for different input sizes
    if torch.cuda.is_available():
        print("Warming up GPU with multi-length synthesis...")
        warmup_texts = [
            "Hello.",                                       # ~5 tokens — short prompt path
            "Hello, how are you doing today?",              # ~20 tokens — medium
            "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.",  # ~50 tokens — longer
        ]
        for text in warmup_texts:
            try:
                with torch.inference_mode():
                    model.generate_custom_voice(
                        text=text,
                        language="English",
                        speaker="vivian",
                        max_new_tokens=256,
                    )
                print(f"  Warmup: synthesized {len(text)} chars")
            except Exception as e:
                print(f"  Warmup failed for '{text[:20]}...': {e}")
        # Clear warmup allocations so steady-state VRAM is clean
        _release_gpu_full()

        # Pre-warm CUDA memory pool — allocate and free a large tensor so the
        # allocator pre-reserves a contiguous block, reducing first-request jitter
        print("Pre-warming CUDA memory pool...")
        try:
            dummy = torch.empty(64 * 1024 * 1024, dtype=torch.bfloat16, device="cuda")
            del dummy
            print("  Allocated and freed 128 MB dummy tensor")
        except Exception as e:
            print(f"  CUDA pool pre-warm failed: {e}")

    _last_used = time.time()
    if _prometheus_available:
        tts_model_loaded.set(1)
    print(f"Model loaded: {model_id}")
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"  GPU Allocated: {allocated:.0f} MB, Reserved: {reserved:.0f} MB")


def _unload_model_sync():
    """Unload model from GPU to free VRAM."""
    global model

    if model is None:
        return

    print("Unloading model (idle timeout)...")
    del model
    model = None
    if _prometheus_available:
        tts_model_loaded.set(0)
    _release_gpu_full()

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"Model unloaded. GPU: Allocated: {allocated:.0f} MB, Reserved: {reserved:.0f} MB")


async def _ensure_model_loaded():
    """Load model if not already loaded. Thread-safe via lock."""
    global _last_used
    if model is not None:
        _last_used = time.time()
        return
    async with _model_lock:
        if model is not None:
            _last_used = time.time()
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(_infer_executor, _load_model_sync)
        _last_used = time.time()


async def _idle_watchdog():
    """Background task that unloads model after IDLE_TIMEOUT seconds of inactivity."""
    while True:
        await asyncio.sleep(30)
        if IDLE_TIMEOUT <= 0 or model is None:
            continue
        if time.time() - _last_used > IDLE_TIMEOUT:
            async with _model_lock:
                if model is not None and time.time() - _last_used > IDLE_TIMEOUT:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(_infer_executor, _unload_model_sync)


def _parse_cpu_cores(spec: str) -> set[int]:
    """Parse CPU core spec like '0-3,6,8-11' into a set of ints."""
    cores = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            cores.update(range(int(lo), int(hi) + 1))
        else:
            cores.add(int(part))
    return cores


def _set_cpu_affinity():
    """Pin process to GPU-adjacent CPU cores for better cache locality.

    Uses os.sched_setaffinity() instead of taskset to avoid command injection
    and to correctly set affinity for the calling process.
    """
    affinity_cores = os.getenv("INFERENCE_CPU_CORES", "")
    if not affinity_cores:
        return
    try:
        cores = _parse_cpu_cores(affinity_cores)
        os.sched_setaffinity(0, cores)
        print(f"CPU affinity set: cores {sorted(cores)}")
    except Exception as e:
        print(f"Could not set CPU affinity: {e}")


@app.get("/health")
async def health():
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_allocated_mb": round(torch.cuda.memory_allocated() / 1024**2),
            "gpu_reserved_mb": round(torch.cuda.memory_reserved() / 1024**2),
        }
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_id": loaded_model_id,
        "cuda": torch.cuda.is_available(),
        "queue_depth": _queue_depth,
        "max_queue_depth": MAX_QUEUE_DEPTH,
        "voices": list(VOICE_MAP.keys()),
        "audio_cache_size": len(_audio_cache),
        "audio_cache_max": _AUDIO_CACHE_MAX,
        "voice_cache_size": len(_voice_prompt_cache),
        "voice_cache_max": VOICE_CACHE_MAX,
        "voice_cache_hits": _voice_cache_hits,
        **gpu_info,
    }


@app.post("/cache/clear")
async def clear_cache():
    """Clear the audio output cache and voice prompt cache."""
    audio_count = len(_audio_cache)
    voice_count = len(_voice_prompt_cache)
    _audio_cache.clear()
    _voice_prompt_cache.clear()
    return {"audio_cleared": audio_count, "voice_cleared": voice_count}


def convert_audio_format(audio_data: np.ndarray, sample_rate: int, output_format: str) -> tuple[bytes, str]:
    """Convert audio data to the requested format."""
    buffer = io.BytesIO()

    if output_format in ("wav", "wave"):
        if _TORCHAUDIO:
            audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)
            torchaudio.save(buffer, audio_tensor, sample_rate, format="wav")
        else:
            sf.write(buffer, audio_data, sample_rate, format="WAV")
        content_type = "audio/wav"
    elif output_format == "mp3":
        if _PydubAudioSegment is not None:
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, audio_data, sample_rate, format="WAV")
            wav_buffer.seek(0)
            audio_segment = _PydubAudioSegment.from_wav(wav_buffer)
            audio_segment.export(buffer, format="mp3")
            content_type = "audio/mpeg"
        else:
            sf.write(buffer, audio_data, sample_rate, format="WAV")
            content_type = "audio/wav"
    elif output_format == "flac":
        sf.write(buffer, audio_data, sample_rate, format="FLAC")
        content_type = "audio/flac"
    elif output_format == "ogg":
        sf.write(buffer, audio_data, sample_rate, format="OGG")
        content_type = "audio/ogg"
    elif output_format == "opus":
        if _PydubAudioSegment is not None:
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, audio_data, sample_rate, format="WAV")
            wav_buffer.seek(0)
            audio_segment = _PydubAudioSegment.from_wav(wav_buffer)
            audio_segment.export(
                buffer, format="opus", codec="libopus",
                parameters=["-b:a", "32k"]
            )
            content_type = "audio/opus"
        else:
            sf.write(buffer, audio_data, sample_rate, format="WAV")
            content_type = "audio/wav"
    else:
        sf.write(buffer, audio_data, sample_rate, format="WAV")
        content_type = "audio/wav"

    buffer.seek(0)
    return buffer.read(), content_type


def resolve_voice(voice: Optional[str]) -> str:
    """Resolve a voice name to a Qwen3-TTS speaker name."""
    if not voice:
        return DEFAULT_VOICE
    voice_lower = voice.lower().replace(" ", "_")
    return VOICE_MAP.get(voice_lower, voice_lower)


_langdetect_model = None


def _get_langdetect():
    """Lazy-load fasttext language detector."""
    global _langdetect_model
    if _langdetect_model is None:
        try:
            from fasttext_langdetect import detect
            _langdetect_model = detect
        except ImportError:
            _langdetect_model = False
    return _langdetect_model


def _detect_language_unicode(text: str) -> str:
    """Fallback: language detection based on Unicode character ranges."""
    for ch in text:
        if '\u4e00' <= ch <= '\u9fff':
            return "Chinese"
        if '\u3040' <= ch <= '\u309f' or '\u30a0' <= ch <= '\u30ff':
            return "Japanese"
        if '\uac00' <= ch <= '\ud7af':
            return "Korean"
    return "English"


# Map fasttext ISO codes to Qwen language names
_LANG_MAP = {
    "zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean",
    "fr": "French", "de": "German", "es": "Spanish", "it": "Italian",
    "pt": "Portuguese", "ru": "Russian",
}


def detect_language(text: str) -> str:
    """Detect language using fasttext if available, falling back to Unicode heuristic."""
    detector = _get_langdetect()
    if detector:
        try:
            result = detector(text, low_memory=False)
            lang = result.get("lang", "en")
            return _LANG_MAP.get(lang, "English")
        except Exception:
            pass
    return _detect_language_unicode(text)


def _adaptive_max_tokens(text: str) -> int:
    """Scale token budget with text length to avoid over-allocating KV cache."""
    cjk_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff' or '\uac00' <= c <= '\ud7af')
    if cjk_chars > len(text) * 0.3:
        return max(128, min(2048, len(text) * 3))
    return max(128, min(2048, len(text.split()) * 8))


def _trim_silence(audio: np.ndarray, sample_rate: int = 24000, threshold_db: float = -40.0) -> np.ndarray:
    """Trim leading and trailing silence from audio array."""
    if not VAD_TRIM:
        return audio
    threshold = 10 ** (threshold_db / 20.0)
    non_silent = np.abs(audio) > threshold
    if not np.any(non_silent):
        return audio  # all silence, return as-is
    start = np.argmax(non_silent)
    end = len(non_silent) - np.argmax(non_silent[::-1])
    # Add small padding (50ms) to avoid cutting speech
    pad = int(0.05 * sample_rate)
    start = max(0, start - pad)
    end = min(len(audio), end + pad)
    return audio[start:end]


def _expand_currency(amount: str, unit: str) -> str:
    """Expand currency amount to words."""
    parts = amount.split('.')
    result = f"{parts[0]} {unit}"
    if len(parts) > 1 and parts[1] != '00':
        result += f" and {parts[1]} cents"
    return result


def _normalize_text(text: str) -> str:
    """Normalize text for TTS: expand numbers, currency, abbreviations."""
    if not TEXT_NORMALIZE:
        return text
    # Currency
    text = re.sub(r'\$(\d+(?:\.\d{2})?)', lambda m: _expand_currency(m.group(1), 'dollars'), text)
    text = re.sub(r'€(\d+)', lambda m: f"{m.group(1)} euros", text)
    text = re.sub(r'£(\d+)', lambda m: f"{m.group(1)} pounds", text)
    # Common abbreviations
    abbrevs = {'Dr.': 'Doctor', 'Mr.': 'Mister', 'Mrs.': 'Missus', 'Prof.': 'Professor',
               'Jr.': 'Junior', 'Sr.': 'Senior', 'St.': 'Saint', 'Ave.': 'Avenue',
               'Blvd.': 'Boulevard', 'Dept.': 'Department', 'Est.': 'Established'}
    for abbr, expansion in abbrevs.items():
        text = text.replace(abbr, expansion)
    # Large numbers with commas: 1,000,000 -> 1000000
    while re.search(r'(\d),(\d{3})', text):
        text = re.sub(r'(\d),(\d{3})', r'\1\2', text)
    return text


def _adjust_speed(audio_data: np.ndarray, sample_rate: int, speed: float) -> np.ndarray:
    """Adjust audio playback speed. Uses pyrubberband (pitch-preserving) if available,
    falling back to scipy resampling."""
    if speed == 1.0:
        return audio_data
    if _pyrubberband is not None:
        return _pyrubberband.time_stretch(audio_data, sample_rate, speed)
    new_length = int(len(audio_data) / speed)
    if new_length > 0:
        return scipy_signal.resample(audio_data, new_length)
    return audio_data


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, aware of common abbreviations and CJK punctuation."""
    pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!|\u3002|\uff01|\uff1f)\s+'
    sentences = re.split(pattern, text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _do_synthesize(text, language, speaker, gen_kwargs, instruct=None):
    """Run TTS inference. No per-request GC — let CUDA reuse cached allocations."""
    with torch.inference_mode():
        wavs, sr = model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct,
            **gen_kwargs,
        )
    return wavs, sr


def _get_cached_voice_prompt(audio_bytes: bytes, ref_text: str | None):
    """Return a cached voice clone prompt, or compute and cache it.

    Uses model.create_voice_clone_prompt() to pre-compute the speaker embedding
    from reference audio. The returned prompt object can be reused across
    generate_voice_clone() calls, skipping the encoder pass entirely.
    """
    global _voice_cache_hits

    cache_key = hashlib.sha256(audio_bytes).hexdigest()

    if VOICE_CACHE_MAX > 0 and cache_key in _voice_prompt_cache:
        _voice_cache_hits += 1
        _voice_prompt_cache.move_to_end(cache_key)
        return _voice_prompt_cache[cache_key]

    # Decode audio to pass to create_voice_clone_prompt
    ref_audio_data, ref_sr = sf.read(io.BytesIO(audio_bytes))
    if len(ref_audio_data.shape) > 1:
        ref_audio_data = ref_audio_data.mean(axis=1)

    # TODO: verify API with: docker compose run --rm qwen3-tts python3 -c \
    #   "from qwen_tts import Qwen3TTSModel; import inspect; \
    #    print(inspect.signature(Qwen3TTSModel.create_voice_clone_prompt))"
    prompt = model.create_voice_clone_prompt(
        ref_audio=(ref_audio_data, ref_sr),
        ref_text=ref_text,
    )

    if VOICE_CACHE_MAX > 0:
        _voice_prompt_cache[cache_key] = prompt
        while len(_voice_prompt_cache) > VOICE_CACHE_MAX:
            _voice_prompt_cache.popitem(last=False)

    return prompt


def _do_voice_clone(text, language, ref_prompt, gen_kwargs):
    """Run voice clone inference using a pre-computed voice prompt.

    No per-request GC — let CUDA reuse cached allocations.
    """
    with torch.inference_mode():
        # TODO: verify kwarg name with: docker compose run --rm qwen3-tts python3 -c \
        #   "from qwen_tts import Qwen3TTSModel; import inspect; \
        #    print(inspect.signature(Qwen3TTSModel.generate_voice_clone))"
        wavs, sr = model.generate_voice_clone(
            text=text,
            language=language,
            ref_prompt=ref_prompt,
            **gen_kwargs,
        )
    return wavs, sr


async def _encode_audio_async(audio_data, sample_rate, output_format):
    """Run audio encoding in the CPU thread pool, overlapping with GPU work."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _encode_executor,
        lambda: convert_audio_format(audio_data, sample_rate, output_format)
    )


@app.post("/v1/audio/speech")
async def synthesize_speech(request: TTSRequest):
    """OpenAI-compatible TTS endpoint using CustomVoice model."""
    global _queue_depth

    request_id = str(uuid.uuid4())[:8]
    t_start = time.perf_counter()

    # Early rejection when queue is full
    if MAX_QUEUE_DEPTH > 0 and _queue_depth >= MAX_QUEUE_DEPTH:
        raise HTTPException(
            status_code=503,
            detail=f"Server busy: {_queue_depth} requests queued. Try again later.",
            headers={"Retry-After": "5"},
        )

    if not request.input or not request.input.strip():
        raise HTTPException(status_code=400, detail="Input text is required")

    speaker = resolve_voice(request.voice)
    text = request.input.strip()
    _queue_depth += 1

    # Fast path: return cached audio without touching the GPU
    cache_key = _audio_cache_key(
        text, speaker, request.speed, request.response_format, request.language or "", request.instruct or ""
    )
    cached = _get_audio_cache(cache_key)
    if cached is not None:
        return Response(
            content=cached[0],
            media_type=cached[1],
            headers={
                "Content-Disposition": f'attachment; filename="speech.{request.response_format}"'
            },
        )

    await _ensure_model_loaded()

    try:
        language = request.language or detect_language(request.input)
        text = _normalize_text(text)
        gen_kwargs = {"max_new_tokens": _adaptive_max_tokens(text)}

        t_queue = time.perf_counter()
        loop = asyncio.get_running_loop()
        async with _infer_semaphore:
            t_queue_done = time.perf_counter()
            wavs, sr = await asyncio.wait_for(
                loop.run_in_executor(
                    _infer_executor,
                    lambda: _do_synthesize(text, language, speaker, gen_kwargs, instruct=request.instruct)
                ),
                timeout=REQUEST_TIMEOUT
            )
        t_infer_done = time.perf_counter()

        audio_data = np.array(wavs[0], dtype=np.float32, copy=True)
        if audio_data.ndim > 1:
            audio_data = audio_data.squeeze()

        audio_data = _trim_silence(audio_data, sr)

        audio_data = _adjust_speed(audio_data, sr, request.speed)

        audio_bytes, content_type = await _encode_audio_async(
            audio_data, sr, request.response_format
        )
        t_encode_done = time.perf_counter()

        logger.info(
            "synthesis_complete",
            extra={"extra_fields": {
                "request_id": request_id,
                "endpoint": "/v1/audio/speech",
                "voice": speaker,
                "language": language,
                "chars": len(text),
                "format": request.response_format,
                "queue_ms": round((t_queue_done - t_queue) * 1000),
                "infer_ms": round((t_infer_done - t_queue_done) * 1000),
                "encode_ms": round((t_encode_done - t_infer_done) * 1000),
                "total_ms": round((t_encode_done - t_start) * 1000),
            }},
        )

        _set_audio_cache(cache_key, audio_bytes, content_type)

        if _prometheus_available:
            tts_requests_total.labels(voice=speaker, format=request.response_format).inc()
            tts_inference_duration.observe(t_infer_done - t_queue_done)

        return Response(
            content=audio_bytes,
            media_type=content_type,
            headers={
                "Content-Disposition": f'attachment; filename="speech.{request.response_format}"'
            },
        )

    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Synthesis timed out")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")
    finally:
        _queue_depth -= 1


@app.post("/v1/audio/speech/stream")
async def synthesize_speech_stream(request: TTSRequest):
    """Sentence-chunked SSE streaming TTS endpoint."""
    global _last_used
    await _ensure_model_loaded()

    if not request.input or not request.input.strip():
        raise HTTPException(status_code=400, detail="Input text is required")

    speaker = resolve_voice(request.voice)
    language = request.language or detect_language(request.input)
    text = request.input.strip()
    sentences = _split_sentences(text)

    if not sentences:
        raise HTTPException(status_code=400, detail="No sentences found in input")

    async def generate():
        for sentence in sentences:
            try:
                gen_kwargs = {"max_new_tokens": _adaptive_max_tokens(sentence)}
                loop = asyncio.get_running_loop()
                async with _infer_semaphore:
                    wavs, sr = await asyncio.wait_for(
                        loop.run_in_executor(
                            _infer_executor,
                            lambda s=sentence: _do_synthesize(
                                s, language, speaker, gen_kwargs,
                                instruct=request.instruct,
                            )
                        ),
                        timeout=REQUEST_TIMEOUT,
                    )

                audio_data = np.array(wavs[0], dtype=np.float32, copy=True)
                if audio_data.ndim > 1:
                    audio_data = audio_data.squeeze()

                if request.speed != 1.0:
                    new_length = int(len(audio_data) / request.speed)
                    if new_length > 0:
                        audio_data = scipy_signal.resample(audio_data, new_length)

                audio_int16 = (audio_data * 32767).astype(np.int16)
                pcm_bytes = audio_int16.tobytes()
                yield f"data: {base64.b64encode(pcm_bytes).decode()}\n\n"

                _last_used = time.time()

            except asyncio.TimeoutError:
                yield "data: [ERROR] Synthesis timed out\n\n"
                return
            except Exception as e:
                yield f"data: [ERROR] {str(e)}\n\n"
                return

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
        },
    )


@app.post("/v1/audio/speech/clone")
async def clone_voice(
    file: UploadFile = File(...),
    input: str = Form(...),
    ref_text: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    response_format: str = Form("wav"),
):
    """Voice cloning endpoint - requires a reference audio file."""
    global _queue_depth

    t_start = time.perf_counter()

    if MAX_QUEUE_DEPTH > 0 and _queue_depth >= MAX_QUEUE_DEPTH:
        raise HTTPException(
            status_code=503,
            detail=f"Server busy: {_queue_depth} requests queued. Try again later.",
            headers={"Retry-After": "5"},
        )

    await _ensure_model_loaded()

    if not input or not input.strip():
        raise HTTPException(status_code=400, detail="Input text is required")

    _queue_depth += 1
    try:
        # Read reference audio and compute/cache speaker embedding
        audio_bytes = await file.read()
        text = input.strip()
        language = language or detect_language(text)
        text = _normalize_text(text)
        gen_kwargs = {"max_new_tokens": _adaptive_max_tokens(text)}

        ref_prompt = _get_cached_voice_prompt(
            audio_bytes, ref_text.strip() if ref_text else None
        )

        t_queue = time.perf_counter()
        loop = asyncio.get_running_loop()
        async with _infer_semaphore:
            t_infer_start = time.perf_counter()
            wavs, sr = await asyncio.wait_for(
                loop.run_in_executor(
                    _infer_executor,
                    lambda: _do_voice_clone(
                        text,
                        language,
                        ref_prompt,
                        gen_kwargs,
                    )
                ),
                timeout=REQUEST_TIMEOUT
            )
        t_infer_end = time.perf_counter()

        audio_data = np.array(wavs[0], dtype=np.float32, copy=True)
        if audio_data.ndim > 1:
            audio_data = audio_data.squeeze()

        audio_data = _trim_silence(audio_data, sr)

        audio_bytes_out, content_type = await _encode_audio_async(
            audio_data, sr, response_format
        )
        t_end = time.perf_counter()

        logger.info(
            "request_complete",
            extra={
                "endpoint": "/v1/audio/speech/clone",
                "queue_ms": round((t_infer_start - t_queue) * 1000, 1),
                "inference_ms": round((t_infer_end - t_infer_start) * 1000, 1),
                "encode_ms": round((t_end - t_infer_end) * 1000, 1),
                "total_ms": round((t_end - t_start) * 1000, 1),
                "chars": len(text),
                "format": response_format,
                "language": language,
            },
        )

        return Response(
            content=audio_bytes_out,
            media_type=content_type,
            headers={
                "Content-Disposition": f'attachment; filename="speech.{response_format}"'
            },
        )

    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Voice clone timed out")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice clone failed: {str(e)}")
    finally:
        _queue_depth -= 1


@app.post("/v1/audio/speech/stream/pcm")
async def synthesize_speech_stream_pcm(request: TTSRequest):
    """Raw PCM streaming TTS endpoint — no SSE framing, no base64.

    Splits text into sentences and streams each as raw int16 PCM bytes.
    Headers report the audio format: 24000 Hz, 16-bit, mono.
    """
    global _last_used
    await _ensure_model_loaded()

    if not request.input or not request.input.strip():
        raise HTTPException(status_code=400, detail="Input text is required")

    speaker = resolve_voice(request.voice)
    language = request.language or detect_language(request.input)
    text = request.input.strip()
    sentences = _split_sentences(text)
    if not sentences:
        raise HTTPException(status_code=400, detail="No sentences found in input")

    async def pcm_generator():
        global _last_used
        for sentence in sentences:
            gen_kwargs = {"max_new_tokens": _adaptive_max_tokens(sentence)}
            loop = asyncio.get_running_loop()
            try:
                async with _infer_semaphore:
                    wavs, sr = await asyncio.wait_for(
                        loop.run_in_executor(
                            _infer_executor,
                            lambda s=sentence: _do_synthesize(
                                s, language, speaker, gen_kwargs
                            ),
                        ),
                        timeout=REQUEST_TIMEOUT,
                    )
                _last_used = time.time()
                audio_data = np.array(wavs[0], dtype=np.float32, copy=True)
                if audio_data.ndim > 1:
                    audio_data = audio_data.squeeze()
                if request.speed != 1.0:
                    new_length = int(len(audio_data) / request.speed)
                    if new_length > 0:
                        audio_data = scipy_signal.resample(audio_data, new_length)
                pcm_data = np.clip(audio_data, -1.0, 1.0)
                pcm_bytes = (pcm_data * 32767).astype(np.int16).tobytes()
                yield pcm_bytes
            except asyncio.TimeoutError:
                break
            except Exception:
                break

    return StreamingResponse(
        pcm_generator(),
        media_type="application/octet-stream",
        headers={
            "X-PCM-Sample-Rate": "24000",
            "X-PCM-Bit-Depth": "16",
            "X-PCM-Channels": "1",
            "Content-Disposition": 'attachment; filename="speech.pcm"',
        },
    )


@app.websocket("/v1/audio/speech/ws")
async def ws_synthesize(websocket: WebSocket):
    """WebSocket streaming endpoint. Send JSON, receive binary PCM per sentence."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            text = (data.get("input") or "").strip()
            if not text:
                await websocket.send_json({"event": "error", "detail": "input is required"})
                continue

            voice = resolve_voice(data.get("voice"))
            language = data.get("language") or detect_language(text)
            speed = float(data.get("speed", 1.0))

            await _ensure_model_loaded()

            sentences = _split_sentences(text)
            if not sentences:
                sentences = [text]

            for sentence in sentences:
                gen_kwargs = {"max_new_tokens": _adaptive_max_tokens(sentence)}
                loop = asyncio.get_running_loop()
                async with _infer_semaphore:
                    wavs, sr = await asyncio.wait_for(
                        loop.run_in_executor(
                            _infer_executor,
                            lambda s=sentence: _do_synthesize(s, language, voice, gen_kwargs)
                        ),
                        timeout=REQUEST_TIMEOUT
                    )

                audio_data = np.array(wavs[0], dtype=np.float32, copy=True)
                if audio_data.ndim > 1:
                    audio_data = audio_data.squeeze()

                if speed != 1.0:
                    new_length = int(len(audio_data) / speed)
                    if new_length > 0:
                        audio_data = scipy_signal.resample(audio_data, new_length)

                pcm = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
                await websocket.send_bytes(pcm)
                global _last_used
                _last_used = time.time()

            await websocket.send_json({"event": "done"})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"event": "error", "detail": str(e)})
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
