from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional
import torch
import soundfile as sf
import io
import os
import gc
import asyncio
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import scipy.signal as scipy_signal

try:
    from pydub import AudioSegment as _PydubAudioSegment
except ImportError:
    _PydubAudioSegment = None

# Enable cudnn autotuner — finds fastest convolution algorithms for the GPU
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True   # 3x faster matmul on Ampere+ GPUs
    torch.backends.cudnn.allow_tf32 = True           # enable TF32 for cuDNN ops

app = FastAPI(title="Qwen3-TTS API")

model = None
loaded_model_id = None

# Single-thread executor for GPU inference — avoids default pool overhead
_infer_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts-infer")

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

# Track last request time
_last_used = 0.0

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


class VoiceCloneRequest(BaseModel):
    input: str
    language: Optional[str] = None
    ref_text: Optional[str] = None
    response_format: str = "wav"


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

    _last_used = time.time()
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


@app.on_event("startup")
async def startup():
    asyncio.create_task(_idle_watchdog())


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
        "voices": list(VOICE_MAP.keys()),
        **gpu_info,
    }


def convert_audio_format(audio_data: np.ndarray, sample_rate: int, output_format: str) -> tuple[bytes, str]:
    """Convert audio data to the requested format."""
    buffer = io.BytesIO()

    if output_format in ("wav", "wave"):
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


def detect_language(text: str) -> str:
    """Simple language detection based on character ranges."""
    for ch in text:
        if '\u4e00' <= ch <= '\u9fff':
            return "Chinese"
        if '\u3040' <= ch <= '\u309f' or '\u30a0' <= ch <= '\u30ff':
            return "Japanese"
        if '\uac00' <= ch <= '\ud7af':
            return "Korean"
    return "English"


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


def _do_voice_clone(text, language, ref_audio, ref_text, gen_kwargs):
    """Run voice clone inference. No per-request GC — let CUDA reuse cached allocations."""
    with torch.inference_mode():
        wavs, sr = model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
            **gen_kwargs,
        )
    return wavs, sr


@app.post("/v1/audio/speech")
async def synthesize_speech(request: TTSRequest):
    """OpenAI-compatible TTS endpoint using CustomVoice model."""
    await _ensure_model_loaded()

    if not request.input or not request.input.strip():
        raise HTTPException(status_code=400, detail="Input text is required")

    try:
        speaker = resolve_voice(request.voice)
        language = request.language or detect_language(request.input)
        gen_kwargs = {"max_new_tokens": 2048}
        text = request.input.strip()

        loop = asyncio.get_running_loop()
        async with _infer_semaphore:
            wavs, sr = await asyncio.wait_for(
                loop.run_in_executor(
                    _infer_executor,
                    lambda: _do_synthesize(text, language, speaker, gen_kwargs, instruct=request.instruct)
                ),
                timeout=REQUEST_TIMEOUT
            )

        audio_data = np.array(wavs[0], dtype=np.float32, copy=True)
        if audio_data.ndim > 1:
            audio_data = audio_data.squeeze()

        audio_data = _trim_silence(audio_data, sr)

        # Speed adjustment via resampling
        if request.speed != 1.0:
            new_length = int(len(audio_data) / request.speed)
            if new_length > 0:
                audio_data = scipy_signal.resample(audio_data, new_length)

        audio_bytes, content_type = convert_audio_format(
            audio_data, sr, request.response_format
        )

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


@app.post("/v1/audio/speech/clone")
async def clone_voice(
    file: UploadFile = File(...),
    input: str = Form(...),
    ref_text: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    response_format: str = Form("wav"),
):
    """Voice cloning endpoint - requires a reference audio file."""
    await _ensure_model_loaded()

    if not input or not input.strip():
        raise HTTPException(status_code=400, detail="Input text is required")

    try:
        # Read reference audio
        audio_bytes = await file.read()
        ref_audio_data, ref_sr = sf.read(io.BytesIO(audio_bytes))
        if len(ref_audio_data.shape) > 1:
            ref_audio_data = ref_audio_data.mean(axis=1)

        language = language or detect_language(input)
        gen_kwargs = {"max_new_tokens": 2048}
        text = input.strip()

        loop = asyncio.get_running_loop()
        async with _infer_semaphore:
            wavs, sr = await asyncio.wait_for(
                loop.run_in_executor(
                    _infer_executor,
                    lambda: _do_voice_clone(
                        text,
                        language,
                        (ref_audio_data, ref_sr),
                        ref_text.strip() if ref_text else None,
                        gen_kwargs,
                    )
                ),
                timeout=REQUEST_TIMEOUT
            )

        audio_data = np.array(wavs[0], dtype=np.float32, copy=True)
        if audio_data.ndim > 1:
            audio_data = audio_data.squeeze()

        audio_data = _trim_silence(audio_data, sr)

        audio_bytes_out, content_type = convert_audio_format(
            audio_data, sr, response_format
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
