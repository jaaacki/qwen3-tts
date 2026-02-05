from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional
import torch
import soundfile as sf
import io
import os
import numpy as np

app = FastAPI(title="Qwen3-TTS API")

model = None
loaded_model_id = None

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


class VoiceCloneRequest(BaseModel):
    input: str
    language: Optional[str] = None
    ref_text: Optional[str] = None
    response_format: str = "wav"


@app.on_event("startup")
async def load_model():
    global model, loaded_model_id
    from qwen_tts import Qwen3TTSModel

    model_id = os.getenv("MODEL_ID", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    loaded_model_id = model_id

    print(f"Loading {model_id}...")
    model = Qwen3TTSModel.from_pretrained(
        model_id,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    print(f"Model loaded: {model_id}")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_id": loaded_model_id,
        "cuda": torch.cuda.is_available(),
        "voices": list(VOICE_MAP.keys()),
    }


def convert_audio_format(audio_data: np.ndarray, sample_rate: int, output_format: str) -> tuple[bytes, str]:
    """Convert audio data to the requested format."""
    buffer = io.BytesIO()

    if output_format in ("wav", "wave"):
        sf.write(buffer, audio_data, sample_rate, format="WAV")
        content_type = "audio/wav"
    elif output_format == "mp3":
        try:
            from pydub import AudioSegment
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, audio_data, sample_rate, format="WAV")
            wav_buffer.seek(0)
            audio_segment = AudioSegment.from_wav(wav_buffer)
            audio_segment.export(buffer, format="mp3")
            content_type = "audio/mpeg"
        except ImportError:
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


@app.post("/v1/audio/speech")
async def synthesize_speech(request: TTSRequest):
    """OpenAI-compatible TTS endpoint using CustomVoice model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.input or not request.input.strip():
        raise HTTPException(status_code=400, detail="Input text is required")

    try:
        speaker = resolve_voice(request.voice)
        language = request.language or detect_language(request.input)

        gen_kwargs = {"max_new_tokens": 2048}

        wavs, sr = model.generate_custom_voice(
            text=request.input.strip(),
            language=language,
            speaker=speaker,
            **gen_kwargs,
        )

        audio_data = np.array(wavs[0], dtype=np.float32)
        if len(audio_data.shape) > 1:
            audio_data = audio_data.squeeze()

        # Speed adjustment via resampling
        if request.speed != 1.0:
            import scipy.signal as signal
            new_length = int(len(audio_data) / request.speed)
            if new_length > 0:
                audio_data = signal.resample(audio_data, new_length)

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
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

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

        wavs, sr = model.generate_voice_clone(
            text=input.strip(),
            language=language,
            ref_audio=(ref_audio_data, ref_sr),
            ref_text=ref_text.strip() if ref_text else None,
            **gen_kwargs,
        )

        audio_data = np.array(wavs[0], dtype=np.float32)
        if len(audio_data.shape) > 1:
            audio_data = audio_data.squeeze()

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

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice clone failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
