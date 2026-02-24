"""Audio utilities for E2E tests.

For TTS tests we mostly need to:
1. Generate reference WAV files for voice clone tests.
2. Validate that server output is valid audio bytes.
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf


# ---------------------------------------------------------------------------
# Reference audio generation (used for /clone endpoint tests)
# ---------------------------------------------------------------------------

def generate_sine_wav(
    duration_s: float = 3.0,
    frequency: float = 220.0,
    sample_rate: int = 24000,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Generate a sine-wave signal at the given frequency and sample rate."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    signal = amplitude * np.sin(2 * np.pi * frequency * t)
    return signal.astype(np.float32)


def generate_ref_audio(
    output_dir: Path,
    filename: str = "ref_3s.wav",
    duration_s: float = 3.0,
    sample_rate: int = 24000,
    force: bool = False,
) -> Path:
    """Generate a simple sine-wave WAV file for voice-clone reference audio.

    Returns the path to the generated file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    if not path.exists() or force:
        audio = generate_sine_wav(duration_s=duration_s, sample_rate=sample_rate)
        sf.write(str(path), audio, sample_rate, subtype="PCM_16")
    return path


# ---------------------------------------------------------------------------
# WAV validation helpers
# ---------------------------------------------------------------------------

def parse_wav_header(data: bytes) -> Optional[dict]:
    """Parse a RIFF/WAVE header.  Returns None if data is not a WAV file."""
    if len(data) < 44:
        return None
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return None
    try:
        # fmt chunk starts at byte 12
        channels   = struct.unpack_from("<H", data, 22)[0]
        sample_rate = struct.unpack_from("<I", data, 24)[0]
        bit_depth  = struct.unpack_from("<H", data, 34)[0]
        return {
            "channels":    channels,
            "sample_rate": sample_rate,
            "bit_depth":   bit_depth,
        }
    except struct.error:
        return None


def is_valid_wav(data: bytes) -> bool:
    """Return True if `data` begins with a valid RIFF/WAVE header."""
    return parse_wav_header(data) is not None


def audio_duration_from_wav(data: bytes) -> Optional[float]:
    """Estimate duration in seconds from raw WAV bytes.  Returns None on error."""
    header = parse_wav_header(data)
    if header is None:
        return None
    # Walk RIFF chunks to find the "data" chunk (handles non-standard headers
    # where extra chunks like LIST appear between fmt and data).
    try:
        offset = 12  # skip RIFF header (4 ID + 4 size + 4 WAVE)
        data_size = None
        while offset + 8 <= len(data):
            chunk_id = data[offset:offset + 4]
            chunk_size = struct.unpack_from("<I", data, offset + 4)[0]
            if chunk_id == b"data":
                data_size = chunk_size
                break
            offset += 8 + chunk_size
        if data_size is None:
            return None
        bytes_per_sample = header["bit_depth"] // 8
        if bytes_per_sample == 0 or header["channels"] == 0:
            return None
        total_samples = data_size / (bytes_per_sample * header["channels"])
        return total_samples / header["sample_rate"]
    except (struct.error, ZeroDivisionError):
        return None
