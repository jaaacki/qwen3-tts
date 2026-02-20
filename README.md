# Qwen3-TTS

OpenAI-compatible Text-to-Speech API server powered by [Qwen3-TTS-0.6B](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice), running in Docker with NVIDIA GPU acceleration.

## Features

- **OpenAI-compatible API** — drop-in replacement for `/v1/audio/speech`
- **9 built-in voices** — vivian, serena, uncle_fu, dylan, eric, ryan, aiden, ono_anna, sohee
- **OpenAI voice aliases** — alloy, echo, fable, onyx, nova, shimmer
- **Voice cloning** — generate speech from a reference audio sample
- **Multi-language** — English, Chinese, Japanese, Korean, German, Italian, Portuguese, Spanish, French, Russian, Beijing dialect, Sichuan dialect
- **Multiple output formats** — WAV, MP3, FLAC, OGG
- **Speed control** — adjustable playback speed via resampling

## Requirements

- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- NVIDIA GPU with CUDA 12.4+ support

## Quick Start

```bash
# Build and start the service
docker compose up -d --build

# Check health
curl http://localhost:8101/health
```

The model (~2.4 GB) downloads automatically on first startup.

## API Endpoints

### `POST /v1/audio/speech`

Generate speech from text.

```bash
curl -X POST http://localhost:8101/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, world!", "voice": "vivian"}' \
  -o speech.wav
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | string | *required* | Text to synthesize |
| `voice` | string | `vivian` | Voice name or OpenAI alias |
| `response_format` | string | `wav` | Output format: `wav`, `mp3`, `flac`, `ogg` |
| `speed` | float | `1.0` | Playback speed multiplier |
| `language` | string | *auto-detect* | Language override |

### `POST /v1/audio/speech/clone`

Generate speech using a cloned voice from a reference audio file.

```bash
curl -X POST http://localhost:8101/v1/audio/speech/clone \
  -F "file=@reference.wav" \
  -F "input=Hello, this is my cloned voice." \
  -F "ref_text=Original text spoken in the reference audio." \
  -o cloned_speech.wav
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | file | *required* | Reference audio file |
| `input` | string | *required* | Text to synthesize |
| `ref_text` | string | *optional* | Transcript of the reference audio |
| `language` | string | *auto-detect* | Language override |
| `response_format` | string | `wav` | Output format |

### `GET /health`

Returns service status, model info, CUDA availability, and available voices.

## Configuration

Environment variables in `compose.yaml`:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | Hugging Face model ID |
| `IDLE_TIMEOUT` | `120` | Seconds of inactivity before unloading model from GPU (0 = disabled) |
| `REQUEST_TIMEOUT` | `300` | Maximum seconds per inference request |
| `TORCH_COMPILE` | `true` | Enable torch.compile optimization (set to false to disable) |

The model cache is persisted to `./models` via volume mount.

### GPU Memory Management

The model loads **on-demand** with the first request and automatically **unloads after idle timeout** to free VRAM for other services. This is ideal for shared GPU environments.

## Testing

```bash
# Run the full test suite
python test_tts.py

# Quick smoke test
bash test_quick.sh
```

## License

Uses the [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice) model — see its license for usage terms.
