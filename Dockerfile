FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# CUDA memory allocator tuning — reduce fragmentation for shared GPU
ENV PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
ENV TOKENIZERS_PARALLELISM=false

# Install system dependencies (sox needed by qwen-tts audio pipeline)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git libsndfile1 ffmpeg sox \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
RUN pip install --no-cache-dir \
    accelerate \
    soundfile \
    scipy \
    fastapi \
    uvicorn \
    pydub \
    python-multipart \
    qwen-tts

COPY server.py /app/server.py

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
