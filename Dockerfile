# Stage 1: builder — install Python deps with build tools
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime AS builder

WORKDIR /build

# git is needed by qwen-tts install
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Optional quantization
RUN pip install --no-cache-dir --prefix=/install "bitsandbytes>=0.43.0" || true

# Stage 2: runtime — lean image with only what's needed
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

# Python runtime tuning
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# CUDA memory allocator tuning — reduce fragmentation for shared GPU
ENV PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
ENV TOKENIZERS_PARALLELISM=false

# Limit CPU thread spawning — GPU does the heavy work, excess threads just contend
ENV OMP_NUM_THREADS=2
ENV MKL_NUM_THREADS=2

# jemalloc replaces ptmalloc2 to reduce RSS bloat from arena fragmentation
# Path assumes x86_64 Linux; for aarch64 use /usr/lib/aarch64-linux-gnu/libjemalloc.so.2
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
ENV MALLOC_CONF=background_thread:true,dirty_decay_ms:1000,muzzy_decay_ms:0

# Install runtime-only system dependencies (no git, no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg sox rubberband-cli libjemalloc2 libopus-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder into conda's site-packages
COPY --from=builder /install/lib/python3.11/site-packages/ /opt/conda/lib/python3.11/site-packages/
COPY --from=builder /install/bin/ /opt/conda/bin/

# Copy application
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
COPY server.py /app/server.py
COPY gateway.py /app/gateway.py
COPY worker.py /app/worker.py
COPY voices/ /app/voices/

EXPOSE 8000

CMD if [ "${GATEWAY_MODE:-false}" = "true" ]; then \
      exec uvicorn gateway:app --host 0.0.0.0 --port 8000 --loop uvloop --http httptools --no-access-log; \
    else \
      exec /app/docker-entrypoint.sh uvicorn server:app --host 0.0.0.0 --port 8000 --loop uvloop --http httptools --no-access-log --timeout-keep-alive 65; \
    fi
