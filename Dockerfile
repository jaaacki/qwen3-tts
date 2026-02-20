# Stage 1: builder — install Python deps with build tools
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime AS builder

WORKDIR /build

# git is needed by qwen-tts install
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: runtime — lean image with only what's needed
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

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
    libsndfile1 ffmpeg sox rubberband-cli libjemalloc2 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
COPY server.py /app/server.py
COPY docker-entrypoint.sh /app/docker-entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["uvicorn", "server:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--loop", "uvloop", \
     "--http", "httptools", \
     "--no-access-log", \
     "--timeout-keep-alive", "65"]
