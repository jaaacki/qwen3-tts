#!/bin/bash
# GPU tuning — best-effort, failures are non-fatal (may need --privileged)

# Keep GPU driver initialized between workloads (eliminates 200-500ms cold-start)
nvidia-smi -pm 1 2>/dev/null || echo "Warning: could not set GPU persistence mode"

# Lock GPU clocks to max boost for consistent latency
if ! nvidia-smi --lock-gpu-clocks=0,9999 2>/dev/null; then
  _MAX_CLOCK=$(nvidia-smi --query-gpu=clocks.max.gr --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [ -n "$_MAX_CLOCK" ]; then
    nvidia-smi -lgc 0,"$_MAX_CLOCK" 2>/dev/null || \
      echo "Warning: could not lock GPU clocks (may need privileged or specific GPU support)"
  else
    echo "Warning: could not lock GPU clocks (may need privileged or specific GPU support)"
  fi
fi

# Enable transparent huge pages if running with sufficient privileges
# THP reduces TLB pressure for large model weights (~2.4GB = thousands of 4KB pages -> fewer 2MB pages)
echo madvise > /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null && \
    echo "THP: enabled (madvise)" || true
echo defer+madvise > /sys/kernel/mm/transparent_hugepage/defrag 2>/dev/null && \
    echo "THP defrag: defer+madvise" || true

# Check jemalloc
if [ -n "$LD_PRELOAD" ] && [ -f "$LD_PRELOAD" ]; then
    echo "jemalloc loaded: $LD_PRELOAD"
fi

# Switch to Unix domain socket mode when UNIX_SOCKET_PATH is set (bypasses TCP stack)
if [ -n "$UNIX_SOCKET_PATH" ]; then
    exec uvicorn server:app --uds "$UNIX_SOCKET_PATH" \
        --loop uvloop --http httptools --no-access-log --timeout-keep-alive 65
fi

# Append TLS args for HTTP/2 support when SSL certs are provided
TLS_ARGS=""
if [ -n "$SSL_KEYFILE" ]; then
    TLS_ARGS="$TLS_ARGS --ssl-keyfile $SSL_KEYFILE"
fi
if [ -n "$SSL_CERTFILE" ]; then
    TLS_ARGS="$TLS_ARGS --ssl-certfile $SSL_CERTFILE"
fi

exec "$@" $TLS_ARGS
