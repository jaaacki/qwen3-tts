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

exec "$@"
