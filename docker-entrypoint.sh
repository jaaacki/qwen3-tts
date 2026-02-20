#!/bin/bash
# GPU tuning — best-effort, failures are non-fatal (may need --privileged)

# Keep GPU driver initialized between workloads (eliminates 200-500ms cold-start)
nvidia-smi -pm 1 2>/dev/null || echo "Warning: could not set GPU persistence mode"

exec "$@"
