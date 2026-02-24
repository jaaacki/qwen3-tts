"""Worker subprocess — runs the full TTS server on an internal port.

Started by gateway.py. Preloads the model eagerly. Gateway manages idle shutdown.
"""
import os
import uvicorn

# Worker always preloads — it exists to serve
os.environ.setdefault("PRELOAD_MODEL", "true")
os.environ.setdefault("IDLE_TIMEOUT", "0")  # Gateway manages idle, not the worker

from server import app  # noqa: E402

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=port,
        loop="uvloop",
        http="httptools",
        no_access_log=True,
    )
