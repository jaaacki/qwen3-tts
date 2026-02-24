"""Gateway proxy — lightweight FastAPI that manages the inference worker subprocess.

GATEWAY_MODE=true: starts with ~30 MB RAM, spawns worker on first request.
The worker loads the model (~1 GB VRAM) and serves requests on WORKER_PORT.
"""
import asyncio
import os
import subprocess
import sys
import time
from contextlib import asynccontextmanager

import aiohttp
from fastapi import FastAPI, Request, Response

WORKER_HOST = os.getenv("WORKER_HOST", "127.0.0.1")
WORKER_PORT = int(os.getenv("WORKER_PORT", "8001"))
IDLE_TIMEOUT = int(os.getenv("IDLE_TIMEOUT", "120"))
WORKER_URL = f"http://{WORKER_HOST}:{WORKER_PORT}"

_worker_process: subprocess.Popen | None = None
_worker_lock = asyncio.Lock()
_last_used: float = 0.0
_session: aiohttp.ClientSession | None = None


async def _ensure_worker():
    global _worker_process, _last_used
    if _worker_process is not None and _worker_process.poll() is None:
        _last_used = time.time()
        return
    async with _worker_lock:
        if _worker_process is not None and _worker_process.poll() is None:
            _last_used = time.time()
            return
        print("Gateway: starting worker subprocess...")
        _worker_process = subprocess.Popen(
            [sys.executable, "worker.py"],
            env={**os.environ, "PORT": str(WORKER_PORT)},
        )
        # Wait for worker to become ready
        for _ in range(60):
            await asyncio.sleep(0.5)
            try:
                async with _session.get(f"{WORKER_URL}/health") as resp:
                    if resp.status == 200:
                        print(f"Gateway: worker ready (pid={_worker_process.pid})")
                        _last_used = time.time()
                        return
            except Exception:
                pass
        _worker_process.kill()
        _worker_process = None
        raise RuntimeError("Worker failed to start within 30s")


async def _check_idle():
    global _worker_process, _last_used
    if IDLE_TIMEOUT <= 0 or _worker_process is None:
        return
    if _worker_process.poll() is not None:
        _worker_process = None
        return
    if time.time() - _last_used > IDLE_TIMEOUT:
        print("Gateway: idle timeout — killing worker")
        _worker_process.kill()
        _worker_process = None


async def _idle_watchdog():
    while True:
        await asyncio.sleep(30)
        await _check_idle()


@asynccontextmanager
async def lifespan(app):
    global _session
    _session = aiohttp.ClientSession()
    asyncio.create_task(_idle_watchdog())
    print("Gateway started")
    yield
    if _worker_process is not None:
        _worker_process.kill()
    await _session.close()
    print("Gateway shutdown")


app = FastAPI(title="Qwen3-TTS Gateway", lifespan=lifespan)


async def _proxy(request: Request, path: str) -> Response:
    global _last_used
    await _ensure_worker()
    _last_used = time.time()
    url = f"{WORKER_URL}/{path}"
    body = await request.body()
    async with _session.request(
        method=request.method,
        url=url,
        headers={k: v for k, v in request.headers.items() if k.lower() != "host"},
        data=body,
        allow_redirects=False,
    ) as resp:
        content = await resp.read()
        return Response(
            content=content,
            status_code=resp.status,
            headers=dict(resp.headers),
            media_type=resp.content_type,
        )


@app.api_route("/{path:path}", methods=["GET", "POST", "DELETE"])
async def proxy_all(request: Request, path: str):
    return await _proxy(request, path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
