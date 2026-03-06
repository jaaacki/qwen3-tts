# Bring Logging, Env Config & Error Handling to Standard

Date: 2026-03-01

## Problem

Audit against operational standards found gaps in three areas:

1. **Logging** — missing `service` field, wrong timestamp format, outputs to stderr instead of stdout, non-standard level names, `request_id` missing on 4 of 5 endpoints
2. **Env config** — no `.gitignore`, no `.env.example`, no startup validation of env vars
3. **Error handling** — using FastAPI default `{"detail": "..."}` instead of standard error shape `{code, message, context, statusCode}`

## Design

### 1. Logging (`_json_sink` + endpoints in server.py)

- Add `"service": "qwen3-tts"` to every JSON log entry
- Timestamp: `"%Y-%m-%dT%H:%M:%S.%f%z"` (ISO 8601 with TZ + fractional seconds)
- Output to `sys.stdout` instead of `sys.stderr`
- Level remap: CRITICAL→fatal, ERROR→error, WARNING→warn, INFO→info, DEBUG→debug, TRACE→trace, SUCCESS→info
- Add `request_id` (uuid4 prefix) to `/stream`, `/stream/pcm`, `/clone`, `/ws` endpoints

### 2. Env Config

- `.gitignore` with `.env`, `models/`, `__pycache__/`, `*.pyc`
- `.env.example` committed, documenting every variable with comments
- `_validate_env()` at startup: validate QUANTIZE, LOG_FORMAT, LOG_LEVEL, numeric vars; fail fast

### 3. Error Handling

- `ErrorResponse` Pydantic model: `{code, message, context, statusCode}`
- `_error_response()` helper + FastAPI exception handler
- Machine-readable codes: QUEUE_FULL, EMPTY_INPUT, UNKNOWN_VOICE, SYNTHESIS_TIMEOUT, SYNTHESIS_FAILED, INVALID_AUDIO, CLONE_FAILED, NO_SENTENCES

### Team

- 3 Builders in worktrees (one per area), Architect coordinates
- GitHub issues filed first, branches from main
