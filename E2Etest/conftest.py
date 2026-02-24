"""Pytest configuration and shared fixtures for Qwen3-TTS E2E tests."""
from __future__ import annotations

import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Generator

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent))


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_BASE_URL = "http://localhost:8101"
DEFAULT_WS_URL   = "ws://localhost:8101/v1/audio/speech/ws"
HEALTH_TIMEOUT   = 30   # seconds to wait for server health check
MODEL_LOAD_TIMEOUT = 300  # seconds for first model load (2.4 GB download on cold)

# Short text used to warm the model — quick, produces audible output
WARMUP_TEXT = "Hello, this is a test."


# =============================================================================
# Per-test SLA budgets (seconds) — tests exceeding these are flagged ⚠️ SLOW
# =============================================================================

_DURATION_SLAS: dict[str, float] = {
    # Health / cache (should be instant)
    "test_health_returns_200":             2.0,
    "test_health_has_expected_fields":     2.0,
    "test_health_status_is_ok":            2.0,
    "test_health_lists_voices":            2.0,
    "test_health_gpu_info_when_available": 2.0,
    "test_health_cache_fields":            2.0,
    "test_clear_cache_returns_counts":     2.0,
    # Error handling (no GPU work)
    "test_empty_input_returns_400":        5.0,
    "test_missing_input_returns_4xx":      5.0,
    "test_queue_depth_in_health":          5.0,
    # Synthesis (first call may load model)
    "test_basic_synthesis_returns_audio":  60.0,
    "test_response_is_valid_wav":          60.0,
    "test_audio_has_nonzero_duration":     60.0,
    "test_synthesis_deterministic":        90.0,
    # Voices
    "test_native_voice_names":             60.0,
    "test_openai_voice_aliases":           60.0,
    "test_unknown_voice_passthrough":      60.0,
    # Formats
    "test_wav_format":                     60.0,
    "test_mp3_format":                     60.0,
    "test_flac_format":                    60.0,
    "test_ogg_format":                     60.0,
    "test_opus_format":                    60.0,
    # Streaming
    "test_sse_stream_returns_events":      90.0,
    "test_sse_stream_has_done_marker":     90.0,
    "test_sse_events_contain_audio":       90.0,
    "test_pcm_stream_returns_bytes":       90.0,
    "test_pcm_stream_headers":             90.0,
    # WebSocket
    "test_ws_synthesize_returns_pcm":      90.0,
    "test_ws_receives_done_event":         90.0,
    "test_ws_empty_input_returns_error":   30.0,
    "test_ws_multiple_requests":           120.0,
    # Clone
    "test_clone_returns_audio":            120.0,
    "test_clone_with_ref_text":            120.0,
    "test_clone_cache_hit":                180.0,
    # Performance
    "test_warm_inference_latency":         60.0,
    "test_health_latency":                 2.0,
    "test_audio_cache_hit_latency":        60.0,
    # Integration
    "test_audio_cache_populated":          90.0,
    "test_cache_cleared_on_request":       30.0,
    "test_speed_parameter_affects_duration": 120.0,
    "test_temperature_param_accepted":     60.0,
}
_DEFAULT_SLA = 60.0


def _sla_for(test_name: str) -> float:
    return _DURATION_SLAS.get(test_name, _DEFAULT_SLA)


# =============================================================================
# Markdown Report Generator
# =============================================================================

class MarkdownReportGenerator:
    """Collects test results and generates a markdown report."""

    def __init__(self):
        self.results: list[dict] = []
        self.session_start: float = 0.0
        self.session_end: float = 0.0

    def add_result(
        self,
        nodeid: str,
        outcome: str,
        duration: float,
        stdout: str = "",
        longrepr: str = "",
        skip_reason: str = "",
    ):
        self.results.append({
            "nodeid":      nodeid,
            "outcome":     outcome,
            "duration":    duration,
            "stdout":      stdout,
            "longrepr":    longrepr,
            "skip_reason": skip_reason,
        })

    def _fetch_server_info(self) -> dict:
        try:
            resp = httpx.get(
                f"{os.getenv('E2E_BASE_URL', DEFAULT_BASE_URL)}/health",
                timeout=5,
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return {}

    def _categorize(self, nodeid: str) -> str:
        fname = nodeid.split("::")[0].rsplit("/", 1)[-1]
        mapping = {
            "test_api_http":   "HTTP API",
            "test_voices":     "Voices",
            "test_formats":    "Formats",
            "test_streaming":  "Streaming",
            "test_websocket":  "WebSocket",
            "test_clone":      "Clone",
            "test_performance": "Performance",
            "test_integration": "Integration",
        }
        for key, label in mapping.items():
            if key in fname:
                return label
        return "Other"

    def _parse_synthesis_results(self) -> list[dict]:
        """Extract synthesis metadata printed by tests."""
        results = []
        for r in self.results:
            if not r["stdout"]:
                continue
            entry: dict = {"test": r["nodeid"].split("::")[-1], "duration": r["duration"]}
            for line in r["stdout"].splitlines():
                m = re.match(r"Voice:\s*(.+)", line)
                if m:
                    entry["voice"] = m.group(1).strip()
                m = re.match(r"Format:\s*(.+)", line)
                if m:
                    entry["format"] = m.group(1).strip()
                m = re.match(r"Audio bytes:\s*(\d+)", line)
                if m:
                    entry["bytes"] = int(m.group(1))
                m = re.match(r"Audio duration:\s*([\d.]+)s", line)
                if m:
                    entry["audio_duration"] = float(m.group(1))
            if any(k in entry for k in ("voice", "bytes", "audio_duration")):
                results.append(entry)
        return results

    def _parse_latency_metrics(self) -> list[dict]:
        """Extract latency numbers printed by performance tests."""
        metrics = []
        for r in self.results:
            if "test_performance" not in r["nodeid"] or not r["stdout"]:
                continue
            for line in r["stdout"].splitlines():
                m = re.search(r"([\w\s]+?):\s*([\d.]+)\s*s", line, re.IGNORECASE)
                if m:
                    metrics.append({
                        "test":   r["nodeid"].split("::")[-1],
                        "metric": m.group(1).strip(),
                        "value":  float(m.group(2)),
                    })
        return metrics

    def generate(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        report_path = output_dir / f"{ts}.md"

        info = self._fetch_server_info()
        total_duration = self.session_end - self.session_start

        passed  = sum(1 for r in self.results if r["outcome"] == "passed")
        failed  = sum(1 for r in self.results if r["outcome"] == "failed")
        skipped = sum(1 for r in self.results if r["outcome"] == "skipped")
        errored = sum(1 for r in self.results if r["outcome"] == "error")
        total   = len(self.results)

        durations = sorted(r["duration"] for r in self.results if r["outcome"] != "skipped")
        p50     = durations[len(durations) // 2] if durations else 0.0
        slowest = max(self.results, key=lambda r: r["duration"]) if self.results else None

        lines: list[str] = []
        lines.append(f"# E2E Test Report — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # ── Summary ──────────────────────────────────────────────────────────
        overall_icon = "✅" if failed == 0 else "❌"
        lines.append(f"## {overall_icon} Summary\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total | {total} |")
        lines.append(f"| ✅ Passed | {passed} |")
        if failed:
            lines.append(f"| ❌ Failed | {failed} |")
        if skipped:
            lines.append(f"| ⏭️ Skipped | {skipped} |")
        if errored:
            lines.append(f"| 💥 Errors | {errored} |")
        lines.append(f"| ⏱️ Total Duration | {total_duration:.1f}s |")
        lines.append(f"| ⏱️ p50 Duration | {p50:.2f}s |")
        if slowest:
            sname = slowest["nodeid"].split("::")[-1]
            lines.append(f"| 🐢 Slowest | {sname} ({slowest['duration']:.1f}s) |")
        model = info.get("model_id") or "N/A"
        lines.append(f"| 🤖 Model | {model} |")
        gpu = info.get("gpu_name") or "N/A"
        lines.append(f"| 🖥️ GPU | {gpu} |")
        lines.append("")

        # ── Failures ─────────────────────────────────────────────────────────
        failures = [r for r in self.results if r["outcome"] in ("failed", "error")]
        if failures:
            lines.append("## ❌ Failures\n")
            lines.append("| Test | Error |")
            lines.append("|------|-------|")
            for r in failures:
                name = r["nodeid"].split("::")[-1]
                error_msg = "—"
                for ln in r["longrepr"].splitlines():
                    stripped = ln.strip()
                    if stripped.startswith("E ") and any(
                        kw in stripped for kw in ("Error", "Exception", "Assert", "Failed")
                    ):
                        error_msg = stripped.lstrip("E").strip()[:150]
                        break
                if error_msg == "—":
                    repr_lines = [l.strip() for l in r["longrepr"].splitlines() if l.strip()]
                    error_msg = repr_lines[-1][:150] if repr_lines else "—"
                lines.append(f"| `{name}` | `{error_msg}` |")
            lines.append("")

        # ── Skipped ──────────────────────────────────────────────────────────
        skips = [r for r in self.results if r["outcome"] == "skipped"]
        if skips:
            lines.append("## ⏭️ Skipped\n")
            lines.append("| Test | Reason |")
            lines.append("|------|--------|")
            for r in skips:
                name   = r["nodeid"].split("::")[-1]
                reason = r["skip_reason"] or "—"
                lines.append(f"| `{name}` | {reason} |")
            lines.append("")

        # ── Synthesis Results ─────────────────────────────────────────────────
        synth_results = self._parse_synthesis_results()
        if synth_results:
            lines.append("## 🔊 Synthesis Results\n")
            lines.append("| Test | Voice | Format | Bytes | Audio Duration | Test Duration |")
            lines.append("|------|-------|--------|-------|----------------|---------------|")
            for s in synth_results:
                name      = s["test"]
                voice     = s.get("voice", "—")
                fmt       = s.get("format", "—")
                byt       = str(s["bytes"]) if "bytes" in s else "—"
                adur      = f"{s['audio_duration']:.2f}s" if "audio_duration" in s else "—"
                tdur      = f"{s['duration']:.1f}s"
                lines.append(f"| `{name}` | {voice} | {fmt} | {byt} | {adur} | {tdur} |")
            lines.append("")

        # ── Latency Metrics ───────────────────────────────────────────────────
        lat_metrics = self._parse_latency_metrics()
        if lat_metrics:
            lines.append("## ⚡ Latency Metrics\n")
            lines.append("| Test | Metric | Value |")
            lines.append("|------|--------|-------|")
            for lm in lat_metrics:
                lines.append(f"| `{lm['test']}` | {lm['metric']} | {lm['value']:.2f}s |")
            lines.append("")

        # ── Results by Category ───────────────────────────────────────────────
        lines.append("## Results by Category\n")
        categories: dict[str, list[dict]] = {}
        for r in self.results:
            cat = self._categorize(r["nodeid"])
            categories.setdefault(cat, []).append(r)

        lines.append("| Category | ✅ Passed | ❌ Failed | ⏭️ Skipped |")
        lines.append("|----------|----------|----------|-----------|")
        for cat in sorted(categories):
            cr = categories[cat]
            cp = sum(1 for r in cr if r["outcome"] == "passed")
            cf = sum(1 for r in cr if r["outcome"] == "failed")
            cs = sum(1 for r in cr if r["outcome"] == "skipped")
            icon = "❌" if cf else ("⏭️" if cs == len(cr) else "✅")
            lines.append(f"| {icon} {cat} | {cp} | {cf} | {cs} |")
        lines.append("")

        # ── All Tests ─────────────────────────────────────────────────────────
        lines.append("## All Tests\n")
        lines.append("| Test | Status | Duration | SLA |")
        lines.append("|------|--------|----------|-----|")
        for r in self.results:
            name    = r["nodeid"].split("::")[-1]
            outcome = r["outcome"]
            dur     = r["duration"]
            sla     = _sla_for(name)

            if outcome == "passed":
                status_icon = "✅"
                sla_icon    = "✅" if dur <= sla else f"⚠️ SLOW (≤{sla:.0f}s)"
            elif outcome == "skipped":
                status_icon = "⏭️"
                sla_icon    = "—"
            else:
                status_icon = "❌"
                sla_icon    = "—"

            lines.append(f"| `{name}` | {status_icon} {outcome.upper()} | {dur:.2f}s | {sla_icon} |")
        lines.append("")

        report_path.write_text("\n".join(lines))
        return report_path


_report_generator = MarkdownReportGenerator()


# =============================================================================
# Pytest hooks for report generation
# =============================================================================

def pytest_sessionstart(session):
    _report_generator.session_start = time.time()


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if rep.when == "call" or (rep.when == "setup" and rep.skipped):
        stdout = ""
        for section_name, section_content in rep.sections:
            if "stdout" in section_name.lower():
                stdout += section_content

        longrepr = ""
        if rep.failed and rep.longrepr:
            longrepr = str(rep.longrepr)

        skip_reason = ""
        if rep.skipped and rep.longrepr:
            if isinstance(rep.longrepr, tuple) and len(rep.longrepr) >= 3:
                reason = rep.longrepr[2]
                skip_reason = reason[len("Skipped: "):] if reason.startswith("Skipped: ") else reason
            else:
                skip_reason = str(rep.longrepr)

        _report_generator.add_result(
            nodeid=rep.nodeid,
            outcome=rep.outcome,
            duration=rep.duration,
            stdout=stdout,
            longrepr=longrepr,
            skip_reason=skip_reason,
        )


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    _report_generator.session_end = time.time()
    reports_dir = Path(__file__).parent / "reports"
    report_path = _report_generator.generate(reports_dir)
    terminalreporter.write_sep("=", "Markdown Report")
    terminalreporter.write_line(f"Report saved to: {report_path}")


# =============================================================================
# Session-scoped fixtures
# =============================================================================

@pytest.fixture(scope="session")
def base_url() -> str:
    return os.getenv("E2E_BASE_URL", DEFAULT_BASE_URL)


@pytest.fixture(scope="session")
def ws_url() -> str:
    return os.getenv("E2E_WS_URL", DEFAULT_WS_URL)


@pytest.fixture(scope="session")
def server_available(base_url: str) -> bool:
    """Non-fatal server availability check."""
    try:
        response = httpx.get(f"{base_url}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="session")
def ensure_server(base_url: str):
    """Skip test if server is not reachable."""
    start = time.time()
    last_error = None
    while time.time() - start < HEALTH_TIMEOUT:
        try:
            response = httpx.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                return
        except Exception as e:
            last_error = e
        time.sleep(1)
    msg = f"Server not available at {base_url}"
    if last_error:
        msg += f": {last_error}"
    pytest.skip(msg)


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def audio_dir(data_dir: Path) -> Path:
    return data_dir / "audio"


@pytest.fixture(scope="session", autouse=True)
def generate_ref_audio_files(audio_dir: Path):
    """Auto-generate reference audio files for clone tests on first run."""
    from utils.audio import generate_ref_audio
    generate_ref_audio(audio_dir, filename="ref_3s.wav", duration_s=3.0)
    generate_ref_audio(audio_dir, filename="ref_5s.wav", duration_s=5.0)


@pytest.fixture(scope="session")
def ref_audio_3s(audio_dir: Path) -> Path:
    """3-second sine-wave WAV for voice clone reference."""
    path = audio_dir / "ref_3s.wav"
    if not path.exists():
        pytest.skip(f"Reference audio not found: {path}")
    return path


@pytest.fixture(scope="session")
def ref_audio_5s(audio_dir: Path) -> Path:
    """5-second sine-wave WAV for voice clone reference."""
    path = audio_dir / "ref_5s.wav"
    if not path.exists():
        pytest.skip(f"Reference audio not found: {path}")
    return path


# =============================================================================
# HTTP client fixtures
# =============================================================================

@pytest.fixture
def http_client(base_url: str) -> Generator[httpx.Client, None, None]:
    """Plain httpx.Client for raw request testing."""
    with httpx.Client(base_url=base_url, timeout=300) as client:
        yield client


# =============================================================================
# Model management fixtures
# =============================================================================

@pytest.fixture(scope="session")
def ensure_model_loaded(base_url: str):
    """Ensure model is loaded by running a quick synthesis if needed."""
    response = httpx.get(f"{base_url}/health", timeout=5)
    if response.status_code == 200 and response.json().get("model_loaded"):
        return  # Already loaded

    # Trigger model load
    httpx.post(
        f"{base_url}/v1/audio/speech",
        json={"model": "qwen3-tts", "input": WARMUP_TEXT, "voice": "alloy"},
        timeout=MODEL_LOAD_TIMEOUT,
    )

    # Wait for confirmation
    start = time.time()
    while time.time() - start < MODEL_LOAD_TIMEOUT:
        try:
            r = httpx.get(f"{base_url}/health", timeout=5)
            if r.status_code == 200 and r.json().get("model_loaded"):
                return
        except Exception:
            pass
        time.sleep(3)

    pytest.skip("Model failed to load within timeout")


# =============================================================================
# Pytest configuration
# =============================================================================

def pytest_configure(config):
    config.addinivalue_line("markers", "smoke: quick smoke tests")
    config.addinivalue_line("markers", "slow: tests that take a long time")
    config.addinivalue_line("markers", "performance: performance and latency tests")
    config.addinivalue_line("markers", "websocket: WebSocket-specific tests")
    config.addinivalue_line("markers", "integration: integration tests")
    config.addinivalue_line("markers", "requires_gpu: tests requiring GPU")


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "performance" in item.nodeid or "latency" in item.nodeid:
            item.add_marker(pytest.mark.performance)
        if "websocket" in item.nodeid or "_ws_" in item.nodeid:
            item.add_marker(pytest.mark.websocket)
