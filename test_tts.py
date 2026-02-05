#!/usr/bin/env python3
"""Test suite for Qwen3-TTS API server."""

import requests
import wave
import io
import sys
import time
import os

BASE_URL = os.getenv("TTS_URL", "http://localhost:8101")

PASS = 0
FAIL = 0


def report(name, ok, detail=""):
    global PASS, FAIL
    status = "PASS" if ok else "FAIL"
    if ok:
        PASS += 1
    else:
        FAIL += 1
    suffix = f" - {detail}" if detail else ""
    print(f"  [{status}] {name}{suffix}")


def test_health():
    print("\n=== Health Check ===")
    r = requests.get(f"{BASE_URL}/health", timeout=10)
    report("GET /health returns 200", r.status_code == 200)

    data = r.json()
    report("status is ok", data.get("status") == "ok")
    report("model is loaded", data.get("model_loaded") is True)
    report("CUDA available", data.get("cuda") is True)
    report("voices list present", len(data.get("voices", [])) > 0, f"{len(data.get('voices', []))} voices")


def test_basic_synthesis():
    print("\n=== Basic Speech Synthesis ===")
    payload = {
        "input": "Hello, this is a test of text to speech.",
        "voice": "ryan",
    }
    start = time.time()
    r = requests.post(f"{BASE_URL}/v1/audio/speech", json=payload, timeout=120)
    elapsed = time.time() - start

    report("POST /v1/audio/speech returns 200", r.status_code == 200, f"{elapsed:.1f}s")
    report("Content-Type is audio/wav", "audio/wav" in r.headers.get("Content-Type", ""))

    # Validate WAV
    try:
        w = wave.open(io.BytesIO(r.content), "rb")
        sr = w.getframerate()
        dur = w.getnframes() / sr
        report("Valid WAV file", True, f"{dur:.1f}s @ {sr}Hz")
        report("Duration > 1s", dur > 1.0)
        w.close()
    except Exception as e:
        report("Valid WAV file", False, str(e))


def test_voices():
    print("\n=== Voice Variants ===")
    voices = ["vivian", "aiden", "serena", "alloy", "nova"]
    for voice in voices:
        payload = {"input": "Testing voice selection.", "voice": voice}
        r = requests.post(f"{BASE_URL}/v1/audio/speech", json=payload, timeout=120)
        ok = r.status_code == 200 and len(r.content) > 1000
        report(f"Voice '{voice}'", ok, f"{len(r.content)} bytes")


def test_languages():
    print("\n=== Multi-language ===")
    cases = [
        ("English", "This is an English speech test.", "ryan"),
        ("Chinese", "你好，这是一个中文语音测试。", "vivian"),
        ("Japanese", "こんにちは、音声テストです。", "ono_anna"),
        ("Korean", "안녕하세요, 음성 테스트입니다.", "sohee"),
    ]
    for lang, text, voice in cases:
        payload = {"input": text, "voice": voice, "language": lang}
        r = requests.post(f"{BASE_URL}/v1/audio/speech", json=payload, timeout=120)
        ok = r.status_code == 200 and len(r.content) > 1000
        report(f"{lang}", ok, f"{len(r.content)} bytes")


def test_formats():
    print("\n=== Output Formats ===")
    for fmt in ["wav", "flac"]:
        payload = {
            "input": "Testing audio format output.",
            "voice": "ryan",
            "response_format": fmt,
        }
        r = requests.post(f"{BASE_URL}/v1/audio/speech", json=payload, timeout=120)
        report(f"Format '{fmt}'", r.status_code == 200, f"{len(r.content)} bytes")


def test_error_handling():
    print("\n=== Error Handling ===")

    # Empty input
    r = requests.post(f"{BASE_URL}/v1/audio/speech", json={"input": ""}, timeout=10)
    report("Empty input returns 400", r.status_code == 400)

    # Whitespace-only input
    r = requests.post(f"{BASE_URL}/v1/audio/speech", json={"input": "   "}, timeout=10)
    report("Whitespace input returns 400", r.status_code == 400)


def test_default_voice():
    print("\n=== Default Voice (no voice param) ===")
    payload = {"input": "Testing without specifying a voice."}
    r = requests.post(f"{BASE_URL}/v1/audio/speech", json=payload, timeout=120)
    ok = r.status_code == 200 and len(r.content) > 1000
    report("Default voice works", ok, f"{len(r.content)} bytes")


def test_long_text():
    print("\n=== Long Text ===")
    long_text = "This is a longer text to test synthesis. " * 5
    payload = {"input": long_text, "voice": "ryan"}
    start = time.time()
    r = requests.post(f"{BASE_URL}/v1/audio/speech", json=payload, timeout=300)
    elapsed = time.time() - start
    ok = r.status_code == 200 and len(r.content) > 5000
    report("Long text synthesis", ok, f"{len(r.content)} bytes in {elapsed:.1f}s")


if __name__ == "__main__":
    print(f"Qwen3-TTS API Test Suite")
    print(f"Target: {BASE_URL}")

    # Check server is up
    try:
        requests.get(f"{BASE_URL}/health", timeout=5)
    except requests.ConnectionError:
        print(f"\nERROR: Cannot connect to {BASE_URL}")
        print("Is the TTS server running? Try: docker compose up -d")
        sys.exit(1)

    test_health()
    test_basic_synthesis()
    test_default_voice()
    test_voices()
    test_languages()
    test_formats()
    test_error_handling()
    test_long_text()

    print(f"\n{'='*40}")
    total = PASS + FAIL
    print(f"Results: {PASS}/{total} passed, {FAIL} failed")
    sys.exit(0 if FAIL == 0 else 1)
