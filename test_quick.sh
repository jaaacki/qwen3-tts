#!/bin/bash
# Quick TTS test - generates a WAV file and shows info
set -e

URL="${TTS_URL:-http://localhost:8101}"
TEXT="${1:-Hello, this is a quick test of the Qwen text to speech system.}"
VOICE="${2:-ryan}"
OUT="${3:-test_output.wav}"

echo "=== Qwen3-TTS Quick Test ==="
echo "URL:   $URL"
echo "Voice: $VOICE"
echo "Text:  $TEXT"
echo ""

# Health check
echo "Health: $(curl -s "$URL/health" | python3 -m json.tool 2>/dev/null || curl -s "$URL/health")"
echo ""

# Generate speech
echo "Generating speech..."
HTTP_CODE=$(curl -s -o "$OUT" -w "%{http_code}" -X POST "$URL/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d "{\"input\": \"$TEXT\", \"voice\": \"$VOICE\"}")

if [ "$HTTP_CODE" -eq 200 ]; then
    echo "Success! HTTP $HTTP_CODE"
    python3 -c "
import wave
with wave.open('$OUT', 'rb') as w:
    print(f'File:        $OUT')
    print(f'Duration:    {w.getnframes() / w.getframerate():.2f}s')
    print(f'Sample rate: {w.getframerate()} Hz')
    print(f'Channels:    {w.getnchannels()}')
    print(f'Size:        {w.getnframes() * w.getsampwidth() * w.getnchannels()} bytes')
"
else
    echo "FAILED! HTTP $HTTP_CODE"
    cat "$OUT"
    exit 1
fi
