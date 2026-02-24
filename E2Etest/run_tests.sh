#!/bin/bash
# Run E2E tests for Qwen3-TTS with optional server management

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Defaults
WITH_SERVER=false
SKIP_SLOW=false
MARKERS=""
VERBOSE="-v"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --with-server)
            WITH_SERVER=true
            shift
            ;;
        --fast)
            MARKERS="-m 'not slow'"
            shift
            ;;
        --smoke)
            MARKERS="-m smoke"
            shift
            ;;
        --http-only)
            MARKERS="-k http"
            shift
            ;;
        --websocket-only)
            MARKERS="-m websocket"
            shift
            ;;
        --performance)
            MARKERS="-m performance"
            shift
            ;;
        --integration)
            MARKERS="-m integration"
            shift
            ;;
        --formats)
            MARKERS="-k formats"
            shift
            ;;
        --clone)
            MARKERS="-k clone"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --with-server     Start/stop Docker container for tests"
            echo "  --smoke           Run only smoke tests"
            echo "  --fast            Skip slow tests"
            echo "  --http-only       Run only HTTP API tests"
            echo "  --websocket-only  Run only WebSocket tests"
            echo "  --performance     Run only performance tests"
            echo "  --integration     Run only integration tests"
            echo "  --formats         Run only format tests"
            echo "  --clone           Run only voice clone tests"
            echo "  --help, -h        Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --with-server          # Full test run with server"
            echo "  $0 --smoke                # Quick smoke tests only"
            echo "  $0 --fast                 # Skip slow tests"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run '$0 --help' for usage"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "Qwen3-TTS E2E Test Runner"
echo "========================================"
echo ""

cd "$PROJECT_DIR"

# Check/install dependencies
echo "Checking dependencies..."
if ! python3 -c "import pytest" 2>/dev/null; then
    echo -e "${YELLOW}Installing test dependencies...${NC}"
    pip install -q -r E2Etest/requirements.txt
fi
echo -e "${GREEN}Dependencies OK${NC}"
echo ""

# Start server if requested
if [ "$WITH_SERVER" = true ]; then
    echo "Starting server..."
    docker compose up -d --build

    echo "Waiting for server to be healthy..."
    for i in {1..60}; do
        if curl -s http://localhost:8101/health >/dev/null 2>&1; then
            echo -e "${GREEN}Server is ready!${NC}"
            break
        fi
        echo -n "."
        sleep 2
    done

    if ! curl -s http://localhost:8101/health >/dev/null 2>&1; then
        echo -e "${RED}Server failed to start${NC}"
        docker compose logs --tail=50
        exit 1
    fi
    echo ""
fi

# Verify server is running
if ! curl -s http://localhost:8101/health >/dev/null 2>&1; then
    echo -e "${YELLOW}Warning: Server not responding at localhost:8101${NC}"
    echo "Tests will be skipped. Start server with: docker compose up -d"
    echo "Or use --with-server to auto-start"
    echo ""
fi

# Build pytest command
PYTEST_CMD="pytest E2Etest/ $VERBOSE"

if [ -n "$MARKERS" ]; then
    PYTEST_CMD="$PYTEST_CMD $MARKERS"
fi

# Run tests
echo "Running tests..."
echo "Command: $PYTEST_CMD"
echo ""
echo "========================================"

set +e
eval $PYTEST_CMD
TEST_EXIT=$?
set -e

echo "========================================"

# Show latest report
LATEST_REPORT=$(ls -t E2Etest/reports/*.md 2>/dev/null | head -1)
if [ -n "$LATEST_REPORT" ]; then
    echo ""
    echo -e "${GREEN}Markdown report:${NC} $LATEST_REPORT"
fi

# Stop server if we started it
if [ "$WITH_SERVER" = true ]; then
    echo ""
    echo "Stopping server..."
    docker compose down
fi

echo ""
if [ $TEST_EXIT -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
else
    echo -e "${RED}Some tests failed (exit code: $TEST_EXIT)${NC}"
fi

exit $TEST_EXIT
