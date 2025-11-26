#!/bin/bash

echo "Running Cactus test suite..."
echo "============================"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

DEFAULT_MODEL="LiquidAI/LFM2-VL-450M"
DEFAULT_TRANSCRIBE_MODEL="openai/whisper-small"

MODEL_NAME="$DEFAULT_MODEL"
TRANSCRIBE_MODEL_NAME="$DEFAULT_TRANSCRIBE_MODEL"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --transcribe_model)
            TRANSCRIBE_MODEL_NAME="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model <name>            Model to use for tests (default: $DEFAULT_MODEL)"
            echo "  --transcribe_model <name> Transcribe model to use (default: $DEFAULT_TRANSCRIBE_MODEL)"
            echo "  --help, -h                Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo ""
echo "Using model: $MODEL_NAME"
echo "Using transcribe model: $TRANSCRIBE_MODEL_NAME"

echo ""
echo "Step 1: Downloading model weights..."
if ! "$PROJECT_ROOT/cli/cactus" download "$MODEL_NAME"; then
    echo "Failed to download model weights"
    exit 1
fi

if ! "$PROJECT_ROOT/cli/cactus" download "$TRANSCRIBE_MODEL_NAME"; then
    echo "Failed to download transcribe model weights"
    exit 1
fi

echo ""
echo "Step 2: Building Cactus library..."
cd "$PROJECT_ROOT"
if ! cactus/build.sh; then
    echo "Failed to build cactus library"
    exit 1
fi

echo ""
echo "Step 3: Building tests..."
cd "$PROJECT_ROOT/tests"

rm -rf build
mkdir -p build
cd build

if ! cmake .. -DCMAKE_RULE_MESSAGES=OFF -DCMAKE_VERBOSE_MAKEFILE=OFF > /dev/null 2>&1; then
    echo "Failed to configure tests"
    exit 1
fi

if ! make -j$(nproc 2>/dev/null || echo 4); then
    echo "Failed to build tests"
    exit 1
fi

echo ""
echo "Step 4: Running tests..."
echo "------------------------"

# Set model path environment variables for tests
MODEL_DIR=$(echo "$MODEL_NAME" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')
TRANSCRIBE_MODEL_DIR=$(echo "$TRANSCRIBE_MODEL_NAME" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')
export CACTUS_TEST_MODEL="$PROJECT_ROOT/weights/$MODEL_DIR"
export CACTUS_TEST_TRANSCRIBE_MODEL="$PROJECT_ROOT/weights/$TRANSCRIBE_MODEL_DIR"
echo "Using model path: $CACTUS_TEST_MODEL"
echo "Using transcribe model path: $CACTUS_TEST_TRANSCRIBE_MODEL"

echo "Discovering test executables..."
test_executables=($(find . -maxdepth 1 -name "test_*" -type f | sort))

executable_tests=()
for test_file in "${test_executables[@]}"; do
    if [ -x "$test_file" ]; then
        executable_tests+=("$test_file")
    fi
done

if [ ${#executable_tests[@]} -eq 0 ]; then
    echo "No test executables found!"
    exit 1
fi

test_executables=("${executable_tests[@]}")

echo "Found ${#test_executables[@]} test executable(s)"

for executable in "${test_executables[@]}"; do
    exec_name=$(basename "$executable")
    ./"$exec_name"
done