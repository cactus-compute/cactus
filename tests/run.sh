#!/bin/bash

echo "Running Cactus test suite..."
echo "============================"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

DEFAULT_MODEL="LiquidAI/LFM2-VL-450M"
DEFAULT_TRANSCRIBE_MODEL="openai/whisper-small"
DEFAULT_VAD_MODEL="snakers4/silero-vad"

MODEL_NAME="$DEFAULT_MODEL"
TRANSCRIBE_MODEL_NAME="$DEFAULT_TRANSCRIBE_MODEL"
VAD_MODEL_NAME="$DEFAULT_VAD_MODEL"
ANDROID_MODE=false
IOS_MODE=false
NO_REBUILD=false
ONLY_EXEC=""

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
        --vad_model)
            VAD_MODEL_NAME="$2"
            shift 2
            ;;
        --android)
            ANDROID_MODE=true
            shift
            ;;
        --ios)
            IOS_MODE=true
            shift
            ;;
        --no-rebuild)
            NO_REBUILD=true
            shift
            ;;
        --only)
            ONLY_EXEC="$2"
            shift 2
            ;;
        --precision)
            PRECISION="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model <name>            Model to use for tests (default: $DEFAULT_MODEL)"
            echo "  --transcribe_model <name> Transcribe model to use (default: $DEFAULT_TRANSCRIBE_MODEL)"
            echo "  --vad_model <name>        VAD model to use (default: $DEFAULT_VAD_MODEL)"
            echo "  --precision <type>        Precision for model conversion (MIXED, FP16, INT8, INT4)"
            echo "  --android                 Run tests on Android device or emulator"
            echo "  --ios                     Run tests on iOS device or simulator"
            echo "  --no-rebuild              Skip building cactus library and tests"
            echo "  --only <test_name>        Only run the specified test (engine, graph, index, kernel, kv_cache, performance, etc)"
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
echo "Using vad model: $VAD_MODEL_NAME"
if [ ! -z "$PRECISION" ]; then
    echo "Using precision: $PRECISION"
    PRECISION_FLAG="--precision $PRECISION"
else
    PRECISION_FLAG=""
fi

echo ""
echo "Step 1: Downloading model weights..."
if ! cactus download "$MODEL_NAME" $PRECISION_FLAG; then
    echo "Failed to download model weights"
    exit 1
fi

if ! cactus download "$TRANSCRIBE_MODEL_NAME" $PRECISION_FLAG; then
    echo "Failed to download transcribe model weights"
    exit 1
fi

if ! cactus download "$VAD_MODEL_NAME" $PRECISION_FLAG; then
    echo "Failed to download VAD model weights"
    exit 1
fi

echo ""
if [ "$ANDROID_MODE" = true ]; then
    exec "$SCRIPT_DIR/android/run.sh" "$MODEL_NAME" "$TRANSCRIBE_MODEL_NAME" "$VAD_MODEL_NAME"
fi

if [ "$IOS_MODE" = true ]; then
    exec "$SCRIPT_DIR/ios/run.sh" "$MODEL_NAME" "$TRANSCRIBE_MODEL_NAME" "$VAD_MODEL_NAME"
fi

if [ "$NO_REBUILD" = false ]; then
    echo "Step 2: Building Cactus library..."
    if ! cactus build; then
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
else
    echo "Skipping build (--no-rebuild)"
    cd "$PROJECT_ROOT/tests/build"
fi

echo ""
echo "Step 4: Running tests..."
echo "------------------------"

# Set model path environment variables for tests
MODEL_DIR=$(echo "$MODEL_NAME" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')
TRANSCRIBE_MODEL_DIR=$(echo "$TRANSCRIBE_MODEL_NAME" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')
VAD_MODEL_DIR=$(echo "$VAD_MODEL_NAME" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')

export CACTUS_TEST_MODEL="$PROJECT_ROOT/weights/$MODEL_DIR"
export CACTUS_TEST_TRANSCRIBE_MODEL="$PROJECT_ROOT/weights/$TRANSCRIBE_MODEL_DIR"
export CACTUS_TEST_VAD_MODEL="$PROJECT_ROOT/weights/$VAD_MODEL_DIR"
export CACTUS_TEST_ASSETS="$PROJECT_ROOT/tests/assets"
export CACTUS_INDEX_PATH="$PROJECT_ROOT/tests/assets"

echo "Using model path: $CACTUS_TEST_MODEL"
echo "Using transcribe model path: $CACTUS_TEST_TRANSCRIBE_MODEL"
echo "Using VAD model path: $CACTUS_TEST_VAD_MODEL"
echo "Using assets path: $CACTUS_TEST_ASSETS"
echo "Using index path: $CACTUS_INDEX_PATH"

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

# If --only is set, execute only the named test
if [ -n "$ONLY_EXEC" ]; then
    allowed=()
    for test_file in "${executable_tests[@]}"; do
        test_name=$(basename "$test_file" | sed 's/^test_//')
        allowed+=("$test_name")
    done

    ok=false
    for a in "${allowed[@]}"; do
        if [ "$a" = "$ONLY_EXEC" ]; then
            ok=true
            break
        fi
    done
    if [ "$ok" = false ]; then
        echo "Unknown test name: $ONLY_EXEC"
        echo "Allowed: ${allowed[*]}"
        exit 1
    fi

    target="./test_$ONLY_EXEC"
    if [ ! -f "$target" ] || [ ! -x "$target" ]; then
        echo "Could not find or execute test: $target"
        exit 1
    fi

    test_executables=("$target")
fi

echo "Found ${#test_executables[@]} test executable(s)"

for executable in "${test_executables[@]}"; do
    exec_name=$(basename "$executable")
    ./"$exec_name"
done
