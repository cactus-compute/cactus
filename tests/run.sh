#!/bin/bash

echo "Running Cactus test suite..."
echo "============================"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

DEFAULT_MODEL="LiquidAI/LFM2-VL-450M"
DEFAULT_TRANSCRIBE_MODEL="nvidia/parakeet-tdt-0.6b-v3"
DEFAULT_WHISPER_MODEL="openai/whisper-small"
DEFAULT_VAD_MODEL="snakers4/silero-vad"
DEFAULT_DIARIZE_MODEL="pyannote/segmentation-3.0"
DEFAULT_EMBED_SPEAKER_MODEL="pyannote/wespeaker-voxceleb-resnet34-LM"

MODEL_NAME="$DEFAULT_MODEL"
TRANSCRIBE_MODEL_NAME="$DEFAULT_TRANSCRIBE_MODEL"
WHISPER_MODEL_NAME="$DEFAULT_WHISPER_MODEL"
VAD_MODEL_NAME="$DEFAULT_VAD_MODEL"
DIARIZE_MODEL_NAME="$DEFAULT_DIARIZE_MODEL"
EMBED_SPEAKER_MODEL_NAME="$DEFAULT_EMBED_SPEAKER_MODEL"
ANDROID_MODE=false
IOS_MODE=false
NO_REBUILD=false
EXHAUSTIVE_MODE=false
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
        --whisper_model)
            WHISPER_MODEL_NAME="$2"
            shift 2
            ;;
        --vad_model)
            VAD_MODEL_NAME="$2"
            shift 2
            ;;
        --diarize_model)
            DIARIZE_MODEL_NAME="$2"
            shift 2
            ;;
        --embed_speaker_model)
            EMBED_SPEAKER_MODEL_NAME="$2"
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
        --exhaustive)
            EXHAUSTIVE_MODE=true
            shift
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
            echo "  --whisper_model <name>    Whisper model for language detection (default: $DEFAULT_WHISPER_MODEL)"
            echo "  --vad_model <name>        VAD model to use (default: $DEFAULT_VAD_MODEL)"
            echo "  --diarize_model <name>    Diarization model to use (default: $DEFAULT_DIARIZE_MODEL)"
            echo "  --embed_speaker_model <name> Speaker embedding model to use (default: $DEFAULT_EMBED_SPEAKER_MODEL)"
            echo "  --precision <type>        Precision for model conversion (MIXED, FP16, INT8, INT4)"
            echo "  --android                 Run tests on Android device or emulator"
            echo "  --ios                     Run tests on iOS device or simulator"
            echo "  --no-rebuild              Skip building cactus library and tests"
            echo "  --exhaustive              Run exhaustive golden tests for all model families and precisions"
            echo "  --only <test_name>        Only run the specified test (llm, vlm, stt, embed, rag, graph, index, kernel, kv_cache, performance)"
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
echo "Using whisper model: $WHISPER_MODEL_NAME"
echo "Using vad model: $VAD_MODEL_NAME"
echo "Using diarize model: $DIARIZE_MODEL_NAME"
echo "Using embed_speaker model: $EMBED_SPEAKER_MODEL_NAME"
if [ -n "$PRECISION" ]; then
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

if ! cactus download "$WHISPER_MODEL_NAME" $PRECISION_FLAG; then
    echo "Failed to download whisper model weights"
    exit 1
fi

if ! cactus download "$VAD_MODEL_NAME" $PRECISION_FLAG; then
    echo "Failed to download VAD model weights"
    exit 1
fi

if ! cactus download "$DIARIZE_MODEL_NAME" $PRECISION_FLAG; then
    echo "Failed to download diarize model weights"
    exit 1
fi

if ! cactus download "$EMBED_SPEAKER_MODEL_NAME" $PRECISION_FLAG; then
    echo "Failed to download embed_speaker model weights"
    exit 1
fi

echo ""
if [ "$ANDROID_MODE" = true ]; then
    export CACTUS_TEST_ONLY="$ONLY_EXEC"
    exec "$SCRIPT_DIR/android/run.sh" "$MODEL_NAME" "$TRANSCRIBE_MODEL_NAME" "$WHISPER_MODEL_NAME" "$VAD_MODEL_NAME" "$DIARIZE_MODEL_NAME" "$EMBED_SPEAKER_MODEL_NAME"
fi

if [ "$IOS_MODE" = true ]; then
    export CACTUS_TEST_ONLY="$ONLY_EXEC"
    exec "$SCRIPT_DIR/ios/run.sh" "$MODEL_NAME" "$TRANSCRIBE_MODEL_NAME" "$WHISPER_MODEL_NAME" "$VAD_MODEL_NAME" "$DIARIZE_MODEL_NAME" "$EMBED_SPEAKER_MODEL_NAME"
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
WHISPER_MODEL_DIR=$(echo "$WHISPER_MODEL_NAME" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')
VAD_MODEL_DIR=$(echo "$VAD_MODEL_NAME" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')
DIARIZE_MODEL_DIR=$(echo "$DIARIZE_MODEL_NAME" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')
EMBED_SPEAKER_MODEL_DIR=$(echo "$EMBED_SPEAKER_MODEL_NAME" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')

export CACTUS_TEST_MODEL="$PROJECT_ROOT/weights/$MODEL_DIR"
export CACTUS_TEST_TRANSCRIBE_MODEL="$PROJECT_ROOT/weights/$TRANSCRIBE_MODEL_DIR"
export CACTUS_TEST_WHISPER_MODEL="$PROJECT_ROOT/weights/$WHISPER_MODEL_DIR"
export CACTUS_TEST_VAD_MODEL="$PROJECT_ROOT/weights/$VAD_MODEL_DIR"
export CACTUS_TEST_DIARIZE_MODEL="$PROJECT_ROOT/weights/$DIARIZE_MODEL_DIR"
export CACTUS_TEST_EMBED_SPEAKER_MODEL="$PROJECT_ROOT/weights/$EMBED_SPEAKER_MODEL_DIR"
export CACTUS_TEST_ASSETS="$PROJECT_ROOT/tests/assets"
export CACTUS_INDEX_PATH="$PROJECT_ROOT/tests/assets"

echo "Using model path: $CACTUS_TEST_MODEL"
echo "Using transcribe model path: $CACTUS_TEST_TRANSCRIBE_MODEL"
echo "Using whisper model path: $CACTUS_TEST_WHISPER_MODEL"
echo "Using VAD model path: $CACTUS_TEST_VAD_MODEL"
echo "Using diarize model path: $CACTUS_TEST_DIARIZE_MODEL"
echo "Using embed_speaker model path: $CACTUS_TEST_EMBED_SPEAKER_MODEL"
echo "Using assets path: $CACTUS_TEST_ASSETS"
echo "Using index path: $CACTUS_INDEX_PATH"

echo "Discovering test executables..."
test_executables=($(find . -maxdepth 1 -name "test_*" ! -name "test_exhaustive" -type f | sort))

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

if [ "$EXHAUSTIVE_MODE" = true ]; then
    echo ""
    echo "Step 5: Running exhaustive tests..."
    echo "------------------------------------"
    exec "$SCRIPT_DIR/golden/generate_exhaustive_golden.sh"
fi
