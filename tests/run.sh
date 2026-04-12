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
SVE_MODE=false
SVE_BENCHMARK_ONLY=false
SVE_SIZES=""
SVE_ITERATIONS=""

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
        --sve)
            SVE_MODE=true
            shift
            ;;
        --sve-sizes)
            SVE_SIZES="$2"
            shift 2
            ;;
        --sve-iterations)
            SVE_ITERATIONS="$2"
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
            echo "  --sve                     Run the focused matmul benchmark, or compare NEON vs scalable matmul for the selected test suite"
            echo "  --sve-sizes <MxKxN,...>   Custom matmul shapes for --sve (example: 1x1024x1024,4x2048x2048)"
            echo "  --sve-iterations <count>  Iterations per shape for --sve"
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

if [ "$SVE_MODE" = true ]; then
    if [ -z "$ONLY_EXEC" ]; then
        ONLY_EXEC="performance"
        SVE_BENCHMARK_ONLY=true
        echo "Scalable matmul benchmark mode: enabled"
        if [ -n "$SVE_SIZES" ]; then
            echo "Scalable matmul sizes: $SVE_SIZES"
        fi
        if [ -n "$SVE_ITERATIONS" ]; then
            echo "Scalable matmul iterations: $SVE_ITERATIONS"
        fi
    else
        echo "Scalable matmul compare requested for test suite '$ONLY_EXEC'"
    fi
    if [ -z "$PRECISION" ]; then
        echo "Note: default model test weights are often INT4."
        echo "      On SME2-capable devices, --sve now compares NEON vs scalable INT4 matmul on those weights."
        echo "      Use --precision INT4 to guarantee regenerated INT4 weights before the comparison."
    fi
fi

echo ""
SKIP_STANDARD_DOWNLOADS=false
if [ "$ONLY_EXEC" = "gemma4_suite" ]; then
    SKIP_STANDARD_DOWNLOADS=true
fi
if [ "$SVE_BENCHMARK_ONLY" = true ]; then
    SKIP_STANDARD_DOWNLOADS=true
fi

if [ "$SKIP_STANDARD_DOWNLOADS" = false ]; then
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
else
    if [ "$SVE_BENCHMARK_ONLY" = true ]; then
        echo "Step 1: Skipping model downloads for --sve scalable benchmark mode"
    else
        echo "Step 1: Skipping standard downloads for --only gemma4_suite"
    fi
fi

if [ "$SVE_BENCHMARK_ONLY" = false ]; then
    if ! cactus download "$DIARIZE_MODEL_NAME" $PRECISION_FLAG; then
        echo "Failed to download diarize model weights"
        exit 1
    fi

    if ! cactus download "$EMBED_SPEAKER_MODEL_NAME" $PRECISION_FLAG; then
        echo "Failed to download embed_speaker model weights"
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

    if [ "$SVE_BENCHMARK_ONLY" = true ]; then
        if ! make -j$(nproc 2>/dev/null || echo 4) test_performance; then
            echo "Failed to build scalable benchmark target"
            exit 1
        fi
    else
        if ! make -j$(nproc 2>/dev/null || echo 4); then
            echo "Failed to build tests"
            exit 1
        fi
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
export CACTUS_TEST_GEMMA4_MODEL="${CACTUS_TEST_GEMMA4_MODEL:-$PROJECT_ROOT/weights/gemma4_int4}"
export CACTUS_TEST_TRANSCRIBE_MODEL="$PROJECT_ROOT/weights/$TRANSCRIBE_MODEL_DIR"
export CACTUS_TEST_WHISPER_MODEL="$PROJECT_ROOT/weights/$WHISPER_MODEL_DIR"
export CACTUS_TEST_VAD_MODEL="$PROJECT_ROOT/weights/$VAD_MODEL_DIR"
export CACTUS_TEST_DIARIZE_MODEL="$PROJECT_ROOT/weights/$DIARIZE_MODEL_DIR"
export CACTUS_TEST_EMBED_SPEAKER_MODEL="$PROJECT_ROOT/weights/$EMBED_SPEAKER_MODEL_DIR"
export CACTUS_TEST_ASSETS="$PROJECT_ROOT/tests/assets"
export CACTUS_INDEX_PATH="$PROJECT_ROOT/tests/assets"
if [ "$SVE_MODE" = true ]; then
    if [ "$SVE_BENCHMARK_ONLY" = true ]; then
        export CACTUS_FORCE_SVE_MATMUL=1
        export CACTUS_FORCE_SVE2_MATMUL=1
        export CACTUS_TEST_SVE_ONLY=1
        if [ -n "$SVE_SIZES" ]; then
            export CACTUS_TEST_SVE_SIZES="$SVE_SIZES"
        fi
        if [ -n "$SVE_ITERATIONS" ]; then
            export CACTUS_TEST_SVE_ITERATIONS="$SVE_ITERATIONS"
        fi
    else
        unset CACTUS_FORCE_NEON_MATMUL
        unset CACTUS_FORCE_SVE_MATMUL
        unset CACTUS_FORCE_SVE2_MATMUL
        unset CACTUS_TEST_SVE_ONLY
        unset CACTUS_TEST_SVE_SIZES
        unset CACTUS_TEST_SVE_ITERATIONS
    fi
fi

echo "Using model path: $CACTUS_TEST_MODEL"
echo "Using transcribe model path: $CACTUS_TEST_TRANSCRIBE_MODEL"
echo "Using whisper model path: $CACTUS_TEST_WHISPER_MODEL"
echo "Using VAD model path: $CACTUS_TEST_VAD_MODEL"
echo "Using diarize model path: $CACTUS_TEST_DIARIZE_MODEL"
echo "Using embed_speaker model path: $CACTUS_TEST_EMBED_SPEAKER_MODEL"
echo "Using assets path: $CACTUS_TEST_ASSETS"
echo "Using index path: $CACTUS_INDEX_PATH"
if [ "$SVE_MODE" = true ]; then
    if [ "$SVE_BENCHMARK_ONLY" = true ]; then
        echo "Using scalable matmul performance mode (SVE/SME2)"
    else
        echo "Scalable matmul compare requested for test execution (SVE/SME2 when available)"
    fi
fi

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

detect_scalable_backend() {
    if command -v sysctl >/dev/null 2>&1; then
        local sme2
        local sve

        sme2=$(sysctl -n hw.optional.arm.FEAT_SME2 2>/dev/null || echo 0)
        if [ "$sme2" = "1" ]; then
            echo "SME2"
            return
        fi

        sve=$(sysctl -n hw.optional.arm.FEAT_SVE 2>/dev/null || echo 0)
        if [ "$sve" = "1" ]; then
            echo "SVE"
            return
        fi
    fi

    echo "SCALABLE"
}

scalable_backend_available() {
    local backend
    backend=$(detect_scalable_backend)
    [ "$backend" != "SCALABLE" ]
}

run_timed_test_executable() {
    local executable="$1"
    local mode_label="$2"
    local neon_override="$3"
    local scalable_override="$4"
    local timing_file

    timing_file=$(mktemp)

    env \
        CACTUS_FORCE_NEON_MATMUL="$neon_override" \
        CACTUS_FORCE_SVE_MATMUL="$scalable_override" \
        CACTUS_FORCE_SVE2_MATMUL="$scalable_override" \
        "$executable" >/dev/null 2>&1
    local warmup_status=$?
    if [ $warmup_status -ne 0 ]; then
        echo "Warmup run failed for $(basename "$executable") [$mode_label]"
        rm -f "$timing_file"
        return $warmup_status
    fi

    echo ""
    echo "Running $(basename "$executable") [$mode_label]..."

    env \
        CACTUS_FORCE_NEON_MATMUL="$neon_override" \
        CACTUS_FORCE_SVE_MATMUL="$scalable_override" \
        CACTUS_FORCE_SVE2_MATMUL="$scalable_override" \
        python3 - "$executable" "$timing_file" <<'PY'
import os
import subprocess
import sys
import time

executable = sys.argv[1]
timing_path = sys.argv[2]

start = time.perf_counter()
result = subprocess.run([executable], env=os.environ.copy())
elapsed_ms = (time.perf_counter() - start) * 1000.0

with open(timing_path, "w", encoding="utf-8") as f:
    f.write(f"{elapsed_ms:.6f}\n")
    f.write(f"{result.returncode}\n")

sys.exit(result.returncode)
PY
    local status=$?

    if [ -f "$timing_file" ]; then
        LAST_TIMED_RUN_MS=$(sed -n '1p' "$timing_file")
        rm -f "$timing_file"
    else
        LAST_TIMED_RUN_MS=""
    fi

    return $status
}

SVE_COMPARE_SUITE=false
if [ "$SVE_MODE" = true ] && [ "$SVE_BENCHMARK_ONLY" = false ]; then
    if scalable_backend_available; then
        SVE_COMPARE_SUITE=true
    else
        echo "No SVE or SME2 backend available on this runtime."
        echo "Running the selected test suite once without NEON vs scalable comparison."
    fi
fi

if [ "$SVE_COMPARE_SUITE" = true ]; then
    SCALABLE_BACKEND_NAME=$(detect_scalable_backend)
    TOTAL_NEON_MS="0"
    TOTAL_SCALABLE_MS="0"

    echo "Scalable backend hint: $SCALABLE_BACKEND_NAME"
    if [ -n "$PRECISION" ]; then
        echo "Precision under comparison: $PRECISION"
    else
        echo "Precision under comparison: existing/default test weights (typically INT4 for LLM/VLM)"
    fi
    echo "Timing method: one warmup run per backend, then one timed run per backend"
fi

for executable in "${test_executables[@]}"; do
    exec_name=$(basename "$executable")
    if [ "$SVE_COMPARE_SUITE" = true ]; then
        run_timed_test_executable "$executable" "NEON" "1" "0"
        status=$?
        if [ $status -ne 0 ]; then
            exit $status
        fi
        neon_time_ms="$LAST_TIMED_RUN_MS"
        TOTAL_NEON_MS=$(awk -v total="$TOTAL_NEON_MS" -v add="$neon_time_ms" 'BEGIN { printf "%.6f", total + add }')

        run_timed_test_executable "$executable" "$SCALABLE_BACKEND_NAME" "0" "1"
        status=$?
        if [ $status -ne 0 ]; then
            exit $status
        fi
        scalable_time_ms="$LAST_TIMED_RUN_MS"
        TOTAL_SCALABLE_MS=$(awk -v total="$TOTAL_SCALABLE_MS" -v add="$scalable_time_ms" 'BEGIN { printf "%.6f", total + add }')

        speedup=$(awk -v neon="$neon_time_ms" -v scalable="$scalable_time_ms" 'BEGIN {
            if (scalable <= 0) print "inf";
            else printf "%.3f", neon / scalable;
        }')
        delta_ms=$(awk -v neon="$neon_time_ms" -v scalable="$scalable_time_ms" 'BEGIN {
            printf "%.3f", neon - scalable;
        }')

        echo ""
        echo "Wall-time comparison for $exec_name:"
        echo "  NEON: ${neon_time_ms}ms"
        echo "  ${SCALABLE_BACKEND_NAME}: ${scalable_time_ms}ms"
        echo "  Speedup (NEON/${SCALABLE_BACKEND_NAME}): ${speedup}x"
        echo "  Delta: ${delta_ms}ms"
    else
        ./"$exec_name"
    fi
done

if [ "$SVE_COMPARE_SUITE" = true ]; then
    total_speedup=$(awk -v neon="$TOTAL_NEON_MS" -v scalable="$TOTAL_SCALABLE_MS" 'BEGIN {
        if (scalable <= 0) print "inf";
        else printf "%.3f", neon / scalable;
    }')
    total_delta_ms=$(awk -v neon="$TOTAL_NEON_MS" -v scalable="$TOTAL_SCALABLE_MS" 'BEGIN {
        printf "%.3f", neon - scalable;
    }')

    echo ""
    echo "Overall suite wall-time comparison:"
    echo "  NEON total: ${TOTAL_NEON_MS}ms"
    echo "  ${SCALABLE_BACKEND_NAME} total: ${TOTAL_SCALABLE_MS}ms"
    echo "  Speedup (NEON/${SCALABLE_BACKEND_NAME}): ${total_speedup}x"
    echo "  Delta: ${total_delta_ms}ms"
fi

if [ "$EXHAUSTIVE_MODE" = true ]; then
    echo ""
    echo "Step 5: Running exhaustive tests..."
    echo "------------------------------------"
    exec "$SCRIPT_DIR/golden/generate_exhaustive_golden.sh"
fi
