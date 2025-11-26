#!/bin/bash

echo "Running Cactus test suite..."
echo "============================"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

WEIGHTS_DIR="$PROJECT_ROOT/weights/lfm2-vl-1.6b"
if [ ! -d "$WEIGHTS_DIR" ] || [ ! -f "$WEIGHTS_DIR/config.txt" ]; then
    echo ""
    echo "Weights not found. Generating weights..."
    echo "============================================="
    cd "$PROJECT_ROOT"

    if ! command -v python3 &> /dev/null; then
        echo "Error: python3 is not installed. Please install Python 3 and try again."
        exit 1
    fi

    VENV_DIR="$PROJECT_ROOT/.venv"
    if [ ! -d "$VENV_DIR" ]; then
        echo "Creating Python virtual environment..."
        python3 -m venv "$VENV_DIR"
    fi

    echo "Activating virtual environment and installing dependencies..."
    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate"
    if python3 -m pip install -r tools/requirements.txt; then
        echo "Running: python3 tools/convert_hf.py LiquidAI/LFM2-VL-1.6B weights/lfm2-vl-1.6b/ --precision INT8"
        if python3 tools/convert_hf.py LiquidAI/LFM2-VL-1.6B weights/lfm2-vl-1.6b/ --precision INT8; then
            echo "Successfully generated Weights"
        else
            echo "Warning: Failed to generate Weights. Tests may fail."
        fi
    else
        echo "Warning: Failed to install python dependencies. Tests may fail."
        echo "Please run manually: python3 tools/convert_hf.py LiquidAI/LFM2-VL-1.6B weights/lfm2-vl-1.6b/ --precision INT8"
    fi
else
    echo ""
    echo "Weights found at $WEIGHTS_DIR"
fi

echo ""
echo "Step 1: Building Cactus library..."
cd "$PROJECT_ROOT"
if ! cactus/build.sh; then
    echo "Failed to build cactus library"
    exit 1
fi

echo ""
echo "Step 2: Building tests..."
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
echo "Step 3: Running tests..."
echo "------------------------"

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