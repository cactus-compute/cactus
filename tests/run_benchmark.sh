#!/bin/bash
# Run cactus matmul + attention benchmarks for the v2-bench graph suite.
#
# Builds tests/build (if needed) and runs both matmul_bench and attn_bench,
# emitting CSV files that can be plotted by external tooling.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

usage() {
    cat <<EOF
Usage: $0 [options]

Build flags (forwarded to cmake):
  --with-ggml         Enable llama.cpp / ggml backend (-DWITH_GGML=ON)
  --with-litert       Enable LiteRT backend          (-DWITH_LITERT=ON)
  --with-executorch   Enable ExecuTorch backend      (-DWITH_EXECUTORCH=ON)
  --with-onnxrt       Enable ONNX Runtime backend    (-DWITH_ONNXRT=ON)
  --external-frameworks
                      Auto-enable any third-party tree found under ../third_party/

Run options:
  --skip-cactus       Don't (re)build the underlying libcactus.a
  --skip-build        Skip CMake/make for tests/
  --no-attn           Run only the matmul benchmark
  --no-matmul         Run only the attention benchmark
  --backends LIST     Comma-separated framework names (cactus,ggml,litert,...)
  --threads N|max     Override thread count
  --csv-dir DIR       Where to write CSVs (default: tests/build)

All other args are forwarded to both benchmark executables.
EOF
}

CMAKE_FLAGS=()
PASS_ARGS=()
RUN_MATMUL=1
RUN_ATTN=1
SKIP_CACTUS=0
SKIP_BUILD=0
EXTERNAL_AUTO=0
CSV_DIR="$BUILD_DIR"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-ggml)        CMAKE_FLAGS+=("-DWITH_GGML=ON"); shift;;
        --with-litert)      CMAKE_FLAGS+=("-DWITH_LITERT=ON"); shift;;
        --with-executorch)  CMAKE_FLAGS+=("-DWITH_EXECUTORCH=ON"); shift;;
        --with-onnxrt)      CMAKE_FLAGS+=("-DWITH_ONNXRT=ON"); shift;;
        --external-frameworks) EXTERNAL_AUTO=1; shift;;
        --no-attn)          RUN_ATTN=0; shift;;
        --no-matmul)        RUN_MATMUL=0; shift;;
        --skip-cactus)      SKIP_CACTUS=1; shift;;
        --skip-build)       SKIP_BUILD=1; shift;;
        --csv-dir)          CSV_DIR="$2"; shift 2;;
        -h|--help)          usage; exit 0;;
        *)                  PASS_ARGS+=("$1"); shift;;
    esac
done

if [[ $EXTERNAL_AUTO -eq 1 ]]; then
    [[ -d "$ROOT_DIR/../third_party/ggml"        ]] && CMAKE_FLAGS+=("-DWITH_GGML=ON")
    [[ -d "$ROOT_DIR/../third_party/litert"      ]] && CMAKE_FLAGS+=("-DWITH_LITERT=ON")
    [[ -d "$ROOT_DIR/../third_party/onnxruntime" ]] && CMAKE_FLAGS+=("-DWITH_ONNXRT=ON")
    CMAKE_FLAGS+=("-DWITH_EXECUTORCH=ON")
fi

if [[ $SKIP_CACTUS -eq 0 ]]; then
    if [[ ! -f "$ROOT_DIR/cactus/build/libcactus.a" ]]; then
        echo "==> Building libcactus.a"
        bash "$ROOT_DIR/cactus/build.sh"
    else
        echo "==> Reusing existing libcactus.a (use --skip-cactus to silence; rebuild with cactus/build.sh)"
    fi
fi

if [[ $SKIP_BUILD -eq 0 ]]; then
    mkdir -p "$BUILD_DIR"
    pushd "$BUILD_DIR" > /dev/null
    cmake "$SCRIPT_DIR" "${CMAKE_FLAGS[@]}"
    make -j$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4) matmul_bench attn_bench
    popd > /dev/null
fi

mkdir -p "$CSV_DIR"

if [[ $RUN_MATMUL -eq 1 ]]; then
    echo
    echo "==> Running matmul_bench"
    "$BUILD_DIR/matmul_bench" --csv "$CSV_DIR/matmul_bench.csv" "${PASS_ARGS[@]}"
fi

if [[ $RUN_ATTN -eq 1 ]]; then
    echo
    echo "==> Running attn_bench"
    "$BUILD_DIR/attn_bench" --csv "$CSV_DIR/attn_bench.csv" "${PASS_ARGS[@]}"
fi

echo
echo "==> CSV results in $CSV_DIR"
ls -la "$CSV_DIR"/*.csv 2>/dev/null || true
