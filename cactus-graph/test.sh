#!/bin/bash
set -e

cd "$(dirname "$0")"

SUITE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --suite) SUITE="${2:?--suite needs an argument}"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

echo "Building and testing cactus-graph..."

rm -rf build
mkdir -p build
cd build

cmake .. -DCACTUS_BUILD_TESTS=ON -DCMAKE_RULE_MESSAGES=OFF -DCMAKE_VERBOSE_MAKEFILE=OFF > /dev/null
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo ""
FAILED=0
if [ -n "$SUITE" ]; then
    target="./test_${SUITE}"
    if [ -x "$target" ]; then
        "$target" || FAILED=1
    else
        echo "Test not found: $target" >&2
        FAILED=1
    fi
else
    for t in ./test_*; do
        [ -x "$t" ] || continue
        "$t" || FAILED=1
    done
fi

exit $FAILED
