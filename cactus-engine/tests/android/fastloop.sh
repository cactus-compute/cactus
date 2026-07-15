#!/bin/bash
# Tight kernel-iteration loop: rebuild test_vulkan, push, run the on-device unit
# suite (needs no model bundle — real-weight tests skip when CACTUS_TEST_MODEL
# is unset). Typical wall time: ~60s build + ~3s device run.
#   usage: ./fastloop.sh            # run full unit suite
#          CACTUS_TEST_MODEL=/data/local/tmp/cactus_models/<bundle> ./fastloop.sh
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
export PATH="$HOME/Library/Android/sdk/platform-tools:$PATH"
cmake --build "$BUILD_DIR" -j 10 --target test_vulkan 2>&1 | grep -E "error" -B2 && exit 1
adb push "$BUILD_DIR/test_vulkan" /data/local/tmp/cactus_tests/ > /dev/null
adb shell "cd /data/local/tmp/cactus_tests && chmod +x test_vulkan && \
  export CACTUS_NO_CLOUD_TELE=1 && export CACTUS_TEST_BACKEND=vulkan && \
  ${CACTUS_TEST_MODEL:+export CACTUS_TEST_MODEL=$CACTUS_TEST_MODEL &&} ./test_vulkan"
