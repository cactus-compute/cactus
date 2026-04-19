#!/usr/bin/env bash
# Build CactusSquarePackage host-side shared library for aarch64 Linux
# (Qualcomm Device Cloud target). Runs from WSL Ubuntu 24.04 aarch64.

set -euo pipefail
cd "$(dirname "$0")"

QAIRT=/mnt/c/Qualcomm/AIStack/QAIRT/2.31.0.250130
QNN_INC=$QAIRT/include/QNN
QNN_LINUX_LIB=$QAIRT/lib/aarch64-oe-linux-gcc11.2
HEXNATIVE_INC=/mnt/c/Qualcomm/Hexagon_SDK/6.4.0.2/tools/HEXAGON_Tools/19.0.04/Tools/libnative/include

OUT=build-wsl-linux-aarch64
mkdir -p "$OUT"

# Force-include the log stub we already use to neutralize the MSVC-hostile
# FILE_BASENAME macros. On Linux g++ has __FILE_NAME__ natively so this may
# not be strictly required, but the stub also guards against other
# host-build gotchas and keeps behavior consistent with the MSVC build.
# Actually on Linux g++ with __FILE_NAME__ the original macros work,
# so we won't force-include log_stub.h here.

CXXFLAGS=(
  -std=c++17
  -O2
  -fPIC
  -Wall -Wno-reorder -Wno-missing-braces -Wno-unused-function
  -Wno-unused-parameter -Wno-unused-variable -Wno-sign-compare
  -Wno-ignored-qualifiers -Wno-missing-field-initializers
  -Wno-stringop-overread
  -fvisibility=default
  -I"$QNN_INC"
  -I"$HEXNATIVE_INC"
  -D__HVXDBL__ -DUSE_OS_LINUX
  -DTHIS_PKG_NAME=CactusSquarePackage
  -DQNN_API="__attribute__((visibility(\"default\")))"
  -D__QAIC_HEADER_EXPORT="__attribute__((visibility(\"default\")))"
)

echo "== compile Interface =="
g++ "${CXXFLAGS[@]}" -c src/CactusSquarePackageInterface.cpp -o "$OUT/Interface.o"

echo "== compile Square =="
g++ "${CXXFLAGS[@]}" -c src/ops/Square.cpp -o "$OUT/Square.o"

echo "== link shared library =="
g++ -shared -fPIC -o "$OUT/libQnnCactusSquarePackage.so" \
  "$OUT/Interface.o" "$OUT/Square.o" \
  -L"$QNN_LINUX_LIB" -lQnnHtpPrepare -lQnnHtp \
  -Wl,-rpath,"$QNN_LINUX_LIB"

echo "== verify =="
file "$OUT/libQnnCactusSquarePackage.so"
echo "-- exports of our InterfaceProvider --"
nm -D "$OUT/libQnnCactusSquarePackage.so" | grep -E "InterfaceProvider|CactusSquare" | head
echo "-- undefined symbols (should resolve against QnnHtpPrepare.so) --"
nm -D "$OUT/libQnnCactusSquarePackage.so" | grep -E "^\s+U " | grep -E "hnnx|Allocator|FakeAllocator|Tensor" | wc -l
echo "DONE"
