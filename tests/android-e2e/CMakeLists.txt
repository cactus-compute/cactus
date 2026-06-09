cmake_minimum_required(VERSION 3.16)
project(CactusE2EAndroid LANGUAGES CXX)

# Minimal cactus-only e2e decode-tps benchmark for Android (arm64-v8a).
# Reuses the e2e bench sources from ../bench.
#
# The CQ4 transpiled bundles (precision code 6, group-128, packed 4-bit) are
# only readable by the cactus-v2 engine; this repo's engine predates that
# format and segfaults loading them. Link against the prebuilt cactus-v2
# Android archives instead (same C FFI: cactus_init/cactus_complete/...).

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT ANDROID_ABI STREQUAL "arm64-v8a")
    message(FATAL_ERROR "Only arm64-v8a is supported (got: ${ANDROID_ABI})")
endif()

set(REPO_ROOT   ${CMAKE_CURRENT_SOURCE_DIR}/../..)
set(BENCH_DIR   ${CMAKE_CURRENT_SOURCE_DIR}/../bench)
set(CACTUS_ROOT ${REPO_ROOT}/cactus)

set(CACTUS_V2_ROOT "$ENV{HOME}/Documents/Desert/cactus-v2" CACHE PATH "cactus-v2 repo root")
set(V2_BUILD       ${CACTUS_V2_ROOT}/android/build)
# Archive layout changed upstream: post-June builds nest module archives under
# cactus-engine/, older builds put four flat archives in android/build.
if(EXISTS ${V2_BUILD}/cactus-engine/libcactus_engine_core.a)
    set(V2_LIBS
        ${V2_BUILD}/libcactus_engine.a
        ${V2_BUILD}/cactus-engine/libcactus_engine_core.a
        ${V2_BUILD}/cactus-engine/cactus-graph/libcactus_graph.a
        ${V2_BUILD}/cactus-engine/cactus-graph/cactus-kernels/libcactus_kernels.a)
else()
    set(V2_LIBS
        ${V2_BUILD}/libcactus.a
        ${V2_BUILD}/libcactus_engine.a
        ${V2_BUILD}/libcactus_graph.a
        ${V2_BUILD}/libcactus_kernels.a)
endif()
set(CURL_LIB       ${CACTUS_V2_ROOT}/cactus-engine/libs/curl/android/${ANDROID_ABI}/libcurl.a)
set(MBEDTLS_ROOT   ${CACTUS_V2_ROOT}/android/mbedtls/${ANDROID_ABI})

foreach(_lib ${V2_LIBS} "${CURL_LIB}"
             "${MBEDTLS_ROOT}/lib/libmbedtls.a"
             "${MBEDTLS_ROOT}/lib/libmbedx509.a"
             "${MBEDTLS_ROOT}/lib/libmbedcrypto.a")
    if(NOT EXISTS "${_lib}")
        message(FATAL_ERROR "Missing required library: ${_lib} (build cactus-v2/android first)")
    endif()
endforeach()

add_executable(e2e_bench
    ${BENCH_DIR}/e2e_bench.cpp
    ${BENCH_DIR}/e2e_driver.cpp
    ${BENCH_DIR}/e2e_backend_cactus.cpp
)

target_include_directories(e2e_bench PRIVATE
    ${BENCH_DIR}
    ${CACTUS_ROOT}
)

target_compile_options(e2e_bench PRIVATE
    -O3 -DNDEBUG -march=armv8.2-a+fp16+simd+dotprod -pthread
)

find_library(LOG_LIB log)

# start/end-group: the v2 module archives have circular references.
target_link_libraries(e2e_bench PRIVATE
    -Wl,--start-group
    ${V2_LIBS}
    -Wl,--end-group
    ${CURL_LIB}
    ${MBEDTLS_ROOT}/lib/libmbedtls.a
    ${MBEDTLS_ROOT}/lib/libmbedx509.a
    ${MBEDTLS_ROOT}/lib/libmbedcrypto.a
    ${LOG_LIB}
    android
)
