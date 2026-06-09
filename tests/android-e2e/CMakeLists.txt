cmake_minimum_required(VERSION 3.16)
project(CactusE2EAndroid LANGUAGES CXX)

# Minimal cactus-only e2e decode-tps benchmark for Android (arm64-v8a).
# Reuses the e2e bench sources from ../bench and links the prebuilt
# Android libcactus.a produced by ../../android/build.sh.

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT ANDROID_ABI STREQUAL "arm64-v8a")
    message(FATAL_ERROR "Only arm64-v8a is supported (got: ${ANDROID_ABI})")
endif()

set(REPO_ROOT   ${CMAKE_CURRENT_SOURCE_DIR}/../..)
set(BENCH_DIR   ${CMAKE_CURRENT_SOURCE_DIR}/../bench)
set(CACTUS_ROOT ${REPO_ROOT}/cactus)

set(CACTUS_LIB     ${REPO_ROOT}/android/libcactus.a)
set(CURL_LIB       ${REPO_ROOT}/libs/curl/android/${ANDROID_ABI}/libcurl.a)
set(MBEDTLS_ROOT   ${REPO_ROOT}/android/mbedtls/${ANDROID_ABI})

foreach(_lib "${CACTUS_LIB}" "${CURL_LIB}"
             "${MBEDTLS_ROOT}/lib/libmbedtls.a"
             "${MBEDTLS_ROOT}/lib/libmbedx509.a"
             "${MBEDTLS_ROOT}/lib/libmbedcrypto.a")
    if(NOT EXISTS "${_lib}")
        message(FATAL_ERROR "Missing required library: ${_lib} (run ../../android/build.sh first)")
    endif()
endforeach()

# libcactus.a's kernel_matmul.o is compiled with -DCACTUS_COMPILE_I8MM/SME2 and
# references the i8mm/sme2 kernels, but CMake does not archive OBJECT-library
# sources into a STATIC lib, so those kernels are missing from libcactus.a.
# Compile them directly into the executable (with their own -march flags) so the
# fast prefill GEMM / decode GEMV paths resolve and actually run on-device.
set(KERNEL_DIR ${CACTUS_ROOT}/kernel)

add_library(bench_i8mm OBJECT ${KERNEL_DIR}/kernel_i8mm.cpp)
target_include_directories(bench_i8mm PRIVATE ${CACTUS_ROOT} ${KERNEL_DIR})
target_compile_options(bench_i8mm PRIVATE
    -O3 -DNDEBUG -march=armv8.2-a+fp16+simd+dotprod+i8mm -pthread)
target_compile_definitions(bench_i8mm PRIVATE
    CACTUS_COMPILE_I8MM __ARM_FEATURE_MATMUL_INT8=1 PLATFORM_CPU_ONLY=1)
set_target_properties(bench_i8mm PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(bench_sme2 OBJECT ${KERNEL_DIR}/kernel_sme2.cpp)
target_include_directories(bench_sme2 PRIVATE ${CACTUS_ROOT} ${KERNEL_DIR})
target_compile_options(bench_sme2 PRIVATE
    -O3 -DNDEBUG -march=armv9.2-a+nosve+nosve2+sme2 -pthread)
target_compile_definitions(bench_sme2 PRIVATE
    CACTUS_COMPILE_SME2 PLATFORM_CPU_ONLY=1)
set_target_properties(bench_sme2 PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(e2e_bench
    ${BENCH_DIR}/e2e_bench.cpp
    ${BENCH_DIR}/e2e_driver.cpp
    ${BENCH_DIR}/e2e_backend_cactus.cpp
    $<TARGET_OBJECTS:bench_i8mm>
    $<TARGET_OBJECTS:bench_sme2>
)

target_include_directories(e2e_bench PRIVATE
    ${BENCH_DIR}
    ${CACTUS_ROOT}
    ${REPO_ROOT}/libs/curl/include
    ${MBEDTLS_ROOT}/include
)

target_compile_options(e2e_bench PRIVATE
    -O3 -DNDEBUG -march=armv8.2-a+fp16+simd+dotprod -pthread
)

find_library(LOG_LIB log)

# Link order: exe objects -> cactus -> curl -> mbedtls.
target_link_libraries(e2e_bench PRIVATE
    ${CACTUS_LIB}
    ${CURL_LIB}
    ${MBEDTLS_ROOT}/lib/libmbedtls.a
    ${MBEDTLS_ROOT}/lib/libmbedx509.a
    ${MBEDTLS_ROOT}/lib/libmbedcrypto.a
    ${LOG_LIB}
    android
)
