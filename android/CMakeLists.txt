cmake_minimum_required(VERSION 3.10)
project(Cactus LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(SOURCE_DIR ${CMAKE_SOURCE_DIR}/../cactus)
set(CACTUS_CURL_ROOT "${CMAKE_SOURCE_DIR}/../libs/curl" CACHE PATH "Path to vendored libcurl artifacts")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
    message(STATUS "CMAKE_BUILD_TYPE was not set, defaulting to Release.")
endif()

file(GLOB ENGINE_SOURCES "${SOURCE_DIR}/engine/*.cpp")
file(GLOB GRAPH_SOURCES "${SOURCE_DIR}/graph/*.cpp")
file(GLOB KERNEL_SOURCES "${SOURCE_DIR}/kernel/*.cpp")
file(GLOB MODEL_SOURCES "${SOURCE_DIR}/models/*.cpp")
file(GLOB FFI_SOURCES "${SOURCE_DIR}/ffi/*.cpp")
file(GLOB TELEMETRY_SOURCES "${SOURCE_DIR}/telemetry/*.cpp")
set(NPU_SOURCES "${SOURCE_DIR}/npu/npu.cpp")

set(HEADER_FILES
    ${SOURCE_DIR}/cactus.h
    ${SOURCE_DIR}/engine/engine.h
    ${SOURCE_DIR}/graph/graph.h
    ${SOURCE_DIR}/kernel/kernel.h
    ${SOURCE_DIR}/kernel/kernel_utils.h
    ${SOURCE_DIR}/ffi/cactus_ffi.h
    ${SOURCE_DIR}/ffi/cactus_utils.h
    ${SOURCE_DIR}/models/model.h
    ${SOURCE_DIR}/npu/npu.h
)

set(COMMON_SOURCES
    ${KERNEL_SOURCES}
    ${GRAPH_SOURCES}
    ${ENGINE_SOURCES}
    ${FFI_SOURCES}
    ${MODEL_SOURCES}
    ${TELEMETRY_SOURCES}
    ${NPU_SOURCES}
)

if(NOT COMMON_SOURCES)
    message(FATAL_ERROR "No source files found! Check SOURCE_DIR: ${SOURCE_DIR}")
endif()

find_library(LOG_LIB log)

add_library(
    cactus
    SHARED
    ${COMMON_SOURCES}
    ${CMAKE_SOURCE_DIR}/cactus_jni.cpp
)

add_library(
    cactus_static
    STATIC
    ${COMMON_SOURCES}
)

set(COMMON_INCLUDE_DIRS
    ${SOURCE_DIR}
    ${SOURCE_DIR}/engine
    ${SOURCE_DIR}/graph
    ${SOURCE_DIR}/kernel
    ${SOURCE_DIR}/ffi
    ${SOURCE_DIR}/models
    ${SOURCE_DIR}/npu
    ${SOURCE_DIR}/telemetry
)

target_include_directories(cactus PRIVATE ${COMMON_INCLUDE_DIRS})
target_include_directories(cactus_static PRIVATE ${COMMON_INCLUDE_DIRS})

target_link_libraries(cactus ${LOG_LIB} android)

set(CACTUS_CURL_INCLUDE_DIR "${CACTUS_CURL_ROOT}/include")
set(CACTUS_CURL_LIBRARY_ANDROID "${CACTUS_CURL_ROOT}/android/${ANDROID_ABI}/libcurl.a")
set(CACTUS_MBEDTLS_ROOT "${CMAKE_SOURCE_DIR}/mbedtls/${ANDROID_ABI}")
if(NOT EXISTS "${CACTUS_MBEDTLS_ROOT}")
    set(CACTUS_MBEDTLS_ROOT "${CMAKE_SOURCE_DIR}/../libs/mbedtls/android/${ANDROID_ABI}")
endif()
set(CACTUS_MBEDTLS_INCLUDE_DIR "${CACTUS_MBEDTLS_ROOT}/include")
set(CACTUS_MBEDTLS_LIBRARY "${CACTUS_MBEDTLS_ROOT}/lib/libmbedtls.a")
set(CACTUS_MBEDX509_LIBRARY "${CACTUS_MBEDTLS_ROOT}/lib/libmbedx509.a")
set(CACTUS_MBEDCRYPTO_LIBRARY "${CACTUS_MBEDTLS_ROOT}/lib/libmbedcrypto.a")
if(EXISTS "${CACTUS_CURL_INCLUDE_DIR}/curl/curl.h" AND EXISTS "${CACTUS_CURL_LIBRARY_ANDROID}")
    message(STATUS "Using vendored libcurl for Android: ${CACTUS_CURL_LIBRARY_ANDROID}")

    if(NOT (EXISTS "${CACTUS_MBEDTLS_LIBRARY}" AND EXISTS "${CACTUS_MBEDX509_LIBRARY}" AND EXISTS "${CACTUS_MBEDCRYPTO_LIBRARY}"))
        message(FATAL_ERROR "Vendored mbedTLS libraries not found. Expected: ${CACTUS_MBEDTLS_LIBRARY}, ${CACTUS_MBEDX509_LIBRARY}, ${CACTUS_MBEDCRYPTO_LIBRARY}")
    endif()

    target_include_directories(cactus PRIVATE "${CACTUS_CURL_INCLUDE_DIR}")
    target_include_directories(cactus_static PRIVATE "${CACTUS_CURL_INCLUDE_DIR}")
    target_include_directories(cactus PRIVATE "${CACTUS_MBEDTLS_INCLUDE_DIR}")
    target_include_directories(cactus_static PRIVATE "${CACTUS_MBEDTLS_INCLUDE_DIR}")
    target_compile_definitions(cactus PRIVATE CACTUS_USE_CURL=1)
    target_compile_definitions(cactus_static PRIVATE CACTUS_USE_CURL=1)

    target_link_libraries(cactus "${CACTUS_CURL_LIBRARY_ANDROID}")
    target_link_libraries(cactus_static "${CACTUS_CURL_LIBRARY_ANDROID}")
    target_link_libraries(cactus "${CACTUS_MBEDTLS_LIBRARY}" "${CACTUS_MBEDX509_LIBRARY}" "${CACTUS_MBEDCRYPTO_LIBRARY}")
    target_link_libraries(cactus_static "${CACTUS_MBEDTLS_LIBRARY}" "${CACTUS_MBEDX509_LIBRARY}" "${CACTUS_MBEDCRYPTO_LIBRARY}")
else()
    message(FATAL_ERROR "Vendored Android libcurl not found. Expected: ${CACTUS_CURL_INCLUDE_DIR}/curl/curl.h and ${CACTUS_CURL_LIBRARY_ANDROID}")
endif()

set(COMMON_COMPILE_DEFINITIONS
    PLATFORM_CPU_ONLY=1
)

target_compile_definitions(cactus PRIVATE ${COMMON_COMPILE_DEFINITIONS})
target_compile_definitions(cactus_static PRIVATE ${COMMON_COMPILE_DEFINITIONS})

set(COMMON_COMPILE_OPTIONS
    -pthread
    -Wall
    -Wextra
    -pedantic
    -Wno-c++20-designator 
    -Wno-missing-field-initializers 
    -fomit-frame-pointer
    -funroll-loops
    -ftree-vectorize
)

target_compile_options(cactus PRIVATE ${COMMON_COMPILE_OPTIONS})
target_compile_options(cactus_static PRIVATE ${COMMON_COMPILE_OPTIONS})

if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    target_compile_options(cactus PRIVATE -DCACTUS_ANDROID_ENABLE_LOGGING)
    target_compile_options(cactus_static PRIVATE -DCACTUS_ANDROID_ENABLE_LOGGING)
endif ()

set(OPTIMIZATION_OPTIONS
    -O3
    -DNDEBUG
    -ffunction-sections
    -fdata-sections
)

target_compile_options(cactus PRIVATE ${OPTIMIZATION_OPTIONS})
target_compile_options(cactus_static PRIVATE ${OPTIMIZATION_OPTIONS})

target_compile_options(cactus PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)

if (${ANDROID_ABI} STREQUAL "arm64-v8a")
    set(ARM_OPTIONS
        -march=armv8.2-a+fp16+simd+dotprod
    )
    set(ARM_DEFINITIONS
        __ARM_NEON=1
        __ARM_FEATURE_FP16_VECTOR_ARITHMETIC=1
        __ARM_FEATURE_DOTPROD=1
    )
    
    target_compile_options(cactus PRIVATE ${ARM_OPTIONS})
    target_compile_options(cactus_static PRIVATE ${ARM_OPTIONS})
    
    target_compile_definitions(cactus PRIVATE ${ARM_DEFINITIONS})
    target_compile_definitions(cactus_static PRIVATE ${ARM_DEFINITIONS})
else()
    message(FATAL_ERROR "Unsupported Android ABI: ${ANDROID_ABI}. Cactus only supports arm64-v8a due to ARM NEON requirements.")
endif()

target_link_options(cactus PRIVATE -Wl,--gc-sections)
target_link_options(cactus PRIVATE -Wl,--exclude-libs,ALL)
target_link_options(cactus PRIVATE -flto)
target_link_options(cactus PRIVATE -Wl,-z,max-page-size=16384)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
