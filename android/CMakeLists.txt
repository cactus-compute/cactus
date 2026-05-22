cmake_minimum_required(VERSION 3.10)
project(CactusAndroid LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Release CACHE STRING "")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if(NOT ${ANDROID_ABI} STREQUAL "arm64-v8a")
    message(FATAL_ERROR "Unsupported Android ABI: ${ANDROID_ABI}. Cactus only supports arm64-v8a.")
endif()

set(CACTUS_MBEDTLS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/mbedtls/${ANDROID_ABI}")
if(NOT EXISTS "${CACTUS_MBEDTLS_ROOT}")
    set(CACTUS_MBEDTLS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../libs/mbedtls/android/${ANDROID_ABI}")
endif()

if(NOT TARGET cactus_engine)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../cactus ${CMAKE_CURRENT_BINARY_DIR}/cactus)
endif()

set(_DEPS cactus_engine cactus_graph cactus_kernels)

add_library(cactus_jni SHARED ${CMAKE_CURRENT_SOURCE_DIR}/cactus_jni.cpp)

set_target_properties(cactus_jni PROPERTIES
    OUTPUT_NAME cactus
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

target_include_directories(cactus_jni PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../cactus
    ${CMAKE_CURRENT_SOURCE_DIR}/../cactus-engine
    ${CMAKE_CURRENT_SOURCE_DIR}/../cactus-engine/src
)

target_link_libraries(cactus_jni PRIVATE ${_DEPS})

find_library(LOG_LIB log)
target_link_libraries(cactus_jni PRIVATE ${LOG_LIB} android)

set(_MBEDTLS "${CACTUS_MBEDTLS_ROOT}/lib/libmbedtls.a")
set(_MBEDX509 "${CACTUS_MBEDTLS_ROOT}/lib/libmbedx509.a")
set(_MBEDCRYPTO "${CACTUS_MBEDTLS_ROOT}/lib/libmbedcrypto.a")
if(EXISTS "${_MBEDTLS}" AND EXISTS "${_MBEDX509}" AND EXISTS "${_MBEDCRYPTO}")
    target_link_libraries(cactus_jni PRIVATE "${_MBEDTLS}" "${_MBEDX509}" "${_MBEDCRYPTO}")
endif()

target_compile_definitions(cactus_jni PRIVATE PLATFORM_CPU_ONLY=1)

set(_CACTUS_VERSION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../CACTUS_VERSION")
if(EXISTS "${_CACTUS_VERSION_FILE}")
    file(READ "${_CACTUS_VERSION_FILE}" _CACTUS_VERSION_CONTENT)
    string(STRIP "${_CACTUS_VERSION_CONTENT}" CACTUS_VERSION_STR)
    target_compile_definitions(cactus_jni PRIVATE CACTUS_COMPILE_TIME_VERSION="${CACTUS_VERSION_STR}")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(cactus_jni PRIVATE CACTUS_ANDROID_ENABLE_LOGGING)
endif()

target_compile_options(cactus_jni PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden -ffunction-sections -fdata-sections)
target_link_options(cactus_jni PRIVATE -Wl,--gc-sections -Wl,--exclude-libs,ALL -flto -Wl,-z,max-page-size=16384)

set(_ARCHIVES $<TARGET_FILE:cactus_engine> $<TARGET_FILE:cactus_graph> $<TARGET_FILE:cactus_kernels>)
file(GENERATE OUTPUT ${CMAKE_BINARY_DIR}/combine_static.mri CONTENT
    "create ${CMAKE_BINARY_DIR}/lib/libcactus_static.a\naddlib $<TARGET_FILE:cactus_engine>\naddlib $<TARGET_FILE:cactus_graph>\naddlib $<TARGET_FILE:cactus_kernels>\nsave\nend\n"
)
add_custom_target(cactus_static ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/lib
    COMMAND ${CMAKE_AR} -M < ${CMAKE_BINARY_DIR}/combine_static.mri
    DEPENDS ${_DEPS}
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(CACTUS_ENGINE_DIR "${PROJECT_ROOT}/cactus-engine")

function(_target_link_cactus_android target_name)
    target_include_directories(${target_name} PRIVATE
        ${PROJECT_ROOT}
        ${PROJECT_ROOT}/cactus
        ${CACTUS_ENGINE_DIR}
        ${CACTUS_ENGINE_DIR}/src
    )
    target_compile_definitions(${target_name} PRIVATE PLATFORM_CPU_ONLY=1)
    target_compile_options(${target_name} PRIVATE -O3 -fvisibility=hidden -fvisibility-inlines-hidden)
    target_link_libraries(${target_name} PRIVATE ${_DEPS} ${LOG_LIB} android)
    if(EXISTS "${_MBEDTLS}" AND EXISTS "${_MBEDX509}" AND EXISTS "${_MBEDCRYPTO}")
        target_link_libraries(${target_name} PRIVATE "${_MBEDTLS}" "${_MBEDX509}" "${_MBEDCRYPTO}")
    endif()
    target_link_options(${target_name} PRIVATE -Wl,-z,max-page-size=16384)
endfunction()

add_executable(cactus_llm_bench
    ${PROJECT_ROOT}/tests/android/cactus_llm_bench.cpp
)
_target_link_cactus_android(cactus_llm_bench)
set_target_properties(cactus_llm_bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_executable(native_transcribe_json
    ${PROJECT_ROOT}/experiments/matrix/native_transcribe_json.cpp
)
_target_link_cactus_android(native_transcribe_json)
set_target_properties(native_transcribe_json PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_executable(cq4_kernel_probe
    ${PROJECT_ROOT}/tests/android/cq4_kernel_probe.cpp
)
target_compile_options(cq4_kernel_probe PRIVATE -O3 -march=armv8.2-a+fp16+simd+dotprod+i8mm)
target_link_options(cq4_kernel_probe PRIVATE -Wl,-z,max-page-size=16384)
set_target_properties(cq4_kernel_probe PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

option(CACTUS_BUILD_XNNPACK_QC4_PROBE "Build optional XNNPACK QC4 probe" OFF)
set(EXECUTORCH_ANDROID_BUILD "${PROJECT_ROOT}/../third_party/executorch/cmake-out-android-arm64-v8a")
if(CACTUS_BUILD_XNNPACK_QC4_PROBE AND EXISTS "${EXECUTORCH_ANDROID_BUILD}/include/xnnpack.h")
    add_executable(xnnpack_qc4_probe
        ${PROJECT_ROOT}/tests/android/xnnpack_qc4_probe.cpp
    )
    target_include_directories(xnnpack_qc4_probe PRIVATE
        ${EXECUTORCH_ANDROID_BUILD}/include
    )
    target_compile_options(xnnpack_qc4_probe PRIVATE -O3 -march=armv8.2-a+fp16+simd+dotprod+i8mm)
    target_link_libraries(xnnpack_qc4_probe PRIVATE
        ${EXECUTORCH_ANDROID_BUILD}/lib/libXNNPACK.a
        ${EXECUTORCH_ANDROID_BUILD}/lib/libxnnpack-microkernels-prod.a
        ${EXECUTORCH_ANDROID_BUILD}/lib/libkleidiai.a
        ${EXECUTORCH_ANDROID_BUILD}/lib/libpthreadpool.a
        ${EXECUTORCH_ANDROID_BUILD}/lib/libcpuinfo.a
        ${LOG_LIB}
        android
    )
    target_link_options(xnnpack_qc4_probe PRIVATE -Wl,-z,max-page-size=16384)
    set_target_properties(xnnpack_qc4_probe PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endif()
