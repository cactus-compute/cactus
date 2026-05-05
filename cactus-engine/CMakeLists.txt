cmake_minimum_required(VERSION 3.10)
project(CactusEngine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES "arm64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -arch arm64 -march=armv8.2-a+fp16+simd+dotprod+i8mm -pthread -Wall -Wextra -pedantic -O3 -Wno-missing-field-initializers")
    add_compile_definitions(
        __ARM_NEON=1
        __ARM_FEATURE_FP16_VECTOR_ARITHMETIC=1
        __ARM_FEATURE_DOTPROD=1
        __ARM_FEATURE_MATMUL_INT8=1
        ACCELERATE_NEW_LAPACK
    )
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.2-a+fp16+simd+dotprod+i8mm -pthread -Wall -Wextra -pedantic -O3 -Wno-missing-field-initializers")
    add_compile_definitions(
        __ARM_NEON=1
        __ARM_FEATURE_FP16_VECTOR_ARITHMETIC=1
        __ARM_FEATURE_DOTPROD=1
        __ARM_FEATURE_MATMUL_INT8=1
    )
endif()

# --- Depend on cactus-graph (which depends on cactus-kernels) ---
set(CACTUS_GRAPH_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../cactus-graph")
if(NOT TARGET cactus_graph)
    add_subdirectory(${CACTUS_GRAPH_DIR} ${CMAKE_CURRENT_BINARY_DIR}/cactus-graph)
endif()

# --- libcurl ---
set(CACTUS_CURL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/libs/curl" CACHE PATH "Path to vendored libcurl")

if(APPLE)
    set(CACTUS_CURL_INCLUDE_DIR "${CACTUS_CURL_ROOT}/include")
    set(CACTUS_CURL_LIBRARY_MACOS "${CACTUS_CURL_ROOT}/macos/libcurl.a")
    if(EXISTS "${CACTUS_CURL_INCLUDE_DIR}/curl/curl.h" AND EXISTS "${CACTUS_CURL_LIBRARY_MACOS}")
        set(CACTUS_HAS_CURL TRUE)
    else()
        set(CACTUS_HAS_CURL FALSE)
        message(WARNING "Vendored libcurl not found at ${CACTUS_CURL_ROOT}; cloud features disabled")
    endif()
else()
    find_package(CURL QUIET)
    set(CACTUS_HAS_CURL ${CURL_FOUND})
endif()

# --- Engine sources ---
set(ENGINE_SOURCES
    src/bpe.cpp
    src/sp.cpp
    src/constraints.cpp
    src/model.cpp
    src/index.cpp
    src/index_ffi.cpp
    src/log.cpp
    src/graph_ffi.cpp
    src/rag.cpp
    src/telemetry.cpp
    src/telemetry_impl.cpp
    src/cloud.cpp
    src/init.cpp
    src/embed.cpp
    src/complete.cpp
    src/transcribe.cpp
    src/tokenizer.cpp
    src/npu.cpp
)

set(MODEL_SOURCES
    models/gemma4/model_gemma4.cpp
    models/gemma4/model_gemma4_mm.cpp
    models/gemma4/model_gemma4_audio.cpp
    models/gemma4/model_gemma4_vision.cpp
)

if(APPLE)
    enable_language(OBJCXX)
    list(APPEND ENGINE_SOURCES src/npu_ane.mm)
    set_source_files_properties(src/npu_ane.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
endif()

add_library(cactus_engine STATIC ${ENGINE_SOURCES} ${MODEL_SOURCES})
set(CACTUS_KERNELS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../cactus-kernels")
target_include_directories(cactus_engine
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/models
            ${CACTUS_KERNELS_DIR}/src
)
target_link_libraries(cactus_engine PUBLIC cactus_graph)
set_target_properties(cactus_engine PROPERTIES POSITION_INDEPENDENT_CODE ON)

# curl
if(CACTUS_HAS_CURL)
    if(APPLE)
        target_link_libraries(cactus_engine PUBLIC ${CACTUS_CURL_LIBRARY_MACOS})
        target_include_directories(cactus_engine PUBLIC ${CACTUS_CURL_INCLUDE_DIR})
    else()
        target_link_libraries(cactus_engine PUBLIC ${CURL_LIBRARIES})
        target_include_directories(cactus_engine PRIVATE ${CURL_INCLUDE_DIRS})
    endif()
    target_compile_definitions(cactus_engine PRIVATE CACTUS_USE_CURL=1)
endif()

# Apple frameworks
if(APPLE)
    find_library(COREML_FRAMEWORK CoreML REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(SECURITY_FRAMEWORK Security REQUIRED)
    find_library(SYSTEMCONFIGURATION_FRAMEWORK SystemConfiguration REQUIRED)
    find_library(CFNETWORK_FRAMEWORK CFNetwork REQUIRED)
    target_link_libraries(cactus_engine PUBLIC
        ${COREML_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
        ${SECURITY_FRAMEWORK}
        ${SYSTEMCONFIGURATION_FRAMEWORK}
        ${CFNETWORK_FRAMEWORK}
    )
endif()

# Version
set(_CACTUS_VERSION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../CACTUS_VERSION")
if(EXISTS "${_CACTUS_VERSION_FILE}")
    file(READ "${_CACTUS_VERSION_FILE}" _CACTUS_VERSION_CONTENT)
    string(STRIP "${_CACTUS_VERSION_CONTENT}" CACTUS_VERSION_STR)
    target_compile_definitions(cactus_engine PRIVATE CACTUS_COMPILE_TIME_VERSION="${CACTUS_VERSION_STR}")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(cactus_engine PRIVATE -Wno-c99-extensions)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(cactus_engine PRIVATE -Wno-pedantic)
endif()
