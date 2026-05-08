cmake_minimum_required(VERSION 3.16)
project(CactusTests LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

option(WITH_LITERT "Build with LiteRT for matmul benchmarks (requires network on first build)" OFF)
option(WITH_GGML "Build with GGML for Q4_0 matmul benchmarks" OFF)
option(WITH_EXECUTORCH "Build with Executorch for matmul benchmarks" OFF)
option(WITH_MLX "Build with MLX for matmul benchmarks (Apple-only)" OFF)
option(WITH_MLC "Build with MLC for matmul benchmarks" OFF)
option(WITH_ONNXRT "Build with ONNX Runtime for matmul benchmarks" OFF)
option(WITH_LLAMA_CPP "Build with llama.cpp for E2E benchmarks" OFF)
option(WITH_ONNXRT_GENAI "Build with ONNX Runtime GenAI for E2E benchmarks" OFF)
option(WITH_E2E_EXECUTORCH "Build with ExecuTorch for E2E benchmarks" OFF)
option(WITH_E2E_LITERT "Build with LiteRT-LM for E2E benchmarks" OFF)

if(WITH_LITERT)
    set(LITERT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/litert)

    include(FetchContent)
    set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        flatbuffers
        GIT_REPOSITORY https://github.com/google/flatbuffers
        GIT_TAG v24.12.23
        GIT_SHALLOW TRUE
        SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers"
    )
    FetchContent_Populate(flatbuffers)
    add_subdirectory("${flatbuffers_SOURCE_DIR}" "${flatbuffers_BINARY_DIR}" EXCLUDE_FROM_ALL)
    include(ExternalProject)
    ExternalProject_Add(flatbuffers-flatc
        PREFIX ${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-flatc
        SOURCE_DIR ${flatbuffers_SOURCE_DIR}
        CMAKE_ARGS -DFLATBUFFERS_BUILD_TESTS=OFF
                   -DFLATBUFFERS_BUILD_FLATLIB=OFF
                   -DFLATBUFFERS_STATIC_FLATC=OFF
                   -DFLATBUFFERS_BUILD_FLATHASH=OFF
        INSTALL_COMMAND ""
        EXCLUDE_FROM_ALL TRUE
    )
    set(FLATBUFFERS_FLATC_EXECUTABLE
        "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-flatc/src/flatbuffers-flatc-build/flatc"
        CACHE STRING "" FORCE)

    set(TFLITE_ENABLE_XNNPACK OFF CACHE BOOL "" FORCE)
    set(TFLITE_ENABLE_GPU OFF CACHE BOOL "" FORCE)
    set(TFLITE_ENABLE_NNAPI OFF CACHE BOOL "" FORCE)
    set(TFLITE_ENABLE_EXTERNAL_DELEGATE OFF CACHE BOOL "" FORCE)
    set(TFLITE_ENABLE_RUY ON CACHE BOOL "" FORCE)
    add_subdirectory(${LITERT_DIR}/tflite ${CMAKE_CURRENT_BINARY_DIR}/tflite EXCLUDE_FROM_ALL)

    add_library(litert_kernels STATIC
        ${LITERT_DIR}/tflite/kernels/internal/optimized/neon_tensor_utils.cc
        ${LITERT_DIR}/tflite/kernels/internal/optimized/cpu_check.cc
        ${LITERT_DIR}/tflite/kernels/internal/reference/portable_tensor_utils.cc
        ${LITERT_DIR}/tflite/kernels/internal/common.cc
        ${LITERT_DIR}/tflite/kernels/internal/quantization_util.cc
        ${LITERT_DIR}/tflite/kernels/cpu_backend_context.cc
        ${LITERT_DIR}/tflite/kernels/internal/optimized/4bit/neon_fully_connected.cc
        ${LITERT_DIR}/tflite/kernels/internal/optimized/4bit/neon_fully_connected_aarch64_sdot.cc
        ${LITERT_DIR}/tflite/kernels/internal/optimized/4bit/neon_fully_connected_aarch64_nosdot.cc
        ${LITERT_DIR}/tflite/kernels/internal/optimized/4bit/fully_connected_reference.cc
    )
    target_include_directories(litert_kernels PUBLIC
        ${LITERT_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}/tflite/tensorflow-src
        ${CMAKE_CURRENT_BINARY_DIR}/cpuinfo
    )
    target_link_libraries(litert_kernels PUBLIC
        Eigen3::Eigen
        ruy::ruy
        gemmlowp::gemmlowp
        cpuinfo::cpuinfo
    )
    target_compile_options(litert_kernels PRIVATE -O3 -DTFLITE_WITH_RUY -DFC_4BIT_NEON)
    message(STATUS "LiteRT enabled from ${LITERT_DIR}")
endif()

if(WITH_GGML)
    set(GGML_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/ggml)
    set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    add_subdirectory(${GGML_DIR} ${CMAKE_CURRENT_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)
    message(STATUS "GGML enabled from ${GGML_DIR}")
endif()

if(WITH_MLX)
    if(NOT APPLE)
        message(FATAL_ERROR "MLX requires macOS (Metal framework)")
    endif()
    set(MLX_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/mlx)
    set(MLX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(MLX_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(MLX_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
    add_subdirectory(${MLX_DIR} ${CMAKE_CURRENT_BINARY_DIR}/mlx EXCLUDE_FROM_ALL)
    message(STATUS "MLX enabled from ${MLX_DIR}")
endif()

if(WITH_EXECUTORCH)
    include(FetchContent)
    set(XNNPACK_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(XNNPACK_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
    set(XNNPACK_LIBRARY_TYPE "static" CACHE STRING "" FORCE)
    FetchContent_Declare(
        xnnpack
        GIT_REPOSITORY https://github.com/google/XNNPACK
        GIT_TAG master
        GIT_SHALLOW TRUE
        SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/xnnpack-src"
    )
    FetchContent_MakeAvailable(xnnpack)
    message(STATUS "Executorch XNNPACK enabled")
endif()

function(cactus_require_vendored_curl include_dir library_path platform_label)
    if(NOT EXISTS "${include_dir}/curl/curl.h" OR NOT EXISTS "${library_path}")
        message(WARNING "==============================================================================")
        message(WARNING " Vendored libcurl is required for ${platform_label} tests but is missing.")
        message(WARNING " Expected header: ${include_dir}/curl/curl.h")
        message(WARNING " Expected library: ${library_path}")
        message(WARNING " Populate libs/curl from your local curl build before running cactus test.")
        message(WARNING "==============================================================================")
        message(FATAL_ERROR "Missing vendored libcurl artifacts for ${platform_label} tests")
    endif()
endfunction()

if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES "arm64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -arch arm64 -march=armv8.2-a+fp16+simd+dotprod+i8mm -pthread -Wall -Wextra -pedantic -O3")
    set(CACTUS_CURL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../libs/curl" CACHE PATH "Path to vendored libcurl artifacts")
    set(CACTUS_CURL_INCLUDE_DIR "${CACTUS_CURL_ROOT}/include")
    set(CACTUS_CURL_LIBRARY_MACOS "${CACTUS_CURL_ROOT}/macos/libcurl.a")
    cactus_require_vendored_curl("${CACTUS_CURL_INCLUDE_DIR}" "${CACTUS_CURL_LIBRARY_MACOS}" "macOS")
else()
    find_package(CURL)
endif()

set(CACTUS_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../cactus)
set(CACTUS_BUILD_DIR ${CACTUS_ROOT}/build)

find_library(CACTUS_LIB
    NAMES cactus
    HINTS ${CACTUS_BUILD_DIR} ${CACTUS_BUILD_DIR}/lib
    REQUIRED
)

set(MMAN_LIB "")
if(NOT APPLE)
    find_library(MMAN_LIB mman)
    if(MMAN_LIB)
        message(STATUS "Found libmman: ${MMAN_LIB}")
    else()
        message(STATUS "libmman not found; assuming mmap symbols are provided by the C runtime")
    endif()
else()
    find_library(COREML_FRAMEWORK CoreML REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
    find_library(SECURITY_FRAMEWORK Security REQUIRED)
    find_library(SYSTEMCONFIGURATION_FRAMEWORK SystemConfiguration REQUIRED)
    find_library(CFNETWORK_FRAMEWORK CFNetwork REQUIRED)
endif()

find_package(SDL2)
if(SDL2_FOUND)
    message(STATUS "Found SDL2: ${SDL2_LIBRARIES}")
else()
    message(STATUS "SDL2 not found; microphone tests will be skipped")
endif()

add_library(bench_lib OBJECT
    bench/bench_common.cpp
    bench/bench_driver.cpp
    bench/backend_cactus.cpp
    bench/backend_executorch.cpp
    bench/backend_mlx.cpp
    bench/backend_mlc.cpp
    bench/backend_onnxrt.cpp
)
target_include_directories(bench_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CACTUS_ROOT}
)
target_link_libraries(bench_lib PUBLIC ${CACTUS_LIB})

if(WITH_LITERT)
    target_sources(bench_lib PRIVATE
        bench/backend_litert.cpp
    )
    target_link_libraries(bench_lib PUBLIC litert_kernels)
    target_compile_definitions(bench_lib PRIVATE FC_4BIT_NEON)
endif()

if(WITH_GGML)
    target_sources(bench_lib PRIVATE bench/backend_ggml.cpp)
    target_link_libraries(bench_lib PUBLIC ggml)
endif()

if(WITH_EXECUTORCH)
    target_link_libraries(bench_lib PUBLIC XNNPACK)
    target_compile_definitions(bench_lib PRIVATE WITH_EXECUTORCH)
endif()

if(WITH_MLX)
    target_link_libraries(bench_lib PUBLIC mlx)
    target_compile_definitions(bench_lib PRIVATE WITH_MLX)
endif()

if(WITH_MLC)
    set(MLC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/mlc)
    set(TVM_DIR ${MLC_DIR}/3rdparty/tvm)
    set(TVM_FFI_DIR ${TVM_DIR}/3rdparty/tvm-ffi)
    find_library(TVM_RUNTIME_LIB tvm_runtime PATHS ${TVM_DIR}/build/lib ${TVM_DIR}/build REQUIRED)
    find_library(TVM_FFI_LIB tvm_ffi PATHS ${TVM_DIR}/build/lib ${TVM_DIR}/build REQUIRED)
    target_include_directories(bench_lib PRIVATE
        ${TVM_DIR}/include
        ${TVM_FFI_DIR}/include
        ${TVM_FFI_DIR}/3rdparty/dlpack/include
        ${TVM_DIR}/3rdparty/dmlc-core/include)
    target_link_libraries(bench_lib PUBLIC ${TVM_RUNTIME_LIB} ${TVM_FFI_LIB})
    target_compile_definitions(bench_lib PRIVATE WITH_MLC)
    message(STATUS "MLC/TVM runtime enabled from ${MLC_DIR}")
endif()

if(WITH_ONNXRT)
    set(ONNXRT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/onnxruntime)
    find_library(ONNXRT_LIB onnxruntime PATHS ${ONNXRT_DIR}/lib REQUIRED)
    target_include_directories(bench_lib PRIVATE ${ONNXRT_DIR}/include)
    target_link_libraries(bench_lib PUBLIC ${ONNXRT_LIB})
    target_compile_definitions(bench_lib PRIVATE WITH_ONNXRT)
    message(STATUS "ONNX Runtime enabled from ${ONNXRT_DIR}")
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench/matmul_bench.cpp")
    add_executable(matmul_bench bench/matmul_bench.cpp test_utils.cpp)
    target_link_libraries(matmul_bench PRIVATE bench_lib)
    target_include_directories(matmul_bench PRIVATE ${CACTUS_ROOT})
    if(WITH_MLC)
        target_link_libraries(matmul_bench PRIVATE ${TVM_RUNTIME_LIB} ${TVM_FFI_LIB})
    endif()
    if(MMAN_LIB)
        target_link_libraries(matmul_bench PRIVATE ${MMAN_LIB})
    endif()
    if(APPLE)
        target_link_libraries(matmul_bench PRIVATE ${CACTUS_CURL_LIBRARY_MACOS})
        target_include_directories(matmul_bench PRIVATE ${CACTUS_CURL_INCLUDE_DIR})
        target_link_libraries(matmul_bench PRIVATE
            ${COREML_FRAMEWORK}
            ${FOUNDATION_FRAMEWORK}
            ${SECURITY_FRAMEWORK}
            ${SYSTEMCONFIGURATION_FRAMEWORK}
            ${CFNETWORK_FRAMEWORK}
        )
    endif()
    message(STATUS "Added benchmark: matmul_bench")
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench/attn_bench.cpp")
    add_executable(attn_bench bench/attn_bench.cpp test_utils.cpp)
    target_link_libraries(attn_bench PRIVATE bench_lib)
    target_include_directories(attn_bench PRIVATE ${CACTUS_ROOT})
    if(WITH_MLC)
        target_link_libraries(attn_bench PRIVATE ${TVM_RUNTIME_LIB} ${TVM_FFI_LIB})
    endif()
    if(MMAN_LIB)
        target_link_libraries(attn_bench PRIVATE ${MMAN_LIB})
    endif()
    if(APPLE)
        target_link_libraries(attn_bench PRIVATE ${CACTUS_CURL_LIBRARY_MACOS})
        target_include_directories(attn_bench PRIVATE ${CACTUS_CURL_INCLUDE_DIR})
        target_link_libraries(attn_bench PRIVATE
            ${COREML_FRAMEWORK}
            ${FOUNDATION_FRAMEWORK}
            ${ACCELERATE_FRAMEWORK}
            ${SECURITY_FRAMEWORK}
            ${SYSTEMCONFIGURATION_FRAMEWORK}
            ${CFNETWORK_FRAMEWORK}
        )
    endif()
    message(STATUS "Added benchmark: attn_bench")
endif()

# ── Sparse Activation GEMV Benchmark (requires GGML) ─────────────────────────
if(WITH_GGML AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench/bench_sparse_actsparse.cpp")
    add_executable(bench_sparse_actsparse bench/bench_sparse_actsparse.cpp)
    target_include_directories(bench_sparse_actsparse PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${GGML_DIR}/include
    )
    target_link_libraries(bench_sparse_actsparse PRIVATE ggml)
    if(MMAN_LIB)
        target_link_libraries(bench_sparse_actsparse PRIVATE ${MMAN_LIB})
    endif()
    message(STATUS "Added benchmark: bench_sparse_actsparse (GGML Q4_0 sparse GEMV + PowerInfer scatter)")
endif()

# ── E2E Generation Benchmark ──────────────────────────────────────────────────
set(E2E_BENCH_SOURCES
    bench/e2e_bench.cpp
    bench/e2e_driver.cpp
    bench/e2e_backend_cactus.cpp
)

if(WITH_LLAMA_CPP)
    list(APPEND E2E_BENCH_SOURCES bench/e2e_backend_llama_cpp.cpp)
endif()
if(WITH_E2E_EXECUTORCH)
    list(APPEND E2E_BENCH_SOURCES bench/e2e_backend_executorch.cpp)
endif()
if(WITH_ONNXRT_GENAI)
    list(APPEND E2E_BENCH_SOURCES bench/e2e_backend_onnxrt.cpp)
endif()
if(WITH_E2E_LITERT)
    list(APPEND E2E_BENCH_SOURCES bench/e2e_backend_litert.cpp)
endif()

add_executable(e2e_bench ${E2E_BENCH_SOURCES})
target_include_directories(e2e_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CACTUS_ROOT}
)
target_link_libraries(e2e_bench PRIVATE ${CACTUS_LIB})

if(MMAN_LIB)
    target_link_libraries(e2e_bench PRIVATE ${MMAN_LIB})
endif()
if(APPLE)
    target_link_libraries(e2e_bench PRIVATE ${CACTUS_CURL_LIBRARY_MACOS})
    target_include_directories(e2e_bench PRIVATE ${CACTUS_CURL_INCLUDE_DIR})
    target_link_libraries(e2e_bench PRIVATE
        ${ACCELERATE_FRAMEWORK}
        ${COREML_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
        ${SECURITY_FRAMEWORK}
        ${SYSTEMCONFIGURATION_FRAMEWORK}
        ${CFNETWORK_FRAMEWORK}
    )
elseif(CURL_FOUND)
    target_link_libraries(e2e_bench PRIVATE ${CURL_LIBRARIES})
    target_include_directories(e2e_bench PRIVATE ${CURL_INCLUDE_DIRS})
endif()

if(WITH_LLAMA_CPP)
    set(LLAMA_CPP_DIR "" CACHE PATH "Path to llama.cpp installation (auto-detected from Homebrew if empty)")
    if(LLAMA_CPP_DIR STREQUAL "")
        # Try Homebrew
        execute_process(COMMAND brew --prefix llama.cpp OUTPUT_VARIABLE LLAMA_CPP_BREW OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        if(LLAMA_CPP_BREW)
            set(LLAMA_CPP_DIR ${LLAMA_CPP_BREW})
        endif()
    endif()
    if(LLAMA_CPP_DIR)
        find_library(LLAMA_LIB llama PATHS ${LLAMA_CPP_DIR}/lib REQUIRED)
        find_library(GGML_LLAMA_LIB ggml PATHS ${LLAMA_CPP_DIR}/lib)
        target_include_directories(e2e_bench PRIVATE ${LLAMA_CPP_DIR}/include)
        target_link_libraries(e2e_bench PRIVATE ${LLAMA_LIB})
        if(GGML_LLAMA_LIB)
            target_link_libraries(e2e_bench PRIVATE ${GGML_LLAMA_LIB})
        endif()
        target_compile_definitions(e2e_bench PRIVATE WITH_LLAMA_CPP)
        message(STATUS "E2E bench: llama.cpp enabled from ${LLAMA_CPP_DIR}")
    else()
        message(WARNING "WITH_LLAMA_CPP enabled but llama.cpp not found")
    endif()
endif()

if(WITH_E2E_EXECUTORCH)
    target_compile_definitions(e2e_bench PRIVATE WITH_EXECUTORCH)
    message(STATUS "E2E bench: ExecuTorch enabled")
endif()

if(WITH_ONNXRT_GENAI)
    set(ONNXRT_GENAI_DIR "" CACHE PATH "Path to ONNX RT GenAI (auto-detected from pip if empty)")
    if(ONNXRT_GENAI_DIR STREQUAL "")
        execute_process(
            COMMAND python3 -c "import onnxruntime_genai as og; import os; print(os.path.dirname(og.__file__))"
            OUTPUT_VARIABLE ONNXRT_GENAI_PIP OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        if(ONNXRT_GENAI_PIP)
            set(ONNXRT_GENAI_DIR ${ONNXRT_GENAI_PIP})
        endif()
    endif()
    if(ONNXRT_GENAI_DIR)
        find_library(ONNXRT_GENAI_LIB onnxruntime-genai PATHS ${ONNXRT_GENAI_DIR} REQUIRED)
        # Use downloaded C header
        target_include_directories(e2e_bench PRIVATE /tmp)
        target_link_libraries(e2e_bench PRIVATE ${ONNXRT_GENAI_LIB})
        target_compile_definitions(e2e_bench PRIVATE WITH_ONNXRT_GENAI)
        message(STATUS "E2E bench: ONNX RT GenAI enabled from ${ONNXRT_GENAI_DIR}")
    else()
        message(WARNING "WITH_ONNXRT_GENAI enabled but onnxruntime-genai not found")
    endif()
endif()

if(WITH_E2E_LITERT)
    target_compile_definitions(e2e_bench PRIVATE WITH_LITERT)
    message(STATUS "E2E bench: LiteRT-LM enabled")
endif()

message(STATUS "Added benchmark: e2e_bench")

file(GLOB TEST_SOURCES "test_*.cpp")
list(REMOVE_ITEM TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test_utils.cpp")
list(REMOVE_ITEM TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test_perplexity.cpp")

foreach(TEST_FILE ${TEST_SOURCES})

    get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_FILE} test_utils.cpp)
    target_link_libraries(${TEST_NAME} PRIVATE ${CACTUS_LIB})

    if(MMAN_LIB)
        target_link_libraries(${TEST_NAME} PRIVATE ${MMAN_LIB})
        message(STATUS "Test ${TEST_NAME} will link with -lmman")
    endif()

    if(SDL2_FOUND)
        target_link_libraries(${TEST_NAME} PRIVATE ${SDL2_LIBRARIES})
        target_include_directories(${TEST_NAME} PRIVATE ${SDL2_INCLUDE_DIRS})
        target_compile_definitions(${TEST_NAME} PRIVATE HAVE_SDL2)
        message(STATUS "Test ${TEST_NAME} will link with SDL2")
    endif()

    if(APPLE)
        target_link_libraries(${TEST_NAME} PRIVATE ${CACTUS_CURL_LIBRARY_MACOS})
        target_include_directories(${TEST_NAME} PRIVATE ${CACTUS_CURL_INCLUDE_DIR})
        message(STATUS "Test ${TEST_NAME} will link with libcurl")
    elseif(CURL_FOUND)
        target_link_libraries(${TEST_NAME} PRIVATE ${CURL_LIBRARIES})
        target_include_directories(${TEST_NAME} PRIVATE ${CURL_INCLUDE_DIRS})
        message(STATUS "Test ${TEST_NAME} will link with system libcurl")
    endif()
    if(APPLE)
        target_link_libraries(${TEST_NAME} PRIVATE
            ${ACCELERATE_FRAMEWORK}
            ${COREML_FRAMEWORK}
            ${FOUNDATION_FRAMEWORK}
            ${SECURITY_FRAMEWORK}
            ${SYSTEMCONFIGURATION_FRAMEWORK}
            ${CFNETWORK_FRAMEWORK}
        )
    endif()

    target_include_directories(${TEST_NAME} PRIVATE ${CACTUS_ROOT})

    message(STATUS "Added test: ${TEST_NAME}")
endforeach()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

foreach(APP_NAME chat asr attn_e2e_bench)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${APP_NAME}.cpp")
        add_executable(${APP_NAME} ${APP_NAME}.cpp)
        target_link_libraries(${APP_NAME} PRIVATE ${CACTUS_LIB})

        if(MMAN_LIB)
            target_link_libraries(${APP_NAME} PRIVATE ${MMAN_LIB})
        endif()

        if(APPLE)
            target_link_libraries(${APP_NAME} PRIVATE ${CACTUS_CURL_LIBRARY_MACOS})
            target_include_directories(${APP_NAME} PRIVATE ${CACTUS_CURL_INCLUDE_DIR})
        elseif(CURL_FOUND)
            target_link_libraries(${APP_NAME} PRIVATE ${CURL_LIBRARIES})
            target_include_directories(${APP_NAME} PRIVATE ${CURL_INCLUDE_DIRS})
        endif()
        if(APPLE)
            target_link_libraries(${APP_NAME} PRIVATE
                ${ACCELERATE_FRAMEWORK}
                ${COREML_FRAMEWORK}
                ${FOUNDATION_FRAMEWORK}
                ${SECURITY_FRAMEWORK}
                ${SYSTEMCONFIGURATION_FRAMEWORK}
                ${CFNETWORK_FRAMEWORK}
            )
        endif()

        if(SDL2_FOUND AND (APP_NAME STREQUAL "asr" OR APP_NAME STREQUAL "chat"))
            target_link_libraries(${APP_NAME} PRIVATE ${SDL2_LIBRARIES})
            target_include_directories(${APP_NAME} PRIVATE ${SDL2_INCLUDE_DIRS})
            target_compile_definitions(${APP_NAME} PRIVATE HAVE_SDL2)
            message(STATUS "${APP_NAME} will link with SDL2 for live audio")
        endif()

        target_include_directories(${APP_NAME} PRIVATE ${CACTUS_ROOT})
        message(STATUS "Added executable: ${APP_NAME}")
    endif()
endforeach()
