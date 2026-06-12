cmake_minimum_required(VERSION 3.16)
project(CactusBench LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

option(WITH_GGML            "Build with llama.cpp / ggml backend"             OFF)
option(WITH_LITERT          "Build with LiteRT (TFLite) backend"              OFF)
option(WITH_EXECUTORCH      "Build with ExecuTorch matmul (XNNPACK) backend"  OFF)
option(WITH_EXECUTORCH_LLM  "Build with ExecuTorch attention (custom_sdpa) — requires a built ExecuTorch tree" OFF)
option(WITH_ONNXRT          "Build with ONNX Runtime backend"                 OFF)

set(CACTUS_PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)

# ── Compile flags ───────────────────────────────────────────────────────────
if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES "arm64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -arch arm64 -march=armv8.2-a+fp16+simd+dotprod+i8mm -pthread -O3 -Wall -Wextra")
    add_compile_definitions(
        __ARM_NEON=1
        __ARM_FEATURE_FP16_VECTOR_ARITHMETIC=1
        __ARM_FEATURE_DOTPROD=1
        __ARM_FEATURE_MATMUL_INT8=1
    )
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.2-a+fp16+simd+dotprod+i8mm -pthread -O3 -Wall -Wextra")
    add_compile_definitions(
        __ARM_NEON=1
        __ARM_FEATURE_FP16_VECTOR_ARITHMETIC=1
        __ARM_FEATURE_DOTPROD=1
        __ARM_FEATURE_MATMUL_INT8=1
    )
endif()

# ── Cactus library (built separately by cactus/build.sh) ────────────────────
set(CACTUS_BUILD_DIR ${CACTUS_PROJECT_ROOT}/cactus/build)
find_library(CACTUS_LIB
    NAMES cactus
    HINTS ${CACTUS_BUILD_DIR} ${CACTUS_BUILD_DIR}/lib
    REQUIRED
)
message(STATUS "Cactus library: ${CACTUS_LIB}")

# ── Apple frameworks (Accelerate, etc.) ─────────────────────────────────────
if(APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
endif()

# ── Optional third-party backend dependencies ───────────────────────────────

# GGML / llama.cpp
if(WITH_GGML)
    set(GGML_DIR "" CACHE PATH "Path to ggml source tree (e.g. ../third_party/ggml)")
    if(NOT GGML_DIR OR NOT EXISTS "${GGML_DIR}/CMakeLists.txt")
        if(EXISTS "${CACTUS_PROJECT_ROOT}/../third_party/ggml/CMakeLists.txt")
            set(GGML_DIR "${CACTUS_PROJECT_ROOT}/../third_party/ggml")
        else()
            message(FATAL_ERROR "WITH_GGML=ON but no ggml source found. Set -DGGML_DIR=/path/to/ggml or clone into ../../third_party/ggml")
        endif()
    endif()
    set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
    set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    add_subdirectory(${GGML_DIR} ${CMAKE_CURRENT_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)
    message(STATUS "GGML enabled from ${GGML_DIR}")
endif()

# LiteRT
if(WITH_LITERT)
    set(LITERT_DIR "" CACHE PATH "Path to LiteRT source tree")
    if(NOT LITERT_DIR OR NOT EXISTS "${LITERT_DIR}/tflite")
        if(EXISTS "${CACTUS_PROJECT_ROOT}/../third_party/litert/tflite")
            set(LITERT_DIR "${CACTUS_PROJECT_ROOT}/../third_party/litert")
        else()
            message(FATAL_ERROR "WITH_LITERT=ON but no LiteRT source found. Set -DLITERT_DIR=... or clone to ../../third_party/litert")
        endif()
    endif()

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
    set(TFLITE_ENABLE_GPU     OFF CACHE BOOL "" FORCE)
    set(TFLITE_ENABLE_NNAPI   OFF CACHE BOOL "" FORCE)
    set(TFLITE_ENABLE_EXTERNAL_DELEGATE OFF CACHE BOOL "" FORCE)
    set(TFLITE_ENABLE_RUY     ON  CACHE BOOL "" FORCE)
    add_subdirectory(${LITERT_DIR}/tflite ${CMAKE_CURRENT_BINARY_DIR}/tflite EXCLUDE_FROM_ALL)

    add_library(litert_kernels STATIC
        ${LITERT_DIR}/tflite/kernels/internal/optimized/neon_tensor_utils.cc
        ${LITERT_DIR}/tflite/kernels/internal/optimized/cpu_check.cc
        ${LITERT_DIR}/tflite/kernels/internal/reference/portable_tensor_utils.cc
        ${LITERT_DIR}/tflite/kernels/internal/common.cc
        ${LITERT_DIR}/tflite/kernels/internal/quantization_util.cc
        ${LITERT_DIR}/tflite/kernels/cpu_backend_context.cc
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
    target_compile_options(litert_kernels PRIVATE -O3 -DTFLITE_WITH_RUY)
    message(STATUS "LiteRT enabled from ${LITERT_DIR}")
endif()

# ExecuTorch / XNNPACK (matmul only — INT8 qc8w GEMM via XNNPACK)
if(WITH_EXECUTORCH)
    include(FetchContent)
    set(XNNPACK_BUILD_TESTS      OFF CACHE BOOL "" FORCE)
    set(XNNPACK_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
    set(XNNPACK_LIBRARY_TYPE     "static" CACHE STRING "" FORCE)
    FetchContent_Declare(
        xnnpack
        GIT_REPOSITORY https://github.com/google/XNNPACK
        GIT_TAG master
        GIT_SHALLOW TRUE
        SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/xnnpack-src"
    )
    FetchContent_MakeAvailable(xnnpack)
    message(STATUS "ExecuTorch (XNNPACK) matmul enabled")
endif()

# ExecuTorch llm/custom_ops — fused SDPA attention kernel.
# Requires a SEPARATELY-BUILT ExecuTorch checkout. Heavy dependency, not
# fetched at configure time. User must:
#   git clone https://github.com/pytorch/executorch ../../third_party/executorch
#   cd ../../third_party/executorch && ./install_executorch.sh
#   cmake .. -DEXECUTORCH_BUILD_KERNELS_LLM=ON -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON
#   cmake --build cmake-out -j
# Then point us at the build:
#   cmake .. -DWITH_EXECUTORCH_LLM=ON -DEXECUTORCH_DIR=../../third_party/executorch
if(WITH_EXECUTORCH_LLM)
    set(EXECUTORCH_DIR "" CACHE PATH "Path to a built ExecuTorch tree (with cmake-out/)")
    if(NOT EXECUTORCH_DIR OR NOT EXISTS "${EXECUTORCH_DIR}/extension/llm/custom_ops/op_sdpa.h")
        if(EXISTS "${CACTUS_PROJECT_ROOT}/../third_party/executorch/extension/llm/custom_ops/op_sdpa.h")
            set(EXECUTORCH_DIR "${CACTUS_PROJECT_ROOT}/../third_party/executorch")
        else()
            message(FATAL_ERROR "WITH_EXECUTORCH_LLM=ON but ExecuTorch tree not found. Set -DEXECUTORCH_DIR=/path/to/executorch (with cmake-out/ already built)")
        endif()
    endif()

    set(ET_BUILD_DIR "${EXECUTORCH_DIR}/cmake-out")
    if(NOT EXISTS "${ET_BUILD_DIR}")
        message(FATAL_ERROR "ExecuTorch must be built first: ${ET_BUILD_DIR} not found")
    endif()

    # ET's cmake-out lays libraries out as <subdir>/<config>/<libname>.a
    # (e.g. .../extension/llm/custom_ops/Release/libcustom_ops.a). Search
    # both the bare and Release/Debug variants.
    set(ET_LIB_HINTS
        ${ET_BUILD_DIR}
        ${ET_BUILD_DIR}/Release
        ${ET_BUILD_DIR}/Debug
        ${ET_BUILD_DIR}/lib
        ${ET_BUILD_DIR}/extension/llm/custom_ops
        ${ET_BUILD_DIR}/extension/llm/custom_ops/Release
        ${ET_BUILD_DIR}/extension/tensor
        ${ET_BUILD_DIR}/extension/tensor/Release
        ${ET_BUILD_DIR}/extension/threadpool
        ${ET_BUILD_DIR}/extension/threadpool/Release
        ${ET_BUILD_DIR}/kernels/optimized
        ${ET_BUILD_DIR}/kernels/optimized/Release
        ${ET_BUILD_DIR}/backends/xnnpack/third-party/pthreadpool
        ${ET_BUILD_DIR}/backends/xnnpack/third-party/pthreadpool/Release
    )
    find_library(ET_CUSTOM_OPS_LIB custom_ops      PATHS ${ET_LIB_HINTS} REQUIRED)
    find_library(ET_CORE_LIB       executorch_core PATHS ${ET_LIB_HINTS} REQUIRED)
    find_library(ET_THREADPOOL_LIB extension_threadpool PATHS ${ET_LIB_HINTS})
    find_library(ET_PTHREADPOOL_LIB pthreadpool   PATHS ${ET_LIB_HINTS})
    find_library(ET_KERNELS_OPT_LIB optimized_kernels PATHS ${ET_LIB_HINTS})
    find_library(ET_CPUBLAS_LIB    cpublas         PATHS ${ET_LIB_HINTS})
    find_library(ET_TENSOR_EXT_LIB extension_tensor PATHS ${ET_LIB_HINTS})

    set(ET_INCLUDE_DIRS
        ${EXECUTORCH_DIR}
        ${EXECUTORCH_DIR}/..       # for <executorch/...> includes
        ${EXECUTORCH_DIR}/runtime/core/portable_type/c10   # for vendored <c10/...>
    )
    message(STATUS "ExecuTorch (LLM/SDPA) enabled from ${EXECUTORCH_DIR}")
endif()

# ONNX Runtime (prebuilt)
if(WITH_ONNXRT)
    set(ONNXRT_DIR "" CACHE PATH "Path to onnxruntime prebuilt tree")
    if(NOT ONNXRT_DIR OR NOT EXISTS "${ONNXRT_DIR}/include")
        if(EXISTS "${CACTUS_PROJECT_ROOT}/../third_party/onnxruntime/include")
            set(ONNXRT_DIR "${CACTUS_PROJECT_ROOT}/../third_party/onnxruntime")
        else()
            message(FATAL_ERROR "WITH_ONNXRT=ON but onnxruntime not found. Set -DONNXRT_DIR=... or extract to ../../third_party/onnxruntime")
        endif()
    endif()
    find_library(ONNXRT_LIB onnxruntime PATHS ${ONNXRT_DIR}/lib REQUIRED)
    message(STATUS "ONNX Runtime enabled from ${ONNXRT_DIR}")
endif()

# ── Build the bench library ─────────────────────────────────────────────────
add_library(bench_lib OBJECT
    bench/bench_common.cpp
    bench/bench_driver.cpp
    bench/backend_cactus.cpp
    bench/backend_ggml.cpp
    bench/backend_litert.cpp
    bench/backend_executorch.cpp
    bench/backend_onnxrt.cpp
)
target_include_directories(bench_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CACTUS_PROJECT_ROOT}
    ${CACTUS_PROJECT_ROOT}/cactus-kernels
)
target_link_libraries(bench_lib PUBLIC ${CACTUS_LIB})
if(APPLE)
    target_link_libraries(bench_lib PUBLIC ${ACCELERATE_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
endif()

if(WITH_GGML)
    target_link_libraries(bench_lib PUBLIC ggml)
    target_compile_definitions(bench_lib PRIVATE WITH_GGML)
endif()
if(WITH_LITERT)
    target_link_libraries(bench_lib PUBLIC litert_kernels)
    target_compile_definitions(bench_lib PRIVATE WITH_LITERT)
endif()
if(WITH_EXECUTORCH)
    target_link_libraries(bench_lib PUBLIC XNNPACK)
    target_compile_definitions(bench_lib PRIVATE WITH_EXECUTORCH)
endif()
if(WITH_EXECUTORCH_LLM)
    target_include_directories(bench_lib PUBLIC ${ET_INCLUDE_DIRS})
    # ET's vendored c10 expects a generated cmake_macros.h; skip it (matches
    # how ET's own targets compile — see executorch/CMakeLists.txt).
    target_compile_definitions(bench_lib PRIVATE C10_USING_CUSTOM_GENERATED_MACROS)
    target_link_libraries(bench_lib PUBLIC
        ${ET_CUSTOM_OPS_LIB}
        ${ET_CORE_LIB}
    )
    if(ET_THREADPOOL_LIB)
        target_link_libraries(bench_lib PUBLIC ${ET_THREADPOOL_LIB})
    endif()
    if(ET_PTHREADPOOL_LIB)
        target_link_libraries(bench_lib PUBLIC ${ET_PTHREADPOOL_LIB})
    endif()
    if(ET_KERNELS_OPT_LIB)
        target_link_libraries(bench_lib PUBLIC ${ET_KERNELS_OPT_LIB})
    endif()
    if(ET_CPUBLAS_LIB)
        target_link_libraries(bench_lib PUBLIC ${ET_CPUBLAS_LIB})
    endif()
    if(ET_TENSOR_EXT_LIB)
        target_link_libraries(bench_lib PUBLIC ${ET_TENSOR_EXT_LIB})
    endif()
    target_compile_definitions(bench_lib PRIVATE WITH_EXECUTORCH_LLM)
endif()
if(WITH_ONNXRT)
    target_include_directories(bench_lib PUBLIC ${ONNXRT_DIR}/include)
    target_link_libraries(bench_lib PUBLIC ${ONNXRT_LIB})
    target_compile_definitions(bench_lib PRIVATE WITH_ONNXRT)
endif()

# ── Executables ─────────────────────────────────────────────────────────────
add_executable(matmul_bench bench/matmul_bench.cpp)
target_link_libraries(matmul_bench PRIVATE bench_lib)

add_executable(attn_bench bench/attn_bench.cpp)
target_link_libraries(attn_bench PRIVATE bench_lib)

add_executable(accuracy_test bench/accuracy_test.cpp)
target_link_libraries(accuracy_test PRIVATE bench_lib)
