cmake_minimum_required(VERSION 3.10)
project(CactusKernels LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(cactus_kernels STATIC
    src/attention.cpp
    src/attention_hybrid.cpp
    src/blas.cpp
    src/conv.cpp
    src/conv2d.cpp
    src/dsp.cpp
    src/fused.cpp
    src/image.cpp
    src/lstm.cpp
    src/matmul.cpp
    src/matmul_sme2.cpp
    src/matmul_sme2_gemv.cpp
    src/nn.cpp
    src/norms_rope.cpp
    src/quants.cpp
    src/reduce.cpp
    src/scalar.cpp
)

set_target_properties(cactus_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(cactus_kernels
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
           ${CMAKE_CURRENT_SOURCE_DIR}/src
           ${CMAKE_CURRENT_SOURCE_DIR}/libs
)

target_compile_options(cactus_kernels PUBLIC
    -march=armv8.2-a+fp16+simd+dotprod+i8mm -pthread -O3
)

target_compile_options(cactus_kernels PRIVATE
    -Wall -Wextra -pedantic -Wno-missing-field-initializers
    $<$<CXX_COMPILER_ID:AppleClang,Clang>:-Wno-c99-extensions>
    $<$<CXX_COMPILER_ID:GNU>:-Wno-pedantic>
)

if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES "arm64")
    target_compile_definitions(cactus_kernels PUBLIC ACCELERATE_NEW_LAPACK)
    target_link_libraries(cactus_kernels PUBLIC "-framework Accelerate")
endif()

# SME2 path: compile only matmul_sme2.cpp with +sme2 so the streaming/ZA code is isolated and the
# base library keeps targeting armv8.2-a (runs on non-SME devices). Use armv8.2-a+...+sme2, NOT
# armv9-a+sme2 — armv9-a implies non-streaming SVE2 that Apple Silicon lacks (SIGILLs at runtime).
include(CheckCXXSourceCompiles)
set(_CACTUS_SME2_FLAG "-march=armv8.2-a+fp16+simd+dotprod+i8mm+sme2")
set(CMAKE_REQUIRED_FLAGS "${_CACTUS_SME2_FLAG}")
check_cxx_source_compiles("
#include <arm_sme.h>
__arm_locally_streaming __arm_new(\"za\") static void f(){ svzero_za(); }
int main(){ return 0; }
" CACTUS_HAS_SME2)
unset(CMAKE_REQUIRED_FLAGS)
if(CACTUS_HAS_SME2)
    set_source_files_properties(src/matmul_sme2.cpp PROPERTIES COMPILE_OPTIONS "${_CACTUS_SME2_FLAG}")
    # vg1x4 GEMV TU: -O1 ONLY — clang 17 mis-compiles the SME2 multi-vector ZA-array dot at -O2/-O3
    # into an illegal encoding that SIGILLs on M4 (correct + fast at -O0/-O1). Isolate it here.
    set_source_files_properties(src/matmul_sme2_gemv.cpp PROPERTIES COMPILE_OPTIONS "${_CACTUS_SME2_FLAG};-O1")
    message(STATUS "cactus_kernels: SME2 enabled (matmul_sme2.cpp; matmul_sme2_gemv.cpp at -O1)")
else()
    message(STATUS "cactus_kernels: SME2 not supported by toolchain; SME TUs use scalar stubs")
endif()

if((BUILD_TESTING OR CACTUS_BUILD_TESTS) AND CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    file(GLOB TEST_SOURCES "tests/test_*.cpp")
    foreach(TEST_SRC ${TEST_SOURCES})
        get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)
        add_executable(${TEST_NAME} ${TEST_SRC})
        target_link_libraries(${TEST_NAME} PRIVATE cactus_kernels)
    endforeach()
endif()
