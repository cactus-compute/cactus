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
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/libs
)

target_compile_options(cactus_kernels PUBLIC
    -march=armv8.2-a+fp16+simd+dotprod+i8mm -pthread -O3
)

target_compile_definitions(cactus_kernels PUBLIC
    __ARM_NEON=1
    __ARM_FEATURE_FP16_VECTOR_ARITHMETIC=1
    __ARM_FEATURE_DOTPROD=1
    __ARM_FEATURE_MATMUL_INT8=1
)

target_compile_options(cactus_kernels PRIVATE
    -Wall -Wextra -pedantic -Wno-missing-field-initializers
    $<$<CXX_COMPILER_ID:AppleClang,Clang>:-Wno-c99-extensions>
    $<$<CXX_COMPILER_ID:GNU>:-Wno-pedantic>
)

if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES "arm64")
    target_compile_definitions(cactus_kernels PUBLIC ACCELERATE_NEW_LAPACK)
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
    target_link_libraries(cactus_kernels PUBLIC ${ACCELERATE_FRAMEWORK})
endif()

if(BUILD_TESTING OR CACTUS_BUILD_TESTS)
    set(TEST_SOURCES
        tests/test_elementwise.cpp
        tests/test_matmul.cpp
        tests/test_attention.cpp
        tests/test_reduce.cpp
        tests/test_conv.cpp
        tests/test_quant.cpp
        tests/test_dsp.cpp
    )
    foreach(TEST_SRC ${TEST_SOURCES})
        get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)
        add_executable(${TEST_NAME} ${TEST_SRC})
        target_link_libraries(${TEST_NAME} PRIVATE cactus_kernels)
    endforeach()
endif()
