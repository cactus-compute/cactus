cmake_minimum_required(VERSION 3.10)
project(CactusKernels LANGUAGES CXX)

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

set(KERNEL_SOURCES
    src/attention.cpp
    src/blas.cpp
    src/conv.cpp
    src/dsp.cpp
    src/fused.cpp
    src/image.cpp
    src/lstm.cpp
    src/matmul.cpp
    src/nn.cpp
    src/quants.cpp
    src/reduce.cpp
    src/scalar.cpp
)

add_library(cactus_kernels STATIC ${KERNEL_SOURCES})
target_include_directories(cactus_kernels
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/libs
)
set_target_properties(cactus_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
    target_link_libraries(cactus_kernels PUBLIC ${ACCELERATE_FRAMEWORK})
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(cactus_kernels PRIVATE -Wno-c99-extensions)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(cactus_kernels PRIVATE -Wno-pedantic)
endif()

# Tests
if(BUILD_TESTING OR CACTUS_BUILD_TESTS)
    if(APPLE)
        find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
    endif()

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
        target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        if(APPLE)
            target_link_libraries(${TEST_NAME} PRIVATE ${ACCELERATE_FRAMEWORK})
        endif()
    endforeach()

endif()
