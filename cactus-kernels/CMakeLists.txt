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
    src/fused.cpp
    src/lstm.cpp
    src/matmul.cpp
    src/nn.cpp
    src/quants.cpp
    src/reduce.cpp
    src/scalar.cpp
)

# i8mm compiled separately with explicit i8mm flags
set(I8MM_SOURCE src/i8mm.cpp)
set(I8MM_MARCH_FLAG "-march=armv8.2-a+fp16+simd+dotprod+i8mm")

add_library(cactus_i8mm_obj OBJECT ${I8MM_SOURCE})
target_include_directories(cactus_i8mm_obj PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(cactus_i8mm_obj PRIVATE ${I8MM_MARCH_FLAG})
set_target_properties(cactus_i8mm_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(cactus_i8mm_obj PRIVATE CACTUS_COMPILE_I8MM __ARM_FEATURE_MATMUL_INT8=1)

add_library(cactus_kernels STATIC ${KERNEL_SOURCES} $<TARGET_OBJECTS:cactus_i8mm_obj>)
target_include_directories(cactus_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(cactus_kernels PRIVATE CACTUS_COMPILE_I8MM)
set_target_properties(cactus_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)

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

    add_executable(test_tq tests/test_tq.cpp)
    target_link_libraries(test_tq PRIVATE cactus_kernels)
    target_include_directories(test_tq PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(test_tq PRIVATE
        CACTUS_TEST_ASSETS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/assets"
    )
    if(APPLE)
        target_link_libraries(test_tq PRIVATE ${ACCELERATE_FRAMEWORK})
    endif()
endif()
