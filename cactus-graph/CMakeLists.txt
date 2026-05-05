cmake_minimum_required(VERSION 3.10)
project(CactusGraph LANGUAGES CXX)

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

set(CACTUS_KERNELS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../cactus-kernels")
if(NOT TARGET cactus_kernels)
    add_subdirectory(${CACTUS_KERNELS_DIR} ${CMAKE_CURRENT_BINARY_DIR}/cactus-kernels)
endif()

set(GRAPH_SOURCES
    src/core.cpp
    src/builder.cpp
    src/execute.cpp
    src/io.cpp
    src/param_io.cpp
    src/ops_math.cpp
    src/ops_nn.cpp
    src/ops_tensor.cpp
    src/ops_sample.cpp
    src/ops_cache.cpp
    src/ops_dsp.cpp
    src/ops_image.cpp
)

add_library(cactus_graph STATIC ${GRAPH_SOURCES})
target_include_directories(cactus_graph
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(cactus_graph PUBLIC cactus_kernels)
set_target_properties(cactus_graph PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(cactus_graph PRIVATE -Wno-c99-extensions)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(cactus_graph PRIVATE -Wno-pedantic)
endif()

if(BUILD_TESTING OR CACTUS_BUILD_TESTS)
    if(APPLE)
        find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
    endif()

    file(GLOB TEST_SOURCES "tests/test_*.cpp")
    foreach(TEST_SRC ${TEST_SOURCES})
        get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)
        add_executable(${TEST_NAME} ${TEST_SRC})
        target_link_libraries(${TEST_NAME} PRIVATE cactus_graph)
        target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        if(APPLE)
            target_link_libraries(${TEST_NAME} PRIVATE ${ACCELERATE_FRAMEWORK})
        endif()
    endforeach()
endif()
