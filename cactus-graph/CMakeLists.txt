cmake_minimum_required(VERSION 3.10)
project(CactusGraph LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT TARGET cactus_kernels)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../cactus-kernels ${CMAKE_CURRENT_BINARY_DIR}/cactus-kernels)
endif()

add_library(cactus_graph STATIC
    src/core.cpp
    src/builder.cpp
    src/execute.cpp
    src/metal_runtime.cpp
    src/metal_plan.cpp
    src/io.cpp
    src/param_io.cpp
    src/ops_math.cpp
    src/ops_nn.cpp
    src/ops_conv.cpp
    src/ops_recurrent.cpp
    src/ops_tensor.cpp
    src/ops_sample.cpp
    src/ops_cache.cpp
    src/ops_dsp.cpp
    src/ops_image.cpp
    src/graph_ffi.cpp
    src/last_error.cpp
)

set_target_properties(cactus_graph PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(cactus_graph
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(cactus_graph PUBLIC cactus_kernels)

target_compile_options(cactus_graph PRIVATE
    -Wall -Wextra -pedantic -Wno-missing-field-initializers
    $<$<CXX_COMPILER_ID:AppleClang,Clang>:-Wno-c99-extensions>
    $<$<CXX_COMPILER_ID:GNU>:-Wno-pedantic>
)

if((BUILD_TESTING OR CACTUS_BUILD_TESTS) AND CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    file(GLOB TEST_SOURCES "tests/test_*.cpp")
    foreach(TEST_SRC ${TEST_SOURCES})
        get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)
        add_executable(${TEST_NAME} ${TEST_SRC})
        target_link_libraries(${TEST_NAME} PRIVATE cactus_graph)
    endforeach()
endif()
