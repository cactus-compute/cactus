cmake_minimum_required(VERSION 3.10)
project(Cactus LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.2-a+fp16+simd+dotprod -pthread -Wall -Wextra -pedantic -O3")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
  message(STATUS "CMAKE_BUILD_TYPE was not set, defaulting to Release.")
endif()

file(GLOB ENGINE_SOURCES "engine/*.cpp")
file(GLOB GRAPH_SOURCES "graph/*.cpp")
file(GLOB KERNEL_SOURCES "kernel/*.cpp")
file(GLOB FFI_SOURCES "ffi/*.cpp")
file(GLOB MODEL_SOURCES "models/*.cpp")

set(HEADER_FILES
    cactus.h
    engine/engine.h
    graph/graph.h
    kernel/kernel.h
    kernel/kernel_utils.h
    ffi/cactus_ffi.h
    models/model.h
)

set(COMMON_SOURCES
    ${KERNEL_SOURCES}
    ${GRAPH_SOURCES}
    ${ENGINE_SOURCES}
    ${FFI_SOURCES}
    ${MODEL_SOURCES}
)

add_library(cactus STATIC ${COMMON_SOURCES})
target_compile_definitions(cactus PUBLIC PLATFORM_CPU_ONLY=1)

target_include_directories(cactus PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    engine
    graph
    kernel
    ffi
    models
)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)