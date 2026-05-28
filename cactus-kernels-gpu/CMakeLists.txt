# Cactus GPU kernels (Metal-backed) — Apple-only static library.
#
# This module is a sibling of cactus-kernels/ but targets the Metal GPU
# instead of CPU NEON. It compiles a .metallib from the .metal sources and
# a static dispatch library from the .mm wrappers.
#
# Build artifact targets:
#   - cactus-kernels-gpu          static lib (libcactus-kernels-gpu.a)
#   - cactus_kernels.metallib     compiled Metal kernel pack
# Both end up in ${CMAKE_BINARY_DIR}/cactus-kernels-gpu/.

cmake_minimum_required(VERSION 3.18)
project(cactus-kernels-gpu LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT APPLE)
  message(STATUS "cactus-kernels-gpu: non-Apple host, building stub-only library")
endif()

# ----- dispatch C++/Objective-C++ static lib --------------------------------
set(DISPATCH_SOURCES
    src/dispatch/context.mm
    src/dispatch/buffer.mm
    src/dispatch/pipeline.mm
    src/dispatch/command_buffer.mm
    src/dispatch/pipeline_factory.mm
)

add_library(cactus-kernels-gpu STATIC ${DISPATCH_SOURCES})
target_include_directories(cactus-kernels-gpu
    PUBLIC  include
    PRIVATE src/dispatch)
target_compile_options(cactus-kernels-gpu PRIVATE -fobjc-arc -Wno-unused-parameter)

if(APPLE)
  find_library(METAL_FRAMEWORK     Metal     REQUIRED)
  find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
  target_link_libraries(cactus-kernels-gpu PUBLIC
      ${METAL_FRAMEWORK}
      ${FOUNDATION_FRAMEWORK})
endif()

# ----- Metal kernel library (.metallib) -------------------------------------
# We invoke the Metal compiler (xcrun metal) on the master source file.
# Apple's Metal toolchain: source.metal --> intermediate.air --> .metallib.
if(APPLE)
  set(METAL_KERNEL_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cactus_kernels.metal)
  set(METAL_KERNEL_AIR    ${CMAKE_BINARY_DIR}/cactus-kernels-gpu/cactus_kernels.air)
  set(METAL_KERNEL_LIB    ${CMAKE_BINARY_DIR}/cactus-kernels-gpu/cactus_kernels.metallib)

  # Gather all .metal sources for dependency tracking.
  file(GLOB METAL_KERNEL_SOURCES CONFIGURE_DEPENDS
       ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/*.metal)

  add_custom_command(
      OUTPUT ${METAL_KERNEL_LIB}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/cactus-kernels-gpu
      COMMAND xcrun -sdk macosx metal
              -c ${METAL_KERNEL_SOURCE}
              -o ${METAL_KERNEL_AIR}
              -I ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels
              -std=metal3.0
              -ffast-math
              -O3
      COMMAND xcrun -sdk macosx metallib ${METAL_KERNEL_AIR} -o ${METAL_KERNEL_LIB}
      DEPENDS ${METAL_KERNEL_SOURCES}
      COMMENT "Compiling cactus Metal kernel library"
      VERBATIM)

  add_custom_target(cactus_kernels_metallib ALL DEPENDS ${METAL_KERNEL_LIB})
  add_dependencies(cactus-kernels-gpu cactus_kernels_metallib)

  # Expose the metallib path to consumers via a CMake variable.
  set(CACTUS_KERNELS_METALLIB_PATH ${METAL_KERNEL_LIB} CACHE INTERNAL
      "Path to compiled Metal kernel library")
endif()

# ----- tests (optional) -----------------------------------------------------
option(CACTUS_KERNELS_GPU_TESTS "Build cactus-kernels-gpu tests" OFF)
if(CACTUS_KERNELS_GPU_TESTS AND APPLE)
  add_subdirectory(tests)
endif()
