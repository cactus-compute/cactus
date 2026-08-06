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
           ${CMAKE_CURRENT_SOURCE_DIR}/libs
)

add_library(cactus_flags INTERFACE)

target_compile_options(cactus_flags INTERFACE
    -march=armv8.2-a+fp16+simd+dotprod+i8mm -pthread -O3
    -fvisibility=hidden -fvisibility-inlines-hidden
    -ffunction-sections -fdata-sections -fno-rtti
    $<$<CXX_COMPILER_ID:Clang,AppleClang>:-flto=thin>
)

target_link_options(cactus_flags INTERFACE
    $<$<CXX_COMPILER_ID:Clang,AppleClang>:-flto=thin>
)

if(APPLE)
    target_link_options(cactus_flags INTERFACE -Wl,-dead_strip)
else()
    target_link_options(cactus_flags INTERFACE -Wl,--gc-sections)
endif()

if(ANDROID)
    target_link_options(cactus_flags INTERFACE -Wl,--exclude-libs,ALL)
endif()

target_link_libraries(cactus_kernels PUBLIC cactus_flags)

target_compile_options(cactus_kernels PRIVATE
    -Wall -Wextra -pedantic -Wno-missing-field-initializers
    $<$<CXX_COMPILER_ID:AppleClang,Clang>:-Wno-c99-extensions>
    $<$<CXX_COMPILER_ID:GNU>:-Wno-pedantic>
)

if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES "arm64")
    target_compile_definitions(cactus_kernels PUBLIC ACCELERATE_NEW_LAPACK)
    target_link_libraries(cactus_kernels PUBLIC "-framework Accelerate")
    enable_language(OBJCXX)
    set(_msl_in  ${CMAKE_CURRENT_SOURCE_DIR}/src/cactus_kernels.metal)
    set(_msl_out ${CMAKE_CURRENT_BINARY_DIR}/generated/cactus_kernels_msl.h)
    add_custom_command(
        OUTPUT  ${_msl_out}
        COMMAND ${CMAKE_COMMAND} -DIN=${_msl_in} -DOUT=${_msl_out} -DDELIM=CACTUS_MSL
                -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_msl.cmake
        DEPENDS ${_msl_in} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_msl.cmake
        VERBATIM)
    target_sources(cactus_kernels PRIVATE src/metal_backend.mm ${_msl_out})
    target_include_directories(cactus_kernels PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)
    set_source_files_properties(src/metal_backend.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
    target_link_libraries(cactus_kernels PUBLIC "-framework Metal" "-framework Foundation" "-framework MetalPerformanceShaders")
else()
    target_sources(cactus_kernels PRIVATE src/metal_backend_stub.cpp)
endif()

if((BUILD_TESTING OR CACTUS_BUILD_TESTS) AND CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    file(GLOB TEST_SOURCES "tests/test_*.cpp")
    foreach(TEST_SRC ${TEST_SOURCES})
        get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)
        add_executable(${TEST_NAME} ${TEST_SRC})
        target_link_libraries(${TEST_NAME} PRIVATE cactus_kernels)
    endforeach()
endif()
