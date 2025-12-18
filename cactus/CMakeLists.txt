cmake_minimum_required(VERSION 3.10)
project(Cactus LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
    message(STATUS "CMAKE_BUILD_TYPE was not set, defaulting to Release.")
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../CACTUS_VERSION" CACTUS_VERSION)
string(STRIP "${CACTUS_VERSION}" CACTUS_VERSION)

if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES "arm64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -arch arm64 -pthread -Wall -Wextra -pedantic -O3 -Wno-missing-designated-field-initializers")

    enable_language(OBJCXX)
    find_package(CURL REQUIRED)
    find_library(COREML_FRAMEWORK CoreML REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.2-a+fp16+simd+dotprod -pthread -Wall -Wextra -pedantic -O3 -Wno-missing-designated-field-initializers")
endif()

file(GLOB ENGINE_SOURCES "engine/*.cpp")
file(GLOB GRAPH_SOURCES "graph/*.cpp")
file(GLOB KERNEL_SOURCES "kernel/*.cpp")
file(GLOB FFI_SOURCES "ffi/*.cpp")
file(GLOB MODEL_SOURCES "models/*.cpp")

file(GLOB NPU_SOURCES "npu/*.cpp")

set(CACTUS_PRO_LIB "${CMAKE_CURRENT_SOURCE_DIR}/../libs/libcactus_pro.a")
if(EXISTS ${CACTUS_PRO_LIB})
    message(STATUS "Found libcactus_pro.a - NPU acceleration enabled")
    set(CACTUS_HAS_PRO TRUE)
else()
    message(STATUS "libcactus_pro.a not found - NPU acceleration disabled (using stubs)")
    set(CACTUS_HAS_PRO FALSE)
endif()

set(COMMON_SOURCES
    ${ENGINE_SOURCES}
    ${GRAPH_SOURCES}
    ${KERNEL_SOURCES}
    ${FFI_SOURCES}
    ${MODEL_SOURCES}
    ${NPU_SOURCES}
)

set(COMMON_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}
    engine
    graph
    kernel
    ffi
    models
    npu
)

set(COMMON_DEFINITIONS
    PLATFORM_CPU_ONLY=1
    CACTUS_VERSION="${CACTUS_VERSION}"
)

function(configure_cactus_target target_name)
    target_compile_definitions(${target_name} PUBLIC ${COMMON_DEFINITIONS})
    target_include_directories(${target_name} PUBLIC ${COMMON_INCLUDES})

    if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(${target_name} PRIVATE -Wno-c99-extensions)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(${target_name} PRIVATE -Wno-pedantic)
    endif()

    if(CACTUS_HAS_PRO)
        if(APPLE)
            target_link_libraries(${target_name} PUBLIC "-force_load ${CACTUS_PRO_LIB}")
        else()
            target_link_libraries(${target_name} PUBLIC
                "-Wl,--whole-archive" ${CACTUS_PRO_LIB} "-Wl,--no-whole-archive")
        endif()
    endif()

    if(APPLE)
        target_link_libraries(${target_name} PUBLIC
            ${COREML_FRAMEWORK}
            ${FOUNDATION_FRAMEWORK}
            ${CURL_LIBRARIES}
        )
        target_include_directories(${target_name} PUBLIC ${CURL_INCLUDE_DIRS})
    endif()
endfunction()

add_library(cactus STATIC ${COMMON_SOURCES})
configure_cactus_target(cactus)

add_library(cactus_ffi SHARED ${COMMON_SOURCES})
configure_cactus_target(cactus_ffi)
set_target_properties(cactus_ffi PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "cactus"
)

