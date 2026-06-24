cmake_minimum_required(VERSION 3.10)
project(CactusAndroid LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Release CACHE STRING "")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if(NOT ${ANDROID_ABI} STREQUAL "arm64-v8a")
    message(FATAL_ERROR "Unsupported Android ABI: ${ANDROID_ABI}. Cactus only supports arm64-v8a.")
endif()

if(NOT TARGET cactus_engine)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../cactus-engine ${CMAKE_CURRENT_BINARY_DIR}/cactus-engine)
endif()

set(_DEPS cactus_engine cactus_graph cactus_kernels)

add_library(cactus_jni SHARED ${CMAKE_CURRENT_SOURCE_DIR}/cactus_jni.cpp)

set_target_properties(cactus_jni PROPERTIES
    OUTPUT_NAME cactus_engine
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

target_include_directories(cactus_jni PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../cactus-engine
    ${CMAKE_CURRENT_SOURCE_DIR}/../cactus-engine/src
)

target_link_libraries(cactus_jni PRIVATE ${_DEPS})

set(_CACTUS_VERSION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../CACTUS_VERSION")
if(EXISTS "${_CACTUS_VERSION_FILE}")
    file(READ "${_CACTUS_VERSION_FILE}" _CACTUS_VERSION_CONTENT)
    string(STRIP "${_CACTUS_VERSION_CONTENT}" CACTUS_VERSION_STR)
    target_compile_definitions(cactus_jni PRIVATE CACTUS_COMPILE_TIME_VERSION="${CACTUS_VERSION_STR}")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(cactus_jni PRIVATE CACTUS_ANDROID_ENABLE_LOGGING)
endif()

target_link_options(cactus_jni PRIVATE -Wl,-z,max-page-size=16384)
