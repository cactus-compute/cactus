cmake_minimum_required(VERSION 3.10)
project(CactusAndroid LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Release CACHE STRING "")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if(NOT ${ANDROID_ABI} STREQUAL "arm64-v8a")
    message(FATAL_ERROR "Unsupported Android ABI: ${ANDROID_ABI}. Cactus only supports arm64-v8a.")
endif()

set(CACTUS_MBEDTLS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/mbedtls/${ANDROID_ABI}")
if(NOT EXISTS "${CACTUS_MBEDTLS_ROOT}")
    set(CACTUS_MBEDTLS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../libs/mbedtls/android/${ANDROID_ABI}")
endif()

if(NOT TARGET cactus_engine)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../cactus ${CMAKE_CURRENT_BINARY_DIR}/cactus)
endif()

set(_DEPS cactus_engine cactus_graph cactus_kernels)

add_library(cactus_jni SHARED ${CMAKE_CURRENT_SOURCE_DIR}/cactus_jni.cpp)

set_target_properties(cactus_jni PROPERTIES
    OUTPUT_NAME cactus
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

target_include_directories(cactus_jni PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../cactus
    ${CMAKE_CURRENT_SOURCE_DIR}/../cactus-engine
    ${CMAKE_CURRENT_SOURCE_DIR}/../cactus-engine/src
)

target_link_libraries(cactus_jni PRIVATE ${_DEPS})

find_library(LOG_LIB log)
target_link_libraries(cactus_jni PRIVATE ${LOG_LIB} android)

set(_MBEDTLS "${CACTUS_MBEDTLS_ROOT}/lib/libmbedtls.a")
set(_MBEDX509 "${CACTUS_MBEDTLS_ROOT}/lib/libmbedx509.a")
set(_MBEDCRYPTO "${CACTUS_MBEDTLS_ROOT}/lib/libmbedcrypto.a")
if(EXISTS "${_MBEDTLS}" AND EXISTS "${_MBEDX509}" AND EXISTS "${_MBEDCRYPTO}")
    target_link_libraries(cactus_jni PRIVATE "${_MBEDTLS}" "${_MBEDX509}" "${_MBEDCRYPTO}")
endif()

target_compile_definitions(cactus_jni PRIVATE PLATFORM_CPU_ONLY=1)

set(_CACTUS_VERSION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../CACTUS_VERSION")
if(EXISTS "${_CACTUS_VERSION_FILE}")
    file(READ "${_CACTUS_VERSION_FILE}" _CACTUS_VERSION_CONTENT)
    string(STRIP "${_CACTUS_VERSION_CONTENT}" CACTUS_VERSION_STR)
    target_compile_definitions(cactus_jni PRIVATE CACTUS_COMPILE_TIME_VERSION="${CACTUS_VERSION_STR}")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(cactus_jni PRIVATE CACTUS_ANDROID_ENABLE_LOGGING)
endif()

target_compile_options(cactus_jni PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden -ffunction-sections -fdata-sections)
target_link_options(cactus_jni PRIVATE -Wl,--gc-sections -Wl,--exclude-libs,ALL -flto -Wl,-z,max-page-size=16384)
