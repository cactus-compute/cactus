cmake_minimum_required(VERSION 3.16)
project(CactusLocalIntegrationTests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT CACTUS_REPO_ROOT)
    get_filename_component(CACTUS_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
endif()

set(CACTUS_LIB "${CACTUS_REPO_ROOT}/cactus/build/libcactus.a")
if(NOT EXISTS "${CACTUS_LIB}")
    message(FATAL_ERROR "Missing ${CACTUS_LIB}; run ./cactus/build.sh first")
endif()

add_executable(integration_runner native_runner.cpp)
target_link_libraries(integration_runner PRIVATE "${CACTUS_LIB}")
target_include_directories(integration_runner PRIVATE
    "${CACTUS_REPO_ROOT}/cactus-engine"
    "${CACTUS_REPO_ROOT}/cactus-engine/src"
    "${CACTUS_REPO_ROOT}/cactus-graph"
    "${CACTUS_REPO_ROOT}/cactus-kernels"
    "${CACTUS_REPO_ROOT}/cactus-kernels/src"
    "${CACTUS_REPO_ROOT}/cactus-kernels/libs"
)

find_package(Threads REQUIRED)
target_link_libraries(integration_runner PRIVATE Threads::Threads)

if(APPLE)
    find_library(COREML_FRAMEWORK CoreML REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
    find_library(SECURITY_FRAMEWORK Security REQUIRED)
    find_library(SYSTEMCONFIGURATION_FRAMEWORK SystemConfiguration REQUIRED)
    find_library(CFNETWORK_FRAMEWORK CFNetwork REQUIRED)
    target_link_libraries(integration_runner PRIVATE
        "${ACCELERATE_FRAMEWORK}"
        "${COREML_FRAMEWORK}"
        "${FOUNDATION_FRAMEWORK}"
        "${SECURITY_FRAMEWORK}"
        "${SYSTEMCONFIGURATION_FRAMEWORK}"
        "${CFNETWORK_FRAMEWORK}"
    )
    set(CACTUS_CURL_ROOT "${CACTUS_REPO_ROOT}/cactus-engine/libs/curl")
    set(CACTUS_CURL_LIBRARY_MACOS "${CACTUS_CURL_ROOT}/macos/libcurl.a")
    if(EXISTS "${CACTUS_CURL_LIBRARY_MACOS}")
        target_link_libraries(integration_runner PRIVATE "${CACTUS_CURL_LIBRARY_MACOS}")
        target_include_directories(integration_runner PRIVATE "${CACTUS_CURL_ROOT}/include")
    else()
        find_library(CURL_LIBRARY NAMES curl)
        if(CURL_LIBRARY)
            target_link_libraries(integration_runner PRIVATE "${CURL_LIBRARY}")
        endif()
    endif()
else()
    find_package(CURL)
    if(CURL_FOUND)
        target_link_libraries(integration_runner PRIVATE ${CURL_LIBRARIES})
        target_include_directories(integration_runner PRIVATE ${CURL_INCLUDE_DIRS})
    endif()
endif()
