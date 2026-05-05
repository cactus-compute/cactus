cmake_minimum_required(VERSION 3.10)
project(CactusEngineTests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CACTUS_ENGINE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(CACTUS_PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)

# --- libcurl ---
if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES "arm64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -arch arm64 -pthread -Wall -Wextra -pedantic -O3")
    set(CACTUS_CURL_ROOT "${CACTUS_ENGINE_ROOT}/libs/curl")
    set(CACTUS_CURL_INCLUDE_DIR "${CACTUS_CURL_ROOT}/include")
    set(CACTUS_CURL_LIBRARY_MACOS "${CACTUS_CURL_ROOT}/macos/libcurl.a")
else()
    find_package(CURL)
endif()

# --- Find the pre-built cactus library ---
find_library(CACTUS_LIB
    NAMES cactus
    HINTS ${CACTUS_PROJECT_ROOT}/cactus/build
    REQUIRED
)

# --- Platform libraries ---
if(NOT APPLE)
    find_library(MMAN_LIB mman)
else()
    find_library(COREML_FRAMEWORK CoreML REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
    find_library(SECURITY_FRAMEWORK Security REQUIRED)
    find_library(SYSTEMCONFIGURATION_FRAMEWORK SystemConfiguration REQUIRED)
    find_library(CFNETWORK_FRAMEWORK CFNetwork REQUIRED)
endif()

# --- Test executables ---
file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "test_*.cpp")
list(REMOVE_ITEM TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test_utils.cpp")

foreach(TEST_FILE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_FILE} test_utils.cpp)
    target_link_libraries(${TEST_NAME} PRIVATE ${CACTUS_LIB})

    target_include_directories(${TEST_NAME} PRIVATE
        ${CACTUS_ENGINE_ROOT}
        ${CACTUS_PROJECT_ROOT}/cactus-graph
        ${CACTUS_PROJECT_ROOT}/cactus-kernels
    )

    if(MMAN_LIB)
        target_link_libraries(${TEST_NAME} PRIVATE ${MMAN_LIB})
    endif()

    if(APPLE)
        target_link_libraries(${TEST_NAME} PRIVATE
            ${CACTUS_CURL_LIBRARY_MACOS}
            ${ACCELERATE_FRAMEWORK}
            ${COREML_FRAMEWORK}
            ${FOUNDATION_FRAMEWORK}
            ${SECURITY_FRAMEWORK}
            ${SYSTEMCONFIGURATION_FRAMEWORK}
            ${CFNETWORK_FRAMEWORK}
        )
        target_include_directories(${TEST_NAME} PRIVATE ${CACTUS_CURL_INCLUDE_DIR})
    elseif(CURL_FOUND)
        target_link_libraries(${TEST_NAME} PRIVATE ${CURL_LIBRARIES})
        target_include_directories(${TEST_NAME} PRIVATE ${CURL_INCLUDE_DIRS})
    endif()
endforeach()
