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
    if(NOT EXISTS "${CACTUS_CURL_LIBRARY_MACOS}")
        unset(CACTUS_CURL_LIBRARY_MACOS)
        find_library(CACTUS_CURL_LIBRARY_MACOS NAMES curl)
    endif()
else()
    find_package(CURL)
endif()

# --- Find the pre-built cactus library ---
set(CACTUS_LIB "${CACTUS_PROJECT_ROOT}/cactus/build/libcactus.a")
if(NOT EXISTS "${CACTUS_LIB}")
    message(FATAL_ERROR "Missing ${CACTUS_LIB}; build the top-level cactus target first")
endif()

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
            ${ACCELERATE_FRAMEWORK}
            ${COREML_FRAMEWORK}
            ${FOUNDATION_FRAMEWORK}
            ${SECURITY_FRAMEWORK}
            ${SYSTEMCONFIGURATION_FRAMEWORK}
            ${CFNETWORK_FRAMEWORK}
        )
        if(CACTUS_CURL_LIBRARY_MACOS)
            target_link_libraries(${TEST_NAME} PRIVATE ${CACTUS_CURL_LIBRARY_MACOS})
            target_include_directories(${TEST_NAME} PRIVATE ${CACTUS_CURL_INCLUDE_DIR})
        endif()
    elseif(CURL_FOUND)
        target_link_libraries(${TEST_NAME} PRIVATE ${CURL_LIBRARIES})
        target_include_directories(${TEST_NAME} PRIVATE ${CURL_INCLUDE_DIRS})
    endif()
endforeach()

# --- chat CLI (depends on SDL2 for audio recording) ---
find_package(SDL2 QUIET CONFIG)
if(NOT SDL2_FOUND)
    find_library(SDL2_LIBRARY NAMES SDL2 PATHS /opt/homebrew/lib /usr/local/lib)
    find_path(SDL2_INCLUDE_DIR SDL.h PATHS /opt/homebrew/include/SDL2 /usr/local/include/SDL2)
    if(SDL2_LIBRARY AND SDL2_INCLUDE_DIR)
        set(SDL2_FOUND TRUE)
    endif()
endif()
if(SDL2_FOUND)
    add_executable(chat chat.cpp test_utils.cpp)
    target_link_libraries(chat PRIVATE ${CACTUS_LIB})
    if(TARGET SDL2::SDL2)
        target_link_libraries(chat PRIVATE SDL2::SDL2)
    else()
        target_link_libraries(chat PRIVATE ${SDL2_LIBRARY})
        target_include_directories(chat PRIVATE ${SDL2_INCLUDE_DIR})
    endif()
    target_include_directories(chat PRIVATE
        ${CACTUS_ENGINE_ROOT}
        ${CACTUS_PROJECT_ROOT}/cactus-graph
        ${CACTUS_PROJECT_ROOT}/cactus-kernels
    )
    if(MMAN_LIB)
        target_link_libraries(chat PRIVATE ${MMAN_LIB})
    endif()
    if(APPLE)
        target_link_libraries(chat PRIVATE
            ${ACCELERATE_FRAMEWORK}
            ${COREML_FRAMEWORK}
            ${FOUNDATION_FRAMEWORK}
            ${SECURITY_FRAMEWORK}
            ${SYSTEMCONFIGURATION_FRAMEWORK}
            ${CFNETWORK_FRAMEWORK}
        )
        if(CACTUS_CURL_LIBRARY_MACOS)
            target_link_libraries(chat PRIVATE ${CACTUS_CURL_LIBRARY_MACOS})
            target_include_directories(chat PRIVATE ${CACTUS_CURL_INCLUDE_DIR})
        endif()
    elseif(CURL_FOUND)
        target_link_libraries(chat PRIVATE ${CURL_LIBRARIES})
        target_include_directories(chat PRIVATE ${CURL_INCLUDE_DIRS})
    endif()
else()
    message(WARNING "SDL2 not found; chat binary will not be built")
endif()

# --- asr CLI (audio transcription via cactus_transcribe) ---
add_executable(asr asr.cpp)
target_link_libraries(asr PRIVATE ${CACTUS_LIB})
target_include_directories(asr PRIVATE
    ${CACTUS_ENGINE_ROOT}
    ${CACTUS_PROJECT_ROOT}/cactus-graph
    ${CACTUS_PROJECT_ROOT}/cactus-kernels
)
if(MMAN_LIB)
    target_link_libraries(asr PRIVATE ${MMAN_LIB})
endif()
if(APPLE)
    target_link_libraries(asr PRIVATE
        ${ACCELERATE_FRAMEWORK}
        ${COREML_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
        ${SECURITY_FRAMEWORK}
        ${SYSTEMCONFIGURATION_FRAMEWORK}
        ${CFNETWORK_FRAMEWORK}
    )
    if(CACTUS_CURL_LIBRARY_MACOS)
        target_link_libraries(asr PRIVATE ${CACTUS_CURL_LIBRARY_MACOS})
        target_include_directories(asr PRIVATE ${CACTUS_CURL_INCLUDE_DIR})
    endif()
elseif(CURL_FOUND)
    target_link_libraries(asr PRIVATE ${CURL_LIBRARIES})
    target_include_directories(asr PRIVATE ${CURL_INCLUDE_DIRS})
endif()
