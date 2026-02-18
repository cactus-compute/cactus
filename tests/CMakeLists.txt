cmake_minimum_required(VERSION 3.10)
project(CactusTests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

function(cactus_require_vendored_curl include_dir library_path platform_label)
    if(NOT EXISTS "${include_dir}/curl/curl.h" OR NOT EXISTS "${library_path}")
        message(WARNING "==============================================================================")
        message(WARNING " Vendored libcurl is required for ${platform_label} tests but is missing.")
        message(WARNING " Expected header: ${include_dir}/curl/curl.h")
        message(WARNING " Expected library: ${library_path}")
        message(WARNING " Populate libs/curl from your local curl build before running cactus test.")
        message(WARNING "==============================================================================")
        message(FATAL_ERROR "Missing vendored libcurl artifacts for ${platform_label} tests")
    endif()
endfunction()

if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES "arm64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -arch arm64 -pthread -Wall -Wextra -pedantic -O3")
    set(CACTUS_CURL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../libs/curl" CACHE PATH "Path to vendored libcurl artifacts")
    set(CACTUS_CURL_INCLUDE_DIR "${CACTUS_CURL_ROOT}/include")
    set(CACTUS_CURL_LIBRARY_MACOS "${CACTUS_CURL_ROOT}/macos/libcurl.a")
    cactus_require_vendored_curl("${CACTUS_CURL_INCLUDE_DIR}" "${CACTUS_CURL_LIBRARY_MACOS}" "macOS")
else()
endif()

set(CACTUS_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../cactus)
set(CACTUS_BUILD_DIR ${CACTUS_ROOT}/build)

find_library(CACTUS_LIB 
    NAMES cactus
    PATHS ${CACTUS_BUILD_DIR} ${CACTUS_BUILD_DIR}/lib
    REQUIRED
)
 
set(MMAN_LIB "")
if(NOT APPLE)
    find_library(MMAN_LIB mman)
    if(MMAN_LIB)
        message(STATUS "Found libmman: ${MMAN_LIB}")
    else()
        message(STATUS "libmman not found; assuming mmap symbols are provided by the C runtime")
    endif()
else()
    find_library(COREML_FRAMEWORK CoreML REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(SECURITY_FRAMEWORK Security REQUIRED)
    find_library(SYSTEMCONFIGURATION_FRAMEWORK SystemConfiguration REQUIRED)
    find_library(CFNETWORK_FRAMEWORK CFNetwork REQUIRED)
endif()

find_package(SDL2)
if(SDL2_FOUND)
    message(STATUS "Found SDL2: ${SDL2_LIBRARIES}")
else()
    message(STATUS "SDL2 not found; microphone tests will be skipped")
endif()

file(GLOB TEST_SOURCES "test_*.cpp")
list(REMOVE_ITEM TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test_utils.cpp")

foreach(TEST_FILE ${TEST_SOURCES})

    get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_FILE} test_utils.cpp)
    target_link_libraries(${TEST_NAME} PRIVATE ${CACTUS_LIB})


    if(MMAN_LIB)
        target_link_libraries(${TEST_NAME} PRIVATE ${MMAN_LIB})
        message(STATUS "Test ${TEST_NAME} will link with -lmman")
    endif()

    if(SDL2_FOUND)
        target_link_libraries(${TEST_NAME} PRIVATE ${SDL2_LIBRARIES})
        target_include_directories(${TEST_NAME} PRIVATE ${SDL2_INCLUDE_DIRS})
        target_compile_definitions(${TEST_NAME} PRIVATE HAVE_SDL2)
        message(STATUS "Test ${TEST_NAME} will link with SDL2")
    endif()

    if(APPLE)
        target_link_libraries(${TEST_NAME} PRIVATE ${CACTUS_CURL_LIBRARY_MACOS})
        target_include_directories(${TEST_NAME} PRIVATE ${CACTUS_CURL_INCLUDE_DIR})
        message(STATUS "Test ${TEST_NAME} will link with libcurl")
    endif()
    if(APPLE)
        target_link_libraries(${TEST_NAME} PRIVATE
            ${COREML_FRAMEWORK}
            ${FOUNDATION_FRAMEWORK}
            ${SECURITY_FRAMEWORK}
            ${SYSTEMCONFIGURATION_FRAMEWORK}
            ${CFNETWORK_FRAMEWORK}
        )
    endif()

    target_include_directories(${TEST_NAME} PRIVATE ${CACTUS_ROOT})

    message(STATUS "Added test: ${TEST_NAME}")
endforeach()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

foreach(APP_NAME chat asr)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${APP_NAME}.cpp")
        add_executable(${APP_NAME} ${APP_NAME}.cpp)
        target_link_libraries(${APP_NAME} PRIVATE ${CACTUS_LIB})

        if(MMAN_LIB)
            target_link_libraries(${APP_NAME} PRIVATE ${MMAN_LIB})
        endif()

        if(APPLE)
            target_link_libraries(${APP_NAME} PRIVATE ${CACTUS_CURL_LIBRARY_MACOS})
            target_include_directories(${APP_NAME} PRIVATE ${CACTUS_CURL_INCLUDE_DIR})
        endif()
        if(APPLE)
            target_link_libraries(${APP_NAME} PRIVATE
                ${COREML_FRAMEWORK}
                ${FOUNDATION_FRAMEWORK}
                ${SECURITY_FRAMEWORK}
                ${SYSTEMCONFIGURATION_FRAMEWORK}
                ${CFNETWORK_FRAMEWORK}
            )
        endif()

        if(SDL2_FOUND AND APP_NAME STREQUAL "asr")
            target_link_libraries(${APP_NAME} PRIVATE ${SDL2_LIBRARIES})
            target_include_directories(${APP_NAME} PRIVATE ${SDL2_INCLUDE_DIRS})
            target_compile_definitions(${APP_NAME} PRIVATE HAVE_SDL2)
            message(STATUS "asr will link with SDL2 for live transcription")
        endif()

        target_include_directories(${APP_NAME} PRIVATE ${CACTUS_ROOT})
        message(STATUS "Added executable: ${APP_NAME}")
    endif()
endforeach()
