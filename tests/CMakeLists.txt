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
    find_package(CURL)
endif()

set(CACTUS_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../cactus)
set(CACTUS_BUILD_DIR ${CACTUS_ROOT}/build)

if(WIN32 AND EXISTS "${CACTUS_BUILD_DIR}/libcactus.a")
    set(CACTUS_LIB "${CACTUS_BUILD_DIR}/libcactus.a")
    message(STATUS "Using static libcactus.a: ${CACTUS_LIB}")
else()
    find_library(CACTUS_LIB
        NAMES cactus
        HINTS ${CACTUS_BUILD_DIR} ${CACTUS_BUILD_DIR}/lib
        REQUIRED
    )
endif()
 
set(MMAN_LIB "")
if(NOT APPLE)
    if(WIN32)
        get_filename_component(_COMPILER_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
        string(REGEX REPLACE "/bin$" "" _MSYS2_PREFIX "${_COMPILER_DIR}")
        set(_MMAN_LIB "${_MSYS2_PREFIX}/lib/libmman.dll.a")
        if(EXISTS "${_MMAN_LIB}")
            set(MMAN_LIB "${_MMAN_LIB}")
            message(STATUS "Found MSYS2 mman: ${MMAN_LIB}")
        endif()
        set(_CURL_INC "${_MSYS2_PREFIX}/include")
        set(_CURL_LIB "${_MSYS2_PREFIX}/lib/libcurl.dll.a")
        if(EXISTS "${_CURL_INC}/curl/curl.h" AND EXISTS "${_CURL_LIB}")
            set(CURL_INCLUDE_DIRS "${_CURL_INC}")
            set(CURL_LIBRARIES    "${_CURL_LIB}")
            set(CURL_FOUND TRUE)
            message(STATUS "Found MSYS2 curl: ${_CURL_LIB}")
        endif()
    else()
        find_library(MMAN_LIB mman)
        if(MMAN_LIB)
            message(STATUS "Found libmman: ${MMAN_LIB}")
        else()
            message(STATUS "libmman not found; assuming mmap symbols are provided by the C runtime")
        endif()
    endif()
else()
    find_library(COREML_FRAMEWORK CoreML REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
    find_library(SECURITY_FRAMEWORK Security REQUIRED)
    find_library(SYSTEMCONFIGURATION_FRAMEWORK SystemConfiguration REQUIRED)
    find_library(CFNETWORK_FRAMEWORK CFNetwork REQUIRED)
endif()

if(WIN32 AND DEFINED _MSYS2_PREFIX)
    set(_SDL2_INC "${_MSYS2_PREFIX}/include/SDL2")
    set(_SDL2_LIB "${_MSYS2_PREFIX}/lib/libSDL2.dll.a")
    if(EXISTS "${_SDL2_INC}/SDL.h" AND EXISTS "${_SDL2_LIB}")
        set(SDL2_INCLUDE_DIRS "${_MSYS2_PREFIX}/include/SDL2")
        set(SDL2_LIBRARIES    "${_SDL2_LIB}")
        set(SDL2_FOUND TRUE)
        message(STATUS "Found MSYS2 SDL2: ${_SDL2_LIB}")
    endif()
endif()
if(NOT SDL2_FOUND)
    find_package(SDL2)
endif()
if(SDL2_FOUND)
    message(STATUS "Found SDL2: ${SDL2_LIBRARIES}")
else()
    message(STATUS "SDL2 not found; microphone tests will be skipped")
endif()

# test_fastrpc is a standalone FastRPC probe — no libcactus/libmman/SDL2 to avoid
# libmman hooking Windows memory APIs that interfere with libcdsprpc.dll dspqueue.
if(WIN32 AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_fastrpc.cpp")
    add_executable(test_fastrpc test_fastrpc.cpp
        ${CACTUS_ROOT}/npu/fastrpc_drv.cpp)
    target_include_directories(test_fastrpc PRIVATE ${CACTUS_ROOT})
    set(_COMPAT_H "${CACTUS_ROOT}/compat_win32.h")
    set(_COMPAT_INC "${CACTUS_ROOT}/compat_include")
    if(EXISTS "${_COMPAT_H}")
        target_compile_options(test_fastrpc PRIVATE -include "${_COMPAT_H}")
    endif()
    if(EXISTS "${_COMPAT_INC}")
        target_include_directories(test_fastrpc BEFORE PRIVATE "${_COMPAT_INC}")
    endif()
    target_link_libraries(test_fastrpc PRIVATE wintrust crypt32)
    message(STATUS "Added standalone test: test_fastrpc")
endif()

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "test_*.cpp")
list(REMOVE_ITEM TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test_utils.cpp")
list(REMOVE_ITEM TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test_fastrpc.cpp")

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
        target_compile_definitions(${TEST_NAME} PRIVATE HAVE_SDL2 SDL_MAIN_HANDLED)
        message(STATUS "Test ${TEST_NAME} will link with SDL2")
    endif()

    if(APPLE)
        target_link_libraries(${TEST_NAME} PRIVATE ${CACTUS_CURL_LIBRARY_MACOS})
        target_include_directories(${TEST_NAME} PRIVATE ${CACTUS_CURL_INCLUDE_DIR})
        message(STATUS "Test ${TEST_NAME} will link with libcurl")
    elseif(CURL_FOUND)
        target_link_libraries(${TEST_NAME} PRIVATE ${CURL_LIBRARIES})
        target_include_directories(${TEST_NAME} PRIVATE ${CURL_INCLUDE_DIRS})
        message(STATUS "Test ${TEST_NAME} will link with system libcurl")
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
    endif()

    target_include_directories(${TEST_NAME} PRIVATE ${CACTUS_ROOT})

    if(WIN32)
        set(_COMPAT_H "${CACTUS_ROOT}/compat_win32.h")
        set(_COMPAT_INC "${CACTUS_ROOT}/compat_include")
        if(EXISTS "${_COMPAT_H}")
            target_compile_options(${TEST_NAME} PRIVATE -include "${_COMPAT_H}")
        endif()
        if(EXISTS "${_COMPAT_INC}")
            target_include_directories(${TEST_NAME} BEFORE PRIVATE "${_COMPAT_INC}")
        endif()
    endif()

    message(STATUS "Added test: ${TEST_NAME}")
endforeach()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

foreach(APP_NAME chat asr bench_npu_enc)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${APP_NAME}.cpp")
        add_executable(${APP_NAME} ${APP_NAME}.cpp)
        target_link_libraries(${APP_NAME} PRIVATE ${CACTUS_LIB})

        if(MMAN_LIB)
            target_link_libraries(${APP_NAME} PRIVATE ${MMAN_LIB})
        endif()

        if(APPLE)
            target_link_libraries(${APP_NAME} PRIVATE ${CACTUS_CURL_LIBRARY_MACOS})
            target_include_directories(${APP_NAME} PRIVATE ${CACTUS_CURL_INCLUDE_DIR})
        elseif(CURL_FOUND)
            target_link_libraries(${APP_NAME} PRIVATE ${CURL_LIBRARIES})
            target_include_directories(${APP_NAME} PRIVATE ${CURL_INCLUDE_DIRS})
        endif()
        if(APPLE)
            target_link_libraries(${APP_NAME} PRIVATE
                ${ACCELERATE_FRAMEWORK}
                ${COREML_FRAMEWORK}
                ${FOUNDATION_FRAMEWORK}
                ${SECURITY_FRAMEWORK}
                ${SYSTEMCONFIGURATION_FRAMEWORK}
                ${CFNETWORK_FRAMEWORK}
            )
        endif()

        if(SDL2_FOUND AND (APP_NAME STREQUAL "asr" OR APP_NAME STREQUAL "chat"))
            target_link_libraries(${APP_NAME} PRIVATE ${SDL2_LIBRARIES})
            target_include_directories(${APP_NAME} PRIVATE ${SDL2_INCLUDE_DIRS})
            target_compile_definitions(${APP_NAME} PRIVATE HAVE_SDL2 SDL_MAIN_HANDLED)
            message(STATUS "${APP_NAME} will link with SDL2 for live audio")
        endif()

        target_include_directories(${APP_NAME} PRIVATE ${CACTUS_ROOT})

        if(WIN32)
            set(_COMPAT_H "${CACTUS_ROOT}/compat_win32.h")
            set(_COMPAT_INC "${CACTUS_ROOT}/compat_include")
            if(EXISTS "${_COMPAT_H}")
                target_compile_options(${APP_NAME} PRIVATE -include "${_COMPAT_H}")
            endif()
            if(EXISTS "${_COMPAT_INC}")
                target_include_directories(${APP_NAME} BEFORE PRIVATE "${_COMPAT_INC}")
            endif()
        endif()

        message(STATUS "Added executable: ${APP_NAME}")
    endif()
endforeach()
