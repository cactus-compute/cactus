cmake_minimum_required(VERSION 3.10)
project(CactusAndroidTests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -Wall -Wextra -pedantic -O3")

set(CACTUS_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../cactus)
set(ANDROID_BUILD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../android/build)
set(CACTUS_LIB ${ANDROID_BUILD_DIR}/libcactus_static.a)
set(CACTUS_CURL_ROOT "$ENV{CACTUS_CURL_ROOT}")
if(NOT CACTUS_CURL_ROOT)
    set(CACTUS_CURL_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../libs/curl)
endif()
set(CACTUS_CURL_INCLUDE_DIR "${CACTUS_CURL_ROOT}/include")
set(CACTUS_CURL_LIBRARY "${CACTUS_CURL_ROOT}/android/arm64-v8a/libcurl.a")
set(CACTUS_MBEDTLS_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../android/mbedtls/arm64-v8a)
if(NOT EXISTS "${CACTUS_MBEDTLS_ROOT}")
    set(CACTUS_MBEDTLS_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../libs/mbedtls/android/arm64-v8a)
endif()
set(CACTUS_MBEDTLS_CRYPTO_LIBRARY "${CACTUS_MBEDTLS_ROOT}/lib/libmbedcrypto.a")
set(CACTUS_MBEDTLS_TLS_LIBRARY "${CACTUS_MBEDTLS_ROOT}/lib/libmbedtls.a")
set(CACTUS_MBEDTLS_X509_LIBRARY "${CACTUS_MBEDTLS_ROOT}/lib/libmbedx509.a")

if(NOT EXISTS ${CACTUS_LIB})
    message(FATAL_ERROR "Cactus library not found at: ${CACTUS_LIB}")
endif()

message(STATUS "Using Cactus library: ${CACTUS_LIB}")
message(STATUS "Using vendored curl root: ${CACTUS_CURL_ROOT}")

set(TESTS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
file(GLOB TEST_SOURCES "${TESTS_DIR}/test*.cpp")
list(REMOVE_ITEM TEST_SOURCES "${TESTS_DIR}/test_utils.cpp")

function(link_vendored_curl_if_present target_name)
    if(CACTUS_CURL_ROOT AND EXISTS "${CACTUS_CURL_INCLUDE_DIR}/curl/curl.h" AND EXISTS "${CACTUS_CURL_LIBRARY}")
        if(NOT EXISTS "${CACTUS_MBEDTLS_CRYPTO_LIBRARY}" OR
           NOT EXISTS "${CACTUS_MBEDTLS_TLS_LIBRARY}" OR
           NOT EXISTS "${CACTUS_MBEDTLS_X509_LIBRARY}")
            message(FATAL_ERROR
                "Vendored Android libcurl is present but mbedTLS static libs are missing.\n"
                "Expected:\n"
                "  ${CACTUS_MBEDTLS_CRYPTO_LIBRARY}\n"
                "  ${CACTUS_MBEDTLS_TLS_LIBRARY}\n"
                "  ${CACTUS_MBEDTLS_X509_LIBRARY}")
        endif()
        target_include_directories(${target_name} PRIVATE "${CACTUS_CURL_INCLUDE_DIR}")
        target_link_libraries(${target_name} PRIVATE
            "${CACTUS_CURL_LIBRARY}"
            "${CACTUS_MBEDTLS_TLS_LIBRARY}"
            "${CACTUS_MBEDTLS_X509_LIBRARY}"
            "${CACTUS_MBEDTLS_CRYPTO_LIBRARY}"
        )
        message(STATUS "Target ${target_name} will link vendored libcurl: ${CACTUS_CURL_LIBRARY}")
    endif()
endfunction()

foreach(TEST_FILE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
    add_executable(${TEST_NAME}
        ${TEST_FILE}
        ${TESTS_DIR}/test_utils.cpp
    )
    target_link_libraries(${TEST_NAME} PRIVATE ${CACTUS_LIB})
    link_vendored_curl_if_present(${TEST_NAME})
    target_include_directories(${TEST_NAME} PRIVATE ${CACTUS_ROOT})
    message(STATUS "Added Android test: ${TEST_NAME}")
endforeach()

add_executable(asr "${TESTS_DIR}/asr.cpp")
target_link_libraries(asr PRIVATE ${CACTUS_LIB})
link_vendored_curl_if_present(asr)
target_include_directories(asr PRIVATE ${CACTUS_ROOT})
message(STATUS "Added Android executable: asr")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
