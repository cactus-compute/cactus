cmake_minimum_required(VERSION 3.10)
project(CactusAndroidTests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT TARGET cactus_engine)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../.. ${CMAKE_CURRENT_BINARY_DIR}/cactus-engine)
endif()

set(TESTS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
file(GLOB TEST_SOURCES "${TESTS_DIR}/test_*.cpp")
list(REMOVE_ITEM TEST_SOURCES "${TESTS_DIR}/test_utils.cpp")

foreach(TEST_FILE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_FILE} ${TESTS_DIR}/test_utils.cpp)
    target_link_libraries(${TEST_NAME} PRIVATE cactus_engine)
endforeach()

if(TARGET test_curl AND EXISTS "${CACTUS_CURL_ROOT}/include/curl/curl.h")
    target_include_directories(test_curl PRIVATE "${CACTUS_CURL_ROOT}/include")
endif()
