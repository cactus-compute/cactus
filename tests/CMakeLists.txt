cmake_minimum_required(VERSION 3.10)
project(CactusTests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES "arm64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -arch arm64 -pthread -Wall -Wextra -pedantic -O3")
    find_package(CURL REQUIRED)
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
endif()

find_package(SDL2)
if(SDL2_FOUND)
    message(STATUS "Found SDL2: ${SDL2_LIBRARIES}")
else()
    message(STATUS "SDL2 not found; microphone tests will be skipped")
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

    if(APPLE AND CURL_FOUND)
        target_link_libraries(${TEST_NAME} PRIVATE ${CURL_LIBRARIES})
        target_include_directories(${TEST_NAME} PRIVATE ${CURL_INCLUDE_DIRS})
        message(STATUS "Test ${TEST_NAME} will link with libcurl")
    endif()


    if(APPLE)
        target_link_libraries(${TEST_NAME} PRIVATE ${COREML_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
    endif()

    target_include_directories(${TEST_NAME} PRIVATE ${CACTUS_ROOT})

    message(STATUS "Added test: ${TEST_NAME}")
endforeach()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
