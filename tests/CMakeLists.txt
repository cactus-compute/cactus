cmake_minimum_required(VERSION 3.10)
project(CactusTests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES "arm64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -arch arm64 -pthread -Wall -Wextra -pedantic -O3")
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
    set(MMAN_LIB "mman")
    message(STATUS "Non-Apple system detected: will attempt to link with -lmman for mmap/munmap support")
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
    
    target_include_directories(${TEST_NAME} PRIVATE ${CACTUS_ROOT})
    
    message(STATUS "Added test: ${TEST_NAME}")
endforeach()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
