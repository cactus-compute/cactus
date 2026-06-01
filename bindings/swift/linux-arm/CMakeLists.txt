cmake_minimum_required(VERSION 3.10)
project(CactusLinuxArm LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../cactus ${CMAKE_CURRENT_BINARY_DIR}/cactus)
