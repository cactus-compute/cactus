cmake_minimum_required(VERSION 3.10)
project(CactusApple LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0" CACHE STRING "Minimum macOS version")
set(CMAKE_XCODE_ATTRIBUTE_IPHONEOS_DEPLOYMENT_TARGET "13.0" CACHE STRING "Minimum iOS version")

set(PROJECT_ROOT "${CMAKE_SOURCE_DIR}/..")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CACTUS_ENGINE_DIR "${PROJECT_ROOT}/cactus-engine")
if(NOT TARGET cactus_engine)
    add_subdirectory(${CACTUS_ENGINE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/cactus-engine)
endif()

if(BUILD_SHARED_LIBS)
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/empty.cpp" "")
    add_library(cactus SHARED "${CMAKE_CURRENT_BINARY_DIR}/empty.cpp")
    add_dependencies(cactus cactus_engine cactus_graph cactus_kernels)

    target_link_options(cactus PRIVATE
        -force_load $<TARGET_FILE:cactus_engine>
        -force_load $<TARGET_FILE:cactus_graph>
        -force_load $<TARGET_FILE:cactus_kernels>
    )

    find_library(COREML_FRAMEWORK CoreML REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
    find_library(SECURITY_FRAMEWORK Security REQUIRED)
    find_library(SYSTEMCONFIGURATION_FRAMEWORK SystemConfiguration REQUIRED)
    find_library(CFNETWORK_FRAMEWORK CFNetwork REQUIRED)
    target_link_libraries(cactus PUBLIC
        ${COREML_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
        ${ACCELERATE_FRAMEWORK}
        ${SECURITY_FRAMEWORK}
        ${SYSTEMCONFIGURATION_FRAMEWORK}
        ${CFNETWORK_FRAMEWORK}
    )

    set(CACTUS_CURL_INCLUDE_DIR "${CACTUS_CURL_ROOT}/include")
    if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
        string(TOLOWER "${CMAKE_OSX_SYSROOT}" _sysroot_lower)
        if(_sysroot_lower MATCHES "simulator")
            set(CACTUS_CURL_LIBRARY "${CACTUS_CURL_ROOT}/ios/simulator/libcurl.a")
        else()
            set(CACTUS_CURL_LIBRARY "${CACTUS_CURL_ROOT}/ios/device/libcurl.a")
        endif()
    else()
        set(CACTUS_CURL_LIBRARY "${CACTUS_CURL_ROOT}/macos/libcurl.a")
    endif()
    if(EXISTS "${CACTUS_CURL_INCLUDE_DIR}/curl/curl.h" AND EXISTS "${CACTUS_CURL_LIBRARY}")
        target_link_libraries(cactus PUBLIC "${CACTUS_CURL_LIBRARY}")
        target_include_directories(cactus PUBLIC "${CACTUS_CURL_INCLUDE_DIR}")
    endif()

    set_target_properties(cactus PROPERTIES
        FRAMEWORK TRUE
        FRAMEWORK_VERSION A
        MACOSX_FRAMEWORK_IDENTIFIER com.cactuscompute.cactus
        VERSION 1.0.0
        SOVERSION 1.0.0
        PUBLIC_HEADER "${CACTUS_ENGINE_DIR}/cactus_engine.h"
        MACOSX_FRAMEWORK_INFO_PLIST "${CMAKE_SOURCE_DIR}/Info.plist"
    )

    if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
        target_compile_definitions(cactus PRIVATE TARGET_OS_IPHONE=1)
    endif()

    target_link_options(cactus PRIVATE -Wl,-dead_strip -flto)
else()
    add_custom_target(cactus ALL
        COMMAND libtool -static -o ${CMAKE_BINARY_DIR}/libcactus.a
            $<TARGET_FILE:cactus_engine>
            $<TARGET_FILE:cactus_graph>
            $<TARGET_FILE:cactus_kernels>
        DEPENDS cactus_engine cactus_graph cactus_kernels
        COMMENT "Merging static libraries into libcactus.a"
    )
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
