cmake_minimum_required(VERSION 3.10)
project(CactusApple LANGUAGES CXX)

set(CMAKE_OSX_DEPLOYMENT_TARGET "13.3" CACHE STRING "")
set(CMAKE_XCODE_ATTRIBUTE_IPHONEOS_DEPLOYMENT_TARGET "16.4" CACHE STRING "")
set(CMAKE_BUILD_TYPE Release CACHE STRING "")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if(NOT TARGET cactus_engine)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../cactus-engine ${CMAKE_CURRENT_BINARY_DIR}/cactus-engine)
endif()

set(_DEPS cactus_engine cactus_graph cactus_kernels)

if(BUILD_SHARED_LIBS)
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/stub.cpp "")
    add_library(cactus_framework SHARED ${CMAKE_CURRENT_BINARY_DIR}/stub.cpp)
    set_target_properties(cactus_framework PROPERTIES OUTPUT_NAME cactus)
    add_dependencies(cactus_framework ${_DEPS})

    target_link_libraries(cactus_framework PRIVATE cactus_flags)

    target_link_options(cactus_framework PRIVATE
        -force_load $<TARGET_FILE:cactus_engine>
        -force_load $<TARGET_FILE:cactus_graph>
        -force_load $<TARGET_FILE:cactus_kernels>
    )

    target_link_libraries(cactus_framework PUBLIC
        "-framework Accelerate" "-framework CoreML" "-framework Foundation" "-framework Metal" "-framework MetalPerformanceShaders"
        "-framework Security" "-framework SystemConfiguration" "-framework CFNetwork"
    )

    if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
        string(TOLOWER "${CMAKE_OSX_SYSROOT}" _sysroot_lower)
        if(_sysroot_lower MATCHES "simulator")
            set(_CURL "${CACTUS_CURL_ROOT}/ios/simulator/libcurl.a")
        else()
            set(_CURL "${CACTUS_CURL_ROOT}/ios/device/libcurl.a")
        endif()
    else()
        set(_CURL "${CACTUS_CURL_ROOT}/macos/libcurl.a")
    endif()
    if(EXISTS "${_CURL}")
        target_link_libraries(cactus_framework PUBLIC "${_CURL}")
    endif()

    set_target_properties(cactus_framework PROPERTIES
        FRAMEWORK TRUE
        FRAMEWORK_VERSION A
        MACOSX_FRAMEWORK_IDENTIFIER com.cactuscompute.cactus
        VERSION 1.0.0
        SOVERSION 1.0.0
        MACOSX_FRAMEWORK_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/Info.plist"
    )

    add_custom_command(TARGET cactus_framework POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:cactus_framework>/Headers"
        COMMAND ${CMAKE_COMMAND} -E copy
            "${CMAKE_CURRENT_SOURCE_DIR}/../cactus-engine/cactus_engine.h"
            "$<TARGET_FILE_DIR:cactus_framework>/Headers"
    )

    if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
        target_compile_definitions(cactus_framework PRIVATE TARGET_OS_IPHONE=1)
    endif()
endif()
