cmake_minimum_required(VERSION 3.10)
project(CactusApple LANGUAGES CXX)

set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0" CACHE STRING "")
set(CMAKE_XCODE_ATTRIBUTE_IPHONEOS_DEPLOYMENT_TARGET "13.0" CACHE STRING "")
set(CMAKE_BUILD_TYPE Release CACHE STRING "")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if(NOT TARGET cactus_engine)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../cactus ${CMAKE_CURRENT_BINARY_DIR}/cactus)
endif()

set(_DEPS cactus_engine cactus_graph cactus_kernels)

if(BUILD_SHARED_LIBS)
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/stub.cpp "")
    add_library(cactus_framework SHARED ${CMAKE_CURRENT_BINARY_DIR}/stub.cpp)
    add_dependencies(cactus_framework ${_DEPS})

    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
    find_library(COREML_FRAMEWORK CoreML REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(SECURITY_FRAMEWORK Security)
    find_library(SYSTEMCONFIGURATION_FRAMEWORK SystemConfiguration)
    find_library(CFNETWORK_FRAMEWORK CFNetwork)

    target_link_options(cactus_framework PRIVATE
        -force_load $<TARGET_FILE:cactus_engine>
        -force_load $<TARGET_FILE:cactus_graph>
        -force_load $<TARGET_FILE:cactus_kernels>
        -Wl,-dead_strip -flto
    )

    target_link_libraries(cactus_framework PUBLIC
        ${ACCELERATE_FRAMEWORK} ${COREML_FRAMEWORK} ${FOUNDATION_FRAMEWORK}
    )

    string(TOLOWER "${CMAKE_OSX_SYSROOT}" _sysroot_lower)
    if(_sysroot_lower MATCHES "iphonesimulator")
        set(_CURL "${CACTUS_CURL_ROOT}/ios/simulator/libcurl.a")
    elseif(_sysroot_lower MATCHES "iphoneos")
        set(_CURL "${CACTUS_CURL_ROOT}/ios/device/libcurl.a")
    elseif(_sysroot_lower MATCHES "appletvsimulator")
        set(_CURL "${CACTUS_CURL_ROOT}/tvos/simulator/libcurl.a")
    elseif(_sysroot_lower MATCHES "appletvos")
        set(_CURL "${CACTUS_CURL_ROOT}/tvos/device/libcurl.a")
    elseif(_sysroot_lower MATCHES "watchsimulator")
        set(_CURL "${CACTUS_CURL_ROOT}/watchos/simulator/libcurl.a")
    elseif(_sysroot_lower MATCHES "watchos")
        string(TOLOWER "${CMAKE_OSX_ARCHITECTURES}" _archs_lower)
        if(_archs_lower MATCHES "arm64_32")
            set(_CURL "${CACTUS_CURL_ROOT}/watchos/device-arm64_32/libcurl.a")
        elseif(_archs_lower MATCHES "arm64")
            set(_CURL "${CACTUS_CURL_ROOT}/watchos/device-arm64/libcurl.a")
        endif()
    elseif(_sysroot_lower MATCHES "xrsimulator")
        set(_CURL "${CACTUS_CURL_ROOT}/visionos/simulator/libcurl.a")
    elseif(_sysroot_lower MATCHES "xros")
        set(_CURL "${CACTUS_CURL_ROOT}/visionos/device/libcurl.a")
    else()
        set(_CURL "${CACTUS_CURL_ROOT}/macos/libcurl.a")
    endif()
    if(EXISTS "${_CURL}")
        if(NOT SECURITY_FRAMEWORK OR NOT CFNETWORK_FRAMEWORK)
            message(FATAL_ERROR "Required Apple networking frameworks were not found for vendored libcurl linking")
        endif()

        target_link_libraries(cactus_framework PUBLIC "${_CURL}")
        target_link_libraries(cactus_framework PUBLIC ${SECURITY_FRAMEWORK} ${CFNETWORK_FRAMEWORK})

        if(SYSTEMCONFIGURATION_FRAMEWORK)
            target_link_libraries(cactus_framework PUBLIC ${SYSTEMCONFIGURATION_FRAMEWORK})
        endif()
    endif()

    set_target_properties(cactus_framework PROPERTIES
        OUTPUT_NAME cactus
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
            "$<TARGET_FILE_DIR:cactus_framework>/Headers/cactus.h"
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:cactus_framework>/Modules"
        COMMAND ${CMAKE_COMMAND} -E copy
            "${CMAKE_CURRENT_SOURCE_DIR}/module.modulemap"
            "$<TARGET_FILE_DIR:cactus_framework>/Modules/module.modulemap"
    )

    if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
        target_compile_definitions(cactus_framework PRIVATE TARGET_OS_IPHONE=1)
    endif()
endif()
