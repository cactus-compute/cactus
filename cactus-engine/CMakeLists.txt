cmake_minimum_required(VERSION 3.10)
project(CactusEngine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT TARGET cactus_graph)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../cactus-graph ${CMAKE_CURRENT_BINARY_DIR}/cactus-graph)
endif()

set(CACTUS_CURL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/libs/curl" CACHE PATH "Path to vendored libcurl")

set(ENGINE_SOURCES
    src/bpe.cpp
    src/sp.cpp
    src/constraints.cpp
    src/model.cpp
    src/kv_compress.cpp
    src/engine_image.cpp
    src/index.cpp
    src/index_ffi.cpp
    src/log.cpp
    src/rag.cpp
    src/telemetry.cpp
    src/telemetry_impl.cpp
    src/cloud.cpp
    src/init.cpp
    src/embed.cpp
    src/complete.cpp
    src/transcribe.cpp
    src/stream.cpp
    src/tokenizer.cpp
)
add_library(cactus_engine STATIC ${ENGINE_SOURCES})
set_target_properties(cactus_engine PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME cactus_engine_core
)

target_include_directories(cactus_engine
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
           ${CMAKE_CURRENT_SOURCE_DIR}/libs
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(cactus_engine PUBLIC cactus_graph)

if(APPLE)
    target_link_libraries(cactus_engine PUBLIC
        "-framework Foundation" "-framework Metal" "-framework MetalPerformanceShaders"
        "-framework Security" "-framework SystemConfiguration" "-framework CFNetwork"
    )

    if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
        string(TOLOWER "${CMAKE_OSX_SYSROOT}" _sysroot)
        if(_sysroot MATCHES "simulator")
            set(_curl "${CACTUS_CURL_ROOT}/ios/simulator/libcurl.a")
        else()
            set(_curl "${CACTUS_CURL_ROOT}/ios/device/libcurl.a")
        endif()
    else()
        set(_curl "${CACTUS_CURL_ROOT}/macos/libcurl.a")
    endif()
    if(EXISTS "${_curl}")
        target_link_libraries(cactus_engine PRIVATE "${_curl}")
        target_include_directories(cactus_engine PRIVATE "${CACTUS_CURL_ROOT}/include")
        target_compile_definitions(cactus_engine PRIVATE CACTUS_USE_CURL=1)
    endif()
elseif(ANDROID)
    set(_mbedtls ${CMAKE_CURRENT_SOURCE_DIR}/../android/mbedtls/${ANDROID_ABI}/lib)
    target_link_libraries(cactus_engine PUBLIC
        ${_mbedtls}/libmbedtls.a
        ${_mbedtls}/libmbedx509.a
        ${_mbedtls}/libmbedcrypto.a
        log
        android
    )

    set(_curl "${CACTUS_CURL_ROOT}/android/${ANDROID_ABI}/libcurl.a")
    if(EXISTS "${_curl}")
        target_link_libraries(cactus_engine PRIVATE "${_curl}")
        target_include_directories(cactus_engine PRIVATE "${CACTUS_CURL_ROOT}/include")
        target_compile_definitions(cactus_engine PRIVATE CACTUS_USE_CURL=1)
    endif()
else()
    find_package(CURL QUIET)
    if(CURL_FOUND)
        target_link_libraries(cactus_engine PRIVATE ${CURL_LIBRARIES})
        target_include_directories(cactus_engine PRIVATE ${CURL_INCLUDE_DIRS})
        target_compile_definitions(cactus_engine PRIVATE CACTUS_USE_CURL=1)
    endif()
endif()

set(_version_file "${CMAKE_CURRENT_SOURCE_DIR}/../CACTUS_VERSION")
if(EXISTS "${_version_file}")
    file(READ "${_version_file}" _version)
    string(STRIP "${_version}" _version)
    target_compile_definitions(cactus_engine PRIVATE CACTUS_COMPILE_TIME_VERSION="${_version}")
endif()

target_compile_options(cactus_engine PRIVATE
    -Wall -Wextra -pedantic -Wno-missing-field-initializers
    $<$<CXX_COMPILER_ID:AppleClang,Clang>:-Wno-c99-extensions>
    $<$<CXX_COMPILER_ID:GNU>:-Wno-pedantic>
)

set(_CACTUS_ENGINE_BUNDLE "${CMAKE_BINARY_DIR}/libcactus_engine.a")
set(_CACTUS_ENGINE_ARCHIVES
    $<TARGET_FILE:cactus_engine>
    $<TARGET_FILE:cactus_graph>
    $<TARGET_FILE:cactus_kernels>
)
set(_CACTUS_ENGINE_DEPS cactus_engine cactus_graph cactus_kernels)

if(APPLE)
    add_custom_target(cactus_engine_bundle ALL
        COMMAND libtool -static -o ${_CACTUS_ENGINE_BUNDLE} ${_CACTUS_ENGINE_ARCHIVES}
        DEPENDS ${_CACTUS_ENGINE_DEPS}
        BYPRODUCTS ${_CACTUS_ENGINE_BUNDLE}
    )
else()
    file(GENERATE OUTPUT ${CMAKE_BINARY_DIR}/cactus_engine_bundle.mri CONTENT
        "create ${_CACTUS_ENGINE_BUNDLE}\naddlib $<TARGET_FILE:cactus_engine>\naddlib $<TARGET_FILE:cactus_graph>\naddlib $<TARGET_FILE:cactus_kernels>\nsave\nend\n"
    )
    add_custom_target(cactus_engine_bundle ALL
        COMMAND ${CMAKE_AR} -M < ${CMAKE_BINARY_DIR}/cactus_engine_bundle.mri
        DEPENDS ${_CACTUS_ENGINE_DEPS}
        BYPRODUCTS ${_CACTUS_ENGINE_BUNDLE}
    )
endif()

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    file(WRITE ${CMAKE_BINARY_DIR}/cactus_engine_shared_stub.cpp "")
    add_library(cactus_engine_shared SHARED ${CMAKE_BINARY_DIR}/cactus_engine_shared_stub.cpp)
    set_target_properties(cactus_engine_shared PROPERTIES OUTPUT_NAME cactus_engine)
    add_dependencies(cactus_engine_shared ${_CACTUS_ENGINE_DEPS})
    target_link_libraries(cactus_engine_shared PRIVATE cactus_flags)

    if(APPLE)
        target_link_options(cactus_engine_shared PRIVATE
            -Wl,-force_load,$<TARGET_FILE:cactus_engine>
            -Wl,-force_load,$<TARGET_FILE:cactus_graph>
            -Wl,-force_load,$<TARGET_FILE:cactus_kernels>
        )
        set(_curl "${CACTUS_CURL_ROOT}/macos/libcurl.a")
        if(EXISTS "${_curl}")
            target_link_libraries(cactus_engine_shared PRIVATE "${_curl}")
        endif()
        target_link_libraries(cactus_engine_shared PRIVATE
            "-framework Accelerate" "-framework Foundation" "-framework Metal" "-framework MetalPerformanceShaders"
            "-framework Security" "-framework SystemConfiguration" "-framework CFNetwork"
        )
    else()
        target_link_options(cactus_engine_shared PRIVATE
            -Wl,--whole-archive ${_CACTUS_ENGINE_ARCHIVES} -Wl,--no-whole-archive
        )
        find_package(CURL QUIET)
        if(CURL_FOUND)
            target_link_libraries(cactus_engine_shared PRIVATE ${CURL_LIBRARIES})
        endif()
    endif()
endif()
