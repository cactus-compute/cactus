cmake_minimum_required(VERSION 3.10)
project(Cactus LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(_CACTUS_VERSION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../CACTUS_VERSION")
if(EXISTS "${_CACTUS_VERSION_FILE}")
    file(READ "${_CACTUS_VERSION_FILE}" _CACTUS_VERSION_CONTENT)
    string(STRIP "${_CACTUS_VERSION_CONTENT}" CACTUS_VERSION_STR)
    message(STATUS "Cactus version: ${CACTUS_VERSION_STR}")
else()
    set(CACTUS_VERSION_STR "")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
    message(STATUS "CMAKE_BUILD_TYPE was not set, defaulting to Release.")
endif()

find_package(CURL QUIET)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CACTUS_CURL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../libs/curl" CACHE PATH "Path to vendored libcurl artifacts")

function(cactus_require_vendored_curl include_dir library_path platform_label)
    if(NOT EXISTS "${include_dir}/curl/curl.h" OR NOT EXISTS "${library_path}")
        message(WARNING "==============================================================================")
        message(WARNING " Vendored libcurl is required for ${platform_label} builds but is missing.")
        message(WARNING " Expected header: ${include_dir}/curl/curl.h")
        message(WARNING " Expected library: ${library_path}")
        message(WARNING " Populate libs/curl from your local curl build before running cactus build.")
        message(WARNING "==============================================================================")
        message(FATAL_ERROR "Missing vendored libcurl artifacts for ${platform_label}")
    endif()
endfunction()

if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES "arm64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -arch arm64 -march=armv8.2-a+fp16+simd+dotprod+i8mm -pthread -Wall -Wextra -pedantic -O3 -Wno-missing-field-initializers")
    add_compile_definitions(
        __ARM_NEON=1
        __ARM_FEATURE_FP16_VECTOR_ARITHMETIC=1
        __ARM_FEATURE_DOTPROD=1
        __ARM_FEATURE_MATMUL_INT8=1
        ACCELERATE_NEW_LAPACK
    )

    enable_language(OBJCXX)
    set(CACTUS_CURL_INCLUDE_DIR "${CACTUS_CURL_ROOT}/include")
    set(CACTUS_CURL_LIBRARY_MACOS "${CACTUS_CURL_ROOT}/macos/libcurl.a")
    cactus_require_vendored_curl("${CACTUS_CURL_INCLUDE_DIR}" "${CACTUS_CURL_LIBRARY_MACOS}" "macOS")
    find_library(COREML_FRAMEWORK CoreML REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
    find_library(SECURITY_FRAMEWORK Security REQUIRED)
    find_library(SYSTEMCONFIGURATION_FRAMEWORK SystemConfiguration REQUIRED)
    find_library(CFNETWORK_FRAMEWORK CFNetwork REQUIRED)
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.2-a+fp16+simd+dotprod+i8mm -pthread -Wall -Wextra -pedantic -O3 -Wno-missing-field-initializers")
    add_compile_definitions(
        __ARM_NEON=1
        __ARM_FEATURE_FP16_VECTOR_ARITHMETIC=1
        __ARM_FEATURE_DOTPROD=1
        __ARM_FEATURE_MATMUL_INT8=1
    )
endif()

file(GLOB ENGINE_SOURCES CONFIGURE_DEPENDS "engine/*.cpp")
file(GLOB GRAPH_SOURCES CONFIGURE_DEPENDS "graph/*.cpp")
file(GLOB KERNEL_SOURCES CONFIGURE_DEPENDS "kernel/*.cpp")
file(GLOB FFI_SOURCES CONFIGURE_DEPENDS "ffi/*.cpp")
file(GLOB MODEL_SOURCES CONFIGURE_DEPENDS "models/*.cpp" "models/gemma4/*.cpp")
file(GLOB TELEMETRY_SOURCES CONFIGURE_DEPENDS "telemetry/*.cpp")
set(I8MM_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernel/kernel_i8mm.cpp")
list(REMOVE_ITEM KERNEL_SOURCES "${I8MM_SOURCE}")


if(APPLE)
    set(NPU_SOURCES "npu/npu_ane.mm")
    set_source_files_properties(npu/npu_ane.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
    message(STATUS "Apple platform detected - NPU/ANE acceleration enabled")
else()
    set(NPU_SOURCES "npu/npu.cpp")
endif()

set(COMMON_SOURCES
    ${ENGINE_SOURCES}
    ${GRAPH_SOURCES}
    ${KERNEL_SOURCES}
    ${FFI_SOURCES}
    ${MODEL_SOURCES}
    ${NPU_SOURCES}
    ${TELEMETRY_SOURCES}
)

set(COMMON_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}
    engine
    graph
    kernel
    ffi
    models
    npu
    telemetry
)

function(configure_cactus_target target_name)
    target_include_directories(${target_name} PUBLIC ${COMMON_INCLUDES})

    if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(${target_name} PRIVATE -Wno-c99-extensions)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(${target_name} PRIVATE -Wno-pedantic)
    endif()

    if(NOT "${CACTUS_VERSION_STR}" STREQUAL "")
        target_compile_definitions(${target_name} PRIVATE CACTUS_COMPILE_TIME_VERSION="${CACTUS_VERSION_STR}")
    endif()

    if(APPLE)
        target_link_libraries(${target_name} PUBLIC
            ${COREML_FRAMEWORK}
            ${FOUNDATION_FRAMEWORK}
            ${ACCELERATE_FRAMEWORK}
            ${SECURITY_FRAMEWORK}
            ${SYSTEMCONFIGURATION_FRAMEWORK}
            ${CFNETWORK_FRAMEWORK}
            ${CACTUS_CURL_LIBRARY_MACOS}
        )
        target_include_directories(${target_name} PUBLIC ${CACTUS_CURL_INCLUDE_DIR})
        target_compile_definitions(${target_name} PRIVATE CACTUS_USE_CURL=1)
    elseif(CURL_FOUND)
        target_include_directories(${target_name} PRIVATE ${CURL_INCLUDE_DIRS})
        target_link_libraries(${target_name} PUBLIC ${CURL_LIBRARIES})
        target_compile_definitions(${target_name} PRIVATE CACTUS_USE_CURL)
    endif()
endfunction()


# I8MM support: compile kernel_i8mm.cpp with +i8mm
set(I8MM_MARCH_FLAG "-march=armv8.2-a+fp16+simd+dotprod+i8mm")
add_library(cactus_i8mm_obj OBJECT ${I8MM_SOURCE})
target_include_directories(cactus_i8mm_obj PRIVATE ${COMMON_INCLUDES})
target_compile_options(cactus_i8mm_obj PRIVATE ${I8MM_MARCH_FLAG})
set_target_properties(cactus_i8mm_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(cactus_i8mm_obj PRIVATE CACTUS_COMPILE_I8MM __ARM_FEATURE_MATMUL_INT8=1)

set(EXTRA_OBJECTS $<TARGET_OBJECTS:cactus_i8mm_obj>)

add_library(cactus STATIC ${COMMON_SOURCES} ${EXTRA_OBJECTS})
configure_cactus_target(cactus)
target_compile_definitions(cactus PRIVATE CACTUS_COMPILE_I8MM)


add_library(cactus_ffi SHARED ${COMMON_SOURCES} ${EXTRA_OBJECTS})
configure_cactus_target(cactus_ffi)
set_target_properties(cactus_ffi PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "cactus"
)
target_compile_definitions(cactus_ffi PRIVATE CACTUS_COMPILE_I8MM)
