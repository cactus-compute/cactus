cmake_minimum_required(VERSION 3.10)
project(CactusAndroid LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(PROJECT_ROOT "${CMAKE_SOURCE_DIR}/..")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT ${ANDROID_ABI} STREQUAL "arm64-v8a")
    message(FATAL_ERROR "Unsupported Android ABI: ${ANDROID_ABI}. Cactus only supports arm64-v8a.")
endif()

set(CACTUS_MBEDTLS_ROOT "${CMAKE_SOURCE_DIR}/mbedtls/${ANDROID_ABI}")
if(NOT EXISTS "${CACTUS_MBEDTLS_ROOT}")
    set(CACTUS_MBEDTLS_ROOT "${PROJECT_ROOT}/libs/mbedtls/android/${ANDROID_ABI}")
endif()
set(CACTUS_MBEDTLS_LIBRARY "${CACTUS_MBEDTLS_ROOT}/lib/libmbedtls.a")
set(CACTUS_MBEDX509_LIBRARY "${CACTUS_MBEDTLS_ROOT}/lib/libmbedx509.a")
set(CACTUS_MBEDCRYPTO_LIBRARY "${CACTUS_MBEDTLS_ROOT}/lib/libmbedcrypto.a")

set(CACTUS_ENGINE_DIR "${PROJECT_ROOT}/cactus-engine")
if(NOT TARGET cactus_engine)
    add_subdirectory(${CACTUS_ENGINE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/cactus-engine)
endif()

find_library(LOG_LIB log)

add_library(cactus SHARED ${CMAKE_SOURCE_DIR}/cactus_jni.cpp)
target_include_directories(cactus PRIVATE ${PROJECT_ROOT}/cactus ${CACTUS_ENGINE_DIR} ${CACTUS_ENGINE_DIR}/src)
target_link_libraries(cactus PRIVATE cactus_engine cactus_graph cactus_kernels)
target_link_libraries(cactus PRIVATE ${LOG_LIB} android)
if(EXISTS "${CACTUS_MBEDTLS_LIBRARY}" AND EXISTS "${CACTUS_MBEDX509_LIBRARY}" AND EXISTS "${CACTUS_MBEDCRYPTO_LIBRARY}")
    target_link_libraries(cactus PRIVATE
        "${CACTUS_MBEDTLS_LIBRARY}"
        "${CACTUS_MBEDX509_LIBRARY}"
        "${CACTUS_MBEDCRYPTO_LIBRARY}"
    )
endif()
target_compile_definitions(cactus PRIVATE PLATFORM_CPU_ONLY=1)

set(_CACTUS_VERSION_FILE "${PROJECT_ROOT}/CACTUS_VERSION")
if(EXISTS "${_CACTUS_VERSION_FILE}")
    file(READ "${_CACTUS_VERSION_FILE}" _CACTUS_VERSION_CONTENT)
    string(STRIP "${_CACTUS_VERSION_CONTENT}" CACTUS_VERSION_STR)
    target_compile_definitions(cactus PRIVATE CACTUS_COMPILE_TIME_VERSION="${CACTUS_VERSION_STR}")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(cactus PRIVATE -DCACTUS_ANDROID_ENABLE_LOGGING)
endif()

target_compile_options(cactus PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
target_link_options(cactus PRIVATE -Wl,--gc-sections -Wl,--exclude-libs,ALL -flto -Wl,-z,max-page-size=16384)

set(MRI_SCRIPT "${CMAKE_BINARY_DIR}/combine.mri")
file(GENERATE OUTPUT ${MRI_SCRIPT} CONTENT
"create ${CMAKE_BINARY_DIR}/lib/libcactus_static.a
addlib $<TARGET_FILE:cactus_engine>
addlib $<TARGET_FILE:cactus_graph>
addlib $<TARGET_FILE:cactus_kernels>
save
end
")
add_custom_target(cactus_static ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/lib
    COMMAND ${CMAKE_AR} -M < ${MRI_SCRIPT}
    DEPENDS cactus_engine cactus_graph cactus_kernels
    COMMENT "Merging static libraries into libcactus_static.a"
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
