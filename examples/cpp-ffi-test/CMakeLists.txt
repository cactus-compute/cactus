cmake_minimum_required(VERSION 3.12)
project(cactus_ffi_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)

set(CACTUS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(LLAMA_ROOT "${CACTUS_ROOT}/llama.cpp")

add_subdirectory(${CACTUS_ROOT}/cactus cactus)

add_executable(cactus_ffi_test
    main.cpp
)

target_link_libraries(cactus_ffi_test 
    cactus
)

target_include_directories(cactus_ffi_test PRIVATE
    ${CACTUS_ROOT}
    ${LLAMA_ROOT}/include
) 