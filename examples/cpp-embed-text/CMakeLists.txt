cmake_minimum_required(VERSION 3.10)
project(cactus_vlm)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    add_compile_options(-g -O0 -fsanitize=address -fno-omit-frame-pointer)
    link_libraries(-fsanitize=address)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_subdirectory(../../cactus ${CMAKE_BINARY_DIR}/cactus_core_build)

add_executable(cactus_embedding
    main.cpp
)

target_link_libraries(cactus_embedding
    PRIVATE
    cactus_core_lib
)