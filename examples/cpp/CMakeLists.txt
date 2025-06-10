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

# Create separate executables for each example
add_executable(cactus_vlm main_vlm.cpp)
add_executable(cactus_vlm_ffi main_vlm_ffi.cpp)
add_executable(cactus_embed main_embed.cpp)
add_executable(cactus_llm main_llm.cpp)
add_executable(cactus_tts main_tts.cpp)

# Link each executable to the core library
target_link_libraries(cactus_vlm PRIVATE cactus_core_lib)
target_link_libraries(cactus_vlm_ffi PRIVATE cactus_core_lib)
target_link_libraries(cactus_embed PRIVATE cactus_core_lib)
target_link_libraries(cactus_llm PRIVATE cactus_core_lib)
target_link_libraries(cactus_tts PRIVATE cactus_core_lib)