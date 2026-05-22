cmake_minimum_required(VERSION 3.10)
project(Cactus LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Release CACHE STRING "")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

if(NOT TARGET cactus_engine)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../cactus-engine ${CMAKE_CURRENT_BINARY_DIR}/cactus-engine)
endif()

set(_ARCHIVES $<TARGET_FILE:cactus_engine> $<TARGET_FILE:cactus_graph> $<TARGET_FILE:cactus_kernels>)
set(_DEPS cactus_engine cactus_graph cactus_kernels)

if(APPLE)
    add_custom_target(cactus ALL
        COMMAND libtool -static -o ${CMAKE_BINARY_DIR}/libcactus.a ${_ARCHIVES}
        DEPENDS ${_DEPS}
    )
else()
    file(GENERATE OUTPUT ${CMAKE_BINARY_DIR}/combine.mri CONTENT
        "create ${CMAKE_BINARY_DIR}/libcactus.a\naddlib $<TARGET_FILE:cactus_engine>\naddlib $<TARGET_FILE:cactus_graph>\naddlib $<TARGET_FILE:cactus_kernels>\nsave\nend\n"
    )
    add_custom_target(cactus ALL
        COMMAND ${CMAKE_AR} -M < ${CMAKE_BINARY_DIR}/combine.mri
        DEPENDS ${_DEPS}
    )
endif()

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
file(WRITE ${CMAKE_BINARY_DIR}/stub.cpp "")
add_library(cactus_shared SHARED ${CMAKE_BINARY_DIR}/stub.cpp)
set_target_properties(cactus_shared PROPERTIES OUTPUT_NAME cactus)
add_dependencies(cactus_shared ${_DEPS})

if(APPLE)
    target_link_options(cactus_shared PRIVATE
        -Wl,-force_load,$<TARGET_FILE:cactus_engine>
        -Wl,-force_load,$<TARGET_FILE:cactus_graph>
        -Wl,-force_load,$<TARGET_FILE:cactus_kernels>
    )
    set(_CURL "${CACTUS_CURL_ROOT}/macos/libcurl.a")
    if(EXISTS "${_CURL}")
        target_link_libraries(cactus_shared PRIVATE "${_CURL}")
    endif()
    target_link_libraries(cactus_shared PRIVATE
        "-framework Accelerate" "-framework CoreML" "-framework Foundation"
        "-framework Security" "-framework SystemConfiguration" "-framework CFNetwork"
    )
else()
    target_link_options(cactus_shared PRIVATE -Wl,--whole-archive ${_ARCHIVES} -Wl,--no-whole-archive)
    find_package(CURL QUIET)
    if(CURL_FOUND)
        target_link_libraries(cactus_shared PRIVATE ${CURL_LIBRARIES})
    endif()
endif()
endif()
