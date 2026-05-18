cmake_minimum_required(VERSION 3.10)
project(Cactus LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/)

# Build the engine (which pulls in graph -> kernels)
set(CACTUS_ENGINE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../cactus-engine")
if(NOT TARGET cactus_engine)
    add_subdirectory(${CACTUS_ENGINE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/cactus-engine)
endif()

# Produce a combined libcactus.a that contains all symbols from the three layers.
# This is what downstream consumers (tests, mobile builds) link against.
if(APPLE)
    # Use libtool to merge static archives
    add_custom_target(cactus ALL
        COMMAND libtool -static -o ${CMAKE_BINARY_DIR}/libcactus.a
            $<TARGET_FILE:cactus_engine>
            $<TARGET_FILE:cactus_graph>
            $<TARGET_FILE:cactus_kernels>
        DEPENDS cactus_engine cactus_graph cactus_kernels
        COMMENT "Merging static libraries into libcactus.a"
    )
else()
    # Use ar to create a combined archive via MRI script
    set(MRI_SCRIPT "${CMAKE_BINARY_DIR}/combine.mri")
    file(GENERATE OUTPUT ${MRI_SCRIPT} CONTENT
"create ${CMAKE_BINARY_DIR}/libcactus.a
addlib $<TARGET_FILE:cactus_engine>
addlib $<TARGET_FILE:cactus_graph>
addlib $<TARGET_FILE:cactus_kernels>
save
end
")
    add_custom_target(cactus ALL
        COMMAND ${CMAKE_AR} -M < ${MRI_SCRIPT}
        DEPENDS cactus_engine cactus_graph cactus_kernels
        COMMENT "Merging static libraries into libcactus.a"
    )
endif()

file(WRITE ${CMAKE_BINARY_DIR}/cactus_shared_stub.cpp "")
add_library(cactus_shared SHARED ${CMAKE_BINARY_DIR}/cactus_shared_stub.cpp)
set_target_properties(cactus_shared PROPERTIES OUTPUT_NAME cactus)
add_dependencies(cactus_shared cactus_engine cactus_graph cactus_kernels)
if(APPLE)
    target_link_options(cactus_shared PRIVATE
        -Wl,-force_load,$<TARGET_FILE:cactus_engine>
        -Wl,-force_load,$<TARGET_FILE:cactus_graph>
        -Wl,-force_load,$<TARGET_FILE:cactus_kernels>
    )
    set(_CACTUS_CURL_MACOS "${CMAKE_CURRENT_SOURCE_DIR}/../cactus-engine/libs/curl/macos/libcurl.a")
    if(EXISTS ${_CACTUS_CURL_MACOS})
        target_link_libraries(cactus_shared PRIVATE ${_CACTUS_CURL_MACOS})
    endif()
    target_link_libraries(cactus_shared PRIVATE
        "-framework CoreML" "-framework Foundation" "-framework Accelerate"
        "-framework Security" "-framework SystemConfiguration" "-framework CFNetwork"
    )
else()
    target_link_options(cactus_shared PRIVATE
        -Wl,--whole-archive
        $<TARGET_FILE:cactus_engine>
        $<TARGET_FILE:cactus_graph>
        $<TARGET_FILE:cactus_kernels>
        -Wl,--no-whole-archive
    )
endif()
