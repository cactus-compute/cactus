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
