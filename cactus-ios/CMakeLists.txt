cmake_minimum_required(VERSION 3.16)
project(cactus VERSION 1.0.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# iOS specific settings
set(CMAKE_OSX_DEPLOYMENT_TARGET 13.0)
set(CMAKE_XCODE_ATTRIBUTE_ENABLE_BITCODE NO)

# Dependencies and compile options
add_definitions(
    -DNDEBUG
    -DO3
    -DLM_GGML_USE_CPU
    -DLM_GGML_USE_ACCELERATE
    -DLM_GGML_USE_METAL
    -DLM_GGML_METAL_USE_BF16
)

# Set the source directory to the cactus directory
set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../cactus)

# Define public headers
set(PUBLIC_HEADERS
    ${SOURCE_DIR}/cactus.h
    ${SOURCE_DIR}/llama.h
    ${SOURCE_DIR}/llama-impl.h
    ${SOURCE_DIR}/ggml.h
)

# Create library target
add_library(cactus SHARED
    ${SOURCE_DIR}/cactus_context.cpp
    ${SOURCE_DIR}/cactus_loader.cpp
    ${SOURCE_DIR}/cactus_completion.cpp
    ${SOURCE_DIR}/cactus_utils.cpp
    ${SOURCE_DIR}/cactus_embedding.cpp
    ${SOURCE_DIR}/cactus_lora.cpp
    ${SOURCE_DIR}/cactus_tokenization.cpp
    ${SOURCE_DIR}/cactus_multimodal.cpp
    ${SOURCE_DIR}/cactus_tts.cpp
    ${SOURCE_DIR}/cactus_bench.cpp
    ${SOURCE_DIR}/cactus_chat.cpp
    ${SOURCE_DIR}/tools/mtmd/mtmd.cpp
    ${SOURCE_DIR}/tools/mtmd/mtmd-audio.cpp
    ${SOURCE_DIR}/tools/mtmd/clip.cpp
    ${SOURCE_DIR}/tools/mtmd/mtmd-helper.cpp
    ${SOURCE_DIR}/llama.cpp
    ${SOURCE_DIR}/llama-mmap.cpp
    ${SOURCE_DIR}/llama-memory.cpp
    ${SOURCE_DIR}/llama-io.cpp
    ${SOURCE_DIR}/llama-cparams.cpp
    ${SOURCE_DIR}/llama-hparams.cpp
    ${SOURCE_DIR}/llama-model.cpp
    ${SOURCE_DIR}/llama-model-loader.cpp
    ${SOURCE_DIR}/llama-model-saver.cpp
    ${SOURCE_DIR}/llama-kv-cache.cpp
    ${SOURCE_DIR}/llama-context.cpp
    ${SOURCE_DIR}/llama-chat.cpp
    ${SOURCE_DIR}/llama-batch.cpp
    ${SOURCE_DIR}/llama-arch.cpp
    ${SOURCE_DIR}/llama-adapter.cpp
    ${SOURCE_DIR}/llama-sampling.cpp
    ${SOURCE_DIR}/llama-grammar.cpp
    ${SOURCE_DIR}/llama-vocab.cpp
    ${SOURCE_DIR}/llama-impl.cpp
    ${SOURCE_DIR}/llama-graph.cpp
    ${SOURCE_DIR}/ggml.c
    ${SOURCE_DIR}/ggml-alloc.c
    ${SOURCE_DIR}/ggml-backend.cpp
    ${SOURCE_DIR}/ggml-quants.c
    ${SOURCE_DIR}/ggml-opt.cpp
    ${SOURCE_DIR}/ggml-threading.cpp
    ${SOURCE_DIR}/ggml-backend-reg.cpp
    ${SOURCE_DIR}/gguf.cpp
    ${SOURCE_DIR}/common.cpp
    ${SOURCE_DIR}/chat.cpp
    ${SOURCE_DIR}/log.cpp
    ${SOURCE_DIR}/sampling.cpp
    ${SOURCE_DIR}/json-schema-to-grammar.cpp
    ${SOURCE_DIR}/unicode.cpp
    ${SOURCE_DIR}/unicode-data.cpp
    ${SOURCE_DIR}/ggml-cpu/amx/amx.cpp
    ${SOURCE_DIR}/ggml-cpu/amx/mmq.cpp
    ${SOURCE_DIR}/ggml-cpu/ggml-cpu.c
    ${SOURCE_DIR}/ggml-cpu/ggml-cpu.cpp
    ${SOURCE_DIR}/ggml-cpu/ggml-cpu-aarch64.cpp
    ${SOURCE_DIR}/ggml-cpu/ggml-cpu-quants.c
    ${SOURCE_DIR}/ggml-cpu/ggml-cpu-traits.cpp
    ${SOURCE_DIR}/ggml-cpu/unary-ops.cpp
    ${SOURCE_DIR}/ggml-cpu/binary-ops.cpp
    ${SOURCE_DIR}/ggml-cpu/sgemm.cpp
    ${SOURCE_DIR}/ggml-cpu/vec.cpp
    ${SOURCE_DIR}/ggml-cpu/ops.cpp
    ${SOURCE_DIR}/ggml-metal.m
    ${SOURCE_DIR}/minja/minja.hpp
    ${SOURCE_DIR}/minja/chat-template.hpp
    ${SOURCE_DIR}/json.hpp
)

# Setup include directories, use the cactus directory
target_include_directories(cactus
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../cactus>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../cactus/ggml-cpu>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../cactus/minja>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../cactus/tools/mtmd>
        $<INSTALL_INTERFACE:include>
)

# Link required frameworks
target_link_libraries(cactus PRIVATE
    "-framework Accelerate"
    "-framework Foundation"
    "-framework Metal"
    "-framework MetalKit"
)

# Set properties for framework
set_target_properties(cactus PROPERTIES
    MACOSX_FRAMEWORK_IDENTIFIER "com.cactus"
    MACOSX_FRAMEWORK_BUNDLE_VERSION 1.0.0
    MACOSX_FRAMEWORK_SHORT_VERSION_STRING 1.0.0
    FRAMEWORK TRUE
    FRAMEWORK_VERSION 1.0.0
    VERSION 1.0.0
    PUBLIC_HEADER "${PUBLIC_HEADERS}"
    XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC NO
)
