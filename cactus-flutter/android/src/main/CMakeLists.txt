cmake_minimum_required(VERSION 3.10)

project(cactus)

set(CMAKE_CXX_STANDARD 17)
set(SOURCE_DIR ${CMAKE_SOURCE_DIR}/../../../../cactus)

include_directories(${SOURCE_DIR})
include_directories(${SOURCE_DIR}/ggml-cpu)
include_directories(${SOURCE_DIR}/minja)
include_directories(${SOURCE_DIR}/tools/mtmd)

set(
    SOURCE_FILES
    # Core cactus files
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
    
    # GGML core files
    ${SOURCE_DIR}/ggml.c
    ${SOURCE_DIR}/ggml-alloc.c
    ${SOURCE_DIR}/ggml-backend.cpp
    ${SOURCE_DIR}/ggml-backend-reg.cpp
    ${SOURCE_DIR}/ggml-opt.cpp
    ${SOURCE_DIR}/ggml-threading.cpp
    ${SOURCE_DIR}/ggml-quants.c
    ${SOURCE_DIR}/gguf.cpp
    
    # GGML CPU files
    ${SOURCE_DIR}/ggml-cpu/ggml-cpu.c
    ${SOURCE_DIR}/ggml-cpu/ggml-cpu.cpp
    ${SOURCE_DIR}/ggml-cpu/ggml-cpu-aarch64.cpp
    ${SOURCE_DIR}/ggml-cpu/ggml-cpu-quants.c
    ${SOURCE_DIR}/ggml-cpu/ggml-cpu-traits.cpp
    ${SOURCE_DIR}/ggml-cpu/amx/amx.cpp
    ${SOURCE_DIR}/ggml-cpu/amx/mmq.cpp
    ${SOURCE_DIR}/ggml-cpu/unary-ops.cpp
    ${SOURCE_DIR}/ggml-cpu/binary-ops.cpp
    ${SOURCE_DIR}/ggml-cpu/sgemm.cpp
    ${SOURCE_DIR}/ggml-cpu/vec.cpp
    ${SOURCE_DIR}/ggml-cpu/ops.cpp
    
    # LLaMA files  
    ${SOURCE_DIR}/log.cpp
    ${SOURCE_DIR}/llama-impl.cpp
    ${SOURCE_DIR}/llama-grammar.cpp
    ${SOURCE_DIR}/llama-sampling.cpp
    ${SOURCE_DIR}/llama-vocab.cpp
    ${SOURCE_DIR}/llama-adapter.cpp
    ${SOURCE_DIR}/llama-chat.cpp
    ${SOURCE_DIR}/llama-context.cpp
    ${SOURCE_DIR}/llama-kv-cache.cpp
    ${SOURCE_DIR}/llama-arch.cpp
    ${SOURCE_DIR}/llama-batch.cpp
    ${SOURCE_DIR}/llama-cparams.cpp
    ${SOURCE_DIR}/llama-hparams.cpp
    ${SOURCE_DIR}/llama.cpp
    ${SOURCE_DIR}/llama-model.cpp
    ${SOURCE_DIR}/llama-model-loader.cpp
    ${SOURCE_DIR}/llama-model-saver.cpp
    ${SOURCE_DIR}/llama-mmap.cpp
    ${SOURCE_DIR}/llama-memory.cpp
    ${SOURCE_DIR}/llama-io.cpp
    ${SOURCE_DIR}/llama-graph.cpp
    
    # Common files
    ${SOURCE_DIR}/sampling.cpp
    ${SOURCE_DIR}/unicode-data.cpp
    ${SOURCE_DIR}/unicode.cpp
    ${SOURCE_DIR}/common.cpp
    ${SOURCE_DIR}/chat.cpp
    ${SOURCE_DIR}/json-schema-to-grammar.cpp
    
    # Template and JSON headers (header-only)
    ${SOURCE_DIR}/minja/minja.hpp
    ${SOURCE_DIR}/minja/chat-template.hpp
    ${SOURCE_DIR}/json.hpp
    
    # Multimodal/MTMD files
    ${SOURCE_DIR}/tools/mtmd/mtmd.cpp
    ${SOURCE_DIR}/tools/mtmd/mtmd-audio.cpp
    ${SOURCE_DIR}/tools/mtmd/clip.cpp
    ${SOURCE_DIR}/tools/mtmd/mtmd-helper.cpp

    # FFI bindings
    ${SOURCE_DIR}/cactus_ffi.cpp
)

find_library(LOG_LIB log)

function(build_library target_name cpu_flags)
    add_library(
        ${target_name}
        SHARED
        ${SOURCE_FILES}
    )

    target_link_libraries(${target_name} ${LOG_LIB} android)

    target_compile_options(${target_name} PRIVATE -DLM_GGML_USE_CPU -DLM_GGML_USE_CPU_AARCH64 -pthread ${cpu_flags})

    if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
        target_compile_options(${target_name} PRIVATE -DCACTUS_ANDROID_ENABLE_LOGGING)
    endif ()

    target_compile_options(${target_name} PRIVATE -O3 -DNDEBUG)
    target_compile_options(${target_name} PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
    target_compile_options(${target_name} PRIVATE -ffunction-sections -fdata-sections)

    target_link_options(${target_name} PRIVATE -Wl,--gc-sections)
    target_link_options(${target_name} PRIVATE -Wl,--exclude-libs,ALL)
    target_link_options(${target_name} PRIVATE -flto)

endfunction()

build_library("cactus" "")

if (${ANDROID_ABI} STREQUAL "arm64-v8a")
    build_library("cactus_v8" "-march=armv8-a")
    build_library("cactus_v8_2" "-march=armv8.2-a")
    build_library("cactus_v8_2_dotprod" "-march=armv8.2-a+dotprod")
    build_library("cactus_v8_2_i8mm" "-march=armv8.2-a+i8mm")
    build_library("cactus_v8_2_dotprod_i8mm" "-march=armv8.2-a+dotprod+i8mm")

elseif (${ANDROID_ABI} STREQUAL "x86_64")
    build_library("cactus_x86_64" "-march=x86-64" "-mtune=intel" "-msse4.2" "-mpopcnt")

endif ()