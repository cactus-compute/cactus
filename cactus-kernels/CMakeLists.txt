cmake_minimum_required(VERSION 3.10)
project(CactusKernels LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(cactus_kernels STATIC
    src/attention.cpp
    src/attention_hybrid.cpp
    src/blas.cpp
    src/conv.cpp
    src/conv2d.cpp
    src/dsp.cpp
    src/fused.cpp
    src/image.cpp
    src/lstm.cpp
    src/matmul.cpp
    src/nn.cpp
    src/norms_rope.cpp
    src/quants.cpp
    src/reduce.cpp
    src/scalar.cpp
)

set_target_properties(cactus_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(cactus_kernels
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
           ${CMAKE_CURRENT_SOURCE_DIR}/src
           ${CMAKE_CURRENT_SOURCE_DIR}/libs
)

add_library(cactus_flags INTERFACE)

target_compile_options(cactus_flags INTERFACE
    -march=armv8.2-a+fp16+simd+dotprod+i8mm -pthread -O3
    -fvisibility=hidden -fvisibility-inlines-hidden
    -ffunction-sections -fdata-sections -fno-rtti
    $<$<CXX_COMPILER_ID:Clang,AppleClang>:-flto=thin>
)

target_link_options(cactus_flags INTERFACE
    $<$<CXX_COMPILER_ID:Clang,AppleClang>:-flto=thin>
)

if(APPLE)
    target_link_options(cactus_flags INTERFACE -Wl,-dead_strip)
else()
    target_link_options(cactus_flags INTERFACE -Wl,--gc-sections)
endif()

if(ANDROID)
    target_link_options(cactus_flags INTERFACE -Wl,--exclude-libs,ALL)
endif()

target_link_libraries(cactus_kernels PUBLIC cactus_flags)

target_compile_options(cactus_kernels PRIVATE
    -Wall -Wextra -pedantic -Wno-missing-field-initializers
    $<$<CXX_COMPILER_ID:AppleClang,Clang>:-Wno-c99-extensions>
    $<$<CXX_COMPILER_ID:GNU>:-Wno-pedantic>
)

target_sources(cactus_kernels PRIVATE src/vulkan_backend.cpp)

if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES "arm64")
    target_compile_definitions(cactus_kernels PUBLIC ACCELERATE_NEW_LAPACK)
    target_link_libraries(cactus_kernels PUBLIC "-framework Accelerate")
    enable_language(OBJCXX)
    set(_msl_in  ${CMAKE_CURRENT_SOURCE_DIR}/src/cactus_kernels.metal)
    set(_msl_out ${CMAKE_CURRENT_BINARY_DIR}/generated/cactus_kernels_msl.h)
    add_custom_command(
        OUTPUT  ${_msl_out}
        COMMAND ${CMAKE_COMMAND} -DIN=${_msl_in} -DOUT=${_msl_out} -DSYM=kCactusMSL
                -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_msl.cmake
        DEPENDS ${_msl_in} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_msl.cmake
        VERBATIM)
    target_sources(cactus_kernels PRIVATE src/metal_backend.mm ${_msl_out})
    target_include_directories(cactus_kernels PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)
    set_source_files_properties(src/metal_backend.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
    target_link_libraries(cactus_kernels PUBLIC "-framework Metal" "-framework Foundation" "-framework MetalPerformanceShaders")
else()
    target_sources(cactus_kernels PRIVATE src/metal_backend_stub.cpp)
    target_include_directories(cactus_kernels PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)
    target_link_libraries(cactus_kernels PUBLIC ${CMAKE_DL_LIBS})
    if(ANDROID)
        find_program(GLSLC glslc HINTS
            ${CMAKE_ANDROID_NDK}/shader-tools/${ANDROID_HOST_TAG}
            ${CMAKE_ANDROID_NDK}/shader-tools/darwin-x86_64
            ${CMAKE_ANDROID_NDK}/shader-tools/linux-x86_64)
        if(NOT GLSLC)
            message(FATAL_ERROR "glslc not found in NDK shader-tools")
        endif()
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/generated)
        set(_vk_comp ${CMAKE_CURRENT_SOURCE_DIR}/src/cactus_kernels.comp)
        foreach(_k binary scalar unary swiglu rms_norm cq4_transform cq4_gemv argmax
                   copy cast strided_copy strided_scatter bcast_binary concat2
                   rms_norm_add rms_norm_add_rms rms_norm_scale rope_full softcap
                   adjust_logits argmax3 kv_append_i8 attn_decode_i8 attn_combine ortho_rotate rope_pair
                   cq4_gemv2 cq4_transform_m cq4_gemm
                   rope_pair_rms rms2_add_clip gather softmax_topk topk_rows gemv_bias rms_norm_add_rows
                   attn_prefill_i8 attn_prefill_ring_i8 attn_f16
                   layer_norm softmax_rows glu conv1d_k3 conv1d_dw conv1d_gen conv1d_nlc_dw
                   batchnorm bias_add_rows gemm_batch rel_pos_bias elemwise_chain conv_cache_append
                   binary_f32 scalar_f32 unary_f32 clamp reduce_axis cumsum gather_f32idx
                   maxpool1d bilinear groupnorm conv2d emb_hadamard_m emb_ortho_m attn_fused_i8 attn_fused_i8_512 cq4_gemv_cat cq_gemv_lowbit deltanet_decode deltanet_prefill moe_transform moe_up moe_down_acc)
            set(_spv ${CMAKE_CURRENT_BINARY_DIR}/generated/vk_${_k}.spv)
            set(_hdr ${CMAKE_CURRENT_BINARY_DIR}/generated/vk_${_k}.h)
            add_custom_command(
                OUTPUT  ${_spv}
                COMMAND ${GLSLC} -fshader-stage=compute --target-env=vulkan1.1 -O
                        -DK_${_k} ${_vk_comp} -o ${_spv}
                DEPENDS ${_vk_comp}
                VERBATIM)
            add_custom_command(
                OUTPUT  ${_hdr}
                COMMAND ${CMAKE_COMMAND} -DIN=${_spv} -DOUT=${_hdr} -DSYM=kSpv_${_k}
                        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_msl.cmake
                DEPENDS ${_spv} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_msl.cmake
                VERBATIM)
            target_sources(cactus_kernels PRIVATE ${_hdr})
        endforeach()
    endif()
endif()

if((BUILD_TESTING OR CACTUS_BUILD_TESTS) AND CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    file(GLOB TEST_SOURCES "tests/test_*.cpp")
    foreach(TEST_SRC ${TEST_SOURCES})
        get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)
        add_executable(${TEST_NAME} ${TEST_SRC})
        target_link_libraries(${TEST_NAME} PRIVATE cactus_kernels)
    endforeach()
endif()
