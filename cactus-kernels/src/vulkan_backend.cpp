#include "vulkan_backend.h"

#if !defined(__ANDROID__)

bool cactus_vulkan_available() { return false; }
const char* cactus_vulkan_device_info() { return "Vulkan: unavailable (non-Android build)"; }
bool cactus_vulkan_op_enabled(const char*) { return false; }
void cactus_vulkan_session_begin() {}
void cactus_vulkan_session_flush() {}
void cactus_vulkan_session_sync() {}
void cactus_vulkan_session_end() {}
void cactus_vulkan_invalidate_host_wraps() {}
void cactus_vulkan_trim_prefill_cache() {}
void* cactus_vulkan_alloc_shared(size_t) { return nullptr; }
void* cactus_vulkan_alloc_pooled(size_t) { return nullptr; }
void cactus_vulkan_free_shared(void*) {}
bool cactus_vulkan_encode_binary_f16(int, void*, const void*, const void*, size_t) { return false; }
bool cactus_vulkan_encode_scalar_f16(int, void*, const void*, size_t, float) { return false; }
bool cactus_vulkan_encode_unary_f16(int, void*, const void*, size_t) { return false; }
bool cactus_vulkan_encode_swiglu_f16(void*, const void*, const void*, size_t, float) { return false; }
bool cactus_vulkan_encode_rms_norm_f16(void*, const void*, const void*, size_t, size_t, float) { return false; }
bool cactus_vulkan_encode_cq_gemv(void*, const void*, const CactusQuantMatrix*) { return false; }
bool cactus_vulkan_encode_copy(void*, const void*, size_t) { return false; }
bool cactus_vulkan_encode_cast(void*, int, const void*, int, size_t) { return false; }
bool cactus_vulkan_encode_strided_copy(void*, const void*, const uint32_t*, const uint32_t*, uint32_t, uint32_t, uint32_t, size_t, size_t) { return false; }
bool cactus_vulkan_encode_strided_scatter(void*, const void*, const uint32_t*, const uint32_t*, uint32_t, uint32_t, uint32_t, size_t, size_t) { return false; }
bool cactus_vulkan_encode_bcast_binary(int, void*, const void*, const void*, const uint32_t*, const uint32_t*, const uint32_t*, uint32_t, uint32_t, size_t, size_t, size_t) { return false; }
bool cactus_vulkan_encode_concat2(void*, const void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) { return false; }
bool cactus_vulkan_encode_rms_norm_add(void*, const void*, const void*, const void*, size_t, size_t, float, float) { return false; }
bool cactus_vulkan_encode_rms_norm_add_rms(void*, void*, const void*, const void*, const void*, const void*, size_t, size_t, float, float) { return false; }
bool cactus_vulkan_encode_rms_norm_scale(void*, const void*, const void*, size_t, size_t, float, float) { return false; }
bool cactus_vulkan_encode_rope_full(void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float, int) { return false; }
bool cactus_vulkan_encode_softcap(void*, const void*, size_t, float) { return false; }
bool cactus_vulkan_encode_adjust_logits(void*, size_t, const uint32_t*, uint32_t, int64_t, float) { return false; }
bool cactus_vulkan_encode_argmax(const void*, uint32_t, void*, const void*) { return false; }
bool cactus_vulkan_encode_kv_append_i8(const void*, void*, void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, size_t, size_t, size_t) { return false; }
bool cactus_vulkan_encode_kv_append_sliding_i8(const void*, void*, void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, size_t, size_t, size_t) { return false; }
bool cactus_vulkan_prewarm_quant(const CactusQuantMatrix*) { return false; }
bool cactus_vulkan_encode_transform_batch(const void*, const CactusQuantMatrix* const*, int, void* const*) { return false; }
bool cactus_vulkan_encode_gemv_precoded(void*, const void*, const CactusQuantMatrix*) { return false; }
bool cactus_vulkan_encode_quant_matmul_m(void*, const void*, const CactusQuantMatrix*, uint32_t) { return false; }
bool cactus_vulkan_encode_rope_pair(void*, const void*, const void*, const void*, uint32_t, uint32_t) { return false; }
bool cactus_vulkan_encode_attention_i8(void*, const void*, const void*, const void*, const void*, const void*, const void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float, size_t, size_t, size_t, size_t) { return false; }
bool cactus_vulkan_encode_rope_pair_rms(void*, const void*, const void*, const void*, const void*, uint32_t, uint32_t, float) { return false; }
bool cactus_vulkan_encode_rms2_add_clip(void*, const void*, const void*, const void*, const void*, size_t, float, float) { return false; }
bool cactus_vulkan_encode_gather_f16(void*, const void*, size_t, const uint32_t*, uint32_t, uint32_t) { return false; }
bool cactus_vulkan_encode_softmax_topk(void*, void*, const void*, size_t, size_t, size_t, float) { return false; }
bool cactus_vulkan_encode_topk_rows(void*, const void*, size_t, size_t, size_t) { return false; }
bool cactus_vulkan_encode_gemv_bias(void*, const void*, const void*, const void*, uint32_t, uint32_t, int) { return false; }
bool cactus_vulkan_encode_rms_norm_add_rows(void*, void*, const void*, const void*, const void*, uint32_t, uint32_t, float, int) { return false; }
bool cactus_vulkan_transform_gemv_fits(uint32_t) { return false; }
bool cactus_vulkan_encode_transform_gemv(void*, const void*, const CactusQuantMatrix*, const void*) { return false; }
bool cactus_vulkan_encode_swiglu_transform(void*, const void*, const void*, const CactusQuantMatrix*, float) { return false; }
bool cactus_vulkan_encode_attention_i8_prefill(void*, const void*, const void*, const void*, const void*, const void*, const void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float, size_t, size_t, size_t, size_t, uint32_t, uint32_t) { return false; }
bool cactus_vulkan_encode_attention_f16(void*, const void*, const void*, const void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float, uint32_t, uint32_t, uint32_t, float, uint32_t) { return false; }
bool cactus_vulkan_encode_layer_norm(void*, const void*, const void*, const void*, size_t, size_t, float) { return false; }
bool cactus_vulkan_encode_softmax_rows(void*, const void*, size_t, size_t) { return false; }
bool cactus_vulkan_encode_glu(void*, const void*, size_t, size_t, size_t) { return false; }
bool cactus_vulkan_encode_conv1d_k3(void*, const void*, const void*, int, const void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) { return false; }
bool cactus_vulkan_encode_conv1d_dw(void*, const void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) { return false; }
bool cactus_vulkan_encode_conv1d_gen(void*, const void*, const void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, int) { return false; }
bool cactus_vulkan_encode_conv1d_nlc_dw(void*, const void*, const void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) { return false; }
bool cactus_vulkan_encode_batchnorm(void*, const void*, const void*, const void*, const void*, const void*, uint32_t, uint32_t, uint32_t, float) { return false; }
bool cactus_vulkan_encode_bias_add_rows(void*, const void*, uint32_t, uint32_t) { return false; }
bool cactus_vulkan_encode_gemm_batch(void*, const void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, int, int) { return false; }
bool cactus_vulkan_encode_gemm_f16(void*, const void*, const void*, uint32_t, uint32_t, uint32_t, int) { return false; }
bool cactus_vulkan_encode_rel_pos_bias(void*, const void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, int, float) { return false; }
bool cactus_vulkan_encode_elemwise_chain(void*, const void*, const float*, uint32_t, const void*, const void*, const void*, const size_t*, size_t, uint32_t, uint32_t) { return false; }
bool cactus_vulkan_encode_conv_cache_append(void*, const void*, void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, int) { return false; }
bool cactus_vulkan_encode_binary_f32(int, void*, const void*, const void*, size_t) { return false; }
bool cactus_vulkan_encode_scalar_f32(int, void*, const void*, size_t, float) { return false; }
bool cactus_vulkan_encode_unary_f32(int, void*, const void*, size_t) { return false; }
bool cactus_vulkan_encode_clamp(void*, const void*, size_t, float, float, int) { return false; }
bool cactus_vulkan_encode_reduce_axis(int, void*, const void*, uint32_t, uint32_t, uint32_t, int) { return false; }
bool cactus_vulkan_encode_cumsum(void*, const void*, uint32_t, uint32_t, uint32_t, int) { return false; }
bool cactus_vulkan_encode_gather_f32idx(void*, const void*, const void*, uint32_t, uint32_t, size_t) { return false; }
bool cactus_vulkan_encode_maxpool1d(void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) { return false; }
bool cactus_vulkan_encode_bilinear(void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, int) { return false; }
bool cactus_vulkan_encode_groupnorm(void*, const void*, const void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, float) { return false; }
bool cactus_vulkan_encode_conv2d(void*, const void*, const void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, int) { return false; }
bool cactus_vulkan_encode_embedding_hadamard_m(void*, const CactusQuantMatrix*, const uint32_t*, uint32_t) { return false; }
bool cactus_vulkan_encode_embedding_hadamard(void*, uint32_t, const CactusQuantMatrix*) { return false; }
bool cactus_vulkan_encode_embedding_ortho_m(void*, const CactusQuantMatrix*, const uint32_t*, uint32_t) { return false; }
bool cactus_vulkan_encode_embedding_ortho(void*, uint32_t, const CactusQuantMatrix*, float) { return false; }
bool cactus_vulkan_encode_attention_fused_i8(void*, const void*, const void*, const void*, void*, void*, void*, void*, const void*, const void*, const void*, const void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float, float, size_t, size_t, size_t, size_t) { return false; }
bool cactus_vulkan_encode_deltanet_decode(void*, const void*, const void*, const void*, const void*, const void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float) { return false; }
bool cactus_vulkan_encode_deltanet_prefill(void*, const void*, const void*, const void*, const void*, const void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float) { return false; }
bool cactus_vulkan_moe_cq4_ready(const CactusQuantMatrix*) { return false; }
bool cactus_vulkan_moe_cq4_build(const CactusQuantMatrix*, const CactusQuantMatrix*, const CactusQuantMatrix*, uint32_t) { return false; }
bool cactus_vulkan_encode_moe_gated_cq4(void*, const void*, const void*, const void*, const CactusQuantMatrix*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float, float) { return false; }
bool cactus_vulkan_binary_f16(int, __fp16*, const __fp16*, const __fp16*, size_t) { return false; }
bool cactus_vulkan_scalar_f16(int, __fp16*, const __fp16*, size_t, float) { return false; }
bool cactus_vulkan_unary_f16(int, __fp16*, const __fp16*, size_t) { return false; }
bool cactus_vulkan_swiglu_f16(__fp16*, const __fp16*, const __fp16*, size_t, float) { return false; }
bool cactus_vulkan_rms_norm_f16(__fp16*, const __fp16*, const __fp16*, size_t, size_t, float) { return false; }
bool cactus_vulkan_cq_gemv(__fp16*, const __fp16*, const CactusQuantMatrix*, int, double*) { return false; }
bool cactus_vulkan_argmax_f16(const __fp16*, uint32_t, uint32_t*, float*) { return false; }

#else

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <map>

#include "vk_binary.h"
#include "vk_scalar.h"
#include "vk_unary.h"
#include "vk_swiglu.h"
#include "vk_rms_norm.h"
#include "vk_cq4_transform.h"
#include "vk_cq4_gemv.h"
#include "vk_argmax.h"
#include "vk_copy.h"
#include "vk_cast.h"
#include "vk_strided_copy.h"
#include "vk_strided_scatter.h"
#include "vk_bcast_binary.h"
#include "vk_concat2.h"
#include "vk_rms_norm_add.h"
#include "vk_rms_norm_add_rms.h"
#include "vk_rms_norm_scale.h"
#include "vk_rope_full.h"
#include "vk_softcap.h"
#include "vk_adjust_logits.h"
#include "vk_argmax3.h"
#include "vk_kv_append_i8.h"
#include "vk_attn_decode_i8.h"
#include "vk_attn_combine.h"
#include "vk_ortho_rotate.h"
#include "vk_rope_pair.h"
#include "vk_cq4_gemv2.h"
#include "vk_cq4_transform_m.h"
#include "vk_cq4_gemm.h"
#include "vk_rope_pair_rms.h"
#include "vk_rms2_add_clip.h"
#include "vk_gather.h"
#include "vk_softmax_topk.h"
#include "vk_topk_rows.h"
#include "vk_gemv_bias.h"
#include "vk_rms_norm_add_rows.h"
#include "vk_attn_prefill_i8.h"
#include "vk_attn_prefill_ring_i8.h"
#include "vk_attn_f16.h"
#include "vk_layer_norm.h"
#include "vk_softmax_rows.h"
#include "vk_glu.h"
#include "vk_conv1d_k3.h"
#include "vk_conv1d_dw.h"
#include "vk_conv1d_gen.h"
#include "vk_conv1d_nlc_dw.h"
#include "vk_batchnorm.h"
#include "vk_bias_add_rows.h"
#include "vk_gemm_batch.h"
#include "vk_rel_pos_bias.h"
#include "vk_elemwise_chain.h"
#include "vk_conv_cache_append.h"
#include "vk_binary_f32.h"
#include "vk_scalar_f32.h"
#include "vk_unary_f32.h"
#include "vk_clamp.h"
#include "vk_reduce_axis.h"
#include "vk_cumsum.h"
#include "vk_gather_f32idx.h"
#include "vk_maxpool1d.h"
#include "vk_bilinear.h"
#include "vk_groupnorm.h"
#include "vk_conv2d.h"
#include "vk_emb_hadamard_m.h"
#include "vk_emb_ortho_m.h"
#include "vk_attn_fused_i8.h"
#include "vk_attn_fused_i8_512.h"
#include "vk_cq_gemv_lowbit.h"
#include "vk_deltanet_decode.h"
#include "vk_deltanet_prefill.h"
#include "vk_moe_transform.h"
#include "vk_moe_up.h"
#include "vk_moe_down_acc.h"

#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

#define VK_FNS_GLOBAL(X) \
    X(vkCreateInstance) X(vkEnumerateInstanceLayerProperties)
#define VK_FNS_INSTANCE(X) \
    X(vkEnumeratePhysicalDevices) X(vkGetPhysicalDeviceProperties2) \
    X(vkGetPhysicalDeviceQueueFamilyProperties) X(vkGetPhysicalDeviceMemoryProperties) \
    X(vkGetPhysicalDeviceFeatures2) X(vkEnumerateDeviceExtensionProperties) \
    X(vkCreateDevice) X(vkGetDeviceProcAddr)
#define VK_FNS_DEVICE(X) \
    X(vkGetDeviceQueue) X(vkCreateShaderModule) X(vkDestroyShaderModule) \
    X(vkCreateDescriptorSetLayout) X(vkCreatePipelineLayout) X(vkCreateComputePipelines) \
    X(vkCreateDescriptorPool) X(vkResetDescriptorPool) X(vkAllocateDescriptorSets) \
    X(vkUpdateDescriptorSets) X(vkCreateCommandPool) X(vkResetCommandPool) \
    X(vkAllocateCommandBuffers) X(vkBeginCommandBuffer) X(vkEndCommandBuffer) \
    X(vkCmdBindPipeline) X(vkCmdBindDescriptorSets) X(vkCmdPushConstants) \
    X(vkCmdDispatch) X(vkCmdPipelineBarrier) X(vkQueueSubmit) \
    X(vkCreateFence) X(vkResetFences) X(vkWaitForFences) \
    X(vkCreateBuffer) X(vkDestroyBuffer) X(vkGetBufferMemoryRequirements) \
    X(vkBindBufferMemory) X(vkAllocateMemory) X(vkFreeMemory) X(vkMapMemory) \
    X(vkFlushMappedMemoryRanges) X(vkInvalidateMappedMemoryRanges) \
    X(vkCreateQueryPool) X(vkCmdResetQueryPool) X(vkCmdWriteTimestamp) X(vkGetQueryPoolResults)

enum { KB, KSC, KU, KSW, KR, KT, KG, KA,
       KCP, KCA, KSTC, KSTS, KBB, KCT, KRA, KRR,
       KRL, KRF, KSO, KAJ, KM3, KKV, KAT, KCB, KROT, KRP, KG2, KTM, KGM,
       KRPS, KR2C, KGH, KSTK, KTKR, KGB, KRAR, KAP, KAPR, KAF,
       KLN, KSMR, KGLU, KCK3, KCDW, KCGN, KCND, KBN2, KBAR, KGMB, KRPB, KEWC, KCCA,
       KB32, KSC32, KU32, KCLMP, KRAX, KCSM, KGF32, KMXP, KBIL, KGN2, KC2D, KEHM, KEOM, KAFU, KLBG, KDND, KDNP, KMOT, KMOU, KMOD, KAF2, KCOUNT };

struct KDef {
    const char* spv;
    size_t len;
    uint32_t nbuf;
    uint32_t push;
    uint32_t write_mask;
};

const KDef kdefs[KCOUNT] = {
    {kSpv_binary,        sizeof(kSpv_binary) - 1,        3, 8,  0x4},
    {kSpv_scalar,        sizeof(kSpv_scalar) - 1,        2, 12, 0x2},
    {kSpv_unary,         sizeof(kSpv_unary) - 1,         2, 8,  0x2},
    {kSpv_swiglu,        sizeof(kSpv_swiglu) - 1,        3, 8,  0x4},
    {kSpv_rms_norm,      sizeof(kSpv_rms_norm) - 1,      3, 8,  0x4},
    {kSpv_cq4_transform, sizeof(kSpv_cq4_transform) - 1, 6, 4,  0x20},
    {kSpv_cq4_gemv,      sizeof(kSpv_cq4_gemv) - 1,      5, 20, 0x10},
    {kSpv_argmax,        sizeof(kSpv_argmax) - 1,        2, 4,  0x2},
    {kSpv_copy,             sizeof(kSpv_copy) - 1,             2, 12,  0x2},
    {kSpv_cast,             sizeof(kSpv_cast) - 1,             2, 8,   0x2},
    {kSpv_strided_copy,     sizeof(kSpv_strided_copy) - 1,     2, 76,  0x2},
    {kSpv_strided_scatter,  sizeof(kSpv_strided_scatter) - 1,  2, 76,  0x2},
    {kSpv_bcast_binary,     sizeof(kSpv_bcast_binary) - 1,     3, 108, 0x4},
    {kSpv_concat2,          sizeof(kSpv_concat2) - 1,          3, 16,  0x4},
    {kSpv_rms_norm_add,     sizeof(kSpv_rms_norm_add) - 1,     4, 12,  0x8},
    {kSpv_rms_norm_add_rms, sizeof(kSpv_rms_norm_add_rms) - 1, 6, 12,  0x28},
    {kSpv_rms_norm_scale,   sizeof(kSpv_rms_norm_scale) - 1,   3, 12,  0x4},
    {kSpv_rope_full,        sizeof(kSpv_rope_full) - 1,        2, 28,  0x2},
    {kSpv_softcap,          sizeof(kSpv_softcap) - 1,          2, 8,   0x2},
    {kSpv_adjust_logits,    sizeof(kSpv_adjust_logits) - 1,    2, 20,  0x1},
    {kSpv_argmax3,          sizeof(kSpv_argmax3) - 1,          3, 8,   0x4},
    {kSpv_kv_append_i8,     sizeof(kSpv_kv_append_i8) - 1,     3, 36,  0x6},
    {kSpv_attn_decode_i8,   sizeof(kSpv_attn_decode_i8) - 1,   10, 52, 0x380},
    {kSpv_attn_combine,     sizeof(kSpv_attn_combine) - 1,     3, 8,   0x4},
    {kSpv_ortho_rotate,     sizeof(kSpv_ortho_rotate) - 1,     4, 8,   0x8},
    {kSpv_rope_pair,        sizeof(kSpv_rope_pair) - 1,        4, 8,   0x8},
    {kSpv_cq4_gemv2,        sizeof(kSpv_cq4_gemv2) - 1,        6, 20,  0x10},
    {kSpv_cq4_transform_m,  sizeof(kSpv_cq4_transform_m) - 1,  6, 8,   0x20},
    {kSpv_cq4_gemm,         sizeof(kSpv_cq4_gemm) - 1,         6, 24,  0x10},
    {kSpv_rope_pair_rms,     sizeof(kSpv_rope_pair_rms) - 1,     5, 8,  0x10},
    {kSpv_rms2_add_clip,     sizeof(kSpv_rms2_add_clip) - 1,     5, 12, 0x10},
    {kSpv_gather,            sizeof(kSpv_gather) - 1,            3, 8,  0x4},
    {kSpv_softmax_topk,      sizeof(kSpv_softmax_topk) - 1,      3, 16, 0x6},
    {kSpv_topk_rows,         sizeof(kSpv_topk_rows) - 1,         2, 12, 0x2},
    {kSpv_gemv_bias,         sizeof(kSpv_gemv_bias) - 1,         4, 12, 0x8},
    {kSpv_rms_norm_add_rows, sizeof(kSpv_rms_norm_add_rows) - 1, 5, 16, 0x18},
    {kSpv_attn_prefill_i8,      sizeof(kSpv_attn_prefill_i8) - 1,      8, 52, 0x80},
    {kSpv_attn_prefill_ring_i8, sizeof(kSpv_attn_prefill_ring_i8) - 1, 8, 60, 0x80},
    {kSpv_attn_f16,             sizeof(kSpv_attn_f16) - 1,             5, 48, 0x8},
    {kSpv_layer_norm,     sizeof(kSpv_layer_norm) - 1,     4, 12, 0x8},
    {kSpv_softmax_rows,   sizeof(kSpv_softmax_rows) - 1,   2, 4,  0x2},
    {kSpv_glu,            sizeof(kSpv_glu) - 1,            2, 12, 0x2},
    {kSpv_conv1d_k3,      sizeof(kSpv_conv1d_k3) - 1,      3, 16, 0x4},
    {kSpv_conv1d_dw,      sizeof(kSpv_conv1d_dw) - 1,      3, 16, 0x4},
    {kSpv_conv1d_gen,     sizeof(kSpv_conv1d_gen) - 1,     4, 36, 0x8},
    {kSpv_conv1d_nlc_dw,  sizeof(kSpv_conv1d_nlc_dw) - 1,  4, 28, 0x8},
    {kSpv_batchnorm,      sizeof(kSpv_batchnorm) - 1,      6, 16, 0x20},
    {kSpv_bias_add_rows,  sizeof(kSpv_bias_add_rows) - 1,  2, 8,  0x1},
    {kSpv_gemm_batch,     sizeof(kSpv_gemm_batch) - 1,     5, 24, 0x18},
    {kSpv_rel_pos_bias,   sizeof(kSpv_rel_pos_bias) - 1,   3, 24, 0x4},
    {kSpv_elemwise_chain, sizeof(kSpv_elemwise_chain) - 1, 11, 16, 0xC},
    {kSpv_conv_cache_append, sizeof(kSpv_conv_cache_append) - 1, 4, 32, 0xC},
    {kSpv_binary_f32,   sizeof(kSpv_binary_f32) - 1,   3, 8,  0x4},
    {kSpv_scalar_f32,   sizeof(kSpv_scalar_f32) - 1,   2, 12, 0x2},
    {kSpv_unary_f32,    sizeof(kSpv_unary_f32) - 1,    2, 8,  0x2},
    {kSpv_clamp,        sizeof(kSpv_clamp) - 1,        4, 16, 0xC},
    {kSpv_reduce_axis,  sizeof(kSpv_reduce_axis) - 1,  4, 20, 0xC},
    {kSpv_cumsum,       sizeof(kSpv_cumsum) - 1,       4, 16, 0xC},
    {kSpv_gather_f32idx, sizeof(kSpv_gather_f32idx) - 1, 3, 8, 0x4},
    {kSpv_maxpool1d,    sizeof(kSpv_maxpool1d) - 1,    2, 16, 0x2},
    {kSpv_bilinear,     sizeof(kSpv_bilinear) - 1,     2, 24, 0x2},
    {kSpv_groupnorm,    sizeof(kSpv_groupnorm) - 1,    4, 20, 0x8},
    {kSpv_conv2d,       sizeof(kSpv_conv2d) - 1,       4, 44, 0x8},
    {kSpv_emb_hadamard_m, sizeof(kSpv_emb_hadamard_m) - 1, 9, 20, 0x100},
    {kSpv_emb_ortho_m,    sizeof(kSpv_emb_ortho_m) - 1,    7, 24, 0x40},
    {kSpv_attn_fused_i8,  sizeof(kSpv_attn_fused_i8) - 1, 15, 56, 0x60F8},
    {kSpv_cq_gemv_lowbit, sizeof(kSpv_cq_gemv_lowbit) - 1, 5, 24, 0x10},
    {kSpv_deltanet_decode,  sizeof(kSpv_deltanet_decode) - 1,  7, 20, 0x40},
    {kSpv_deltanet_prefill, sizeof(kSpv_deltanet_prefill) - 1, 8, 24, 0xC0},
    {kSpv_moe_transform, sizeof(kSpv_moe_transform) - 1, 5, 40, 0x10},
    {kSpv_moe_up,        sizeof(kSpv_moe_up) - 1,        8, 48, 0x80},
    {kSpv_moe_down_acc,  sizeof(kSpv_moe_down_acc) - 1,  6, 44, 0x20},
    {kSpv_attn_fused_i8_512, sizeof(kSpv_attn_fused_i8_512) - 1, 15, 56, 0x60F8},
};

struct Slot {
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkDescriptorPool desc = VK_NULL_HANDLE;
    bool recording = false;
    bool in_flight = false;
    bool dirty = true;
};

struct VK {
    void* lib = nullptr;
    PFN_vkGetInstanceProcAddr gipa = nullptr;
#define D(f) PFN_##f f = nullptr;
    VK_FNS_GLOBAL(D)
    VK_FNS_INSTANCE(D)
    VK_FNS_DEVICE(D)
#undef D

    VkInstance inst = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice dev = VK_NULL_HANDLE;
    VkQueue q = VK_NULL_HANDLE;
    uint32_t qfam = 0;
    VkPhysicalDeviceMemoryProperties memprops = {};
    VkDeviceSize sb_align = 256;
    Slot slots[3];
    int cur = 0;
    VkQueryPool query = VK_NULL_HANDLE;
    float ts_period = 0.0f;
    uint32_t sg_size = 0;
    uint32_t shmem = 0;
    bool attn_ok = false;
    bool gemv2_ok = false;
    struct Pipe { VkDescriptorSetLayout dsl = VK_NULL_HANDLE; VkPipelineLayout pl = VK_NULL_HANDLE; VkPipeline p = VK_NULL_HANDLE; } pipes[KCOUNT];
    std::string info = "Vulkan: not initialized";
    bool ok = false;

    VK() { init(); }

    void init() {
        lib = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
        if (!lib) { info = "Vulkan: libvulkan.so not found"; return; }
        gipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(lib, "vkGetInstanceProcAddr"));
        if (!gipa) { info = "Vulkan: vkGetInstanceProcAddr missing"; return; }
#define R(f) f = reinterpret_cast<PFN_##f>(gipa(nullptr, #f)); if (!f) { info = "Vulkan: missing " #f; return; }
        VK_FNS_GLOBAL(R)
#undef R

        const char* layers[1] = {"VK_LAYER_KHRONOS_validation"};
        uint32_t nlayers = 0;
        if (getenv("CACTUS_VK_VALIDATE")) {
            uint32_t avail = 0;
            vkEnumerateInstanceLayerProperties(&avail, nullptr);
            std::vector<VkLayerProperties> props(avail);
            vkEnumerateInstanceLayerProperties(&avail, props.data());
            for (const auto& lp : props)
                if (std::strcmp(lp.layerName, layers[0]) == 0) nlayers = 1;
        }

        VkApplicationInfo app = {};
        app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.pApplicationName = "cactus";
        app.apiVersion = VK_API_VERSION_1_1;
        VkInstanceCreateInfo ici = {};
        ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ici.pApplicationInfo = &app;
        ici.enabledLayerCount = nlayers;
        ici.ppEnabledLayerNames = layers;
        if (vkCreateInstance(&ici, nullptr, &inst) != VK_SUCCESS || !inst) {
            info = "Vulkan: instance creation failed";
            return;
        }
#define R(f) f = reinterpret_cast<PFN_##f>(gipa(inst, #f)); if (!f) { info = "Vulkan: missing " #f; return; }
        VK_FNS_INSTANCE(R)
#undef R

        uint32_t ndev = 0;
        vkEnumeratePhysicalDevices(inst, &ndev, nullptr);
        if (ndev == 0) { info = "Vulkan: no physical devices"; return; }
        std::vector<VkPhysicalDevice> devs(ndev);
        vkEnumeratePhysicalDevices(inst, &ndev, devs.data());

        VkPhysicalDeviceSubgroupProperties subp = {};
        subp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
        VkPhysicalDeviceProperties2 props2 = {};
        props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props2.pNext = &subp;
        VkPhysicalDevice16BitStorageFeatures s16 = {};
        for (VkPhysicalDevice d : devs) {
            VkPhysicalDeviceProperties2 p2 = props2;
            vkGetPhysicalDeviceProperties2(d, &p2);
            if (p2.properties.apiVersion < VK_API_VERSION_1_1) continue;
            uint32_t nq = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(d, &nq, nullptr);
            std::vector<VkQueueFamilyProperties> qs(nq);
            vkGetPhysicalDeviceQueueFamilyProperties(d, &nq, qs.data());
            uint32_t fam = UINT32_MAX;
            for (uint32_t i = 0; i < nq; ++i)
                if (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { fam = i; break; }
            if (fam == UINT32_MAX) continue;
            s16 = {};
            s16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
            VkPhysicalDeviceFeatures2 f2 = {};
            f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            f2.pNext = &s16;
            vkGetPhysicalDeviceFeatures2(d, &f2);
            if (!s16.storageBuffer16BitAccess) continue;
            phys = d;
            qfam = fam;
            props2 = p2;
            break;
        }
        if (!phys) { info = "Vulkan: no compute device with 16-bit storage"; return; }

        vkGetPhysicalDeviceMemoryProperties(phys, &memprops);
        sb_align = props2.properties.limits.minStorageBufferOffsetAlignment;
        if (sb_align < 16) sb_align = 16;
        if (props2.properties.limits.maxPushConstantsSize < 128) {
            info = "Vulkan: push constant limit below 128";
            return;
        }
        sg_size = subp.subgroupSize;
        shmem = props2.properties.limits.maxComputeSharedMemorySize;
        attn_ok = (subp.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT)
               && (subp.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT)
               && sg_size >= 16
               && props2.properties.limits.maxComputeSharedMemorySize >= 4112u * 4u;
        gemv2_ok = attn_ok
               && props2.properties.limits.maxComputeSharedMemorySize >= 1120u * 16u + 256u
               && !getenv("CACTUS_VK_GEMV");
        if (props2.properties.limits.timestampComputeAndGraphics)
            ts_period = props2.properties.limits.timestampPeriod;

        bool f16ext = props2.properties.apiVersion >= VK_API_VERSION_1_2;
        if (!f16ext) {
            uint32_t next = 0;
            vkEnumerateDeviceExtensionProperties(phys, nullptr, &next, nullptr);
            std::vector<VkExtensionProperties> exts(next);
            vkEnumerateDeviceExtensionProperties(phys, nullptr, &next, exts.data());
            for (const auto& e : exts)
                if (std::strcmp(e.extensionName, "VK_KHR_shader_float16_int8") == 0) f16ext = true;
        }
        bool shader_f16 = false;
        if (f16ext) {
            VkPhysicalDeviceShaderFloat16Int8Features hf = {};
            hf.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
            VkPhysicalDeviceFeatures2 f2 = {};
            f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            f2.pNext = &hf;
            vkGetPhysicalDeviceFeatures2(phys, &f2);
            shader_f16 = hf.shaderFloat16;
        }

        char buf[256];
        std::snprintf(buf, sizeof(buf), "%s | Vulkan %u.%u | driver %u | 16bitStorage:yes shaderFloat16:%s subgroup:%u sharedMem:%u",
                      props2.properties.deviceName,
                      VK_API_VERSION_MAJOR(props2.properties.apiVersion),
                      VK_API_VERSION_MINOR(props2.properties.apiVersion),
                      props2.properties.driverVersion,
                      shader_f16 ? "yes" : "no",
                      subp.subgroupSize,
                      props2.properties.limits.maxComputeSharedMemorySize);
        info = buf;

        float prio = 1.0f;
        VkDeviceQueueCreateInfo qci = {};
        qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = qfam;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;
        VkPhysicalDevice16BitStorageFeatures en16 = {};
        en16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
        en16.storageBuffer16BitAccess = VK_TRUE;
        VkPhysicalDeviceFeatures2 enf2 = {};
        enf2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        enf2.pNext = &en16;
        VkDeviceCreateInfo dci = {};
        dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dci.pNext = &enf2;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        if (vkCreateDevice(phys, &dci, nullptr, &dev) != VK_SUCCESS || !dev) {
            info += " | device creation failed";
            return;
        }
#define R(f) f = reinterpret_cast<PFN_##f>(vkGetDeviceProcAddr(dev, #f)); if (!f) { info += " | missing " #f; return; }
        VK_FNS_DEVICE(R)
#undef R
        vkGetDeviceQueue(dev, qfam, 0, &q);

        for (Slot& s : slots) {
            VkCommandPoolCreateInfo cpci = {};
            cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            cpci.queueFamilyIndex = qfam;
            if (vkCreateCommandPool(dev, &cpci, nullptr, &s.pool) != VK_SUCCESS) { info += " | command pool failed"; return; }
            VkCommandBufferAllocateInfo cbai = {};
            cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            cbai.commandPool = s.pool;
            cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cbai.commandBufferCount = 1;
            if (vkAllocateCommandBuffers(dev, &cbai, &s.cmd) != VK_SUCCESS) { info += " | command buffer failed"; return; }
            VkFenceCreateInfo fci = {};
            fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            if (vkCreateFence(dev, &fci, nullptr, &s.fence) != VK_SUCCESS) { info += " | fence failed"; return; }
            VkDescriptorPoolSize psz = {};
            psz.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            psz.descriptorCount = 8192;
            VkDescriptorPoolCreateInfo dpci = {};
            dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            dpci.maxSets = 2048;
            dpci.poolSizeCount = 1;
            dpci.pPoolSizes = &psz;
            if (vkCreateDescriptorPool(dev, &dpci, nullptr, &s.desc) != VK_SUCCESS) { info += " | descriptor pool failed"; return; }
        }

        if (ts_period > 0.0f) {
            VkQueryPoolCreateInfo qpci = {};
            qpci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
            qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
            qpci.queryCount = 2;
            if (vkCreateQueryPool(dev, &qpci, nullptr, &query) != VK_SUCCESS) { query = VK_NULL_HANDLE; ts_period = 0.0f; }
        }

        for (int k = 0; k < KCOUNT; ++k) {
            if (k == KAT && !attn_ok) continue;
            if ((k == KG2 || k == KGM) && !gemv2_ok) continue;
            if (k == KAPR && shmem < (7936u + 128u) * 4u) continue;
            if (k == KAFU && (!attn_ok || shmem < 22000u)) continue;
            if (k == KAF2 && (!attn_ok || shmem < 23200u || sg_size < 16u)) continue;
            const KDef& kd = kdefs[k];
            if (kd.len % 4 != 0) { info += " | spv size misaligned"; return; }
            std::vector<VkDescriptorSetLayoutBinding> binds(kd.nbuf);
            for (uint32_t i = 0; i < kd.nbuf; ++i) {
                binds[i] = {};
                binds[i].binding = i;
                binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                binds[i].descriptorCount = 1;
                binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            }
            VkDescriptorSetLayoutCreateInfo dslci = {};
            dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            dslci.bindingCount = kd.nbuf;
            dslci.pBindings = binds.data();
            if (vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &pipes[k].dsl) != VK_SUCCESS) { info += " | dsl failed"; return; }
            VkPushConstantRange range = {};
            range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            range.size = kd.push;
            VkPipelineLayoutCreateInfo plci = {};
            plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            plci.setLayoutCount = 1;
            plci.pSetLayouts = &pipes[k].dsl;
            plci.pushConstantRangeCount = 1;
            plci.pPushConstantRanges = &range;
            if (vkCreatePipelineLayout(dev, &plci, nullptr, &pipes[k].pl) != VK_SUCCESS) { info += " | pipeline layout failed"; return; }
            std::vector<uint32_t> code(kd.len / 4);
            std::memcpy(code.data(), kd.spv, kd.len);
            VkShaderModuleCreateInfo smci = {};
            smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            smci.codeSize = kd.len;
            smci.pCode = code.data();
            VkShaderModule mod = VK_NULL_HANDLE;
            if (vkCreateShaderModule(dev, &smci, nullptr, &mod) != VK_SUCCESS) { info += " | shader module failed"; return; }
            VkComputePipelineCreateInfo cpi = {};
            cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            cpi.stage.module = mod;
            cpi.stage.pName = "main";
            cpi.layout = pipes[k].pl;
            VkResult pr = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipes[k].p);
            vkDestroyShaderModule(dev, mod, nullptr);
            if (pr != VK_SUCCESS) { info += " | pipeline creation failed"; return; }
        }
        ok = true;
        info += attn_ok ? " | attn:ok" : " | attn:off";
        info += " | build:ok";
    }
};

VK& vk() { static VK v; return v; }
std::recursive_mutex g_mu;

constexpr size_t BUCKET = 16384;
constexpr size_t SLAB_BYTES = 32u << 20;
constexpr size_t POOL_MAX = 4u << 20;

struct MemBlock {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    char* map = nullptr;
    size_t cap = 0;
    bool slab = false;
    bool coherent = true;
    size_t used = 0;
    uint32_t live = 0;
};

bool g_noncoherent = false;
std::map<uintptr_t, MemBlock*> g_blocks;
std::unordered_map<size_t, std::vector<MemBlock*>> g_free_blocks;
std::vector<MemBlock*> g_pending;
std::vector<MemBlock*> g_slabs;
MemBlock* g_cur_slab = nullptr;

uint32_t mem_type(uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < vk().memprops.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (vk().memprops.memoryTypes[i].propertyFlags & want) == want) return i;
    return UINT32_MAX;
}

bool create_block(size_t bytes, MemBlock& b, bool cached) {
    VkBufferCreateInfo bci = {};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = bytes ? bytes : 4;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vk().vkCreateBuffer(vk().dev, &bci, nullptr, &b.buf) != VK_SUCCESS) return false;
    VkMemoryRequirements req = {};
    vk().vkGetBufferMemoryRequirements(vk().dev, b.buf, &req);
    uint32_t type = UINT32_MAX;
    if (cached) {
        type = mem_type(req.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
        if (type == UINT32_MAX)
            type = mem_type(req.memoryTypeBits,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
        static const bool force_cached = [] {
            const char* v = getenv("CACTUS_VK_CACHED");
            return v && std::strcmp(v, "force") == 0;
        }();
        if (type == UINT32_MAX && force_cached) {
            type = mem_type(req.memoryTypeBits,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
            if (type == UINT32_MAX)
                type = mem_type(req.memoryTypeBits,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
        }
    }
    if (type == UINT32_MAX)
        type = mem_type(req.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (type == UINT32_MAX)
        type = mem_type(req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (type == UINT32_MAX) { vk().vkDestroyBuffer(vk().dev, b.buf, nullptr); b.buf = VK_NULL_HANDLE; return false; }
    VkMemoryPropertyFlags mprops = vk().memprops.memoryTypes[type].propertyFlags;
    b.coherent = (mprops & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
    if (!b.coherent) g_noncoherent = true;
    if (cached) {
        static bool logged = false;
        if (!logged) {
            logged = true;
            char m[48];
            std::snprintf(m, sizeof(m), " | actMem:%s%s%s",
                          (mprops & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ? "dl+" : "",
                          (mprops & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) ? "cached" : "wc",
                          b.coherent ? "+coh" : "");
            vk().info += m;
        }
    }
    VkMemoryAllocateInfo mai = {};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = type;
    void* map = nullptr;
    if (vk().vkAllocateMemory(vk().dev, &mai, nullptr, &b.mem) != VK_SUCCESS
        || vk().vkBindBufferMemory(vk().dev, b.buf, b.mem, 0) != VK_SUCCESS
        || vk().vkMapMemory(vk().dev, b.mem, 0, VK_WHOLE_SIZE, 0, &map) != VK_SUCCESS) {
        if (b.mem) vk().vkFreeMemory(vk().dev, b.mem, nullptr);
        vk().vkDestroyBuffer(vk().dev, b.buf, nullptr);
        b = MemBlock();
        return false;
    }
    b.map = static_cast<char*>(map);
    b.cap = bytes;
    return true;
}

void* alloc_shared_i(size_t bytes) {
    size_t bk = (bytes ? bytes : 1);
    bk = (bk + BUCKET - 1) & ~(BUCKET - 1);
    auto& fl = g_free_blocks[bk];
    MemBlock* b;
    if (!fl.empty()) {
        b = fl.back();
        fl.pop_back();
    } else {
        b = new MemBlock();
        if (!create_block(bk, *b, true)) { delete b; return nullptr; }
        g_blocks[(uintptr_t)b->map] = b;
    }
    b->live = 1;
    return b->map;
}

void* alloc_pooled_i(size_t bytes) {
    if (bytes > POOL_MAX) return alloc_shared_i(bytes);
    size_t need = (bytes + 255) & ~(size_t)255;
    if (!g_cur_slab || g_cur_slab->used + need > g_cur_slab->cap) {
        for (MemBlock* s : g_slabs)
            if (s != g_cur_slab && s->live == 0 && s->cap >= need) { s->used = 0; g_cur_slab = s; break; }
        if (!g_cur_slab || g_cur_slab->used + need > g_cur_slab->cap) {
            MemBlock* s = new MemBlock();
            if (!create_block(SLAB_BYTES, *s, true)) { delete s; return alloc_shared_i(bytes); }
            s->slab = true;
            g_blocks[(uintptr_t)s->map] = s;
            g_slabs.push_back(s);
            g_cur_slab = s;
        }
    }
    void* p = g_cur_slab->map + g_cur_slab->used;
    g_cur_slab->used += need;
    g_cur_slab->live++;
    return p;
}

MemBlock* block_of(const void* p, size_t* off) {
    uintptr_t a = (uintptr_t)p;
    auto it = g_blocks.upper_bound(a);
    if (it == g_blocks.begin()) return nullptr;
    --it;
    MemBlock* b = it->second;
    if (a >= it->first + b->cap) return nullptr;
    *off = a - it->first;
    return b;
}

void free_shared_i(void* p) {
    if (!p) return;
    size_t off = 0;
    MemBlock* b = block_of(p, &off);
    if (!b) return;
    if (b->slab) {
        if (b->live) b->live--;
        return;
    }
    b->live = 0;
    g_pending.push_back(b);
}

void flush_noncoherent(bool invalidate) {
    if (!g_noncoherent) return;
    std::vector<VkMappedMemoryRange> rs;
    for (auto& kv : g_blocks) {
        if (kv.second->coherent) continue;
        VkMappedMemoryRange r = {};
        r.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        r.memory = kv.second->mem;
        r.size = VK_WHOLE_SIZE;
        rs.push_back(r);
    }
    if (rs.empty()) return;
    if (invalidate) vk().vkInvalidateMappedMemoryRanges(vk().dev, (uint32_t)rs.size(), rs.data());
    else vk().vkFlushMappedMemoryRanges(vk().dev, (uint32_t)rs.size(), rs.data());
}

bool submit_slot(Slot& s) {
    flush_noncoherent(false);
    VkResult r = vk().vkEndCommandBuffer(s.cmd);
    if (r != VK_SUCCESS) { vk().ok = false; std::fprintf(stderr, "[vkdead] vkEndCommandBuffer r=%d\n", (int)r); return false; }
    r = vk().vkResetFences(vk().dev, 1, &s.fence);
    if (r != VK_SUCCESS) { vk().ok = false; std::fprintf(stderr, "[vkdead] vkResetFences r=%d\n", (int)r); return false; }
    VkSubmitInfo si = {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &s.cmd;
    r = vk().vkQueueSubmit(vk().q, 1, &si, s.fence);
    if (r != VK_SUCCESS) { vk().ok = false; std::fprintf(stderr, "[vkdead] vkQueueSubmit r=%d\n", (int)r); return false; }
    s.recording = false;
    s.in_flight = true;
    s.dirty = true;
    return true;
}

bool wait_slot(Slot& s) {
    if (!s.in_flight) return true;
    // Metal parity: waitUntilCompleted blocks for as long as the work takes.
    // Long submissions (e.g. warmup prefill on slow GPUs) legitimately exceed
    // 5s, so VK_TIMEOUT retries; only a real error kills the device.
    VkResult r = VK_TIMEOUT;
    for (int tries = 0; r == VK_TIMEOUT && tries < 60; ++tries) {
        r = vk().vkWaitForFences(vk().dev, 1, &s.fence, VK_TRUE, 5000000000ull);
        if (r == VK_TIMEOUT && std::getenv("CACTUS_VK_STATS"))
            std::fprintf(stderr, "[vkslow] fence wait %ds...\n", (tries + 1) * 5);
    }
    s.in_flight = false;
    if (r != VK_SUCCESS) { vk().ok = false; std::fprintf(stderr, "[vkdead] vkWaitForFences r=%d\n", (int)r); return false; }
    return true;
}

bool ensure_cmd() {
    Slot& s = vk().slots[vk().cur];
    if (s.recording) return true;
    if (!wait_slot(s)) return false;
    if (s.dirty) {
        if (vk().vkResetCommandPool(vk().dev, s.pool, 0) != VK_SUCCESS) return false;
        if (vk().vkResetDescriptorPool(vk().dev, s.desc, 0) != VK_SUCCESS) return false;
        s.dirty = false;
    }
    VkCommandBufferBeginInfo bi = {};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vk().vkBeginCommandBuffer(s.cmd, &bi) != VK_SUCCESS) return false;
    s.recording = true;
    return true;
}

void flush_i() {
    Slot& s = vk().slots[vk().cur];
    if (!s.recording) return;
    if (!submit_slot(s)) return;
    vk().cur = (vk().cur + 1) % 3;
}

void recycle_pending() {
    for (MemBlock* b : g_pending) g_free_blocks[b->cap].push_back(b);
    g_pending.clear();
    for (MemBlock* s : g_slabs)
        if (s->live == 0) s->used = 0;
}

// Host-buffer overrides for the fused-embed fold path. Metal wraps arbitrary
// host memory zero-copy, so fold kernels write straight into the engine's
// host input vectors; Vulkan cannot. Instead, binds of those host pointers
// are redirected to a device shadow: created inside a fold scope, refreshed
// from host when no GPU write is pending, written back to host at sync so
// CPU consumers observe the kernel results.
struct OvEntry { size_t size = 0; MemBlock* blk = nullptr; bool dirty = false; };
std::map<const void*, OvEntry> g_ov;
MemBlock g_stage;
size_t g_stage_off = 0;

void sync_i() {
    flush_i();
    for (Slot& s : vk().slots) wait_slot(s);
    flush_noncoherent(true);
    for (auto& kv : g_ov) {
        if (!kv.second.dirty) continue;
        std::memcpy(const_cast<void*>(kv.first), kv.second.blk->map, kv.second.size);
        kv.second.dirty = false;
    }
    g_stage_off = 0;
    recycle_pending();
}

VkCommandBuffer cur_cmd() { return vk().slots[vk().cur].cmd; }

VkDescriptorSet make_set(int k, const VkDescriptorBufferInfo* infos, uint32_t count) {
    VkDescriptorSetAllocateInfo ai = {};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = vk().slots[vk().cur].desc;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &vk().pipes[k].dsl;
    VkDescriptorSet set = VK_NULL_HANDLE;
    if (vk().vkAllocateDescriptorSets(vk().dev, &ai, &set) != VK_SUCCESS) return VK_NULL_HANDLE;
    VkWriteDescriptorSet writes[16] = {};
    for (uint32_t i = 0; i < count; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vk().vkUpdateDescriptorSets(vk().dev, count, writes, 0, nullptr);
    return set;
}

void mem_barrier() {
    VkMemoryBarrier mb = {};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vk().vkCmdPipelineBarrier(cur_cmd(), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
}

struct AccessRange {
    VkBuffer buf;
    VkDeviceSize beg;
    VkDeviceSize end;
};
std::vector<AccessRange> g_writes, g_reads;

bool ranges_hit(const std::vector<AccessRange>& v, VkBuffer buf, VkDeviceSize beg, VkDeviceSize end) {
    for (const AccessRange& r : v)
        if (r.buf == buf && beg < r.end && r.beg < end) return true;
    return false;
}

void dispatch(int k, VkDescriptorSet set, const void* push, uint32_t gx, uint32_t gy,
              const VkDescriptorBufferInfo* infos, uint32_t count, uint32_t gz = 1) {
    static const bool barrier_all = [] {
        const char* v = getenv("CACTUS_VK_BARRIER");
        return v && std::strcmp(v, "all") == 0;
    }();
    const uint32_t wm = kdefs[k].write_mask;
    bool hazard = barrier_all || g_writes.size() + g_reads.size() > 1024;
    for (uint32_t i = 0; i < count && !hazard; ++i) {
        VkDeviceSize beg = infos[i].offset, end = beg + infos[i].range;
        hazard = ranges_hit(g_writes, infos[i].buffer, beg, end)
              || (((wm >> i) & 1u) && ranges_hit(g_reads, infos[i].buffer, beg, end));
    }
    if (hazard) {
        mem_barrier();
        g_writes.clear();
        g_reads.clear();
    }
    for (uint32_t i = 0; i < count; ++i) {
        VkDeviceSize beg = infos[i].offset, end = beg + infos[i].range;
        (((wm >> i) & 1u) ? g_writes : g_reads).push_back({infos[i].buffer, beg, end});
    }
    vk().vkCmdBindPipeline(cur_cmd(), VK_PIPELINE_BIND_POINT_COMPUTE, vk().pipes[k].p);
    vk().vkCmdBindDescriptorSets(cur_cmd(), VK_PIPELINE_BIND_POINT_COMPUTE, vk().pipes[k].pl, 0, 1, &set, 0, nullptr);
    vk().vkCmdPushConstants(cur_cmd(), vk().pipes[k].pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, kdefs[k].push, push);
    vk().vkCmdDispatch(cur_cmd(), gx, gy, gz);
}

OvEntry* ov_bind(const void* p, size_t bytes, size_t* delta_out) {
    if (!p || bytes == 0 || g_ov.empty()) return nullptr;
    auto it = g_ov.upper_bound(p);
    if (it == g_ov.begin()) return nullptr;
    --it;
    const char* base = (const char*)it->first;
    if ((const char*)p < base || (const char*)p >= base + it->second.size) return nullptr;
    size_t delta = (size_t)((const char*)p - base);
    if (delta + bytes > it->second.size) return nullptr;
    OvEntry* e = &it->second;
    if (!e->dirty) std::memcpy((char*)e->blk->map + delta, p, bytes);
    e->dirty = true;
    if (delta_out) *delta_out = delta;
    return e;
}

void ov_register(void* p, size_t bytes) {
    if (!p || bytes == 0 || bytes > (1u << 22)) return;
    auto it = g_ov.find(p);
    if (it != g_ov.end() && it->second.size >= bytes) return;
    if (it != g_ov.end()) {
        if (it->second.dirty) sync_i();
        if (it->second.blk) g_pending.push_back(it->second.blk);
        g_ov.erase(it);
    }
    MemBlock* b = new MemBlock();
    if (!create_block(bytes, *b, false)) { delete b; return; }
    std::memcpy(b->map, p, bytes);
    g_ov.emplace(p, OvEntry{bytes, b, false});
}

bool dinfo(const void* p, size_t bytes, VkDescriptorBufferInfo* out) {
    size_t off = 0;
    MemBlock* b = block_of(p, &off);
    if (!b) {
        size_t delta = 0;
        OvEntry* e = ov_bind(p, bytes ? bytes : 4, &delta);
        if (!e || delta % vk().sb_align != 0) return false;
        out->buffer = e->blk->buf;
        out->offset = delta;
        out->range = bytes ? bytes : 4;
        return true;
    }
    if (off % vk().sb_align != 0) return false;
    out->buffer = b->buf;
    out->offset = off;
    out->range = bytes ? bytes : 4;
    return true;
}

bool dinfo_off(const void* p, size_t bytes, VkDescriptorBufferInfo* out, uint32_t* delta4) {
    size_t off = 0;
    MemBlock* b = block_of(p, &off);
    if (!b) {
        size_t delta = 0;
        OvEntry* e = ov_bind(p, bytes ? bytes : 4, &delta);
        if (!e) return false;
        size_t obind = delta & ~(size_t)(vk().sb_align - 1);
        size_t d = delta - obind;
        if (d & 3u) return false;
        out->buffer = e->blk->buf;
        out->offset = obind;
        out->range = (bytes ? bytes : 4) + d;
        *delta4 = (uint32_t)(d >> 2);
        return true;
    }
    size_t bind = off & ~(size_t)(vk().sb_align - 1);
    size_t d = off - bind;
    if (d & 3u) return false;
    out->buffer = b->buf;
    out->offset = bind;
    out->range = (bytes ? bytes : 4) + d;
    *delta4 = (uint32_t)(d >> 2);
    return true;
}

MemBlock g_dummy, g_attn_po, g_attn_ml, g_recent;
std::unordered_map<const void*, MemBlock*> g_wraps;

bool winfo(const void* p, size_t bytes, VkDescriptorBufferInfo* out) {
    if (dinfo(p, bytes, out)) return true;
    if (!p || bytes == 0 || bytes > (1u << 20)) return false;
    auto it = g_wraps.find(p);
    if (it != g_wraps.end() && it->second->cap >= bytes) {
        if (std::memcmp(it->second->map, p, bytes) != 0) {
            sync_i();
            std::memcpy(it->second->map, p, bytes);
        }
        out->buffer = it->second->buf;
        out->offset = 0;
        out->range = bytes;
        return true;
    }
    MemBlock* b = new MemBlock();
    if (!create_block(bytes, *b, false)) { delete b; return false; }
    std::memcpy(b->map, p, bytes);
    if (it != g_wraps.end()) {
        sync_i();
        vk().vkDestroyBuffer(vk().dev, it->second->buf, nullptr);
        vk().vkFreeMemory(vk().dev, it->second->mem, nullptr);
        delete it->second;
        it->second = b;
    } else {
        g_wraps.emplace(p, b);
    }
    out->buffer = b->buf;
    out->offset = 0;
    out->range = bytes;
    return true;
}

bool dummy_info(VkDescriptorBufferInfo* out) {
    if (!g_dummy.buf && !create_block(BUCKET, g_dummy, true)) return false;
    out->buffer = g_dummy.buf;
    out->offset = 0;
    out->range = BUCKET;
    return true;
}

// Per-encode upload staging. scratch_info() reuses one block, so consecutive
// encodes that memcpy fresh host content before submission would overwrite
// each other (deferred-execution race). The ring bump-allocates a distinct
// 256-aligned slot per upload; sync_i resets the cursor once the GPU has
// consumed everything.
void* stage_alloc(size_t bytes) {
    if (bytes == 0) return nullptr;
    size_t need = (bytes + 255u) & ~(size_t)255u;
    if (g_stage_off + need > g_stage.cap) {
        // Grow from the previous capacity, never back down: a wrap that
        // recreated the ring at BUCKET would fill instantly and force a
        // pipeline-drain sync every few encodes.
        size_t cap = g_stage.cap ? g_stage.cap * 2 : BUCKET;
        while (cap < need) cap <<= 1;
        if (g_stage.buf) {
            sync_i();
            g_blocks.erase((uintptr_t)g_stage.map);
            vk().vkDestroyBuffer(vk().dev, g_stage.buf, nullptr);
            vk().vkFreeMemory(vk().dev, g_stage.mem, nullptr);
            g_stage = MemBlock();
            g_stage_off = 0;
        }
        if (!create_block(cap, g_stage, true)) return nullptr;
        g_stage.live = 1;
        g_blocks[(uintptr_t)g_stage.map] = &g_stage;
    }
    char* dst = (char*)g_stage.map + g_stage_off;
    g_stage_off += need;
    return dst;
}

const void* stage_upload(const void* src, size_t bytes) {
    if (!src) return nullptr;
    void* dst = stage_alloc(bytes);
    if (dst) std::memcpy(dst, src, bytes);
    return dst;
}

bool dinfo(const void* p, size_t bytes, VkDescriptorBufferInfo* out);

// Read-only operand bind: device memory directly, host memory via a staged
// per-encode snapshot (stable for the duration of the graph execute — the
// engine only rewrites inputs between executes). Metal reads host memory
// zero-copy; this is the Vulkan equivalent for small operands. Never use for
// outputs — writes would land in the snapshot.
bool sinfo(const void* p, size_t bytes, VkDescriptorBufferInfo* out) {
    if (dinfo(p, bytes, out)) return true;
    if (!p || bytes == 0 || bytes > (4u << 20)) return false;
    const void* st = stage_upload(p, bytes);
    return st && dinfo(st, bytes, out);
}

bool scratch_info(MemBlock& b, size_t bytes, VkDescriptorBufferInfo* out) {
    if (b.cap < bytes) {
        if (b.buf) {
            sync_i();
            g_blocks.erase((uintptr_t)b.map);
            vk().vkDestroyBuffer(vk().dev, b.buf, nullptr);
            vk().vkFreeMemory(vk().dev, b.mem, nullptr);
            b = MemBlock();
        }
        size_t cap = BUCKET;
        while (cap < bytes) cap <<= 1;
        if (!create_block(cap, b, true)) return false;
        b.live = 1;
        g_blocks[(uintptr_t)b.map] = &b;
    }
    out->buffer = b.buf;
    out->offset = 0;
    out->range = bytes;
    return true;
}

uint32_t ew_groups(size_t n) {
    size_t g = (n + 63) / 64;
    return (uint32_t)(g > 65535 ? 65535 : g);
}

bool enc_ew(int k, void* y, size_t out_bytes,
            const void* i0, size_t b0, const void* i1, size_t b1,
            const void* push, uint32_t groups) {
    if (!vk().ok) return false;
    VkDescriptorBufferInfo infos[3];
    uint32_t idx = 0;
    if (!sinfo(i0, b0, &infos[idx++])) return false;
    if (i1 && !sinfo(i1, b1, &infos[idx++])) return false;
    if (!dinfo(y, out_bytes, &infos[idx++])) return false;
    if (!ensure_cmd()) return false;
    VkDescriptorSet set = make_set(k, infos, idx);
    if (!set) return false;
    dispatch(k, set, push, groups, 1, infos, idx);
    return true;
}

struct ResW {
    MemBlock b;
    VkDeviceSize off[9] = {};
    VkDeviceSize sz[9] = {};
    bool ok = false;
};
std::unordered_map<uint64_t, ResW> g_resident;

uint64_t resident_key(const CactusQuantMatrix* W) {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](uint64_t v) { h ^= v; h *= 1099511628211ull; };
    mix(W->bits); mix(W->K); mix(W->N); mix(W->group_size); mix(W->num_groups); mix(W->flags);
    const uint8_t* p = W->packed_indices;
    size_t pkb = (size_t)W->N * W->num_groups * cactus_quant_packed_group_bytes(W->bits, W->group_size);
    if (!p || pkb == 0) return h;
    size_t take = pkb < 64 ? pkb : 64;
    for (size_t i = 0; i < take; ++i) mix(p[i]);
    for (size_t i = pkb - take; i < pkb; ++i) mix(p[i]);
    return h;
}

enum { W_PACKED, W_NORMS, W_CB, W_RECIP, W_LS, W_RS, W_PERM, W_CODE, W_ROT, W_SECT };

ResW& resident(const CactusQuantMatrix* W) {
    uint64_t key = resident_key(W);
    auto it = g_resident.find(key);
    if (it != g_resident.end()) return it->second;
    const uint32_t gs = W->group_size, ng = W->num_groups, N = W->N, K = W->K;
    const uint32_t pgb = cactus_quant_packed_group_bytes(W->bits, gs);
    const bool il = (W->flags & CACTUS_QUANT_FLAG_INTERLEAVED_4ROW) != 0;
    const bool ortho = (W->flags & CACTUS_QUANT_FLAG_ORTHOGONAL) != 0;
    const size_t rows = il ? ((size_t)(N + 3) & ~(size_t)3) : N;
    ResW r;
    const void* srcs[W_SECT] = {W->packed_indices, W->norms, W->codebook, W->input_scale_recip,
                                W->left_signs, W->right_signs, W->permutation, nullptr,
                                ortho ? W->rotation : nullptr};
    r.sz[W_PACKED] = rows * ng * pgb;
    r.sz[W_NORMS] = rows * ng * sizeof(__fp16);
    r.sz[W_CB] = 16 * sizeof(__fp16);
    r.sz[W_RECIP] = (size_t)K * sizeof(__fp16);
    r.sz[W_LS] = gs;
    r.sz[W_RS] = gs;
    r.sz[W_PERM] = (size_t)gs * sizeof(uint32_t);
    r.sz[W_CODE] = (size_t)K * sizeof(__fp16);
    r.sz[W_ROT] = ortho ? (size_t)K * K * sizeof(__fp16) : 0;
    VkDeviceSize total = 0;
    for (int i = 0; i < W_SECT; ++i) {
        total = (total + vk().sb_align - 1) & ~(vk().sb_align - 1);
        r.off[i] = total;
        total += r.sz[i];
    }
    if (create_block(total, r.b, false)) {
        for (int i = 0; i < W_SECT; ++i)
            if (srcs[i]) std::memcpy(r.b.map + r.off[i], srcs[i], r.sz[i]);
        r.ok = true;
    }
    return g_resident.emplace(key, r).first->second;
}

bool enc_cq_gemv(void* y, const void* x, const CactusQuantMatrix* W) {
    if (!vk().ok || !W || (W->bits != 4 && W->bits != 2 && W->bits != 3)) return false;
    if (W->bits != 4 && (W->flags & CACTUS_QUANT_FLAG_ORTHOGONAL)) return false;
    if (!W->codebook || !W->norms || !W->packed_indices) return false;
    const uint32_t gs = W->group_size, ng = W->num_groups, N = W->N, K = W->K;
    if ((size_t)ng * gs != K) return false;
    const bool ortho = (W->flags & CACTUS_QUANT_FLAG_ORTHOGONAL) != 0;
    if (ortho) {
        if (!W->rotation || ng != 1 || gs != K || (K & 31u) || K > 4096) return false;
    } else {
        if (!W->input_scale_recip || !W->left_signs || !W->right_signs || !W->permutation) return false;
        if (gs < 16 || gs > 512 || (gs & (gs - 1)) != 0) return false;
    }
    if ((W->flags & CACTUS_QUANT_FLAG_INTERLEAVED_4ROW) && (N & 3)) return false;
    ResW& r = resident(W);
    if (!r.ok) return false;
    VkDescriptorBufferInfo xb, yb;
    if (!dinfo(x, (size_t)K * sizeof(__fp16), &xb)) return false;
    if (!dinfo(y, (size_t)N * sizeof(__fp16), &yb)) return false;
    auto ri = [&](int i) {
        VkDescriptorBufferInfo d = {};
        d.buffer = r.b.buf;
        d.offset = r.off[i];
        d.range = r.sz[i] ? r.sz[i] : 4;
        return d;
    };
    if (!ensure_cmd()) return false;
    if (ortho) {
        VkDescriptorBufferInfo rinfos[4] = {xb, ri(W_RECIP), ri(W_ROT), ri(W_CODE)};
        VkDescriptorSet rset = make_set(KROT, rinfos, 4);
        if (!rset) return false;
        struct { uint32_t K, has_recip; } rpush = {K, W->input_scale_recip ? 1u : 0u};
        dispatch(KROT, rset, &rpush, (K + 63) / 64, 1, rinfos, 4);
    } else {
        VkDescriptorBufferInfo tinfos[6] = {xb, ri(W_RECIP), ri(W_LS), ri(W_RS), ri(W_PERM), ri(W_CODE)};
        VkDescriptorSet tset = make_set(KT, tinfos, 6);
        if (!tset) return false;
        struct { uint32_t gs; } tpush = {gs};
        dispatch(KT, tset, &tpush, ng, 1, tinfos, 6);
    }
    const uint32_t pgb = cactus_quant_packed_group_bytes(W->bits, gs);
    const uint32_t il = (W->flags & CACTUS_QUANT_FLAG_INTERLEAVED_4ROW) ? 1 : 0;
    if (W->bits != 4) {
        VkDescriptorBufferInfo ginfos[5] = {ri(W_CODE), ri(W_PACKED), ri(W_CB), ri(W_NORMS), yb};
        VkDescriptorSet gset = make_set(KLBG, ginfos, 5);
        if (!gset) return false;
        struct { uint32_t gs, ng, pgb, N, bits, il; } gpush = {gs, ng, pgb, N, W->bits, il};
        const uint32_t rows4 = (N + 3) / 4;
        dispatch(KLBG, gset, &gpush, rows4 > 65535 ? 65535 : rows4, 1, ginfos, 5);
        return true;
    }
    const int gk = (vk().gemv2_ok && K <= 8960u) ? KG2 : KG;
    VkDescriptorBufferInfo ginfos[6] = {ri(W_CODE), ri(W_PACKED), ri(W_CB), ri(W_NORMS), yb, ri(W_PACKED)};
    VkDescriptorSet gset = make_set(gk, ginfos, gk == KG2 ? 6 : 5);
    if (!gset) return false;
    struct { uint32_t gs, ng, pgb, N, il; } gpush = {gs, ng, pgb, N, il};
    const uint32_t rows = gk == KG2 ? (N + 15) / 16 : (N + 3) / 4;
    dispatch(gk, gset, &gpush, rows > 65535 ? 65535 : rows, 1, ginfos, gk == KG2 ? 6 : 5);
    return true;
}

struct HostIn {
    void* p = nullptr;
    HostIn(const void* src, size_t bytes) {
        p = alloc_shared_i(bytes);
        if (p && src) std::memcpy(p, src, bytes);
    }
    ~HostIn() { if (p) free_shared_i(p); }
};

}

bool cactus_vulkan_available() { return vk().ok; }

const char* cactus_vulkan_device_info() {
    if (vk().ok) {
        std::lock_guard<std::recursive_mutex> lk(g_mu);
        void* p = alloc_shared_i(1);
        if (p) free_shared_i(p);
    }
    return vk().info.c_str();
}

bool cactus_vulkan_op_enabled(const char* name) {
    static const char* filter = getenv("CACTUS_VK_OPS");
    if (!filter || !*filter) return true;
    return std::strstr(filter, name) != nullptr;
}

void cactus_vulkan_session_begin() {}

void cactus_vulkan_session_flush() {
    if (!vk().ok) return;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    flush_i();
}

void cactus_vulkan_session_sync() {
    if (!vk().ok) return;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    sync_i();
}

void cactus_vulkan_session_end() {
    cactus_vulkan_session_sync();
}

void cactus_vulkan_fold_buffers(void* h, size_t hbytes, void* ple, size_t plebytes) {
    if (!vk().ok) return;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    ov_register(h, hbytes);
    ov_register(ple, plebytes);
}

void cactus_vulkan_invalidate_host_wraps() {
    if (!vk().ok) return;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    sync_i();
    for (auto& kv : g_ov) {
        if (kv.second.blk) g_pending.push_back(kv.second.blk);
    }
    g_ov.clear();
    for (auto& kv : g_wraps) {
        if (kv.second->buf) vk().vkDestroyBuffer(vk().dev, kv.second->buf, nullptr);
        if (kv.second->mem) vk().vkFreeMemory(vk().dev, kv.second->mem, nullptr);
        delete kv.second;
    }
    g_wraps.clear();
}

void cactus_vulkan_trim_prefill_cache() {}

void* cactus_vulkan_alloc_shared(size_t bytes) {
    if (!vk().ok) return nullptr;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    return alloc_shared_i(bytes);
}

void* cactus_vulkan_alloc_pooled(size_t bytes) {
    if (!vk().ok) return nullptr;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    return alloc_pooled_i(bytes);
}

void cactus_vulkan_free_shared(void* p) {
    if (!vk().ok) return;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    free_shared_i(p);
}

bool cactus_vulkan_encode_binary_f16(int op, void* y, const void* a, const void* b, size_t n) {
    if (n == 0) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    struct { uint32_t n; int32_t op; } push = {(uint32_t)n, op};
    return enc_ew(KB, y, n * 2, a, n * 2, b, n * 2, &push, ew_groups(n));
}

bool cactus_vulkan_encode_scalar_f16(int op, void* y, const void* in, size_t n, float p) {
    if (n == 0) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    struct { uint32_t n; int32_t op; float p; } push = {(uint32_t)n, op, p};
    return enc_ew(KSC, y, n * 2, in, n * 2, nullptr, 0, &push, ew_groups(n));
}

bool cactus_vulkan_encode_unary_f16(int op, void* y, const void* in, size_t n) {
    if (n == 0) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    struct { uint32_t n; int32_t op; } push = {(uint32_t)n, op};
    return enc_ew(KU, y, n * 2, in, n * 2, nullptr, 0, &push, ew_groups(n));
}

bool cactus_vulkan_encode_swiglu_f16(void* y, const void* gate, const void* up, size_t n, float scale) {
    if (n == 0) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    struct { uint32_t n; float scale; } push = {(uint32_t)n, scale};
    return enc_ew(KSW, y, n * 2, gate, n * 2, up, n * 2, &push, ew_groups(n));
}

bool cactus_vulkan_encode_rms_norm_f16(void* y, const void* in, const void* w,
                                       size_t rows, size_t dim, float eps) {
    if (!vk().ok || rows == 0 || dim == 0 || rows > 65535) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    VkDescriptorBufferInfo infos[3];
    if (!sinfo(in, rows * dim * 2, &infos[0])) return false;
    if (!winfo(w, dim * 2, &infos[1])) return false;
    if (!dinfo(y, rows * dim * 2, &infos[2])) return false;
    if (!ensure_cmd()) return false;
    VkDescriptorSet set = make_set(KR, infos, 3);
    if (!set) return false;
    struct { uint32_t dim; float eps; } push = {(uint32_t)dim, eps};
    dispatch(KR, set, &push, (uint32_t)rows, 1, infos, 3);
    return true;
}

bool cactus_vulkan_encode_cq_gemv(void* y, const void* x, const CactusQuantMatrix* W) {
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    return enc_cq_gemv(y, x, W);
}

bool cactus_vulkan_binary_f16(int op, __fp16* out, const __fp16* a, const __fp16* b, size_t n) {
    if (!vk().ok || n == 0) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    HostIn ia(a, n * 2), ib(b, n * 2), oy(nullptr, n * 2);
    if (!ia.p || !ib.p || !oy.p) return false;
    struct { uint32_t n; int32_t op; } push = {(uint32_t)n, op};
    bool r = enc_ew(KB, oy.p, n * 2, ia.p, n * 2, ib.p, n * 2, &push, ew_groups(n));
    sync_i();
    if (r) std::memcpy(out, oy.p, n * 2);
    return r && vk().ok;
}

bool cactus_vulkan_scalar_f16(int op, __fp16* out, const __fp16* in, size_t n, float p) {
    if (!vk().ok || n == 0) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    HostIn ii(in, n * 2), oy(nullptr, n * 2);
    if (!ii.p || !oy.p) return false;
    struct { uint32_t n; int32_t op; float p; } push = {(uint32_t)n, op, p};
    bool r = enc_ew(KSC, oy.p, n * 2, ii.p, n * 2, nullptr, 0, &push, ew_groups(n));
    sync_i();
    if (r) std::memcpy(out, oy.p, n * 2);
    return r && vk().ok;
}

bool cactus_vulkan_unary_f16(int op, __fp16* out, const __fp16* in, size_t n) {
    if (!vk().ok || n == 0) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    HostIn ii(in, n * 2), oy(nullptr, n * 2);
    if (!ii.p || !oy.p) return false;
    struct { uint32_t n; int32_t op; } push = {(uint32_t)n, op};
    bool r = enc_ew(KU, oy.p, n * 2, ii.p, n * 2, nullptr, 0, &push, ew_groups(n));
    sync_i();
    if (r) std::memcpy(out, oy.p, n * 2);
    return r && vk().ok;
}

bool cactus_vulkan_swiglu_f16(__fp16* out, const __fp16* gate, const __fp16* up, size_t n, float scale) {
    if (!vk().ok || n == 0) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    HostIn ig(gate, n * 2), iu(up, n * 2), oy(nullptr, n * 2);
    if (!ig.p || !iu.p || !oy.p) return false;
    struct { uint32_t n; float scale; } push = {(uint32_t)n, scale};
    bool r = enc_ew(KSW, oy.p, n * 2, ig.p, n * 2, iu.p, n * 2, &push, ew_groups(n));
    sync_i();
    if (r) std::memcpy(out, oy.p, n * 2);
    return r && vk().ok;
}

bool cactus_vulkan_rms_norm_f16(__fp16* out, const __fp16* in, const __fp16* weight,
                                size_t rows, size_t dim, float eps) {
    if (!vk().ok || rows == 0 || dim == 0 || rows > 65535) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    HostIn ii(in, rows * dim * 2), iw(weight, dim * 2), oy(nullptr, rows * dim * 2);
    if (!ii.p || !iw.p || !oy.p) return false;
    struct { uint32_t dim; float eps; } push = {(uint32_t)dim, eps};
    bool r = enc_ew(KR, oy.p, rows * dim * 2, ii.p, rows * dim * 2, iw.p, dim * 2, &push, (uint32_t)rows);
    sync_i();
    if (r) std::memcpy(out, oy.p, rows * dim * 2);
    return r && vk().ok;
}

bool cactus_vulkan_cq_gemv(__fp16* y, const __fp16* x, const CactusQuantMatrix* W,
                           int iters, double* kernel_ms) {
    if (!vk().ok || !W) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    const size_t xb = (size_t)W->K * 2, yb = (size_t)W->N * 2;
    HostIn ix(x, xb), oy(nullptr, yb);
    if (!ix.p || !oy.p) return false;
    const int reps = iters > 1 ? iters : 1;
    const bool ts = kernel_ms && vk().ts_period > 0.0f;
    if (!ensure_cmd()) return false;
    if (ts) {
        vk().vkCmdResetQueryPool(cur_cmd(), vk().query, 0, 2);
        vk().vkCmdWriteTimestamp(cur_cmd(), VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, vk().query, 0);
    }
    for (int i = 0; i < reps; ++i)
        if (!enc_cq_gemv(oy.p, ix.p, W)) return false;
    if (ts)
        vk().vkCmdWriteTimestamp(cur_cmd(), VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, vk().query, 1);
    sync_i();
    if (!vk().ok) return false;
    std::memcpy(y, oy.p, yb);
    if (ts) {
        uint64_t stamps[2] = {};
        if (vk().vkGetQueryPoolResults(vk().dev, vk().query, 0, 2, sizeof(stamps), stamps,
                                       sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS)
            *kernel_ms = (double)(stamps[1] - stamps[0]) * vk().ts_period / 1e6;
        else
            *kernel_ms = 0.0;
    } else if (kernel_ms) {
        *kernel_ms = 0.0;
    }
    return true;
}

bool cactus_vulkan_encode_copy(void* out, const void* in, size_t bytes) {
    if (bytes == 0 || (bytes & 3u)) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    uint32_t n = (uint32_t)(bytes >> 2);
    struct { uint32_t n, o_in, o_out; } push = {n, 0, 0};
    return enc_ew(KCP, out, bytes, in, bytes, nullptr, 0, &push, ew_groups(n));
}

bool enc_copy_off(void* out, size_t out_bytes, const void* in, size_t in_bytes, size_t bytes) {
    if (bytes == 0 || (bytes & 3u)) return false;
    VkDescriptorBufferInfo infos[2];
    uint32_t o_in = 0, o_out = 0;
    if (!dinfo_off(in, in_bytes, &infos[0], &o_in)) return false;
    if (!dinfo_off(out, out_bytes, &infos[1], &o_out)) return false;
    if (!ensure_cmd()) return false;
    VkDescriptorSet set = make_set(KCP, infos, 2);
    if (!set) return false;
    uint32_t n = (uint32_t)(bytes >> 2);
    struct { uint32_t n, o_in, o_out; } push = {n, o_in, o_out};
    dispatch(KCP, set, &push, ew_groups(n), 1, infos, 2);
    return true;
}

MemBlock g_slide;

bool cactus_vulkan_encode_kv_append_sliding_i8(const void* src, void* int8base, void* scalebase,
        uint32_t kv_heads, uint32_t hdim, uint32_t keep_sink, uint32_t remaining, uint32_t shift_src,
        uint32_t group_size, uint32_t M, size_t src_bytes, size_t int8_bytes, size_t scale_bytes) {
    if (!vk().ok || M == 0 || kv_heads == 0 || group_size == 0 || (group_size & 3u)
        || hdim == 0 || (hdim % group_size)) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    const size_t i8s = (size_t)kv_heads * hdim;
    const size_t scs = (size_t)kv_heads * (hdim / group_size) * 4;
    if (remaining > 0) {
        const size_t di8 = remaining * i8s, dsc = remaining * scs;
        VkDescriptorBufferInfo sinfo;
        if (!scratch_info(g_slide, di8 + dsc, &sinfo)) return false;
        char* i8b = (char*)int8base;
        char* scb = (char*)scalebase;
        if (!enc_copy_off(g_slide.map, di8 + dsc, i8b + shift_src * i8s, int8_bytes - shift_src * i8s, di8)) return false;
        if (!enc_copy_off(g_slide.map + di8, dsc, scb + shift_src * scs, scale_bytes - shift_src * scs, dsc)) return false;
        if (!enc_copy_off(i8b + keep_sink * i8s, int8_bytes - keep_sink * i8s, g_slide.map, di8, di8)) return false;
        if (!enc_copy_off(scb + keep_sink * scs, scale_bytes - keep_sink * scs, g_slide.map + di8, dsc, dsc)) return false;
    }
    return cactus_vulkan_encode_kv_append_i8(src, int8base, scalebase, kv_heads, hdim,
        keep_sink + remaining, group_size, M, 0, 0, src_bytes, int8_bytes, scale_bytes);
}

bool cactus_vulkan_prewarm_quant(const CactusQuantMatrix* W) {
    if (!vk().ok || !W || W->bits != 4 || !W->packed_indices) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    return resident(W).ok;
}

bool cactus_vulkan_encode_transform_batch(const void* x, const CactusQuantMatrix* const* Ws,
                                          int B, void* const* codes) {
    if (!vk().ok || B < 1 || B > 3 || !x) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    for (int bi = 0; bi < B; ++bi) {
        const CactusQuantMatrix* W = Ws[bi];
        if (!W || W->bits != 4 || (W->flags & CACTUS_QUANT_FLAG_ORTHOGONAL)) return false;
        if (!W->input_scale_recip || !W->left_signs || !W->right_signs || !W->permutation) return false;
        const uint32_t gs = W->group_size, ng = W->num_groups, K = W->K;
        if (gs < 16 || gs > 512 || (gs & (gs - 1)) != 0 || (size_t)ng * gs != K) return false;
        ResW& r = resident(W);
        if (!r.ok) return false;
        VkDescriptorBufferInfo xb, cb;
        if (!dinfo(x, (size_t)K * 2, &xb)) return false;
        if (!dinfo(codes[bi], (size_t)K * 2, &cb)) return false;
        auto ri = [&](int i) {
            VkDescriptorBufferInfo d = {};
            d.buffer = r.b.buf;
            d.offset = r.off[i];
            d.range = r.sz[i] ? r.sz[i] : 4;
            return d;
        };
        if (!ensure_cmd()) return false;
        VkDescriptorBufferInfo tinfos[6] = {xb, ri(W_RECIP), ri(W_LS), ri(W_RS), ri(W_PERM), cb};
        VkDescriptorSet tset = make_set(KT, tinfos, 6);
        if (!tset) return false;
        struct { uint32_t gs; } tpush = {gs};
        dispatch(KT, tset, &tpush, ng, 1, tinfos, 6);
    }
    return true;
}

bool cactus_vulkan_encode_gemv_precoded(void* out, const void* code, const CactusQuantMatrix* W) {
    if (!vk().ok || !W || W->bits != 4 || (W->flags & CACTUS_QUANT_FLAG_ORTHOGONAL)) return false;
    const uint32_t gs = W->group_size, ng = W->num_groups, N = W->N, K = W->K;
    if ((size_t)ng * gs != K || ((W->flags & CACTUS_QUANT_FLAG_INTERLEAVED_4ROW) && (N & 3))) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    ResW& r = resident(W);
    if (!r.ok) return false;
    VkDescriptorBufferInfo cb, yb;
    if (!dinfo(code, (size_t)K * 2, &cb)) return false;
    if (!dinfo(out, (size_t)N * 2, &yb)) return false;
    auto ri = [&](int i) {
        VkDescriptorBufferInfo d = {};
        d.buffer = r.b.buf;
        d.offset = r.off[i];
        d.range = r.sz[i] ? r.sz[i] : 4;
        return d;
    };
    if (!ensure_cmd()) return false;
    const int gk = (vk().gemv2_ok && K <= 8960u) ? KG2 : KG;
    VkDescriptorBufferInfo ginfos[6] = {cb, ri(W_PACKED), ri(W_CB), ri(W_NORMS), yb, ri(W_PACKED)};
    VkDescriptorSet gset = make_set(gk, ginfos, gk == KG2 ? 6 : 5);
    if (!gset) return false;
    const uint32_t pgb = cactus_quant_packed_group_bytes(W->bits, gs);
    const uint32_t il = (W->flags & CACTUS_QUANT_FLAG_INTERLEAVED_4ROW) ? 1 : 0;
    struct { uint32_t gs, ng, pgb, N, il; } gpush = {gs, ng, pgb, N, il};
    const uint32_t rows = gk == KG2 ? (N + 15) / 16 : (N + 3) / 4;
    dispatch(gk, gset, &gpush, rows > 65535 ? 65535 : rows, 1, ginfos, gk == KG2 ? 6 : 5);
    return true;
}

MemBlock g_code_m;

bool cactus_vulkan_encode_quant_matmul_m(void* out, const void* lhs, const CactusQuantMatrix* W, uint32_t M) {
    if (!vk().ok || !W || M == 0) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    if (M == 1) return enc_cq_gemv(out, lhs, W);
    if (!vk().gemv2_ok || W->bits != 4 || (W->flags & CACTUS_QUANT_FLAG_ORTHOGONAL)) return false;
    if (!W->input_scale_recip || !W->left_signs || !W->right_signs || !W->permutation
        || !W->codebook || !W->norms || !W->packed_indices) return false;
    const uint32_t gs = W->group_size, ng = W->num_groups, N = W->N, K = W->K;
    if (gs < 16 || gs > 512 || (gs & (gs - 1)) != 0 || (size_t)ng * gs != K) return false;
    if ((W->flags & CACTUS_QUANT_FLAG_INTERLEAVED_4ROW) && (N & 3)) return false;
    if (M > 65535) return false;
    ResW& r = resident(W);
    if (!r.ok) return false;
    VkDescriptorBufferInfo xb, yb, cinfo;
    if (!dinfo(lhs, (size_t)M * K * 2, &xb)) return false;
    if (!dinfo(out, (size_t)M * N * 2, &yb)) return false;
    if (!scratch_info(g_code_m, (size_t)M * K * 2, &cinfo)) return false;
    auto ri = [&](int i) {
        VkDescriptorBufferInfo d = {};
        d.buffer = r.b.buf;
        d.offset = r.off[i];
        d.range = r.sz[i] ? r.sz[i] : 4;
        return d;
    };
    if (!ensure_cmd()) return false;
    VkDescriptorBufferInfo tinfos[6] = {xb, ri(W_RECIP), ri(W_LS), ri(W_RS), ri(W_PERM), cinfo};
    VkDescriptorSet tset = make_set(KTM, tinfos, 6);
    if (!tset) return false;
    struct { uint32_t gs, K; } tpush = {gs, K};
    dispatch(KTM, tset, &tpush, ng, M, tinfos, 6);
    VkDescriptorBufferInfo ginfos[6] = {cinfo, ri(W_PACKED), ri(W_CB), ri(W_NORMS), yb, ri(W_PACKED)};
    VkDescriptorSet gset = make_set(KGM, ginfos, 6);
    if (!gset) return false;
    const uint32_t pgb = cactus_quant_packed_group_bytes(W->bits, gs);
    const uint32_t il = (W->flags & CACTUS_QUANT_FLAG_INTERLEAVED_4ROW) ? 1 : 0;
    struct { uint32_t gs, ng, pgb, N, il, M; } gpush = {gs, ng, pgb, N, il, M};
    const uint32_t rows4 = (N + 3) / 4;
    dispatch(KGM, gset, &gpush, rows4 > 65535 ? 65535 : rows4, (M + 7) / 8, ginfos, 6);
    return true;
}

bool cactus_vulkan_encode_rope_pair(void* out, const void* x, const void* c, const void* s,
                                    uint32_t H, uint32_t D) {
    if (!vk().ok || H == 0 || H > 65535 || D == 0 || (D & 1u)) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    VkDescriptorBufferInfo infos[4];
    if (!dinfo(x, (size_t)H * D * 2, &infos[0])) return false;
    if (!dinfo(c, (size_t)D * 2, &infos[1])) return false;
    if (!dinfo(s, (size_t)D * 2, &infos[2])) return false;
    if (!dinfo(out, (size_t)H * D * 2, &infos[3])) return false;
    if (!ensure_cmd()) return false;
    VkDescriptorSet set = make_set(KRP, infos, 4);
    if (!set) return false;
    struct { uint32_t H, D; } push = {H, D};
    dispatch(KRP, set, &push, (D + 63) / 64, H, infos, 4);
    return true;
}

bool cactus_vulkan_encode_rope_pair_rms(void* out, const void* x, const void* w,
                                        const void* c, const void* s,
                                        uint32_t H, uint32_t D, float eps) {
    if (!vk().ok || H == 0 || H > 65535 || D == 0 || (D & 1u) || D > 1024) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    VkDescriptorBufferInfo infos[5];
    if (!dinfo(x, (size_t)H * D * 2, &infos[0])) return false;
    if (!winfo(w, (size_t)D * 2, &infos[1])) return false;
    if (!winfo(c, (size_t)D * 2, &infos[2])) return false;
    if (!winfo(s, (size_t)D * 2, &infos[3])) return false;
    if (!dinfo(out, (size_t)H * D * 2, &infos[4])) return false;
    if (!ensure_cmd()) return false;
    VkDescriptorSet set = make_set(KRPS, infos, 5);
    if (!set) return false;
    struct { uint32_t D; float eps; } push = {D, eps};
    dispatch(KRPS, set, &push, H, 1, infos, 5);
    return true;
}

bool cactus_vulkan_encode_rms2_add_clip(void* out, const void* a, const void* wa,
                                        const void* b, const void* wb, size_t dim,
                                        float eps_a, float eps_b) {
    if (!vk().ok || dim == 0) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    VkDescriptorBufferInfo infos[5];
    if (!dinfo(a, dim * 2, &infos[0])) return false;
    if (!winfo(wa, dim * 2, &infos[1])) return false;
    if (!dinfo(b, dim * 2, &infos[2])) return false;
    if (!winfo(wb, dim * 2, &infos[3])) return false;
    if (!dinfo(out, dim * 2, &infos[4])) return false;
    if (!ensure_cmd()) return false;
    VkDescriptorSet set = make_set(KR2C, infos, 5);
    if (!set) return false;
    struct { uint32_t D; float ea, eb; } push = {(uint32_t)dim, eps_a, eps_b};
    dispatch(KR2C, set, &push, 1, 1, infos, 5);
    return true;
}

bool cactus_vulkan_encode_gather_f16(void* out, const void* table, size_t table_bytes,
                                     const uint32_t* rows, uint32_t M, uint32_t D) {
    if (!vk().ok || M == 0 || D == 0 || !rows) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    uint32_t n = M * D;
    VkDescriptorBufferInfo infos[3];
    if (!dinfo(table, table_bytes, &infos[0])) {
        // Host-resident table (mmap weight): gather the M rows on CPU into the
        // upload ring — bytes moved are M*D*2, not the table — and encode a
        // device copy. Metal reads the table zero-copy; this is the Vulkan
        // equivalent without a giant residency allocation.
        char* g = (char*)stage_alloc((size_t)n * 2);
        if (!g) return false;
        const char* tb = (const char*)table;
        for (uint32_t i = 0; i < M; ++i) {
            size_t off = (size_t)rows[i] * D * 2;
            if (off + (size_t)D * 2 > table_bytes) return false;
            std::memcpy(g + (size_t)i * D * 2, tb + off, (size_t)D * 2);
        }
        return cactus_vulkan_encode_copy(out, g, (size_t)n * 2);
    }
    const void* rblob = stage_upload(rows, (size_t)M * 4);
    if (!rblob || !dinfo(rblob, (size_t)M * 4, &infos[1])) return false;
    if (!dinfo(out, (size_t)n * 2, &infos[2])) return false;
    if (!ensure_cmd()) return false;
    VkDescriptorSet set = make_set(KGH, infos, 3);
    if (!set) return false;
    struct { uint32_t D, n; } push = {D, n};
    uint32_t gx = (n + 63) / 64;
    dispatch(KGH, set, &push, gx > 65535 ? 65535 : gx, 1, infos, 3);
    return true;
}

bool cactus_vulkan_encode_softmax_topk(void* probs, void* topk, const void* in,
                                       size_t rows, size_t cols, size_t k, float scale) {
    if (!vk().ok || k == 0 || k > 16 || rows == 0 || rows > 65535 || cols == 0) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    VkDescriptorBufferInfo infos[3];
    if (!dinfo(in, rows * cols * 2, &infos[0])) return false;
    if (!dinfo(probs, rows * cols * 2, &infos[1])) return false;
    if (!dinfo(topk, rows * k * 2 * sizeof(float), &infos[2])) return false;
    if (!ensure_cmd()) return false;
    VkDescriptorSet set = make_set(KSTK, infos, 3);
    if (!set) return false;
    struct { uint32_t E, k, B; float scale; } push = {(uint32_t)cols, (uint32_t)k, (uint32_t)rows, scale};
    dispatch(KSTK, set, &push, (uint32_t)rows, 1, infos, 3);
    return true;
}

bool cactus_vulkan_encode_topk_rows(void* out, const void* in, size_t rows, size_t cols, size_t k) {
    if (!vk().ok || k == 0 || k > 16 || rows == 0 || rows > 65535 || cols == 0) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    VkDescriptorBufferInfo infos[2];
    if (!dinfo(in, rows * cols * 2, &infos[0])) return false;
    if (!dinfo(out, rows * k * 2 * sizeof(float), &infos[1])) return false;
    if (!ensure_cmd()) return false;
    VkDescriptorSet set = make_set(KTKR, infos, 2);
    if (!set) return false;
    struct { uint32_t F, k, B; } push = {(uint32_t)cols, (uint32_t)k, (uint32_t)rows};
    dispatch(KTKR, set, &push, (uint32_t)rows, 1, infos, 2);
    return true;
}

bool cactus_vulkan_encode_gemv_bias(void* out, const void* x, const void* w, const void* bias,
                                    uint32_t K, uint32_t N, int tr) {
    if (!vk().ok || K == 0 || N == 0) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    VkDescriptorBufferInfo infos[4];
    if (!dinfo(x, (size_t)K * 2, &infos[0])) return false;
    if (!winfo(w, (size_t)K * N * 2, &infos[1])) return false;
    if (!winfo(bias, (size_t)N * 2, &infos[2])) return false;
    if (!dinfo(out, (size_t)N * 2, &infos[3])) return false;
    if (!ensure_cmd()) return false;
    VkDescriptorSet set = make_set(KGB, infos, 4);
    if (!set) return false;
    struct { uint32_t K, N, tr; } push = {K, N, tr ? 1u : 0u};
    uint32_t gx = (N + 63) / 64;
    dispatch(KGB, set, &push, gx > 65535 ? 65535 : gx, 1, infos, 4);
    return true;
}

bool cactus_vulkan_encode_rms_norm_add_rows(void* ysum, void* ynorm, const void* x, const void* res,
                                            const void* w, uint32_t rows, uint32_t dim, float eps,
                                            int clipped) {
    if (!vk().ok || rows == 0 || rows > 65535 || dim == 0) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    VkDescriptorBufferInfo infos[5];
    if (!dinfo(x, (size_t)rows * dim * 2, &infos[0])) return false;
    if (!dinfo(res, (size_t)rows * dim * 2, &infos[1])) return false;
    if (!winfo(w, (size_t)dim * 2, &infos[2])) return false;
    if (!dinfo(ysum, (size_t)rows * dim * 2, &infos[3])) return false;
    if (!dinfo(ynorm, (size_t)rows * dim * 2, &infos[4])) return false;
    if (!ensure_cmd()) return false;
    VkDescriptorSet set = make_set(KRAR, infos, 5);
    if (!set) return false;
    struct { uint32_t dim; float eps; uint32_t rows, clipped; } push = {dim, eps, rows, clipped ? 1u : 0u};
    dispatch(KRAR, set, &push, rows, 1, infos, 5);
    return true;
}

bool cactus_vulkan_encode_attention_i8_prefill(
        void* out, const void* q, const void* knew, const void* vnew,
        const void* kc, const void* vc, const void* ks, const void* vs,
        uint32_t nqh, uint32_t nkvh, uint32_t hd, uint32_t vhd,
        uint32_t hist, uint32_t new_len, uint32_t q_pos0, uint32_t window, uint32_t is_causal,
        uint32_t M, float scale, size_t kc_bytes, size_t vc_bytes, size_t ks_bytes, size_t vs_bytes,
        uint32_t sink, uint32_t ring) {
    (void)window;
    if (!vk().ok || M == 0 || nkvh == 0 || (nqh % nkvh) != 0) return false;
    uint32_t total_keys = hist + new_len;
    if (total_keys == 0) return false;
    if ((hd & 3u) || (vhd & 3u) || hd > 1024u || vhd > 1024u) return false;
    if ((size_t)M * nqh > 65535u) return false;
    const int kk = ring > 0u ? KAPR : KAP;
    if (!vk().pipes[kk].p) return false;
    if (ring > 0u) {
        uint32_t maxsc = total_keys > sink + ring ? sink + ring : total_keys;
        if (maxsc > 7936u) return false;
    }
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    VkDescriptorBufferInfo infos[8];
    uint32_t okc = 0, ovc = 0, oks = 0, ovs = 0;
    if (!dinfo(q, (size_t)M * nqh * hd * 2, &infos[0])) return false;
    if (new_len > 0 && knew && vnew) {
        if (!dinfo(knew, (size_t)new_len * nkvh * hd * 2, &infos[1])) return false;
        if (!dinfo(vnew, (size_t)new_len * nkvh * vhd * 2, &infos[2])) return false;
    } else {
        if (!dummy_info(&infos[1]) || !dummy_info(&infos[2])) return false;
    }
    if (kc && hist) { if (!dinfo_off(kc, kc_bytes, &infos[3], &okc)) return false; }
    else if (!dummy_info(&infos[3])) return false;
    if (vc && hist) { if (!dinfo_off(vc, vc_bytes, &infos[4], &ovc)) return false; }
    else if (!dummy_info(&infos[4])) return false;
    if (ks && hist) { if (!dinfo_off(ks, ks_bytes, &infos[5], &oks)) return false; }
    else if (!dummy_info(&infos[5])) return false;
    if (vs && hist) { if (!dinfo_off(vs, vs_bytes, &infos[6], &ovs)) return false; }
    else if (!dummy_info(&infos[6])) return false;
    if (!dinfo(out, (size_t)M * nqh * vhd * 2, &infos[7])) return false;
    if (!ensure_cmd()) return false;
    VkDescriptorSet set = make_set(kk, infos, 8);
    if (!set) return false;
    if (ring > 0u) {
        struct { uint32_t nqh, nkvh, hd, vhd, hist; float scale;
                 uint32_t q_pos0, new_len, is_causal, sinkN, ringR, okc, ovc, oks, ovs; } push =
            {nqh, nkvh, hd, vhd, hist, scale, q_pos0, new_len, is_causal, sink, ring, okc, ovc, oks, ovs};
        dispatch(kk, set, &push, M * nqh, 1, infos, 8);
    } else {
        struct { uint32_t nqh, nkvh, hd, vhd, hist; float scale;
                 uint32_t q_pos0, new_len, is_causal, okc, ovc, oks, ovs; } push =
            {nqh, nkvh, hd, vhd, hist, scale, q_pos0, new_len, is_causal, okc, ovc, oks, ovs};
        dispatch(kk, set, &push, M * nqh, 1, infos, 8);
    }
    return true;
}

bool cactus_vulkan_encode_layer_norm(void* out, const void* in, const void* w, const void* b,
                                     size_t rows, size_t dim, float eps) {
    if (!vk().ok || rows == 0 || rows > 65535 || dim == 0) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    VkDescriptorBufferInfo infos[4];
    if (!dinfo(in, rows * dim * 2, &infos[0])) return false;
    if (!winfo(w, dim * 2, &infos[1])) return false;
    if (b) { if (!winfo(b, dim * 2, &infos[2])) return false; }
    else if (!dummy_info(&infos[2])) return false;
    if (!dinfo(out, rows * dim * 2, &infos[3])) return false;
    if (!ensure_cmd()) return false;
    VkDescriptorSet set = make_set(KLN, infos, 4);
    if (!set) return false;
    struct { uint32_t dim; float eps; uint32_t hb; } push = {(uint32_t)dim, eps, b ? 1u : 0u};
    dispatch(KLN, set, &push, (uint32_t)rows, 1, infos, 4);
    return true;
}

bool cactus_vulkan_encode_softmax_rows(void* out, const void* in, size_t rows, size_t cols) {
    if (!vk().ok || rows == 0 || rows > 65535 || cols == 0) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    VkDescriptorBufferInfo infos[2];
    if (!dinfo(in, rows * cols * 2, &infos[0])) return false;
    if (!dinfo(out, rows * cols * 2, &infos[1])) return false;
    if (!ensure_cmd()) return false;
    VkDescriptorSet set = make_set(KSMR, infos, 2);
    if (!set) return false;
    struct { uint32_t cols; } push = {(uint32_t)cols};
    dispatch(KSMR, set, &push, (uint32_t)rows, 1, infos, 2);
    return true;
}

bool cactus_vulkan_encode_glu(void* out, const void* in, size_t split, size_t inner, size_t n_out) {
    if (!vk().ok || n_out == 0 || split == 0 || inner == 0) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    VkDescriptorBufferInfo infos[2];
    if (!dinfo(in, n_out * 4, &infos[0])) return false;
    if (!dinfo(out, n_out * 2, &infos[1])) return false;
    if (!ensure_cmd()) return false;
    VkDescriptorSet set = make_set(KGLU, infos, 2);
    if (!set) return false;
    struct { uint32_t split, inner, n; } push = {(uint32_t)split, (uint32_t)inner, (uint32_t)n_out};
    uint32_t gx = (uint32_t)((n_out + 63) / 64);
    dispatch(KGLU, set, &push, gx > 65535 ? 65535 : gx, 1, infos, 2);
    return true;
}

bool cactus_vulkan_encode_conv1d_k3(void* out, const void* x, const void* w, int w_int8,
                                    const void* w_scales, uint32_t w_gs,
                                    uint32_t Cin, uint32_t L, uint32_t Cout, uint32_t Lout, uint32_t stride) {
    (void)w_scales; (void)w_gs;
    if (!vk().ok || w_int8 || Cin == 0 || L == 0 || Cout == 0 || Cout > 65535 || Lout == 0) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    VkDescriptorBufferInfo infos[3];
    if (!dinfo(x, (size_t)Cin * L * 2, &infos[0])) return false;
    if (!winfo(w, (size_t)Cout * Cin * 3 * 2, &infos[1])) return false;
    if (!dinfo(out, (size_t)Cout * Lout * 2, &infos[2])) return false;
    if (!ensure_cmd()) return false;
    VkDescriptorSet set = make_set(KCK3, infos, 3);
    if (!set) return false;
    struct { uint32_t Cin, L, Lout, stride; } push = {Cin, L, Lout, stride};
    dispatch(KCK3, set, &push, (Lout + 63) / 64, Cout, infos, 3);
    return true;
}

bool cactus_vulkan_encode_conv1d_dw(void* out, const void* x, const void* w,
                                    uint32_t C, uint32_t L, uint32_t Lout, uint32_t K, uint32_t stride) {
    if (!vk().ok || C == 0 || C > 65535 || L == 0 || Lout == 0 || K == 0) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    VkDescriptorBufferInfo infos[3];
    if (!dinfo(x, (size_t)C * L * 2, &infos[0])) return false;
    if (!winfo(w, (size_t)C * K * 2, &infos[1])) return false;
    if (!dinfo(out, (size_t)C * Lout * 2, &infos[2])) return false;
    if (!ensure_cmd()) return false;
    VkDescriptorSet set = make_set(KCDW, infos, 3);
    if (!set) return false;
    struct { uint32_t L, Lout, K, stride; } push = {L, Lout, K, stride};
    dispatch(KCDW, set, &push, (Lout + 63) / 64, C, infos, 3);
    return true;
}

bool cactus_vulkan_encode_conv1d_gen(void* out, const void* x, const void* w, const void* bias,
                                     uint32_t N, uint32_t Cin, uint32_t L, uint32_t Cout,
                                     uint32_t Lout, uint32_t K, uint32_t stride, int w_ck_co) {
    if (!vk().ok || N == 0 || Cin == 0 || L == 0 || Cout == 0 || Cout > 65535 || Lout == 0 || K == 0) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    VkDescriptorBufferInfo infos[4];
    if (!dinfo(x, (size_t)N * Cin * L * 2, &infos[0])) return false;
    if (!winfo(w, (size_t)Cout * Cin * K * 2, &infos[1])) return false;
    if (bias) { if (!winfo(bias, (size_t)Cout * 2, &infos[2])) return false; }
    else if (!dummy_info(&infos[2])) return false;
    if (!dinfo(out, (size_t)N * Cout * Lout * 2, &infos[3])) return false;
    if (!ensure_cmd()) return false;
    VkDescriptorSet set = make_set(KCGN, infos, 4);
    if (!set) return false;
    struct { uint32_t N, Cin, L, Cout, Lout, K, stride, hb, wl; } push =
        {N, Cin, L, Cout, Lout, K, stride, bias ? 1u : 0u, w_ck_co ? 1u : 0u};
    dispatch(KCGN, set, &push, (Lout + 63) / 64, Cout, infos, 4);
    return true;
}

bool cactus_vulkan_encode_conv1d_nlc_dw(void* out, const void* x, const void* w, const void* bias,
                                        uint32_t N, uint32_t L, uint32_t C, uint32_t K,
                                        uint32_t dil, uint32_t pad) {
    if (!vk().ok || N == 0 || L == 0 || L > 65535 || C == 0 || K == 0) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    VkDescriptorBufferInfo infos[4];
    if (!dinfo(x, (size_t)N * L * C * 2, &infos[0])) return false;
    if (!winfo(w, (size_t)C * K * 2, &infos[1])) return false;
    if (bias) { if (!winfo(bias, (size_t)C * 2, &infos[2])) return false; }
    else if (!dummy_info(&infos[2])) return false;
    if (!dinfo(out, (size_t)N * L * C * 2, &infos[3])) return false;
    if (!ensure_cmd()) return false;
    VkDescriptorSet set = make_set(KCND, infos, 4);
    if (!set) return false;
    struct { uint32_t N, L, C, K, dil, pad, hb; } push = {N, L, C, K, dil, pad, bias ? 1u : 0u};
    dispatch(KCND, set, &push, (C + 63) / 64, L, infos, 4);
    return true;
}

bool cactus_vulkan_encode_batchnorm(void* out, const void* x, const void* w, const void* b,
                                    const void* rm, const void* rv, uint32_t C, uint32_t inner,
                                    uint32_t total, float eps) {
    if (!vk().ok || C == 0 || inner == 0 || total == 0) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    VkDescriptorBufferInfo infos[6];
    if (!dinfo(x, (size_t)total * 2, &infos[0])) return false;
    if (!winfo(w, (size_t)C * 2, &infos[1])) return false;
    if (!winfo(b, (size_t)C * 2, &infos[2])) return false;
    if (!winfo(rm, (size_t)C * 2, &infos[3])) return false;
    if (!winfo(rv, (size_t)C * 2, &infos[4])) return false;
    if (!dinfo(out, (size_t)total * 2, &infos[5])) return false;
    if (!ensure_cmd()) return false;
    VkDescriptorSet set = make_set(KBN2, infos, 6);
    if (!set) return false;
    struct { uint32_t C, inner; float eps; uint32_t n; } push = {C, inner, eps, total};
    uint32_t gx = (total + 63) / 64;
    dispatch(KBN2, set, &push, gx > 65535 ? 65535 : gx, 1, infos, 6);
    return true;
}

bool cactus_vulkan_encode_bias_add_rows(void* y, const void* bias, uint32_t C, uint32_t total) {
    if (!vk().ok || C == 0 || total == 0) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    VkDescriptorBufferInfo infos[2];
    if (!dinfo(y, (size_t)total * 2, &infos[0])) return false;
    if (!winfo(bias, (size_t)C * 2, &infos[1])) return false;
    if (!ensure_cmd()) return false;
    VkDescriptorSet set = make_set(KBAR, infos, 2);
    if (!set) return false;
    struct { uint32_t C, n; } push = {C, total};
    uint32_t gx = (total + 63) / 64;
    dispatch(KBAR, set, &push, gx > 65535 ? 65535 : gx, 1, infos, 2);
    return true;
}

bool cactus_vulkan_encode_gemm_batch(void* out, const void* a, const void* b,
                                     uint32_t M, uint32_t K, uint32_t N, uint32_t batch,
                                     int f32out, int f32a) {
    if (!vk().ok || M == 0 || K == 0 || N == 0 || batch == 0 || batch > 65535) return false;
    if ((size_t)M * N > (size_t)65535 * 64) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    VkDescriptorBufferInfo infos[5];
    size_t ab = (size_t)batch * M * K * (f32a ? 4 : 2);
    if (!dinfo(a, ab, &infos[0])) return false;
    infos[1] = infos[0];
    if (!dinfo(b, (size_t)batch * K * N * 2, &infos[2])) return false;
    if (!dinfo(out, (size_t)batch * M * N * (f32out ? 4 : 2), &infos[3])) return false;
    infos[4] = infos[3];
    if (!ensure_cmd()) return false;
    VkDescriptorSet set = make_set(KGMB, infos, 5);
    if (!set) return false;
    struct { uint32_t M, K, N, f32out, f32a, tr; } push = {M, K, N, f32out ? 1u : 0u, f32a ? 1u : 0u, 0u};
    dispatch(KGMB, set, &push, ((uint32_t)((size_t)M * N) + 63) / 64, batch, infos, 5);
    return true;
}

bool cactus_vulkan_encode_gemm_f16(void* out, const void* lhs, const void* rhs,
                                   uint32_t M, uint32_t K, uint32_t N, int pretransposed) {
    if (!vk().ok || M == 0 || K == 0 || N == 0) return false;
    if ((size_t)M * N > (size_t)65535 * 64) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    VkDescriptorBufferInfo infos[5];
    if (!dinfo(lhs, (size_t)M * K * 2, &infos[0])) return false;
    infos[1] = infos[0];
    if (!winfo(rhs, (size_t)K * N * 2, &infos[2])) return false;
    if (!dinfo(out, (size_t)M * N * 2, &infos[3])) return false;
    infos[4] = infos[3];
    if (!ensure_cmd()) return false;
    VkDescriptorSet set = make_set(KGMB, infos, 5);
    if (!set) return false;
    struct { uint32_t M, K, N, f32out, f32a, tr; } push = {M, K, N, 0u, 0u, pretransposed ? 1u : 0u};
    dispatch(KGMB, set, &push, ((uint32_t)((size_t)M * N) + 63) / 64, 1, infos, 5);
    return true;
}

bool cactus_vulkan_encode_rel_pos_bias(void* y, const void* q, const void* r,
                                       uint32_t B, uint32_t T, uint32_t H, uint32_t D,
                                       uint32_t R, int r_batched, float scale) {
    if (!vk().ok || B == 0 || T == 0 || H == 0 || H > 65535 || D == 0) return false;
    if ((size_t)T * T > (size_t)65535 * 64) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    VkDescriptorBufferInfo infos[3];
    if (!dinfo(q, (size_t)B * T * H * D * 2, &infos[0])) return false;
    size_t rbytes = (size_t)(r_batched ? B : 1) * R * H * D * 2;
    if (!winfo(r, rbytes, &infos[1])) return false;
    if (!dinfo(y, (size_t)B * H * T * T * 2, &infos[2])) return false;
    if (!ensure_cmd()) return false;
    VkDescriptorSet set = make_set(KRPB, infos, 3);
    if (!set) return false;
    struct { uint32_t T, H, D, rbs; float scale; uint32_t B; } push =
        {T, H, D, r_batched ? R * H * D : 0u, scale, B};
    dispatch(KRPB, set, &push, ((T * T) + 63) / 64, H, infos, 3);
    return true;
}

MemBlock g_ewsteps;

bool cactus_vulkan_encode_elemwise_chain(void* out, const void* in, const float* steps,
                                         uint32_t nsteps, const void* side0, const void* side1,
                                         const void* side2, const size_t* side_elems,
                                         size_t n, uint32_t flags, uint32_t inner) {
    if (!vk().ok || nsteps == 0 || nsteps > 12 || n == 0) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    size_t ein = (flags & 1u) ? 4 : 2, eout = (flags & 2u) ? 4 : 2;
    VkDescriptorBufferInfo infos[11];
    if (!dinfo(in, n * ein, &infos[0])) return false;
    infos[1] = infos[0];
    if (!dinfo(out, n * eout, &infos[2])) return false;
    infos[3] = infos[2];
    const void* sblob = stage_upload(steps, (size_t)nsteps * 16);
    if (!sblob || !dinfo(sblob, (size_t)nsteps * 16, &infos[4])) return false;
    const void* sides[3] = {side0, side1, side2};
    for (int si = 0; si < 3; ++si) {
        int base = 5 + si * 2;
        if (sides[si]) {
            size_t sf32 = 2;
            for (uint32_t s = 0; s < nsteps; ++s) {
                const uint32_t* st = reinterpret_cast<const uint32_t*>(steps + s * 4);
                if ((int)st[0] == 2 && (int)((st[1] >> 6) & 3u) == si && (st[1] & 32u)) sf32 = 4;
            }
            if (!winfo(sides[si], side_elems[si] * sf32, &infos[base])) return false;
            infos[base + 1] = infos[base];
        } else {
            if (!dummy_info(&infos[base])) return false;
            infos[base + 1] = infos[base];
        }
    }
    if (!ensure_cmd()) return false;
    VkDescriptorSet set = make_set(KEWC, infos, 11);
    if (!set) return false;
    struct { uint32_t nsteps, n, flags, inner; } push = {nsteps, (uint32_t)n, flags, inner};
    uint32_t gx = ((uint32_t)n + 63) / 64;
    dispatch(KEWC, set, &push, gx > 65535 ? 65535 : gx, 1, infos, 11);
    return true;
}

bool cactus_vulkan_encode_attention_f16(void* out, const void* q, const void* k, const void* v,
        const void* mask, uint32_t B, uint32_t T, uint32_t S, uint32_t HQ, uint32_t HKV,
        uint32_t D, uint32_t DV, float scale, uint32_t causal, uint32_t pos_off,
        uint32_t window, float logit_cap, uint32_t mask_mode) {
    if (!vk().ok || !vk().pipes[KAF].p) return false;
    if (B == 0 || T == 0 || S == 0 || HQ == 0 || HKV == 0 || (HQ % HKV) != 0) return false;
    if (D > 1024u || DV > 1024u) return false;
    if ((size_t)B * T * HQ > 65535u) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    VkDescriptorBufferInfo infos[5];
    if (!dinfo(q, (size_t)B * T * HQ * D * 2, &infos[0])) return false;
    if (!dinfo(k, (size_t)B * S * HKV * D * 2, &infos[1])) return false;
    if (!dinfo(v, (size_t)B * S * HKV * DV * 2, &infos[2])) return false;
    if (!dinfo(out, (size_t)B * T * HQ * DV * 2, &infos[3])) return false;
    if (mask_mode != 0u && mask) {
        size_t mb = (mask_mode >= 3u) ? (size_t)B * HQ * T * S * 2 : (size_t)B * T * S * 2;
        if (!dinfo(mask, mb, &infos[4])) return false;
    } else {
        if (!dummy_info(&infos[4])) return false;
    }
    if (!ensure_cmd()) return false;
    VkDescriptorSet set = make_set(KAF, infos, 5);
    if (!set) return false;
    struct { uint32_t T, S, HQ, HKV, D, DV; float scale; uint32_t causal, pos_off, window;
             float logit_cap; uint32_t mask_mode; } push =
        {T, S, HQ, HKV, D, DV, scale, causal, pos_off, window, logit_cap, mask_mode};
    dispatch(KAF, set, &push, B * T * HQ, 1, infos, 5);
    return true;
}

bool cactus_vulkan_encode_conv_cache_append(void* out, const void* src, void* ring,
        uint32_t hd, uint32_t ws, uint32_t nnew, uint32_t head0, uint32_t count_new,
        uint32_t num_rows, int src_f32) {
    if (!vk().ok || hd == 0 || ws == 0 || ws + nnew > 65535) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    VkDescriptorBufferInfo infos[4];
    uint32_t ring_off4 = 0;
    if (!dinfo(src, (size_t)num_rows * hd * (src_f32 ? 4 : 2), &infos[0])) return false;
    infos[1] = infos[0];
    if (!dinfo_off(ring, (size_t)ws * hd * 2, &infos[2], &ring_off4)) return false;
    if (!dinfo(out, (size_t)ws * hd * 2, &infos[3])) return false;
    if (!ensure_cmd()) return false;
    VkDescriptorSet set = make_set(KCCA, infos, 4);
    if (!set) return false;
    struct { uint32_t hd, ws, nnew, head0, count_new, num_rows, f32, ring_off; } push =
        {hd, ws, nnew, head0, count_new, num_rows, src_f32 ? 1u : 0u, ring_off4 * 2u};
    dispatch(KCCA, set, &push, (hd + 63) / 64, ws + nnew, infos, 4);
    return true;
}

static bool enc_simple(int kk, const void* a, size_t ab, const void* b, size_t bb,
                       void* y, size_t yb, const void* push, uint32_t gx, uint32_t gy) {
    if (!vk().ok) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    VkDescriptorBufferInfo infos[3];
    int cnt = 0;
    if (!dinfo(a, ab, &infos[cnt++])) return false;
    if (b && !dinfo(b, bb, &infos[cnt++])) return false;
    if (!dinfo(y, yb, &infos[cnt++])) return false;
    if (!ensure_cmd()) return false;
    VkDescriptorSet set = make_set(kk, infos, cnt);
    if (!set) return false;
    dispatch(kk, set, push, gx > 65535 ? 65535 : gx, gy, infos, cnt);
    return true;
}

bool cactus_vulkan_encode_binary_f32(int op, void* y, const void* a, const void* b, size_t n) {
    if (n == 0) return false;
    struct { uint32_t n; int32_t op; } push = {(uint32_t)n, op};
    return enc_simple(KB32, a, n * 4, b, n * 4, y, n * 4, &push, ((uint32_t)n + 63) / 64, 1);
}

bool cactus_vulkan_encode_scalar_f32(int op, void* y, const void* in, size_t n, float p) {
    if (n == 0) return false;
    struct { uint32_t n; int32_t op; float p; } push = {(uint32_t)n, op, p};
    return enc_simple(KSC32, in, n * 4, nullptr, 0, y, n * 4, &push, ((uint32_t)n + 63) / 64, 1);
}

bool cactus_vulkan_encode_unary_f32(int op, void* y, const void* in, size_t n) {
    if (n == 0) return false;
    struct { uint32_t n; int32_t op; } push = {(uint32_t)n, op};
    return enc_simple(KU32, in, n * 4, nullptr, 0, y, n * 4, &push, ((uint32_t)n + 63) / 64, 1);
}

bool cactus_vulkan_encode_clamp(void* out, const void* in, size_t n, float lo, float hi, int f32) {
    if (!vk().ok || n == 0) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    size_t eb = f32 ? 4 : 2;
    VkDescriptorBufferInfo infos[4];
    if (!dinfo(in, n * eb, &infos[0])) return false;
    infos[1] = infos[0];
    if (!dinfo(out, n * eb, &infos[2])) return false;
    infos[3] = infos[2];
    if (!ensure_cmd()) return false;
    VkDescriptorSet set = make_set(KCLMP, infos, 4);
    if (!set) return false;
    struct { uint32_t n; float lo, hi; uint32_t f32; } push = {(uint32_t)n, lo, hi, f32 ? 1u : 0u};
    uint32_t gx = ((uint32_t)n + 63) / 64;
    dispatch(KCLMP, set, &push, gx > 65535 ? 65535 : gx, 1, infos, 4);
    return true;
}

bool cactus_vulkan_encode_reduce_axis(int op, void* out, const void* in, uint32_t outer,
                                      uint32_t axis_size, uint32_t inner, int f32) {
    if (!vk().ok || axis_size == 0 || inner == 0) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    size_t eb = f32 ? 4 : 2;
    uint32_t n = outer * inner;
    if (n == 0) return false;
    VkDescriptorBufferInfo infos[4];
    if (!dinfo(in, (size_t)outer * axis_size * inner * eb, &infos[0])) return false;
    infos[1] = infos[0];
    if (!dinfo(out, (size_t)n * eb, &infos[2])) return false;
    infos[3] = infos[2];
    if (!ensure_cmd()) return false;
    VkDescriptorSet set = make_set(KRAX, infos, 4);
    if (!set) return false;
    struct { uint32_t axis_size, inner, n; int32_t op; uint32_t f32; } push =
        {axis_size, inner, n, op, f32 ? 1u : 0u};
    uint32_t gx = (n + 63) / 64;
    dispatch(KRAX, set, &push, gx > 65535 ? 65535 : gx, 1, infos, 4);
    return true;
}

bool cactus_vulkan_encode_cumsum(void* out, const void* in, uint32_t outer,
                                 uint32_t axis_size, uint32_t inner, int f32) {
    if (!vk().ok || axis_size == 0 || inner == 0) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    size_t eb = f32 ? 4 : 2;
    uint32_t n = outer * inner;
    if (n == 0) return false;
    size_t total_b = (size_t)outer * axis_size * inner * eb;
    VkDescriptorBufferInfo infos[4];
    if (!dinfo(in, total_b, &infos[0])) return false;
    infos[1] = infos[0];
    if (!dinfo(out, total_b, &infos[2])) return false;
    infos[3] = infos[2];
    if (!ensure_cmd()) return false;
    VkDescriptorSet set = make_set(KCSM, infos, 4);
    if (!set) return false;
    struct { uint32_t axis_size, inner, n, f32; } push = {axis_size, inner, n, f32 ? 1u : 0u};
    uint32_t gx = (n + 63) / 64;
    dispatch(KCSM, set, &push, gx > 65535 ? 65535 : gx, 1, infos, 4);
    return true;
}

bool cactus_vulkan_encode_gather_f32idx(void* out, const void* table, const void* idx,
                                        uint32_t rows, uint32_t D, size_t table_bytes) {
    if (!vk().ok || rows == 0 || D == 0) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    uint32_t n = rows * D;
    VkDescriptorBufferInfo infos[3];
    if (!winfo(table, table_bytes, &infos[0])) return false;
    if (!dinfo(idx, (size_t)rows * 4, &infos[1])) return false;
    if (!dinfo(out, (size_t)n * 2, &infos[2])) return false;
    if (!ensure_cmd()) return false;
    VkDescriptorSet set = make_set(KGF32, infos, 3);
    if (!set) return false;
    struct { uint32_t D, n; } push = {D, n};
    uint32_t gx = (n + 63) / 64;
    dispatch(KGF32, set, &push, gx > 65535 ? 65535 : gx, 1, infos, 3);
    return true;
}

bool cactus_vulkan_encode_maxpool1d(void* out, const void* in, uint32_t NC, uint32_t L,
                                    uint32_t Lout, uint32_t K, uint32_t stride) {
    if (!vk().ok || NC == 0 || NC > 65535 || L == 0 || Lout == 0 || K == 0) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    VkDescriptorBufferInfo infos[2];
    if (!dinfo(in, (size_t)NC * L * 2, &infos[0])) return false;
    if (!dinfo(out, (size_t)NC * Lout * 2, &infos[1])) return false;
    if (!ensure_cmd()) return false;
    VkDescriptorSet set = make_set(KMXP, infos, 2);
    if (!set) return false;
    struct { uint32_t L, Lout, K, stride; } push = {L, Lout, K, stride};
    dispatch(KMXP, set, &push, (Lout + 63) / 64, NC, infos, 2);
    return true;
}

bool cactus_vulkan_encode_bilinear(void* out, const void* in, uint32_t sh, uint32_t sw,
                                   uint32_t dh, uint32_t dw, uint32_t E, int align) {
    if (!vk().ok || sh == 0 || sw == 0 || dh == 0 || dw == 0 || E == 0) return false;
    if ((size_t)dh * dw > 65535) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    VkDescriptorBufferInfo infos[2];
    if (!dinfo(in, (size_t)sh * sw * E * 2, &infos[0])) return false;
    if (!dinfo(out, (size_t)dh * dw * E * 2, &infos[1])) return false;
    if (!ensure_cmd()) return false;
    VkDescriptorSet set = make_set(KBIL, infos, 2);
    if (!set) return false;
    struct { uint32_t sh, sw, dh, dw, E, align; } push = {sh, sw, dh, dw, E, align ? 1u : 0u};
    dispatch(KBIL, set, &push, (E + 63) / 64, dh * dw, infos, 2);
    return true;
}

bool cactus_vulkan_encode_groupnorm(void* out, const void* x, const void* w, const void* b,
                                    uint32_t N, uint32_t C, uint32_t S, uint32_t groups, float eps) {
    if (!vk().ok || N == 0 || C == 0 || S == 0 || groups == 0 || C % groups != 0) return false;
    if ((size_t)N * groups > 65535) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    uint32_t cpg = C / groups;
    VkDescriptorBufferInfo infos[4];
    if (!dinfo(x, (size_t)N * C * S * 2, &infos[0])) return false;
    if (!winfo(w, (size_t)C * 2, &infos[1])) return false;
    if (!winfo(b, (size_t)C * 2, &infos[2])) return false;
    if (!dinfo(out, (size_t)N * C * S * 2, &infos[3])) return false;
    if (!ensure_cmd()) return false;
    VkDescriptorSet set = make_set(KGN2, infos, 4);
    if (!set) return false;
    struct { uint32_t cpg, S, C; float eps; uint32_t groups; } push = {cpg, S, C, eps, groups};
    dispatch(KGN2, set, &push, N * groups, 1, infos, 4);
    return true;
}

bool cactus_vulkan_encode_conv2d(void* out, const void* x, const void* w, const void* bias,
                                 uint32_t N, uint32_t Cin, uint32_t H, uint32_t W, uint32_t Cout,
                                 uint32_t Ho, uint32_t Wo, uint32_t K, uint32_t stride,
                                 uint32_t pad, int dw) {
    if (!vk().ok || N == 0 || Cin == 0 || Cout == 0 || Ho == 0 || Wo == 0 || K == 0) return false;
    if ((size_t)N * Cout * Ho > 65535) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    VkDescriptorBufferInfo infos[4];
    if (!dinfo(x, (size_t)N * Cin * H * W * 2, &infos[0])) return false;
    size_t wb = (size_t)(dw ? Cout : Cout * Cin) * K * K * 2;
    if (!winfo(w, wb, &infos[1])) return false;
    if (bias) { if (!winfo(bias, (size_t)Cout * 2, &infos[2])) return false; }
    else if (!dummy_info(&infos[2])) return false;
    if (!dinfo(out, (size_t)N * Cout * Ho * Wo * 2, &infos[3])) return false;
    if (!ensure_cmd()) return false;
    VkDescriptorSet set = make_set(KC2D, infos, 4);
    if (!set) return false;
    struct { uint32_t Cin, H, W, Cout, Ho, Wo, K, stride, pad, dw, hb; } push =
        {Cin, H, W, Cout, Ho, Wo, K, stride, pad, dw ? 1u : 0u, bias ? 1u : 0u};
    dispatch(KC2D, set, &push, (Wo + 63) / 64, N * Cout * Ho, infos, 4);
    return true;
}

MemBlock g_emb_rows;

static bool emb_hadamard_common(void* out, const CactusQuantMatrix* W, const uint32_t* rows, uint32_t M) {
    if (!vk().ok || !W || M == 0 || M > 65535) return false;
    if (W->bits != 4 && W->bits != 2 && W->bits != 3) return false;
    if (W->flags & (CACTUS_QUANT_FLAG_ORTHOGONAL | CACTUS_QUANT_FLAG_INTERLEAVED_4ROW)) return false;
    const uint32_t gs = W->group_size, ng = W->num_groups, K = W->K;
    if (gs > 512 || (gs & (gs - 1)) != 0 || ng == 0 || ng > 65535) return false;
    if (!W->packed_indices || !W->norms || !W->codebook || !W->left_signs
        || !W->right_signs || !W->permutation || !W->input_scale_recip) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    ResW& r = resident(W);
    if (!r.ok) return false;
    auto ri = [&](int i) {
        VkDescriptorBufferInfo d = {};
        d.buffer = r.b.buf; d.offset = r.off[i]; d.range = r.sz[i] ? r.sz[i] : 4;
        return d;
    };
    VkDescriptorBufferInfo infos[9];
    infos[0] = ri(W_PACKED); infos[1] = ri(W_CB); infos[2] = ri(W_NORMS);
    infos[3] = ri(W_RECIP);  infos[4] = ri(W_LS); infos[5] = ri(W_RS); infos[6] = ri(W_PERM);
    const void* rblob = stage_upload(rows, (size_t)M * 4);
    if (!rblob || !dinfo(rblob, (size_t)M * 4, &infos[7])) return false;
    if (!dinfo(out, (size_t)M * K * 2, &infos[8])) return false;
    if (!ensure_cmd()) return false;
    VkDescriptorSet set = make_set(KEHM, infos, 9);
    if (!set) return false;
    struct { uint32_t gs, ng, K, bits, pk_base; } push = {gs, ng, K, W->bits, 0u};
    dispatch(KEHM, set, &push, ng, M, infos, 9);
    return true;
}

bool cactus_vulkan_encode_embedding_hadamard_m(void* out, const CactusQuantMatrix* W,
                                               const uint32_t* rows, uint32_t M) {
    return emb_hadamard_common(out, W, rows, M);
}

bool cactus_vulkan_encode_embedding_hadamard(void* out, uint32_t row, const CactusQuantMatrix* W) {
    return emb_hadamard_common(out, W, &row, 1);
}

static bool emb_ortho_common(void* out, const CactusQuantMatrix* W, const uint32_t* rows,
                             uint32_t M, float oscale) {
    if (!vk().ok || !W || M == 0 || M > 65535 || !W->rotation) return false;
    if (W->bits != 4 && W->bits != 2 && W->bits != 3) return false;
    const uint32_t K = W->K, ng = W->num_groups, gs = W->group_size;
    if (ng != 1 || gs != K || K > 2048) return false;
    const bool il = (W->flags & CACTUS_QUANT_FLAG_INTERLEAVED_4ROW) != 0;
    if (il && W->bits != 4) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    ResW& r = resident(W);
    if (!r.ok || !r.sz[W_ROT]) return false;
    auto ri = [&](int i) {
        VkDescriptorBufferInfo d = {};
        d.buffer = r.b.buf; d.offset = r.off[i]; d.range = r.sz[i] ? r.sz[i] : 4;
        return d;
    };
    VkDescriptorBufferInfo infos[7];
    infos[0] = ri(W_PACKED); infos[1] = ri(W_CB); infos[2] = ri(W_NORMS);
    infos[3] = ri(W_RECIP);  infos[4] = ri(W_ROT);
    const void* rblob = stage_upload(rows, (size_t)M * 4);
    if (!rblob || !dinfo(rblob, (size_t)M * 4, &infos[5])) return false;
    if (!dinfo(out, (size_t)M * K * 2, &infos[6])) return false;
    if (!ensure_cmd()) return false;
    VkDescriptorSet set = make_set(KEOM, infos, 7);
    if (!set) return false;
    struct { uint32_t K, M, bits, il, pk_base; float oscale; } push =
        {K, M, W->bits, il ? 1u : 0u, 0u, oscale};
    dispatch(KEOM, set, &push, (K + 255) / 256, M, infos, 7);
    return true;
}

bool cactus_vulkan_encode_embedding_ortho_m(void* out, const CactusQuantMatrix* W,
                                            const uint32_t* rows, uint32_t M) {
    return emb_ortho_common(out, W, rows, M, 1.0f);
}

bool cactus_vulkan_encode_embedding_ortho(void* out, uint32_t row, const CactusQuantMatrix* W, float scale) {
    return emb_ortho_common(out, W, &row, 1, scale);
}

bool cactus_vulkan_encode_attention_fused_i8(
        void* out, const void* q, const void* kraw, const void* vraw,
        void* kc, void* vc, void* ks, void* vs,
        const void* qw, const void* kw, const void* vw, const void* cs, const void* sn,
        uint32_t nqh, uint32_t hd, uint32_t vhd,
        uint32_t kv_start, uint32_t kv_end, uint32_t slot, uint32_t has_new,
        float eps, float scale,
        size_t kc_bytes, size_t vc_bytes, size_t ks_bytes, size_t vs_bytes) {
    static int afdbg = std::getenv("CACTUS_VK_STATS") ? 8 : 0;
    #define AF_REF(tag) do { if (afdbg > 0) { --afdbg; \
        std::fprintf(stderr, "[vkafref] %s nqh=%u hd=%u vhd=%u kv=[%u,%u) slot=%u new=%u\n", \
                     tag, nqh, hd, vhd, kv_start, kv_end, slot, has_new); } } while (0)
    if (!vk().ok) { AF_REF("pipe"); return false; }
    if (kv_end <= kv_start || nqh == 0 || hd == 0 || vhd == 0) { AF_REF("dims"); return false; }
    if (hd > 512u || vhd > 512u || (hd & 31u) || (vhd & 31u)) { AF_REF("hd"); return false; }
    const int kaf = (hd > 256u || vhd > 256u) ? KAF2 : KAFU;
    if (!vk().pipes[kaf].p) { AF_REF("pipe"); return false; }
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    VkDescriptorBufferInfo infos[15];
    uint32_t okc = 0, ovc = 0, oks = 0, ovs = 0;
    if (!dinfo(q, (size_t)nqh * hd * 2, &infos[0])) { AF_REF("q"); return false; }
    if (has_new && kraw && vraw) {
        if (!dinfo(kraw, (size_t)hd * 2, &infos[1])) { AF_REF("kraw"); return false; }
        if (!dinfo(vraw, (size_t)vhd * 2, &infos[2])) { AF_REF("vraw"); return false; }
    } else {
        if (!dummy_info(&infos[1]) || !dummy_info(&infos[2])) { AF_REF("dummy"); return false; }
    }
    if (!dinfo_off(kc, kc_bytes, &infos[3], &okc)) { AF_REF("kc"); return false; }
    if (!dinfo_off(vc, vc_bytes, &infos[4], &ovc)) { AF_REF("vc"); return false; }
    if (!dinfo_off(ks, ks_bytes, &infos[5], &oks)) { AF_REF("ks"); return false; }
    if (!dinfo_off(vs, vs_bytes, &infos[6], &ovs)) { AF_REF("vs"); return false; }
    if (!dinfo(out, (size_t)nqh * vhd * 2, &infos[7])) { AF_REF("out"); return false; }
    if (!winfo(qw, (size_t)hd * 2, &infos[8])) { AF_REF("qw"); return false; }
    if (has_new && kw) { if (!winfo(kw, (size_t)hd * 2, &infos[9])) return false; }
    else if (!dummy_info(&infos[9])) return false;
    if (has_new && vw) { if (!winfo(vw, (size_t)vhd * 2, &infos[10])) return false; }
    else if (!dummy_info(&infos[10])) return false;
    if (!dinfo(cs, (size_t)hd * 2, &infos[11]) && !winfo(cs, (size_t)hd * 2, &infos[11])) return false;
    if (!dinfo(sn, (size_t)hd * 2, &infos[12]) && !winfo(sn, (size_t)hd * 2, &infos[12])) return false;
    uint32_t R = kv_end - kv_start;
    uint32_t nwg = R / 24u; if (nwg < 1u) nwg = 1u; if (nwg > 32u) nwg = 32u;
    if (nwg > 1u) {
        if (!scratch_info(g_attn_po, (size_t)nqh * nwg * vhd * 4, &infos[13])) return false;
        if (!scratch_info(g_attn_ml, (size_t)nqh * nwg * 8, &infos[14])) return false;
    } else {
        if (!dummy_info(&infos[13]) || !dummy_info(&infos[14])) return false;
    }
    if (!ensure_cmd()) return false;
    VkDescriptorSet set = make_set(kaf, infos, 15);
    if (!set) return false;
    struct { uint32_t nqh, hd, vhd, kv_start, kv_end, nwg, slot, has_new; float eps, scale;
             uint32_t okc, ovc, oks, ovs; } push =
        {nqh, hd, vhd, kv_start, kv_end, nwg, slot, has_new, eps, scale, okc, ovc, oks, ovs};
    dispatch(kaf, set, &push, nqh * nwg, 1, infos, 15);
    if (nwg > 1u) {
        VkDescriptorBufferInfo cinfos[3] = {infos[13], infos[14], infos[7]};
        VkDescriptorSet cset = make_set(KCB, cinfos, 3);
        if (!cset) return false;
        struct { uint32_t vhd, nwg; } cpush = {vhd, nwg};
        dispatch(KCB, cset, &cpush, nqh, 1, cinfos, 3);
    }
    return true;
}

MemBlock g_dn_state;

bool cactus_vulkan_encode_deltanet_decode(void* out, const void* q, const void* k, const void* v,
                                          const void* g, const void* b, const void* s,
                                          uint32_t B, uint32_t Hq, uint32_t Hv,
                                          uint32_t K, uint32_t V, float scale) {
    if (!vk().ok || !vk().pipes[KDND].p) return false;
    if (B == 0 || Hq == 0 || Hv == 0 || K == 0 || K > 512 || V == 0 || V > 1024 || (Hv % Hq) != 0) return false;
    if (Hv > 65535 || B > 65535) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    VkDescriptorBufferInfo infos[7];
    if (!dinfo(q, (size_t)B * Hq * K * 2, &infos[0])) return false;
    if (!dinfo(k, (size_t)B * Hq * K * 2, &infos[1])) return false;
    if (!dinfo(v, (size_t)B * Hv * V * 2, &infos[2])) return false;
    if (!dinfo(g, (size_t)B * Hv * 2, &infos[3])) return false;
    if (!dinfo(b, (size_t)B * Hv * 2, &infos[4])) return false;
    if (!dinfo(s, (size_t)B * K * Hv * V * 2, &infos[5])) return false;
    if (!dinfo(out, (size_t)B * (1 + K) * Hv * V * 2, &infos[6])) return false;
    if (!ensure_cmd()) return false;
    VkDescriptorSet set = make_set(KDND, infos, 7);
    if (!set) return false;
    struct { uint32_t Hq, Hv, K, V; float scale; } push = {Hq, Hv, K, V, scale};
    dispatch(KDND, set, &push, Hv, B, infos, 7);
    return true;
}

bool cactus_vulkan_encode_deltanet_prefill(void* out, const void* q, const void* k, const void* v,
                                           const void* g, const void* b, const void* s,
                                           uint32_t B, uint32_t T, uint32_t Hq, uint32_t Hv,
                                           uint32_t K, uint32_t V, float scale) {
    if (!vk().ok || !vk().pipes[KDNP].p) return false;
    if (B == 0 || T == 0 || Hq == 0 || Hv == 0 || K == 0 || K > 512 || V == 0 || V > 1024 || (Hv % Hq) != 0) return false;
    if (Hv > 65535 || B > 65535) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    VkDescriptorBufferInfo infos[8];
    if (!dinfo(q, (size_t)B * T * Hq * K * 2, &infos[0])) return false;
    if (!dinfo(k, (size_t)B * T * Hq * K * 2, &infos[1])) return false;
    if (!dinfo(v, (size_t)B * T * Hv * V * 2, &infos[2])) return false;
    if (!dinfo(g, (size_t)B * T * Hv * 2, &infos[3])) return false;
    if (!dinfo(b, (size_t)B * T * Hv * 2, &infos[4])) return false;
    if (!dinfo(s, (size_t)B * K * Hv * V * 2, &infos[5])) return false;
    if (!dinfo(out, (size_t)B * (T + K) * Hv * V * 2, &infos[6])) return false;
    if (!scratch_info(g_dn_state, (size_t)B * Hv * K * V * sizeof(float), &infos[7])) return false;
    if (!ensure_cmd()) return false;
    VkDescriptorSet set = make_set(KDNP, infos, 8);
    if (!set) return false;
    struct { uint32_t T, Hq, Hv, K, V; float scale; } push = {T, Hq, Hv, K, V, scale};
    dispatch(KDNP, set, &push, Hv, B, infos, 8);
    return true;
}

struct VkMoeSet {
    MemBlock arena;
    size_t slot = 0;
    size_t offs[7] = {};   // pk, nm, cb, rc, ls, rs, pm (byte offsets within an expert slot)
    uint32_t K = 0, N = 0;
    bool ok = false;
};
struct VkMoeCat { VkMoeSet w1, w3, w2; bool ok = false; };
std::map<const void*, VkMoeCat> g_vk_moe;
MemBlock g_moe_code1, g_moe_code3, g_moe_gu, g_moe_code2;

static bool vk_moe_component_span(const CactusQuantMatrix& W, uintptr_t& lo, uintptr_t& hi, size_t* offs) {
    const uint32_t gs = W.group_size, ng = W.num_groups, bits = W.bits;
    const uint32_t pgb = (gs * bits + 7u) / 8u;
    struct { const void* p; size_t bytes; } sec[7] = {
        {W.packed_indices, (size_t)W.N * ng * pgb},
        {W.norms, (size_t)W.N * ng * 2},
        {W.codebook, (size_t)(1u << bits) * 2},
        {W.input_scale_recip, (size_t)W.K * 2},
        {W.left_signs, gs},
        {W.right_signs, gs},
        {W.permutation, (size_t)gs * 4},
    };
    lo = UINTPTR_MAX; hi = 0;
    for (int i = 0; i < 7; ++i) {
        if (!sec[i].p || sec[i].bytes == 0) return false;
        uintptr_t a = (uintptr_t)sec[i].p;
        lo = std::min(lo, a);
        hi = std::max(hi, a + sec[i].bytes);
    }
    for (int i = 0; i < 7; ++i) offs[i] = (uintptr_t)sec[i].p - lo;
    return hi > lo && hi - lo <= (256u << 20);
}

static bool vk_moe_build_set(VkMoeSet& S, const CactusQuantMatrix* Ws, uint32_t E) {
    if (E == 0) return false;
    if (Ws[0].bits != 4 || Ws[0].group_size != 128) return false;
    if (!(Ws[0].flags & CACTUS_QUANT_FLAG_INTERLEAVED_4ROW)) return false;
    uintptr_t lo0, hi0;
    size_t offs[7];
    if (!vk_moe_component_span(Ws[0], lo0, hi0, offs)) return false;
    size_t slot = hi0 - lo0;
    for (uint32_t e = 1; e < E; ++e) {
        uintptr_t lo, hi; size_t o[7];
        if (!vk_moe_component_span(Ws[e], lo, hi, o)) return false;
        if (hi - lo != slot) return false;
        for (int i = 0; i < 7; ++i) if (o[i] != offs[i]) return false;
        if (Ws[e].K != Ws[0].K || Ws[e].N != Ws[0].N) return false;
    }
    if ((offs[0] & 3u) || (offs[1] & 1u) || (offs[2] & 1u) || (offs[3] & 1u) || (offs[6] & 3u)) return false;
    if (slot & 3u) slot = (slot + 3u) & ~size_t(3);
    if (!create_block(slot * E, S.arena, false)) return false;
    for (uint32_t e = 0; e < E; ++e) {
        uintptr_t lo, hi; size_t o[7];
        vk_moe_component_span(Ws[e], lo, hi, o);
        std::memcpy(S.arena.map + slot * e, (const void*)lo, hi - lo);
    }
    S.slot = slot;
    for (int i = 0; i < 7; ++i) S.offs[i] = offs[i];
    S.K = Ws[0].K; S.N = Ws[0].N;
    S.ok = true;
    return true;
}

bool cactus_vulkan_moe_cq4_ready(const CactusQuantMatrix* w1_0) {
    if (!vk().ok || !w1_0) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    auto it = g_vk_moe.find(w1_0->packed_indices);
    return it != g_vk_moe.end() && it->second.ok;
}

bool cactus_vulkan_moe_cq4_build(const CactusQuantMatrix* w1s, const CactusQuantMatrix* w3s,
                                 const CactusQuantMatrix* w2s, uint32_t E) {
    if (!vk().ok || !vk().pipes[KMOT].p || !vk().pipes[KMOU].p || !vk().pipes[KMOD].p) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    auto it = g_vk_moe.find(w1s[0].packed_indices);
    if (it != g_vk_moe.end()) return it->second.ok;
    VkMoeCat cat;
    cat.ok = vk_moe_build_set(cat.w1, w1s, E)
          && vk_moe_build_set(cat.w3, w3s, E)
          && vk_moe_build_set(cat.w2, w2s, E)
          && cat.w1.K == cat.w3.K && cat.w1.N == cat.w3.N && cat.w2.K >= cat.w1.N;
    auto ins = g_vk_moe.emplace(w1s[0].packed_indices, std::move(cat));
    return ins.first->second.ok;
}

bool cactus_vulkan_encode_moe_gated_cq4(void* out, const void* hidden, const void* probs,
                                        const void* topk, const CactusQuantMatrix* w1_0,
                                        uint32_t E, uint32_t top_k, uint32_t tokens,
                                        uint32_t act, uint32_t normalize, float eps, float scaling) {
    if (!vk().ok || top_k == 0 || top_k > 16 || tokens == 0 || tokens > 65535) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    auto it = g_vk_moe.find(w1_0 ? w1_0->packed_indices : nullptr);
    if (it == g_vk_moe.end() || !it->second.ok) return false;
    VkMoeCat& C = it->second;
    const uint32_t K1 = C.w1.K, N1 = C.w1.N, K2 = C.w2.K, N2 = C.w2.N;
    if (K1 % 128u || K2 % 128u || (N1 & 3u) || (N2 & 1u)) return false;
    const size_t slots = (size_t)tokens * top_k;
    VkDescriptorBufferInfo hb, tb, pb, ob, cd1, cd3, gub, cd2;
    if (!dinfo(hidden, (size_t)tokens * K1 * 2, &hb)) return false;
    if (!dinfo(topk, slots * 4, &tb)) return false;
    if (!dinfo(probs, (size_t)tokens * E * 2, &pb)) return false;
    if (!dinfo(out, (size_t)tokens * N2 * 2, &ob)) return false;
    if (!scratch_info(g_moe_code1, slots * K1 * 2, &cd1)) return false;
    if (!scratch_info(g_moe_code3, slots * K1 * 2, &cd3)) return false;
    if (!scratch_info(g_moe_gu, slots * N1 * 2, &gub)) return false;
    if (!scratch_info(g_moe_code2, slots * K2 * 2, &cd2)) return false;
    auto arena_info = [&](VkMoeSet& S) {
        VkDescriptorBufferInfo d = {};
        d.buffer = S.arena.buf; d.offset = 0; d.range = S.arena.cap;
        return d;
    };
    if (!ensure_cmd()) return false;
    for (int m = 0; m < 2; ++m) {
        VkMoeSet& S = m == 0 ? C.w1 : C.w3;
        VkDescriptorBufferInfo infos[5] = {hb, tb, arena_info(S), arena_info(S), m == 0 ? cd1 : cd3};
        VkDescriptorSet set = make_set(KMOT, infos, 5);
        if (!set) return false;
        struct { uint32_t K, k_valid, xz, xy, tk, estride, o_rc, o_ls, o_rs, o_pm; } push =
            {K1, K1, K1, 0u, top_k, (uint32_t)S.slot,
             (uint32_t)S.offs[3], (uint32_t)S.offs[4], (uint32_t)S.offs[5], (uint32_t)S.offs[6]};
        dispatch(KMOT, set, &push, K1 / 128u, top_k, infos, 5, tokens);
    }
    {
        VkDescriptorBufferInfo infos[8] = {cd1, cd3, tb,
            arena_info(C.w1), arena_info(C.w1), arena_info(C.w3), arena_info(C.w3), gub};
        VkDescriptorSet set = make_set(KMOU, infos, 8);
        if (!set) return false;
        struct { uint32_t K, N, act, tk, es1, es3, o_pk1, o_nm1, o_cb1, o_pk3, o_nm3, o_cb3; } push =
            {K1, N1, act, top_k, (uint32_t)C.w1.slot, (uint32_t)C.w3.slot,
             (uint32_t)C.w1.offs[0], (uint32_t)C.w1.offs[1], (uint32_t)C.w1.offs[2],
             (uint32_t)C.w3.offs[0], (uint32_t)C.w3.offs[1], (uint32_t)C.w3.offs[2]};
        uint32_t nsg = 256u / (vk().sg_size ? vk().sg_size : 16u);
        dispatch(KMOU, set, &push, (N1 + nsg * 4u - 1u) / (nsg * 4u), top_k, infos, 8, tokens);
    }
    {
        VkMoeSet& S = C.w2;
        VkDescriptorBufferInfo infos[5] = {gub, tb, arena_info(S), arena_info(S), cd2};
        VkDescriptorSet set = make_set(KMOT, infos, 5);
        if (!set) return false;
        struct { uint32_t K, k_valid, xz, xy, tk, estride, o_rc, o_ls, o_rs, o_pm; } push =
            {K2, N1, top_k * N1, N1, top_k, (uint32_t)S.slot,
             (uint32_t)S.offs[3], (uint32_t)S.offs[4], (uint32_t)S.offs[5], (uint32_t)S.offs[6]};
        dispatch(KMOT, set, &push, K2 / 128u, top_k, infos, 5, tokens);
    }
    {
        VkMoeSet& S = C.w2;
        VkDescriptorBufferInfo infos[6] = {cd2, tb, pb, arena_info(S), arena_info(S), ob};
        VkDescriptorSet set = make_set(KMOD, infos, 6);
        if (!set) return false;
        struct { uint32_t K, N, tk, normalize; float eps, scaling; uint32_t E, estride, o_pk, o_nm, o_cb; } push =
            {K2, N2, top_k, normalize, eps, scaling, E, (uint32_t)S.slot,
             (uint32_t)S.offs[0], (uint32_t)S.offs[1], (uint32_t)S.offs[2]};
        uint32_t nsg = 256u / (vk().sg_size ? vk().sg_size : 16u);
        dispatch(KMOD, set, &push, (N2 + nsg * 2u - 1u) / (nsg * 2u), 1u, infos, 6, tokens);
    }
    return true;
}

bool cactus_vulkan_transform_gemv_fits(uint32_t K) {
    return vk().ok && vk().gemv2_ok && K <= 8960u;
}

bool cactus_vulkan_encode_transform_gemv(void* out, const void* x, const CactusQuantMatrix* W,
                                         const void* osw) {
    static int dbg = getenv("CACTUS_VK_TGV_DBG") ? 8 : 0;
    if (!vk().ok || !W || (W->flags & CACTUS_QUANT_FLAG_ORTHOGONAL)) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    if (!enc_cq_gemv(out, x, W)) {
        if (dbg > 0) { --dbg; std::fprintf(stderr, "[tgv] gemv refused K=%u N=%u gs=%u fl=%u bits=%u\n",
                                           W->K, W->N, W->group_size, W->flags, W->bits); }
        return false;
    }
    if (!osw) return true;
    const uint32_t N = W->N;
    if (!cactus_vulkan_encode_unary_f16(0, out, out, N)) {
        if (dbg > 0) { --dbg; std::fprintf(stderr, "[tgv] unary refused N=%u\n", N); }
        return false;
    }
    if (cactus_vulkan_encode_binary_f16(3, out, out, osw, N)) return true;
    // osw may live in host memory (CPU-computed PLE row) or at an unbindable
    // offset; stage a per-encode snapshot through the upload ring.
    const void* staged = nullptr;
    {
        std::lock_guard<std::recursive_mutex> lk2(g_mu);
        staged = stage_upload(osw, (size_t)N * 2);
    }
    if (!staged || !cactus_vulkan_encode_binary_f16(3, out, out, staged, N)) {
        if (dbg > 0) { --dbg; std::fprintf(stderr, "[tgv] binary refused after staging N=%u\n", N); }
        return false;
    }
    return true;
}

void* g_swig_p = nullptr;
size_t g_swig_cap = 0;

bool cactus_vulkan_encode_swiglu_transform(void* code, const void* gate, const void* up,
                                           const CactusQuantMatrix* W, float scale) {
    if (!vk().ok || !W || (W->flags & CACTUS_QUANT_FLAG_ORTHOGONAL)) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    const uint32_t K = W->K;
    size_t need = (size_t)K * 2;
    if (g_swig_cap < need) {
        if (g_swig_p) free_shared_i(g_swig_p);
        g_swig_p = alloc_shared_i(need);
        g_swig_cap = g_swig_p ? need : 0;
    }
    if (!g_swig_p) return false;
    if (!cactus_vulkan_encode_swiglu_f16(g_swig_p, gate, up, K, scale)) return false;
    const CactusQuantMatrix* Ws[1] = {W};
    void* codes[1] = {code};
    return cactus_vulkan_encode_transform_batch(g_swig_p, Ws, 1, codes);
}

bool cactus_vulkan_encode_cast(void* out, int out_prec, const void* in, int in_prec, size_t n) {
    if (n == 0 || (n & 1u)) return false;
    int32_t mode;
    size_t ib, ob;
    if (in_prec == 1 && out_prec == 2) { mode = 0; ib = n * 2; ob = n * 4; }
    else if (in_prec == 2 && out_prec == 1) { mode = 1; ib = n * 4; ob = n * 2; }
    else return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    struct { uint32_t n; int32_t mode; } push = {(uint32_t)n, mode};
    return enc_ew(KCA, out, ob, in, ib, nullptr, 0, &push, ew_groups(n / 2));
}

bool cactus_vulkan_encode_strided_copy(void* out, const void* in, const uint32_t* oshape,
        const uint32_t* sstride, uint32_t ndim, uint32_t total, uint32_t base,
        size_t in_bytes, size_t out_bytes) {
    if (ndim == 0 || ndim > 8 || total == 0) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    struct { uint32_t ndim, total, base; uint32_t oshape[8]; uint32_t sstride[8]; } push = {};
    push.ndim = ndim; push.total = total; push.base = base;
    for (uint32_t d = 0; d < ndim; ++d) { push.oshape[d] = oshape[d]; push.sstride[d] = sstride[d]; }
    return enc_ew(KSTC, out, out_bytes, in, in_bytes, nullptr, 0, &push, ew_groups(total));
}

bool cactus_vulkan_encode_strided_scatter(void* out, const void* in, const uint32_t* ishape,
        const uint32_t* ostride, uint32_t ndim, uint32_t total, uint32_t base,
        size_t in_bytes, size_t out_bytes) {
    if (ndim == 0 || ndim > 8 || total == 0) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    struct { uint32_t ndim, total, base; uint32_t oshape[8]; uint32_t sstride[8]; } push = {};
    push.ndim = ndim; push.total = total; push.base = base;
    for (uint32_t d = 0; d < ndim; ++d) { push.oshape[d] = ishape[d]; push.sstride[d] = ostride[d]; }
    return enc_ew(KSTS, out, out_bytes, in, in_bytes, nullptr, 0, &push, ew_groups(total));
}

bool cactus_vulkan_encode_bcast_binary(int op, void* out, const void* a, const void* b,
        const uint32_t* oshape, const uint32_t* astride, const uint32_t* bstride, uint32_t ndim, uint32_t total,
        size_t a_bytes, size_t b_bytes, size_t out_bytes) {
    if (ndim == 0 || ndim > 8 || total == 0) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    struct { int32_t op; uint32_t ndim, total; uint32_t oshape[8]; uint32_t astr[8]; uint32_t bstr[8]; } push = {};
    push.op = op; push.ndim = ndim; push.total = total;
    for (uint32_t d = 0; d < ndim; ++d) { push.oshape[d] = oshape[d]; push.astr[d] = astride[d]; push.bstr[d] = bstride[d]; }
    if (!vk().ok) return false;
    VkDescriptorBufferInfo infos[3];
    if (!winfo(a, a_bytes, &infos[0])) return false;
    if (!winfo(b, b_bytes, &infos[1])) return false;
    if (!dinfo(out, out_bytes, &infos[2])) return false;
    if (!ensure_cmd()) return false;
    VkDescriptorSet set = make_set(KBB, infos, 3);
    if (!set) return false;
    dispatch(KBB, set, &push, ew_groups(total), 1, infos, 3);
    return true;
}

bool cactus_vulkan_encode_concat2(void* out, const void* a, const void* b,
        uint32_t a_outer, uint32_t b_outer, uint32_t a_axis, uint32_t b_axis, uint32_t inner) {
    if (a_outer == 0 || a_outer != b_outer || inner == 0 || a_axis + b_axis == 0) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    uint32_t total = a_outer * (a_axis + b_axis) * inner;
    struct { uint32_t outer, a_axis, b_axis, inner; } push = {a_outer, a_axis, b_axis, inner};
    return enc_ew(KCT, out, (size_t)total * 2, a, (size_t)a_outer * a_axis * inner * 2,
                  b, (size_t)b_outer * b_axis * inner * 2, &push, ew_groups(total));
}

bool cactus_vulkan_encode_rms_norm_add(void* out, const void* in, const void* w, const void* res,
        size_t rows, size_t dim, float eps, float out_scale) {
    if (!vk().ok || rows == 0 || dim == 0 || rows > 65535) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    VkDescriptorBufferInfo infos[4];
    if (!sinfo(in, rows * dim * 2, &infos[0])) return false;
    if (!winfo(w, dim * 2, &infos[1])) return false;
    if (!sinfo(res, rows * dim * 2, &infos[2])) return false;
    if (!dinfo(out, rows * dim * 2, &infos[3])) return false;
    if (!ensure_cmd()) return false;
    VkDescriptorSet set = make_set(KRA, infos, 4);
    if (!set) return false;
    struct { uint32_t dim; float eps, out_scale; } push = {(uint32_t)dim, eps, out_scale};
    dispatch(KRA, set, &push, (uint32_t)rows, 1, infos, 4);
    return true;
}

bool cactus_vulkan_encode_rms_norm_add_rms(void* h_out, void* xn_out, const void* in, const void* w1,
        const void* res, const void* w2, size_t rows, size_t dim, float eps, float out_scale) {
    if (!vk().ok || rows == 0 || dim == 0 || rows > 65535) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    VkDescriptorBufferInfo infos[6];
    if (!dinfo(in, rows * dim * 2, &infos[0])) return false;
    if (!winfo(w1, dim * 2, &infos[1])) return false;
    if (!dinfo(res, rows * dim * 2, &infos[2])) return false;
    if (!dinfo(h_out, rows * dim * 2, &infos[3])) return false;
    if (!winfo(w2, dim * 2, &infos[4])) return false;
    if (!dinfo(xn_out, rows * dim * 2, &infos[5])) return false;
    if (!ensure_cmd()) return false;
    VkDescriptorSet set = make_set(KRR, infos, 6);
    if (!set) return false;
    struct { uint32_t dim; float eps, out_scale; } push = {(uint32_t)dim, eps, out_scale};
    dispatch(KRR, set, &push, (uint32_t)rows, 1, infos, 6);
    return true;
}

bool cactus_vulkan_encode_rms_norm_scale(void* out, const void* in, const void* w,
        size_t rows, size_t dim, float eps, float oscale) {
    if (!vk().ok || rows == 0 || dim == 0 || rows > 65535) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    VkDescriptorBufferInfo infos[3];
    if (!dinfo(in, rows * dim * 2, &infos[0])) return false;
    if (!winfo(w, dim * 2, &infos[1])) return false;
    if (!dinfo(out, rows * dim * 2, &infos[2])) return false;
    if (!ensure_cmd()) return false;
    VkDescriptorSet set = make_set(KRL, infos, 3);
    if (!set) return false;
    struct { uint32_t dim; float eps, oscale; } push = {(uint32_t)dim, eps, oscale};
    dispatch(KRL, set, &push, (uint32_t)rows, 1, infos, 3);
    return true;
}

bool cactus_vulkan_encode_rope_full(void* out, const void* in, uint32_t tokens, uint32_t S,
        uint32_t H, uint32_t D, uint32_t rot, uint32_t pos0, float theta, int gptj) {
    if (!vk().ok || tokens == 0 || tokens > 65535 || D == 0 || rot < 2 || H == 0 || S == 0) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    VkDescriptorBufferInfo infos[2];
    if (!dinfo(in, (size_t)tokens * D * 2, &infos[0])) return false;
    if (!dinfo(out, (size_t)tokens * D * 2, &infos[1])) return false;
    if (!ensure_cmd()) return false;
    VkDescriptorSet set = make_set(KRF, infos, 2);
    if (!set) return false;
    struct { uint32_t S, H, D, rot, pos0, gptj; float theta; } push =
        {S, H, D, rot, pos0, gptj ? 1u : 0u, theta};
    uint32_t span = rot / 2 + (D - rot);
    dispatch(KRF, set, &push, (span + 63) / 64, tokens, infos, 2);
    return true;
}

bool cactus_vulkan_encode_softcap(void* out, const void* in, size_t n, float cap) {
    if (n == 0 || cap == 0.0f) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    struct { uint32_t n; float cap; } push = {(uint32_t)n, cap};
    return enc_ew(KSO, out, n * 2, in, n * 2, nullptr, 0, &push, ew_groups(n));
}

bool cactus_vulkan_encode_adjust_logits(void* logits, size_t vocab, const uint32_t* recent,
        uint32_t n_recent, int64_t suppressed, float penalty) {
    if (!vk().ok || vocab == 0) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    VkDescriptorBufferInfo infos[2];
    if (!dinfo(logits, vocab * 2, &infos[0])) return false;
    if (recent && n_recent) {
        const void* rec = stage_upload(recent, (size_t)n_recent * 4);
        if (!rec || !dinfo(rec, (size_t)n_recent * 4, &infos[1])) return false;
    } else {
        if (!scratch_info(g_recent, 4, &infos[1])) return false;
    }
    if (!ensure_cmd()) return false;
    VkDescriptorSet set = make_set(KAJ, infos, 2);
    if (!set) return false;
    struct { uint32_t n_recent, sflag, sid, vocab; float penalty; } push =
        {n_recent, suppressed >= 0 ? 1u : 0u, suppressed >= 0 ? (uint32_t)suppressed : 0u,
         (uint32_t)vocab, penalty};
    dispatch(KAJ, set, &push, 1, 1, infos, 2);
    return true;
}

bool cactus_vulkan_encode_argmax(const void* logits, uint32_t vocab, void* out3, const void* bias) {
    if (!vk().ok || vocab == 0) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    VkDescriptorBufferInfo infos[3];
    if (!dinfo(logits, (size_t)vocab * 2, &infos[0])) return false;
    uint32_t has_bias = 0;
    if (bias) {
        if (!dinfo(bias, (size_t)vocab * 4, &infos[1])) return false;
        has_bias = 1;
    } else if (!dummy_info(&infos[1])) {
        return false;
    }
    if (!dinfo(out3, 12, &infos[2])) return false;
    if (!ensure_cmd()) return false;
    VkDescriptorSet set = make_set(KM3, infos, 3);
    if (!set) return false;
    struct { uint32_t V, has_bias; } push = {vocab, has_bias};
    dispatch(KM3, set, &push, 1, 1, infos, 3);
    return true;
}

bool cactus_vulkan_encode_kv_append_i8(const void* src, void* int8base, void* scalebase,
        uint32_t kv_heads, uint32_t hdim, uint32_t current_len, uint32_t group_size, uint32_t M,
        uint32_t sink, uint32_t W, size_t src_bytes, size_t int8_bytes, size_t scale_bytes) {
    if (!vk().ok || M == 0 || kv_heads == 0 || group_size == 0 || (group_size & 3u)
        || hdim == 0 || (hdim % group_size)) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    VkDescriptorBufferInfo infos[3];
    uint32_t o_i8 = 0, o_sc = 0;
    if (!dinfo(src, src_bytes, &infos[0])) return false;
    if (!dinfo_off(int8base, int8_bytes, &infos[1], &o_i8)) return false;
    if (!dinfo_off(scalebase, scale_bytes, &infos[2], &o_sc)) return false;
    if (!ensure_cmd()) return false;
    VkDescriptorSet set = make_set(KKV, infos, 3);
    if (!set) return false;
    struct { uint32_t kvh, hdim, cur, gs, M, sink, W, o_i8, o_sc; } push =
        {kv_heads, hdim, current_len, group_size, M, sink, W, o_i8, o_sc};
    uint32_t work = M * kv_heads * (hdim / group_size);
    dispatch(KKV, set, &push, (work + 63) / 64, 1, infos, 3);
    return true;
}

bool cactus_vulkan_encode_attention_i8(
        void* out, const void* q, const void* knew, const void* vnew,
        const void* kc, const void* vc, const void* ks, const void* vs,
        uint32_t num_q_heads, uint32_t num_kv_heads, uint32_t head_dim, uint32_t v_hdim,
        uint32_t history_len, uint32_t total_keys, uint32_t kv_start, uint32_t kv_end,
        float scale, size_t kc_bytes, size_t vc_bytes, size_t ks_bytes, size_t vs_bytes) {
    if (!vk().ok || !vk().attn_ok || !vk().pipes[KAT].p) return false;
    if (kv_end <= kv_start || num_kv_heads == 0 || (num_q_heads % num_kv_heads) != 0) return false;
    if (head_dim > 512u || v_hdim > 512u || (head_dim & 31u) || (v_hdim & 31u)) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    VkDescriptorBufferInfo infos[10];
    uint32_t okc = 0, ovc = 0, oks = 0, ovs = 0;
    if (!dinfo(q, (size_t)num_q_heads * head_dim * 2, &infos[0])) return false;
    if (total_keys > history_len && knew && vnew) {
        if (!dinfo(knew, (size_t)(total_keys - history_len) * num_kv_heads * head_dim * 2, &infos[1])) return false;
        if (!dinfo(vnew, (size_t)(total_keys - history_len) * num_kv_heads * v_hdim * 2, &infos[2])) return false;
    } else {
        if (!dummy_info(&infos[1]) || !dummy_info(&infos[2])) return false;
    }
    if (!dinfo_off(kc, kc_bytes, &infos[3], &okc)) return false;
    if (!dinfo_off(vc, vc_bytes, &infos[4], &ovc)) return false;
    if (!dinfo_off(ks, ks_bytes, &infos[5], &oks)) return false;
    if (!dinfo_off(vs, vs_bytes, &infos[6], &ovs)) return false;
    if (!dinfo(out, (size_t)num_q_heads * v_hdim * 2, &infos[7])) return false;
    static const int nwg_env = [] {
        const char* v = getenv("CACTUS_VK_ATTN_NWG");
        return v ? std::atoi(v) : 0;
    }();
    uint32_t R = kv_end - kv_start;
    uint32_t nwg = nwg_env > 0 ? (uint32_t)nwg_env : R / 24u;
    if (nwg < 1u) nwg = 1u;
    if (nwg > 32u) nwg = 32u;
    if (nwg > 1u) {
        if (!scratch_info(g_attn_po, (size_t)num_q_heads * nwg * v_hdim * 4, &infos[8])) return false;
        if (!scratch_info(g_attn_ml, (size_t)num_q_heads * nwg * 8, &infos[9])) return false;
    } else {
        if (!dummy_info(&infos[8]) || !dummy_info(&infos[9])) return false;
    }
    if (!ensure_cmd()) return false;
    VkDescriptorSet set = make_set(KAT, infos, 10);
    if (!set) return false;
    struct { uint32_t nqh, nkvh, hd, vhd, hist, kv_start, kv_end, nwg; float scale;
             uint32_t okc, ovc, oks, ovs; } push =
        {num_q_heads, num_kv_heads, head_dim, v_hdim, history_len, kv_start, kv_end, nwg, scale,
         okc, ovc, oks, ovs};
    dispatch(KAT, set, &push, num_q_heads * nwg, 1, infos, 10);
    if (nwg > 1u) {
        VkDescriptorBufferInfo cinfos[3] = {infos[8], infos[9], infos[7]};
        VkDescriptorSet cset = make_set(KCB, cinfos, 3);
        if (!cset) return false;
        struct { uint32_t vhd, nwg; } cpush = {v_hdim, nwg};
        dispatch(KCB, cset, &cpush, num_q_heads, 1, cinfos, 3);
    }
    return true;
}

bool cactus_vulkan_argmax_f16(const __fp16* logits, uint32_t n, uint32_t* idx, float* best) {
    if (!vk().ok || n == 0) return false;
    std::lock_guard<std::recursive_mutex> lk(g_mu);
    const uint32_t G = 64;
    HostIn il(logits, (size_t)n * 2), ob(nullptr, G * 2 * sizeof(float));
    if (!il.p || !ob.p) return false;
    VkDescriptorBufferInfo infos[2];
    if (!dinfo(il.p, (size_t)n * 2, &infos[0]) || !dinfo(ob.p, G * 2 * sizeof(float), &infos[1])) return false;
    if (!ensure_cmd()) return false;
    VkDescriptorSet set = make_set(KA, infos, 2);
    if (!set) return false;
    struct { uint32_t n; } push = {n};
    dispatch(KA, set, &push, G, 1, infos, 2);
    sync_i();
    if (!vk().ok) return false;
    const float* parts = (const float*)ob.p;
    float b = parts[0];
    uint32_t bi;
    std::memcpy(&bi, &parts[1], sizeof(bi));
    for (uint32_t i = 1; i < G; ++i) {
        uint32_t ci;
        std::memcpy(&ci, &parts[2 * i + 1], sizeof(ci));
        if (parts[2 * i] > b || (parts[2 * i] == b && ci < bi)) { b = parts[2 * i]; bi = ci; }
    }
    *best = b;
    *idx = bi;
    return true;
}

#endif
