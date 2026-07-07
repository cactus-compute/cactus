

#include "metal_backend.h"

#if CACTUS_HAS_METAL

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <TargetConditionals.h>
#include <mach/vm_map.h>
#if TARGET_OS_OSX
#include <mach/mach_vm.h>
#endif
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <vector>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <utility>
#include <cstdint>
#include <mutex>
#include <atomic>

#include "cactus_kernels_msl.h"
#include "threading.h"
#include <sys/mman.h>

namespace {

struct MetalCtx {
    id<MTLDevice> dev = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> psoT = nil;
    id<MTLComputePipelineState> psoTsimd = nil;
    id<MTLComputePipelineState> psoTbatch = nil;
    id<MTLComputePipelineState> psoMega = nil, psoSwiT = nil, psoCat = nil;
    id<MTLComputePipelineState> psoG = nil, psoGmr = nil, psoGmrLow = nil;
    id<MTLComputePipelineState> psoActQ = nil, psoG2i8 = nil;
    id<MTLComputePipelineState> psoTm = nil;
    id<MTLComputePipelineState> psoGmma = nil, psoGdense = nil;
    id<MTLComputePipelineState> psoRotW = nil, psoEmbOW = nil, psoRmsAddRms = nil;
    id<MTLComputePipelineState> psoEmbH = nil;
    id<MTLComputePipelineState> psoEmbOm = nil, psoEmbHm = nil;
    id<MTLComputePipelineState> psoCopy=nil, psoBinary=nil, psoScalar=nil, psoUnary=nil, psoRms=nil, psoSwiglu=nil, psoRmsAdd=nil, psoRmsAddScale=nil;
    id<MTLComputePipelineState> psoCF16F32=nil, psoCF32F16=nil, psoCI8F16=nil, psoCF16I8=nil;
    id<MTLComputePipelineState> psoAttn=nil, psoAttnC=nil, psoAttnF=nil, psoStrided=nil, psoScatter=nil, psoBcast=nil, psoKvAppend=nil;
    id<MTLComputePipelineState> psoAttnPre=nil, psoAttnPreMma2=nil, psoAttnPreHd256=nil;
    id<MTLComputePipelineState> psoKvAppendM=nil, psoKvAppendRingM=nil;
    id<MTLComputePipelineState> psoSlideS=nil, psoSlideR=nil, psoSlideRM=nil;
    id<MTLComputePipelineState> psoArgmax=nil;
    id<MTLComputePipelineState> psoArgmaxP=nil, psoArgmaxF=nil, psoSoftcap=nil, psoAdjust=nil;
    id<MTLComputePipelineState> psoGather=nil;
    id<MTLComputePipelineState> psoBinF32=nil, psoScaF32=nil, psoUnaF32=nil;
    id<MTLComputePipelineState> psoClampF16=nil, psoClampF32=nil, psoGlu=nil;
    id<MTLComputePipelineState> psoLayerNorm=nil, psoSoftmaxR=nil, psoConv1dK3=nil, psoAttnDense=nil, psoAttnD64=nil;
    id<MTLComputePipelineState> psoGemmBatch=nil, psoGemmBatchF32A=nil, psoConvDw=nil, psoStridedRows=nil, psoMaskFill=nil;
    id<MTLComputePipelineState> psoReduceF16=nil, psoReduceF32=nil, psoCumsumF16=nil, psoCumsumF32=nil;
    id<MTLComputePipelineState> psoConcat2=nil, psoGatherF32Idx=nil, psoRopeFull=nil, psoMaxpool1d=nil;
    id<MTLComputePipelineState> psoBilinear=nil, psoConv1dGen=nil, psoConv1dNlcDw=nil, psoConv2d=nil;
    id<MTLComputePipelineState> psoBatchnorm=nil, psoGroupnorm=nil, psoBiasAddRows=nil, psoEwChain=nil;
    id<MTLComputePipelineState> psoAttnFlash=nil, psoRmsSimd=nil, psoScatterRows=nil;
    id<MTLComputePipelineState> psoTranspose2d=nil, psoBcastRows=nil, psoRmsAddSimd=nil;
    id<MTLComputePipelineState> psoConvCacheAppend=nil, psoRelPosBias=nil, psoGemvBias=nil;
    id<MTLComputePipelineState> psoTopkRows=nil, psoMoeT=nil, psoMoeUp=nil;
    id<MTLComputePipelineState> psoMoeT2=nil, psoMoeDownAcc=nil, psoRopePair=nil;
    id<MTLComputePipelineState> psoRmsScale=nil, psoSoftmaxTopk=nil, psoRopePairRms=nil;
    id<MTLComputePipelineState> psoRms2AddClip=nil, psoDeltanet=nil, psoDeltanetPre=nil;
    id<MTLBuffer> dummy=nil;
    bool ok = false;

    MetalCtx() { @autoreleasepool {
        dev = MTLCreateSystemDefaultDevice();
        if (!dev) return;
        queue = [dev newCommandQueue];
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kCactusMSL]
                                               options:nil error:&err];
        if (!lib) { if (err) fprintf(stderr,"[cactus-metal] MSL compile failed: %s\n",[[err localizedDescription] UTF8String]); return; }
        auto pso = [&](const char* name) -> id<MTLComputePipelineState> {
            NSError* e=nil;
            id<MTLFunction> f=[lib newFunctionWithName:[NSString stringWithUTF8String:name]];
            id<MTLComputePipelineState> p = f ? [dev newComputePipelineStateWithFunction:f error:&e] : nil;
            if (!p) fprintf(stderr,"[cactus-metal] pipeline '%s' failed: %s\n", name, e?[[e localizedDescription] UTF8String]:"function not found");
            return p;
        };
        psoT=pso("cq4_transform"); psoTsimd=pso("cq4_transform_simd"); psoG=pso("cq4_gemv"); psoTbatch=pso("cq4_transform_batch"); psoGmr=pso("cq4_gemv_mr");
        psoMega=pso("cq4_transform_gemv"); psoSwiT=pso("cq4_swiglu_transform"); psoCat=pso("cq4_gemv_cat");
        psoTm=pso("cq4_transform_m"); psoGmma=pso("cq4_gemm_mma");
        psoGdense=pso("cq4_gemm_dense_f16");
        psoGmrLow=pso("cq_gemv_mr_lowbit");
        psoActQ=pso("cq_act_quant_i8"); psoG2i8=pso("cq2_gemv_i8");
        psoRotW=pso("lmhead_rotate_wide"); psoEmbOW=pso("emb_ortho_wide"); psoRmsAddRms=pso("rms_norm_add_rms_f16");
        psoEmbH=pso("emb_hadamard");
        psoEmbOm=pso("emb_ortho_m"); psoEmbHm=pso("emb_hadamard_m");
        psoGather=pso("gather_f16");
        psoCopy=pso("copy_bytes"); psoBinary=pso("binary_f16"); psoScalar=pso("scalar_f16");
        psoUnary=pso("unary_f16"); psoRms=pso("rms_norm_f16"); psoSwiglu=pso("swiglu_f16"); psoRmsAdd=pso("rms_norm_add_f16"); psoRmsAddScale=pso("rms_norm_add_scale_f16");
        psoCF16F32=pso("cast_f16_f32"); psoCF32F16=pso("cast_f32_f16");
        psoCI8F16=pso("cast_i8_f16"); psoCF16I8=pso("cast_f16_i8");
        psoAttn=pso("attn_decode_i8"); psoAttnC=pso("attn_decode_combine"); psoAttnF=pso("attn_decode_fused_i8");
        psoStrided=pso("strided_copy_f16"); psoBcast=pso("bcast_binary_f16");
        psoAttnPre=pso("attn_prefill_i8"); psoAttnPreMma2=pso("attn_prefill_mma2"); psoAttnPreHd256=pso("attn_prefill_mma_hd256"); psoKvAppendM=pso("kv_append_i8_m");
        psoKvAppendRingM=pso("kv_append_ring_i8_m");
        psoSlideS=pso("kv_slide_save"); psoSlideR=pso("kv_slide_restore"); psoSlideRM=pso("kv_slide_restore_m");
        psoScatter=pso("strided_scatter_f16"); psoKvAppend=pso("kv_append_i8");
        psoArgmax=pso("argmax_logits");
        psoArgmaxP=pso("argmax_part"); psoArgmaxF=pso("argmax_final"); psoSoftcap=pso("softcap_f16");
        psoAdjust=pso("adjust_logits");
        psoBinF32=pso("binary_f32"); psoScaF32=pso("scalar_f32"); psoUnaF32=pso("unary_f32");
        psoClampF16=pso("clamp_f16"); psoClampF32=pso("clamp_f32"); psoGlu=pso("glu_f16");
        psoLayerNorm=pso("layer_norm_f16"); psoSoftmaxR=pso("softmax_rows_f16");
        psoConv1dK3=pso("conv1d_k3_f16"); psoAttnDense=pso("attn_f16"); psoAttnD64=pso("attn_f16_d64");
        psoGemmBatch=pso("gemm_batch_f16"); psoGemmBatchF32A=pso("gemm_batch_f32a"); psoConvDw=pso("conv1d_dw_f16");
        psoStridedRows=pso("strided_copy_rows_f16"); psoMaskFill=pso("attn_maskfill_f16");
        psoReduceF16=pso("reduce_axis_f16"); psoReduceF32=pso("reduce_axis_f32");
        psoCumsumF16=pso("cumsum_f16"); psoCumsumF32=pso("cumsum_f32");
        psoConcat2=pso("concat2_f16"); psoGatherF32Idx=pso("gather_f32idx_f16");
        psoRopeFull=pso("rope_full_f16"); psoMaxpool1d=pso("maxpool1d_f16");
        psoBilinear=pso("bilinear_f16"); psoConv1dGen=pso("conv1d_gen_f16");
        psoConv1dNlcDw=pso("conv1d_nlc_dw_f16"); psoConv2d=pso("conv2d_f16");
        psoBatchnorm=pso("batchnorm_f16"); psoGroupnorm=pso("groupnorm_f16");
        psoBiasAddRows=pso("bias_add_rows_f16"); psoEwChain=pso("elemwise_chain_f16");
        psoAttnFlash=pso("attn_flash_f16"); psoRmsSimd=pso("rms_norm_simd_f16");
        psoScatterRows=pso("strided_scatter_rows_f16"); psoTranspose2d=pso("transpose2d_f16");
        psoBcastRows=pso("bcast_binary_rows_f16"); psoRmsAddSimd=pso("rms_norm_add_simd_f16");
        psoConvCacheAppend=pso("conv_cache_append_f16"); psoRelPosBias=pso("rel_pos_bias_f16");
        psoGemvBias=pso("gemv_bias_f16");
        psoTopkRows=pso("topk_rows_f16"); psoMoeT=pso("cq4_moe_transform");
        psoMoeUp=pso("cq4_moe_gemv_up"); psoMoeT2=pso("cq4_moe_transform2");
        psoMoeDownAcc=pso("cq4_moe_gemv_down_acc"); psoRopePair=pso("rope_pair_f16");
        psoRmsScale=pso("rms_norm_scale_f16"); psoSoftmaxTopk=pso("softmax_topk_f16");
        psoRopePairRms=pso("rope_pair_rms_f16"); psoRms2AddClip=pso("rms2_add_clip_f16");
        psoDeltanet=pso("gated_deltanet_decode_f16"); psoDeltanetPre=pso("gated_deltanet_prefill_f16");
        dummy=[dev newBufferWithLength:16 options:MTLResourceStorageModeShared];
        ok = psoT&&psoG&&psoTm&&psoGmma&&psoRotW&&psoEmbH&&psoEmbOm&&psoEmbHm&&psoCopy&&psoBinary&&psoScalar&&psoUnary&&psoRms&&psoSwiglu&&psoRmsAdd&&psoCF16F32&&psoCF32F16&&psoCI8F16&&psoCF16I8
             &&psoAttn&&psoAttnC&&psoAttnPre&&psoAttnPreMma2&&psoKvAppendM&&psoKvAppendRingM&&psoSlideS&&psoSlideR&&psoSlideRM&&psoStrided&&psoBcast&&psoScatter&&psoKvAppend&&psoArgmax&&psoGather;
    }}
};

MetalCtx& ctx() { static MetalCtx c; return c; }

id<MTLBuffer> owned_shared(size_t len) {
    size_t alen = ((len ? len : 1) + 16383) & ~(size_t)16383;
    void* p = mmap(nullptr, alen, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
    if (p == MAP_FAILED)
        return [ctx().dev newBufferWithLength:alen options:MTLResourceStorageModeShared];
    id<MTLBuffer> b = [ctx().dev newBufferWithBytesNoCopy:p length:alen
                       options:MTLResourceStorageModeShared
                       deallocator:^(void* ptr, NSUInteger l){ munmap(ptr, l); }];
    if (!b) {
        munmap(p, alen);
        return [ctx().dev newBufferWithLength:alen options:MTLResourceStorageModeShared];
    }
    return b;
}

inline id<MTLBuffer> buf(const void* p, size_t bytes) {
    if (!p || bytes == 0) return nil;
    id<MTLBuffer> b = owned_shared(bytes);
    if (b) std::memcpy([b contents], p, bytes);
    return b;
}

struct ResW {
    id<MTLBuffer> packed=nil, norms=nil, codebook=nil, lsign=nil, rsign=nil, perm=nil, recip=nil;
    id<MTLBuffer> rotation=nil;
    id<MTLBuffer> cb_i8=nil;
    float cb_scale=1.f;
    size_t packed_off=0, norms_off=0;
    size_t cb_off=0, ls_off=0, rs_off=0, pm_off=0, rc_off=0, rot_off=0;
    uint32_t il=0;
    bool ok=false;
};
std::unordered_map<uint64_t, ResW> g_resident;

static uint64_t resident_key(const CactusQuantMatrix* W) {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](uint64_t v) { h ^= v; h *= 1099511628211ull; };
    mix(W->bits); mix(W->K); mix(W->N); mix(W->group_size); mix(W->num_groups); mix(W->flags);
    const uint8_t* p = (const uint8_t*)W->packed_indices;
    size_t pkb = (size_t)W->N * W->num_groups * ((W->group_size * W->bits + 7u) / 8u);
    if (!p || pkb == 0) return h;
    size_t take = pkb < 64 ? pkb : 64;
    for (size_t i = 0; i < take; ++i) mix(p[i]);
    for (size_t i = pkb - take; i < pkb; ++i) mix(p[i]);
    return h;
}
id<MTLBuffer> g_attn_scores = nil;
std::unordered_map<const void*, ResW> g_resident_emb;
std::mutex g_resident_mu;

struct MoESet4 {
    id<MTLBuffer> pk=nil, nm=nil, rc=nil, ls=nil, rs=nil, pm=nil, cb=nil;
    size_t pk_off=0, nm_off=0, rc_off=0, ls_off=0, rs_off=0, pm_off=0, cb_off=0;
    uint32_t K=0, N=0;
    uint32_t stride=0;
};
struct MoECat4 { MoESet4 w1, w3, w2; bool ok=false; };
std::unordered_map<uint64_t, MoECat4> g_moe_cat;

std::pair<id<MTLBuffer>, size_t> wrapHostPtr(const void* p, size_t bytes);

bool bind_component(const void* p, size_t bytes, size_t align,
                    id<MTLBuffer> __strong& b, size_t& off) {
    if (!p || bytes == 0) return true;
    if (((uintptr_t)p & (align-1)) != 0) return false;
    auto w = wrapHostPtr(p, bytes);
    if (!w.first) return false;
    b = w.first;
    off = w.second;
    return true;
}

ResW& resident(const CactusQuantMatrix* W) {
    std::lock_guard<std::mutex> lk(g_resident_mu);
    uint64_t key = resident_key(W);
    auto it = g_resident.find(key);
    if (it != g_resident.end()) return it->second;
    const uint32_t K=W->K, N=W->N, ng=W->num_groups, gs=W->group_size, bits=W->bits;
    const uint32_t pgb=(gs*bits+7u)/8u;
    ResW r;
    r.ok = bind_component(W->codebook, (size_t)(1u<<bits)*sizeof(__fp16), 2, r.codebook, r.cb_off)
        && bind_component(W->left_signs, gs, 1, r.lsign, r.ls_off)
        && bind_component(W->right_signs, gs, 1, r.rsign, r.rs_off)
        && bind_component(W->permutation, (size_t)gs*sizeof(uint32_t), 4, r.perm, r.pm_off)
        && bind_component(W->input_scale_recip, (size_t)K*sizeof(__fp16), 2, r.recip, r.rc_off)
        && (!W->rotation || bind_component(W->rotation, (size_t)K*K*sizeof(__fp16), 8, r.rotation, r.rot_off));
    if (bits == 2u) {
        int8_t cbq[4];
        float max_abs = 0.f;
        for (uint32_t i = 0; i < 4u; i++) {
            float v = std::fabs((float)W->codebook[i]);
            if (v > max_abs) max_abs = v;
        }
        float scale = max_abs / 127.f;
        if (scale < 1e-10f) scale = 1e-10f;
        float inv = 1.f / scale;
        for (uint32_t i = 0; i < 4u; i++)
            cbq[i] = (int8_t)std::round((float)W->codebook[i] * inv);
        r.cb_i8 = buf(cbq, 4);
        r.cb_scale = scale;
    }
    size_t pkb=(size_t)N*ng*pgb, nmb=(size_t)N*ng*sizeof(__fp16);
    const bool interleaved = (W->flags & CACTUS_QUANT_FLAG_INTERLEAVED_4ROW) != 0;
    if (((uintptr_t)W->packed_indices & 7u) != 0 || ((uintptr_t)W->norms & 1u) != 0) {
        r.ok = false;
    } else if (r.ok) {
        auto wp = wrapHostPtr(W->packed_indices, pkb);
        auto wn = wrapHostPtr(W->norms, nmb);
        if (wp.first && wn.first) {
            r.packed = wp.first; r.packed_off = wp.second;
            r.norms  = wn.first; r.norms_off  = wn.second;
            r.il = interleaved ? 1 : 0;
        } else {
            r.ok = false;
        }
    }
    return g_resident.emplace(key, r).first->second;
}

id<MTLCommandBuffer> g_cmd = nil;
id<MTLComputeCommandEncoder> g_enc = nil;
std::unordered_map<size_t, std::vector<id<MTLBuffer>>> g_free;
std::vector<id<MTLBuffer>> g_pending;
std::map<uintptr_t, id<MTLBuffer>> g_shared;
bool g_active = false;
id<MTLBuffer> g_code_buf = nil;
id<MTLBuffer> g_code_buf_m = nil;

inline size_t bucket(size_t b) { return (b + 16383) & ~size_t(16383); }

static const uint32_t g_mr_rows = 8u * 2u;

inline bool tg_mem_ok(size_t bytes) {
    return bytes <= (size_t)[ctx().dev maxThreadgroupMemoryLength];
}

id<MTLComputeCommandEncoder> ensureEncoder() {
    if (!g_enc) {
        if (!g_cmd) g_cmd = [ctx().queue commandBuffer];
        g_enc = [g_cmd computeCommandEncoder];
    }
    return g_enc;
}

id<MTLBuffer> recycled(size_t bytes) {
    size_t bk = bucket(bytes ? bytes : 1);
    auto it = g_free.find(bk);
    id<MTLBuffer> b;
    if (it != g_free.end() && !it->second.empty()) { b = it->second.back(); it->second.pop_back(); }
    else b = owned_shared(bk);
    g_pending.push_back(b);
    return b;
}

std::map<uintptr_t, id<MTLBuffer>> g_wrapped;
std::unordered_map<const void*, id<MTLBuffer>> g_readonly;
std::map<std::tuple<void*,size_t,size_t,size_t>, MPSMatrix*> g_mps_mats;
std::pair<id<MTLBuffer>, size_t> wrapHostPtr(const void* p, size_t bytes) {
    uintptr_t a = (uintptr_t)p, base = a & ~(uintptr_t)16383u;
    size_t need = (a - base) + (bytes ? bytes : 1);
    auto it = g_wrapped.upper_bound(a);
    if (it != g_wrapped.begin()) {
        --it;
        if (a >= it->first && a + (bytes?bytes:1) <= it->first + (size_t)it->second.length)
            return { it->second, (size_t)(a - it->first) };
    }
    size_t wraplen = (need + 16383u) & ~(size_t)16383u;
    uintptr_t lo = base, hi = base + wraplen;
#if TARGET_OS_OSX
    // Expand the wrap to the whole VM region so neighboring host pointers hit
    // the same MTLBuffer. mach_vm.h is macOS-only; elsewhere the page-aligned
    // span above is used as-is (correct, just more wrap entries).
    {
        mach_vm_address_t raddr = a;
        mach_vm_size_t rsize = 0;
        vm_region_basic_info_data_64_t info;
        mach_msg_type_number_t cnt = VM_REGION_BASIC_INFO_COUNT_64;
        mach_port_t obj = MACH_PORT_NULL;
        if (mach_vm_region(mach_task_self(), &raddr, &rsize, VM_REGION_BASIC_INFO_64,
                           (vm_region_info_t)&info, &cnt, &obj) == KERN_SUCCESS
            && raddr <= base && raddr + rsize >= hi
            && (info.protection & VM_PROT_READ) && (info.protection & VM_PROT_WRITE)
            && rsize <= (512u << 20)) {
            lo = (uintptr_t)raddr;
            hi = (uintptr_t)(raddr + rsize);
        }
        if (obj != MACH_PORT_NULL) mach_port_deallocate(mach_task_self(), obj);
    }
#endif
    auto ov = g_wrapped.lower_bound(lo);
    if (ov != g_wrapped.begin()) {
        auto prev = std::prev(ov);
        if (prev->first + (size_t)prev->second.length > lo) ov = prev;
    }
    while (ov != g_wrapped.end() && ov->first < hi) {
        if (ov->first + (size_t)ov->second.length > lo) {
            if (ov->first < lo) lo = ov->first;
            if (ov->first + (size_t)ov->second.length > hi) hi = ov->first + (size_t)ov->second.length;
            ov = g_wrapped.erase(ov);
        } else ++ov;
    }
    id<MTLBuffer> b = [ctx().dev newBufferWithBytesNoCopy:(void*)lo length:(size_t)(hi - lo)
                       options:MTLResourceStorageModeShared deallocator:nil];
    if (!b) return { nil, 0 };
    g_wrapped[lo] = b;
    return { b, (size_t)(a - lo) };
}

std::pair<id<MTLBuffer>, size_t> bufForPtrOff(const void* p, size_t bytes) {
    uintptr_t a = reinterpret_cast<uintptr_t>(p);
    auto it = g_shared.upper_bound(a);
    if (it != g_shared.begin()) {
        --it;
        uintptr_t base = it->first;
        if (a < base + static_cast<uintptr_t>(it->second.length))
            return { it->second, static_cast<size_t>(a - base) };
    }

    auto wr = wrapHostPtr(p, bytes);
    if (wr.first) return wr;

    size_t nb = bytes ? bytes : 1;
    id<MTLBuffer> b = recycled(bytes);
    std::memcpy([b contents], p, nb);
    return { b, 0 };
}

inline void setBufAt(const void* p, size_t bytes, int idx) {
    auto pr = bufForPtrOff(p, bytes);
    [g_enc setBuffer:pr.first offset:pr.second atIndex:idx];
}

}

bool cactus_metal_available() { return ctx().ok; }

void cactus_metal_set_active(bool a) { g_active = a; }
bool cactus_metal_active_mode() { return ctx().ok && g_active; }

extern "C" void* objc_autoreleasePoolPush(void);
extern "C" void objc_autoreleasePoolPop(void*);
static void* g_arpool = nullptr;

void cactus_metal_session_begin() {
    if (!g_arpool) g_arpool = objc_autoreleasePoolPush();
}
static id<MTLCommandBuffer> g_last_cmd = nil;
static std::atomic<bool> g_cmd_failed{false};
static void watch_cmd_errors(id<MTLCommandBuffer> cmd) {
    [cmd addCompletedHandler:^(id<MTLCommandBuffer> cb) {
        if (cb.status == MTLCommandBufferStatusError) {
            g_cmd_failed.store(true, std::memory_order_relaxed);
            fprintf(stderr, "[cactus-metal] command buffer failed: %s\n",
                    cb.error ? [[cb.error localizedDescription] UTF8String] : "unknown");
        }
    }];
}
static void reclaim_dead_slabs();
void cactus_metal_session_sync() {
    if (g_enc) { [g_enc endEncoding]; g_enc = nil; }
    if (g_cmd) {
        watch_cmd_errors(g_cmd);
        [g_cmd commit];
        g_last_cmd = g_cmd;
        g_cmd = nil;
    }
    if (g_last_cmd) {
        [g_last_cmd waitUntilCompleted];
        g_last_cmd = nil;
    }
    for (id<MTLBuffer> b : g_pending) g_free[(size_t)b.length].push_back(b);
    g_pending.clear();
    reclaim_dead_slabs();
}
void cactus_metal_session_end() {
    cactus_metal_session_sync();
    if (g_arpool) { objc_autoreleasePoolPop(g_arpool); g_arpool = nullptr; }
}
void cactus_metal_invalidate_host_wraps() {
    std::lock_guard<std::mutex> lk(g_resident_mu);
    g_attn_scores = nil;
    g_wrapped.clear();
    g_readonly.clear();
    g_resident.clear();
    g_resident_emb.clear();
    g_mps_mats.clear();
    g_moe_cat.clear();
}

void cactus_metal_session_flush() {
    if (g_enc) { [g_enc endEncoding]; g_enc = nil; }
    if (g_cmd) { watch_cmd_errors(g_cmd); [g_cmd commit]; g_last_cmd = g_cmd; g_cmd = nil; }
}


void* cactus_metal_alloc_shared(size_t bytes) {
    if (!ctx().ok) return nullptr;
    size_t bk = bucket(bytes ? bytes : 1);
    id<MTLBuffer> b = nil;
    auto it = g_free.find(bk);
    if (it != g_free.end() && !it->second.empty()) { b = it->second.back(); it->second.pop_back(); }
    else b = owned_shared(bk);
    void* c = [b contents];
    g_shared[reinterpret_cast<uintptr_t>(c)] = b;
    return c;
}

struct SlabInfo { size_t cap = 0, used = 0, live = 0; };
static char* g_slab = nullptr;
static std::map<uintptr_t, SlabInfo> g_slabs;

static void reclaim_dead_slabs() {
    for (auto it = g_slabs.begin(); it != g_slabs.end();) {
        SlabInfo& s = it->second;
        if (s.live != 0 || s.used == 0) { ++it; continue; }
        if ((char*)it->first == g_slab) { s.used = 0; ++it; continue; }
        g_shared.erase(it->first);
        it = g_slabs.erase(it);
    }
}

void* cactus_metal_alloc_pooled(size_t bytes) {
    if (!ctx().ok) return nullptr;
    if (bytes > (4u << 20)) return cactus_metal_alloc_shared(bytes);
    size_t need = (bytes + 255) & ~size_t(255);
    SlabInfo* cur = nullptr;
    if (g_slab) {
        auto it = g_slabs.find(reinterpret_cast<uintptr_t>(g_slab));
        if (it != g_slabs.end()) cur = &it->second;
    }
    if (!cur || cur->used + need > cur->cap) {
        size_t cap = 32u << 20;
        id<MTLBuffer> b = owned_shared(cap);
        if (!b) return cactus_metal_alloc_shared(bytes);
        g_slab = (char*)[b contents];
        g_shared[reinterpret_cast<uintptr_t>(g_slab)] = b;
        cur = &g_slabs[reinterpret_cast<uintptr_t>(g_slab)];
        *cur = SlabInfo{cap, 0, 0};
    }
    void* p = g_slab + cur->used;
    cur->used += need;
    cur->live += 1;
    return p;
}
void cactus_metal_free_shared(void* contents) {
    uintptr_t a = reinterpret_cast<uintptr_t>(contents);
    auto sit = g_slabs.upper_bound(a);
    if (sit != g_slabs.begin()) {
        --sit;
        if (a < sit->first + sit->second.cap) {
            if (sit->second.live > 0) --sit->second.live;
            return;
        }
    }
    auto it = g_shared.find(a);
    if (it != g_shared.end()) { g_pending.push_back(it->second); g_shared.erase(it); }
}

bool cactus_metal_encode_copy(void* out, const void* in, size_t bytes) {
    if (!ctx().ok) return false;
    ensureEncoder();
    uint32_t n=(uint32_t)bytes;
    [g_enc setComputePipelineState:ctx().psoCopy];
    setBufAt(in, bytes, 0);
    setBufAt(out, bytes, 1);
    [g_enc setBytes:&n length:4 atIndex:2];
    [g_enc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    return true;
}
bool cactus_metal_encode_binary(int op, void* out, const void* a, const void* b, size_t n) {
    if (!ctx().ok) return false;
    ensureEncoder();
    uint32_t nn=(uint32_t)n; int o=op;
    [g_enc setComputePipelineState:ctx().psoBinary];
    setBufAt(a, n*2, 0); setBufAt(b, n*2, 1); setBufAt(out, n*2, 2);
    [g_enc setBytes:&nn length:4 atIndex:3]; [g_enc setBytes:&o length:4 atIndex:4];
    [g_enc dispatchThreads:MTLSizeMake((nn+3)/4,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    return true;
}
bool cactus_metal_encode_scalar(int op, void* out, const void* in, size_t n, float param) {
    if (!ctx().ok) return false;
    ensureEncoder();
    uint32_t nn=(uint32_t)n; int o=op; float p=param;
    [g_enc setComputePipelineState:ctx().psoScalar];
    setBufAt(in, n*2, 0); setBufAt(out, n*2, 1);
    [g_enc setBytes:&nn length:4 atIndex:2]; [g_enc setBytes:&o length:4 atIndex:3]; [g_enc setBytes:&p length:4 atIndex:4];
    [g_enc dispatchThreads:MTLSizeMake((nn+3)/4,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    return true;
}
bool cactus_metal_encode_unary(int op, void* out, const void* in, size_t n) {
    if (!ctx().ok) return false;
    ensureEncoder();
    uint32_t nn=(uint32_t)n; int o=op;
    [g_enc setComputePipelineState:ctx().psoUnary];
    setBufAt(in, n*2, 0); setBufAt(out, n*2, 1);
    [g_enc setBytes:&nn length:4 atIndex:2]; [g_enc setBytes:&o length:4 atIndex:3];
    [g_enc dispatchThreads:MTLSizeMake((nn+3)/4,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    return true;
}
bool cactus_metal_encode_swiglu(void* out, const void* gate, const void* up, size_t n, float scale) {
    if (!ctx().ok) return false;
    ensureEncoder();
    uint32_t nn=(uint32_t)n; float s=scale;
    [g_enc setComputePipelineState:ctx().psoSwiglu];
    setBufAt(gate, n*2, 0); setBufAt(up, n*2, 1); setBufAt(out, n*2, 2);
    [g_enc setBytes:&nn length:4 atIndex:3]; [g_enc setBytes:&s length:4 atIndex:4];
    [g_enc dispatchThreads:MTLSizeMake(nn,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    return true;
}
bool cactus_metal_encode_rms_norm(void* out, const void* in, const void* weight,
                                  size_t rows, size_t dim, float eps) {
    if (!ctx().ok) return false;
    ensureEncoder();
    uint32_t d=(uint32_t)dim; float e=eps;
    if (rows > 1 && ctx().psoRmsSimd) {
        uint32_t r=(uint32_t)rows;
        [g_enc setComputePipelineState:ctx().psoRmsSimd];
        setBufAt(in, rows*dim*2, 0); setBufAt(weight, dim*2, 1); setBufAt(out, rows*dim*2, 2);
        [g_enc setBytes:&d length:4 atIndex:3]; [g_enc setBytes:&e length:4 atIndex:4];
        [g_enc setBytes:&r length:4 atIndex:5];
        [g_enc dispatchThreadgroups:MTLSizeMake((rows+3)/4,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
        return true;
    }
    [g_enc setComputePipelineState:ctx().psoRms];
    setBufAt(in, rows*dim*2, 0); setBufAt(weight, dim*2, 1); setBufAt(out, rows*dim*2, 2);
    [g_enc setBytes:&d length:4 atIndex:3]; [g_enc setBytes:&e length:4 atIndex:4];
    [g_enc setThreadgroupMemoryLength:256*sizeof(float) atIndex:0];
    [g_enc dispatchThreadgroups:MTLSizeMake(rows,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    return true;
}
bool cactus_metal_encode_rms_norm_add(void* out, const void* in, const void* weight, const void* res,
                                      size_t rows, size_t dim, float eps) {
    if (!ctx().ok) return false;
    ensureEncoder();
    uint32_t d=(uint32_t)dim; float e=eps;
    [g_enc setComputePipelineState:ctx().psoRmsAdd];
    setBufAt(in, rows*dim*2, 0); setBufAt(weight, dim*2, 1); setBufAt(res, rows*dim*2, 2); setBufAt(out, rows*dim*2, 3);
    [g_enc setBytes:&d length:4 atIndex:4]; [g_enc setBytes:&e length:4 atIndex:5];
    [g_enc setThreadgroupMemoryLength:256*sizeof(float) atIndex:0];
    [g_enc dispatchThreadgroups:MTLSizeMake(rows,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    return true;
}
bool cactus_metal_encode_rms_norm_add_scale(void* out, const void* in, const void* weight, const void* res,
                                            size_t rows, size_t dim, float eps, float out_scale) {
    if (!ctx().ok || !ctx().psoRmsAddScale) return false;
    ensureEncoder();
    uint32_t d=(uint32_t)dim; float e=eps, os=out_scale;
    [g_enc setComputePipelineState:ctx().psoRmsAddScale];
    setBufAt(in, rows*dim*2, 0); setBufAt(weight, dim*2, 1); setBufAt(res, rows*dim*2, 2); setBufAt(out, rows*dim*2, 3);
    [g_enc setBytes:&d length:4 atIndex:4]; [g_enc setBytes:&e length:4 atIndex:5]; [g_enc setBytes:&os length:4 atIndex:6];
    [g_enc setThreadgroupMemoryLength:256*sizeof(float) atIndex:0];
    [g_enc dispatchThreadgroups:MTLSizeMake(rows,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    return true;
}
bool cactus_metal_encode_rms_norm_add_rms(void* h_out, void* xn_out, const void* in, const void* w1,
                                          const void* res, const void* w2,
                                          size_t rows, size_t dim, float eps, float out_scale) {
    if (!ctx().ok || !ctx().psoRmsAddRms) return false;
    ensureEncoder();
    uint32_t d=(uint32_t)dim; float e=eps, os=out_scale;
    [g_enc setComputePipelineState:ctx().psoRmsAddRms];
    setBufAt(in, rows*dim*2, 0); setBufAt(w1, dim*2, 1); setBufAt(res, rows*dim*2, 2);
    setBufAt(h_out, rows*dim*2, 3); setBufAt(w2, dim*2, 4); setBufAt(xn_out, rows*dim*2, 5);
    [g_enc setBytes:&d length:4 atIndex:6]; [g_enc setBytes:&e length:4 atIndex:7]; [g_enc setBytes:&os length:4 atIndex:8];
    [g_enc setThreadgroupMemoryLength:256*sizeof(float) atIndex:0];
    [g_enc dispatchThreadgroups:MTLSizeMake(rows,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    return true;
}

bool cactus_metal_encode_argmax(const void* logits, uint32_t vocab, void* out3, const void* bias) {
    if (!ctx().ok) return false;
    if (bias && !(ctx().psoArgmaxP && ctx().psoArgmaxF && vocab >= 32768u)) return false;
    ensureEncoder();
    uint32_t V = vocab;
    if (ctx().psoArgmaxP && ctx().psoArgmaxF && V >= 32768u) {
        const uint32_t NP = 128u, T = 256u;
        uint32_t chunk = (V + NP - 1u)/NP;
        uint32_t has_bias = bias ? 1u : 0u;
        id<MTLBuffer> part = recycled((size_t)NP*3*sizeof(float));
        [g_enc setComputePipelineState:ctx().psoArgmaxP];
        setBufAt(logits, (size_t)vocab*2, 0);
        [g_enc setBuffer:part offset:0 atIndex:1];
        [g_enc setBytes:&V length:4 atIndex:2]; [g_enc setBytes:&chunk length:4 atIndex:3];
        if (bias) setBufAt(bias, (size_t)vocab*sizeof(float), 4); else [g_enc setBuffer:ctx().dummy offset:0 atIndex:4];
        [g_enc setBytes:&has_bias length:4 atIndex:5];
        [g_enc setThreadgroupMemoryLength:T*sizeof(float) atIndex:0];
        [g_enc setThreadgroupMemoryLength:T*sizeof(uint) atIndex:1];
        [g_enc setThreadgroupMemoryLength:T*sizeof(float) atIndex:2];
        [g_enc dispatchThreadgroups:MTLSizeMake(NP,1,1) threadsPerThreadgroup:MTLSizeMake(T,1,1)];
        [g_enc setComputePipelineState:ctx().psoArgmaxF];
        [g_enc setBuffer:part offset:0 atIndex:0];
        setBufAt(out3, 3*sizeof(float), 1);
        [g_enc setBytes:&NP length:4 atIndex:2];
        [g_enc setThreadgroupMemoryLength:NP*sizeof(float) atIndex:0];
        [g_enc setThreadgroupMemoryLength:NP*sizeof(uint) atIndex:1];
        [g_enc setThreadgroupMemoryLength:NP*sizeof(float) atIndex:2];
        [g_enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(NP,1,1)];
        return true;
    }
    uint32_t T = 1024;
    [g_enc setComputePipelineState:ctx().psoArgmax];
    setBufAt(logits, (size_t)vocab*2, 0);
    setBufAt(out3, 3*sizeof(float), 1);
    [g_enc setBytes:&V length:4 atIndex:2];
    [g_enc setThreadgroupMemoryLength:T*sizeof(float) atIndex:0];
    [g_enc setThreadgroupMemoryLength:T*sizeof(uint) atIndex:1];
    [g_enc setThreadgroupMemoryLength:T*sizeof(float) atIndex:2];
    [g_enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(T,1,1)];
    return true;
}
bool cactus_metal_encode_adjust_logits(void* logits, size_t vocab,
                                       const uint32_t* recent, uint32_t n_recent,
                                       int64_t suppressed, float penalty) {
    if (!ctx().ok || !ctx().psoAdjust) return false;
    if (n_recent > 4096u) return false;
    ensureEncoder();
    id<MTLBuffer> rb = ctx().dummy;
    if (n_recent) { rb = recycled((size_t)n_recent*4); std::memcpy([rb contents], recent, (size_t)n_recent*4); }
    struct { uint32_t n_recent, suppress_flag, suppress_id, vocab_n; float penalty; } U =
        { n_recent,
          (uint32_t)(suppressed >= 0 ? 1 : 0), (uint32_t)(suppressed >= 0 ? suppressed : 0),
          (uint32_t)vocab, penalty };
    [g_enc setComputePipelineState:ctx().psoAdjust];
    setBufAt(logits, vocab*2, 0);
    [g_enc setBuffer:rb offset:0 atIndex:1];
    [g_enc setBytes:&U length:sizeof(U) atIndex:2];
    [g_enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1024,1,1)];
    return true;
}

bool cactus_metal_encode_softcap(void* out, const void* in, size_t n, float cap) {
    if (!ctx().ok || !ctx().psoSoftcap) return false;
    ensureEncoder();
    uint32_t nn=(uint32_t)n; float c=cap;
    [g_enc setComputePipelineState:ctx().psoSoftcap];
    setBufAt(in, n*2, 0); setBufAt(out, n*2, 1);
    [g_enc setBytes:&nn length:4 atIndex:2]; [g_enc setBytes:&c length:4 atIndex:3];
    [g_enc dispatchThreads:MTLSizeMake(nn,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    return true;
}
bool cactus_metal_encode_cast(void* out, int out_prec, const void* in, int in_prec, size_t n) {
    if (!ctx().ok) return false;

    auto esz=[](int p){ return p==0?1: p==1?2: p==2?4: 0; };
    if (in_prec==out_prec) return cactus_metal_encode_copy(out, in, n*esz(in_prec));
    id<MTLComputePipelineState> pso=nil;
    if (in_prec==1&&out_prec==2) pso=ctx().psoCF16F32;
    else if (in_prec==2&&out_prec==1) pso=ctx().psoCF32F16;
    else if (in_prec==0&&out_prec==1) pso=ctx().psoCI8F16;
    else if (in_prec==1&&out_prec==0) pso=ctx().psoCF16I8;
    else return false;
    ensureEncoder();
    uint32_t nn=(uint32_t)n;
    [g_enc setComputePipelineState:pso];
    setBufAt(in, n*esz(in_prec), 0); setBufAt(out, n*esz(out_prec), 1);
    [g_enc setBytes:&nn length:4 atIndex:2];
    [g_enc dispatchThreads:MTLSizeMake(nn,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    return true;
}
static inline bool quant_fast_eligible(const CactusQuantMatrix* W) {
    return W->bits == 4 && W->group_size >= 128 && (W->group_size % 128) == 0 && (W->N % 4) == 0 &&
           (W->flags & CACTUS_QUANT_FLAG_INTERLEAVED_4ROW) &&
           !(W->flags & CACTUS_QUANT_FLAG_ORTHOGONAL) && W->input_scale_recip && W->left_signs &&
           W->right_signs && W->permutation && W->codebook && W->norms && W->packed_indices;
}

static inline bool quant_lowbit_eligible(const CactusQuantMatrix* W) {
    return (W->bits == 2 || W->bits == 3) && W->group_size == 128 && (W->N % 16) == 0 &&
           (W->flags & CACTUS_QUANT_FLAG_INTERLEAVED_4ROW) &&
           !(W->flags & CACTUS_QUANT_FLAG_ORTHOGONAL) &&
           W->input_scale_recip && W->left_signs && W->right_signs && W->permutation &&
           W->codebook && W->norms && W->packed_indices;
}

void cactus_metal_trim_prefill_cache() {
    std::lock_guard<std::mutex> lk(g_resident_mu);
    g_mps_mats.clear();
    g_free.clear();
    g_attn_scores = nil;
    g_code_buf = nil;
    g_code_buf_m = nil;
}

bool cactus_metal_encode_quant_matmul(void* out, const void* lhs, const CactusQuantMatrix* W) {
    if (!ctx().ok || !W) return false;
    const uint32_t gs=W->group_size, K=W->K, N=W->N, ng=W->num_groups;
    if ((W->flags & CACTUS_QUANT_FLAG_ORTHOGONAL) && W->rotation && ng==1 && gs==K && (N%4)==0) {
        static void* ortho_code = nullptr; static uint32_t ortho_code_k = 0;
        if (ortho_code_k < K) {
            if (ortho_code) cactus_metal_free_shared(ortho_code);
            ortho_code = cactus_metal_alloc_shared((size_t)K*sizeof(__fp16));
            ortho_code_k = K;
        }
        if (ortho_code) return cactus_metal_encode_quant_matmul_ortho(out, lhs, ortho_code, W);
    }
    const bool fast = quant_fast_eligible(W);
    const bool lowbit = !fast && ctx().psoGmrLow && quant_lowbit_eligible(W);
    if (!fast && !lowbit) return false;
    const uint32_t bits = W->bits, pgbw = (gs*bits+7u)/8u;
    ResW& rw = resident(W);
    if (!rw.ok) return false;

    size_t code_bytes = (size_t)ng*gs*sizeof(__fp16);
    if (!g_code_buf || (size_t)g_code_buf.length < code_bytes)
        g_code_buf = owned_shared(code_bytes);
    if (!g_code_buf) return false;
    ensureEncoder();
    bool simdT = (gs==128u && ctx().psoTsimd);
    [g_enc setComputePipelineState:(simdT?ctx().psoTsimd:ctx().psoT)];
    setBufAt(lhs, (size_t)K*2, 0);                 [g_enc setBuffer:rw.recip offset:rw.rc_off atIndex:1];
    [g_enc setBuffer:rw.lsign offset:rw.ls_off atIndex:2]; [g_enc setBuffer:rw.rsign offset:rw.rs_off atIndex:3];
    [g_enc setBuffer:rw.perm offset:rw.pm_off atIndex:4];  [g_enc setBuffer:g_code_buf offset:0 atIndex:5];
    [g_enc setBytes:&gs length:4 atIndex:6]; [g_enc setThreadgroupMemoryLength:gs*sizeof(float) atIndex:0];
    [g_enc dispatchThreadgroups:MTLSizeMake(ng,1,1) threadsPerThreadgroup:MTLSizeMake(simdT?32u:(gs>1024u?1024u:gs),1,1)];
    if (lowbit && bits == 2u && gs == 128u && ctx().psoActQ && ctx().psoG2i8 && rw.cb_i8) {
        id<MTLBuffer> actq = recycled((size_t)ng*gs);
        id<MTLBuffer> ascl = recycled((size_t)ng*sizeof(float));
        [g_enc setComputePipelineState:ctx().psoActQ];
        [g_enc setBuffer:g_code_buf offset:0 atIndex:0];
        [g_enc setBuffer:actq offset:0 atIndex:1]; [g_enc setBuffer:ascl offset:0 atIndex:2];
        [g_enc setBytes:&gs length:4 atIndex:3];
        [g_enc dispatchThreadgroups:MTLSizeMake(ng,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [g_enc setComputePipelineState:ctx().psoG2i8];
        [g_enc setBuffer:actq offset:0 atIndex:0]; [g_enc setBuffer:ascl offset:0 atIndex:1];
        [g_enc setBuffer:rw.packed offset:rw.packed_off atIndex:2]; [g_enc setBuffer:rw.cb_i8 offset:0 atIndex:3];
        [g_enc setBuffer:rw.norms offset:rw.norms_off atIndex:4];
        setBufAt(out, (size_t)N*2, 5);
        [g_enc setBytes:&ng length:4 atIndex:6]; [g_enc setBytes:&N length:4 atIndex:7];
        [g_enc setBytes:&rw.cb_scale length:4 atIndex:8];
        [g_enc dispatchThreadgroups:MTLSizeMake((N+7u)/8u,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        return true;
    }
    bool umr = ctx().psoGmr && (N % g_mr_rows == 0u);
    [g_enc setComputePipelineState:(lowbit?ctx().psoGmrLow:(umr?ctx().psoGmr:ctx().psoG))];
    [g_enc setBuffer:g_code_buf offset:0 atIndex:0]; [g_enc setBuffer:rw.packed offset:rw.packed_off atIndex:1];
    [g_enc setBuffer:rw.codebook offset:rw.cb_off atIndex:2]; [g_enc setBuffer:rw.norms offset:rw.norms_off atIndex:3];
    setBufAt(out, (size_t)N*2, 4);
    [g_enc setBytes:&gs length:4 atIndex:5]; [g_enc setBytes:&ng length:4 atIndex:6];
    [g_enc setBytes:&pgbw length:4 atIndex:7]; [g_enc setBytes:&N length:4 atIndex:8];
    if (lowbit) { [g_enc setBytes:&bits length:4 atIndex:9]; [g_enc setBytes:&rw.il length:4 atIndex:10]; }
    else [g_enc setBytes:&rw.il length:4 atIndex:9];
    uint32_t ROWS=8;
    uint32_t grid = (lowbit || umr) ? (N+g_mr_rows-1u)/g_mr_rows : (N+ROWS-1u)/ROWS;
    [g_enc dispatchThreadgroups:MTLSizeMake(grid,1,1) threadsPerThreadgroup:MTLSizeMake(ROWS*32,1,1)];
    return true;
}

bool cactus_metal_encode_transform_batch(const void* x, const CactusQuantMatrix* const* Ws, int B, void* const* codes) {
    if (!ctx().ok || B < 1 || B > 3 || !ctx().psoTbatch) return false;
    const uint32_t gs=Ws[0]->group_size, K=Ws[0]->K, ng=Ws[0]->num_groups;
    if (gs != 128u || !quant_fast_eligible(Ws[0])) return false;
    ResW& r0 = resident(Ws[0]);
    ResW* rw[3] = { &r0, &r0, &r0 };
    for (int b=1;b<B;b++) rw[b] = &resident(Ws[b]);
    for (int b=0;b<B;b++) if (!rw[b]->ok) return false;
    ensureEncoder();
    [g_enc setComputePipelineState:ctx().psoTbatch];
    setBufAt(x, (size_t)K*2, 0);
    [g_enc setBuffer:r0.recip offset:r0.rc_off atIndex:1];
    for (int b=0;b<3;b++) {
        ResW* R = rw[b<B?b:0];
        [g_enc setBuffer:R->lsign offset:R->ls_off atIndex:2+b];
        [g_enc setBuffer:R->rsign offset:R->rs_off atIndex:5+b];
        [g_enc setBuffer:R->perm  offset:R->pm_off atIndex:8+b];
        setBufAt(codes[b<B?b:0], (size_t)K*2, 11+b);
    }
    [g_enc setBytes:&ng length:4 atIndex:14];
    [g_enc setThreadgroupMemoryLength:gs*sizeof(float) atIndex:0];
    [g_enc dispatchThreadgroups:MTLSizeMake((size_t)ng*B,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    return true;
}

bool cactus_metal_encode_gemv_precoded(void* out, const void* code, const CactusQuantMatrix* W) {
    if (!ctx().ok) return false;
    const uint32_t gs=W->group_size, K=W->K, N=W->N, ng=W->num_groups, pgb=(gs*4u+7u)/8u;
    if (!quant_fast_eligible(W)) return false;
    ResW& rw = resident(W);
    if (!rw.ok) return false;
    ensureEncoder();
    bool umr = ctx().psoGmr && (N % g_mr_rows == 0u);
    [g_enc setComputePipelineState:(umr?ctx().psoGmr:ctx().psoG)];
    setBufAt(code, (size_t)K*2, 0); [g_enc setBuffer:rw.packed offset:rw.packed_off atIndex:1];
    [g_enc setBuffer:rw.codebook offset:rw.cb_off atIndex:2]; [g_enc setBuffer:rw.norms offset:rw.norms_off atIndex:3];
    setBufAt(out, (size_t)N*2, 4);
    [g_enc setBytes:&gs length:4 atIndex:5]; [g_enc setBytes:&ng length:4 atIndex:6];
    [g_enc setBytes:&pgb length:4 atIndex:7]; [g_enc setBytes:&N length:4 atIndex:8];
    [g_enc setBytes:&rw.il length:4 atIndex:9];
    uint32_t ROWS=8;
    uint32_t grid = umr ? (N+g_mr_rows-1u)/g_mr_rows : (N+ROWS-1u)/ROWS;
    [g_enc dispatchThreadgroups:MTLSizeMake(grid,1,1) threadsPerThreadgroup:MTLSizeMake(ROWS*32,1,1)];
    return true;
}

bool cactus_metal_transform_gemv_fits(uint32_t K) {
    return ctx().ok && tg_mem_ok((size_t)K*2 + 8*128*sizeof(float) + 64);
}

bool cactus_metal_encode_transform_gemv(void* out, const void* x, const CactusQuantMatrix* W, const void* osw) {
    if (!ctx().ok || !ctx().psoMega) return false;
    const uint32_t gs=W->group_size, K=W->K, ng=W->num_groups, N=W->N;
    if (gs != 128u || !quant_fast_eligible(W) || (N % 16u) != 0u) return false;
    if (!cactus_metal_transform_gemv_fits(K)) return false;
    ResW& rw = resident(W);
    if (!rw.ok) return false;
    ensureEncoder();
    struct { uint32_t K, ng, oswi; } U = { K, ng, (uint32_t)(osw ? 1 : 0) };
    [g_enc setComputePipelineState:ctx().psoMega];
    setBufAt(x, (size_t)K*2, 0);
    [g_enc setBuffer:rw.recip offset:rw.rc_off atIndex:1];
    [g_enc setBuffer:rw.lsign offset:rw.ls_off atIndex:2]; [g_enc setBuffer:rw.rsign offset:rw.rs_off atIndex:3];
    [g_enc setBuffer:rw.perm offset:rw.pm_off atIndex:4];  [g_enc setBuffer:rw.codebook offset:rw.cb_off atIndex:5];
    [g_enc setBuffer:rw.norms offset:rw.norms_off atIndex:6]; [g_enc setBuffer:rw.packed offset:rw.packed_off atIndex:7];
    setBufAt(out, (size_t)N*2, 8);
    if (osw) setBufAt(osw, (size_t)N*2, 9); else [g_enc setBuffer:ctx().dummy offset:0 atIndex:9];
    [g_enc setBytes:&U length:sizeof(U) atIndex:10];
    [g_enc setThreadgroupMemoryLength:(size_t)K*2 atIndex:0];
    [g_enc setThreadgroupMemoryLength:8*128*sizeof(float) atIndex:1];
    [g_enc dispatchThreadgroups:MTLSizeMake(N/16u,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    return true;
}

bool cactus_metal_encode_swiglu_transform(void* code, const void* gate, const void* up,
                                          const CactusQuantMatrix* W, float scale) {
    if (!ctx().ok || !ctx().psoSwiT || W->group_size != 128u || !quant_fast_eligible(W)) return false;
    const uint32_t K=W->K, ng=W->num_groups;
    ResW& rw = resident(W);
    if (!rw.ok) return false;
    ensureEncoder();
    float s = scale;
    [g_enc setComputePipelineState:ctx().psoSwiT];
    setBufAt(gate, (size_t)K*2, 0); setBufAt(up, (size_t)K*2, 1);
    [g_enc setBuffer:rw.recip offset:rw.rc_off atIndex:2];
    [g_enc setBuffer:rw.lsign offset:rw.ls_off atIndex:3]; [g_enc setBuffer:rw.rsign offset:rw.rs_off atIndex:4];
    [g_enc setBuffer:rw.perm offset:rw.pm_off atIndex:5];
    setBufAt(code, (size_t)K*2, 6);
    [g_enc setBytes:&s length:4 atIndex:7];
    [g_enc setThreadgroupMemoryLength:128*sizeof(float) atIndex:0];
    [g_enc dispatchThreadgroups:MTLSizeMake(ng,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    return true;
}

bool cactus_metal_encode_gemv_cat(void* const* outs, const void* const* codes,
                                  const CactusQuantMatrix* const* Ws, int B) {
    if (!ctx().ok || !ctx().psoCat || B < 1 || B > 3) return false;
    const uint32_t gs=Ws[0]->group_size, K=Ws[0]->K, ng=Ws[0]->num_groups;
    if (gs != 128u) return false;
    uint32_t Ns[3]={0,0,0};
    for (int i=0;i<B;i++){
        if (!quant_fast_eligible(Ws[i]) || Ws[i]->K != K || (Ws[i]->N % 16u) != 0u) return false;
        Ns[i]=Ws[i]->N;
    }
    ResW* rw[3]; rw[0]=&resident(Ws[0]); rw[1]=rw[0]; rw[2]=rw[0];
    for (int i=1;i<B;i++) rw[i]=&resident(Ws[i]);
    for (int i=0;i<B;i++) if (!rw[i]->ok) return false;
    ensureEncoder();
    struct { uint32_t ng, N0, N1, N2; } U = { ng, Ns[0], Ns[1], Ns[2] };
    [g_enc setComputePipelineState:ctx().psoCat];
    for (int i=0;i<3;i++){
        int j = i<B?i:0;
        setBufAt(codes[j], (size_t)K*2, 0+i);
        [g_enc setBuffer:rw[j]->packed offset:rw[j]->packed_off atIndex:3+i];
        [g_enc setBuffer:rw[j]->codebook offset:rw[j]->cb_off atIndex:6+i];
        [g_enc setBuffer:rw[j]->norms offset:rw[j]->norms_off atIndex:9+i];
        setBufAt(outs[j], (size_t)Ns[j]*2, 12+i);
    }
    [g_enc setBytes:&U length:sizeof(U) atIndex:15];
    uint32_t total = Ns[0]+Ns[1]+Ns[2];
    [g_enc dispatchThreadgroups:MTLSizeMake(total/16u,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    return true;
}

bool cactus_metal_prewarm_quant(const CactusQuantMatrix* W) {
    if (!ctx().ok) return false;
    if (!quant_fast_eligible(W) && !quant_lowbit_eligible(W)) return false;
    return resident(W).ok;
}

namespace {
bool moe_component_offsets(const CactusQuantMatrix& W, uintptr_t& lo, uintptr_t& hi, size_t offs[7]) {
    const uintptr_t pg = 16384;
    struct Span { const void* p; size_t len; };
    Span spans[7] = {
        { W.packed_indices, (size_t)W.N*W.num_groups*64u },
        { W.norms, (size_t)W.N*W.num_groups*2 },
        { W.input_scale_recip, (size_t)W.K*2 },
        { W.left_signs, 128 },
        { W.right_signs, 128 },
        { W.permutation, 512 },
        { W.codebook, 32 },
    };
    uintptr_t mn = UINTPTR_MAX, mx = 0;
    for (const auto& s : spans) {
        if (!s.p) return false;
        uintptr_t a = (uintptr_t)s.p;
        if (a < mn) mn = a;
        if (a + s.len > mx) mx = a + s.len;
    }
    lo = mn & ~(pg - 1);
    hi = (mx + pg - 1) & ~(pg - 1);
    for (int i = 0; i < 7; i++) offs[i] = (uintptr_t)spans[i].p - lo;
    return true;
}

bool moe_build_set4_zerocopy(MoESet4& S, const CactusQuantMatrix* Ws, uint32_t count) {
    if (count == 0) return false;
    uintptr_t lo0, hi0;
    size_t offs[7];
    if (!moe_component_offsets(Ws[0], lo0, hi0, offs)) return false;
    const size_t slot = hi0 - lo0;
    if (slot > 0xFFFFFFFFull) return false;
    if ((offs[0] & 7u) || (offs[1] & 1u) || (offs[2] & 1u) || (offs[6] & 1u) || (offs[5] & 3u)) return false;
    for (uint32_t e = 1; e < count; ++e) {
        uintptr_t lo, hi;
        size_t o[7];
        if (!moe_component_offsets(Ws[e], lo, hi, o)) return false;
        if (hi - lo != slot) return false;
        for (int i = 0; i < 7; i++) if (o[i] != offs[i]) return false;
    }
    const size_t total = slot * count;
    void* arena = mmap(nullptr, total, PROT_NONE, MAP_ANON | MAP_PRIVATE, -1, 0);
    if (arena == MAP_FAILED) return false;
    for (uint32_t e = 0; e < count; ++e) {
        uintptr_t lo, hi;
        size_t o[7];
        moe_component_offsets(Ws[e], lo, hi, o);
        vm_address_t dst = (vm_address_t)((uintptr_t)arena + (size_t)e*slot);
        vm_prot_t curp = 0, maxp = 0;
        kern_return_t kr = vm_remap(mach_task_self(), &dst, slot, 0,
            VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE, mach_task_self(),
            (vm_address_t)lo, FALSE, &curp, &maxp, VM_INHERIT_NONE);
        if (kr != KERN_SUCCESS || dst != (vm_address_t)((uintptr_t)arena + (size_t)e*slot)
            || !(curp & VM_PROT_READ)) {
            munmap(arena, total);
            return false;
        }
    }
    id<MTLBuffer> ab = [ctx().dev newBufferWithBytesNoCopy:arena length:total
                        options:MTLResourceStorageModeShared deallocator:nil];
    if (!ab) { munmap(arena, total); return false; }
    CactusThreading::parallel_for(count, CactusThreading::ParallelConfig(1, 1),
        [&](size_t start, size_t end) {
            volatile uint64_t acc = 0;
            for (size_t e = start; e < end; ++e)
                for (size_t o = 0; o < slot; o += 16384)
                    acc += *((const volatile uint8_t*)arena + e*slot + o);
            (void)acc;
        });
    S.K=Ws[0].K; S.N=Ws[0].N;
    S.pk=S.nm=S.rc=S.ls=S.rs=S.pm=S.cb=ab;
    S.pk_off=offs[0]; S.nm_off=offs[1]; S.rc_off=offs[2]; S.ls_off=offs[3];
    S.rs_off=offs[4]; S.pm_off=offs[5]; S.cb_off=offs[6];
    S.stride = (uint32_t)slot;
    return true;
}

bool moe_build_set4(MoESet4& S, const CactusQuantMatrix* Ws, uint32_t count) {
    const uint32_t K=Ws[0].K, N=Ws[0].N, gs=Ws[0].group_size, ng=Ws[0].num_groups;
    if (gs != 128u || Ws[0].bits != 4u || (N % 4u) != 0u || (K % 128u) != 0u) return false;
    for (uint32_t e = 0; e < count; ++e) {
        const CactusQuantMatrix* W = &Ws[e];
        if (!quant_fast_eligible(W) || W->K != K || W->N != N || W->num_groups != ng) return false;
    }
    return moe_build_set4_zerocopy(S, Ws, count);
}
}

bool cactus_metal_moe_cq4_ready(const CactusQuantMatrix* w1_0) {
    std::lock_guard<std::mutex> lk(g_resident_mu);
    auto it = g_moe_cat.find(resident_key(w1_0));
    return it != g_moe_cat.end() && it->second.ok;
}

bool cactus_metal_moe_cq4_build(const CactusQuantMatrix* w1s, const CactusQuantMatrix* w3s,
                                const CactusQuantMatrix* w2s, uint32_t num_experts) {
    if (!ctx().ok || !ctx().psoMoeT || !ctx().psoMoeT2 || !ctx().psoMoeUp || !ctx().psoMoeDownAcc)
        return false;
    const uint64_t key = resident_key(&w1s[0]);
    {
        std::lock_guard<std::mutex> lk(g_resident_mu);
        auto it = g_moe_cat.find(key);
        if (it != g_moe_cat.end()) return it->second.ok;
    }
    MoECat4 cat;
    cat.ok = moe_build_set4(cat.w1, w1s, num_experts)
          && moe_build_set4(cat.w3, w3s, num_experts)
          && moe_build_set4(cat.w2, w2s, num_experts)
          && cat.w1.K == cat.w3.K && cat.w1.N == cat.w3.N && cat.w2.K >= cat.w1.N;
    if (!cat.ok) cat = MoECat4{};
    std::lock_guard<std::mutex> lk(g_resident_mu);
    auto ins = g_moe_cat.emplace(key, cat);
    return ins.first->second.ok;
}

bool cactus_metal_encode_moe_gated_cq4(void* out, const void* hidden, const void* probs,
                                       const void* topk, const CactusQuantMatrix* w1_0,
                                       uint32_t num_experts, uint32_t top_k, uint32_t tokens,
                                       uint32_t act, uint32_t normalize, float eps, float scaling) {
    if (!ctx().ok || !ctx().psoMoeT || top_k == 0 || top_k > 16 || tokens == 0) return false;
    g_resident_mu.lock();
    auto it = g_moe_cat.find(resident_key(w1_0));
    bool hit = it != g_moe_cat.end() && it->second.ok;
    g_resident_mu.unlock();
    if (!hit) return false;
    MoECat4& C = it->second;
    const uint32_t K1=C.w1.K, N1=C.w1.N, K2=C.w2.K, N2=C.w2.N;
    const size_t slots = (size_t)tokens*top_k;
    ensureEncoder();

    id<MTLBuffer> code1 = recycled(slots*K1*2);
    id<MTLBuffer> code3 = recycled(slots*K1*2);
    id<MTLBuffer> gu    = recycled(slots*N1*2);
    id<MTLBuffer> code2 = recycled(slots*K2*2);

    {
        [g_enc setComputePipelineState:ctx().psoMoeT2];
        setBufAt(hidden, (size_t)tokens*K1*2, 0);
        setBufAt(topk, slots*sizeof(float), 1);
        [g_enc setBuffer:C.w1.rc offset:C.w1.rc_off atIndex:2];
        [g_enc setBuffer:C.w1.ls offset:C.w1.ls_off atIndex:3];
        [g_enc setBuffer:C.w1.rs offset:C.w1.rs_off atIndex:4];
        [g_enc setBuffer:C.w1.pm offset:C.w1.pm_off atIndex:5];
        [g_enc setBuffer:C.w3.rc offset:C.w3.rc_off atIndex:6];
        [g_enc setBuffer:C.w3.ls offset:C.w3.ls_off atIndex:7];
        [g_enc setBuffer:C.w3.rs offset:C.w3.rs_off atIndex:8];
        [g_enc setBuffer:C.w3.pm offset:C.w3.pm_off atIndex:9];
        [g_enc setBuffer:code1 offset:0 atIndex:10];
        [g_enc setBuffer:code3 offset:0 atIndex:11];
        uint32_t Kc=K1, tk=top_k;
        [g_enc setBytes:&Kc length:4 atIndex:12]; [g_enc setBytes:&tk length:4 atIndex:13];
        [g_enc setBytes:&C.w1.stride length:4 atIndex:14];
        [g_enc setBytes:&C.w3.stride length:4 atIndex:15];
        [g_enc setThreadgroupMemoryLength:128*sizeof(float) atIndex:0];
        [g_enc dispatchThreadgroups:MTLSizeMake(K1/128u, 2u*top_k, tokens)
              threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    }
    {
        [g_enc setComputePipelineState:ctx().psoMoeUp];
        [g_enc setBuffer:code1 offset:0 atIndex:0]; [g_enc setBuffer:code3 offset:0 atIndex:1];
        setBufAt(topk, slots*sizeof(float), 2);
        [g_enc setBuffer:C.w1.pk offset:C.w1.pk_off atIndex:3]; [g_enc setBuffer:C.w1.nm offset:C.w1.nm_off atIndex:4]; [g_enc setBuffer:C.w1.cb offset:C.w1.cb_off atIndex:5];
        [g_enc setBuffer:C.w3.pk offset:C.w3.pk_off atIndex:6]; [g_enc setBuffer:C.w3.nm offset:C.w3.nm_off atIndex:7]; [g_enc setBuffer:C.w3.cb offset:C.w3.cb_off atIndex:8];
        [g_enc setBuffer:gu offset:0 atIndex:9];
        uint32_t Kc=K1, Nc=N1, a=act, tk=top_k;
        [g_enc setBytes:&Kc length:4 atIndex:10]; [g_enc setBytes:&Nc length:4 atIndex:11];
        [g_enc setBytes:&a length:4 atIndex:12]; [g_enc setBytes:&tk length:4 atIndex:13];
        [g_enc setBytes:&C.w1.stride length:4 atIndex:14];
        [g_enc setBytes:&C.w3.stride length:4 atIndex:15];
        [g_enc dispatchThreadgroups:MTLSizeMake((N1+31u)/32u, top_k, tokens)
              threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    }
    {
        [g_enc setComputePipelineState:ctx().psoMoeT];
        [g_enc setBuffer:gu offset:0 atIndex:0];
        setBufAt(topk, slots*sizeof(float), 1);
        [g_enc setBuffer:C.w2.rc offset:C.w2.rc_off atIndex:2];
        [g_enc setBuffer:C.w2.ls offset:C.w2.ls_off atIndex:3];
        [g_enc setBuffer:C.w2.rs offset:C.w2.rs_off atIndex:4];
        [g_enc setBuffer:C.w2.pm offset:C.w2.pm_off atIndex:5];
        [g_enc setBuffer:code2 offset:0 atIndex:6];
        uint32_t Kc=K2, kv=N1, xz=top_k*N1, xy=N1, tk=top_k;
        [g_enc setBytes:&Kc length:4 atIndex:7]; [g_enc setBytes:&kv length:4 atIndex:8];
        [g_enc setBytes:&xz length:4 atIndex:9]; [g_enc setBytes:&xy length:4 atIndex:10];
        [g_enc setBytes:&tk length:4 atIndex:11];
        [g_enc setBytes:&C.w2.stride length:4 atIndex:12];
        [g_enc setThreadgroupMemoryLength:128*sizeof(float) atIndex:0];
        [g_enc dispatchThreadgroups:MTLSizeMake(K2/128u, top_k, tokens)
              threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    }
    {
        [g_enc setComputePipelineState:ctx().psoMoeDownAcc];
        [g_enc setBuffer:code2 offset:0 atIndex:0];
        setBufAt(topk, slots*sizeof(float), 1);
        setBufAt(probs, (size_t)tokens*num_experts*2, 2);
        [g_enc setBuffer:C.w2.pk offset:C.w2.pk_off atIndex:3]; [g_enc setBuffer:C.w2.nm offset:C.w2.nm_off atIndex:4]; [g_enc setBuffer:C.w2.cb offset:C.w2.cb_off atIndex:5];
        setBufAt(out, (size_t)tokens*N2*2, 6);
        uint32_t Kc=K2, Nc=N2, tk=top_k, nz=normalize, Ec=num_experts;
        [g_enc setBytes:&Kc length:4 atIndex:7]; [g_enc setBytes:&Nc length:4 atIndex:8];
        [g_enc setBytes:&tk length:4 atIndex:9]; [g_enc setBytes:&nz length:4 atIndex:10];
        [g_enc setBytes:&eps length:4 atIndex:11]; [g_enc setBytes:&scaling length:4 atIndex:12];
        [g_enc setBytes:&Ec length:4 atIndex:13];
        [g_enc setBytes:&C.w2.stride length:4 atIndex:14];
        [g_enc setThreadgroupMemoryLength:top_k*16*sizeof(float) atIndex:0];
        [g_enc dispatchThreadgroups:MTLSizeMake((N2+15u)/16u, 1, tokens)
              threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    }
    return true;
}

bool cactus_metal_encode_deltanet_decode(void* out, const void* q, const void* k, const void* v,
                                         const void* g, const void* b, const void* s,
                                         uint32_t B, uint32_t Hq, uint32_t Hv,
                                         uint32_t K, uint32_t V, float scale) {
    if (!ctx().ok || !ctx().psoDeltanet) return false;
    if (B == 0 || Hq == 0 || Hv == 0 || K == 0 || V == 0 || V > 1024 || (Hv % Hq) != 0) return false;
    ensureEncoder();
    [g_enc setComputePipelineState:ctx().psoDeltanet];
    setBufAt(q, (size_t)B*Hq*K*2, 0);
    setBufAt(k, (size_t)B*Hq*K*2, 1);
    setBufAt(v, (size_t)B*Hv*V*2, 2);
    setBufAt(g, (size_t)B*Hv*2, 3);
    setBufAt(b, (size_t)B*Hv*2, 4);
    setBufAt(s, (size_t)B*K*Hv*V*2, 5);
    setBufAt(out, (size_t)B*(1+K)*Hv*V*2, 6);
    [g_enc setBytes:&Hq length:4 atIndex:7]; [g_enc setBytes:&Hv length:4 atIndex:8];
    [g_enc setBytes:&K length:4 atIndex:9]; [g_enc setBytes:&V length:4 atIndex:10];
    [g_enc setBytes:&scale length:4 atIndex:11];
    [g_enc setThreadgroupMemoryLength:(size_t)K*4 atIndex:0];
    [g_enc dispatchThreadgroups:MTLSizeMake(Hv, B, 1) threadsPerThreadgroup:MTLSizeMake(V, 1, 1)];
    return true;
}

bool cactus_metal_encode_deltanet_prefill(void* out, const void* q, const void* k, const void* v,
                                          const void* g, const void* b, const void* s,
                                          uint32_t B, uint32_t T, uint32_t Hq, uint32_t Hv,
                                          uint32_t K, uint32_t V, float scale) {
    if (!ctx().ok || !ctx().psoDeltanetPre) return false;
    if (B == 0 || T == 0 || Hq == 0 || Hv == 0 || K == 0 || V == 0 || V > 1024 || (Hv % Hq) != 0) return false;
    ensureEncoder();
    id<MTLBuffer> scratch = recycled((size_t)B*Hv*K*V*sizeof(float));
    [g_enc setComputePipelineState:ctx().psoDeltanetPre];
    setBufAt(q, (size_t)B*T*Hq*K*2, 0);
    setBufAt(k, (size_t)B*T*Hq*K*2, 1);
    setBufAt(v, (size_t)B*T*Hv*V*2, 2);
    setBufAt(g, (size_t)B*T*Hv*2, 3);
    setBufAt(b, (size_t)B*T*Hv*2, 4);
    setBufAt(s, (size_t)B*K*Hv*V*2, 5);
    setBufAt(out, (size_t)B*(T+K)*Hv*V*2, 6);
    [g_enc setBuffer:scratch offset:0 atIndex:7];
    [g_enc setBytes:&T length:4 atIndex:8]; [g_enc setBytes:&Hq length:4 atIndex:9];
    [g_enc setBytes:&Hv length:4 atIndex:10]; [g_enc setBytes:&K length:4 atIndex:11];
    [g_enc setBytes:&V length:4 atIndex:12]; [g_enc setBytes:&scale length:4 atIndex:13];
    [g_enc setThreadgroupMemoryLength:(size_t)K*4 atIndex:0];
    [g_enc dispatchThreadgroups:MTLSizeMake(Hv, B, 1) threadsPerThreadgroup:MTLSizeMake(V, 1, 1)];
    return true;
}

bool cactus_metal_encode_rms2_add_clip(void* out, const void* a, const void* wa,
                                       const void* b, const void* wb, size_t dim,
                                       float eps_a, float eps_b) {
    if (!ctx().ok || !ctx().psoRms2AddClip || dim == 0) return false;
    ensureEncoder();
    [g_enc setComputePipelineState:ctx().psoRms2AddClip];
    setBufAt(a, dim*2, 0); setBufAt(wa, dim*2, 1);
    setBufAt(b, dim*2, 2); setBufAt(wb, dim*2, 3);
    setBufAt(out, dim*2, 4);
    uint32_t d=(uint32_t)dim; float ea=eps_a, eb=eps_b;
    [g_enc setBytes:&d length:4 atIndex:5]; [g_enc setBytes:&ea length:4 atIndex:6];
    [g_enc setBytes:&eb length:4 atIndex:7];
    [g_enc setThreadgroupMemoryLength:512*sizeof(float) atIndex:0];
    [g_enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    return true;
}

bool cactus_metal_encode_rms_norm_scale(void* out, const void* in, const void* weight,
                                        size_t rows, size_t dim, float eps, float oscale) {
    if (!ctx().ok || !ctx().psoRmsScale || rows == 0 || dim == 0) return false;
    ensureEncoder();
    [g_enc setComputePipelineState:ctx().psoRmsScale];
    setBufAt(in, rows*dim*2, 0); setBufAt(weight, dim*2, 1); setBufAt(out, rows*dim*2, 2);
    uint32_t d=(uint32_t)dim; float e=eps, sc=oscale;
    [g_enc setBytes:&d length:4 atIndex:3]; [g_enc setBytes:&e length:4 atIndex:4];
    [g_enc setBytes:&sc length:4 atIndex:5];
    [g_enc setThreadgroupMemoryLength:256*sizeof(float) atIndex:0];
    [g_enc dispatchThreadgroups:MTLSizeMake(rows,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    return true;
}

bool cactus_metal_encode_softmax_topk(void* probs, void* topk, const void* in,
                                      size_t rows, size_t cols, size_t k, float scale) {
    if (!ctx().ok || !ctx().psoSoftmaxTopk || k == 0 || k > 16 || rows == 0 || cols == 0) return false;
    ensureEncoder();
    [g_enc setComputePipelineState:ctx().psoSoftmaxTopk];
    setBufAt(in, rows*cols*2, 0);
    setBufAt(probs, rows*cols*2, 1);
    setBufAt(topk, rows*k*2*sizeof(float), 2);
    uint32_t E=(uint32_t)cols, kc=(uint32_t)k, B=(uint32_t)rows; float sc=scale;
    [g_enc setBytes:&E length:4 atIndex:3];
    [g_enc setBytes:&kc length:4 atIndex:4];
    [g_enc setBytes:&B length:4 atIndex:5];
    [g_enc setBytes:&sc length:4 atIndex:6];
    [g_enc dispatchThreadgroups:MTLSizeMake(rows,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    return true;
}

bool cactus_metal_encode_rope_pair(void* out, const void* x, const void* c, const void* s,
                                   uint32_t H, uint32_t D) {
    if (!ctx().ok || !ctx().psoRopePair || H == 0 || D == 0 || (D % 2) != 0) return false;
    ensureEncoder();
    [g_enc setComputePipelineState:ctx().psoRopePair];
    setBufAt(x, (size_t)H*D*2, 0);
    setBufAt(c, (size_t)D*2, 1);
    setBufAt(s, (size_t)D*2, 2);
    setBufAt(out, (size_t)H*D*2, 3);
    [g_enc setBytes:&H length:4 atIndex:4];
    [g_enc setBytes:&D length:4 atIndex:5];
    [g_enc dispatchThreads:MTLSizeMake(D, H, 1) threadsPerThreadgroup:MTLSizeMake(64, 4, 1)];
    return true;
}

bool cactus_metal_encode_rope_pair_rms(void* out, const void* x, const void* w,
                                       const void* c, const void* s,
                                       uint32_t H, uint32_t D, float eps) {
    if (!ctx().ok || !ctx().psoRopePairRms || H == 0 || D == 0 || (D % 2) != 0 || D > 1024) return false;
    ensureEncoder();
    [g_enc setComputePipelineState:ctx().psoRopePairRms];
    setBufAt(x, (size_t)H*D*2, 0);
    setBufAt(w, (size_t)D*2, 1);
    setBufAt(c, (size_t)D*2, 2);
    setBufAt(s, (size_t)D*2, 3);
    setBufAt(out, (size_t)H*D*2, 4);
    float e = eps;
    [g_enc setBytes:&D length:4 atIndex:5];
    [g_enc setBytes:&e length:4 atIndex:6];
    uint32_t T = D < 256 ? D : 256;
    [g_enc setThreadgroupMemoryLength:T*sizeof(float) atIndex:0];
    [g_enc setThreadgroupMemoryLength:(size_t)D*2 atIndex:1];
    [g_enc dispatchThreadgroups:MTLSizeMake(H,1,1) threadsPerThreadgroup:MTLSizeMake(T,1,1)];
    return true;
}

bool cactus_metal_encode_topk_rows(void* out, const void* in, size_t rows, size_t cols, size_t k) {
    if (!ctx().ok || !ctx().psoTopkRows || k == 0 || k > 16 || rows == 0 || cols == 0) return false;
    ensureEncoder();
    [g_enc setComputePipelineState:ctx().psoTopkRows];
    setBufAt(in, rows*cols*2, 0);
    setBufAt(out, rows*k*2*sizeof(float), 1);
    uint32_t F=(uint32_t)cols, kc=(uint32_t)k, B=(uint32_t)rows;
    [g_enc setBytes:&F length:4 atIndex:2];
    [g_enc setBytes:&kc length:4 atIndex:3];
    [g_enc setBytes:&B length:4 atIndex:4];
    [g_enc dispatchThreadgroups:MTLSizeMake(rows,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    return true;
}

bool cactus_metal_encode_quant_matmul_m(void* out, const void* lhs, const CactusQuantMatrix* W, uint32_t M) {
    if (!ctx().ok || !W) return false;
    if (M == 1 && (W->flags & CACTUS_QUANT_FLAG_ORTHOGONAL))
        return cactus_metal_encode_quant_matmul(out, lhs, W);

    const uint32_t gs=W->group_size, K=W->K, N=W->N, ng=W->num_groups;
    const uint32_t bits=W->bits, pgb=(gs*bits+7u)/8u;
    const bool fast = ctx().ok && quant_fast_eligible(W);
    if (!fast || !ctx().psoGmma) return false;
    ResW& rw = resident(W);
    if (!rw.ok) return false;
    size_t code_bytes = (size_t)M*ng*gs*sizeof(__fp16);
    if (!g_code_buf_m || (size_t)g_code_buf_m.length < code_bytes)
        g_code_buf_m = owned_shared(code_bytes);
    if (!g_code_buf_m) return false;
    ensureEncoder();

    [g_enc setComputePipelineState:ctx().psoTm];
    setBufAt(lhs, (size_t)M*K*2, 0);               [g_enc setBuffer:rw.recip offset:rw.rc_off atIndex:1];
    [g_enc setBuffer:rw.lsign offset:rw.ls_off atIndex:2]; [g_enc setBuffer:rw.rsign offset:rw.rs_off atIndex:3];
    [g_enc setBuffer:rw.perm offset:rw.pm_off atIndex:4];  [g_enc setBuffer:g_code_buf_m offset:0 atIndex:5];
    [g_enc setBytes:&gs length:4 atIndex:6]; [g_enc setBytes:&K length:4 atIndex:7];
    [g_enc setThreadgroupMemoryLength:gs*sizeof(float) atIndex:0];
    [g_enc dispatchThreadgroups:MTLSizeMake((size_t)ng*M,1,1) threadsPerThreadgroup:MTLSizeMake(gs>1024u?1024u:gs,1,1)];

    [g_enc setComputePipelineState:ctx().psoGmma];
    [g_enc setBuffer:g_code_buf_m offset:0 atIndex:0]; [g_enc setBuffer:rw.packed offset:rw.packed_off atIndex:1];
    [g_enc setBuffer:rw.codebook offset:rw.cb_off atIndex:2]; [g_enc setBuffer:rw.norms offset:rw.norms_off atIndex:3];
    setBufAt(out, (size_t)M*N*2, 4);
    [g_enc setBytes:&gs length:4 atIndex:5]; [g_enc setBytes:&ng length:4 atIndex:6];
    [g_enc setBytes:&pgb length:4 atIndex:7]; [g_enc setBytes:&N length:4 atIndex:8]; [g_enc setBytes:&M length:4 atIndex:9];
    [g_enc dispatchThreadgroups:MTLSizeMake((M+31)/32,(N+63)/64,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
    return true;
}

bool cactus_metal_encode_quant_matmul_ortho(void* out, const void* act, void* code,
                                            const CactusQuantMatrix* W) {
    if (!ctx().ok || !W->rotation) return false;
    const uint32_t K=W->K, N=W->N, ng=W->num_groups, gs=W->group_size, bits=W->bits;
    const uint32_t pgb=(gs*bits+7u)/8u;
    const bool lowbit = (bits == 2 || bits == 3);
    if (bits != 4 && !lowbit) return false;
    if (lowbit && (!ctx().psoGmrLow || (N % 16) != 0)) return false;
    if (ng != 1 || gs != K || (N % 4) != 0) return false;
    ResW& rw = resident(W);
    if (!rw.ok || !rw.rotation || !rw.packed) return false;
    ensureEncoder();

    [g_enc setComputePipelineState:ctx().psoRotW];
    setBufAt(act, (size_t)K*2, 0); [g_enc setBuffer:rw.recip offset:rw.rc_off atIndex:1];
    [g_enc setBuffer:rw.rotation offset:rw.rot_off atIndex:2]; setBufAt(code, (size_t)K*2, 3);
    [g_enc setBytes:&K length:4 atIndex:4];
    [g_enc setThreadgroupMemoryLength:256*sizeof(float) atIndex:0];
    [g_enc dispatchThreadgroups:MTLSizeMake((K+31)/32,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];

    bool umr = ctx().psoGmr && (N % g_mr_rows == 0u);
    [g_enc setComputePipelineState:(lowbit?ctx().psoGmrLow:(umr?ctx().psoGmr:ctx().psoG))];
    setBufAt(code, (size_t)K*2, 0); [g_enc setBuffer:rw.packed offset:rw.packed_off atIndex:1];
    [g_enc setBuffer:rw.codebook offset:rw.cb_off atIndex:2]; [g_enc setBuffer:rw.norms offset:rw.norms_off atIndex:3];
    setBufAt(out, (size_t)N*2, 4);
    [g_enc setBytes:&gs length:4 atIndex:5]; [g_enc setBytes:&ng length:4 atIndex:6];
    [g_enc setBytes:&pgb length:4 atIndex:7]; [g_enc setBytes:&N length:4 atIndex:8];
    if (lowbit) { [g_enc setBytes:&bits length:4 atIndex:9]; [g_enc setBytes:&rw.il length:4 atIndex:10]; }
    else [g_enc setBytes:&rw.il length:4 atIndex:9];
    uint32_t ROWS=8;
    uint32_t grid = (lowbit || umr) ? (N+g_mr_rows-1u)/g_mr_rows : (N+ROWS-1u)/ROWS;
    [g_enc dispatchThreadgroups:MTLSizeMake(grid,1,1) threadsPerThreadgroup:MTLSizeMake(ROWS*32,1,1)];
    return true;
}

bool cactus_metal_encode_embedding_ortho(void* out, uint32_t row, const CactusQuantMatrix* W, float scale) {
    if (!ctx().ok || !W->rotation || (W->bits != 4 && W->bits != 2 && W->bits != 3)) return false;
    const uint32_t K=W->K, ng=W->num_groups, gs=W->group_size, bits=W->bits;
    if (ng != 1 || gs != K) return false;
    if (!tg_mem_ok((size_t)K*sizeof(float))) return false;
    if (bits != 4 && !(ctx().psoEmbOW && (K % 8u) == 0u)) return false;
    if (!ctx().psoEmbOW || (K % 8u) != 0u) return false;
    ResW& rw = resident(W);
    if (!rw.ok || !rw.packed || !rw.rotation || !rw.codebook || !rw.norms || !rw.recip) return false;
    ensureEncoder();
    {
        [g_enc setComputePipelineState:ctx().psoEmbOW];
        [g_enc setBuffer:rw.packed offset:rw.packed_off atIndex:0]; [g_enc setBuffer:rw.codebook offset:rw.cb_off atIndex:1];
        [g_enc setBuffer:rw.norms offset:rw.norms_off atIndex:2]; [g_enc setBuffer:rw.recip offset:rw.rc_off atIndex:3];
        [g_enc setBuffer:rw.rotation offset:rw.rot_off atIndex:4]; setBufAt(out, (size_t)K*2, 5);
        [g_enc setBytes:&K length:4 atIndex:6]; [g_enc setBytes:&row length:4 atIndex:7];
        [g_enc setBytes:&scale length:4 atIndex:8]; [g_enc setBytes:&bits length:4 atIndex:9];
        [g_enc setBytes:&rw.il length:4 atIndex:10];
        [g_enc setThreadgroupMemoryLength:(size_t)K*sizeof(float) atIndex:0];
        [g_enc dispatchThreadgroups:MTLSizeMake(K/8u,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        return true;
    }
}

ResW& resident_emb_meta(const CactusQuantMatrix* W) {
    std::lock_guard<std::mutex> lk(g_resident_mu);
    auto it = g_resident_emb.find(W->packed_indices);
    if (it != g_resident_emb.end()) return it->second;
    const uint32_t K=W->K, gs=W->group_size, ng=W->num_groups, bits=W->bits;
    const uint32_t pgb=(gs*bits+7u)/8u;
    ResW r;
    bind_component(W->codebook, (size_t)(1u<<W->bits)*sizeof(__fp16), 2, r.codebook, r.cb_off);
    bind_component(W->left_signs, gs, 1, r.lsign, r.ls_off);
    bind_component(W->right_signs, gs, 1, r.rsign, r.rs_off);
    bind_component(W->permutation, (size_t)gs*sizeof(uint32_t), 4, r.perm, r.pm_off);
    bind_component(W->input_scale_recip, (size_t)K*sizeof(__fp16), 2, r.recip, r.rc_off);
    if (W->N && ng) {
        auto wp = wrapHostPtr(W->packed_indices, (size_t)W->N*ng*pgb);
        if (wp.first && ((uintptr_t)W->norms & 1u) == 0) {
            auto wn = wrapHostPtr(W->norms, (size_t)W->N*ng*sizeof(__fp16));
            if (wn.first) {
                r.packed = wp.first; r.packed_off = wp.second;
                r.norms  = wn.first; r.norms_off  = wn.second;
            }
        }
    }
    return g_resident_emb.emplace(W->packed_indices, r).first->second;
}

bool cactus_metal_encode_embedding_hadamard(void* out, uint32_t row, const CactusQuantMatrix* W) {
    if (!ctx().ok || (W->bits != 4 && W->bits != 2 && W->bits != 3) ||
        (W->flags & (CACTUS_QUANT_FLAG_ORTHOGONAL | CACTUS_QUANT_FLAG_INTERLEAVED_4ROW))) return false;
    const uint32_t K=W->K, gs=W->group_size, ng=W->num_groups, bits=W->bits, pgb=(gs*bits+7u)/8u;
    if (gs > 256 || (gs & (gs-1)) != 0 || !W->packed_indices || !W->norms || !W->codebook
        || !W->left_signs || !W->right_signs || !W->permutation || !W->input_scale_recip) return false;
    ResW& rm = resident_emb_meta(W);
    if (!rm.codebook || !rm.recip || !rm.lsign || !rm.rsign || !rm.perm) return false;
    ensureEncoder();
    size_t row_bytes = (size_t)ng*pgb;
    if (!rm.packed || !rm.norms) return false;
    [g_enc setComputePipelineState:ctx().psoEmbH];
    [g_enc setBuffer:rm.packed offset:rm.packed_off + (size_t)row*row_bytes atIndex:0];
    [g_enc setBuffer:rm.norms offset:rm.norms_off + (size_t)row*ng*sizeof(__fp16) atIndex:2];
    [g_enc setBuffer:rm.codebook offset:rm.cb_off atIndex:1];
    [g_enc setBuffer:rm.recip offset:rm.rc_off atIndex:3];
    [g_enc setBuffer:rm.lsign offset:rm.ls_off atIndex:4]; [g_enc setBuffer:rm.rsign offset:rm.rs_off atIndex:5];
    [g_enc setBuffer:rm.perm offset:rm.pm_off atIndex:6]; setBufAt(out, (size_t)K*2, 7);
    [g_enc setBytes:&gs length:4 atIndex:8]; [g_enc setBytes:&bits length:4 atIndex:9];
    [g_enc setThreadgroupMemoryLength:(size_t)gs*sizeof(float) atIndex:0];
    [g_enc dispatchThreadgroups:MTLSizeMake(ng,1,1) threadsPerThreadgroup:MTLSizeMake(gs,1,1)];
    return true;
}

static id<MTLBuffer> registerReadonly(const void* p, size_t bytes) {
    if (!p) return nil;
    auto it = g_readonly.find(p);
    if (it != g_readonly.end()) return it->second;
    uintptr_t a = (uintptr_t)p, base = a & ~(uintptr_t)16383u;
    size_t wraplen = (((a - base) + bytes + 16383u) & ~(size_t)16383u);
    id<MTLBuffer> b = [ctx().dev newBufferWithBytesNoCopy:(void*)base length:wraplen
                       options:MTLResourceStorageModeShared deallocator:nil];
    if (b) g_readonly[p] = b;
    return b;
}

bool cactus_metal_encode_embedding_ortho_m(void* out, const CactusQuantMatrix* W, const uint32_t* rows, uint32_t M) {
    if (!ctx().ok || !ctx().psoEmbOm || !W->rotation || (W->bits != 4 && W->bits != 2 && W->bits != 3)) return false;
    const uint32_t K=W->K, ng=W->num_groups, gs=W->group_size, bits=W->bits;
    if (ng != 1 || gs != K) return false;
    ResW& rw = resident(W);
    if (!rw.packed || !rw.rotation || !rw.codebook || !rw.norms || !rw.recip) return false;
    ensureEncoder();
    id<MTLBuffer> rb = recycled((size_t)M*sizeof(uint32_t));
    std::memcpy([rb contents], rows, (size_t)M*sizeof(uint32_t));
    [g_enc setComputePipelineState:ctx().psoEmbOm];
    [g_enc setBuffer:rw.packed offset:rw.packed_off atIndex:0]; [g_enc setBuffer:rw.codebook offset:rw.cb_off atIndex:1];
    [g_enc setBuffer:rw.norms offset:rw.norms_off atIndex:2]; [g_enc setBuffer:rw.recip offset:rw.rc_off atIndex:3];
    [g_enc setBuffer:rw.rotation offset:rw.rot_off atIndex:4]; [g_enc setBuffer:rb offset:0 atIndex:5];
    setBufAt(out, (size_t)M*K*2, 6); [g_enc setBytes:&K length:4 atIndex:7]; [g_enc setBytes:&M length:4 atIndex:8];
    [g_enc setBytes:&bits length:4 atIndex:9];
    [g_enc setBytes:&rw.il length:4 atIndex:10];
    [g_enc dispatchThreadgroups:MTLSizeMake(((K+15)/16)*((M+15)/16),1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    return true;
}

bool cactus_metal_encode_embedding_hadamard_m(void* out, const CactusQuantMatrix* W, const uint32_t* rows, uint32_t M) {
    if (!ctx().ok || !ctx().psoEmbHm || (W->bits != 4 && W->bits != 2 && W->bits != 3) ||
        (W->flags & (CACTUS_QUANT_FLAG_ORTHOGONAL | CACTUS_QUANT_FLAG_INTERLEAVED_4ROW))) return false;
    const uint32_t K=W->K, gs=W->group_size, ng=W->num_groups, bits=W->bits, pgb=(gs*bits+7u)/8u;
    if (gs > 256 || (gs & (gs-1)) != 0 || !W->packed_indices || !W->norms || !W->codebook
        || !W->left_signs || !W->right_signs || !W->permutation || !W->input_scale_recip) return false;
    ResW& rm = resident_emb_meta(W);
    if (!rm.codebook || !rm.recip || !rm.lsign || !rm.rsign || !rm.perm) return false;
    ensureEncoder();
    size_t rowb = (size_t)ng*pgb;
    id<MTLBuffer> pk = recycled((size_t)M*rowb);
    id<MTLBuffer> nm = recycled((size_t)M*ng*sizeof(__fp16));
    for (uint32_t m=0; m<M; ++m) {
        std::memcpy((uint8_t*)[pk contents]+(size_t)m*rowb, (const uint8_t*)W->packed_indices+(size_t)rows[m]*rowb, rowb);
        std::memcpy((__fp16*)[nm contents]+(size_t)m*ng, (const __fp16*)W->norms+(size_t)rows[m]*ng, (size_t)ng*sizeof(__fp16));
    }
    [g_enc setComputePipelineState:ctx().psoEmbHm];
    [g_enc setBuffer:pk offset:0 atIndex:0]; [g_enc setBuffer:rm.codebook offset:rm.cb_off atIndex:1];
    [g_enc setBuffer:nm offset:0 atIndex:2]; [g_enc setBuffer:rm.recip offset:rm.rc_off atIndex:3];
    [g_enc setBuffer:rm.lsign offset:rm.ls_off atIndex:4]; [g_enc setBuffer:rm.rsign offset:rm.rs_off atIndex:5];
    [g_enc setBuffer:rm.perm offset:rm.pm_off atIndex:6]; setBufAt(out, (size_t)M*K*2, 7);
    [g_enc setBytes:&gs length:4 atIndex:8]; [g_enc setBytes:&ng length:4 atIndex:9]; [g_enc setBytes:&K length:4 atIndex:10];
    [g_enc setBytes:&bits length:4 atIndex:11];
    [g_enc setThreadgroupMemoryLength:(size_t)gs*sizeof(float) atIndex:0];
    [g_enc dispatchThreadgroups:MTLSizeMake((size_t)ng*M,1,1) threadsPerThreadgroup:MTLSizeMake(gs,1,1)];
    return true;
}


bool cactus_metal_encode_gather_f16(void* out, const void* table, size_t table_bytes,
                                    const uint32_t* rows, uint32_t M, uint32_t D) {
    if (!ctx().ok || !ctx().psoGather || M == 0 || D == 0) return false;
    id<MTLBuffer> tb = registerReadonly(table, table_bytes);
    if (!tb) return false;
    size_t toff = (uintptr_t)table & (uintptr_t)16383u;
    ensureEncoder();
    id<MTLBuffer> rb = recycled((size_t)M*sizeof(uint32_t));
    std::memcpy([rb contents], rows, (size_t)M*sizeof(uint32_t));
    uint32_t n = M*D;
    [g_enc setComputePipelineState:ctx().psoGather];
    [g_enc setBuffer:tb offset:toff atIndex:0]; [g_enc setBuffer:rb offset:0 atIndex:1];
    setBufAt(out, (size_t)n*2, 2);
    [g_enc setBytes:&D length:4 atIndex:3]; [g_enc setBytes:&n length:4 atIndex:4];
    [g_enc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    return true;
}

static void setCacheAt(const void* p, size_t bytes, int idx) {
    if (p && bytes) setBufAt(p, bytes, idx);
    else [g_enc setBuffer:ctx().dummy offset:0 atIndex:idx];
}

bool cactus_metal_encode_attention_i8(
    void* out, const void* q, const void* knew, const void* vnew,
    const void* kc, const void* vc, const void* ks, const void* vs,
    uint32_t num_q_heads, uint32_t num_kv_heads, uint32_t head_dim, uint32_t v_hdim,
    uint32_t history_len, uint32_t total_keys, uint32_t kv_start, uint32_t kv_end,
    float scale, size_t kc_bytes, size_t vc_bytes, size_t ks_bytes, size_t vs_bytes) {
    if (!ctx().ok) return false;
    if (kv_end <= kv_start || (num_q_heads % num_kv_heads) != 0) return false;
    if (head_dim > 512u || v_hdim > 512u) return false;
    ensureEncoder();
    auto setInputs = [&]() {
        setBufAt(q, (size_t)num_q_heads*head_dim*2, 0);
        if (total_keys > history_len && knew && vnew) {
            setBufAt(knew, (size_t)num_kv_heads*head_dim*2, 1);
            setBufAt(vnew, (size_t)num_kv_heads*v_hdim*2, 2);
        } else {
            [g_enc setBuffer:ctx().dummy offset:0 atIndex:1]; [g_enc setBuffer:ctx().dummy offset:0 atIndex:2];
        }
        setCacheAt(kc, kc_bytes, 3); setCacheAt(vc, vc_bytes, 4);
        setCacheAt(ks, ks_bytes, 5); setCacheAt(vs, vs_bytes, 6);
    };

    const uint32_t T = 256, nsg = T / 32u;
    const uint32_t R = kv_end - kv_start;
    uint32_t nwg = R / 24u; if (nwg < 1u) nwg = 1u; if (nwg > 32u) nwg = 32u;
    id<MTLBuffer> partO = ctx().dummy, partML = ctx().dummy;
    if (nwg > 1u) {
        partO  = recycled((size_t)num_q_heads*nwg*v_hdim*sizeof(float));
        partML = recycled((size_t)num_q_heads*nwg*2*sizeof(float));
    }
    [g_enc setComputePipelineState:ctx().psoAttn];
    setInputs();
    setBufAt(out, (size_t)num_q_heads*v_hdim*2, 7);
    [g_enc setBytes:&num_q_heads length:4 atIndex:8]; [g_enc setBytes:&num_kv_heads length:4 atIndex:9];
    [g_enc setBytes:&head_dim length:4 atIndex:10];   [g_enc setBytes:&v_hdim length:4 atIndex:11];
    [g_enc setBytes:&history_len length:4 atIndex:12];[g_enc setBytes:&scale length:4 atIndex:13];
    [g_enc setBytes:&kv_start length:4 atIndex:14];   [g_enc setBytes:&kv_end length:4 atIndex:15];
    [g_enc setBuffer:partO offset:0 atIndex:16];      [g_enc setBuffer:partML offset:0 atIndex:17];
    [g_enc setBytes:&nwg length:4 atIndex:18];
    [g_enc setThreadgroupMemoryLength:((size_t)nsg*v_hdim + 2*nsg)*sizeof(float) atIndex:0];
    [g_enc dispatchThreadgroups:MTLSizeMake(num_q_heads*nwg,1,1) threadsPerThreadgroup:MTLSizeMake(T,1,1)];
    if (nwg > 1u) {
        [g_enc setComputePipelineState:ctx().psoAttnC];
        [g_enc setBuffer:partO offset:0 atIndex:0];  [g_enc setBuffer:partML offset:0 atIndex:1];
        setBufAt(out, (size_t)num_q_heads*v_hdim*2, 2);
        [g_enc setBytes:&v_hdim length:4 atIndex:3]; [g_enc setBytes:&nwg length:4 atIndex:4];
        [g_enc dispatchThreadgroups:MTLSizeMake(num_q_heads,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    }
    return true;
}

bool cactus_metal_encode_attention_fused_i8(
    void* out, const void* q, const void* kraw, const void* vraw,
    void* kc, void* vc, void* ks, void* vs,
    const void* qw, const void* kw, const void* vw, const void* cs, const void* sn,
    uint32_t nqh, uint32_t hd, uint32_t vhd,
    uint32_t kv_start, uint32_t kv_end, uint32_t slot, uint32_t has_new,
    float eps, float scale,
    size_t kc_bytes, size_t vc_bytes, size_t ks_bytes, size_t vs_bytes) {
    if (!ctx().ok || !ctx().psoAttnF) return false;
    if (kv_end <= kv_start) return false;
    ensureEncoder();
    const uint32_t T = 256, nsg = T / 32u;
    const uint32_t R = kv_end - kv_start;
    uint32_t nwg = R / 24u; if (nwg < 1u) nwg = 1u; if (nwg > 32u) nwg = 32u;
    id<MTLBuffer> partO = ctx().dummy, partML = ctx().dummy;
    if (nwg > 1u) {
        partO  = recycled((size_t)nqh*nwg*vhd*sizeof(float));
        partML = recycled((size_t)nqh*nwg*2*sizeof(float));
    }
    struct { uint32_t nqh, nkvh, hd, vhd, hist, kv_start, kv_end, nwg, slot, has_new; float eps, scale; } U =
        { nqh, 1u, hd, vhd, 0u, kv_start, kv_end, nwg, slot, has_new, eps, scale };
    [g_enc setComputePipelineState:ctx().psoAttnF];
    setBufAt(q, (size_t)nqh*hd*2, 0);
    if (has_new && kraw && vraw) { setBufAt(kraw, (size_t)hd*2, 1); setBufAt(vraw, (size_t)vhd*2, 2); }
    else { [g_enc setBuffer:ctx().dummy offset:0 atIndex:1]; [g_enc setBuffer:ctx().dummy offset:0 atIndex:2]; }
    setBufAt(kc, kc_bytes, 3); setBufAt(vc, vc_bytes, 4);
    setBufAt(ks, ks_bytes, 5); setBufAt(vs, vs_bytes, 6);
    setBufAt(out, (size_t)nqh*vhd*2, 7);
    setBufAt(qw, (size_t)hd*2, 8);
    if (has_new && kw) setBufAt(kw, (size_t)hd*2, 9); else [g_enc setBuffer:ctx().dummy offset:0 atIndex:9];
    if (has_new && vw) setBufAt(vw, (size_t)vhd*2, 10); else [g_enc setBuffer:ctx().dummy offset:0 atIndex:10];
    setBufAt(cs, (size_t)hd*2, 11); setBufAt(sn, (size_t)hd*2, 12);
    [g_enc setBuffer:partO offset:0 atIndex:13]; [g_enc setBuffer:partML offset:0 atIndex:14];
    [g_enc setBytes:&U length:sizeof(U) atIndex:15];
    [g_enc setThreadgroupMemoryLength:((size_t)nsg*vhd + 2*nsg + 256)*sizeof(float) atIndex:0];
    [g_enc setThreadgroupMemoryLength:(size_t)(2*hd + vhd)*2 atIndex:1];
    [g_enc dispatchThreadgroups:MTLSizeMake(nqh*nwg,1,1) threadsPerThreadgroup:MTLSizeMake(T,1,1)];
    if (nwg > 1u) {
        [g_enc setComputePipelineState:ctx().psoAttnC];
        [g_enc setBuffer:partO offset:0 atIndex:0];  [g_enc setBuffer:partML offset:0 atIndex:1];
        setBufAt(out, (size_t)nqh*vhd*2, 2);
        [g_enc setBytes:&vhd length:4 atIndex:3]; [g_enc setBytes:&nwg length:4 atIndex:4];
        [g_enc dispatchThreadgroups:MTLSizeMake(nqh,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    }
    return true;
}

bool cactus_metal_encode_attention_i8_prefill(
    void* out, const void* q, const void* knew, const void* vnew,
    const void* kc, const void* vc, const void* ks, const void* vs,
    uint32_t num_q_heads, uint32_t num_kv_heads, uint32_t head_dim, uint32_t v_hdim,
    uint32_t history_len, uint32_t new_len, uint32_t q_pos0, uint32_t window, uint32_t is_causal, uint32_t M,
    float scale, size_t kc_bytes, size_t vc_bytes, size_t ks_bytes, size_t vs_bytes,
    uint32_t sink, uint32_t ring) {
    if (!ctx().ok) return false;
    uint32_t total_keys = history_len + new_len;
    const uint32_t T = 128;
    if (total_keys == 0 || M == 0 || (num_q_heads % num_kv_heads) != 0) return false;
    uint32_t maxsc = (ring > 0u) ? ((total_keys > sink + ring) ? (sink + ring) : total_keys) : 256u;
    if (maxsc > 7936u) return false;
    bool mma2 = (ring == 0u && window == 0u && is_causal && head_dim == 512u && v_hdim == 512u
                 && num_q_heads == 8u && num_kv_heads == 1u);
    bool hd256 = (ctx().psoAttnPreHd256 != nil && is_causal && head_dim == 256u && v_hdim == 256u
                  && num_q_heads == 8u && num_kv_heads == 1u);
    if (mma2 || hd256) {
        ensureEncoder();
        [g_enc setComputePipelineState:(hd256 ? ctx().psoAttnPreHd256 : ctx().psoAttnPreMma2)];
        setBufAt(q, (size_t)M*num_q_heads*head_dim*2, 0);
        if (new_len > 0 && knew && vnew) {
            setBufAt(knew, (size_t)new_len*num_kv_heads*head_dim*2, 1);
            setBufAt(vnew, (size_t)new_len*num_kv_heads*v_hdim*2, 2);
        } else {
            [g_enc setBuffer:ctx().dummy offset:0 atIndex:1]; [g_enc setBuffer:ctx().dummy offset:0 atIndex:2];
        }
        setCacheAt(kc, kc_bytes, 3); setCacheAt(vc, vc_bytes, 4); setCacheAt(ks, ks_bytes, 5); setCacheAt(vs, vs_bytes, 6);
        setBufAt(out, (size_t)M*num_q_heads*v_hdim*2, 7);
        [g_enc setBytes:&num_q_heads length:4 atIndex:8]; [g_enc setBytes:&num_kv_heads length:4 atIndex:9];
        [g_enc setBytes:&head_dim length:4 atIndex:10];   [g_enc setBytes:&v_hdim length:4 atIndex:11];
        [g_enc setBytes:&history_len length:4 atIndex:12];[g_enc setBytes:&scale length:4 atIndex:13];
        [g_enc setBytes:&q_pos0 length:4 atIndex:14];     [g_enc setBytes:&new_len length:4 atIndex:15];
        [g_enc setBytes:&M length:4 atIndex:16];
        if (hd256) { [g_enc setBytes:&sink length:4 atIndex:17]; [g_enc setBytes:&ring length:4 atIndex:18]; }
        uint32_t mtiles = (M + 7u)/8u;
        [g_enc dispatchThreadgroups:MTLSizeMake(mtiles, 1, 1) threadsPerThreadgroup:MTLSizeMake(hd256?256u:512u,1,1)];
        return true;
    }
    ensureEncoder();
    [g_enc setComputePipelineState:ctx().psoAttnPre];
    setBufAt(q, (size_t)M*num_q_heads*head_dim*2, 0);
    if (new_len > 0 && knew && vnew) {
        setBufAt(knew, (size_t)new_len*num_kv_heads*head_dim*2, 1);
        setBufAt(vnew, (size_t)new_len*num_kv_heads*v_hdim*2, 2);
    } else {
        [g_enc setBuffer:ctx().dummy offset:0 atIndex:1]; [g_enc setBuffer:ctx().dummy offset:0 atIndex:2];
    }
    setCacheAt(kc, kc_bytes, 3); setCacheAt(vc, vc_bytes, 4); setCacheAt(ks, ks_bytes, 5); setCacheAt(vs, vs_bytes, 6);
    setBufAt(out, (size_t)M*num_q_heads*v_hdim*2, 7);
    [g_enc setBytes:&num_q_heads length:4 atIndex:8]; [g_enc setBytes:&num_kv_heads length:4 atIndex:9];
    [g_enc setBytes:&head_dim length:4 atIndex:10];   [g_enc setBytes:&v_hdim length:4 atIndex:11];
    [g_enc setBytes:&history_len length:4 atIndex:12];[g_enc setBytes:&scale length:4 atIndex:13];
    [g_enc setBytes:&q_pos0 length:4 atIndex:14];     [g_enc setBytes:&new_len length:4 atIndex:15];
    [g_enc setBytes:&window length:4 atIndex:16];     [g_enc setBytes:&is_causal length:4 atIndex:17];
    [g_enc setBytes:&maxsc length:4 atIndex:18];
    [g_enc setBytes:&sink length:4 atIndex:19]; [g_enc setBytes:&ring length:4 atIndex:20];
    [g_enc setThreadgroupMemoryLength:((size_t)maxsc + T)*sizeof(float) atIndex:0];
    [g_enc dispatchThreadgroups:MTLSizeMake(M*num_q_heads,1,1) threadsPerThreadgroup:MTLSizeMake(T,1,1)];
    return true;
}

bool cactus_metal_encode_binary_f32(int op, void* out, const void* a, const void* b, size_t n) {
    if (!ctx().ok || !ctx().psoBinF32) return false;
    ensureEncoder();
    uint32_t nn=(uint32_t)n; int o=op;
    [g_enc setComputePipelineState:ctx().psoBinF32];
    setBufAt(a, n*4, 0); setBufAt(b, n*4, 1); setBufAt(out, n*4, 2);
    [g_enc setBytes:&nn length:4 atIndex:3]; [g_enc setBytes:&o length:4 atIndex:4];
    [g_enc dispatchThreads:MTLSizeMake((n+3)/4,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    return true;
}
bool cactus_metal_encode_scalar_f32(int op, void* out, const void* in, size_t n, float p) {
    if (!ctx().ok || !ctx().psoScaF32) return false;
    ensureEncoder();
    uint32_t nn=(uint32_t)n; int o=op;
    [g_enc setComputePipelineState:ctx().psoScaF32];
    setBufAt(in, n*4, 0); setBufAt(out, n*4, 1);
    [g_enc setBytes:&nn length:4 atIndex:2]; [g_enc setBytes:&o length:4 atIndex:3]; [g_enc setBytes:&p length:4 atIndex:4];
    [g_enc dispatchThreads:MTLSizeMake((n+3)/4,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    return true;
}
bool cactus_metal_encode_unary_f32(int op, void* out, const void* in, size_t n) {
    if (!ctx().ok || !ctx().psoUnaF32) return false;
    ensureEncoder();
    uint32_t nn=(uint32_t)n; int o=op;
    [g_enc setComputePipelineState:ctx().psoUnaF32];
    setBufAt(in, n*4, 0); setBufAt(out, n*4, 1);
    [g_enc setBytes:&nn length:4 atIndex:2]; [g_enc setBytes:&o length:4 atIndex:3];
    [g_enc dispatchThreads:MTLSizeMake((n+3)/4,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    return true;
}
bool cactus_metal_encode_clamp(void* out, const void* in, size_t n, float lo, float hi, int f32) {
    id<MTLComputePipelineState> pso = f32 ? ctx().psoClampF32 : ctx().psoClampF16;
    if (!ctx().ok || !pso) return false;
    ensureEncoder();
    uint32_t nn=(uint32_t)n; size_t es = f32?4:2;
    [g_enc setComputePipelineState:pso];
    setBufAt(in, n*es, 0); setBufAt(out, n*es, 1);
    [g_enc setBytes:&nn length:4 atIndex:2]; [g_enc setBytes:&lo length:4 atIndex:3]; [g_enc setBytes:&hi length:4 atIndex:4];
    size_t grid = (n + 3) / 4;
    [g_enc dispatchThreads:MTLSizeMake(grid,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    return true;
}
bool cactus_metal_encode_glu(void* out, const void* in, size_t split, size_t inner, size_t n_out) {
    if (!ctx().ok || !ctx().psoGlu) return false;
    ensureEncoder();
    uint32_t sp=(uint32_t)split, in_=(uint32_t)inner, nn=(uint32_t)n_out;
    [g_enc setComputePipelineState:ctx().psoGlu];
    setBufAt(in, n_out*4, 0); setBufAt(out, n_out*2, 1);
    [g_enc setBytes:&sp length:4 atIndex:2]; [g_enc setBytes:&in_ length:4 atIndex:3]; [g_enc setBytes:&nn length:4 atIndex:4];
    [g_enc dispatchThreads:MTLSizeMake(n_out,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    return true;
}
bool cactus_metal_encode_layer_norm(void* out, const void* in, const void* w, const void* b,
                                    size_t rows, size_t dim, float eps) {
    if (!ctx().ok || !ctx().psoLayerNorm) return false;
    ensureEncoder();
    uint32_t d=(uint32_t)dim, hb = b ? 1u : 0u;
    uint32_t T = dim >= 1024 ? 1024u : 256u;
    [g_enc setComputePipelineState:ctx().psoLayerNorm];
    setBufAt(in, rows*dim*2, 0); setBufAt(w, dim*2, 1);
    if (b) setBufAt(b, dim*2, 2); else [g_enc setBuffer:ctx().dummy offset:0 atIndex:2];
    setBufAt(out, rows*dim*2, 3);
    [g_enc setBytes:&d length:4 atIndex:4]; [g_enc setBytes:&eps length:4 atIndex:5]; [g_enc setBytes:&hb length:4 atIndex:6];
    [g_enc setThreadgroupMemoryLength:2*T*sizeof(float) atIndex:0];
    [g_enc dispatchThreadgroups:MTLSizeMake(rows,1,1) threadsPerThreadgroup:MTLSizeMake(T,1,1)];
    return true;
}
bool cactus_metal_encode_softmax_rows(void* out, const void* in, size_t rows, size_t cols) {
    if (!ctx().ok || !ctx().psoSoftmaxR) return false;
    ensureEncoder();
    uint32_t c=(uint32_t)cols;
    uint32_t T = cols >= 1024 ? 1024u : 256u;
    [g_enc setComputePipelineState:ctx().psoSoftmaxR];
    setBufAt(in, rows*cols*2, 0); setBufAt(out, rows*cols*2, 1);
    [g_enc setBytes:&c length:4 atIndex:2];
    [g_enc setThreadgroupMemoryLength:T*sizeof(float) atIndex:0];
    [g_enc dispatchThreadgroups:MTLSizeMake(rows,1,1) threadsPerThreadgroup:MTLSizeMake(T,1,1)];
    return true;
}
bool cactus_metal_encode_conv1d_k3(void* out, const void* x, const void* w, int w_int8,
                                   const void* w_scales, uint32_t w_gs,
                                   uint32_t Cin, uint32_t L, uint32_t Cout, uint32_t Lout, uint32_t stride) {
    if (!ctx().ok || !ctx().psoConv1dK3) return false;
    ensureEncoder();
    id<MTLBuffer> dq = nil;
    if (w_int8) {
        size_t rows = Cout, cols = (size_t)Cin*3;
        dq = recycled(rows*cols*2);
        if (!dq) return false;
        __fp16* dst = (__fp16*)[dq contents];
        const int8_t* src = (const int8_t*)w;
        if (w_gs == 0 || !w_scales) {
            for (size_t i = 0; i < rows*cols; ++i) dst[i] = (__fp16)src[i];
        } else {
            const __fp16* scales = (const __fp16*)w_scales;
            size_t ngr = cols / w_gs;
            for (size_t r = 0; r < rows; ++r)
                for (size_t c = 0; c < cols; ++c)
                    dst[r*cols+c] = (__fp16)((float)src[r*cols+c] * (float)scales[r*ngr + c/w_gs]);
        }
    }
    [g_enc setComputePipelineState:ctx().psoConv1dK3];
    setBufAt(x, (size_t)Cin*L*2, 0);
    if (dq) [g_enc setBuffer:dq offset:0 atIndex:1]; else setBufAt(w, (size_t)Cout*Cin*3*2, 1);
    setBufAt(out, (size_t)Cout*Lout*2, 2);
    [g_enc setBytes:&Cin length:4 atIndex:3]; [g_enc setBytes:&L length:4 atIndex:4];
    [g_enc setBytes:&Lout length:4 atIndex:5]; [g_enc setBytes:&stride length:4 atIndex:6];
    [g_enc dispatchThreads:MTLSizeMake(Lout,Cout,1) threadsPerThreadgroup:MTLSizeMake(64,4,1)];
    return true;
}
static MPSMatrixMultiplication* mps_gemm_ab(uint32_t M, uint32_t N, uint32_t K, bool tr, float alpha, float beta) {
    static std::map<std::tuple<uint32_t,uint32_t,uint32_t,bool,uint64_t>, MPSMatrixMultiplication*> cache;
    uint32_t ab, bb;
    std::memcpy(&ab, &alpha, 4);
    std::memcpy(&bb, &beta, 4);
    auto key = std::make_tuple(M,N,K,tr,((uint64_t)ab<<32)|bb);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    MPSMatrixMultiplication* mm = [[MPSMatrixMultiplication alloc]
        initWithDevice:ctx().dev transposeLeft:false transposeRight:tr
        resultRows:M resultColumns:N interiorColumns:K alpha:alpha beta:beta];
    cache[key] = mm;
    return mm;
}
static MPSMatrixMultiplication* mps_gemm_a(uint32_t M, uint32_t N, uint32_t K, bool tr, float alpha) {
    return mps_gemm_ab(M, N, K, tr, alpha, 0.0f);
}
static MPSMatrixMultiplication* mps_gemm(uint32_t M, uint32_t N, uint32_t K, bool tr) {
    return mps_gemm_a(M, N, K, tr, 1.0f);
}

bool cactus_metal_encode_gemm_f16(void* out, const void* lhs, const void* rhs,
                                  uint32_t M, uint32_t K, uint32_t N, int pretransposed) {
    if (!ctx().ok) return false;
    static const bool mps_ok = MPSSupportsMTLDevice(ctx().dev);
    auto a = bufForPtrOff(lhs, (size_t)M*K*2);
    auto b = bufForPtrOff(rhs, (size_t)N*K*2);
    auto c = bufForPtrOff(out, (size_t)M*N*2);
    int use_tr = pretransposed;
    if (mps_ok && (a.second % 16)==0 && (b.second % 16)==0 && (c.second % 16)==0
        && (K % 8)==0 && (N % 8)==0) {
        if (g_enc) { [g_enc endEncoding]; g_enc = nil; }
        if (!g_cmd) g_cmd = [ctx().queue commandBuffer];
        MPSMatrixDescriptor* da = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:K
            rowBytes:(size_t)K*2 dataType:MPSDataTypeFloat16];
        MPSMatrixDescriptor* db = use_tr
            ? [MPSMatrixDescriptor matrixDescriptorWithRows:N columns:K rowBytes:(size_t)K*2 dataType:MPSDataTypeFloat16]
            : [MPSMatrixDescriptor matrixDescriptorWithRows:K columns:N rowBytes:(size_t)N*2 dataType:MPSDataTypeFloat16];
        MPSMatrixDescriptor* dc = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N
            rowBytes:(size_t)N*2 dataType:MPSDataTypeFloat16];
        auto mat = [](id<MTLBuffer> buf, size_t off, MPSMatrixDescriptor* d) -> MPSMatrix* {
            auto key = std::make_tuple((__bridge void*)buf, off, (size_t)d.rows, (size_t)d.columns);
            auto it = g_mps_mats.find(key);
            if (it != g_mps_mats.end()) return it->second;
            if (g_mps_mats.size() > 4096) g_mps_mats.clear();
            MPSMatrix* m = [[MPSMatrix alloc] initWithBuffer:buf offset:off descriptor:d];
            g_mps_mats[key] = m;
            return m;
        };
        MPSMatrix* ma = mat(a.first, a.second, da);
        MPSMatrix* mb = mat(b.first, b.second, db);
        MPSMatrix* mc = mat(c.first, c.second, dc);
        [mps_gemm(M, N, K, use_tr != 0) encodeToCommandBuffer:g_cmd
            leftMatrix:ma rightMatrix:mb resultMatrix:mc];
        return true;
    }
    if (!ctx().psoGdense || !ctx().psoStrided) return false;
    ensureEncoder();
    id<MTLBuffer> tbuf = nil;
    if (!pretransposed) {
        tbuf = recycled((size_t)K*N*2);
        if (!tbuf) return false;
        uint32_t oshape[2] = { N, K }, sstride[2] = { 1u, N };
        uint32_t total = N*K;
        [g_enc setComputePipelineState:ctx().psoStrided];
        setBufAt(rhs, (size_t)K*N*2, 0); [g_enc setBuffer:tbuf offset:0 atIndex:1];
        [g_enc setBytes:oshape length:8 atIndex:2]; [g_enc setBytes:sstride length:8 atIndex:3];
        uint32_t nd=2, base=0;
        [g_enc setBytes:&nd length:4 atIndex:4]; [g_enc setBytes:&total length:4 atIndex:5]; [g_enc setBytes:&base length:4 atIndex:6];
        [g_enc dispatchThreads:MTLSizeMake(total,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    }
    [g_enc setComputePipelineState:ctx().psoGdense];
    setBufAt(lhs, (size_t)M*K*2, 0);
    if (tbuf) [g_enc setBuffer:tbuf offset:0 atIndex:1]; else setBufAt(rhs, (size_t)N*K*2, 1);
    setBufAt(out, (size_t)M*N*2, 2);
    [g_enc setBytes:&K length:4 atIndex:3]; [g_enc setBytes:&N length:4 atIndex:4]; [g_enc setBytes:&M length:4 atIndex:5];
    [g_enc dispatchThreadgroups:MTLSizeMake((M+31)/32,(N+63)/64,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
    return true;
}
bool cactus_metal_encode_attention_f16(void* out, const void* q, const void* k, const void* v, const void* mask,
    uint32_t B, uint32_t T, uint32_t S, uint32_t HQ, uint32_t HKV, uint32_t D, uint32_t DV,
    float scale, uint32_t causal, uint32_t pos_off, uint32_t window, float logit_cap, uint32_t mask_mode) {
    if (!ctx().ok || !ctx().psoAttnDense) return false;
    if (D > 128 || DV > 128 || (D % 4) != 0 || (DV % 4) != 0 || HKV == 0 || HQ % HKV != 0) return false;
    static const bool mps_attn_ok = MPSSupportsMTLDevice(ctx().dev);
    bool mask_ok = mask_mode == 0 || (mask_mode == 2 && mask && ctx().psoMaskFill);
    if (ctx().psoAttnFlash && B == 1 && HQ == HKV && !causal && window == 0
        && logit_cap == 0.0f && D == 64 && DV == 64 && S >= 256
        && (mask_mode == 0 || (mask_mode == 2 && mask))) {
        ensureEncoder();
        [g_enc setComputePipelineState:ctx().psoAttnFlash];
        setBufAt(q, (size_t)T*HQ*D*2, 0); setBufAt(k, (size_t)S*HKV*D*2, 1);
        setBufAt(v, (size_t)S*HKV*DV*2, 2); setBufAt(out, (size_t)T*HQ*DV*2, 3);
        if (mask_mode == 2 && mask) setBufAt(mask, (size_t)T*S*2, 4);
        else [g_enc setBuffer:ctx().dummy offset:0 atIndex:4];
        [g_enc setBytes:&T length:4 atIndex:5]; [g_enc setBytes:&S length:4 atIndex:6];
        [g_enc setBytes:&HQ length:4 atIndex:7]; [g_enc setBytes:&scale length:4 atIndex:8];
        [g_enc setBytes:&mask_mode length:4 atIndex:9];
        [g_enc dispatchThreadgroups:MTLSizeMake((T + 63) / 64, HQ, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        return true;
    }
    if (mps_attn_ok && B == 1 && HQ == HKV && !causal && mask_ok && window == 0
        && logit_cap == 0.0f && S >= 512 && (S % 8) == 0 && (D % 8) == 0 && (DV % 8) == 0
        && ctx().psoSoftmaxR) {
        auto qb = bufForPtrOff(q, (size_t)T*HQ*D*2);
        auto kb = bufForPtrOff(k, (size_t)S*HKV*D*2);
        auto vb = bufForPtrOff(v, (size_t)S*HKV*DV*2);
        auto ob = bufForPtrOff(out, (size_t)T*HQ*DV*2);
        bool aligned = (qb.second % 16)==0 && (kb.second % 16)==0 && (vb.second % 16)==0 && (ob.second % 16)==0
            && ((D*2) % 16)==0 && ((DV*2) % 16)==0;
        if (aligned) {
            size_t need = (size_t)HQ*T*S*2;
            if (!g_attn_scores || g_attn_scores.length < need)
                g_attn_scores = owned_shared(need);
            id<MTLBuffer> sbuf = g_attn_scores;
            if (sbuf) {
                float beta = 0.0f;
                if (mask_mode == 2) {
                    ensureEncoder();
                    [g_enc setComputePipelineState:ctx().psoMaskFill];
                    setBufAt(mask, (size_t)T*S*2, 0);
                    [g_enc setBuffer:sbuf offset:0 atIndex:1];
                    uint32_t ts = T*S;
                    [g_enc setBytes:&ts length:4 atIndex:2];
                    [g_enc dispatchThreads:MTLSizeMake(ts, HQ, 1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                    beta = 1.0f;
                }
                if (g_enc) { [g_enc endEncoding]; g_enc = nil; }
                if (!g_cmd) g_cmd = [ctx().queue commandBuffer];
                MPSMatrixDescriptor* dq = [MPSMatrixDescriptor matrixDescriptorWithRows:T columns:D
                    rowBytes:(size_t)HQ*D*2 dataType:MPSDataTypeFloat16];
                MPSMatrixDescriptor* dk = [MPSMatrixDescriptor matrixDescriptorWithRows:S columns:D
                    rowBytes:(size_t)HKV*D*2 dataType:MPSDataTypeFloat16];
                MPSMatrixDescriptor* ds = [MPSMatrixDescriptor matrixDescriptorWithRows:T columns:S
                    rowBytes:(size_t)S*2 dataType:MPSDataTypeFloat16];
                MPSMatrixDescriptor* dv = [MPSMatrixDescriptor matrixDescriptorWithRows:S columns:DV
                    rowBytes:(size_t)HKV*DV*2 dataType:MPSDataTypeFloat16];
                MPSMatrixDescriptor* do2 = [MPSMatrixDescriptor matrixDescriptorWithRows:T columns:DV
                    rowBytes:(size_t)HQ*DV*2 dataType:MPSDataTypeFloat16];
                MPSMatrixMultiplication* mqk = mps_gemm_ab(T, S, D, true, scale, beta);
                for (uint32_t h = 0; h < HQ; ++h) {
                    MPSMatrix* mq = [[MPSMatrix alloc] initWithBuffer:qb.first offset:qb.second + (size_t)h*D*2 descriptor:dq];
                    MPSMatrix* mk = [[MPSMatrix alloc] initWithBuffer:kb.first offset:kb.second + (size_t)h*D*2 descriptor:dk];
                    MPSMatrix* msc = [[MPSMatrix alloc] initWithBuffer:sbuf offset:(size_t)h*T*S*2 descriptor:ds];
                    [mqk encodeToCommandBuffer:g_cmd leftMatrix:mq rightMatrix:mk resultMatrix:msc];
                }
                ensureEncoder();
                [g_enc setComputePipelineState:ctx().psoSoftmaxR];
                [g_enc setBuffer:sbuf offset:0 atIndex:0];
                [g_enc setBuffer:sbuf offset:0 atIndex:1];
                uint32_t cols = S;
                [g_enc setBytes:&cols length:4 atIndex:2];
                [g_enc setThreadgroupMemoryLength:256*4 atIndex:0];
                [g_enc dispatchThreadgroups:MTLSizeMake((size_t)HQ*T,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                [g_enc endEncoding]; g_enc = nil;
                MPSMatrixMultiplication* mpv = mps_gemm(T, DV, S, false);
                for (uint32_t h = 0; h < HQ; ++h) {
                    MPSMatrix* msc = [[MPSMatrix alloc] initWithBuffer:sbuf offset:(size_t)h*T*S*2 descriptor:ds];
                    MPSMatrix* mv = [[MPSMatrix alloc] initWithBuffer:vb.first offset:vb.second + (size_t)h*DV*2 descriptor:dv];
                    MPSMatrix* mo = [[MPSMatrix alloc] initWithBuffer:ob.first offset:ob.second + (size_t)h*DV*2 descriptor:do2];
                    [mpv encodeToCommandBuffer:g_cmd leftMatrix:msc rightMatrix:mv resultMatrix:mo];
                }
                return true;
            }
        }
    }
    ensureEncoder();
    struct { uint32_t T,S,HQ,HKV,D,DV; float scale; uint32_t causal,pos_off,window; float logit_cap; uint32_t mask_mode; } U =
        { T,S,HQ,HKV,D,DV, scale, causal,pos_off,window, logit_cap, mask_mode };
    bool d64 = (D <= 64 && DV <= 64 && ctx().psoAttnD64);
    [g_enc setComputePipelineState:(d64 ? ctx().psoAttnD64 : ctx().psoAttnDense)];
    setBufAt(q, (size_t)B*T*HQ*D*2, 0); setBufAt(k, (size_t)B*S*HKV*D*2, 1);
    setBufAt(v, (size_t)B*S*HKV*DV*2, 2); setBufAt(out, (size_t)B*T*HQ*DV*2, 3);
    if (mask_mode && mask) {
        size_t mb = (mask_mode>=3u) ? (size_t)B*HQ*T*S*2 : (size_t)B*T*S*2;
        setBufAt(mask, mb, 4);
    } else [g_enc setBuffer:ctx().dummy offset:0 atIndex:4];
    [g_enc setBytes:&U length:sizeof(U) atIndex:5];
    [g_enc dispatchThreadgroups:MTLSizeMake(d64 ? (T+7)/8 : (T+3)/4,HQ,B) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    return true;
}

bool cactus_metal_encode_reduce_axis(int op, void* out, const void* in, uint32_t outer,
                                     uint32_t axis_size, uint32_t inner, int f32) {
    if (!ctx().ok || !ctx().psoReduceF16 || !ctx().psoReduceF32) return false;
    ensureEncoder();
    size_t es = f32 ? 4 : 2;
    uint32_t n = outer * inner;
    [g_enc setComputePipelineState:(f32 ? ctx().psoReduceF32 : ctx().psoReduceF16)];
    setBufAt(in, (size_t)outer*axis_size*inner*es, 0); setBufAt(out, (size_t)n*es, 1);
    [g_enc setBytes:&axis_size length:4 atIndex:2]; [g_enc setBytes:&inner length:4 atIndex:3];
    [g_enc setBytes:&n length:4 atIndex:4]; [g_enc setBytes:&op length:4 atIndex:5];
    [g_enc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    return true;
}
bool cactus_metal_encode_cumsum(void* out, const void* in, uint32_t outer,
                                uint32_t axis_size, uint32_t inner, int f32) {
    if (!ctx().ok || !ctx().psoCumsumF16 || !ctx().psoCumsumF32) return false;
    ensureEncoder();
    size_t es = f32 ? 4 : 2;
    uint32_t n = outer * inner;
    size_t bytes = (size_t)outer*axis_size*inner*es;
    [g_enc setComputePipelineState:(f32 ? ctx().psoCumsumF32 : ctx().psoCumsumF16)];
    setBufAt(in, bytes, 0); setBufAt(out, bytes, 1);
    [g_enc setBytes:&axis_size length:4 atIndex:2]; [g_enc setBytes:&inner length:4 atIndex:3];
    [g_enc setBytes:&n length:4 atIndex:4];
    [g_enc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    return true;
}
bool cactus_metal_encode_concat2(void* out, const void* a, const void* b,
                                 uint32_t a_outer, uint32_t b_outer,
                                 uint32_t a_axis, uint32_t b_axis, uint32_t inner) {
    if (!ctx().ok || !ctx().psoConcat2 || a_outer == 0 || b_outer == 0) return false;
    ensureEncoder();
    uint32_t outer = a_outer > b_outer ? a_outer : b_outer;
    uint32_t n = outer * (a_axis + b_axis) * inner;
    [g_enc setComputePipelineState:ctx().psoConcat2];
    setBufAt(a, (size_t)a_outer*a_axis*inner*2, 0); setBufAt(b, (size_t)b_outer*b_axis*inner*2, 1);
    setBufAt(out, (size_t)n*2, 2);
    [g_enc setBytes:&a_axis length:4 atIndex:3]; [g_enc setBytes:&b_axis length:4 atIndex:4];
    [g_enc setBytes:&inner length:4 atIndex:5]; [g_enc setBytes:&n length:4 atIndex:6];
    [g_enc setBytes:&a_outer length:4 atIndex:7]; [g_enc setBytes:&b_outer length:4 atIndex:8];
    [g_enc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    return true;
}
bool cactus_metal_encode_gather_f32idx(void* out, const void* table, const void* idx,
                                       uint32_t rows, uint32_t D, size_t table_bytes) {
    if (!ctx().ok || !ctx().psoGatherF32Idx) return false;
    ensureEncoder();
    uint32_t n = rows * D;
    [g_enc setComputePipelineState:ctx().psoGatherF32Idx];
    setBufAt(table, table_bytes, 0); setBufAt(idx, (size_t)rows*4, 1); setBufAt(out, (size_t)n*2, 2);
    [g_enc setBytes:&D length:4 atIndex:3]; [g_enc setBytes:&n length:4 atIndex:4];
    [g_enc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    return true;
}
bool cactus_metal_encode_rope_full(void* out, const void* in, uint32_t tokens, uint32_t S,
                                   uint32_t H, uint32_t D, uint32_t rot, uint32_t pos0,
                                   float theta, int gptj) {
    if (!ctx().ok || !ctx().psoRopeFull) return false;
    ensureEncoder();
    uint32_t gj = gptj ? 1u : 0u;
    [g_enc setComputePipelineState:ctx().psoRopeFull];
    setBufAt(in, (size_t)tokens*D*2, 0); setBufAt(out, (size_t)tokens*D*2, 1);
    [g_enc setBytes:&S length:4 atIndex:2]; [g_enc setBytes:&H length:4 atIndex:3];
    [g_enc setBytes:&D length:4 atIndex:4]; [g_enc setBytes:&rot length:4 atIndex:5];
    [g_enc setBytes:&pos0 length:4 atIndex:6]; [g_enc setBytes:&theta length:4 atIndex:7];
    [g_enc setBytes:&gj length:4 atIndex:8];
    uint32_t gx = rot/2 + (D > rot ? D - rot : 0);
    if (gx == 0) gx = 1;
    [g_enc dispatchThreads:MTLSizeMake(gx, tokens, 1) threadsPerThreadgroup:MTLSizeMake(std::min(gx, 64u), 4, 1)];
    return true;
}
bool cactus_metal_encode_maxpool1d(void* out, const void* in, uint32_t NC, uint32_t L,
                                   uint32_t Lout, uint32_t K, uint32_t stride) {
    if (!ctx().ok || !ctx().psoMaxpool1d) return false;
    ensureEncoder();
    [g_enc setComputePipelineState:ctx().psoMaxpool1d];
    setBufAt(in, (size_t)NC*L*2, 0); setBufAt(out, (size_t)NC*Lout*2, 1);
    [g_enc setBytes:&L length:4 atIndex:2]; [g_enc setBytes:&Lout length:4 atIndex:3];
    [g_enc setBytes:&K length:4 atIndex:4]; [g_enc setBytes:&stride length:4 atIndex:5];
    [g_enc dispatchThreads:MTLSizeMake(Lout, NC, 1) threadsPerThreadgroup:MTLSizeMake(std::min(Lout, 64u), 4, 1)];
    return true;
}
bool cactus_metal_encode_bilinear(void* out, const void* in, uint32_t sh, uint32_t sw,
                                  uint32_t dh, uint32_t dw, uint32_t E, int align) {
    if (!ctx().ok || !ctx().psoBilinear) return false;
    ensureEncoder();
    uint32_t al = align ? 1u : 0u;
    [g_enc setComputePipelineState:ctx().psoBilinear];
    setBufAt(in, (size_t)sh*sw*E*2, 0); setBufAt(out, (size_t)dh*dw*E*2, 1);
    [g_enc setBytes:&sh length:4 atIndex:2]; [g_enc setBytes:&sw length:4 atIndex:3];
    [g_enc setBytes:&dh length:4 atIndex:4]; [g_enc setBytes:&dw length:4 atIndex:5];
    [g_enc setBytes:&E length:4 atIndex:6]; [g_enc setBytes:&al length:4 atIndex:7];
    [g_enc dispatchThreads:MTLSizeMake(E, dh*dw, 1) threadsPerThreadgroup:MTLSizeMake(std::min(E, 64u), 4, 1)];
    return true;
}
bool cactus_metal_encode_conv1d_gen(void* out, const void* x, const void* w, const void* bias,
                                    uint32_t N, uint32_t Cin, uint32_t L, uint32_t Cout,
                                    uint32_t Lout, uint32_t K, uint32_t stride, int w_ck_co) {
    if (!ctx().ok || !ctx().psoConv1dGen) return false;
    ensureEncoder();
    uint32_t hb = bias ? 1u : 0u, wl = w_ck_co ? 1u : 0u;
    [g_enc setComputePipelineState:ctx().psoConv1dGen];
    setBufAt(x, (size_t)N*Cin*L*2, 0); setBufAt(w, (size_t)Cout*Cin*K*2, 1);
    if (bias) setBufAt(bias, (size_t)Cout*2, 2); else [g_enc setBuffer:ctx().dummy offset:0 atIndex:2];
    setBufAt(out, (size_t)N*Cout*Lout*2, 3);
    [g_enc setBytes:&Cin length:4 atIndex:4]; [g_enc setBytes:&L length:4 atIndex:5];
    [g_enc setBytes:&Cout length:4 atIndex:6]; [g_enc setBytes:&Lout length:4 atIndex:7];
    [g_enc setBytes:&K length:4 atIndex:8]; [g_enc setBytes:&stride length:4 atIndex:9];
    [g_enc setBytes:&hb length:4 atIndex:10]; [g_enc setBytes:&wl length:4 atIndex:11];
    [g_enc dispatchThreads:MTLSizeMake(Lout, Cout, N) threadsPerThreadgroup:MTLSizeMake(std::min(Lout, 64u), 4, 1)];
    return true;
}
bool cactus_metal_encode_conv1d_nlc_dw(void* out, const void* x, const void* w, const void* bias,
                                       uint32_t N, uint32_t L, uint32_t C, uint32_t K,
                                       uint32_t dil, uint32_t pad) {
    if (!ctx().ok || !ctx().psoConv1dNlcDw) return false;
    ensureEncoder();
    uint32_t hb = bias ? 1u : 0u;
    [g_enc setComputePipelineState:ctx().psoConv1dNlcDw];
    setBufAt(x, (size_t)N*L*C*2, 0); setBufAt(w, (size_t)C*K*2, 1);
    if (bias) setBufAt(bias, (size_t)C*2, 2); else [g_enc setBuffer:ctx().dummy offset:0 atIndex:2];
    setBufAt(out, (size_t)N*L*C*2, 3);
    [g_enc setBytes:&L length:4 atIndex:4]; [g_enc setBytes:&C length:4 atIndex:5];
    [g_enc setBytes:&K length:4 atIndex:6]; [g_enc setBytes:&dil length:4 atIndex:7];
    [g_enc setBytes:&pad length:4 atIndex:8]; [g_enc setBytes:&hb length:4 atIndex:9];
    [g_enc dispatchThreads:MTLSizeMake(C, L, N) threadsPerThreadgroup:MTLSizeMake(std::min(C, 64u), 4, 1)];
    return true;
}
bool cactus_metal_encode_conv2d(void* out, const void* x, const void* w, const void* bias,
                                uint32_t N, uint32_t Cin, uint32_t H, uint32_t W, uint32_t Cout,
                                uint32_t Ho, uint32_t Wo, uint32_t K, uint32_t stride,
                                uint32_t pad, int dw) {
    if (!ctx().ok || !ctx().psoConv2d) return false;
    ensureEncoder();
    uint32_t hb = bias ? 1u : 0u, dwf = dw ? 1u : 0u;
    size_t wbytes = dw ? (size_t)Cout*K*K*2 : (size_t)Cout*Cin*K*K*2;
    [g_enc setComputePipelineState:ctx().psoConv2d];
    setBufAt(x, (size_t)N*Cin*H*W*2, 0); setBufAt(w, wbytes, 1);
    if (bias) setBufAt(bias, (size_t)Cout*2, 2); else [g_enc setBuffer:ctx().dummy offset:0 atIndex:2];
    setBufAt(out, (size_t)N*Cout*Ho*Wo*2, 3);
    [g_enc setBytes:&Cin length:4 atIndex:4]; [g_enc setBytes:&H length:4 atIndex:5];
    [g_enc setBytes:&W length:4 atIndex:6]; [g_enc setBytes:&Cout length:4 atIndex:7];
    [g_enc setBytes:&Ho length:4 atIndex:8]; [g_enc setBytes:&Wo length:4 atIndex:9];
    [g_enc setBytes:&K length:4 atIndex:10]; [g_enc setBytes:&stride length:4 atIndex:11];
    [g_enc setBytes:&pad length:4 atIndex:12]; [g_enc setBytes:&dwf length:4 atIndex:13];
    [g_enc setBytes:&hb length:4 atIndex:14];
    uint32_t wq = (Wo + 3) / 4;
    [g_enc dispatchThreads:MTLSizeMake(wq, Ho, (size_t)N*Cout) threadsPerThreadgroup:MTLSizeMake(std::min(wq, 32u), std::min(Ho, 8u), 1)];
    return true;
}
bool cactus_metal_encode_batchnorm(void* out, const void* x, const void* w, const void* b,
                                   const void* rm, const void* rv, uint32_t C, uint32_t inner,
                                   uint32_t total, float eps) {
    if (!ctx().ok || !ctx().psoBatchnorm) return false;
    ensureEncoder();
    [g_enc setComputePipelineState:ctx().psoBatchnorm];
    setBufAt(x, (size_t)total*2, 0); setBufAt(w, (size_t)C*2, 1); setBufAt(b, (size_t)C*2, 2);
    setBufAt(rm, (size_t)C*2, 3); setBufAt(rv, (size_t)C*2, 4); setBufAt(out, (size_t)total*2, 5);
    [g_enc setBytes:&C length:4 atIndex:6]; [g_enc setBytes:&inner length:4 atIndex:7];
    [g_enc setBytes:&eps length:4 atIndex:8]; [g_enc setBytes:&total length:4 atIndex:9];
    [g_enc dispatchThreads:MTLSizeMake(total,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    return true;
}
bool cactus_metal_encode_groupnorm(void* out, const void* x, const void* w, const void* b,
                                   uint32_t N, uint32_t C, uint32_t S, uint32_t groups, float eps) {
    if (!ctx().ok || !ctx().psoGroupnorm || groups == 0 || C % groups != 0) return false;
    ensureEncoder();
    uint32_t cpg = C / groups;
    [g_enc setComputePipelineState:ctx().psoGroupnorm];
    setBufAt(x, (size_t)N*C*S*2, 0); setBufAt(w, (size_t)C*2, 1); setBufAt(b, (size_t)C*2, 2);
    setBufAt(out, (size_t)N*C*S*2, 3);
    [g_enc setBytes:&cpg length:4 atIndex:4]; [g_enc setBytes:&S length:4 atIndex:5];
    [g_enc setBytes:&C length:4 atIndex:6]; [g_enc setBytes:&eps length:4 atIndex:7];
    [g_enc setThreadgroupMemoryLength:256*4 atIndex:0];
    [g_enc dispatchThreadgroups:MTLSizeMake(groups, N, 1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    return true;
}
bool cactus_metal_encode_bias_add_rows(void* y, const void* bias, uint32_t C, uint32_t total) {
    if (!ctx().ok || !ctx().psoBiasAddRows) return false;
    ensureEncoder();
    [g_enc setComputePipelineState:ctx().psoBiasAddRows];
    setBufAt(y, (size_t)total*2, 0); setBufAt(bias, (size_t)C*2, 1);
    [g_enc setBytes:&C length:4 atIndex:2]; [g_enc setBytes:&total length:4 atIndex:3];
    [g_enc dispatchThreads:MTLSizeMake(total,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    return true;
}
bool cactus_metal_encode_elemwise_chain(void* out, const void* in, const float* steps,
                                        uint32_t nsteps, const void* side0, const void* side1,
                                        const void* side2, const size_t* side_elems,
                                        size_t n, uint32_t flags, uint32_t inner) {
    if (!ctx().ok || !ctx().psoEwChain || nsteps == 0 || nsteps > 12) return false;
    ensureEncoder();
    uint32_t ns = nsteps, nn = (uint32_t)n;
    size_t ein = (flags & 1u) ? 4 : 2, eout = (flags & 2u) ? 4 : 2;
    [g_enc setComputePipelineState:ctx().psoEwChain];
    setBufAt(in, n*ein, 0); setBufAt(out, n*eout, 1);
    [g_enc setBytes:steps length:nsteps*16 atIndex:2];
    [g_enc setBytes:&ns length:4 atIndex:3];
    setBufAt(side0 ? side0 : in, (side0 ? side_elems[0] : n)*((flags & 4u) ? 4 : 2), 4);
    setBufAt(side1 ? side1 : in, (side1 ? side_elems[1] : n)*((flags & 8u) ? 4 : 2), 5);
    setBufAt(side2 ? side2 : in, (side2 ? side_elems[2] : n)*((flags & 16u) ? 4 : 2), 6);
    [g_enc setBytes:&nn length:4 atIndex:7];
    [g_enc setBytes:&flags length:4 atIndex:8];
    uint32_t innr = inner;
    [g_enc setBytes:&innr length:4 atIndex:9];
    [g_enc dispatchThreads:MTLSizeMake((nn+3)/4,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    return true;
}

bool cactus_metal_encode_rms_norm_add_rows(void* ysum, void* ynorm, const void* x, const void* res,
                                           const void* w, uint32_t rows, uint32_t dim, float eps,
                                           int clipped) {
    if (!ctx().ok || !ctx().psoRmsAddSimd) return false;
    ensureEncoder();
    uint32_t cl = clipped ? 1u : 0u;
    [g_enc setComputePipelineState:ctx().psoRmsAddSimd];
    setBufAt(x, (size_t)rows*dim*2, 0); setBufAt(res, (size_t)rows*dim*2, 1);
    setBufAt(w, (size_t)dim*2, 2); setBufAt(ysum, (size_t)rows*dim*2, 3);
    setBufAt(ynorm, (size_t)rows*dim*2, 4);
    [g_enc setBytes:&dim length:4 atIndex:5]; [g_enc setBytes:&eps length:4 atIndex:6];
    [g_enc setBytes:&rows length:4 atIndex:7]; [g_enc setBytes:&cl length:4 atIndex:8];
    [g_enc dispatchThreadgroups:MTLSizeMake((rows+3)/4,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
    return true;
}
bool cactus_metal_encode_gemm_batch(void* out, const void* a, const void* b,
                                    uint32_t M, uint32_t K, uint32_t N, uint32_t batch, int f32out, int f32a) {
    if (!ctx().ok || !ctx().psoGemmBatch || !ctx().psoGemmBatchF32A) return false;
    ensureEncoder();
    uint32_t fo = f32out ? 1u : 0u;
    [g_enc setComputePipelineState:(f32a ? ctx().psoGemmBatchF32A : ctx().psoGemmBatch)];
    setBufAt(a, (size_t)batch*M*K*(f32a?4:2), 0); setBufAt(b, (size_t)batch*K*N*2, 1);
    setBufAt(out, (size_t)batch*M*N*(f32out?4:2), 2);
    [g_enc setBytes:&M length:4 atIndex:3]; [g_enc setBytes:&K length:4 atIndex:4];
    [g_enc setBytes:&N length:4 atIndex:5]; [g_enc setBytes:&fo length:4 atIndex:6];
    [g_enc dispatchThreads:MTLSizeMake((size_t)M*N, batch, 1) threadsPerThreadgroup:MTLSizeMake(64,4,1)];
    return true;
}
bool cactus_metal_encode_conv1d_dw(void* out, const void* x, const void* w,
                                   uint32_t C, uint32_t L, uint32_t Lout, uint32_t K, uint32_t stride) {
    if (!ctx().ok || !ctx().psoConvDw) return false;
    ensureEncoder();
    [g_enc setComputePipelineState:ctx().psoConvDw];
    setBufAt(x, (size_t)C*L*2, 0); setBufAt(w, (size_t)C*K*2, 1); setBufAt(out, (size_t)C*Lout*2, 2);
    [g_enc setBytes:&L length:4 atIndex:3]; [g_enc setBytes:&Lout length:4 atIndex:4];
    [g_enc setBytes:&K length:4 atIndex:5]; [g_enc setBytes:&stride length:4 atIndex:6];
    [g_enc dispatchThreads:MTLSizeMake(Lout, C, 1) threadsPerThreadgroup:MTLSizeMake(64,4,1)];
    return true;
}
bool cactus_metal_encode_transpose2d(void* out, const void* in, uint32_t batch, uint32_t R, uint32_t C) {
    if (!ctx().ok || !ctx().psoTranspose2d) return false;
    ensureEncoder();
    [g_enc setComputePipelineState:ctx().psoTranspose2d];
    setBufAt(in, (size_t)batch*R*C*2, 0); setBufAt(out, (size_t)batch*R*C*2, 1);
    [g_enc setBytes:&R length:4 atIndex:2]; [g_enc setBytes:&C length:4 atIndex:3];
    [g_enc dispatchThreadgroups:MTLSizeMake((C+31)/32, (R+31)/32, batch) threadsPerThreadgroup:MTLSizeMake(32,8,1)];
    return true;
}
bool cactus_metal_encode_strided_copy(void* out, const void* in, const uint32_t* oshape,
    const uint32_t* sstride, uint32_t ndim, uint32_t total, uint32_t base, size_t in_bytes, size_t out_bytes) {
    if (!ctx().ok || ndim == 0 || ndim > 8) return false;
    ensureEncoder();
    if (ndim >= 2 && sstride[ndim-1] == 1 && (oshape[ndim-1] % 4) == 0 && ctx().psoStridedRows) {
        uint32_t inner = oshape[ndim-1];
        uint32_t inner4 = inner / 4;
        uint32_t nd2 = ndim - 1;
        uint32_t rows = total / inner;
        [g_enc setComputePipelineState:ctx().psoStridedRows];
        setBufAt(in, in_bytes, 0); setBufAt(out, out_bytes, 1);
        [g_enc setBytes:oshape length:nd2*4 atIndex:2]; [g_enc setBytes:sstride length:nd2*4 atIndex:3];
        [g_enc setBytes:&nd2 length:4 atIndex:4]; [g_enc setBytes:&rows length:4 atIndex:5];
        [g_enc setBytes:&base length:4 atIndex:6]; [g_enc setBytes:&inner4 length:4 atIndex:7];
        [g_enc dispatchThreads:MTLSizeMake(inner4, rows, 1) threadsPerThreadgroup:MTLSizeMake(std::min(inner4, 64u), 4, 1)];
        return true;
    }
    [g_enc setComputePipelineState:ctx().psoStrided];
    setBufAt(in, in_bytes, 0); setBufAt(out, out_bytes, 1);
    [g_enc setBytes:oshape length:ndim*4 atIndex:2]; [g_enc setBytes:sstride length:ndim*4 atIndex:3];
    [g_enc setBytes:&ndim length:4 atIndex:4]; [g_enc setBytes:&total length:4 atIndex:5]; [g_enc setBytes:&base length:4 atIndex:6];
    [g_enc dispatchThreads:MTLSizeMake(total,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    return true;
}
bool cactus_metal_encode_bcast_binary(int op, void* out, const void* a, const void* b,
    const uint32_t* oshape, const uint32_t* astride, const uint32_t* bstride, uint32_t ndim, uint32_t total,
    size_t a_bytes, size_t b_bytes, size_t out_bytes) {
    if (!ctx().ok || ndim == 0 || ndim > 8) return false;
    ensureEncoder();
    int o = op;
    if (ndim >= 2 && oshape[ndim-1] > 1 && ctx().psoBcastRows) {
        uint32_t inner = oshape[ndim-1];
        uint32_t ainner = astride[ndim-1], binner = bstride[ndim-1];
        uint32_t nd2 = ndim - 1;
        uint32_t rows = total / inner;
        [g_enc setComputePipelineState:ctx().psoBcastRows];
        setBufAt(a, a_bytes, 0); setBufAt(b, b_bytes, 1); setBufAt(out, out_bytes, 2);
        [g_enc setBytes:oshape length:nd2*4 atIndex:3]; [g_enc setBytes:astride length:nd2*4 atIndex:4];
        [g_enc setBytes:bstride length:nd2*4 atIndex:5];
        [g_enc setBytes:&nd2 length:4 atIndex:6]; [g_enc setBytes:&rows length:4 atIndex:7];
        [g_enc setBytes:&o length:4 atIndex:8]; [g_enc setBytes:&inner length:4 atIndex:9];
        [g_enc setBytes:&ainner length:4 atIndex:10]; [g_enc setBytes:&binner length:4 atIndex:11];
        [g_enc dispatchThreads:MTLSizeMake(inner, rows, 1) threadsPerThreadgroup:MTLSizeMake(std::min(inner, 64u), 4, 1)];
        return true;
    }
    [g_enc setComputePipelineState:ctx().psoBcast];
    setBufAt(a, a_bytes, 0); setBufAt(b, b_bytes, 1); setBufAt(out, out_bytes, 2);
    [g_enc setBytes:oshape length:ndim*4 atIndex:3]; [g_enc setBytes:astride length:ndim*4 atIndex:4]; [g_enc setBytes:bstride length:ndim*4 atIndex:5];
    [g_enc setBytes:&ndim length:4 atIndex:6]; [g_enc setBytes:&total length:4 atIndex:7]; [g_enc setBytes:&o length:4 atIndex:8];
    [g_enc dispatchThreads:MTLSizeMake(total,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    return true;
}
bool cactus_metal_encode_strided_scatter(void* out, const void* in, const uint32_t* ishape,
    const uint32_t* ostride, uint32_t ndim, uint32_t total, uint32_t base, size_t in_bytes, size_t out_bytes) {
    if (!ctx().ok || ndim == 0 || ndim > 8) return false;
    ensureEncoder();
    if (ndim >= 2 && ostride[ndim-1] == 1 && (ishape[ndim-1] % 4) == 0 && ctx().psoScatterRows) {
        uint32_t inner = ishape[ndim-1];
        uint32_t inner4 = inner / 4;
        uint32_t nd2 = ndim - 1;
        uint32_t rows = total / inner;
        [g_enc setComputePipelineState:ctx().psoScatterRows];
        setBufAt(in, in_bytes, 0); setBufAt(out, out_bytes, 1);
        [g_enc setBytes:ishape length:nd2*4 atIndex:2]; [g_enc setBytes:ostride length:nd2*4 atIndex:3];
        [g_enc setBytes:&nd2 length:4 atIndex:4]; [g_enc setBytes:&rows length:4 atIndex:5];
        [g_enc setBytes:&base length:4 atIndex:6]; [g_enc setBytes:&inner4 length:4 atIndex:7];
        [g_enc dispatchThreads:MTLSizeMake(inner4, rows, 1) threadsPerThreadgroup:MTLSizeMake(std::min(inner4, 64u), 4, 1)];
        return true;
    }
    [g_enc setComputePipelineState:ctx().psoScatter];
    setBufAt(in, in_bytes, 0); setBufAt(out, out_bytes, 1);
    [g_enc setBytes:ishape length:ndim*4 atIndex:2]; [g_enc setBytes:ostride length:ndim*4 atIndex:3];
    [g_enc setBytes:&ndim length:4 atIndex:4]; [g_enc setBytes:&total length:4 atIndex:5]; [g_enc setBytes:&base length:4 atIndex:6];
    [g_enc dispatchThreads:MTLSizeMake(total,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    return true;
}
static bool encode_kv_append(const void* src, void* int8base, void* scalebase,
    uint32_t kv_heads, uint32_t hdim, uint32_t current_len, uint32_t group_size, uint32_t M,
    size_t src_bytes, size_t int8_bytes, size_t scale_bytes) {
    if (!ctx().ok) return false;
    uint32_t num_groups = (hdim + group_size - 1)/group_size, n = M*kv_heads*num_groups;
    ensureEncoder();
    [g_enc setComputePipelineState:(M == 1 ? ctx().psoKvAppend : ctx().psoKvAppendM)];
    setBufAt(src, src_bytes, 0); setBufAt(int8base, int8_bytes, 1); setBufAt(scalebase, scale_bytes, 2);
    [g_enc setBytes:&kv_heads length:4 atIndex:3]; [g_enc setBytes:&hdim length:4 atIndex:4];
    [g_enc setBytes:&current_len length:4 atIndex:5]; [g_enc setBytes:&group_size length:4 atIndex:6];
    if (M != 1) [g_enc setBytes:&M length:4 atIndex:7];
    [g_enc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(n<256?n:256,1,1)];
    return true;
}

bool cactus_metal_encode_kv_append_i8(const void* src, void* int8base, void* scalebase,
    uint32_t kv_heads, uint32_t hdim, uint32_t current_len, uint32_t group_size,
    size_t src_bytes, size_t int8_bytes, size_t scale_bytes) {
    return encode_kv_append(src, int8base, scalebase, kv_heads, hdim, current_len, group_size, 1,
        src_bytes, int8_bytes, scale_bytes);
}

bool cactus_metal_encode_kv_append_i8_m(const void* src, void* int8base, void* scalebase,
    uint32_t kv_heads, uint32_t hdim, uint32_t current_len, uint32_t group_size, uint32_t M,
    size_t src_bytes, size_t int8_bytes, size_t scale_bytes) {
    return encode_kv_append(src, int8base, scalebase, kv_heads, hdim, current_len, group_size, M,
        src_bytes, int8_bytes, scale_bytes);
}

bool cactus_metal_encode_gemv_bias(void* out, const void* x, const void* w, const void* bias,
    uint32_t K, uint32_t N, int pretransposed) {
    if (!ctx().ok || !ctx().psoGemvBias || !bias) return false;
    ensureEncoder();
    [g_enc setComputePipelineState:ctx().psoGemvBias];
    setBufAt(x, (size_t)K*2, 0);
    setBufAt(w, (size_t)K*N*2, 1);
    setBufAt(bias, (size_t)N*2, 2);
    setBufAt(out, (size_t)N*2, 3);
    uint32_t trv = pretransposed ? 1u : 0u;
    [g_enc setBytes:&K length:4 atIndex:4]; [g_enc setBytes:&N length:4 atIndex:5];
    [g_enc setBytes:&trv length:4 atIndex:6];
    [g_enc dispatchThreadgroups:MTLSizeMake((N+3)/4,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
    return true;
}

bool cactus_metal_encode_rel_pos_bias(void* y, const void* q, const void* r,
    uint32_t B, uint32_t T, uint32_t H, uint32_t D, uint32_t R, int r_batched, float scale) {
    if (!ctx().ok || !ctx().psoRelPosBias) return false;
    ensureEncoder();
    [g_enc setComputePipelineState:ctx().psoRelPosBias];
    setBufAt(q, (size_t)B*T*H*D*2, 0);
    setBufAt(r, (size_t)(r_batched?B:1)*R*H*D*2, 1);
    setBufAt(y, (size_t)B*H*T*T*2, 2);
    uint32_t rbs = r_batched ? R*H*D : 0u;
    [g_enc setBytes:&T length:4 atIndex:3]; [g_enc setBytes:&H length:4 atIndex:4];
    [g_enc setBytes:&D length:4 atIndex:5]; [g_enc setBytes:&rbs length:4 atIndex:6];
    [g_enc setBytes:&scale length:4 atIndex:7];
    [g_enc dispatchThreads:MTLSizeMake((size_t)T*T, H, B)
        threadsPerThreadgroup:MTLSizeMake(64, H < 4 ? H : 4, 1)];
    return true;
}

bool cactus_metal_encode_conv_cache_append(void* out, const void* src, void* ring,
    uint32_t hd, uint32_t ws, uint32_t nnew, uint32_t head0, uint32_t count_new,
    uint32_t num_rows, int src_f32) {
    if (!ctx().ok || !ctx().psoConvCacheAppend) return false;
    ensureEncoder();
    [g_enc setComputePipelineState:ctx().psoConvCacheAppend];
    setBufAt(src, (size_t)num_rows*hd*(src_f32?4:2), 0);
    setBufAt(ring, (size_t)ws*hd*2, 1);
    setBufAt(out, (size_t)ws*hd*2, 2);
    uint32_t f32 = src_f32 ? 1u : 0u;
    [g_enc setBytes:&hd length:4 atIndex:3]; [g_enc setBytes:&ws length:4 atIndex:4];
    [g_enc setBytes:&nnew length:4 atIndex:5]; [g_enc setBytes:&head0 length:4 atIndex:6];
    [g_enc setBytes:&count_new length:4 atIndex:7]; [g_enc setBytes:&num_rows length:4 atIndex:8];
    [g_enc setBytes:&f32 length:4 atIndex:9];
    [g_enc dispatchThreads:MTLSizeMake(hd, ws + nnew, 1)
        threadsPerThreadgroup:MTLSizeMake(hd < 64 ? hd : 64, 4, 1)];
    return true;
}

bool cactus_metal_encode_kv_append_ring_i8_m(const void* src, void* int8base, void* scalebase,
    uint32_t kv_heads, uint32_t hdim, uint32_t current_len, uint32_t group_size, uint32_t M,
    uint32_t sink, uint32_t W, size_t src_bytes, size_t int8_bytes, size_t scale_bytes) {
    if (!ctx().ok) return false;
    uint32_t num_groups = (hdim + group_size - 1)/group_size, n = M*kv_heads*num_groups;
    ensureEncoder();
    [g_enc setComputePipelineState:ctx().psoKvAppendRingM];
    setBufAt(src, src_bytes, 0); setBufAt(int8base, int8_bytes, 1); setBufAt(scalebase, scale_bytes, 2);
    [g_enc setBytes:&kv_heads length:4 atIndex:3]; [g_enc setBytes:&hdim length:4 atIndex:4];
    [g_enc setBytes:&current_len length:4 atIndex:5]; [g_enc setBytes:&group_size length:4 atIndex:6];
    [g_enc setBytes:&M length:4 atIndex:7];
    [g_enc setBytes:&sink length:4 atIndex:8]; [g_enc setBytes:&W length:4 atIndex:9];
    [g_enc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(n<256?n:256,1,1)];
    return true;
}

static bool encode_kv_append_sliding(const void* src, void* int8base, void* scalebase,
    uint32_t kv_heads, uint32_t hdim, uint32_t keep_sink, uint32_t remaining, uint32_t shift_src,
    uint32_t group_size, uint32_t M, size_t src_bytes, size_t int8_bytes, size_t scale_bytes) {
    if (!ctx().ok) return false;
    uint32_t num_groups = (hdim + group_size - 1)/group_size, per = kv_heads*num_groups;
    id<MTLBuffer> scrI8 = recycled((size_t)(remaining?remaining:1)*kv_heads*hdim);
    id<MTLBuffer> scrSc = recycled((size_t)(remaining?remaining:1)*kv_heads*num_groups*sizeof(float));
    ensureEncoder();
    if (remaining > 0) {
        uint32_t n = remaining*per;
        [g_enc setComputePipelineState:ctx().psoSlideS];
        setBufAt(int8base, int8_bytes, 0); setBufAt(scalebase, scale_bytes, 1);
        [g_enc setBuffer:scrI8 offset:0 atIndex:2]; [g_enc setBuffer:scrSc offset:0 atIndex:3];
        [g_enc setBytes:&kv_heads length:4 atIndex:4]; [g_enc setBytes:&hdim length:4 atIndex:5];
        [g_enc setBytes:&group_size length:4 atIndex:6]; [g_enc setBytes:&shift_src length:4 atIndex:7];
        [g_enc setBytes:&remaining length:4 atIndex:8];
        [g_enc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(n<256?n:256,1,1)];
    }
    uint32_t n2 = (remaining+M)*per;
    [g_enc setComputePipelineState:(M == 1 ? ctx().psoSlideR : ctx().psoSlideRM)];
    setBufAt(src, src_bytes, 0); setBufAt(int8base, int8_bytes, 1); setBufAt(scalebase, scale_bytes, 2);
    [g_enc setBuffer:scrI8 offset:0 atIndex:3]; [g_enc setBuffer:scrSc offset:0 atIndex:4];
    [g_enc setBytes:&kv_heads length:4 atIndex:5]; [g_enc setBytes:&hdim length:4 atIndex:6];
    [g_enc setBytes:&group_size length:4 atIndex:7]; [g_enc setBytes:&keep_sink length:4 atIndex:8];
    [g_enc setBytes:&remaining length:4 atIndex:9];
    if (M != 1) [g_enc setBytes:&M length:4 atIndex:10];
    [g_enc dispatchThreads:MTLSizeMake(n2,1,1) threadsPerThreadgroup:MTLSizeMake(n2<256?n2:256,1,1)];
    return true;
}

bool cactus_metal_encode_kv_append_sliding_i8(const void* src, void* int8base, void* scalebase,
    uint32_t kv_heads, uint32_t hdim, uint32_t keep_sink, uint32_t remaining, uint32_t shift_src,
    uint32_t group_size, size_t src_bytes, size_t int8_bytes, size_t scale_bytes) {
    return encode_kv_append_sliding(src, int8base, scalebase, kv_heads, hdim, keep_sink, remaining,
        shift_src, group_size, 1, src_bytes, int8_bytes, scale_bytes);
}

bool cactus_metal_encode_kv_append_sliding_i8_m(const void* src, void* int8base, void* scalebase,
    uint32_t kv_heads, uint32_t hdim, uint32_t keep_sink, uint32_t remaining, uint32_t shift_src,
    uint32_t group_size, uint32_t M, size_t src_bytes, size_t int8_bytes, size_t scale_bytes) {
    return encode_kv_append_sliding(src, int8base, scalebase, kv_heads, hdim, keep_sink, remaining,
        shift_src, group_size, M, src_bytes, int8_bytes, scale_bytes);
}

void cactus_metal_quant_matmul(const CactusQuantMatrix* W, const __fp16* A,
                               uint32_t M, __fp16* C) {
    const uint32_t gs = W->group_size;
    const bool fast = ctx().ok && M == 1 && quant_fast_eligible(W);

    if (!fast) { cactus_quant_matmul(W, A, M, C); return; }

    @autoreleasepool {
        const uint32_t K = W->K, N = W->N, ng = W->num_groups;
        const uint32_t pgb = (gs * 4u + 7u) / 8u;

        ResW& rw = resident(W);
        if (!rw.ok) { cactus_quant_matmul(W, A, M, C); return; }
        id<MTLBuffer> bx     = buf(A, (size_t)K * sizeof(__fp16));
        id<MTLBuffer> brecip = rw.recip;
        id<MTLBuffer> bls    = rw.lsign;
        id<MTLBuffer> brs    = rw.rsign;
        id<MTLBuffer> bperm  = rw.perm;
        id<MTLBuffer> bpk    = rw.packed;
        id<MTLBuffer> bcb    = rw.codebook;
        id<MTLBuffer> bnorm  = rw.norms;
        id<MTLBuffer> bcode  = owned_shared((size_t)ng * gs * sizeof(__fp16));
        id<MTLBuffer> by     = owned_shared((size_t)N * sizeof(__fp16));

        id<MTLCommandBuffer> cmd = [ctx().queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];

        bool simdT = (gs==128u && ctx().psoTsimd);
        [e setComputePipelineState:(simdT?ctx().psoTsimd:ctx().psoT)];
        [e setBuffer:bx offset:0 atIndex:0];  [e setBuffer:brecip offset:rw.rc_off atIndex:1];
        [e setBuffer:bls offset:rw.ls_off atIndex:2]; [e setBuffer:brs offset:rw.rs_off atIndex:3];
        [e setBuffer:bperm offset:rw.pm_off atIndex:4]; [e setBuffer:bcode offset:0 atIndex:5];
        [e setBytes:&gs length:sizeof(uint32_t) atIndex:6];
        [e setThreadgroupMemoryLength:gs * sizeof(float) atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake(ng,1,1) threadsPerThreadgroup:MTLSizeMake(simdT?32u:(gs>1024u?1024u:gs),1,1)];

        [e setComputePipelineState:ctx().psoG];
        [e setBuffer:bcode offset:0 atIndex:0]; [e setBuffer:bpk offset:rw.packed_off atIndex:1];
        [e setBuffer:bcb offset:rw.cb_off atIndex:2];   [e setBuffer:bnorm offset:rw.norms_off atIndex:3];
        [e setBuffer:by offset:0 atIndex:4];
        [e setBytes:&gs length:sizeof(uint32_t) atIndex:5];
        [e setBytes:&ng length:sizeof(uint32_t) atIndex:6];
        [e setBytes:&pgb length:sizeof(uint32_t) atIndex:7];
        [e setBytes:&N length:sizeof(uint32_t) atIndex:8];
        [e setBytes:&rw.il length:sizeof(uint32_t) atIndex:9];
        const uint32_t ROWS = 8;
        [e dispatchThreadgroups:MTLSizeMake((N+ROWS-1)/ROWS,1,1)
            threadsPerThreadgroup:MTLSizeMake(ROWS*32,1,1)];

        [e endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        std::memcpy(C, [by contents], (size_t)N * sizeof(__fp16));
    }
}

#endif
