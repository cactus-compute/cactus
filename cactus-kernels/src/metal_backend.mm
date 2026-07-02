

#include "metal_backend.h"

#if CACTUS_HAS_METAL

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <vector>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <unordered_map>
#include <map>
#include <utility>
#include <cstdint>
#include <mutex>

#include "cactus_kernels_msl.h"

namespace {

struct MetalCtx {
    id<MTLDevice> dev = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> psoT = nil;
    id<MTLComputePipelineState> psoTsimd = nil;
    id<MTLComputePipelineState> psoG = nil, psoGn1 = nil;
    id<MTLComputePipelineState> psoG2 = nil, psoG2n1 = nil, psoGF16 = nil;
    id<MTLComputePipelineState> psoSoftmaxR = nil, psoTopk = nil, psoMoeT = nil, psoMoeUp = nil, psoMoeDown = nil, psoMoeAcc = nil;
    id<MTLComputePipelineState> psoG2fused = nil, psoRms2Add = nil;
    id<MTLComputePipelineState> psoGfused = nil;
    id<MTLComputePipelineState> psoTm = nil, psoGm = nil;
    id<MTLComputePipelineState> psoGmma = nil;
    id<MTLComputePipelineState> psoRotate = nil;
    id<MTLComputePipelineState> psoEmbDq = nil, psoRowdot = nil, psoColdot = nil, psoColdotC = nil;
    id<MTLComputePipelineState> psoSoftcap = nil, psoRmsRope = nil;
    id<MTLComputePipelineState> psoArgP = nil, psoArgC = nil;
    id<MTLComputePipelineState> psoRmsAddRms = nil, psoTswi = nil, psoGfusedSwi = nil;
    id<MTLComputePipelineState> psoTmulti = nil, psoGfusedResid = nil;
    id<MTLComputePipelineState> psoEmbO = nil, psoEmbH = nil;
    id<MTLComputePipelineState> psoEmbOm = nil, psoEmbHm = nil;
    id<MTLComputePipelineState> psoCopy=nil, psoBinary=nil, psoScalar=nil, psoUnary=nil, psoRms=nil, psoSwiglu=nil, psoRmsAdd=nil;
    id<MTLComputePipelineState> psoCF16F32=nil, psoCF32F16=nil, psoCI8F16=nil, psoCF16I8=nil;
    id<MTLComputePipelineState> psoAttn=nil, psoAttnC=nil, psoAttnFused=nil, psoStrided=nil, psoScatter=nil, psoBcast=nil, psoKvAppend=nil;
    id<MTLComputePipelineState> psoAttnPre=nil, psoAttnPreMma2=nil;
    id<MTLComputePipelineState> psoKvAppendM=nil, psoKvAppendRingM=nil;
    id<MTLComputePipelineState> psoSlideS=nil, psoSlideR=nil, psoSlideRM=nil;
    id<MTLComputePipelineState> psoRope=nil;
    id<MTLComputePipelineState> psoArgmax=nil;
    id<MTLComputePipelineState> psoGather=nil;
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
        psoT=pso("cq4_transform"); psoTsimd=pso("cq4_transform_simd"); psoG=pso("cq4_gemv"); psoGn1=pso("cq4_gemv_n1");
        psoG2=pso("cq2_gemv"); psoG2n1=pso("cq2_gemv_n1"); psoGF16=pso("gemv_f16");
        psoG2fused=pso("cq2_gemv_fused"); psoRms2Add=pso("rms2_add_clip_f16");
        psoSoftmaxR=pso("softmax_rows_f16"); psoTopk=pso("topk_row_f16");
        psoMoeT=pso("cq2_moe_transform"); psoMoeUp=pso("cq2_moe_gemv_up");
        psoMoeDown=pso("cq2_moe_gemv_down"); psoMoeAcc=pso("moe_accum_f16");
        psoGfused=pso("cq4_gemv_fused");
        psoTm=pso("cq4_transform_m"); psoGm=pso("cq4_gemm"); psoGmma=pso("cq4_gemm_mma");
        psoRotate=pso("lmhead_rotate");
        psoEmbDq=pso("emb_ortho_dequant"); psoRowdot=pso("rot_rowdot");
        psoColdot=pso("rot_coldot"); psoColdotC=pso("rot_coldot_combine");
        psoSoftcap=pso("softcap_f16"); psoRmsRope=pso("rms_rope_f16");
        psoArgP=pso("argmax_part"); psoArgC=pso("argmax_combine");
        psoRmsAddRms=pso("rms_norm_add_rms_f16"); psoTswi=pso("cq4_transform_swiglu"); psoGfusedSwi=pso("cq4_gemv_fused_swiglu");
        psoTmulti=pso("cq4_transform_multi_resid"); psoGfusedResid=pso("cq4_gemv_fused_resid");
        psoEmbO=pso("emb_ortho"); psoEmbH=pso("emb_hadamard");
        psoEmbOm=pso("emb_ortho_m"); psoEmbHm=pso("emb_hadamard_m");
        psoGather=pso("gather_f16");
        psoCopy=pso("copy_bytes"); psoBinary=pso("binary_f16"); psoScalar=pso("scalar_f16");
        psoUnary=pso("unary_f16"); psoRms=pso("rms_norm_f16"); psoSwiglu=pso("swiglu_f16"); psoRmsAdd=pso("rms_norm_add_f16");
        psoCF16F32=pso("cast_f16_f32"); psoCF32F16=pso("cast_f32_f16");
        psoCI8F16=pso("cast_i8_f16"); psoCF16I8=pso("cast_f16_i8");
        psoAttn=pso("attn_decode_i8"); psoAttnC=pso("attn_decode_combine"); psoAttnFused=pso("attn_decode_fused_i8");
        psoStrided=pso("strided_copy_f16"); psoBcast=pso("bcast_binary_f16");
        psoAttnPre=pso("attn_prefill_i8"); psoAttnPreMma2=pso("attn_prefill_mma2"); psoKvAppendM=pso("kv_append_i8_m");
        psoKvAppendRingM=pso("kv_append_ring_i8_m");
        psoSlideS=pso("kv_slide_save"); psoSlideR=pso("kv_slide_restore"); psoSlideRM=pso("kv_slide_restore_m");
        psoScatter=pso("strided_scatter_f16"); psoKvAppend=pso("kv_append_i8"); psoRope=pso("rope_f16");
        psoArgmax=pso("argmax_logits");
        dummy=[dev newBufferWithLength:16 options:MTLResourceStorageModeShared];
        ok = psoT&&psoG&&psoGfused&&psoTm&&psoGm&&psoRotate&&psoEmbO&&psoEmbH&&psoEmbOm&&psoEmbHm&&psoCopy&&psoBinary&&psoScalar&&psoUnary&&psoRms&&psoSwiglu&&psoRmsAdd&&psoCF16F32&&psoCF32F16&&psoCI8F16&&psoCF16I8
             &&psoAttn&&psoAttnC&&psoAttnPre&&psoAttnPreMma2&&psoKvAppendM&&psoKvAppendRingM&&psoSlideS&&psoSlideR&&psoSlideRM&&psoStrided&&psoBcast&&psoScatter&&psoKvAppend&&psoRope&&psoArgmax&&psoGather
             &&psoEmbDq&&psoRowdot&&psoColdot&&psoColdotC&&psoSoftcap&&psoRmsRope&&psoArgP&&psoArgC
             &&psoRmsAddRms&&psoTswi&&psoGfusedSwi&&psoAttnFused&&psoGn1&&psoTmulti&&psoGfusedResid&&psoG2&&psoG2n1&&psoGF16
             &&psoSoftmaxR&&psoTopk&&psoMoeT&&psoMoeUp&&psoMoeDown&&psoMoeAcc&&psoG2fused&&psoRms2Add;
    }}
};

MetalCtx& ctx() { static MetalCtx c; return c; }

inline id<MTLBuffer> buf(const void* p, size_t bytes) {
    if (!p || bytes == 0) return nil;
    return [ctx().dev newBufferWithBytes:p length:bytes options:MTLResourceStorageModeShared];
}

struct ResW {
    id<MTLBuffer> packed=nil, norms=nil, codebook=nil, lsign=nil, rsign=nil, perm=nil, recip=nil;
    id<MTLBuffer> rotation=nil;
};
std::unordered_map<const void*, ResW> g_resident;
std::mutex g_resident_mu;

void deinterleave_4row(const CactusQuantMatrix* W,
                       uint8_t* packed_nat, __fp16* norms_nat) {
    const uint32_t N=W->N, ng=W->num_groups, gs=W->group_size, pgb=(gs*4u+7u)/8u, NB=N/4;
    std::memset(packed_nat, 0, (size_t)N*ng*pgb);
    const uint8_t* il=W->packed_indices;
    for (uint32_t nb=0; nb<NB; ++nb) for (uint32_t g=0; g<ng; ++g) {
        const uint8_t* panel = il + ((size_t)nb*ng+g)*4*(size_t)pgb;
        for (uint32_t r=0; r<4; ++r) {
            uint32_t n = nb*4+r;
            norms_nat[(size_t)n*ng+g] = W->norms[((size_t)nb*ng+g)*4 + r];
            uint8_t* row = packed_nat + ((size_t)n*ng+g)*pgb;
            for (uint32_t v=0; v<gs/16; ++v) for (uint32_t b=0; b<4; ++b) {
                uint8_t lo=panel[(2*v)*16+r*4+b], hi=panel[(2*v+1)*16+r*4+b];
                uint32_t k0=16*v+b, k1=16*v+4+b, k2=16*v+8+b, k3=16*v+12+b;
                row[k0>>1] |= (k0&1)?((lo&0xF)<<4):(lo&0xF);
                row[k1>>1] |= (k1&1)?((lo>>4)<<4):(lo>>4);
                row[k2>>1] |= (k2&1)?((hi&0xF)<<4):(hi&0xF);
                row[k3>>1] |= (k3&1)?((hi>>4)<<4):(hi>>4);
            }
        }
    }
}

// 2-bit interleaved panels: byte at panel[chunk*16 + set*4 + r] holds row r's
// four consecutive weights k = chunk*16 + set*4 .. +3 (2-bit little-endian),
// which is exactly one byte of the natural row-major packing.
void deinterleave_4row_2bit(const CactusQuantMatrix* W,
                            uint8_t* packed_nat, __fp16* norms_nat) {
    const uint32_t N=W->N, ng=W->num_groups, gs=W->group_size, pgb=(gs*2u+7u)/8u, NB=N/4;
    const uint8_t* il=W->packed_indices;
    for (uint32_t nb=0; nb<NB; ++nb) for (uint32_t g=0; g<ng; ++g) {
        const uint8_t* panel = il + ((size_t)nb*ng+g)*4*(size_t)pgb;
        for (uint32_t r=0; r<4; ++r) {
            uint32_t n = nb*4+r;
            norms_nat[(size_t)n*ng+g] = W->norms[((size_t)nb*ng+g)*4 + r];
            uint8_t* row = packed_nat + ((size_t)n*ng+g)*pgb;
            for (uint32_t chunk=0; chunk<gs/16; ++chunk)
                for (uint32_t set=0; set<4; ++set)
                    row[chunk*4+set] = panel[chunk*16 + set*4 + r];
        }
    }
}

ResW& resident(const CactusQuantMatrix* W) {
    std::lock_guard<std::mutex> lk(g_resident_mu);
    auto it = g_resident.find(W->packed_indices);
    if (it != g_resident.end()) return it->second;
    const uint32_t K=W->K, N=W->N, ng=W->num_groups, gs=W->group_size, pgb=(gs*W->bits+7u)/8u;
    ResW r;
    r.codebook = buf(W->codebook, 16*sizeof(__fp16));
    r.lsign    = buf(W->left_signs, gs);
    r.rsign    = buf(W->right_signs, gs);
    r.perm     = buf(W->permutation, (size_t)gs*sizeof(uint32_t));
    r.recip    = buf(W->input_scale_recip, (size_t)K*sizeof(__fp16));
    if (W->rotation) r.rotation = buf(W->rotation, (size_t)K*K*sizeof(__fp16));
    if (W->flags & CACTUS_QUANT_FLAG_INTERLEAVED_4ROW) {
        size_t pkb=(size_t)N*ng*pgb, nmb=(size_t)N*ng*sizeof(__fp16);
        r.packed = [ctx().dev newBufferWithLength:pkb options:MTLResourceStorageModeShared];
        r.norms  = [ctx().dev newBufferWithLength:nmb options:MTLResourceStorageModeShared];
        if (W->bits == 4)
            deinterleave_4row(W, (uint8_t*)r.packed.contents, (__fp16*)r.norms.contents);
        else if (W->bits == 2)
            deinterleave_4row_2bit(W, (uint8_t*)r.packed.contents, (__fp16*)r.norms.contents);
        else {
            r.packed = nil; r.norms = nil;
        }
    } else {
        r.packed = buf(W->packed_indices, (size_t)N*ng*pgb);
        r.norms  = buf(W->norms, (size_t)N*ng*sizeof(__fp16));
    }
    return g_resident.emplace(W->packed_indices, r).first->second;
}

id<MTLCommandBuffer> g_cmd = nil;
id<MTLComputeCommandEncoder> g_enc = nil;
std::vector<id<MTLCommandBuffer>> g_inflight;
std::unordered_map<size_t, std::vector<id<MTLBuffer>>> g_free;
std::vector<id<MTLBuffer>> g_pending;
std::map<uintptr_t, id<MTLBuffer>> g_shared;
bool g_active = false;
id<MTLBuffer> g_code_buf = nil;
id<MTLBuffer> g_code_buf_m = nil;

inline size_t bucket(size_t b) { return (b + 4095) & ~size_t(4095); }

static const bool g_concurrent = (std::getenv("CACTUS_GPU_CONCURRENT") != nullptr);
bool g_manual = false;  // fused decode path: concurrent encoder, explicit barriers

id<MTLComputeCommandEncoder> ensureEncoder() {
    if (!g_enc) {
        // unretained references: every bound resource is kept alive by the
        // resident/recycle registries, so skip per-resource retain/release
        g_cmd = g_manual ? [ctx().queue commandBufferWithUnretainedReferences]
                         : [ctx().queue commandBuffer];
        g_enc = (g_concurrent || g_manual)
            ? [g_cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent]
            : [g_cmd computeCommandEncoder];
    }
    return g_enc;
}

// hazard(): ordering between dependent dispatches inside one logical op.
// Needed whenever the encoder is concurrent (env override or manual mode).
inline void hazard() {
    if ((g_concurrent || g_manual) && g_enc) [g_enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}
// barrier(): trailing barrier after an op. In manual mode the caller inserts
// explicit barriers at dependency edges instead, so this is a no-op there.
inline void barrier() {
    static const bool force = (std::getenv("CACTUS_FORCE_BARRIER") != nullptr);
    if (g_manual) return;
    if (g_concurrent || force) [g_enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

id<MTLBuffer> recycled(size_t bytes) {
    size_t bk = bucket(bytes ? bytes : 1);
    auto it = g_free.find(bk);
    id<MTLBuffer> b;
    if (it != g_free.end() && !it->second.empty()) { b = it->second.back(); it->second.pop_back(); }
    else b = [ctx().dev newBufferWithLength:bk options:MTLResourceStorageModeShared];
    g_pending.push_back(b);
    return b;
}

std::unordered_map<const void*, id<MTLBuffer>> g_readonly;
id<MTLBuffer> registerReadonly(const void* p, size_t bytes) {
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

std::pair<id<MTLBuffer>, size_t> bufForPtrOff(const void* p, size_t bytes) {
    uintptr_t a = reinterpret_cast<uintptr_t>(p);
    auto it = g_shared.upper_bound(a);
    if (it != g_shared.begin()) {
        --it;
        uintptr_t base = it->first;
        if (a < base + static_cast<uintptr_t>(it->second.length))
            return { it->second, static_cast<size_t>(a - base) };
    }
    auto rit = g_readonly.find(p);
    if (rit != g_readonly.end())
        return { rit->second, static_cast<size_t>(a & (uintptr_t)16383u) };

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

bool cactus_metal_concurrent() { return g_concurrent; }
void cactus_metal_barrier() { if ((g_concurrent || g_manual) && g_enc) [g_enc memoryBarrierWithScope:MTLBarrierScopeBuffers]; }

void cactus_metal_manual_begin() { g_manual = true; }
void cactus_metal_manual_end()   { g_manual = false; }

static const bool g_prof = (std::getenv("CACTUS_METAL_PROF") != nullptr);
static double g_prof_gpu_s = 0.0, g_prof_wall_s = 0.0;
static uint64_t g_prof_syncs = 0;
extern "C" void cactus_metal_prof_report() {
    if (!g_prof) return;
    fprintf(stderr, "[cactus-metal-prof] syncs=%llu gpu_busy=%.3fs sync_wall=%.3fs\n",
            (unsigned long long)g_prof_syncs, g_prof_gpu_s, g_prof_wall_s);
    g_prof_gpu_s = g_prof_wall_s = 0.0; g_prof_syncs = 0;
}

void cactus_metal_session_begin() {}

// Commit the current command buffer without waiting and keep encoding into a
// fresh one. Buffers on the same queue execute in submission order, so
// dependencies established by barriers still hold across the split.
void cactus_metal_session_flush() {
    if (!g_enc) return;
    [g_enc endEncoding]; [g_cmd commit];
    g_inflight.push_back(g_cmd);
    g_enc = nil; g_cmd = nil;
}

void cactus_metal_session_sync() {
    if (g_enc) {
        [g_enc endEncoding]; [g_cmd commit];
        g_inflight.push_back(g_cmd);
        g_enc = nil; g_cmd = nil;
    }
    if (!g_inflight.empty()) {
        if (g_prof) {
            static bool reg = (std::atexit([]{ cactus_metal_prof_report(); }), true); (void)reg;
            NSDate* t0 = [NSDate date];
            for (id<MTLCommandBuffer> c : g_inflight) [c waitUntilCompleted];
            g_prof_wall_s += -[t0 timeIntervalSinceNow];
            for (id<MTLCommandBuffer> c : g_inflight)
                g_prof_gpu_s += c.GPUEndTime - c.GPUStartTime;
            ++g_prof_syncs;
        } else {
            for (id<MTLCommandBuffer> c : g_inflight) [c waitUntilCompleted];
        }
        g_inflight.clear();
    }
    for (id<MTLBuffer> b : g_pending) g_free[(size_t)b.length].push_back(b);
    g_pending.clear();
}
void cactus_metal_session_end() {
    cactus_metal_session_sync();
}

void* cactus_metal_alloc_shared(size_t bytes) {
    if (!ctx().ok) return nullptr;
    size_t bk = bucket(bytes ? bytes : 1);
    id<MTLBuffer> b = nil;
    auto it = g_free.find(bk);
    if (it != g_free.end() && !it->second.empty()) { b = it->second.back(); it->second.pop_back(); }
    else b = [ctx().dev newBufferWithLength:bk options:MTLResourceStorageModeShared];
    void* c = [b contents];
    g_shared[reinterpret_cast<uintptr_t>(c)] = b;
    return c;
}
void cactus_metal_free_shared(void* contents) {
    auto it = g_shared.find(reinterpret_cast<uintptr_t>(contents));
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
    barrier();
    return true;
}
bool cactus_metal_encode_binary(int op, void* out, const void* a, const void* b, size_t n) {
    if (!ctx().ok) return false;
    ensureEncoder();
    uint32_t nn=(uint32_t)n; int o=op;
    [g_enc setComputePipelineState:ctx().psoBinary];
    setBufAt(a, n*2, 0); setBufAt(b, n*2, 1); setBufAt(out, n*2, 2);
    [g_enc setBytes:&nn length:4 atIndex:3]; [g_enc setBytes:&o length:4 atIndex:4];
    [g_enc dispatchThreads:MTLSizeMake(nn,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    barrier();
    return true;
}
bool cactus_metal_encode_scalar(int op, void* out, const void* in, size_t n, float param) {
    if (!ctx().ok) return false;
    ensureEncoder();
    uint32_t nn=(uint32_t)n; int o=op; float p=param;
    [g_enc setComputePipelineState:ctx().psoScalar];
    setBufAt(in, n*2, 0); setBufAt(out, n*2, 1);
    [g_enc setBytes:&nn length:4 atIndex:2]; [g_enc setBytes:&o length:4 atIndex:3]; [g_enc setBytes:&p length:4 atIndex:4];
    [g_enc dispatchThreads:MTLSizeMake(nn,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    barrier();
    return true;
}
bool cactus_metal_encode_unary(int op, void* out, const void* in, size_t n) {
    if (!ctx().ok) return false;
    ensureEncoder();
    uint32_t nn=(uint32_t)n; int o=op;
    [g_enc setComputePipelineState:ctx().psoUnary];
    setBufAt(in, n*2, 0); setBufAt(out, n*2, 1);
    [g_enc setBytes:&nn length:4 atIndex:2]; [g_enc setBytes:&o length:4 atIndex:3];
    [g_enc dispatchThreads:MTLSizeMake(nn,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    barrier();
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
    barrier();
    return true;
}
bool cactus_metal_encode_rope(void* out, const void* x, const void* cos, const void* sin,
                              uint32_t heads, uint32_t head_dim) {
    if (!ctx().ok) return false;
    ensureEncoder();
    uint32_t total=heads*head_dim;
    [g_enc setComputePipelineState:ctx().psoRope];
    setBufAt(x, (size_t)total*2, 0); setBufAt(out, (size_t)total*2, 1);
    setBufAt(cos, (size_t)head_dim*2, 2); setBufAt(sin, (size_t)head_dim*2, 3);
    [g_enc setBytes:&heads length:4 atIndex:4]; [g_enc setBytes:&head_dim length:4 atIndex:5];
    [g_enc dispatchThreads:MTLSizeMake(total,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    barrier();
    return true;
}
bool cactus_metal_encode_rms_norm(void* out, const void* in, const void* weight,
                                  size_t rows, size_t dim, float eps) {
    if (!ctx().ok) return false;
    ensureEncoder();
    uint32_t d=(uint32_t)dim; float e=eps;
    [g_enc setComputePipelineState:ctx().psoRms];
    setBufAt(in, rows*dim*2, 0); setBufAt(weight, dim*2, 1); setBufAt(out, rows*dim*2, 2);
    [g_enc setBytes:&d length:4 atIndex:3]; [g_enc setBytes:&e length:4 atIndex:4];
    [g_enc setThreadgroupMemoryLength:256*sizeof(float) atIndex:0];
    [g_enc dispatchThreadgroups:MTLSizeMake(rows,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    barrier();
    return true;
}
bool cactus_metal_encode_rms_norm_add(void* out, const void* in, const void* weight, const void* res,
                                      size_t rows, size_t dim, float eps, float out_scale) {
    if (!ctx().ok) return false;
    ensureEncoder();
    uint32_t d=(uint32_t)dim; float e=eps; float s=out_scale;
    [g_enc setComputePipelineState:ctx().psoRmsAdd];
    setBufAt(in, rows*dim*2, 0); setBufAt(weight, dim*2, 1); setBufAt(res, rows*dim*2, 2); setBufAt(out, rows*dim*2, 3);
    [g_enc setBytes:&d length:4 atIndex:4]; [g_enc setBytes:&e length:4 atIndex:5]; [g_enc setBytes:&s length:4 atIndex:6];
    [g_enc setThreadgroupMemoryLength:256*sizeof(float) atIndex:0];
    [g_enc dispatchThreadgroups:MTLSizeMake(rows,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    barrier();
    return true;
}

static inline bool quant_fast_eligible(const CactusQuantMatrix* W);
static id<MTLComputePipelineState> gemv_pso_for(uint32_t N, uint32_t& rows_per_tg, uint32_t bits = 4);

bool cactus_metal_encode_rms_norm_add_rms(void* h_out, void* xn_out, const void* in, const void* w1,
                                          const void* res, const void* w2,
                                          size_t rows, size_t dim, float eps, float out_scale) {
    if (!ctx().ok || !ctx().psoRmsAddRms) return false;
    ensureEncoder();
    uint32_t d=(uint32_t)dim; float e=eps; float s=out_scale;
    [g_enc setComputePipelineState:ctx().psoRmsAddRms];
    setBufAt(in, rows*dim*2, 0); setBufAt(w1, dim*2, 1); setBufAt(res, rows*dim*2, 2); setBufAt(h_out, rows*dim*2, 3);
    setBufAt(w2, dim*2, 4); setBufAt(xn_out, rows*dim*2, 5);
    [g_enc setBytes:&d length:4 atIndex:6]; [g_enc setBytes:&e length:4 atIndex:7]; [g_enc setBytes:&s length:4 atIndex:8];
    [g_enc setThreadgroupMemoryLength:256*sizeof(float) atIndex:0];
    [g_enc dispatchThreadgroups:MTLSizeMake(rows,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    barrier();
    return true;
}

bool cactus_metal_encode_quant_matmul_swiglu(void* out, const void* gate, const void* up,
                                             float swi_scale, const CactusQuantMatrix* W) {
    const uint32_t gs=W->group_size, K=W->K, N=W->N, ng=W->num_groups, pgb=(gs*4u+7u)/8u;
    if (!ctx().ok || gs != 128u || W->bits != 4 || !quant_fast_eligible(W)) return false;
    ResW& rw = resident(W);
    ensureEncoder();
    uint32_t ROWS=8;
    if (K<=4096u && ctx().psoGfusedSwi) {
        [g_enc setComputePipelineState:ctx().psoGfusedSwi];
        setBufAt(gate, (size_t)K*2, 0);                  [g_enc setBuffer:rw.packed offset:0 atIndex:1];
        [g_enc setBuffer:rw.codebook offset:0 atIndex:2];[g_enc setBuffer:rw.norms offset:0 atIndex:3];
        setBufAt(out, (size_t)N*2, 4);
        [g_enc setBytes:&gs length:4 atIndex:5]; [g_enc setBytes:&ng length:4 atIndex:6];
        [g_enc setBytes:&pgb length:4 atIndex:7]; [g_enc setBytes:&N length:4 atIndex:8];
        [g_enc setBuffer:rw.recip offset:0 atIndex:9]; [g_enc setBuffer:rw.lsign offset:0 atIndex:10];
        [g_enc setBuffer:rw.rsign offset:0 atIndex:11]; [g_enc setBuffer:rw.perm offset:0 atIndex:12];
        setBufAt(up, (size_t)K*2, 13);
        [g_enc setBytes:&swi_scale length:4 atIndex:14];
        [g_enc setThreadgroupMemoryLength:(size_t)K*2 + ROWS*128*sizeof(float) atIndex:0];
        [g_enc dispatchThreadgroups:MTLSizeMake((N+ROWS*4-1)/(ROWS*4),1,1) threadsPerThreadgroup:MTLSizeMake(ROWS*32,1,1)];
        barrier();
        return true;
    }
    if (!ctx().psoTswi) return false;
    size_t code_bytes = (size_t)ng*gs*sizeof(__fp16);
    if (!g_code_buf || (size_t)g_code_buf.length < code_bytes)
        g_code_buf = [ctx().dev newBufferWithLength:code_bytes options:MTLResourceStorageModeShared];
    [g_enc setComputePipelineState:ctx().psoTswi];
    setBufAt(gate, (size_t)K*2, 0); setBufAt(up, (size_t)K*2, 1);
    [g_enc setBuffer:rw.recip offset:0 atIndex:2];
    [g_enc setBuffer:rw.lsign offset:0 atIndex:3]; [g_enc setBuffer:rw.rsign offset:0 atIndex:4];
    [g_enc setBuffer:rw.perm offset:0 atIndex:5];  [g_enc setBuffer:g_code_buf offset:0 atIndex:6];
    [g_enc setBytes:&swi_scale length:4 atIndex:7];
    [g_enc setThreadgroupMemoryLength:gs*sizeof(float) atIndex:0];
    [g_enc dispatchThreadgroups:MTLSizeMake(ng,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    hazard();
    uint32_t rpt_; [g_enc setComputePipelineState:gemv_pso_for(N, rpt_)];
    [g_enc setBuffer:g_code_buf offset:0 atIndex:0]; [g_enc setBuffer:rw.packed offset:0 atIndex:1];
    [g_enc setBuffer:rw.codebook offset:0 atIndex:2]; [g_enc setBuffer:rw.norms offset:0 atIndex:3];
    setBufAt(out, (size_t)N*2, 4);
    [g_enc setBytes:&gs length:4 atIndex:5]; [g_enc setBytes:&ng length:4 atIndex:6];
    [g_enc setBytes:&pgb length:4 atIndex:7]; [g_enc setBytes:&N length:4 atIndex:8];
    [g_enc dispatchThreadgroups:MTLSizeMake((N+rpt_-1)/rpt_,1,1) threadsPerThreadgroup:MTLSizeMake(ROWS*32,1,1)];
    barrier();
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
    barrier();
    return true;
}

bool cactus_metal_encode_rms_rope(void* out, const void* in, const void* weight,
                                  const void* cos, const void* sin,
                                  uint32_t heads, uint32_t head_dim, float eps) {
    if (!ctx().ok || !ctx().psoRmsRope) return false;
    ensureEncoder();
    float e=eps;
    [g_enc setComputePipelineState:ctx().psoRmsRope];
    setBufAt(in, (size_t)heads*head_dim*2, 0); setBufAt(weight, (size_t)head_dim*2, 1);
    setBufAt(cos, (size_t)head_dim*2, 2); setBufAt(sin, (size_t)head_dim*2, 3);
    setBufAt(out, (size_t)heads*head_dim*2, 4);
    [g_enc setBytes:&head_dim length:4 atIndex:5]; [g_enc setBytes:&e length:4 atIndex:6];
    [g_enc setThreadgroupMemoryLength:256*sizeof(float) atIndex:0];
    [g_enc dispatchThreadgroups:MTLSizeMake(heads,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    barrier();
    return true;
}
bool cactus_metal_encode_argmax(const void* logits, uint32_t vocab, void* out3) {
    if (!ctx().ok) return false;
    ensureEncoder();
    uint32_t V = vocab;
    if (ctx().psoArgP && ctx().psoArgC && V >= 65536u) {
        const uint32_t NC = 32, T = 256;
        id<MTLBuffer> parts = recycled((size_t)NC*3*sizeof(float));
        [g_enc setComputePipelineState:ctx().psoArgP];
        setBufAt(logits, (size_t)vocab*2, 0);
        [g_enc setBuffer:parts offset:0 atIndex:1];
        [g_enc setBytes:&V length:4 atIndex:2]; [g_enc setBytes:&NC length:4 atIndex:3];
        [g_enc setThreadgroupMemoryLength:T*sizeof(float) atIndex:0];
        [g_enc setThreadgroupMemoryLength:T*sizeof(uint) atIndex:1];
        [g_enc setThreadgroupMemoryLength:T*sizeof(float) atIndex:2];
        [g_enc dispatchThreadgroups:MTLSizeMake(NC,1,1) threadsPerThreadgroup:MTLSizeMake(T,1,1)];
        hazard();
        [g_enc setComputePipelineState:ctx().psoArgC];
        [g_enc setBuffer:parts offset:0 atIndex:0];
        setBufAt(out3, 3*sizeof(float), 1);
        [g_enc setBytes:&NC length:4 atIndex:2];
        [g_enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        barrier();
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
    barrier();
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
    barrier();
    return true;
}
static inline bool quant_fast_eligible(const CactusQuantMatrix* W) {
    bool bits_ok = (W->bits == 4 && W->group_size >= 128 && (W->group_size % 128) == 0)
                || (W->bits == 2 && W->group_size == 128);
    return bits_ok && (W->N % 4) == 0 &&
           !(W->flags & CACTUS_QUANT_FLAG_ORTHOGONAL) && W->input_scale_recip && W->left_signs &&
           W->right_signs && W->permutation && W->codebook && W->norms && W->packed_indices;
}

static id<MTLComputePipelineState> gemv_pso_for(uint32_t N, uint32_t& rows_per_tg, uint32_t bits) {
    if (bits == 2) {
        if (N <= 512u) { rows_per_tg = 8u;  return ctx().psoG2n1; }
        rows_per_tg = 32u; return ctx().psoG2;
    }
    if (N <= 512u) { rows_per_tg = 8u;  return ctx().psoGn1; }
    rows_per_tg = 32u; return ctx().psoG;
}

bool cactus_metal_encode_quant_matmul(void* out, const void* lhs, const CactusQuantMatrix* W) {
    const uint32_t gs=W->group_size, K=W->K, N=W->N, ng=W->num_groups, pgb=(gs*W->bits+7u)/8u;
    if (ctx().ok && (W->flags & CACTUS_QUANT_FLAG_ORTHOGONAL) && W->rotation && ng==1 && gs==K && (N%4)==0) {
        static void* ortho_code = nullptr; static uint32_t ortho_code_k = 0;
        if (ortho_code_k < K) { ortho_code = cactus_metal_alloc_shared((size_t)K*sizeof(__fp16)); ortho_code_k = K; }
        if (ortho_code) return cactus_metal_encode_quant_matmul_ortho(out, lhs, ortho_code, W);
    }
    bool fast = ctx().ok && quant_fast_eligible(W);
    if (!fast) return false;
    ResW& rw = resident(W);
    ensureEncoder();
    uint32_t ROWS=8;
    static const uint32_t fuse_kmax = []{
        const char* e = std::getenv("CACTUS_GEMV_FUSE_KMAX");
        return e ? (uint32_t)atoi(e) : 4096u;
    }();
    if (W->bits==2 && gs==128u && K<=fuse_kmax && ctx().psoG2fused) {
        [g_enc setComputePipelineState:ctx().psoG2fused];
        setBufAt(lhs, (size_t)K*2, 0);                   [g_enc setBuffer:rw.packed offset:0 atIndex:1];
        [g_enc setBuffer:rw.codebook offset:0 atIndex:2];[g_enc setBuffer:rw.norms offset:0 atIndex:3];
        setBufAt(out, (size_t)N*2, 4);
        [g_enc setBytes:&gs length:4 atIndex:5]; [g_enc setBytes:&ng length:4 atIndex:6];
        [g_enc setBytes:&pgb length:4 atIndex:7]; [g_enc setBytes:&N length:4 atIndex:8];
        [g_enc setBuffer:rw.recip offset:0 atIndex:9]; [g_enc setBuffer:rw.lsign offset:0 atIndex:10];
        [g_enc setBuffer:rw.rsign offset:0 atIndex:11]; [g_enc setBuffer:rw.perm offset:0 atIndex:12];
        [g_enc setThreadgroupMemoryLength:(size_t)K*2 + ROWS*128*sizeof(float) atIndex:0];
        [g_enc dispatchThreadgroups:MTLSizeMake((N+ROWS*4-1)/(ROWS*4),1,1) threadsPerThreadgroup:MTLSizeMake(ROWS*32,1,1)];
        barrier();
        return true;
    }
    if (W->bits==4 && gs==128u && K<=fuse_kmax && ctx().psoGfused) {
        // single-dispatch: each threadgroup transforms the input into
        // threadgroup memory, then computes its 8 output rows
        [g_enc setComputePipelineState:ctx().psoGfused];
        setBufAt(lhs, (size_t)K*2, 0);                   [g_enc setBuffer:rw.packed offset:0 atIndex:1];
        [g_enc setBuffer:rw.codebook offset:0 atIndex:2];[g_enc setBuffer:rw.norms offset:0 atIndex:3];
        setBufAt(out, (size_t)N*2, 4);
        [g_enc setBytes:&gs length:4 atIndex:5]; [g_enc setBytes:&ng length:4 atIndex:6];
        [g_enc setBytes:&pgb length:4 atIndex:7]; [g_enc setBytes:&N length:4 atIndex:8];
        [g_enc setBuffer:rw.recip offset:0 atIndex:9]; [g_enc setBuffer:rw.lsign offset:0 atIndex:10];
        [g_enc setBuffer:rw.rsign offset:0 atIndex:11]; [g_enc setBuffer:rw.perm offset:0 atIndex:12];
        [g_enc setThreadgroupMemoryLength:(size_t)K*2 + ROWS*128*sizeof(float) atIndex:0];
        [g_enc dispatchThreadgroups:MTLSizeMake((N+ROWS*4-1)/(ROWS*4),1,1) threadsPerThreadgroup:MTLSizeMake(ROWS*32,1,1)];
        barrier();
        return true;
    }

    size_t code_bytes = (size_t)ng*gs*sizeof(__fp16);
    id<MTLBuffer> code_buf;
    if (g_manual) {
        // distinct per-call buffer so independent matmuls in one stage don't alias
        code_buf = recycled(code_bytes);
    } else {
        if (!g_code_buf || (size_t)g_code_buf.length < code_bytes)
            g_code_buf = [ctx().dev newBufferWithLength:code_bytes options:MTLResourceStorageModeShared];
        code_buf = g_code_buf;
    }
    bool simdT = (gs==128u && ctx().psoTsimd);
    [g_enc setComputePipelineState:(simdT?ctx().psoTsimd:ctx().psoT)];
    setBufAt(lhs, (size_t)K*2, 0);                 [g_enc setBuffer:rw.recip offset:0 atIndex:1];
    [g_enc setBuffer:rw.lsign offset:0 atIndex:2]; [g_enc setBuffer:rw.rsign offset:0 atIndex:3];
    [g_enc setBuffer:rw.perm offset:0 atIndex:4];  [g_enc setBuffer:code_buf offset:0 atIndex:5];
    [g_enc setBytes:&gs length:4 atIndex:6]; [g_enc setThreadgroupMemoryLength:gs*sizeof(float) atIndex:0];
    [g_enc dispatchThreadgroups:MTLSizeMake(ng,1,1) threadsPerThreadgroup:MTLSizeMake(simdT?32u:(gs>1024u?1024u:gs),1,1)];
    hazard();
    uint32_t rpt_; [g_enc setComputePipelineState:gemv_pso_for(N, rpt_, W->bits)];
    [g_enc setBuffer:code_buf offset:0 atIndex:0]; [g_enc setBuffer:rw.packed offset:0 atIndex:1];
    [g_enc setBuffer:rw.codebook offset:0 atIndex:2]; [g_enc setBuffer:rw.norms offset:0 atIndex:3];
    setBufAt(out, (size_t)N*2, 4);
    [g_enc setBytes:&gs length:4 atIndex:5]; [g_enc setBytes:&ng length:4 atIndex:6];
    [g_enc setBytes:&pgb length:4 atIndex:7]; [g_enc setBytes:&N length:4 atIndex:8];
    [g_enc dispatchThreadgroups:MTLSizeMake((N+rpt_-1)/rpt_,1,1) threadsPerThreadgroup:MTLSizeMake(ROWS*32,1,1)];
    barrier();
    return true;
}

bool cactus_metal_encode_quant_matmul_many(void* const* outs, const void* lhs,
                                           const CactusQuantMatrix* const* Ws, uint32_t count) {
    if (!ctx().ok || count == 0) return false;
    for (uint32_t i = 0; i < count; ++i)
        if (!quant_fast_eligible(Ws[i]) || Ws[i]->group_size != 128u || Ws[i]->bits != 4) return false;
    ensureEncoder();
    const uint32_t ROWS = 8;
    std::vector<id<MTLBuffer>> codes(count);
    for (uint32_t i = 0; i < count; ++i) {
        const CactusQuantMatrix* W = Ws[i];
        const uint32_t gs=W->group_size, K=W->K, ng=W->num_groups;
        ResW& rw = resident(W);
        codes[i] = recycled((size_t)ng*gs*sizeof(__fp16));
        [g_enc setComputePipelineState:ctx().psoTsimd];
        setBufAt(lhs, (size_t)K*2, 0);                 [g_enc setBuffer:rw.recip offset:0 atIndex:1];
        [g_enc setBuffer:rw.lsign offset:0 atIndex:2]; [g_enc setBuffer:rw.rsign offset:0 atIndex:3];
        [g_enc setBuffer:rw.perm offset:0 atIndex:4];  [g_enc setBuffer:codes[i] offset:0 atIndex:5];
        [g_enc setBytes:&gs length:4 atIndex:6]; [g_enc setThreadgroupMemoryLength:gs*sizeof(float) atIndex:0];
        [g_enc dispatchThreadgroups:MTLSizeMake(ng,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    }
    hazard();
    for (uint32_t i = 0; i < count; ++i) {
        const CactusQuantMatrix* W = Ws[i];
        const uint32_t gs=W->group_size, N=W->N, ng=W->num_groups, pgb=(gs*4u+7u)/8u;
        ResW& rw = resident(W);
        uint32_t rpt_; [g_enc setComputePipelineState:gemv_pso_for(N, rpt_)];
        [g_enc setBuffer:codes[i] offset:0 atIndex:0]; [g_enc setBuffer:rw.packed offset:0 atIndex:1];
        [g_enc setBuffer:rw.codebook offset:0 atIndex:2]; [g_enc setBuffer:rw.norms offset:0 atIndex:3];
        setBufAt(outs[i], (size_t)N*2, 4);
        [g_enc setBytes:&gs length:4 atIndex:5]; [g_enc setBytes:&ng length:4 atIndex:6];
        [g_enc setBytes:&pgb length:4 atIndex:7]; [g_enc setBytes:&N length:4 atIndex:8];
        [g_enc dispatchThreadgroups:MTLSizeMake((N+rpt_-1)/rpt_,1,1) threadsPerThreadgroup:MTLSizeMake(ROWS*32,1,1)];
    }
    barrier();
    return true;
}

// Encode the residual+norm prologue fused into the transforms of up to 3
// matrices (one dispatch), then the matmuls' gemvs (after one hazard).
// h_out receives clamp(res + rms(in)*w1)*out_scale; the transforms consume
// xn = rms(h_out)*w2 computed in threadgroup memory.
bool cactus_metal_encode_quant_matmul_many_resid(void* const* outs, const CactusQuantMatrix* const* Ws, uint32_t count,
                                                 const void* in, const void* w1, const void* res, const void* w2,
                                                 void* h_out, float out_scale, float eps) {
    static const bool disabled = (std::getenv("CACTUS_NO_RESID") != nullptr);
    if (disabled || !ctx().ok || !ctx().psoTmulti || count == 0 || count > 3) return false;
    const uint32_t K = Ws[0]->K;
    for (uint32_t i = 0; i < count; ++i)
        if (!quant_fast_eligible(Ws[i]) || Ws[i]->group_size != 128u || Ws[i]->K != K || Ws[i]->bits != 4) return false;
    ensureEncoder();
    std::vector<id<MTLBuffer>> codes(count);
    ResW* rws[3];
    for (uint32_t i = 0; i < count; ++i) {
        rws[i] = &resident(Ws[i]);
        codes[i] = recycled((size_t)K*sizeof(__fp16));
    }
    uint32_t last = count - 1;
    [g_enc setComputePipelineState:ctx().psoTmulti];
    setBufAt(in, (size_t)K*2, 0); setBufAt(w1, (size_t)K*2, 1);
    setBufAt(res, (size_t)K*2, 2); setBufAt(w2, (size_t)K*2, 3);
    for (uint32_t i = 0; i < 3; ++i) {
        uint32_t m = i < count ? i : last;
        [g_enc setBuffer:rws[m]->recip offset:0 atIndex:4+i];
        [g_enc setBuffer:rws[m]->lsign offset:0 atIndex:7+i];
        [g_enc setBuffer:rws[m]->rsign offset:0 atIndex:10+i];
        [g_enc setBuffer:rws[m]->perm offset:0 atIndex:13+i];
        [g_enc setBuffer:codes[m] offset:0 atIndex:16+i];
    }
    setBufAt(h_out, (size_t)K*2, 19);
    [g_enc setBytes:&K length:4 atIndex:20];
    [g_enc setBytes:&out_scale length:4 atIndex:21];
    [g_enc setBytes:&eps length:4 atIndex:22];
    [g_enc setThreadgroupMemoryLength:(size_t)K*4 + (8*128+8)*sizeof(float) atIndex:0];
    [g_enc dispatchThreadgroups:MTLSizeMake((K/128u+7u)/8u, count, 1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    hazard();
    for (uint32_t i = 0; i < count; ++i) {
        const CactusQuantMatrix* W = Ws[i];
        const uint32_t gs=W->group_size, N=W->N, ng=W->num_groups, pgb=(gs*4u+7u)/8u;
        uint32_t rpt; id<MTLComputePipelineState> pso = gemv_pso_for(N, rpt);
        [g_enc setComputePipelineState:pso];
        [g_enc setBuffer:codes[i] offset:0 atIndex:0]; [g_enc setBuffer:rws[i]->packed offset:0 atIndex:1];
        [g_enc setBuffer:rws[i]->codebook offset:0 atIndex:2]; [g_enc setBuffer:rws[i]->norms offset:0 atIndex:3];
        setBufAt(outs[i], (size_t)N*2, 4);
        [g_enc setBytes:&gs length:4 atIndex:5]; [g_enc setBytes:&ng length:4 atIndex:6];
        [g_enc setBytes:&pgb length:4 atIndex:7]; [g_enc setBytes:&N length:4 atIndex:8];
        [g_enc dispatchThreadgroups:MTLSizeMake((N+rpt-1)/rpt,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    }
    barrier();
    return true;
}

// Single-dispatch matmul whose input is the residual h2 = clamp(res + rms(in)*w1),
// also written to h_out by threadgroup 0. gs==128, K<=4096 only.
bool cactus_metal_encode_quant_matmul_resid(void* out, const void* in, const void* w1, const void* res,
                                            void* h_out, float eps, const CactusQuantMatrix* W) {
    const uint32_t gs=W->group_size, K=W->K, N=W->N, ng=W->num_groups, pgb=(gs*4u+7u)/8u;
    static const bool disabled = (std::getenv("CACTUS_NO_RESID") != nullptr);
    if (disabled || !ctx().ok || !ctx().psoGfusedResid || gs != 128u || K > 4096u || !quant_fast_eligible(W)) return false;
    ResW& rw = resident(W);
    ensureEncoder();
    uint32_t ROWS=8;
    [g_enc setComputePipelineState:ctx().psoGfusedResid];
    setBufAt(in, (size_t)K*2, 0);                   [g_enc setBuffer:rw.packed offset:0 atIndex:1];
    [g_enc setBuffer:rw.codebook offset:0 atIndex:2];[g_enc setBuffer:rw.norms offset:0 atIndex:3];
    setBufAt(out, (size_t)N*2, 4);
    [g_enc setBytes:&gs length:4 atIndex:5]; [g_enc setBytes:&ng length:4 atIndex:6];
    [g_enc setBytes:&pgb length:4 atIndex:7]; [g_enc setBytes:&N length:4 atIndex:8];
    [g_enc setBuffer:rw.recip offset:0 atIndex:9]; [g_enc setBuffer:rw.lsign offset:0 atIndex:10];
    [g_enc setBuffer:rw.rsign offset:0 atIndex:11]; [g_enc setBuffer:rw.perm offset:0 atIndex:12];
    setBufAt(w1, (size_t)K*2, 13); setBufAt(res, (size_t)K*2, 14);
    setBufAt(h_out, (size_t)K*2, 15);
    [g_enc setBytes:&eps length:4 atIndex:16];
    [g_enc setThreadgroupMemoryLength:(size_t)K*4 + (ROWS*128+ROWS)*sizeof(float) atIndex:0];
    [g_enc dispatchThreadgroups:MTLSizeMake((N+ROWS*4-1)/(ROWS*4),1,1) threadsPerThreadgroup:MTLSizeMake(ROWS*32,1,1)];
    barrier();
    return true;
}

bool cactus_metal_encode_matmul_f16_gemv(void* out, const void* x, const void* w_rows,
                                          uint32_t K, uint32_t N) {
    if (!ctx().ok || !ctx().psoGF16 || (K % 4u) != 0) return false;
    ensureEncoder();
    [g_enc setComputePipelineState:ctx().psoGF16];
    setBufAt(x, (size_t)K*2, 0);
    setBufAt(w_rows, (size_t)N*K*2, 1);
    setBufAt(out, (size_t)N*2, 2);
    [g_enc setBytes:&K length:4 atIndex:3]; [g_enc setBytes:&N length:4 atIndex:4];
    [g_enc dispatchThreadgroups:MTLSizeMake((N+7)/8,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    barrier();
    return true;
}


bool cactus_metal_encode_rms2_add_clip(void* out, const void* a, const void* wa,
                                       const void* b, const void* wb, size_t dim, float eps) {
    if (!ctx().ok || !ctx().psoRms2Add) return false;
    ensureEncoder();
    uint32_t d=(uint32_t)dim; float e=eps;
    [g_enc setComputePipelineState:ctx().psoRms2Add];
    setBufAt(a, dim*2, 0); setBufAt(wa, dim*2, 1);
    setBufAt(b, dim*2, 2); setBufAt(wb, dim*2, 3);
    setBufAt(out, dim*2, 4);
    [g_enc setBytes:&d length:4 atIndex:5]; [g_enc setBytes:&e length:4 atIndex:6];
    [g_enc setThreadgroupMemoryLength:64*sizeof(float) atIndex:0];
    [g_enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    barrier();
    return true;
}

bool cactus_metal_encode_softmax_rows(void* out, const void* in, size_t rows, size_t dim) {
    if (!ctx().ok || !ctx().psoSoftmaxR || dim == 0) return false;
    ensureEncoder();
    uint32_t d=(uint32_t)dim;
    [g_enc setComputePipelineState:ctx().psoSoftmaxR];
    setBufAt(in, rows*dim*2, 0); setBufAt(out, rows*dim*2, 1);
    [g_enc setBytes:&d length:4 atIndex:2];
    [g_enc setThreadgroupMemoryLength:256*sizeof(float) atIndex:0];
    [g_enc dispatchThreadgroups:MTLSizeMake(rows,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    barrier();
    return true;
}

bool cactus_metal_encode_topk_row(void* out, const void* in, uint32_t features, uint32_t k) {
    if (!ctx().ok || !ctx().psoTopk || k == 0 || k > 32 || features == 0) return false;
    ensureEncoder();
    [g_enc setComputePipelineState:ctx().psoTopk];
    setBufAt(in, (size_t)features*2, 0); setBufAt(out, (size_t)k*2*sizeof(float), 1);
    [g_enc setBytes:&features length:4 atIndex:2]; [g_enc setBytes:&k length:4 atIndex:3];
    [g_enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    barrier();
    return true;
}

namespace {
// Per-MoE-node concatenation of all experts' 2-bit weights and transform
// params into stride-indexed buffers so kernels can select experts by the
// GPU-computed top-k indices.
struct MoESet {
    id<MTLBuffer> pk=nil, nm=nil, rc=nil, ls=nil, rs=nil, pm=nil, cb=nil;
    uint32_t K=0, N=0, ng=0;
};
struct MoECat { MoESet w1, w3, w2; bool ok=false; };
std::unordered_map<const void*, MoECat> g_moe_cat;

bool moe_build_set(MoESet& S, const CactusQuantMatrix* Ws, uint32_t count) {
    const uint32_t K=Ws[0].K, N=Ws[0].N, gs=Ws[0].group_size, ng=Ws[0].num_groups;
    if (gs != 128u || Ws[0].bits != 2 || (N % 4) != 0) return false;
    const uint32_t pgb = 32;
    size_t pk_stride = (size_t)N*ng*pgb, nm_stride=(size_t)N*ng;
    S.K=K; S.N=N; S.ng=ng;
    S.pk=[ctx().dev newBufferWithLength:pk_stride*count options:MTLResourceStorageModeShared];
    S.nm=[ctx().dev newBufferWithLength:nm_stride*count*2 options:MTLResourceStorageModeShared];
    S.rc=[ctx().dev newBufferWithLength:(size_t)K*count*2 options:MTLResourceStorageModeShared];
    S.ls=[ctx().dev newBufferWithLength:(size_t)128*count options:MTLResourceStorageModeShared];
    S.rs=[ctx().dev newBufferWithLength:(size_t)128*count options:MTLResourceStorageModeShared];
    S.pm=[ctx().dev newBufferWithLength:(size_t)128*count*4 options:MTLResourceStorageModeShared];
    S.cb=[ctx().dev newBufferWithLength:(size_t)8*count*2 options:MTLResourceStorageModeShared];
    if (!S.pk||!S.nm||!S.rc||!S.ls||!S.rs||!S.pm||!S.cb) return false;
    for (uint32_t e=0; e<count; ++e) {
        const CactusQuantMatrix* W=&Ws[e];
        if (W->bits!=2 || W->K!=K || W->N!=N || W->group_size!=gs) return false;
        uint8_t* pk_dst = (uint8_t*)S.pk.contents + pk_stride*e;
        __fp16*  nm_dst = (__fp16*)S.nm.contents + nm_stride*e;
        if (W->flags & CACTUS_QUANT_FLAG_INTERLEAVED_4ROW) {
            CactusQuantMatrix tmp=*W;
            deinterleave_4row_2bit(&tmp, pk_dst, nm_dst);
        } else {
            std::memcpy(pk_dst, W->packed_indices, pk_stride);
            std::memcpy(nm_dst, W->norms, nm_stride*2);
        }
        std::memcpy((__fp16*)S.rc.contents + (size_t)K*e, W->input_scale_recip, (size_t)K*2);
        std::memcpy((char*)S.ls.contents + 128*e, W->left_signs, 128);
        std::memcpy((char*)S.rs.contents + 128*e, W->right_signs, 128);
        std::memcpy((uint32_t*)S.pm.contents + 128*e, W->permutation, 128*4);
        std::memcpy((__fp16*)S.cb.contents + 8*e, W->codebook, 4*2);
    }
    return true;
}
}

// gated MoE for a single token: probs [E] fp16, topk [k idx][k vals] fp32.
// act: 0=SILU, 1=GELU(tanh).
bool cactus_metal_encode_moe_gated_cq2(void* out, const void* hidden, const void* probs, const void* topk,
                                       const CactusQuantMatrix* w1s, const CactusQuantMatrix* w3s,
                                       const CactusQuantMatrix* w2s, uint32_t num_experts, uint32_t top_k,
                                       uint32_t act, uint32_t normalize, float eps, float scaling) {
    if (!ctx().ok || !ctx().psoMoeT || top_k == 0 || top_k > 16) return false;
    auto it = g_moe_cat.find(w1s[0].packed_indices);
    if (it == g_moe_cat.end()) {
        MoECat cat;
        cat.ok = moe_build_set(cat.w1, w1s, num_experts)
              && moe_build_set(cat.w3, w3s, num_experts)
              && moe_build_set(cat.w2, w2s, num_experts);
        it = g_moe_cat.emplace(w1s[0].packed_indices, cat).first;
    }
    MoECat& C = it->second;
    if (!C.ok) return false;
    const uint32_t K1=C.w1.K, N1=C.w1.N, K2=C.w2.K, N2=C.w2.N;
    if (C.w3.K!=K1 || C.w3.N!=N1 || K2 < N1) return false;
    ensureEncoder();

    id<MTLBuffer> code1 = recycled((size_t)top_k*K1*2);
    id<MTLBuffer> code3 = recycled((size_t)top_k*K1*2);
    id<MTLBuffer> gu    = recycled((size_t)top_k*N1*2);
    id<MTLBuffer> code2 = recycled((size_t)top_k*K2*2);
    id<MTLBuffer> eout  = recycled((size_t)top_k*N2*2);

    auto transform=[&](const MoESet& S, const void* x, id<MTLBuffer> code, uint32_t k_valid, uint32_t x_stride, bool x_is_buf, id<MTLBuffer> xbuf){
        [g_enc setComputePipelineState:ctx().psoMoeT];
        if (x_is_buf) [g_enc setBuffer:xbuf offset:0 atIndex:0];
        else setBufAt(x, (size_t)k_valid*2, 0);
        setBufAt(topk, (size_t)top_k*2*sizeof(float), 1);
        [g_enc setBuffer:S.rc offset:0 atIndex:2];
        [g_enc setBuffer:S.ls offset:0 atIndex:3];
        [g_enc setBuffer:S.rs offset:0 atIndex:4];
        [g_enc setBuffer:S.pm offset:0 atIndex:5];
        [g_enc setBuffer:code offset:0 atIndex:6];
        uint32_t Kc=S.K, kv=k_valid, xs=x_stride;
        [g_enc setBytes:&Kc length:4 atIndex:7]; [g_enc setBytes:&kv length:4 atIndex:8];
        [g_enc setBytes:&xs length:4 atIndex:9];
        [g_enc setThreadgroupMemoryLength:128*sizeof(float) atIndex:0];
        [g_enc dispatchThreadgroups:MTLSizeMake(S.K/128u, top_k, 1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    };

    // up path: transforms for w1 and w3, shared hidden (x_stride 0)
    transform(C.w1, hidden, code1, K1, 0, false, nil);
    transform(C.w3, hidden, code3, K1, 0, false, nil);
    hazard();
    {
        [g_enc setComputePipelineState:ctx().psoMoeUp];
        [g_enc setBuffer:code1 offset:0 atIndex:0]; [g_enc setBuffer:code3 offset:0 atIndex:1];
        setBufAt(topk, (size_t)top_k*2*sizeof(float), 2);
        [g_enc setBuffer:C.w1.pk offset:0 atIndex:3]; [g_enc setBuffer:C.w1.nm offset:0 atIndex:4]; [g_enc setBuffer:C.w1.cb offset:0 atIndex:5];
        [g_enc setBuffer:C.w3.pk offset:0 atIndex:6]; [g_enc setBuffer:C.w3.nm offset:0 atIndex:7]; [g_enc setBuffer:C.w3.cb offset:0 atIndex:8];
        [g_enc setBuffer:gu offset:0 atIndex:9];
        uint32_t Kc=K1, Nc=N1, a=act;
        [g_enc setBytes:&Kc length:4 atIndex:10]; [g_enc setBytes:&Nc length:4 atIndex:11]; [g_enc setBytes:&a length:4 atIndex:12];
        [g_enc dispatchThreadgroups:MTLSizeMake((N1+31)/32, top_k, 1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    }
    hazard();
    // down path: transform of gate*up (per-slot rows, zero-padded to K2)
    transform(C.w2, nullptr, code2, N1, N1, true, gu);
    hazard();
    {
        [g_enc setComputePipelineState:ctx().psoMoeDown];
        [g_enc setBuffer:code2 offset:0 atIndex:0];
        setBufAt(topk, (size_t)top_k*2*sizeof(float), 1);
        [g_enc setBuffer:C.w2.pk offset:0 atIndex:2]; [g_enc setBuffer:C.w2.nm offset:0 atIndex:3]; [g_enc setBuffer:C.w2.cb offset:0 atIndex:4];
        [g_enc setBuffer:eout offset:0 atIndex:5];
        uint32_t Kc=K2, Nc=N2;
        [g_enc setBytes:&Kc length:4 atIndex:6]; [g_enc setBytes:&Nc length:4 atIndex:7];
        [g_enc dispatchThreadgroups:MTLSizeMake((N2+63)/64, top_k, 1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    }
    hazard();
    {
        [g_enc setComputePipelineState:ctx().psoMoeAcc];
        [g_enc setBuffer:eout offset:0 atIndex:0];
        setBufAt(probs, (size_t)num_experts*2, 1);
        setBufAt(topk, (size_t)top_k*2*sizeof(float), 2);
        setBufAt(out, (size_t)N2*2, 3);
        uint32_t Dc=N2, kc=top_k, nz=normalize;
        [g_enc setBytes:&Dc length:4 atIndex:4]; [g_enc setBytes:&kc length:4 atIndex:5];
        [g_enc setBytes:&nz length:4 atIndex:6]; [g_enc setBytes:&eps length:4 atIndex:7];
        [g_enc setBytes:&scaling length:4 atIndex:8];
        [g_enc dispatchThreads:MTLSizeMake(N2,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    }
    barrier();
    return true;
}

bool cactus_metal_prewarm_quant(const CactusQuantMatrix* W) {
    if (!ctx().ok) return false;
    if (!quant_fast_eligible(W)) return false;
    resident(W);
    return true;
}

bool cactus_metal_encode_quant_matmul_m(void* out, const void* lhs, const CactusQuantMatrix* W, uint32_t M) {
    if (M == 1) return cactus_metal_encode_quant_matmul(out, lhs, W);
    const uint32_t gs=W->group_size, K=W->K, N=W->N, ng=W->num_groups, pgb=(gs*4u+7u)/8u;
    bool fast = ctx().ok && W->bits == 4 && quant_fast_eligible(W);
    if (!fast) return false;
    ResW& rw = resident(W);
    size_t code_bytes = (size_t)M*ng*gs*sizeof(__fp16);
    if (!g_code_buf_m || (size_t)g_code_buf_m.length < code_bytes)
        g_code_buf_m = [ctx().dev newBufferWithLength:code_bytes options:MTLResourceStorageModeShared];
    ensureEncoder();

    [g_enc setComputePipelineState:ctx().psoTm];
    setBufAt(lhs, (size_t)M*K*2, 0);               [g_enc setBuffer:rw.recip offset:0 atIndex:1];
    [g_enc setBuffer:rw.lsign offset:0 atIndex:2]; [g_enc setBuffer:rw.rsign offset:0 atIndex:3];
    [g_enc setBuffer:rw.perm offset:0 atIndex:4];  [g_enc setBuffer:g_code_buf_m offset:0 atIndex:5];
    [g_enc setBytes:&gs length:4 atIndex:6]; [g_enc setBytes:&K length:4 atIndex:7];
    [g_enc setThreadgroupMemoryLength:gs*sizeof(float) atIndex:0];
    [g_enc dispatchThreadgroups:MTLSizeMake((size_t)ng*M,1,1) threadsPerThreadgroup:MTLSizeMake(gs>1024u?1024u:gs,1,1)];
    hazard();

    bool mma = (ctx().psoGmma != nil);
    [g_enc setComputePipelineState:(mma?ctx().psoGmma:ctx().psoGm)];
    [g_enc setBuffer:g_code_buf_m offset:0 atIndex:0]; [g_enc setBuffer:rw.packed offset:0 atIndex:1];
    [g_enc setBuffer:rw.codebook offset:0 atIndex:2]; [g_enc setBuffer:rw.norms offset:0 atIndex:3];
    setBufAt(out, (size_t)M*N*2, 4);
    [g_enc setBytes:&gs length:4 atIndex:5]; [g_enc setBytes:&ng length:4 atIndex:6];
    [g_enc setBytes:&pgb length:4 atIndex:7]; [g_enc setBytes:&N length:4 atIndex:8]; [g_enc setBytes:&M length:4 atIndex:9];
    if (mma) {
        [g_enc dispatchThreadgroups:MTLSizeMake((N+31)/32,(M+63)/64,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
    } else {
        uint32_t ROWS=8, MT=16;
        [g_enc dispatchThreadgroups:MTLSizeMake((N+ROWS-1)/ROWS,(M+MT-1)/MT,1) threadsPerThreadgroup:MTLSizeMake(ROWS*32,1,1)];
    }
    barrier();
    return true;
}

bool cactus_metal_encode_quant_matmul_ortho(void* out, const void* act, void* code,
                                            const CactusQuantMatrix* W) {
    if (!ctx().ok || !W->rotation || W->bits != 4) return false;
    const uint32_t K=W->K, N=W->N, ng=W->num_groups, gs=W->group_size, pgb=(gs*4u+7u)/8u;
    if (ng != 1 || gs != K || (N % 4) != 0) return false;
    ResW& rw = resident(W);
    if (!rw.rotation || !rw.packed) return false;
    ensureEncoder();

    // code = rotation^T . (act*recip), k-chunked for parallelism, then combined
    const uint32_t NC = 8;
    id<MTLBuffer> part = recycled((size_t)NC*K*sizeof(float));
    [g_enc setComputePipelineState:ctx().psoColdot];
    setBufAt(act, (size_t)K*2, 0); [g_enc setBuffer:rw.recip offset:0 atIndex:1];
    [g_enc setBuffer:rw.rotation offset:0 atIndex:2]; [g_enc setBuffer:part offset:0 atIndex:3];
    [g_enc setBytes:&K length:4 atIndex:4]; [g_enc setBytes:&NC length:4 atIndex:5];
    [g_enc dispatchThreadgroups:MTLSizeMake((K+255)/256,NC,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    hazard();
    [g_enc setComputePipelineState:ctx().psoColdotC];
    [g_enc setBuffer:part offset:0 atIndex:0]; setBufAt(code, (size_t)K*2, 1);
    [g_enc setBytes:&K length:4 atIndex:2]; [g_enc setBytes:&NC length:4 atIndex:3];
    [g_enc dispatchThreads:MTLSizeMake(K,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    hazard();

    uint32_t rpt_; [g_enc setComputePipelineState:gemv_pso_for(N, rpt_)];
    setBufAt(code, (size_t)K*2, 0); [g_enc setBuffer:rw.packed offset:0 atIndex:1];
    [g_enc setBuffer:rw.codebook offset:0 atIndex:2]; [g_enc setBuffer:rw.norms offset:0 atIndex:3];
    setBufAt(out, (size_t)N*2, 4);
    [g_enc setBytes:&gs length:4 atIndex:5]; [g_enc setBytes:&ng length:4 atIndex:6];
    [g_enc setBytes:&pgb length:4 atIndex:7]; [g_enc setBytes:&N length:4 atIndex:8];
    uint32_t ROWS=8;
    [g_enc dispatchThreadgroups:MTLSizeMake((N+rpt_-1)/rpt_,1,1) threadsPerThreadgroup:MTLSizeMake(ROWS*32,1,1)];
    barrier();
    return true;
}

bool cactus_metal_encode_embedding_ortho(void* out, uint32_t row, const CactusQuantMatrix* W, float scale) {
    if (!ctx().ok || !W->rotation || W->bits != 4) return false;
    const uint32_t K=W->K, ng=W->num_groups, gs=W->group_size;
    if (ng != 1 || gs != K) return false;
    ResW& rw = resident(W);
    if (!rw.packed || !rw.rotation || !rw.codebook || !rw.norms || !rw.recip) return false;
    ensureEncoder();
    // dequantize the row, then out[j] = dot(dq, rotation_row_j)*recip[j]*scale
    // (one simdgroup per output element instead of a single threadgroup)
    id<MTLBuffer> dq = recycled((size_t)K*2);
    [g_enc setComputePipelineState:ctx().psoEmbDq];
    [g_enc setBuffer:rw.packed offset:0 atIndex:0]; [g_enc setBuffer:rw.codebook offset:0 atIndex:1];
    [g_enc setBuffer:dq offset:0 atIndex:2];
    [g_enc setBytes:&K length:4 atIndex:3]; [g_enc setBytes:&row length:4 atIndex:4];
    [g_enc dispatchThreads:MTLSizeMake(K,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    hazard();
    float scale_total = scale * (float)((const __fp16*)W->norms)[row];
    [g_enc setComputePipelineState:ctx().psoRowdot];
    [g_enc setBuffer:dq offset:0 atIndex:0]; [g_enc setBuffer:rw.rotation offset:0 atIndex:1];
    [g_enc setBuffer:rw.recip offset:0 atIndex:2]; setBufAt(out, (size_t)K*2, 3);
    [g_enc setBytes:&K length:4 atIndex:4]; [g_enc setBytes:&scale_total length:4 atIndex:5];
    [g_enc dispatchThreadgroups:MTLSizeMake((K+7)/8,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    barrier();
    return true;
}

std::unordered_map<const void*, ResW> g_resident_emb;
ResW& resident_emb_meta(const CactusQuantMatrix* W) {
    std::lock_guard<std::mutex> lk(g_resident_mu);
    auto it = g_resident_emb.find(W->packed_indices);
    if (it != g_resident_emb.end()) return it->second;
    const uint32_t K=W->K, gs=W->group_size;
    ResW r;
    r.codebook = buf(W->codebook, 16*sizeof(__fp16));
    r.lsign    = buf(W->left_signs, gs);
    r.rsign    = buf(W->right_signs, gs);
    r.perm     = buf(W->permutation, (size_t)gs*sizeof(uint32_t));
    r.recip    = buf(W->input_scale_recip, (size_t)K*sizeof(__fp16));
    return g_resident_emb.emplace(W->packed_indices, r).first->second;
}

bool cactus_metal_encode_embedding_hadamard(void* out, uint32_t row, const CactusQuantMatrix* W) {
    if (!ctx().ok || W->bits != 4 || (W->flags & CACTUS_QUANT_FLAG_ORTHOGONAL)) return false;
    const uint32_t K=W->K, gs=W->group_size, ng=W->num_groups, pgb=(gs*4u+7u)/8u;
    if (gs > 256 || (gs & (gs-1)) != 0 || !W->packed_indices || !W->norms || !W->codebook
        || !W->left_signs || !W->right_signs || !W->permutation || !W->input_scale_recip) return false;
    ResW& rm = resident_emb_meta(W);
    if (!rm.codebook || !rm.recip || !rm.lsign || !rm.rsign || !rm.perm) return false;
    ensureEncoder();
    size_t row_bytes = (size_t)ng*pgb;
    id<MTLBuffer> prow = recycled(row_bytes);
    std::memcpy([prow contents], (const uint8_t*)W->packed_indices + (size_t)row*ng*pgb, row_bytes);
    id<MTLBuffer> nrow = recycled((size_t)ng*sizeof(__fp16));
    std::memcpy([nrow contents], (const __fp16*)W->norms + (size_t)row*ng, (size_t)ng*sizeof(__fp16));
    [g_enc setComputePipelineState:ctx().psoEmbH];
    [g_enc setBuffer:prow offset:0 atIndex:0]; [g_enc setBuffer:rm.codebook offset:0 atIndex:1];
    [g_enc setBuffer:nrow offset:0 atIndex:2]; [g_enc setBuffer:rm.recip offset:0 atIndex:3];
    [g_enc setBuffer:rm.lsign offset:0 atIndex:4]; [g_enc setBuffer:rm.rsign offset:0 atIndex:5];
    [g_enc setBuffer:rm.perm offset:0 atIndex:6]; setBufAt(out, (size_t)K*2, 7);
    [g_enc setBytes:&gs length:4 atIndex:8];
    [g_enc setThreadgroupMemoryLength:(size_t)gs*sizeof(float) atIndex:0];
    [g_enc dispatchThreadgroups:MTLSizeMake(ng,1,1) threadsPerThreadgroup:MTLSizeMake(gs,1,1)];
    barrier();
    return true;
}

bool cactus_metal_encode_embedding_ortho_m(void* out, const CactusQuantMatrix* W, const uint32_t* rows, uint32_t M) {
    if (!ctx().ok || !ctx().psoEmbOm || !W->rotation || W->bits != 4) return false;
    const uint32_t K=W->K, ng=W->num_groups, gs=W->group_size;
    if (ng != 1 || gs != K) return false;
    ResW& rw = resident(W);
    if (!rw.packed || !rw.rotation || !rw.codebook || !rw.norms || !rw.recip) return false;
    ensureEncoder();
    id<MTLBuffer> rb = recycled((size_t)M*sizeof(uint32_t));
    std::memcpy([rb contents], rows, (size_t)M*sizeof(uint32_t));
    [g_enc setComputePipelineState:ctx().psoEmbOm];
    [g_enc setBuffer:rw.packed offset:0 atIndex:0]; [g_enc setBuffer:rw.codebook offset:0 atIndex:1];
    [g_enc setBuffer:rw.norms offset:0 atIndex:2]; [g_enc setBuffer:rw.recip offset:0 atIndex:3];
    [g_enc setBuffer:rw.rotation offset:0 atIndex:4]; [g_enc setBuffer:rb offset:0 atIndex:5];
    setBufAt(out, (size_t)M*K*2, 6); [g_enc setBytes:&K length:4 atIndex:7]; [g_enc setBytes:&M length:4 atIndex:8];
    [g_enc dispatchThreadgroups:MTLSizeMake(((K+15)/16)*((M+15)/16),1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    barrier();
    return true;
}

bool cactus_metal_encode_embedding_hadamard_m(void* out, const CactusQuantMatrix* W, const uint32_t* rows, uint32_t M) {
    if (!ctx().ok || !ctx().psoEmbHm || W->bits != 4 || (W->flags & CACTUS_QUANT_FLAG_ORTHOGONAL)) return false;
    const uint32_t K=W->K, gs=W->group_size, ng=W->num_groups, pgb=(gs*4u+7u)/8u;
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
    [g_enc setBuffer:pk offset:0 atIndex:0]; [g_enc setBuffer:rm.codebook offset:0 atIndex:1];
    [g_enc setBuffer:nm offset:0 atIndex:2]; [g_enc setBuffer:rm.recip offset:0 atIndex:3];
    [g_enc setBuffer:rm.lsign offset:0 atIndex:4]; [g_enc setBuffer:rm.rsign offset:0 atIndex:5];
    [g_enc setBuffer:rm.perm offset:0 atIndex:6]; setBufAt(out, (size_t)M*K*2, 7);
    [g_enc setBytes:&gs length:4 atIndex:8]; [g_enc setBytes:&ng length:4 atIndex:9]; [g_enc setBytes:&K length:4 atIndex:10];
    [g_enc setThreadgroupMemoryLength:(size_t)gs*sizeof(float) atIndex:0];
    [g_enc dispatchThreadgroups:MTLSizeMake((size_t)ng*M,1,1) threadsPerThreadgroup:MTLSizeMake(gs,1,1)];
    barrier();
    return true;
}

bool cactus_metal_register_readonly(const void* p, size_t bytes) {
    if (!ctx().ok) return false;
    return registerReadonly(p, bytes) != nil;
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
    barrier();
    return true;
}

bool cactus_metal_encode_attention_fused_i8(
    void* out, const void* qraw, const void* kraw, const void* vraw,
    const void* qw, const void* kw, const void* vw,
    const void* cosb, const void* sinb,
    void* kc, void* vc, void* ks, void* vs,
    uint32_t num_q_heads, uint32_t head_dim, float scale, float eps,
    uint32_t kv_start, uint32_t kv_end, uint32_t use_local, uint32_t local_slot,
    size_t kc_bytes, size_t vc_bytes, size_t ks_bytes, size_t vs_bytes) {
    if (!ctx().ok || !ctx().psoAttnFused) return false;
    if (kv_end <= kv_start || (head_dim % 32u) != 0 || head_dim > 512u) return false;
    ensureEncoder();
    const uint32_t T = 128, nsg = T / 32u;
    const uint32_t R = kv_end - kv_start;
    static const uint32_t div = []{ const char* e = std::getenv("CACTUS_ATTN_DIV"); return e ? (uint32_t)atoi(e) : 32u; }();
    uint32_t nwg = (R <= div) ? 1u : (R + div - 1u) / div; if (nwg > 32u) nwg = 32u;
    id<MTLBuffer> partO = ctx().dummy, partML = ctx().dummy;
    if (nwg > 1u) {
        partO  = recycled((size_t)num_q_heads*nwg*head_dim*sizeof(float));
        partML = recycled((size_t)num_q_heads*nwg*2*sizeof(float));
    }
    auto setOpt = [&](const void* p, size_t bytes, int idx) {
        if (p && bytes) setBufAt(p, bytes, idx);
        else [g_enc setBuffer:ctx().dummy offset:0 atIndex:idx];
    };
    [g_enc setComputePipelineState:ctx().psoAttnFused];
    setBufAt(qraw, (size_t)num_q_heads*head_dim*2, 0);
    setOpt(kraw, (size_t)head_dim*2, 1); setOpt(vraw, (size_t)head_dim*2, 2);
    setBufAt(qw, (size_t)head_dim*2, 3);
    setOpt(kw, (size_t)head_dim*2, 4); setOpt(vw, (size_t)head_dim*2, 5);
    setBufAt(cosb, (size_t)head_dim*2, 6); setBufAt(sinb, (size_t)head_dim*2, 7);
    setBufAt(kc, kc_bytes, 8); setBufAt(vc, vc_bytes, 9);
    setBufAt(ks, ks_bytes, 10); setBufAt(vs, vs_bytes, 11);
    setBufAt(out, (size_t)num_q_heads*head_dim*2, 12);
    [g_enc setBytes:&num_q_heads length:4 atIndex:13];
    [g_enc setBytes:&head_dim length:4 atIndex:14];
    [g_enc setBytes:&scale length:4 atIndex:15];
    [g_enc setBytes:&kv_start length:4 atIndex:16]; [g_enc setBytes:&kv_end length:4 atIndex:17];
    [g_enc setBytes:&use_local length:4 atIndex:18]; [g_enc setBytes:&local_slot length:4 atIndex:19];
    [g_enc setBytes:&eps length:4 atIndex:20];
    [g_enc setBuffer:partO offset:0 atIndex:21]; [g_enc setBuffer:partML offset:0 atIndex:22];
    [g_enc setBytes:&nwg length:4 atIndex:23];
    [g_enc setThreadgroupMemoryLength:((size_t)nsg*head_dim + 2*nsg + 2*head_dim + 256)*sizeof(float) atIndex:0];
    [g_enc dispatchThreadgroups:MTLSizeMake(num_q_heads*nwg,1,1) threadsPerThreadgroup:MTLSizeMake(T,1,1)];
    if (nwg > 1u) {
        hazard();
        [g_enc setComputePipelineState:ctx().psoAttnC];
        [g_enc setBuffer:partO offset:0 atIndex:0];  [g_enc setBuffer:partML offset:0 atIndex:1];
        setBufAt(out, (size_t)num_q_heads*head_dim*2, 2);
        [g_enc setBytes:&head_dim length:4 atIndex:3]; [g_enc setBytes:&nwg length:4 atIndex:4];
        [g_enc dispatchThreadgroups:MTLSizeMake(num_q_heads,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    }
    barrier();
    return true;
}

bool cactus_metal_encode_attention_i8(
    void* out, const void* q, const void* knew, const void* vnew,
    const void* kc, const void* vc, const void* ks, const void* vs,
    uint32_t num_q_heads, uint32_t num_kv_heads, uint32_t head_dim, uint32_t v_hdim,
    uint32_t history_len, uint32_t total_keys, uint32_t kv_start, uint32_t kv_end,
    float scale, size_t kc_bytes, size_t vc_bytes, size_t ks_bytes, size_t vs_bytes) {
    if (!ctx().ok) return false;
    if (kv_end <= kv_start || (num_q_heads % num_kv_heads) != 0) return false;
    ensureEncoder();
    auto setCache = [&](const void* p, size_t bytes, int idx) {

        if (p && bytes) setBufAt(p, bytes, idx);
        else [g_enc setBuffer:ctx().dummy offset:0 atIndex:idx];
    };
    auto setInputs = [&]() {
        setBufAt(q, (size_t)num_q_heads*head_dim*2, 0);
        if (total_keys > history_len && knew && vnew) {
            setBufAt(knew, (size_t)num_kv_heads*head_dim*2, 1);
            setBufAt(vnew, (size_t)num_kv_heads*v_hdim*2, 2);
        } else {
            [g_enc setBuffer:ctx().dummy offset:0 atIndex:1]; [g_enc setBuffer:ctx().dummy offset:0 atIndex:2];
        }
        setCache(kc, kc_bytes, 3); setCache(vc, vc_bytes, 4);
        setCache(ks, ks_bytes, 5); setCache(vs, vs_bytes, 6);
    };

    const uint32_t T = 256, nsg = T / 32u;
    const uint32_t R = kv_end - kv_start;
    static const uint32_t div = []{ const char* e = std::getenv("CACTUS_ATTN_DIV"); return e ? (uint32_t)atoi(e) : 32u; }();
    uint32_t nwg = (R <= div) ? 1u : (R + div - 1u) / div; if (nwg > 32u) nwg = 32u;
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
        hazard();
        [g_enc setComputePipelineState:ctx().psoAttnC];
        [g_enc setBuffer:partO offset:0 atIndex:0];  [g_enc setBuffer:partML offset:0 atIndex:1];
        setBufAt(out, (size_t)num_q_heads*v_hdim*2, 2);
        [g_enc setBytes:&v_hdim length:4 atIndex:3]; [g_enc setBytes:&nwg length:4 atIndex:4];
        [g_enc dispatchThreadgroups:MTLSizeMake(num_q_heads,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    }
    barrier();
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
    bool mma2 = (ring == 0u && is_causal && total_keys >= 2048u && head_dim == 512u && v_hdim == 512u && num_q_heads == 8u && num_kv_heads == 1u);
    if (mma2) {
        ensureEncoder();
        [g_enc setComputePipelineState:ctx().psoAttnPreMma2];
        setBufAt(q, (size_t)M*num_q_heads*head_dim*2, 0);
        if (new_len > 0 && knew && vnew) {
            setBufAt(knew, (size_t)new_len*num_kv_heads*head_dim*2, 1);
            setBufAt(vnew, (size_t)new_len*num_kv_heads*v_hdim*2, 2);
        } else {
            [g_enc setBuffer:ctx().dummy offset:0 atIndex:1]; [g_enc setBuffer:ctx().dummy offset:0 atIndex:2];
        }
        auto setCacheM = [&](const void* p, size_t bytes, int idx) {
            if (p && bytes) setBufAt(p, bytes, idx); else [g_enc setBuffer:ctx().dummy offset:0 atIndex:idx];
        };
        setCacheM(kc, kc_bytes, 3); setCacheM(vc, vc_bytes, 4); setCacheM(ks, ks_bytes, 5); setCacheM(vs, vs_bytes, 6);
        setBufAt(out, (size_t)M*num_q_heads*v_hdim*2, 7);
        [g_enc setBytes:&num_q_heads length:4 atIndex:8]; [g_enc setBytes:&num_kv_heads length:4 atIndex:9];
        [g_enc setBytes:&head_dim length:4 atIndex:10];   [g_enc setBytes:&v_hdim length:4 atIndex:11];
        [g_enc setBytes:&history_len length:4 atIndex:12];[g_enc setBytes:&scale length:4 atIndex:13];
        [g_enc setBytes:&q_pos0 length:4 atIndex:14];     [g_enc setBytes:&new_len length:4 atIndex:15];
        [g_enc setBytes:&M length:4 atIndex:16];
        uint32_t mtiles = (M + 7u)/8u;
        [g_enc dispatchThreadgroups:MTLSizeMake(mtiles, 1, 1) threadsPerThreadgroup:MTLSizeMake(512,1,1)];
        barrier();
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
    auto setCache = [&](const void* p, size_t bytes, int idx) {
        if (p && bytes) setBufAt(p, bytes, idx); else [g_enc setBuffer:ctx().dummy offset:0 atIndex:idx];
    };
    setCache(kc, kc_bytes, 3); setCache(vc, vc_bytes, 4); setCache(ks, ks_bytes, 5); setCache(vs, vs_bytes, 6);
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
    barrier();
    return true;
}

bool cactus_metal_encode_strided_copy(void* out, const void* in, const uint32_t* oshape,
    const uint32_t* sstride, uint32_t ndim, uint32_t total, uint32_t base, size_t in_bytes, size_t out_bytes) {
    if (!ctx().ok || ndim == 0 || ndim > 8) return false;
    ensureEncoder();
    [g_enc setComputePipelineState:ctx().psoStrided];
    setBufAt(in, in_bytes, 0); setBufAt(out, out_bytes, 1);
    [g_enc setBytes:oshape length:ndim*4 atIndex:2]; [g_enc setBytes:sstride length:ndim*4 atIndex:3];
    [g_enc setBytes:&ndim length:4 atIndex:4]; [g_enc setBytes:&total length:4 atIndex:5]; [g_enc setBytes:&base length:4 atIndex:6];
    [g_enc dispatchThreads:MTLSizeMake(total,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    barrier();
    return true;
}
bool cactus_metal_encode_bcast_binary(int op, void* out, const void* a, const void* b,
    const uint32_t* oshape, const uint32_t* astride, const uint32_t* bstride, uint32_t ndim, uint32_t total,
    size_t a_bytes, size_t b_bytes, size_t out_bytes) {
    if (!ctx().ok || ndim == 0 || ndim > 8) return false;
    ensureEncoder();
    int o = op;
    [g_enc setComputePipelineState:ctx().psoBcast];
    setBufAt(a, a_bytes, 0); setBufAt(b, b_bytes, 1); setBufAt(out, out_bytes, 2);
    [g_enc setBytes:oshape length:ndim*4 atIndex:3]; [g_enc setBytes:astride length:ndim*4 atIndex:4]; [g_enc setBytes:bstride length:ndim*4 atIndex:5];
    [g_enc setBytes:&ndim length:4 atIndex:6]; [g_enc setBytes:&total length:4 atIndex:7]; [g_enc setBytes:&o length:4 atIndex:8];
    [g_enc dispatchThreads:MTLSizeMake(total,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    barrier();
    return true;
}
bool cactus_metal_encode_strided_scatter(void* out, const void* in, const uint32_t* ishape,
    const uint32_t* ostride, uint32_t ndim, uint32_t total, uint32_t base, size_t in_bytes, size_t out_bytes) {
    if (!ctx().ok || ndim == 0 || ndim > 8) return false;
    ensureEncoder();
    [g_enc setComputePipelineState:ctx().psoScatter];
    setBufAt(in, in_bytes, 0); setBufAt(out, out_bytes, 1);
    [g_enc setBytes:ishape length:ndim*4 atIndex:2]; [g_enc setBytes:ostride length:ndim*4 atIndex:3];
    [g_enc setBytes:&ndim length:4 atIndex:4]; [g_enc setBytes:&total length:4 atIndex:5]; [g_enc setBytes:&base length:4 atIndex:6];
    [g_enc dispatchThreads:MTLSizeMake(total,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    barrier();
    return true;
}
bool cactus_metal_encode_kv_append_i8(const void* src, void* int8base, void* scalebase,
    uint32_t kv_heads, uint32_t hdim, uint32_t current_len, uint32_t group_size,
    size_t src_bytes, size_t int8_bytes, size_t scale_bytes) {
    if (!ctx().ok) return false;
    uint32_t num_groups = (hdim + group_size - 1)/group_size, n = kv_heads*num_groups;
    ensureEncoder();
    [g_enc setComputePipelineState:ctx().psoKvAppend];
    setBufAt(src, src_bytes, 0); setBufAt(int8base, int8_bytes, 1); setBufAt(scalebase, scale_bytes, 2);
    [g_enc setBytes:&kv_heads length:4 atIndex:3]; [g_enc setBytes:&hdim length:4 atIndex:4];
    [g_enc setBytes:&current_len length:4 atIndex:5]; [g_enc setBytes:&group_size length:4 atIndex:6];
    [g_enc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(n<256?n:256,1,1)];
    barrier();
    return true;
}

bool cactus_metal_encode_kv_append_i8_m(const void* src, void* int8base, void* scalebase,
    uint32_t kv_heads, uint32_t hdim, uint32_t current_len, uint32_t group_size, uint32_t M,
    size_t src_bytes, size_t int8_bytes, size_t scale_bytes) {
    if (!ctx().ok) return false;
    uint32_t num_groups = (hdim + group_size - 1)/group_size, n = M*kv_heads*num_groups;
    ensureEncoder();
    [g_enc setComputePipelineState:ctx().psoKvAppendM];
    setBufAt(src, src_bytes, 0); setBufAt(int8base, int8_bytes, 1); setBufAt(scalebase, scale_bytes, 2);
    [g_enc setBytes:&kv_heads length:4 atIndex:3]; [g_enc setBytes:&hdim length:4 atIndex:4];
    [g_enc setBytes:&current_len length:4 atIndex:5]; [g_enc setBytes:&group_size length:4 atIndex:6];
    [g_enc setBytes:&M length:4 atIndex:7];
    [g_enc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(n<256?n:256,1,1)];
    barrier();
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
    barrier();
    return true;
}

bool cactus_metal_encode_kv_append_sliding_i8(const void* src, void* int8base, void* scalebase,
    uint32_t kv_heads, uint32_t hdim, uint32_t keep_sink, uint32_t remaining, uint32_t shift_src,
    uint32_t group_size, size_t src_bytes, size_t int8_bytes, size_t scale_bytes) {
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
        hazard();
    }
    uint32_t n2 = (remaining+1)*per;
    [g_enc setComputePipelineState:ctx().psoSlideR];
    setBufAt(src, src_bytes, 0); setBufAt(int8base, int8_bytes, 1); setBufAt(scalebase, scale_bytes, 2);
    [g_enc setBuffer:scrI8 offset:0 atIndex:3]; [g_enc setBuffer:scrSc offset:0 atIndex:4];
    [g_enc setBytes:&kv_heads length:4 atIndex:5]; [g_enc setBytes:&hdim length:4 atIndex:6];
    [g_enc setBytes:&group_size length:4 atIndex:7]; [g_enc setBytes:&keep_sink length:4 atIndex:8];
    [g_enc setBytes:&remaining length:4 atIndex:9];
    [g_enc dispatchThreads:MTLSizeMake(n2,1,1) threadsPerThreadgroup:MTLSizeMake(n2<256?n2:256,1,1)];
    barrier();
    return true;
}

bool cactus_metal_encode_kv_append_sliding_i8_m(const void* src, void* int8base, void* scalebase,
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
        hazard();
    }
    uint32_t n2 = (remaining+M)*per;
    [g_enc setComputePipelineState:ctx().psoSlideRM];
    setBufAt(src, src_bytes, 0); setBufAt(int8base, int8_bytes, 1); setBufAt(scalebase, scale_bytes, 2);
    [g_enc setBuffer:scrI8 offset:0 atIndex:3]; [g_enc setBuffer:scrSc offset:0 atIndex:4];
    [g_enc setBytes:&kv_heads length:4 atIndex:5]; [g_enc setBytes:&hdim length:4 atIndex:6];
    [g_enc setBytes:&group_size length:4 atIndex:7]; [g_enc setBytes:&keep_sink length:4 atIndex:8];
    [g_enc setBytes:&remaining length:4 atIndex:9]; [g_enc setBytes:&M length:4 atIndex:10];
    [g_enc dispatchThreads:MTLSizeMake(n2,1,1) threadsPerThreadgroup:MTLSizeMake(n2<256?n2:256,1,1)];
    barrier();
    return true;
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
        id<MTLBuffer> bx     = buf(A, (size_t)K * sizeof(__fp16));
        id<MTLBuffer> brecip = rw.recip;
        id<MTLBuffer> bls    = rw.lsign;
        id<MTLBuffer> brs    = rw.rsign;
        id<MTLBuffer> bperm  = rw.perm;
        id<MTLBuffer> bpk    = rw.packed;
        id<MTLBuffer> bcb    = rw.codebook;
        id<MTLBuffer> bnorm  = rw.norms;
        id<MTLBuffer> bcode  = [ctx().dev newBufferWithLength:(size_t)ng * gs * sizeof(__fp16)
                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> by     = [ctx().dev newBufferWithLength:(size_t)N * sizeof(__fp16)
                                                      options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmd = [ctx().queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];

        bool simdT = (gs==128u && ctx().psoTsimd);
        [e setComputePipelineState:(simdT?ctx().psoTsimd:ctx().psoT)];
        [e setBuffer:bx offset:0 atIndex:0];  [e setBuffer:brecip offset:0 atIndex:1];
        [e setBuffer:bls offset:0 atIndex:2]; [e setBuffer:brs offset:0 atIndex:3];
        [e setBuffer:bperm offset:0 atIndex:4]; [e setBuffer:bcode offset:0 atIndex:5];
        [e setBytes:&gs length:sizeof(uint32_t) atIndex:6];
        [e setThreadgroupMemoryLength:gs * sizeof(float) atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake(ng,1,1) threadsPerThreadgroup:MTLSizeMake(simdT?32u:(gs>1024u?1024u:gs),1,1)];

        [e setComputePipelineState:ctx().psoG];
        [e setBuffer:bcode offset:0 atIndex:0]; [e setBuffer:bpk offset:0 atIndex:1];
        [e setBuffer:bcb offset:0 atIndex:2];   [e setBuffer:bnorm offset:0 atIndex:3];
        [e setBuffer:by offset:0 atIndex:4];
        [e setBytes:&gs length:sizeof(uint32_t) atIndex:5];
        [e setBytes:&ng length:sizeof(uint32_t) atIndex:6];
        [e setBytes:&pgb length:sizeof(uint32_t) atIndex:7];
        [e setBytes:&N length:sizeof(uint32_t) atIndex:8];
        const uint32_t ROWS = 8;
        [e dispatchThreadgroups:MTLSizeMake((N+ROWS*4-1)/(ROWS*4),1,1)
            threadsPerThreadgroup:MTLSizeMake(ROWS*32,1,1)];

        [e endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        std::memcpy(C, [by contents], (size_t)N * sizeof(__fp16));
    }
}

#endif
