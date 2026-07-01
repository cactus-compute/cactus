

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
#include <chrono>

#include "cactus_kernels_msl.h"

namespace {

struct MetalCtx {
    id<MTLDevice> dev = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> psoT = nil;
    id<MTLComputePipelineState> psoTsimd = nil;
    id<MTLComputePipelineState> psoTbatch = nil;
    id<MTLComputePipelineState> psoG = nil, psoGmr = nil;
    id<MTLComputePipelineState> psoTm = nil, psoGm = nil;
    id<MTLComputePipelineState> psoGmma = nil, psoGdense = nil, psoDeq = nil;
    id<MTLComputePipelineState> psoRotate = nil;
    id<MTLComputePipelineState> psoEmbO = nil, psoEmbH = nil;
    id<MTLComputePipelineState> psoEmbOm = nil, psoEmbHm = nil;
    id<MTLComputePipelineState> psoCopy=nil, psoBinary=nil, psoScalar=nil, psoUnary=nil, psoRms=nil, psoSwiglu=nil, psoRmsAdd=nil, psoRmsAddScale=nil;
    id<MTLComputePipelineState> psoCF16F32=nil, psoCF32F16=nil, psoCI8F16=nil, psoCF16I8=nil;
    id<MTLComputePipelineState> psoAttn=nil, psoAttnC=nil, psoStrided=nil, psoScatter=nil, psoBcast=nil, psoKvAppend=nil;
    id<MTLComputePipelineState> psoAttnPre=nil, psoAttnPreMma2=nil, psoAttnPreHd256=nil;
    id<MTLComputePipelineState> psoKvAppendM=nil, psoKvAppendRingM=nil;
    id<MTLComputePipelineState> psoSlideS=nil, psoSlideR=nil, psoSlideRM=nil;
    id<MTLComputePipelineState> psoRope=nil, psoRopeM=nil, psoRmsRope=nil;
    id<MTLComputePipelineState> psoArgmax=nil;
    id<MTLComputePipelineState> psoGather=nil;
    id<MTLBuffer> dummy=nil;
    bool ok = false;

    MetalCtx() { @autoreleasepool {
        const bool prof = getenv("CACTUS_PROF_INIT")!=nullptr;
        auto t0 = std::chrono::high_resolution_clock::now();
        auto ms = [&](){ auto n=std::chrono::high_resolution_clock::now(); return std::chrono::duration<double,std::milli>(n-t0).count(); };
        dev = MTLCreateSystemDefaultDevice();
        if (!dev) return;
        queue = [dev newCommandQueue];
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kCactusMSL]
                                               options:nil error:&err];
        if (prof) fprintf(stderr,"[prof-init] MSL compile: %.1fms\n", ms());

        if (!lib) { if (err) fprintf(stderr,"[cactus-metal] MSL compile failed: %s\n",[[err localizedDescription] UTF8String]); return; }
        auto pso = [&](const char* name) -> id<MTLComputePipelineState> {
            NSError* e=nil;
            id<MTLFunction> f=[lib newFunctionWithName:[NSString stringWithUTF8String:name]];
            id<MTLComputePipelineState> p = f ? [dev newComputePipelineStateWithFunction:f error:&e] : nil;
            if (!p) fprintf(stderr,"[cactus-metal] pipeline '%s' failed: %s\n", name, e?[[e localizedDescription] UTF8String]:"function not found");
            return p;
        };
        psoT=pso("cq4_transform"); psoTsimd=pso("cq4_transform_simd"); psoG=pso("cq4_gemv"); psoTbatch=pso("cq4_transform_batch"); psoGmr=pso("cq4_gemv_mr");
        psoTm=pso("cq4_transform_m"); psoGm=pso("cq4_gemm"); psoGmma=pso("cq4_gemm_mma");
        psoGdense=pso("cq4_gemm_dense_f16"); psoDeq=pso("cq4_dequant_f16");
        psoRotate=pso("lmhead_rotate");
        psoEmbO=pso("emb_ortho"); psoEmbH=pso("emb_hadamard");
        psoEmbOm=pso("emb_ortho_m"); psoEmbHm=pso("emb_hadamard_m");
        psoGather=pso("gather_f16");
        psoCopy=pso("copy_bytes"); psoBinary=pso("binary_f16"); psoScalar=pso("scalar_f16");
        psoUnary=pso("unary_f16"); psoRms=pso("rms_norm_f16"); psoSwiglu=pso("swiglu_f16"); psoRmsAdd=pso("rms_norm_add_f16"); psoRmsAddScale=pso("rms_norm_add_scale_f16");
        psoCF16F32=pso("cast_f16_f32"); psoCF32F16=pso("cast_f32_f16");
        psoCI8F16=pso("cast_i8_f16"); psoCF16I8=pso("cast_f16_i8");
        psoAttn=pso("attn_decode_i8"); psoAttnC=pso("attn_decode_combine");
        psoStrided=pso("strided_copy_f16"); psoBcast=pso("bcast_binary_f16");
        psoAttnPre=pso("attn_prefill_i8"); psoAttnPreMma2=pso("attn_prefill_mma2"); psoAttnPreHd256=pso("attn_prefill_mma_hd256"); psoKvAppendM=pso("kv_append_i8_m");
        psoKvAppendRingM=pso("kv_append_ring_i8_m");
        psoSlideS=pso("kv_slide_save"); psoSlideR=pso("kv_slide_restore"); psoSlideRM=pso("kv_slide_restore_m");
        psoScatter=pso("strided_scatter_f16"); psoKvAppend=pso("kv_append_i8"); psoRope=pso("rope_f16"); psoRopeM=pso("rope_f16_m"); psoRmsRope=pso("rms_norm_rope_f16");
        psoArgmax=pso("argmax_logits");
        if (prof) fprintf(stderr,"[prof-init] pso creation: %.1fms total\n", ms());
        dummy=[dev newBufferWithLength:16 options:MTLResourceStorageModeShared];
        ok = psoT&&psoG&&psoTm&&psoGm&&psoRotate&&psoEmbO&&psoEmbH&&psoEmbOm&&psoEmbHm&&psoCopy&&psoBinary&&psoScalar&&psoUnary&&psoRms&&psoSwiglu&&psoRmsAdd&&psoCF16F32&&psoCF32F16&&psoCI8F16&&psoCF16I8
             &&psoAttn&&psoAttnC&&psoAttnPre&&psoAttnPreMma2&&psoKvAppendM&&psoKvAppendRingM&&psoSlideS&&psoSlideR&&psoSlideRM&&psoStrided&&psoBcast&&psoScatter&&psoKvAppend&&psoRope&&psoRopeM&&psoArgmax&&psoGather;
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
    id<MTLBuffer> wf16=nil;   // lazily-dequantized dense fp16 weight (prefill pre-dequant path)
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

ResW& resident(const CactusQuantMatrix* W) {
    std::lock_guard<std::mutex> lk(g_resident_mu);
    auto it = g_resident.find(W->packed_indices);
    if (it != g_resident.end()) return it->second;
    const uint32_t K=W->K, N=W->N, ng=W->num_groups, gs=W->group_size, pgb=(gs*4u+7u)/8u;
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
        deinterleave_4row(W, (uint8_t*)r.packed.contents, (__fp16*)r.norms.contents);
    } else {
        r.packed = buf(W->packed_indices, (size_t)N*ng*pgb);
        r.norms  = buf(W->norms, (size_t)N*ng*sizeof(__fp16));
    }
    return g_resident.emplace(W->packed_indices, r).first->second;
}

id<MTLCommandBuffer> g_cmd = nil;
id<MTLComputeCommandEncoder> g_enc = nil;
std::unordered_map<size_t, std::vector<id<MTLBuffer>>> g_free;
std::vector<id<MTLBuffer>> g_pending;
std::map<uintptr_t, id<MTLBuffer>> g_shared;
bool g_active = false;
id<MTLBuffer> g_code_buf = nil;
id<MTLBuffer> g_code_buf_m = nil;

inline size_t bucket(size_t b) { return (b + 4095) & ~size_t(4095); }

struct GemmProf {
    double gpu_ns=0; long cmdbufs=0;   // non-perturbing: total GPU-busy across all committed command buffers
    ~GemmProf(){ if(std::getenv("CACTUS_PROF_GPU") && cmdbufs>0)
        fprintf(stderr,"[PROF gpu-busy] cmdbufs=%ld total_gpu=%.2fms\n", cmdbufs, gpu_ns/1e6); }
};
static GemmProf g_gemmprof;
static const bool g_prof_gpu = (std::getenv("CACTUS_PROF_GPU") != nullptr);

// Per-op-type GPU-time profiler: flush+time each dispatch, attribute to the current op tag g_op.
// Perturbing (serializes), but gives an exact per-category GPU-time breakdown of a forward pass.
static const bool g_prof_ops = (std::getenv("CACTUS_PROF_OPS") != nullptr);
static const bool g_gemv_mr = (std::getenv("CACTUS_NO_GEMV_MR") == nullptr);   // multi-row M=1 gemv (default on; CACTUS_NO_GEMV_MR disables)
static const uint32_t g_mr_rows = 8u * 2u;   // ROWS * CQ4_NR — MUST match cactus_kernels.metal CQ4_NR
static const bool g_skip_mm=(std::getenv("CACTUS_SKIP_MM")!=nullptr), g_skip_attn=(std::getenv("CACTUS_SKIP_ATTN")!=nullptr), g_skip_norm=(std::getenv("CACTUS_SKIP_NORM")!=nullptr);
static const bool g_skip_transform=(std::getenv("CACTUS_SKIP_TRANSFORM")!=nullptr);
static const bool g_predequant=(std::getenv("CACTUS_NO_PREDEQUANT")==nullptr);   // prefill: dense fp16 gemm on pre-dequantized weights (default on; +~1.25GB RAM)
static const char* g_op = "other";
struct OpProf {
    std::map<std::string,std::pair<double,long>> t;
    ~OpProf(){ if(!g_prof_ops || t.empty()) return;
        double tot=0; for(auto&kv:t) tot+=kv.second.first;
        fprintf(stderr,"[PROF ops] total=%.1fms\n", tot/1e6);
        for(auto&kv:t) fprintf(stderr,"  %-12s %8.2fms  %5.1f%%  (n=%ld)\n",
            kv.first.c_str(), kv.second.first/1e6, 100.0*kv.second.first/tot, kv.second.second); }
};
static OpProf g_opprof;

static const bool g_concurrent = (std::getenv("CACTUS_GPU_CONCURRENT") != nullptr);

id<MTLComputeCommandEncoder> ensureEncoder() {
    if (!g_enc) {
        g_cmd = [ctx().queue commandBuffer];
        g_enc = g_concurrent ? [g_cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent]
                             : [g_cmd computeCommandEncoder];
    }
    return g_enc;
}

inline void barrier() {
    static const bool force = (std::getenv("CACTUS_FORCE_BARRIER") != nullptr);
    if (g_prof_ops && g_enc) {   // flush + time this single dispatch, attribute to g_op
        [g_enc endEncoding]; [g_cmd commit]; [g_cmd waitUntilCompleted];
        g_opprof.t[g_op].first += (g_cmd.GPUEndTime - g_cmd.GPUStartTime)*1e9;
        g_opprof.t[g_op].second += 1;
        g_enc = nil; g_cmd = nil;
        for (id<MTLBuffer> b : g_pending) g_free[(size_t)b.length].push_back(b);
        g_pending.clear();
        g_op = "other";
        return;
    }
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

std::pair<id<MTLBuffer>, size_t> bufForPtrOff(const void* p, size_t bytes) {
    uintptr_t a = reinterpret_cast<uintptr_t>(p);
    auto it = g_shared.upper_bound(a);
    if (it != g_shared.begin()) {
        --it;
        uintptr_t base = it->first;
        if (a < base + static_cast<uintptr_t>(it->second.length))
            return { it->second, static_cast<size_t>(a - base) };
    }

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
void cactus_metal_barrier() { if (g_concurrent && g_enc) [g_enc memoryBarrierWithScope:MTLBarrierScopeBuffers]; }

void cactus_metal_session_begin() {}
void cactus_metal_session_sync() {
    if (g_enc) {
        [g_enc endEncoding];
        if (g_prof_gpu) [g_cmd addCompletedHandler:^(id<MTLCommandBuffer> cb){
            g_gemmprof.gpu_ns += (cb.GPUEndTime-cb.GPUStartTime)*1e9; g_gemmprof.cmdbufs += 1; }];
        [g_cmd commit]; [g_cmd waitUntilCompleted];
        g_enc = nil; g_cmd = nil;
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
    g_op="swiglu";
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
    g_op="rope";
    uint32_t total=heads*head_dim;
    [g_enc setComputePipelineState:ctx().psoRope];
    setBufAt(x, (size_t)total*2, 0); setBufAt(out, (size_t)total*2, 1);
    setBufAt(cos, (size_t)head_dim*2, 2); setBufAt(sin, (size_t)head_dim*2, 3);
    [g_enc setBytes:&heads length:4 atIndex:4]; [g_enc setBytes:&head_dim length:4 atIndex:5];
    [g_enc dispatchThreads:MTLSizeMake(total,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    barrier();
    return true;
}
bool cactus_metal_encode_rope_m(void* out, const void* x, const void* cos_m, const void* sin_m,
                                uint32_t heads, uint32_t head_dim, uint32_t M) {
    if (!ctx().ok) return false;
    ensureEncoder();
    g_op="rope";
    uint32_t total=M*heads*head_dim;
    [g_enc setComputePipelineState:ctx().psoRopeM];
    setBufAt(x, (size_t)total*2, 0); setBufAt(out, (size_t)total*2, 1);
    setBufAt(cos_m, (size_t)M*head_dim*2, 2); setBufAt(sin_m, (size_t)M*head_dim*2, 3);
    [g_enc setBytes:&heads length:4 atIndex:4]; [g_enc setBytes:&head_dim length:4 atIndex:5]; [g_enc setBytes:&M length:4 atIndex:6];
    [g_enc dispatchThreads:MTLSizeMake(total,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    barrier();
    return true;
}
bool cactus_metal_encode_rms_norm_rope(void* out, const void* x, const void* weight,
                                       const void* cos, const void* sin,
                                       uint32_t heads, uint32_t head_dim, float eps) {
    if (!ctx().ok || !ctx().psoRmsRope) return false;
    ensureEncoder();
    g_op="rmsnorm";
    uint32_t total=heads*head_dim; uint32_t hd=head_dim; float e=eps;
    [g_enc setComputePipelineState:ctx().psoRmsRope];
    setBufAt(x, (size_t)total*2, 0); setBufAt(weight, (size_t)head_dim*2, 1);
    setBufAt(cos, (size_t)head_dim*2, 2); setBufAt(sin, (size_t)head_dim*2, 3);
    setBufAt(out, (size_t)total*2, 4);
    [g_enc setBytes:&hd length:4 atIndex:5]; [g_enc setBytes:&e length:4 atIndex:6];
    [g_enc setThreadgroupMemoryLength:256*sizeof(float) atIndex:0];
    if(!g_skip_norm) [g_enc dispatchThreadgroups:MTLSizeMake(heads,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    barrier();
    return true;
}
bool cactus_metal_encode_rms_norm(void* out, const void* in, const void* weight,
                                  size_t rows, size_t dim, float eps) {
    if (!ctx().ok) return false;
    ensureEncoder();
    g_op="rmsnorm";
    uint32_t d=(uint32_t)dim; float e=eps;
    [g_enc setComputePipelineState:ctx().psoRms];
    setBufAt(in, rows*dim*2, 0); setBufAt(weight, dim*2, 1); setBufAt(out, rows*dim*2, 2);
    [g_enc setBytes:&d length:4 atIndex:3]; [g_enc setBytes:&e length:4 atIndex:4];
    [g_enc setThreadgroupMemoryLength:256*sizeof(float) atIndex:0];
    if(!g_skip_norm) [g_enc dispatchThreadgroups:MTLSizeMake(rows,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    barrier();
    return true;
}
bool cactus_metal_encode_rms_norm_add(void* out, const void* in, const void* weight, const void* res,
                                      size_t rows, size_t dim, float eps) {
    if (!ctx().ok) return false;
    ensureEncoder();
    g_op="rmsnorm";
    uint32_t d=(uint32_t)dim; float e=eps;
    [g_enc setComputePipelineState:ctx().psoRmsAdd];
    setBufAt(in, rows*dim*2, 0); setBufAt(weight, dim*2, 1); setBufAt(res, rows*dim*2, 2); setBufAt(out, rows*dim*2, 3);
    [g_enc setBytes:&d length:4 atIndex:4]; [g_enc setBytes:&e length:4 atIndex:5];
    [g_enc setThreadgroupMemoryLength:256*sizeof(float) atIndex:0];
    if(!g_skip_norm) [g_enc dispatchThreadgroups:MTLSizeMake(rows,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    barrier();
    return true;
}
bool cactus_metal_encode_rms_norm_add_scale(void* out, const void* in, const void* weight, const void* res,
                                            size_t rows, size_t dim, float eps, float out_scale) {
    if (!ctx().ok || !ctx().psoRmsAddScale) return false;
    ensureEncoder();
    g_op="rmsnorm";
    uint32_t d=(uint32_t)dim; float e=eps, os=out_scale;
    [g_enc setComputePipelineState:ctx().psoRmsAddScale];
    setBufAt(in, rows*dim*2, 0); setBufAt(weight, dim*2, 1); setBufAt(res, rows*dim*2, 2); setBufAt(out, rows*dim*2, 3);
    [g_enc setBytes:&d length:4 atIndex:4]; [g_enc setBytes:&e length:4 atIndex:5]; [g_enc setBytes:&os length:4 atIndex:6];
    [g_enc setThreadgroupMemoryLength:256*sizeof(float) atIndex:0];
    if(!g_skip_norm) [g_enc dispatchThreadgroups:MTLSizeMake(rows,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    barrier();
    return true;
}
bool cactus_metal_encode_argmax(const void* logits, uint32_t vocab, void* out3) {
    if (!ctx().ok) return false;
    ensureEncoder();
    g_op="argmax";
    uint32_t V = vocab, T = 1024;
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
    return W->bits == 4 && W->group_size >= 128 && (W->group_size % 128) == 0 && (W->N % 4) == 0 &&
           !(W->flags & CACTUS_QUANT_FLAG_ORTHOGONAL) && W->input_scale_recip && W->left_signs &&
           W->right_signs && W->permutation && W->codebook && W->norms && W->packed_indices;
}

bool cactus_metal_encode_quant_matmul(void* out, const void* lhs, const CactusQuantMatrix* W) {
    const uint32_t gs=W->group_size, K=W->K, N=W->N, ng=W->num_groups, pgb=(gs*4u+7u)/8u;
    if (ctx().ok && (W->flags & CACTUS_QUANT_FLAG_ORTHOGONAL) && W->rotation && ng==1 && gs==K && (N%4)==0) {
        static void* ortho_code = nullptr; static uint32_t ortho_code_k = 0;
        if (ortho_code_k < K) { ortho_code = cactus_metal_alloc_shared((size_t)K*sizeof(__fp16)); ortho_code_k = K; }
        if (ortho_code) return cactus_metal_encode_quant_matmul_ortho(out, lhs, ortho_code, W);
    }
    bool fast = ctx().ok && quant_fast_eligible(W);
    if (!fast) return false;
    ResW& rw = resident(W);

    size_t code_bytes = (size_t)ng*gs*sizeof(__fp16);
    if (!g_code_buf || (size_t)g_code_buf.length < code_bytes)
        g_code_buf = [ctx().dev newBufferWithLength:code_bytes options:MTLResourceStorageModeShared];
    ensureEncoder();
    g_op="matmul";
    bool simdT = (gs==128u && ctx().psoTsimd);
    [g_enc setComputePipelineState:(simdT?ctx().psoTsimd:ctx().psoT)];
    setBufAt(lhs, (size_t)K*2, 0);                 [g_enc setBuffer:rw.recip offset:0 atIndex:1];
    [g_enc setBuffer:rw.lsign offset:0 atIndex:2]; [g_enc setBuffer:rw.rsign offset:0 atIndex:3];
    [g_enc setBuffer:rw.perm offset:0 atIndex:4];  [g_enc setBuffer:g_code_buf offset:0 atIndex:5];
    [g_enc setBytes:&gs length:4 atIndex:6]; [g_enc setThreadgroupMemoryLength:gs*sizeof(float) atIndex:0];
    [g_enc dispatchThreadgroups:MTLSizeMake(ng,1,1) threadsPerThreadgroup:MTLSizeMake(simdT?32u:(gs>1024u?1024u:gs),1,1)];
    barrier();
    bool umr = g_gemv_mr && ctx().psoGmr && (N % g_mr_rows == 0u);
    [g_enc setComputePipelineState:(umr?ctx().psoGmr:ctx().psoG)];
    [g_enc setBuffer:g_code_buf offset:0 atIndex:0]; [g_enc setBuffer:rw.packed offset:0 atIndex:1];
    [g_enc setBuffer:rw.codebook offset:0 atIndex:2]; [g_enc setBuffer:rw.norms offset:0 atIndex:3];
    setBufAt(out, (size_t)N*2, 4);
    [g_enc setBytes:&gs length:4 atIndex:5]; [g_enc setBytes:&ng length:4 atIndex:6];
    [g_enc setBytes:&pgb length:4 atIndex:7]; [g_enc setBytes:&N length:4 atIndex:8];
    uint32_t ROWS=8;
    g_op="matmul";
    uint32_t grid = umr ? (N+g_mr_rows-1u)/g_mr_rows : (N+ROWS-1u)/ROWS;
    if(!g_skip_mm) [g_enc dispatchThreadgroups:MTLSizeMake(grid,1,1) threadsPerThreadgroup:MTLSizeMake(ROWS*32,1,1)];
    barrier();
    return true;
}

// Batched CQ4 transform for up to 3 weights sharing the input `x` (recip). One dispatch instead of B.
bool cactus_metal_encode_transform_batch(const void* x, const CactusQuantMatrix* const* Ws, int B, void* const* codes) {
    if (!ctx().ok || B < 1 || B > 3 || !ctx().psoTbatch) return false;
    const uint32_t gs=Ws[0]->group_size, K=Ws[0]->K, ng=Ws[0]->num_groups;
    if (gs != 128u || !quant_fast_eligible(Ws[0])) return false;
    ResW& r0 = resident(Ws[0]);
    ResW* rw[3] = { &r0, &r0, &r0 };
    for (int b=1;b<B;b++) rw[b] = &resident(Ws[b]);
    ensureEncoder();
    g_op = "transform";
    [g_enc setComputePipelineState:ctx().psoTbatch];
    setBufAt(x, (size_t)K*2, 0);
    [g_enc setBuffer:r0.recip offset:0 atIndex:1];
    for (int b=0;b<3;b++) {
        ResW* R = rw[b<B?b:0];
        [g_enc setBuffer:R->lsign offset:0 atIndex:2+b];
        [g_enc setBuffer:R->rsign offset:0 atIndex:5+b];
        [g_enc setBuffer:R->perm  offset:0 atIndex:8+b];
        setBufAt(codes[b<B?b:0], (size_t)K*2, 11+b);
    }
    [g_enc setBytes:&ng length:4 atIndex:14];
    [g_enc setThreadgroupMemoryLength:gs*sizeof(float) atIndex:0];
    [g_enc dispatchThreadgroups:MTLSizeMake((size_t)ng*B,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    barrier();
    return true;
}

// CQ4 gemv with a pre-computed transformed activation `code` (no transform dispatch here).
bool cactus_metal_encode_gemv_precoded(void* out, const void* code, const CactusQuantMatrix* W) {
    if (!ctx().ok) return false;
    const uint32_t gs=W->group_size, K=W->K, N=W->N, ng=W->num_groups, pgb=(gs*4u+7u)/8u;
    if (!quant_fast_eligible(W)) return false;
    ResW& rw = resident(W);
    ensureEncoder();
    g_op = "matmul";
    bool umr = g_gemv_mr && ctx().psoGmr && (N % g_mr_rows == 0u);
    [g_enc setComputePipelineState:(umr?ctx().psoGmr:ctx().psoG)];
    setBufAt(code, (size_t)K*2, 0); [g_enc setBuffer:rw.packed offset:0 atIndex:1];
    [g_enc setBuffer:rw.codebook offset:0 atIndex:2]; [g_enc setBuffer:rw.norms offset:0 atIndex:3];
    setBufAt(out, (size_t)N*2, 4);
    [g_enc setBytes:&gs length:4 atIndex:5]; [g_enc setBytes:&ng length:4 atIndex:6];
    [g_enc setBytes:&pgb length:4 atIndex:7]; [g_enc setBytes:&N length:4 atIndex:8];
    uint32_t ROWS=8;
    uint32_t grid = umr ? (N+g_mr_rows-1u)/g_mr_rows : (N+ROWS-1u)/ROWS;
    if(!g_skip_mm) [g_enc dispatchThreadgroups:MTLSizeMake(grid,1,1) threadsPerThreadgroup:MTLSizeMake(ROWS*32,1,1)];
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
    bool fast = ctx().ok && quant_fast_eligible(W);
    if (!fast) return false;
    ResW& rw = resident(W);
    size_t code_bytes = (size_t)M*ng*gs*sizeof(__fp16);
    if (!g_code_buf_m || (size_t)g_code_buf_m.length < code_bytes)
        g_code_buf_m = [ctx().dev newBufferWithLength:code_bytes options:MTLResourceStorageModeShared];
    ensureEncoder();
    g_op="matmul";

    [g_enc setComputePipelineState:ctx().psoTm];
    setBufAt(lhs, (size_t)M*K*2, 0);               [g_enc setBuffer:rw.recip offset:0 atIndex:1];
    [g_enc setBuffer:rw.lsign offset:0 atIndex:2]; [g_enc setBuffer:rw.rsign offset:0 atIndex:3];
    [g_enc setBuffer:rw.perm offset:0 atIndex:4];  [g_enc setBuffer:g_code_buf_m offset:0 atIndex:5];
    [g_enc setBytes:&gs length:4 atIndex:6]; [g_enc setBytes:&K length:4 atIndex:7];
    [g_enc setThreadgroupMemoryLength:gs*sizeof(float) atIndex:0];
    g_op="transform";
    if (!g_skip_transform) [g_enc dispatchThreadgroups:MTLSizeMake((size_t)ng*M,1,1) threadsPerThreadgroup:MTLSizeMake(gs>1024u?1024u:gs,1,1)];
    barrier();

    // Pre-dequant dense path: dequantize the weight to fp16 ONCE (cached in rw.wf16), then run a plain
    // fp16 SG_MAT gemm — avoids the codebook LUT in the gemm fill (~33% faster gemm). The one-time dequant
    // is triggered inside the load-time prefill warmup, so real prefill only pays the fast dense gemm.
    if (g_predequant && ctx().psoGdense && ctx().psoDeq) {
        if (!rw.wf16) {
            rw.wf16 = [ctx().dev newBufferWithLength:(size_t)N*K*2 options:MTLResourceStorageModeShared];
            [g_enc setComputePipelineState:ctx().psoDeq];
            [g_enc setBuffer:rw.packed offset:0 atIndex:0]; [g_enc setBuffer:rw.codebook offset:0 atIndex:1];
            [g_enc setBuffer:rw.norms offset:0 atIndex:2]; [g_enc setBuffer:rw.wf16 offset:0 atIndex:3];
            [g_enc setBytes:&gs length:4 atIndex:4]; [g_enc setBytes:&ng length:4 atIndex:5];
            [g_enc setBytes:&pgb length:4 atIndex:6]; [g_enc setBytes:&N length:4 atIndex:7];
            [g_enc dispatchThreads:MTLSizeMake((size_t)N*ng,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            barrier();
        }
        [g_enc setComputePipelineState:ctx().psoGdense];
        [g_enc setBuffer:g_code_buf_m offset:0 atIndex:0]; [g_enc setBuffer:rw.wf16 offset:0 atIndex:1];
        setBufAt(out, (size_t)M*N*2, 2);
        [g_enc setBytes:&K length:4 atIndex:3]; [g_enc setBytes:&N length:4 atIndex:4]; [g_enc setBytes:&M length:4 atIndex:5];
        g_op="matmul";
        if (!g_skip_mm) [g_enc dispatchThreadgroups:MTLSizeMake((M+31)/32,(N+63)/64,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
        barrier();
        return true;
    }

    bool mma = (ctx().psoGmma != nil);
    [g_enc setComputePipelineState:(mma?ctx().psoGmma:ctx().psoGm)];
    [g_enc setBuffer:g_code_buf_m offset:0 atIndex:0]; [g_enc setBuffer:rw.packed offset:0 atIndex:1];
    [g_enc setBuffer:rw.codebook offset:0 atIndex:2]; [g_enc setBuffer:rw.norms offset:0 atIndex:3];
    setBufAt(out, (size_t)M*N*2, 4);
    [g_enc setBytes:&gs length:4 atIndex:5]; [g_enc setBytes:&ng length:4 atIndex:6];
    [g_enc setBytes:&pgb length:4 atIndex:7]; [g_enc setBytes:&N length:4 atIndex:8]; [g_enc setBytes:&M length:4 atIndex:9];
    g_op="matmul";
    if (!g_skip_mm) { if (mma) {
        [g_enc dispatchThreadgroups:MTLSizeMake((M+31)/32,(N+63)/64,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
    } else {
        uint32_t ROWS=8, MT=16;
        [g_enc dispatchThreadgroups:MTLSizeMake((N+ROWS-1)/ROWS,(M+MT-1)/MT,1) threadsPerThreadgroup:MTLSizeMake(ROWS*32,1,1)];
    } }
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

    [g_enc setComputePipelineState:ctx().psoRotate];
    setBufAt(act, (size_t)K*2, 0); [g_enc setBuffer:rw.recip offset:0 atIndex:1];
    [g_enc setBuffer:rw.rotation offset:0 atIndex:2]; setBufAt(code, (size_t)K*2, 3);
    [g_enc setBytes:&K length:4 atIndex:4];
    [g_enc dispatchThreads:MTLSizeMake(K,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    barrier();

    bool umr = g_gemv_mr && ctx().psoGmr && (N % g_mr_rows == 0u);
    [g_enc setComputePipelineState:(umr?ctx().psoGmr:ctx().psoG)];
    setBufAt(code, (size_t)K*2, 0); [g_enc setBuffer:rw.packed offset:0 atIndex:1];
    [g_enc setBuffer:rw.codebook offset:0 atIndex:2]; [g_enc setBuffer:rw.norms offset:0 atIndex:3];
    setBufAt(out, (size_t)N*2, 4);
    [g_enc setBytes:&gs length:4 atIndex:5]; [g_enc setBytes:&ng length:4 atIndex:6];
    [g_enc setBytes:&pgb length:4 atIndex:7]; [g_enc setBytes:&N length:4 atIndex:8];
    uint32_t ROWS=8;
    uint32_t grid = umr ? (N+g_mr_rows-1u)/g_mr_rows : (N+ROWS-1u)/ROWS;
    if(!g_skip_mm) [g_enc dispatchThreadgroups:MTLSizeMake(grid,1,1) threadsPerThreadgroup:MTLSizeMake(ROWS*32,1,1)];
    barrier();
    return true;
}

bool cactus_metal_encode_embedding_ortho(void* out, uint32_t row, const CactusQuantMatrix* W) {
    if (!ctx().ok || !W->rotation || W->bits != 4) return false;
    const uint32_t K=W->K, ng=W->num_groups, gs=W->group_size;
    if (ng != 1 || gs != K) return false;
    ResW& rw = resident(W);
    if (!rw.packed || !rw.rotation || !rw.codebook || !rw.norms || !rw.recip) return false;
    ensureEncoder();
    [g_enc setComputePipelineState:ctx().psoEmbO];
    [g_enc setBuffer:rw.packed offset:0 atIndex:0]; [g_enc setBuffer:rw.codebook offset:0 atIndex:1];
    [g_enc setBuffer:rw.norms offset:0 atIndex:2]; [g_enc setBuffer:rw.recip offset:0 atIndex:3];
    [g_enc setBuffer:rw.rotation offset:0 atIndex:4]; setBufAt(out, (size_t)K*2, 5);
    [g_enc setBytes:&K length:4 atIndex:6]; [g_enc setBytes:&row length:4 atIndex:7];
    uint32_t T = K>512u ? 512u : K;
    [g_enc setThreadgroupMemoryLength:(size_t)K*sizeof(float) atIndex:0];
    [g_enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(T,1,1)];
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

static std::unordered_map<const void*, id<MTLBuffer>> g_readonly;
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

bool cactus_metal_encode_attention_i8(
    void* out, const void* q, const void* knew, const void* vnew,
    const void* kc, const void* vc, const void* ks, const void* vs,
    uint32_t num_q_heads, uint32_t num_kv_heads, uint32_t head_dim, uint32_t v_hdim,
    uint32_t history_len, uint32_t total_keys, uint32_t kv_start, uint32_t kv_end,
    float scale, size_t kc_bytes, size_t vc_bytes, size_t ks_bytes, size_t vs_bytes) {
    if (!ctx().ok) return false;
    if (kv_end <= kv_start || (num_q_heads % num_kv_heads) != 0) return false;
    ensureEncoder();
    g_op="attention";
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
    uint32_t nwg = (R <= 256u) ? 1u : R / 96u; if (nwg > 32u) nwg = 32u;
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
    if(!g_skip_attn)[g_enc dispatchThreadgroups:MTLSizeMake(num_q_heads*nwg,1,1) threadsPerThreadgroup:MTLSizeMake(T,1,1)];
    barrier();
    if (nwg > 1u) {
        [g_enc setComputePipelineState:ctx().psoAttnC];
        [g_enc setBuffer:partO offset:0 atIndex:0];  [g_enc setBuffer:partML offset:0 atIndex:1];
        setBufAt(out, (size_t)num_q_heads*v_hdim*2, 2);
        [g_enc setBytes:&v_hdim length:4 atIndex:3]; [g_enc setBytes:&nwg length:4 atIndex:4];
        [g_enc dispatchThreadgroups:MTLSizeMake(num_q_heads,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        barrier();
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
    static const bool en_hd256 = (std::getenv("CACTUS_NO_HD256") == nullptr);  // tiled local (hd=256) prefill attention (default on)
    bool mma2 = (ring == 0u && is_causal && total_keys >= 2048u && head_dim == 512u && v_hdim == 512u && num_q_heads == 8u && num_kv_heads == 1u);
    // Route ALL global (hd=512, full-attention / window==0) layers through the tiled mma2 kernel, not just
    // the long-context ones — the tiled kernel loads each KV tile into shared once and reuses it across a
    // block of 8 queries, vs the scalar path re-reading the whole KV cache per query. Always correct for
    // window==0 (no sliding-window eviction). Biggest prefill attention win (pp2048 +22%).
    if (!mma2 && is_causal && head_dim == 512u && v_hdim == 512u && num_q_heads == 8u && num_kv_heads == 1u
        && window == 0u) mma2 = true;
    // hd256 now handles the sliding-window (sink + ring slot mapping), so it's valid for local layers too.
    bool hd256 = (en_hd256 && ctx().psoAttnPreHd256 != nil && is_causal && head_dim == 256u && v_hdim == 256u
                  && num_q_heads == 8u && num_kv_heads == 1u);
    if (mma2 || hd256) {
        ensureEncoder();
    g_op="attention";
        [g_enc setComputePipelineState:(hd256 ? ctx().psoAttnPreHd256 : ctx().psoAttnPreMma2)];
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
        if (hd256) { [g_enc setBytes:&sink length:4 atIndex:17]; [g_enc setBytes:&ring length:4 atIndex:18]; }
        uint32_t mtiles = (M + 7u)/8u;
        if(!g_skip_attn)[g_enc dispatchThreadgroups:MTLSizeMake(mtiles, 1, 1) threadsPerThreadgroup:MTLSizeMake(hd256?256u:512u,1,1)];
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
    if(!g_skip_attn)[g_enc dispatchThreadgroups:MTLSizeMake(M*num_q_heads,1,1) threadsPerThreadgroup:MTLSizeMake(T,1,1)];
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
    g_op="kv";
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
    g_op="kv";
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
    g_op="kv";
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
    g_op="kv";
    if (remaining > 0) {
        uint32_t n = remaining*per;
        [g_enc setComputePipelineState:ctx().psoSlideS];
        setBufAt(int8base, int8_bytes, 0); setBufAt(scalebase, scale_bytes, 1);
        [g_enc setBuffer:scrI8 offset:0 atIndex:2]; [g_enc setBuffer:scrSc offset:0 atIndex:3];
        [g_enc setBytes:&kv_heads length:4 atIndex:4]; [g_enc setBytes:&hdim length:4 atIndex:5];
        [g_enc setBytes:&group_size length:4 atIndex:6]; [g_enc setBytes:&shift_src length:4 atIndex:7];
        [g_enc setBytes:&remaining length:4 atIndex:8];
        [g_enc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(n<256?n:256,1,1)];
        barrier();
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
    g_op="kv";
    if (remaining > 0) {
        uint32_t n = remaining*per;
        [g_enc setComputePipelineState:ctx().psoSlideS];
        setBufAt(int8base, int8_bytes, 0); setBufAt(scalebase, scale_bytes, 1);
        [g_enc setBuffer:scrI8 offset:0 atIndex:2]; [g_enc setBuffer:scrSc offset:0 atIndex:3];
        [g_enc setBytes:&kv_heads length:4 atIndex:4]; [g_enc setBytes:&hdim length:4 atIndex:5];
        [g_enc setBytes:&group_size length:4 atIndex:6]; [g_enc setBytes:&shift_src length:4 atIndex:7];
        [g_enc setBytes:&remaining length:4 atIndex:8];
        [g_enc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(n<256?n:256,1,1)];
        barrier();
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
        [e dispatchThreadgroups:MTLSizeMake((N+ROWS-1)/ROWS,1,1)
            threadsPerThreadgroup:MTLSizeMake(ROWS*32,1,1)];

        [e endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        std::memcpy(C, [by contents], (size_t)N * sizeof(__fp16));
    }
}

#endif
