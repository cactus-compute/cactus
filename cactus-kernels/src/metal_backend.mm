

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
    id<MTLComputePipelineState> psoG = nil;
    id<MTLComputePipelineState> psoTm = nil, psoGm = nil;
    id<MTLComputePipelineState> psoGmma = nil;
    id<MTLComputePipelineState> psoRotate = nil;
    id<MTLComputePipelineState> psoEmbO = nil, psoEmbH = nil;
    id<MTLComputePipelineState> psoEmbOm = nil, psoEmbHm = nil;
    id<MTLComputePipelineState> psoCopy=nil, psoBinary=nil, psoScalar=nil, psoUnary=nil, psoRms=nil, psoSwiglu=nil, psoRmsAdd=nil;
    id<MTLComputePipelineState> psoCF16F32=nil, psoCF32F16=nil, psoCI8F16=nil, psoCF16I8=nil;
    id<MTLComputePipelineState> psoAttn=nil, psoStrided=nil, psoScatter=nil, psoBcast=nil, psoKvAppend=nil;
    id<MTLComputePipelineState> psoAttnP=nil, psoAttnC=nil;
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
        psoT=pso("cq4_transform"); psoTsimd=pso("cq4_transform_simd"); psoG=pso("cq4_gemv");
        psoTm=pso("cq4_transform_m"); psoGm=pso("cq4_gemm"); psoGmma=pso("cq4_gemm_mma");
        psoRotate=pso("lmhead_rotate");
        psoEmbO=pso("emb_ortho"); psoEmbH=pso("emb_hadamard");
        psoEmbOm=pso("emb_ortho_m"); psoEmbHm=pso("emb_hadamard_m");
        psoGather=pso("gather_f16");
        psoCopy=pso("copy_bytes"); psoBinary=pso("binary_f16"); psoScalar=pso("scalar_f16");
        psoUnary=pso("unary_f16"); psoRms=pso("rms_norm_f16"); psoSwiglu=pso("swiglu_f16"); psoRmsAdd=pso("rms_norm_add_f16");
        psoCF16F32=pso("cast_f16_f32"); psoCF32F16=pso("cast_f32_f16");
        psoCI8F16=pso("cast_i8_f16"); psoCF16I8=pso("cast_f16_i8");
        psoAttn=pso("attn_decode_i8"); psoStrided=pso("strided_copy_f16"); psoBcast=pso("bcast_binary_f16");
        psoAttnP=pso("attn_decode_i8_partial"); psoAttnC=pso("attn_decode_combine");
        psoAttnPre=pso("attn_prefill_i8"); psoAttnPreMma2=pso("attn_prefill_mma2"); psoKvAppendM=pso("kv_append_i8_m");
        psoKvAppendRingM=pso("kv_append_ring_i8_m");
        psoSlideS=pso("kv_slide_save"); psoSlideR=pso("kv_slide_restore"); psoSlideRM=pso("kv_slide_restore_m");
        psoScatter=pso("strided_scatter_f16"); psoKvAppend=pso("kv_append_i8"); psoRope=pso("rope_f16");
        psoArgmax=pso("argmax_logits");
        dummy=[dev newBufferWithLength:16 options:MTLResourceStorageModeShared];
        ok = psoT&&psoG&&psoTm&&psoGm&&psoRotate&&psoEmbO&&psoEmbH&&psoEmbOm&&psoEmbHm&&psoCopy&&psoBinary&&psoScalar&&psoUnary&&psoRms&&psoSwiglu&&psoRmsAdd&&psoCF16F32&&psoCF32F16&&psoCI8F16&&psoCF16I8
             &&psoAttn&&psoAttnP&&psoAttnC&&psoAttnPre&&psoAttnPreMma2&&psoKvAppendM&&psoKvAppendRingM&&psoSlideS&&psoSlideR&&psoStrided&&psoBcast&&psoScatter&&psoKvAppend&&psoRope&&psoArgmax&&psoGather;
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

    static const bool g_replay = (std::getenv("CACTUS_GPU_REPLAY") != nullptr);
    size_t nb = bytes ? bytes : 1;
    if (g_replay) {
        static std::unordered_map<const void*, id<MTLBuffer>> g_snap;
        auto sit = g_snap.find(p);
        id<MTLBuffer> b;
        if (sit != g_snap.end() && (size_t)sit->second.length >= nb) b = sit->second;
        else { b = [ctx().dev newBufferWithLength:nb options:MTLResourceStorageModeShared]; g_snap[p] = b; }
        std::memcpy([b contents], p, nb);
        return { b, 0 };
    }
    id<MTLBuffer> b = recycled(bytes);
    std::memcpy([b contents], p, nb);
    return { b, 0 };
}

inline void setBufAt(const void* p, size_t bytes, int idx) {
    auto pr = bufForPtrOff(p, bytes);
    [g_enc setBuffer:pr.first offset:pr.second atIndex:idx];
}

inline void setBufFresh(const void* p, size_t bytes, int idx) {
    id<MTLBuffer> b = recycled(bytes);
    std::memcpy([b contents], p, bytes ? bytes : 1);
    [g_enc setBuffer:b offset:0 atIndex:idx];
}

}

bool cactus_metal_available() { return ctx().ok; }

void cactus_metal_set_active(bool a) { g_active = a; }
bool cactus_metal_active_mode() { return ctx().ok && g_active; }

bool cactus_metal_concurrent() { return g_concurrent; }
void cactus_metal_barrier() { if (g_concurrent && g_enc) [g_enc memoryBarrierWithScope:MTLBarrierScopeBuffers]; }

void cactus_metal_session_begin() {  }
static int g_sync_count = 0;
static double g_gpu_ms = 0;
void cactus_metal_session_sync() {
    if (g_enc) {
        [g_enc endEncoding]; [g_cmd commit]; [g_cmd waitUntilCompleted];
        g_gpu_ms += ([g_cmd GPUEndTime] - [g_cmd GPUStartTime]) * 1000.0;
        g_enc = nil; g_cmd = nil; ++g_sync_count;
    }
    for (id<MTLBuffer> b : g_pending) g_free[(size_t)b.length].push_back(b);
    g_pending.clear();
}
void cactus_metal_session_end() {
    cactus_metal_session_sync();
    if (getenv("CACTUS_GPU_STATS")) {
        fprintf(stderr, "[gpu-stats] syncs=%d  on-GPU=%.3f ms\n", g_sync_count, g_gpu_ms);
        g_sync_count = 0; g_gpu_ms = 0;
    }
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
                                      size_t rows, size_t dim, float eps) {
    if (!ctx().ok) return false;
    ensureEncoder();
    uint32_t d=(uint32_t)dim; float e=eps;
    [g_enc setComputePipelineState:ctx().psoRmsAdd];
    setBufAt(in, rows*dim*2, 0); setBufAt(weight, dim*2, 1); setBufAt(res, rows*dim*2, 2); setBufAt(out, rows*dim*2, 3);
    [g_enc setBytes:&d length:4 atIndex:4]; [g_enc setBytes:&e length:4 atIndex:5];
    [g_enc setThreadgroupMemoryLength:256*sizeof(float) atIndex:0];
    [g_enc dispatchThreadgroups:MTLSizeMake(rows,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    barrier();
    return true;
}
bool cactus_metal_encode_argmax(const void* logits, uint32_t vocab, void* out3) {
    if (!ctx().ok) return false;
    ensureEncoder();
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
bool cactus_metal_encode_quant_matmul(void* out, const void* lhs, const CactusQuantMatrix* W) {
    const uint32_t gs=W->group_size, K=W->K, N=W->N, ng=W->num_groups, pgb=(gs*4u+7u)/8u;
    if (ctx().ok && (W->flags & CACTUS_QUANT_FLAG_ORTHOGONAL) && W->rotation && ng==1 && gs==K && (N%4)==0) {
        static void* ortho_code = nullptr; static uint32_t ortho_code_k = 0;
        if (ortho_code_k < K) { ortho_code = cactus_metal_alloc_shared((size_t)K*sizeof(__fp16)); ortho_code_k = K; }
        if (ortho_code) return cactus_metal_encode_quant_matmul_ortho(out, lhs, ortho_code, W);
    }
    bool fast = ctx().ok && W->bits==4 && gs>=128 && (gs%128)==0 && (N%4)==0 &&
        !(W->flags & CACTUS_QUANT_FLAG_ORTHOGONAL) && W->input_scale_recip && W->left_signs &&
        W->right_signs && W->permutation && W->codebook && W->norms && W->packed_indices;
    if (!fast) return false;
    ResW& rw = resident(W);

    size_t code_bytes = (size_t)ng*gs*sizeof(__fp16);
    if (!g_code_buf || (size_t)g_code_buf.length < code_bytes)
        g_code_buf = [ctx().dev newBufferWithLength:code_bytes options:MTLResourceStorageModeShared];
    ensureEncoder();
    bool simdT = (gs==128u && ctx().psoTsimd);
    [g_enc setComputePipelineState:(simdT?ctx().psoTsimd:ctx().psoT)];
    setBufAt(lhs, (size_t)K*2, 0);                 [g_enc setBuffer:rw.recip offset:0 atIndex:1];
    [g_enc setBuffer:rw.lsign offset:0 atIndex:2]; [g_enc setBuffer:rw.rsign offset:0 atIndex:3];
    [g_enc setBuffer:rw.perm offset:0 atIndex:4];  [g_enc setBuffer:g_code_buf offset:0 atIndex:5];
    [g_enc setBytes:&gs length:4 atIndex:6]; [g_enc setThreadgroupMemoryLength:gs*sizeof(float) atIndex:0];
    [g_enc dispatchThreadgroups:MTLSizeMake(ng,1,1) threadsPerThreadgroup:MTLSizeMake(simdT?32u:(gs>1024u?1024u:gs),1,1)];
    barrier();
    [g_enc setComputePipelineState:ctx().psoG];
    [g_enc setBuffer:g_code_buf offset:0 atIndex:0]; [g_enc setBuffer:rw.packed offset:0 atIndex:1];
    [g_enc setBuffer:rw.codebook offset:0 atIndex:2]; [g_enc setBuffer:rw.norms offset:0 atIndex:3];
    setBufAt(out, (size_t)N*2, 4);
    [g_enc setBytes:&gs length:4 atIndex:5]; [g_enc setBytes:&ng length:4 atIndex:6];
    [g_enc setBytes:&pgb length:4 atIndex:7]; [g_enc setBytes:&N length:4 atIndex:8];
    uint32_t ROWS=8;
    [g_enc dispatchThreadgroups:MTLSizeMake((N+ROWS-1)/ROWS,1,1) threadsPerThreadgroup:MTLSizeMake(ROWS*32,1,1)];
    barrier();
    return true;
}

bool cactus_metal_prewarm_quant(const CactusQuantMatrix* W) {
    if (!ctx().ok) return false;
    const uint32_t gs=W->group_size, N=W->N;
    bool fast = W->bits==4 && gs>=128 && (gs%128)==0 && (N%4)==0 &&
        !(W->flags & CACTUS_QUANT_FLAG_ORTHOGONAL) && W->input_scale_recip && W->left_signs &&
        W->right_signs && W->permutation && W->codebook && W->norms && W->packed_indices;
    if (!fast) return false;
    resident(W);
    return true;
}

bool cactus_metal_encode_quant_matmul_m(void* out, const void* lhs, const CactusQuantMatrix* W, uint32_t M) {
    if (M == 1) return cactus_metal_encode_quant_matmul(out, lhs, W);
    const uint32_t gs=W->group_size, K=W->K, N=W->N, ng=W->num_groups, pgb=(gs*4u+7u)/8u;
    bool fast = ctx().ok && W->bits==4 && gs>=128 && (gs%128)==0 && (N%4)==0 &&
        !(W->flags & CACTUS_QUANT_FLAG_ORTHOGONAL) && W->input_scale_recip && W->left_signs &&
        W->right_signs && W->permutation && W->codebook && W->norms && W->packed_indices;
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
    barrier();

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

    [g_enc setComputePipelineState:ctx().psoRotate];
    setBufAt(act, (size_t)K*2, 0); [g_enc setBuffer:rw.recip offset:0 atIndex:1];
    [g_enc setBuffer:rw.rotation offset:0 atIndex:2]; setBufAt(code, (size_t)K*2, 3);
    [g_enc setBytes:&K length:4 atIndex:4];
    [g_enc dispatchThreads:MTLSizeMake(K,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    barrier();

    [g_enc setComputePipelineState:ctx().psoG];
    setBufAt(code, (size_t)K*2, 0); [g_enc setBuffer:rw.packed offset:0 atIndex:1];
    [g_enc setBuffer:rw.codebook offset:0 atIndex:2]; [g_enc setBuffer:rw.norms offset:0 atIndex:3];
    setBufAt(out, (size_t)N*2, 4);
    [g_enc setBytes:&gs length:4 atIndex:5]; [g_enc setBytes:&ng length:4 atIndex:6];
    [g_enc setBytes:&pgb length:4 atIndex:7]; [g_enc setBytes:&N length:4 atIndex:8];
    uint32_t ROWS=8;
    [g_enc dispatchThreadgroups:MTLSizeMake((N+ROWS-1)/ROWS,1,1) threadsPerThreadgroup:MTLSizeMake(ROWS*32,1,1)];
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

    static const bool g_nosplit = (std::getenv("CACTUS_ATTN_NOSPLIT") != nullptr);
    static const uint32_t CHUNK_BASE = std::getenv("CACTUS_ATTN_CHUNK") ? (uint32_t)atoi(std::getenv("CACTUS_ATTN_CHUNK")) : 96;
    static const uint32_t C_MAX = std::getenv("CACTUS_ATTN_CMAX") ? (uint32_t)atoi(std::getenv("CACTUS_ATTN_CMAX")) : 64;
    const uint32_t R = kv_end - kv_start;
    uint32_t chunk = CHUNK_BASE, C = (R + chunk - 1) / chunk;
    if (C > C_MAX) { C = C_MAX; chunk = (R + C - 1) / C; }
    if (g_nosplit) { C = 1; chunk = R; }

    if (C <= 1) {
        const uint32_t T = 256;
        [g_enc setComputePipelineState:ctx().psoAttn];
        setInputs();
        setBufAt(out, (size_t)num_q_heads*v_hdim*2, 7);
        [g_enc setBytes:&num_q_heads length:4 atIndex:8]; [g_enc setBytes:&num_kv_heads length:4 atIndex:9];
        [g_enc setBytes:&head_dim length:4 atIndex:10];   [g_enc setBytes:&v_hdim length:4 atIndex:11];
        [g_enc setBytes:&history_len length:4 atIndex:12];[g_enc setBytes:&scale length:4 atIndex:13];
        [g_enc setBytes:&kv_start length:4 atIndex:14];   [g_enc setBytes:&kv_end length:4 atIndex:15];
        [g_enc setThreadgroupMemoryLength:((size_t)kv_end + T)*sizeof(float) atIndex:0];
        [g_enc dispatchThreadgroups:MTLSizeMake(num_q_heads,1,1) threadsPerThreadgroup:MTLSizeMake(T,1,1)];
        barrier();
        return true;
    }

    uint32_t TP = chunk < 128 ? chunk : 128;
    TP = (TP + 31u) & ~31u;
    if (TP < 32) TP = 32; if (TP > 128) TP = 128;
    id<MTLBuffer> partO = recycled((size_t)num_q_heads*C*v_hdim*sizeof(float));
    id<MTLBuffer> partM = recycled((size_t)num_q_heads*C*sizeof(float));
    id<MTLBuffer> partL = recycled((size_t)num_q_heads*C*sizeof(float));
    [g_enc setComputePipelineState:ctx().psoAttnP];
    setInputs();
    [g_enc setBuffer:partO offset:0 atIndex:7];
    [g_enc setBuffer:partM offset:0 atIndex:8];
    [g_enc setBuffer:partL offset:0 atIndex:9];
    [g_enc setBytes:&num_q_heads length:4 atIndex:10]; [g_enc setBytes:&num_kv_heads length:4 atIndex:11];
    [g_enc setBytes:&head_dim length:4 atIndex:12];    [g_enc setBytes:&v_hdim length:4 atIndex:13];
    [g_enc setBytes:&history_len length:4 atIndex:14]; [g_enc setBytes:&scale length:4 atIndex:15];
    [g_enc setBytes:&kv_start length:4 atIndex:16];    [g_enc setBytes:&kv_end length:4 atIndex:17];
    [g_enc setBytes:&chunk length:4 atIndex:18];       [g_enc setBytes:&C length:4 atIndex:19];
    [g_enc setThreadgroupMemoryLength:((size_t)chunk + TP)*sizeof(float) atIndex:0];
    [g_enc dispatchThreadgroups:MTLSizeMake(num_q_heads*C,1,1) threadsPerThreadgroup:MTLSizeMake(TP,1,1)];
    barrier();

    const uint32_t TC = 256;
    [g_enc setComputePipelineState:ctx().psoAttnC];
    [g_enc setBuffer:partO offset:0 atIndex:0];
    [g_enc setBuffer:partM offset:0 atIndex:1];
    [g_enc setBuffer:partL offset:0 atIndex:2];
    setBufAt(out, (size_t)num_q_heads*v_hdim*2, 3);
    [g_enc setBytes:&num_q_heads length:4 atIndex:4]; [g_enc setBytes:&v_hdim length:4 atIndex:5];
    [g_enc setBytes:&C length:4 atIndex:6];
    [g_enc setThreadgroupMemoryLength:(size_t)C*sizeof(float) atIndex:0];
    [g_enc dispatchThreadgroups:MTLSizeMake(num_q_heads,1,1) threadsPerThreadgroup:MTLSizeMake(TC,1,1)];
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
    const bool fast =
        ctx().ok && M == 1 && W->bits == 4 && gs >= 128 && (gs % 128) == 0 &&
        !(W->flags & CACTUS_QUANT_FLAG_ORTHOGONAL) && (W->N % 4) == 0 &&
        W->input_scale_recip != nullptr && W->left_signs != nullptr &&
        W->right_signs != nullptr && W->permutation != nullptr &&
        W->codebook != nullptr && W->norms != nullptr && W->packed_indices != nullptr;

    if (getenv("CACTUS_GPU_DEBUG") && M == 1) {
        static std::atomic<int> dbg{0};
        if (dbg++ < 14)
            fprintf(stderr, "[metal] M=%u bits=%u gs=%u flags=0x%x recip=%d ls=%d rs=%d perm=%d -> %s\n",
                    M, W->bits, gs, W->flags, W->input_scale_recip != nullptr,
                    W->left_signs != nullptr, W->right_signs != nullptr,
                    W->permutation != nullptr, fast ? "GPU" : "CPU-fallback");
    }

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
