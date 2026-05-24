#ifdef __APPLE__

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <atomic>
#include "kernel.h"
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

static NSString* const kCactusMSL = @R"(
#include <metal_stdlib>
using namespace metal;

kernel void cactus_gemv_int4(
    device const half* A [[buffer(0)]],
    device const uchar* B_packed [[buffer(1)]],
    device const half* B_scales [[buffer(2)]],
    device half* C [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& group_size [[buffer(6)]],
    uint tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const uint n_block = tgpig;
    const uint n_base = n_block << 2;
    if (n_base >= N) return;
    const uint num_groups = K / group_size;

    float partials[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint k = tiisg; k < K; k += 32) {
        float a_val = float(A[k]);
        uint g = k / group_size;
        uint k_local = k - g * group_size;
        uint k_super = k_local >> 3;
        uint k_in_slab = k_local & 7;
        bool high_nibble = k_in_slab >= 4;
        uint byte_offset_base = (n_block * K + g * group_size) * 2 + k_super * 16 + (k_in_slab & 3);

        #pragma unroll
        for (uint c = 0; c < 4; ++c) {
            uchar b = B_packed[byte_offset_base + c * 4];
            int nibble = high_nibble ? int(b >> 4) : int(b & 0xF);
            if (nibble >= 8) nibble -= 16;
            float scale = float(B_scales[(n_block * num_groups + g) * 4 + c]);
            partials[c] += a_val * float(nibble) * scale;
        }
    }

    #pragma unroll
    for (uint c = 0; c < 4; ++c) {
        float total = simd_sum(partials[c]);
        if (tiisg == 0 && n_base + c < N) C[n_base + c] = (half)total;
    }
}

kernel void cactus_dequant_int4(
    device const uchar* B_packed [[buffer(0)]],
    device const half* B_scales [[buffer(1)]],
    device half* B_out [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    constant uint& group_size [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint k_slab = gid.x;
    const uint n      = gid.y;
    const uint k_base = k_slab << 3;

    const uint num_groups   = K / group_size;
    const uint nb           = n >> 2;
    const uint c            = n & 3u;
    const uint g            = k_base / group_size;
    const uint kl           = k_base - g * group_size;
    const uint slab_in_grp  = kl >> 3;
    const uint byte_offset  = (nb * K + g * group_size) * 2u + slab_in_grp * 16u + c * 4u;

    uchar4 packed = *(device const uchar4*)(B_packed + byte_offset);
    half scl = B_scales[(nb * num_groups + g) * 4u + c];

    half4 lo4, hi4;
    #pragma unroll
    for (uint i = 0; i < 4; ++i) {
        int lo = int(packed[i] & 0xFu);
        int hi = int(packed[i] >> 4);
        if (lo >= 8) lo -= 16;
        if (hi >= 8) hi -= 16;
        lo4[i] = half(lo) * scl;
        hi4[i] = half(hi) * scl;
    }

    device half4* dst = (device half4*)(B_out + n * K + k_base);
    dst[0] = lo4;
    dst[1] = hi4;
}


constant uint A2_Q_BLOCK = 8;
constant uint A2_KV_BLOCK = 16;
constant uint A2_NSG = 2;

kernel void cactus_flash_attn_f16_v2(
    device const half* q_in [[buffer(0)]],
    device const half* k_in [[buffer(1)]],
    device const half* v_in [[buffer(2)]],
    device half* o_out [[buffer(3)]],
    constant uint& seq_len [[buffer(4)]],
    constant uint& kv_seq_len [[buffer(5)]],
    constant uint& nqh [[buffer(6)]],
    constant uint& nkh [[buffer(7)]],
    constant uint& head_dim [[buffer(8)]],
    constant float& scale [[buffer(9)]],
    constant uint& position_offset [[buffer(10)]],
    threadgroup char* shmem [[threadgroup(0)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const uint q_block_idx = tgpig.x;
    const uint q_head = tgpig.y;
    const uint q_start = q_block_idx * A2_Q_BLOCK;
    if (q_start >= seq_len) return;

    const uint kv_head = q_head * nkh / nqh;
    const uint q_stride = nqh * head_dim;
    const uint kv_stride = nkh * head_dim;

    const ushort tid_in_tg = sgitg * 32 + tiisg;
    const ushort total_threads = A2_NSG * 32;

    threadgroup half* sQ = (threadgroup half*)shmem;
    threadgroup half* sK = sQ + A2_Q_BLOCK * head_dim;
    threadgroup half* sV = sK + A2_KV_BLOCK * head_dim;
    threadgroup half* sP = sV + A2_KV_BLOCK * head_dim;
    threadgroup float* sO = (threadgroup float*)(sP + A2_Q_BLOCK * A2_KV_BLOCK);
    threadgroup float* sM = sO + A2_Q_BLOCK * head_dim;
    threadgroup float* sL = sM + A2_Q_BLOCK;
    threadgroup float* sScratch = sL + A2_Q_BLOCK;

    if (tid_in_tg < A2_Q_BLOCK) {
        sM[tid_in_tg] = -INFINITY;
        sL[tid_in_tg] = 0.0f;
    }
    for (uint i = tid_in_tg; i < A2_Q_BLOCK * head_dim; i += total_threads) {
        sO[i] = 0.0f;
    }

    for (uint j = 0; j < A2_Q_BLOCK; ++j) {
        const uint q_pos = q_start + j;
        const bool valid = q_pos < seq_len;
        for (uint d = tid_in_tg; d < head_dim; d += total_threads) {
            sQ[j * head_dim + d] = valid ? q_in[q_pos * q_stride + q_head * head_dim + d] : (half)0;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint d_blocks = head_dim / 8;
    const uint kv_end = min(kv_seq_len, position_offset + q_start + A2_Q_BLOCK);

    for (uint kv0 = 0; kv0 < kv_end; kv0 += A2_KV_BLOCK) {
        for (uint i = 0; i < A2_KV_BLOCK; ++i) {
            const uint kv_pos = kv0 + i;
            const bool valid = kv_pos < kv_seq_len;
            for (uint d = tid_in_tg; d < head_dim; d += total_threads) {
                sK[i * head_dim + d] = valid ? k_in[kv_pos * kv_stride + kv_head * head_dim + d] : (half)0;
                sV[i * head_dim + d] = valid ? v_in[kv_pos * kv_stride + kv_head * head_dim + d] : (half)0;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const ushort kv_col_off = sgitg * 8;
        simdgroup_float8x8 qk_acc = make_filled_simdgroup_matrix<float, 8>(0.0f);
        for (uint b = 0; b < d_blocks; ++b) {
            simdgroup_half8x8 q_tile, k_tile;
            simdgroup_load(q_tile, sQ + b * 8, head_dim);
            simdgroup_load(k_tile, sK + kv_col_off * head_dim + b * 8, head_dim, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(qk_acc, q_tile, k_tile, qk_acc);
        }
        simdgroup_store(qk_acc, sScratch + kv_col_off, A2_KV_BLOCK);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg < 4) {
            const ushort row = sgitg * 4 + tiisg;
            const uint q_pos = q_start + row;
            float scores[A2_KV_BLOCK];
            #pragma unroll
            for (ushort c = 0; c < A2_KV_BLOCK; ++c) {
                const uint kv_pos = kv0 + c;
                float s = sScratch[row * A2_KV_BLOCK + c] * scale;
                if (kv_pos > position_offset + q_pos || q_pos >= seq_len || kv_pos >= kv_seq_len) {
                    s = -INFINITY;
                }
                scores[c] = s;
            }
            float row_max = scores[0];
            for (ushort c = 1; c < A2_KV_BLOCK; ++c) row_max = max(row_max, scores[c]);

            const float old_m = sM[row];
            const float new_m = max(old_m, row_max);
            const float alpha = (old_m == -INFINITY) ? 0.0f : exp(old_m - new_m);

            float new_l = alpha * sL[row];
            for (ushort c = 0; c < A2_KV_BLOCK; ++c) {
                const float p = (new_m == -INFINITY) ? 0.0f : exp(scores[c] - new_m);
                sP[row * A2_KV_BLOCK + c] = (half)p;
                new_l += p;
            }
            sM[row] = new_m;
            sL[row] = new_l;
            for (uint d = 0; d < head_dim; ++d) {
                sO[row * head_dim + d] *= alpha;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint d_off = sgitg * (head_dim / 2);
        const uint sg_d_blocks = head_dim / 16;
        for (uint b = 0; b < sg_d_blocks; ++b) {
            simdgroup_float8x8 o_tile;
            simdgroup_load(o_tile, sO + d_off + b * 8, head_dim);
            simdgroup_half8x8 p0, p1, v0, v1;
            simdgroup_load(p0, sP, A2_KV_BLOCK);
            simdgroup_load(p1, sP + 8, A2_KV_BLOCK);
            simdgroup_load(v0, sV + d_off + b * 8, head_dim);
            simdgroup_load(v1, sV + 8 * head_dim + d_off + b * 8, head_dim);
            simdgroup_multiply_accumulate(o_tile, p0, v0, o_tile);
            simdgroup_multiply_accumulate(o_tile, p1, v1, o_tile);
            simdgroup_store(o_tile, sO + d_off + b * 8, head_dim);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint j = 0; j < A2_Q_BLOCK; ++j) {
        const uint q_pos = q_start + j;
        if (q_pos >= seq_len) continue;
        const float l = sL[j];
        const float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
        for (uint d = tid_in_tg; d < head_dim; d += total_threads) {
            o_out[q_pos * q_stride + q_head * head_dim + d] = (half)(sO[j * head_dim + d] * inv_l);
        }
    }
}

)";

static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;
static id<MTLComputePipelineState> g_dequant_pso = nil;
static id<MTLComputePipelineState> g_gemv_pso = nil;
static id<MTLComputePipelineState> g_attn_v2_pso = nil;
static id<MTLCommandBuffer> g_pending_cmd = nil;
static id<MTLCommandBuffer> g_last_committed = nil;
static NSMutableDictionary<NSValue*, id<MTLBuffer>>* g_buffer_views = nil;
static id<MTLBuffer> g_dequant_scratch = nil;
static size_t g_dequant_scratch_size = 0;
static NSMutableDictionary<NSNumber*, MPSMatrixMultiplication*>* g_mm_cache = nil;
static NSMutableDictionary<NSNumber*, MPSMatrixDescriptor*>* g_desc_cache = nil;
static NSMutableDictionary<NSNumber*, MPSMatrix*>* g_dequant_mat_cache = nil;
static size_t g_page_size = 0;
static dispatch_once_t g_once;

static id<MTLBuffer> cactus_get_dequant_scratch(size_t bytes) {
    if (g_dequant_scratch && g_dequant_scratch_size >= bytes) return g_dequant_scratch;
    size_t want = bytes;
    if (g_dequant_scratch_size > 0) {
        size_t doubled = g_dequant_scratch_size * 2;
        if (doubled > want) want = doubled;
    }
    g_dequant_scratch = [g_device newBufferWithLength:want options:MTLResourceStorageModeShared];
    g_dequant_scratch_size = want;
    [g_dequant_mat_cache removeAllObjects];
    return g_dequant_scratch;
}

static MPSMatrix* cactus_get_dequant_mat(uint32_t N, uint32_t K) {
    uint64_t key = ((uint64_t)N << 32) | (uint64_t)K;
    NSNumber* k = @(key);
    MPSMatrix* m = g_dequant_mat_cache[k];
    if (!m) {
        MPSMatrixDescriptor* d = [MPSMatrixDescriptor matrixDescriptorWithRows:N
                                                                       columns:K
                                                                      rowBytes:K * sizeof(__fp16)
                                                                      dataType:MPSDataTypeFloat16];
        m = [[MPSMatrix alloc] initWithBuffer:g_dequant_scratch offset:0 descriptor:d];
        if (m) g_dequant_mat_cache[k] = m;
    }
    return m;
}

static MPSMatrixDescriptor* cactus_get_desc(uint32_t rows, uint32_t cols) {
    uint64_t key = ((uint64_t)rows << 32) | (uint64_t)cols;
    NSNumber* k = @(key);
    MPSMatrixDescriptor* d = g_desc_cache[k];
    if (!d) {
        d = [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                                   columns:cols
                                                  rowBytes:cols * sizeof(__fp16)
                                                  dataType:MPSDataTypeFloat16];
        if (d) g_desc_cache[k] = d;
    }
    return d;
}

static MPSMatrixMultiplication* cactus_get_mm(uint32_t M, uint32_t K, uint32_t N) {
    uint64_t key = ((uint64_t)M << 42) | ((uint64_t)K << 21) | (uint64_t)N;
    NSNumber* k = @(key);
    MPSMatrixMultiplication* mm = g_mm_cache[k];
    if (!mm) {
        mm = [[MPSMatrixMultiplication alloc] initWithDevice:g_device
            transposeLeft:NO transposeRight:YES resultRows:M resultColumns:N interiorColumns:K alpha:1.0 beta:0.0];
        if (mm) g_mm_cache[k] = mm;
    }
    return mm;
}

static void cactus_mps_init() {
    dispatch_once(&g_once, ^{
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) return;
        g_queue = [g_device newCommandQueue];
        g_buffer_views = [NSMutableDictionary new];
        g_mm_cache = [NSMutableDictionary new];
        g_desc_cache = [NSMutableDictionary new];
        g_dequant_mat_cache = [NSMutableDictionary new];
        g_page_size = sysconf(_SC_PAGESIZE);
        NSError* err = nil;
        id<MTLLibrary> lib = [g_device newLibraryWithSource:kCactusMSL options:nil error:&err];
        if (!lib) return;
        id<MTLFunction> dq = [lib newFunctionWithName:@"cactus_dequant_int4"];
        if (dq) g_dequant_pso = [g_device newComputePipelineStateWithFunction:dq error:&err];
        id<MTLFunction> gv = [lib newFunctionWithName:@"cactus_gemv_int4"];
        if (gv) g_gemv_pso = [g_device newComputePipelineStateWithFunction:gv error:&err];
        id<MTLFunction> at2 = [lib newFunctionWithName:@"cactus_flash_attn_f16_v2"];
        if (at2) g_attn_v2_pso = [g_device newComputePipelineStateWithFunction:at2 error:&err];
    });
}

static id<MTLBuffer> cactus_buffer_view(const void* ptr, size_t len, size_t* out_offset) {
    uintptr_t addr = (uintptr_t)ptr;
    uintptr_t aligned_addr = addr & ~(g_page_size - 1);
    *out_offset = addr - aligned_addr;
    size_t aligned_len = (*out_offset + len + g_page_size - 1) & ~(g_page_size - 1);
    NSValue* key = [NSValue valueWithPointer:(void*)aligned_addr];
    @synchronized(g_buffer_views) {
        id<MTLBuffer> buf = g_buffer_views[key];
        if (buf && [buf length] >= aligned_len) return buf;
        buf = [g_device newBufferWithBytesNoCopy:(void*)aligned_addr
                                          length:aligned_len
                                         options:MTLResourceStorageModeShared
                                     deallocator:nil];
        if (buf) g_buffer_views[key] = buf;
        return buf;
    }
}

static bool g_mps_enabled = true;

struct CactusMPSTraceCounters {
    std::atomic<uint64_t> matmul_f16{0};
    std::atomic<uint64_t> gemv_int4{0};
    std::atomic<uint64_t> matmul_int4{0};
    std::atomic<uint64_t> attention_f16{0};
    std::atomic<uint64_t> attention_graph{0};
};

static CactusMPSTraceCounters g_mps_trace_counters;
static std::atomic<uint64_t> g_mps_trace_event_index{0};

static bool parse_env_bool(const char* name, bool* out_value) {
    const char* raw = std::getenv(name);
    if (!raw || !*raw || !out_value) {
        return false;
    }

    std::string value(raw);
    for (char& ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }

    if (value == "1" || value == "true" || value == "yes" || value == "on") {
        *out_value = true;
        return true;
    }
    if (value == "0" || value == "false" || value == "no" || value == "off") {
        *out_value = false;
        return true;
    }

    return false;
}

static bool cactus_mps_env_enabled() {
    bool enabled = true;
    if (parse_env_bool("CACTUS_MPS", &enabled)) {
        return enabled;
    }

    bool disabled = false;
    if (parse_env_bool("CACTUS_DISABLE_MPS", &disabled)) {
        return !disabled;
    }

    return true;
}

static bool cactus_mps_trace_enabled() {
    bool enabled = false;
    return parse_env_bool("CACTUS_MPS_TRACE", &enabled) && enabled;
}

static bool cactus_mps_trace_summary_enabled() {
    bool enabled = false;
    if (parse_env_bool("CACTUS_MPS_TRACE_SUMMARY", &enabled)) {
        return enabled;
    }
    return cactus_mps_trace_enabled();
}

static void cactus_mps_trace_dump_summary() {
    if (!cactus_mps_trace_summary_enabled()) {
        return;
    }

    cactus_mps_init();
    const int enabled = (g_mps_enabled && cactus_mps_env_enabled()) ? 1 : 0;
    const int available = (g_device != nil && g_queue != nil) ? 1 : 0;

    const unsigned long long matmul_f16 =
        static_cast<unsigned long long>(g_mps_trace_counters.matmul_f16.load(std::memory_order_relaxed));
    const unsigned long long gemv_int4 =
        static_cast<unsigned long long>(g_mps_trace_counters.gemv_int4.load(std::memory_order_relaxed));
    const unsigned long long matmul_int4 =
        static_cast<unsigned long long>(g_mps_trace_counters.matmul_int4.load(std::memory_order_relaxed));
    const unsigned long long attention_f16 =
        static_cast<unsigned long long>(g_mps_trace_counters.attention_f16.load(std::memory_order_relaxed));
    const unsigned long long attention_graph =
        static_cast<unsigned long long>(g_mps_trace_counters.attention_graph.load(std::memory_order_relaxed));
    const unsigned long long total =
        matmul_f16 + gemv_int4 + matmul_int4 + attention_f16 + attention_graph;

    std::fprintf(stderr,
                 "[MPS_TRACE_SUMMARY] enabled=%d available=%d total=%llu matmul_f16=%llu gemv_int4=%llu "
                 "matmul_int4=%llu attention_f16=%llu attention_graph=%llu\n",
                 enabled, available, total, matmul_f16, gemv_int4, matmul_int4, attention_f16, attention_graph);
    std::fflush(stderr);
}

static void cactus_mps_trace_register_summary() {
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        std::atexit(cactus_mps_trace_dump_summary);
    });
}

static void cactus_mps_trace_log(const char* kernel_name, const std::string& details) {
    if (!cactus_mps_trace_enabled() && !cactus_mps_trace_summary_enabled()) {
        return;
    }

    cactus_mps_trace_register_summary();

    if (!cactus_mps_trace_enabled()) {
        return;
    }

    const unsigned long long event_index =
        static_cast<unsigned long long>(1 + g_mps_trace_event_index.fetch_add(1, std::memory_order_relaxed));
    std::fprintf(stderr, "[MPS_TRACE] #%llu %s %s\n", event_index, kernel_name, details.c_str());
    std::fflush(stderr);
}

bool cactus_mps_available() {
    if (cactus_mps_trace_summary_enabled()) {
        cactus_mps_trace_register_summary();
    }
    cactus_mps_init();
    return g_device != nil && g_queue != nil;
}

void cactus_mps_set_enabled(bool enabled) { g_mps_enabled = enabled; }
bool cactus_mps_enabled() { return g_mps_enabled && cactus_mps_env_enabled(); }

static id<MTLCommandBuffer> cactus_mps_active_cmd() {
    if (!g_pending_cmd) {
        g_pending_cmd = [g_queue commandBuffer];
    }
    return g_pending_cmd;
}

void cactus_mps_flush() {
    if (g_pending_cmd) {
        [g_pending_cmd commit];
        g_last_committed = g_pending_cmd;
        g_pending_cmd = nil;
    }
}

void cactus_mps_synchronize() {
    if (g_pending_cmd) {
        [g_pending_cmd commit];
        g_last_committed = g_pending_cmd;
        g_pending_cmd = nil;
    }
    if (g_last_committed) {
        [g_last_committed waitUntilCompleted];
        g_last_committed = nil;
    }
}

void cactus_matmul_f16_mps(const __fp16* A, const __fp16* B_T, __fp16* C,
                           size_t M, size_t K, size_t N) {
    cactus_mps_init();
    if (!g_device || !g_queue) return;

    g_mps_trace_counters.matmul_f16.fetch_add(1, std::memory_order_relaxed);
    cactus_mps_trace_log("matmul_f16",
                         "M=" + std::to_string(M) +
                         " K=" + std::to_string(K) +
                         " N=" + std::to_string(N));

    @autoreleasepool {
        const size_t fp16 = sizeof(__fp16);
        size_t offA, offB, offC;
        id<MTLBuffer> bufA = cactus_buffer_view(A, M*K*fp16, &offA);
        id<MTLBuffer> bufB = cactus_buffer_view(B_T, N*K*fp16, &offB);
        id<MTLBuffer> bufC = cactus_buffer_view(C, M*N*fp16, &offC);

        MPSMatrixDescriptor* dA = cactus_get_desc((uint32_t)M, (uint32_t)K);
        MPSMatrixDescriptor* dB = cactus_get_desc((uint32_t)N, (uint32_t)K);
        MPSMatrixDescriptor* dC = cactus_get_desc((uint32_t)M, (uint32_t)N);

        MPSMatrix* mA = [[MPSMatrix alloc] initWithBuffer:bufA offset:offA descriptor:dA];
        MPSMatrix* mB = [[MPSMatrix alloc] initWithBuffer:bufB offset:offB descriptor:dB];
        MPSMatrix* mC = [[MPSMatrix alloc] initWithBuffer:bufC offset:offC descriptor:dC];

        MPSMatrixMultiplication* mm = cactus_get_mm((uint32_t)M, (uint32_t)K, (uint32_t)N);

        id<MTLCommandBuffer> cmd = cactus_mps_active_cmd();
        [mm encodeToCommandBuffer:cmd leftMatrix:mA rightMatrix:mB resultMatrix:mC];
    }
}

void cactus_gemv_int4_mps(const __fp16* A, const int8_t* B_packed, const __fp16* B_scales,
                          __fp16* C, size_t K, size_t N, size_t group_size) {
    cactus_mps_init();
    if (!g_device || !g_queue || !g_gemv_pso) return;
    if (N % 4 != 0 || K % group_size != 0) return;

    g_mps_trace_counters.gemv_int4.fetch_add(1, std::memory_order_relaxed);
    cactus_mps_trace_log("gemv_int4",
                         "K=" + std::to_string(K) +
                         " N=" + std::to_string(N) +
                         " group_size=" + std::to_string(group_size));

    @autoreleasepool {
        const size_t fp16 = sizeof(__fp16);
        const size_t packed_bytes = (N / 4) * K * 2;
        const size_t num_groups = K / group_size;
        const size_t scales_bytes = (N / 4) * num_groups * 4 * fp16;

        size_t offA, offBp, offBs, offC;
        id<MTLBuffer> bufA = cactus_buffer_view(A, K*fp16, &offA);
        id<MTLBuffer> bufBp = cactus_buffer_view(B_packed, packed_bytes, &offBp);
        id<MTLBuffer> bufBs = cactus_buffer_view(B_scales, scales_bytes, &offBs);
        id<MTLBuffer> bufC = cactus_buffer_view(C, N*fp16, &offC);

        id<MTLCommandBuffer> cmd = cactus_mps_active_cmd();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_gemv_pso];
        [enc setBuffer:bufA offset:offA atIndex:0];
        [enc setBuffer:bufBp offset:offBp atIndex:1];
        [enc setBuffer:bufBs offset:offBs atIndex:2];
        [enc setBuffer:bufC offset:offC atIndex:3];
        uint32_t Ku = (uint32_t)K, Nu = (uint32_t)N, Gu = (uint32_t)group_size;
        [enc setBytes:&Ku length:sizeof(Ku) atIndex:4];
        [enc setBytes:&Nu length:sizeof(Nu) atIndex:5];
        [enc setBytes:&Gu length:sizeof(Gu) atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(N / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [enc endEncoding];
    }
}

void cactus_matmul_int4_mps(const __fp16* A, const int8_t* B_packed, const __fp16* B_scales,
                            __fp16* C, size_t M, size_t K, size_t N, size_t group_size) {
    cactus_mps_init();
    if (!g_device || !g_queue || !g_dequant_pso) return;
    if (N % 4 != 0 || K % group_size != 0) return;

    g_mps_trace_counters.matmul_int4.fetch_add(1, std::memory_order_relaxed);
    cactus_mps_trace_log("matmul_int4",
                         "M=" + std::to_string(M) +
                         " K=" + std::to_string(K) +
                         " N=" + std::to_string(N) +
                         " group_size=" + std::to_string(group_size));

    @autoreleasepool {
        const size_t fp16 = sizeof(__fp16);
        const size_t packed_bytes = (N / 4) * K * 2;
        const size_t num_groups = K / group_size;
        const size_t scales_bytes = (N / 4) * num_groups * 4 * fp16;

        size_t offA, offBp, offBs, offC;
        id<MTLBuffer> bufA = cactus_buffer_view(A, M*K*fp16, &offA);
        id<MTLBuffer> bufBp = cactus_buffer_view(B_packed, packed_bytes, &offBp);
        id<MTLBuffer> bufBs = cactus_buffer_view(B_scales, scales_bytes, &offBs);
        id<MTLBuffer> bufBd = cactus_get_dequant_scratch(N*K*fp16);
        id<MTLBuffer> bufC = cactus_buffer_view(C, M*N*fp16, &offC);

        id<MTLCommandBuffer> cmd = cactus_mps_active_cmd();

        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_dequant_pso];
        [enc setBuffer:bufBp offset:offBp atIndex:0];
        [enc setBuffer:bufBs offset:offBs atIndex:1];
        [enc setBuffer:bufBd offset:0 atIndex:2];
        uint32_t Ku = (uint32_t)K;
        uint32_t Gu = (uint32_t)group_size;
        [enc setBytes:&Ku length:sizeof(Ku) atIndex:3];
        [enc setBytes:&Gu length:sizeof(Gu) atIndex:4];
        [enc dispatchThreads:MTLSizeMake(K / 8, N, 1) threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
        [enc endEncoding];

        MPSMatrixDescriptor* dA = cactus_get_desc((uint32_t)M, (uint32_t)K);
        MPSMatrixDescriptor* dC = cactus_get_desc((uint32_t)M, (uint32_t)N);
        MPSMatrix* mA = [[MPSMatrix alloc] initWithBuffer:bufA offset:offA descriptor:dA];
        MPSMatrix* mB = cactus_get_dequant_mat((uint32_t)N, (uint32_t)K);
        MPSMatrix* mC = [[MPSMatrix alloc] initWithBuffer:bufC offset:offC descriptor:dC];
        MPSMatrixMultiplication* mm = cactus_get_mm((uint32_t)M, (uint32_t)K, (uint32_t)N);
        [mm encodeToCommandBuffer:cmd leftMatrix:mA rightMatrix:mB resultMatrix:mC];
    }
}

void cactus_attention_f16_mps(const __fp16* Q, const __fp16* K, const __fp16* V, __fp16* O,
                              size_t seq_len, size_t kv_seq_len,
                              size_t num_q_heads, size_t num_kv_heads,
                              size_t head_dim, float scale, size_t position_offset) {
    cactus_mps_init();
    if (!g_device || !g_queue || !g_attn_v2_pso) return;
    if (head_dim > 256 || head_dim % 16 != 0) return;

    g_mps_trace_counters.attention_f16.fetch_add(1, std::memory_order_relaxed);
    cactus_mps_trace_log("attention_f16",
                         "seq_len=" + std::to_string(seq_len) +
                         " kv_seq_len=" + std::to_string(kv_seq_len) +
                         " num_q_heads=" + std::to_string(num_q_heads) +
                         " num_kv_heads=" + std::to_string(num_kv_heads) +
                         " head_dim=" + std::to_string(head_dim) +
                         " position_offset=" + std::to_string(position_offset));

    @autoreleasepool {
        const size_t fp16 = sizeof(__fp16);
        const size_t q_bytes = seq_len * num_q_heads * head_dim * fp16;
        const size_t kv_bytes = kv_seq_len * num_kv_heads * head_dim * fp16;
        const size_t o_bytes = seq_len * num_q_heads * head_dim * fp16;

        size_t offQ, offK, offV, offO;
        id<MTLBuffer> bufQ = cactus_buffer_view(Q, q_bytes, &offQ);
        id<MTLBuffer> bufK = cactus_buffer_view(K, kv_bytes, &offK);
        id<MTLBuffer> bufV = cactus_buffer_view(V, kv_bytes, &offV);
        id<MTLBuffer> bufO = cactus_buffer_view(O, o_bytes, &offO);

        id<MTLCommandBuffer> cmd = cactus_mps_active_cmd();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_attn_v2_pso];
        [enc setBuffer:bufQ offset:offQ atIndex:0];
        [enc setBuffer:bufK offset:offK atIndex:1];
        [enc setBuffer:bufV offset:offV atIndex:2];
        [enc setBuffer:bufO offset:offO atIndex:3];
        uint32_t SL = (uint32_t)seq_len, KL = (uint32_t)kv_seq_len;
        uint32_t QH = (uint32_t)num_q_heads, KH = (uint32_t)num_kv_heads;
        uint32_t HD = (uint32_t)head_dim;
        uint32_t PO = (uint32_t)position_offset;
        [enc setBytes:&SL length:sizeof(SL) atIndex:4];
        [enc setBytes:&KL length:sizeof(KL) atIndex:5];
        [enc setBytes:&QH length:sizeof(QH) atIndex:6];
        [enc setBytes:&KH length:sizeof(KH) atIndex:7];
        [enc setBytes:&HD length:sizeof(HD) atIndex:8];
        [enc setBytes:&scale length:sizeof(scale) atIndex:9];
        [enc setBytes:&PO length:sizeof(PO) atIndex:10];
        const size_t Q_BLOCK_V2 = 8, KV_BLOCK_V2 = 16;
        const size_t shmem_bytes =
            (Q_BLOCK_V2 + KV_BLOCK_V2 + KV_BLOCK_V2) * head_dim * sizeof(__fp16) +
            Q_BLOCK_V2 * KV_BLOCK_V2 * sizeof(__fp16) +
            Q_BLOCK_V2 * head_dim * sizeof(float) +
            (Q_BLOCK_V2 + Q_BLOCK_V2 + Q_BLOCK_V2 * KV_BLOCK_V2) * sizeof(float);
        [enc setThreadgroupMemoryLength:shmem_bytes atIndex:0];
        const size_t q_blocks = (seq_len + Q_BLOCK_V2 - 1) / Q_BLOCK_V2;
        [enc dispatchThreadgroups:MTLSizeMake(q_blocks, num_q_heads, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [enc endEncoding];
    }
}

API_AVAILABLE(macos(14.0), ios(17.0))
@interface CactusSDPAGraph : NSObject
@property (nonatomic, strong) MPSGraph* graph;
@property (nonatomic, strong) MPSGraphTensor* qPh;
@property (nonatomic, strong) MPSGraphTensor* kPh;
@property (nonatomic, strong) MPSGraphTensor* vPh;
@property (nonatomic, strong) MPSGraphTensor* outT;
@end
@implementation CactusSDPAGraph
@end

static NSMutableDictionary<NSString*, CactusSDPAGraph*>* g_sdpa_graph_cache = nil;

API_AVAILABLE(macos(14.0), ios(17.0))
static CactusSDPAGraph* cactus_get_sdpa_graph(size_t seq_len, size_t kv_seq_len,
                                               size_t num_q_heads, size_t num_kv_heads,
                                               size_t head_dim, float scale, size_t position_offset) {
    if (!g_sdpa_graph_cache) g_sdpa_graph_cache = [NSMutableDictionary new];
    NSString* key = [NSString stringWithFormat:@"%zu_%zu_%zu_%zu_%zu_%zu_%a",
                     seq_len, kv_seq_len, num_q_heads, num_kv_heads,
                     head_dim, position_offset, (double)scale];
    CactusSDPAGraph* cached = g_sdpa_graph_cache[key];
    if (cached) return cached;

    MPSGraph* graph = [[MPSGraph alloc] init];

    MPSGraphTensor* qPh = [graph placeholderWithShape:@[@(seq_len), @(num_q_heads), @(head_dim)]
                                             dataType:MPSDataTypeFloat16 name:nil];
    MPSGraphTensor* kPh = [graph placeholderWithShape:@[@(kv_seq_len), @(num_kv_heads), @(head_dim)]
                                             dataType:MPSDataTypeFloat16 name:nil];
    MPSGraphTensor* vPh = [graph placeholderWithShape:@[@(kv_seq_len), @(num_kv_heads), @(head_dim)]
                                             dataType:MPSDataTypeFloat16 name:nil];

    MPSGraphTensor* qT = [graph transposeTensor:qPh permutation:@[@1, @0, @2] name:nil];
    MPSGraphTensor* kT = [graph transposeTensor:kPh permutation:@[@1, @0, @2] name:nil];
    MPSGraphTensor* vT = [graph transposeTensor:vPh permutation:@[@1, @0, @2] name:nil];

    if (num_q_heads != num_kv_heads) {
        const NSUInteger groups = num_q_heads / num_kv_heads;
        kT = [graph reshapeTensor:kT withShape:@[@(num_kv_heads), @1, @(kv_seq_len), @(head_dim)] name:nil];
        kT = [graph broadcastTensor:kT toShape:@[@(num_kv_heads), @(groups), @(kv_seq_len), @(head_dim)] name:nil];
        kT = [graph reshapeTensor:kT withShape:@[@(num_q_heads), @(kv_seq_len), @(head_dim)] name:nil];
        vT = [graph reshapeTensor:vT withShape:@[@(num_kv_heads), @1, @(kv_seq_len), @(head_dim)] name:nil];
        vT = [graph broadcastTensor:vT toShape:@[@(num_kv_heads), @(groups), @(kv_seq_len), @(head_dim)] name:nil];
        vT = [graph reshapeTensor:vT withShape:@[@(num_q_heads), @(kv_seq_len), @(head_dim)] name:nil];
    }

    std::vector<__fp16> mask_data(seq_len * kv_seq_len);
    const __fp16 neg_inf = (__fp16)(-65504.0f);
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < kv_seq_len; ++j) {
            bool allowed = (j <= position_offset + i);
            mask_data[i * kv_seq_len + j] = allowed ? (__fp16)0.0f : neg_inf;
        }
    }
    NSData* maskNSData = [NSData dataWithBytes:mask_data.data() length:mask_data.size() * sizeof(__fp16)];
    MPSGraphTensor* maskTensor = [graph constantWithData:maskNSData
                                                   shape:@[@(seq_len), @(kv_seq_len)]
                                                dataType:MPSDataTypeFloat16];

    MPSGraphTensor* outT = [graph scaledDotProductAttentionWithQueryTensor:qT
                                                                keyTensor:kT
                                                              valueTensor:vT
                                                               maskTensor:maskTensor
                                                                    scale:scale
                                                                     name:nil];
    outT = [graph transposeTensor:outT permutation:@[@1, @0, @2] name:nil];

    CactusSDPAGraph* entry = [CactusSDPAGraph new];
    entry.graph = graph;
    entry.qPh = qPh;
    entry.kPh = kPh;
    entry.vPh = vPh;
    entry.outT = outT;
    g_sdpa_graph_cache[key] = entry;
    return entry;
}

void cactus_attention_f16_mpsgraph(const __fp16* Q, const __fp16* K, const __fp16* V, __fp16* O,
                                    size_t seq_len, size_t kv_seq_len,
                                    size_t num_q_heads, size_t num_kv_heads,
                                    size_t head_dim, float scale, size_t position_offset) {
    cactus_mps_init();
    if (!g_device || !g_queue) return;

    g_mps_trace_counters.attention_graph.fetch_add(1, std::memory_order_relaxed);
    cactus_mps_trace_log("attention_graph",
                         "seq_len=" + std::to_string(seq_len) +
                         " kv_seq_len=" + std::to_string(kv_seq_len) +
                         " num_q_heads=" + std::to_string(num_q_heads) +
                         " num_kv_heads=" + std::to_string(num_kv_heads) +
                         " head_dim=" + std::to_string(head_dim) +
                         " position_offset=" + std::to_string(position_offset));

    @autoreleasepool {
        const size_t fp16 = sizeof(__fp16);
        const size_t q_bytes = seq_len * num_q_heads * head_dim * fp16;
        const size_t kv_bytes = kv_seq_len * num_kv_heads * head_dim * fp16;
        const size_t o_bytes = seq_len * num_q_heads * head_dim * fp16;

        cactus_mps_synchronize();

        id<MTLBuffer> bufQ = [g_device newBufferWithBytes:Q length:q_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufK = [g_device newBufferWithBytes:K length:kv_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufV = [g_device newBufferWithBytes:V length:kv_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufO = [g_device newBufferWithLength:o_bytes options:MTLResourceStorageModeShared];

        CactusSDPAGraph* entry = cactus_get_sdpa_graph(seq_len, kv_seq_len,
                                                       num_q_heads, num_kv_heads,
                                                       head_dim, scale, position_offset);

        MPSGraphTensorData* qData = [[MPSGraphTensorData alloc] initWithMTLBuffer:bufQ
                                                                            shape:@[@(seq_len), @(num_q_heads), @(head_dim)]
                                                                         dataType:MPSDataTypeFloat16];
        MPSGraphTensorData* kData = [[MPSGraphTensorData alloc] initWithMTLBuffer:bufK
                                                                            shape:@[@(kv_seq_len), @(num_kv_heads), @(head_dim)]
                                                                         dataType:MPSDataTypeFloat16];
        MPSGraphTensorData* vData = [[MPSGraphTensorData alloc] initWithMTLBuffer:bufV
                                                                            shape:@[@(kv_seq_len), @(num_kv_heads), @(head_dim)]
                                                                         dataType:MPSDataTypeFloat16];
        MPSGraphTensorData* oData = [[MPSGraphTensorData alloc] initWithMTLBuffer:bufO
                                                                            shape:@[@(seq_len), @(num_q_heads), @(head_dim)]
                                                                         dataType:MPSDataTypeFloat16];

        NSDictionary* feeds = @{entry.qPh: qData, entry.kPh: kData, entry.vPh: vData};
        NSDictionary* results = @{entry.outT: oData};

        [entry.graph runWithMTLCommandQueue:g_queue
                                      feeds:feeds
                           targetOperations:nil
                          resultsDictionary:results];

        memcpy(O, [bufO contents], o_bytes);
    }
}

#endif
