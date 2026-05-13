#ifdef __APPLE__

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "../cactus_kernels.h"
#include <cstring>

static id<MTLDevice> g_mps_device = nil;
static id<MTLCommandQueue> g_mps_queue = nil;
static id<MTLCommandQueue> g_mps_queue2 = nil;
static int g_queue_idx = 0;
static NSMutableDictionary<NSNumber*, MPSMatrixMultiplication*>* g_mps_mm_cache = nil;
static NSMutableDictionary<NSNumber*, MPSMatrixDescriptor*>* g_mps_desc_cache = nil;
static dispatch_once_t g_mps_once;
static bool g_mps_enabled_flag = true;

static id<MTLComputePipelineState> g_cq4_dequant_pso = nil;
static id<MTLComputePipelineState> g_cq4_matmul_fused_pso = nil;
static NSMutableDictionary<NSValue*, id<MTLBuffer>>* g_cq4_packed_cache = nil;
static NSMutableDictionary<NSValue*, id<MTLBuffer>>* g_cq4_codebook_cache = nil;
static NSMutableDictionary<NSValue*, id<MTLBuffer>>* g_cq4_norms_cache = nil;

static id<MTLBuffer> g_scratch_decoded_w = nil;
static id<MTLBuffer> g_scratch_a = nil;
static id<MTLBuffer> g_scratch_c = nil;

// Per-W decoded fp16 weight cache (private storage, persists across calls).
static NSMutableDictionary<NSValue*, id<MTLBuffer>>* g_cq4_decoded_cache = nil;

struct PendingMps {
    __strong id<MTLCommandBuffer> cb;
    __strong id<MTLBuffer>        gpu_result;
    void*                         cpu_dest;
    size_t                        byte_size;
};
#include <vector>
static std::vector<PendingMps>* g_pending_list = nullptr;

static NSString* const CQ4_DEQUANT_METAL = @R"METAL(
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// Standalone dequant kernel (kept for fallback paths)
kernel void cactus_cq4_dequant(
    device const uchar*   packed     [[ buffer(0) ]],
    constant half*        codebook   [[ buffer(1) ]],
    device const half*    norms      [[ buffer(2) ]],
    device half4*         out4       [[ buffer(3) ]],
    constant uint&        K          [[ buffer(4) ]],
    constant uint&        num_groups [[ buffer(5) ]],
    constant uint&        group_size [[ buffer(6) ]],
    constant uint&        pgb        [[ buffer(7) ]],
    uint2 gid [[ thread_position_in_grid ]])
{
    uint k_v = gid.x;
    uint row = gid.y;
    uint k = k_v * 8;
    uint group = k / group_size;
    uint k_in = k - group * group_size;
    uint byte_off = k_in >> 1;
    uint base = row * num_groups * pgb + group * pgb + byte_off;
    uchar b0 = packed[base];
    uchar b1 = packed[base + 1];
    uchar b2 = packed[base + 2];
    uchar b3 = packed[base + 3];
    half nm = norms[row * num_groups + group];
    half4 r0;
    r0.x = codebook[b0 & 0xF] * nm;
    r0.y = codebook[b0 >> 4]  * nm;
    r0.z = codebook[b1 & 0xF] * nm;
    r0.w = codebook[b1 >> 4]  * nm;
    half4 r1;
    r1.x = codebook[b2 & 0xF] * nm;
    r1.y = codebook[b2 >> 4]  * nm;
    r1.z = codebook[b3 & 0xF] * nm;
    r1.w = codebook[b3 >> 4]  * nm;
    uint out4_base = (row * K + k) >> 2;
    out4[out4_base + 0] = r0;
    out4[out4_base + 1] = r1;
}

// Fused CQ4 dequant + matmul.
// Computes C[M,N] = A[M,K] @ W^T where W is CQ4-encoded.
// Big tiles: TM=64, TN=64, TK=32 with 1024-thread TG (32 simdgroups).
// Each sg owns 1 row block (8 rows) × 2 col blocks (16 cols) of output → 2 simdgroup_matrix tiles.
kernel void cactus_cq4_matmul_fused(
    device const half*  A           [[ buffer(0) ]],   // [M, K]
    device const uchar* packed      [[ buffer(1) ]],   // [N, num_groups*pgb]
    constant half*      codebook    [[ buffer(2) ]],   // [16]
    device const half*  norms       [[ buffer(3) ]],   // [N, num_groups]
    device half*        C           [[ buffer(4) ]],   // [M, N]
    constant uint&      M           [[ buffer(5) ]],
    constant uint&      N           [[ buffer(6) ]],
    constant uint&      K           [[ buffer(7) ]],
    constant uint&      group_size  [[ buffer(8) ]],
    constant uint&      num_groups  [[ buffer(9) ]],
    constant uint&      pgb         [[ buffer(10) ]],
    uint2 tg_id  [[ threadgroup_position_in_grid ]],
    uint  tid    [[ thread_index_in_threadgroup ]],
    uint  sg_id  [[ simdgroup_index_in_threadgroup ]])
{
    constexpr int TM = 64;
    constexpr int TN = 64;
    constexpr int TK = 32;
    constexpr int TK_BYTES = TK / 2;       // 16 bytes per N row in tile
    constexpr int TG_THREADS = 1024;
    constexpr int N_SG = TG_THREADS / 32;  // 32 simdgroups

    threadgroup half a_tile[TM][TK + 4];
    threadgroup half w_tile[TK][TN + 4];

    uint m_base = tg_id.y * TM;
    uint n_base = tg_id.x * TN;

    // 32 simdgroups: 8 row blocks (sg_m=0..7) × 4 col-pair blocks (sg_n=0..3)
    uint sg_m = sg_id / 4;
    uint sg_n = sg_id % 4;

    // Each simdgroup accumulates 2 simdgroup_matrix tiles (cols sg_n*16..+15)
    simdgroup_half8x8 acc[2];
    acc[0] = make_filled_simdgroup_matrix<half, 8, 8>((half)0);
    acc[1] = make_filled_simdgroup_matrix<half, 8, 8>((half)0);

    for (uint k_base = 0; k_base < K; k_base += TK) {
        // Hoisted: group is constant within this k-iteration when TK <= group_size.
        // (For Gemma-4 E2B: TK=32, group_size=128 → group_size/TK=4 K-iters per group).
        uint group = k_base / group_size;
        uint k_offset_in_group = k_base - group * group_size;
        uint group_packed_base = group * pgb;

        // Load A_tile [TM × TK] = [64 × 32] = 2048 halfs.
        // 1024 threads × 2 halfs each.
        for (uint i = 0; i < 2; ++i) {
            uint flat = tid + i * TG_THREADS;
            uint mm = flat / TK;
            uint kk = flat - mm * TK;
            uint m = m_base + mm;
            uint k = k_base + kk;
            a_tile[mm][kk] = (m < M && k < K) ? A[m * K + k] : (half)0;
        }

        // Load W_tile [TK × TN] = [32 × 64] = 2048 halfs (decoded).
        // 1024 threads × 1 packed byte each → 2 decoded halfs.
        {
            uint byte_in_row = tid % TK_BYTES;       // 0..15
            uint n_in_tile   = tid / TK_BYTES;       // 0..63

            uint n  = n_base + n_in_tile;
            uint b_idx = (k_offset_in_group >> 1) + byte_in_row;

            half v0 = (half)0, v1 = (half)0;
            if ((k_base + byte_in_row * 2) < K && n < N) {
                uchar b = packed[n * num_groups * pgb + group_packed_base + b_idx];
                half nm = norms[n * num_groups + group];
                v0 = codebook[b & 0xF] * nm;
                v1 = codebook[b >> 4]  * nm;
            }
            w_tile[byte_in_row * 2 + 0][n_in_tile] = v0;
            w_tile[byte_in_row * 2 + 1][n_in_tile] = v1;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // K substeps of 8 → TK/8 = 4 substeps.
        for (int kk_tile = 0; kk_tile < 4; ++kk_tile) {
            simdgroup_half8x8 a_mat;
            simdgroup_load(a_mat, &a_tile[sg_m * 8][kk_tile * 8], TK + 4);

            simdgroup_half8x8 b_mat0, b_mat1;
            simdgroup_load(b_mat0, &w_tile[kk_tile * 8][sg_n * 16 + 0], TN + 4);
            simdgroup_load(b_mat1, &w_tile[kk_tile * 8][sg_n * 16 + 8], TN + 4);

            simdgroup_multiply_accumulate(acc[0], a_mat, b_mat0, acc[0]);
            simdgroup_multiply_accumulate(acc[1], a_mat, b_mat1, acc[1]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup half c_tile[TM][TN + 4];
    simdgroup_store(acc[0], &c_tile[sg_m * 8][sg_n * 16 + 0], TN + 4);
    simdgroup_store(acc[1], &c_tile[sg_m * 8][sg_n * 16 + 8], TN + 4);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write c_tile back to C: TM*TN = 4096 elements, 1024 threads × 4 each.
    for (uint i = 0; i < 4; ++i) {
        uint flat = tid + i * TG_THREADS;
        uint mm = flat / TN;
        uint nn = flat - mm * TN;
        uint m = m_base + mm;
        uint n = n_base + nn;
        if (m < M && n < N) {
            C[m * N + n] = c_tile[mm][nn];
        }
    }
}
)METAL";

static void cactus_mps_init() {
    dispatch_once(&g_mps_once, ^{
        g_mps_device = MTLCreateSystemDefaultDevice();
        if (!g_mps_device) return;
        g_mps_queue = [g_mps_device newCommandQueue];
        g_mps_queue2 = [g_mps_device newCommandQueue];
        g_mps_mm_cache = [NSMutableDictionary new];
        g_mps_desc_cache = [NSMutableDictionary new];
        g_cq4_packed_cache = [NSMutableDictionary new];
        g_cq4_codebook_cache = [NSMutableDictionary new];
        g_cq4_norms_cache = [NSMutableDictionary new];
        g_cq4_decoded_cache = [NSMutableDictionary new];
        g_pending_list = new std::vector<PendingMps>();
        g_pending_list->reserve(8);

        NSError* err = nil;
        id<MTLLibrary> lib = [g_mps_device newLibraryWithSource:CQ4_DEQUANT_METAL options:nil error:&err];
        if (!lib) {
            NSLog(@"[cactus] failed to compile metal library: %@", err);
        }
        if (lib) {
            id<MTLFunction> fn1 = [lib newFunctionWithName:@"cactus_cq4_dequant"];
            if (fn1) {
                g_cq4_dequant_pso = [g_mps_device newComputePipelineStateWithFunction:fn1 error:&err];
            }
            id<MTLFunction> fn2 = [lib newFunctionWithName:@"cactus_cq4_matmul_fused"];
            if (fn2) {
                g_cq4_matmul_fused_pso = [g_mps_device newComputePipelineStateWithFunction:fn2 error:&err];
                if (!g_cq4_matmul_fused_pso) {
                    NSLog(@"[cactus] failed to compile fused matmul pso: %@", err);
                }
            }
        }
    });
}

extern "C" bool cactus_mps_available(void) {
    cactus_mps_init();
    return g_mps_device != nil && g_mps_queue != nil;
}

extern "C" void cactus_mps_set_enabled(bool enabled) { g_mps_enabled_flag = enabled; }
extern "C" bool cactus_mps_enabled(void) { return g_mps_enabled_flag; }

static MPSMatrixDescriptor* mps_get_desc(uint32_t rows, uint32_t cols) {
    uint64_t key = ((uint64_t)rows << 32) | (uint64_t)cols;
    NSNumber* k = @(key);
    MPSMatrixDescriptor* d = g_mps_desc_cache[k];
    if (!d) {
        d = [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                                   columns:cols
                                                  rowBytes:cols * sizeof(__fp16)
                                                  dataType:MPSDataTypeFloat16];
        if (d) g_mps_desc_cache[k] = d;
    }
    return d;
}

static MPSMatrixMultiplication* mps_get_mm(uint32_t M, uint32_t K, uint32_t N) {
    uint64_t key = ((uint64_t)M << 42) | ((uint64_t)K << 21) | (uint64_t)N;
    NSNumber* k = @(key);
    MPSMatrixMultiplication* mm = g_mps_mm_cache[k];
    if (!mm) {
        mm = [[MPSMatrixMultiplication alloc] initWithDevice:g_mps_device
            transposeLeft:NO transposeRight:YES
            resultRows:M resultColumns:N interiorColumns:K
            alpha:1.0 beta:0.0];
        if (mm) g_mps_mm_cache[k] = mm;
    }
    return mm;
}

void cactus_matmul_f16_mps(
    const __fp16* A, const __fp16* B_T, __fp16* C,
    size_t M, size_t K, size_t N)
{
    cactus_mps_init();
    if (!g_mps_device || !g_mps_queue) return;

    @autoreleasepool {
        const size_t fp16 = sizeof(__fp16);
        size_t a_size = M*K*fp16;
        size_t b_size = N*K*fp16;
        size_t c_size = M*N*fp16;

        if (g_scratch_a == nil || g_scratch_a.length < a_size) {
            g_scratch_a = [g_mps_device newBufferWithLength:a_size options:MTLResourceStorageModeShared];
        }
        memcpy([g_scratch_a contents], A, a_size);

        id<MTLBuffer> bufB = [g_mps_device newBufferWithBytes:B_T length:b_size options:MTLResourceStorageModeShared];

        if (g_scratch_c == nil || g_scratch_c.length < c_size) {
            g_scratch_c = [g_mps_device newBufferWithLength:c_size options:MTLResourceStorageModeShared];
        }

        MPSMatrixDescriptor* dA = mps_get_desc((uint32_t)M, (uint32_t)K);
        MPSMatrixDescriptor* dB = mps_get_desc((uint32_t)N, (uint32_t)K);
        MPSMatrixDescriptor* dC = mps_get_desc((uint32_t)M, (uint32_t)N);

        MPSMatrix* mA = [[MPSMatrix alloc] initWithBuffer:g_scratch_a descriptor:dA];
        MPSMatrix* mB = [[MPSMatrix alloc] initWithBuffer:bufB descriptor:dB];
        MPSMatrix* mC = [[MPSMatrix alloc] initWithBuffer:g_scratch_c descriptor:dC];

        MPSMatrixMultiplication* mm = mps_get_mm((uint32_t)M, (uint32_t)K, (uint32_t)N);

        id<MTLCommandBuffer> cmd = [g_mps_queue commandBuffer];
        [mm encodeToCommandBuffer:cmd leftMatrix:mA rightMatrix:mB resultMatrix:mC];
        [cmd commit];
        [cmd waitUntilCompleted];
        memcpy(C, [g_scratch_c contents], c_size);
    }
}

extern "C" bool cactus_cq4_matmul_fused_mps(
    const void* packed_indices, const void* codebook, const void* norms,
    const __fp16* hadamard_A, __fp16* C,
    uint32_t M, uint32_t K, uint32_t N,
    uint32_t group_size, uint32_t num_groups)
{
    cactus_mps_init();
    if (!g_mps_device || !g_mps_queue || !g_cq4_matmul_fused_pso) return false;
    if ((K & 31) != 0) return false;   // require K divisible by 32 (TK)

    @autoreleasepool {
        const size_t fp16 = sizeof(__fp16);
        const uint32_t pgb = group_size / 2;

        NSValue* w_key = [NSValue valueWithPointer:packed_indices];
        id<MTLBuffer> bufPacked = g_cq4_packed_cache[w_key];
        id<MTLBuffer> bufCodebook = g_cq4_codebook_cache[w_key];
        id<MTLBuffer> bufNorms = g_cq4_norms_cache[w_key];
        if (!bufPacked) {
            size_t packed_bytes = (size_t)N * num_groups * pgb;
            bufPacked = [g_mps_device newBufferWithBytes:packed_indices length:packed_bytes options:MTLResourceStorageModeShared];
            g_cq4_packed_cache[w_key] = bufPacked;
        }
        if (!bufCodebook) {
            bufCodebook = [g_mps_device newBufferWithBytes:codebook length:16*fp16 options:MTLResourceStorageModeShared];
            g_cq4_codebook_cache[w_key] = bufCodebook;
        }
        if (!bufNorms) {
            size_t norms_bytes = (size_t)N * num_groups * fp16;
            bufNorms = [g_mps_device newBufferWithBytes:norms length:norms_bytes options:MTLResourceStorageModeShared];
            g_cq4_norms_cache[w_key] = bufNorms;
        }

        size_t a_size = (size_t)M * K * fp16;
        if (g_scratch_a == nil || g_scratch_a.length < a_size) {
            g_scratch_a = [g_mps_device newBufferWithLength:a_size options:MTLResourceStorageModeShared];
        }
        memcpy([g_scratch_a contents], hadamard_A, a_size);

        size_t c_size = (size_t)M * N * fp16;
        if (g_scratch_c == nil || g_scratch_c.length < c_size) {
            g_scratch_c = [g_mps_device newBufferWithLength:c_size options:MTLResourceStorageModeShared];
        }

        id<MTLCommandBuffer> cmd = [g_mps_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_cq4_matmul_fused_pso];
        [enc setBuffer:g_scratch_a   offset:0 atIndex:0];
        [enc setBuffer:bufPacked     offset:0 atIndex:1];
        [enc setBuffer:bufCodebook   offset:0 atIndex:2];
        [enc setBuffer:bufNorms      offset:0 atIndex:3];
        [enc setBuffer:g_scratch_c   offset:0 atIndex:4];
        [enc setBytes:&M          length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&N          length:sizeof(uint32_t) atIndex:6];
        [enc setBytes:&K          length:sizeof(uint32_t) atIndex:7];
        [enc setBytes:&group_size length:sizeof(uint32_t) atIndex:8];
        [enc setBytes:&num_groups length:sizeof(uint32_t) atIndex:9];
        [enc setBytes:&pgb        length:sizeof(uint32_t) atIndex:10];

        // Threadgroup: 1024 threads (32 simdgroups). Tile: 64x64.
        MTLSize tg = MTLSizeMake(1024, 1, 1);
        MTLSize grid = MTLSizeMake(((N + 63) / 64) * 1024, (M + 63) / 64, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];

        [cmd commit];
        [cmd waitUntilCompleted];
        memcpy(C, [g_scratch_c contents], c_size);
    }
    return true;
}

// Multi-slot pending writebacks. Each MPS call uses a per-call ephemeral
// A buffer and C buffer, submitted on an alternating queue so multiple
// matmuls can run in parallel on the GPU.
static void cactus_mps_flush_one(int idx) {
    if (!g_pending_list || idx < 0 || (size_t)idx >= g_pending_list->size()) return;
    PendingMps& p = (*g_pending_list)[idx];
    [p.cb waitUntilCompleted];
    if (p.cpu_dest && p.byte_size > 0) {
        memcpy(p.cpu_dest, [p.gpu_result contents], p.byte_size);
    }
    g_pending_list->erase(g_pending_list->begin() + idx);
}

static void cactus_mps_flush_pending() {
    if (!g_pending_list) return;
    while (!g_pending_list->empty()) {
        cactus_mps_flush_one(0);
    }
}

static void cactus_mps_flush_for_dest(void* dest) {
    if (!g_pending_list) return;
    for (int i = (int)g_pending_list->size() - 1; i >= 0; --i) {
        if ((*g_pending_list)[i].cpu_dest == dest) {
            cactus_mps_flush_one(i);
        }
    }
}

extern "C" void cactus_mps_flush(void) { cactus_mps_flush_pending(); }
extern "C" void cactus_mps_flush_for(void* dest) { cactus_mps_flush_for_dest(dest); }

extern "C" void cactus_cq4_matmul_mps_gpu_dequant(
    const void* packed_indices, const void* codebook, const void* norms,
    const __fp16* hadamard_A, __fp16* C,
    uint32_t M, uint32_t K, uint32_t N,
    uint32_t group_size, uint32_t num_groups)
{
    cactus_mps_init();
    if (!g_mps_device || !g_mps_queue || !g_cq4_dequant_pso) return;

    // Sync any prior pending writeback before reusing shared scratch.
    cactus_mps_flush_pending();

    @autoreleasepool {
        const size_t fp16 = sizeof(__fp16);
        const uint32_t pgb = group_size / 2;

        NSValue* w_key = [NSValue valueWithPointer:packed_indices];
        id<MTLBuffer> bufPacked = g_cq4_packed_cache[w_key];
        id<MTLBuffer> bufCodebook = g_cq4_codebook_cache[w_key];
        id<MTLBuffer> bufNorms = g_cq4_norms_cache[w_key];
        if (!bufPacked) {
            size_t packed_bytes = (size_t)N * num_groups * pgb;
            bufPacked = [g_mps_device newBufferWithBytes:packed_indices length:packed_bytes options:MTLResourceStorageModeShared];
            g_cq4_packed_cache[w_key] = bufPacked;
        }
        if (!bufCodebook) {
            bufCodebook = [g_mps_device newBufferWithBytes:codebook length:16*fp16 options:MTLResourceStorageModeShared];
            g_cq4_codebook_cache[w_key] = bufCodebook;
        }
        if (!bufNorms) {
            size_t norms_bytes = (size_t)N * num_groups * fp16;
            bufNorms = [g_mps_device newBufferWithBytes:norms length:norms_bytes options:MTLResourceStorageModeShared];
            g_cq4_norms_cache[w_key] = bufNorms;
        }

        size_t decoded_size = (size_t)N * K * fp16;
        id<MTLBuffer> bufDecoded = g_cq4_decoded_cache[w_key];
        bool needsDequant = (bufDecoded == nil) || bufDecoded.length < decoded_size;
        if (needsDequant) {
            bufDecoded = [g_mps_device newBufferWithLength:decoded_size options:MTLResourceStorageModePrivate];
            g_cq4_decoded_cache[w_key] = bufDecoded;
        }

        size_t a_size = (size_t)M * K * fp16;
        if (g_scratch_a == nil || g_scratch_a.length < a_size) {
            g_scratch_a = [g_mps_device newBufferWithLength:a_size options:MTLResourceStorageModeShared];
        }
        memcpy([g_scratch_a contents], hadamard_A, a_size);

        size_t c_size = (size_t)M * N * fp16;
        if (g_scratch_c == nil || g_scratch_c.length < c_size) {
            g_scratch_c = [g_mps_device newBufferWithLength:c_size options:MTLResourceStorageModeShared];
        }

        id<MTLBuffer> bufA = g_scratch_a;
        id<MTLBuffer> bufC = g_scratch_c;

        id<MTLCommandBuffer> cmd = [g_mps_queue commandBuffer];

        if (needsDequant) {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:g_cq4_dequant_pso];
            [enc setBuffer:bufPacked   offset:0 atIndex:0];
            [enc setBuffer:bufCodebook offset:0 atIndex:1];
            [enc setBuffer:bufNorms    offset:0 atIndex:2];
            [enc setBuffer:bufDecoded  offset:0 atIndex:3];
            [enc setBytes:&K          length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&num_groups length:sizeof(uint32_t) atIndex:5];
            [enc setBytes:&group_size length:sizeof(uint32_t) atIndex:6];
            [enc setBytes:&pgb        length:sizeof(uint32_t) atIndex:7];

            MTLSize gridSize = MTLSizeMake(K / 8, N, 1);
            NSUInteger w = g_cq4_dequant_pso.threadExecutionWidth;
            NSUInteger h = g_cq4_dequant_pso.maxTotalThreadsPerThreadgroup / w;
            if (h == 0) h = 1;
            MTLSize tg = MTLSizeMake(w, h, 1);
            [enc dispatchThreads:gridSize threadsPerThreadgroup:tg];
            [enc endEncoding];
        }

        MPSMatrixDescriptor* dA = mps_get_desc(M, K);
        MPSMatrixDescriptor* dB = mps_get_desc(N, K);
        MPSMatrixDescriptor* dC = mps_get_desc(M, N);
        MPSMatrix* mA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:dA];
        MPSMatrix* mB = [[MPSMatrix alloc] initWithBuffer:bufDecoded descriptor:dB];
        MPSMatrix* mC = [[MPSMatrix alloc] initWithBuffer:bufC descriptor:dC];

        MPSMatrixMultiplication* mm = mps_get_mm(M, K, N);
        [mm encodeToCommandBuffer:cmd leftMatrix:mA rightMatrix:mB resultMatrix:mC];

        [cmd commit];

        PendingMps p;
        p.cb = cmd;
        p.gpu_result = bufC;
        p.cpu_dest = C;
        p.byte_size = c_size;
        if (g_pending_list) g_pending_list->push_back(p);
    }
}

#endif
