#ifdef __APPLE__

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "kernel.h"
#include <cstring>

static NSString* const kCactusMSL = @R"(
#include <metal_stdlib>
using namespace metal;

kernel void cactus_dequant_int4(
    device const uchar* B_packed [[buffer(0)]],
    device const half* B_scales [[buffer(1)]],
    device half* B_out [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    constant uint& group_size [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint k = gid.x;
    uint n = gid.y;
    uint num_groups = K / group_size;
    uint n_block = n >> 2;
    uint c = n & 3;
    uint g = k / group_size;
    uint k_local = k - g * group_size;
    uint k_super = k_local >> 3;
    uint k_in_slab = k_local & 7;
    uint byte_in_group = k_super * 16 + c * 4 + (k_in_slab & 3);
    uint byte_offset = (n_block * K + g * group_size) * 2 + byte_in_group;
    uchar b = B_packed[byte_offset];
    int nibble = (k_in_slab < 4) ? int(b & 0xF) : int(b >> 4);
    if (nibble >= 8) nibble -= 16;
    half scale = B_scales[(n_block * num_groups + g) * 4 + c];
    B_out[n * K + k] = half(nibble) * scale;
}
)";

static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;
static id<MTLComputePipelineState> g_dequant_pso = nil;
static dispatch_once_t g_once;

static void cactus_mps_init() {
    dispatch_once(&g_once, ^{
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) return;
        g_queue = [g_device newCommandQueue];
        NSError* err = nil;
        id<MTLLibrary> lib = [g_device newLibraryWithSource:kCactusMSL options:nil error:&err];
        if (!lib) return;
        id<MTLFunction> fn = [lib newFunctionWithName:@"cactus_dequant_int4"];
        if (!fn) return;
        g_dequant_pso = [g_device newComputePipelineStateWithFunction:fn error:&err];
    });
}

bool cactus_mps_available() {
    cactus_mps_init();
    return g_device != nil && g_queue != nil;
}

void cactus_matmul_f16_mps(const __fp16* A, const __fp16* B_T, __fp16* C,
                           size_t M, size_t K, size_t N) {
    cactus_mps_init();
    if (!g_device || !g_queue) return;

    @autoreleasepool {
        const size_t fp16 = sizeof(__fp16);
        id<MTLBuffer> bufA = [g_device newBufferWithBytes:A length:M*K*fp16 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufB = [g_device newBufferWithBytes:B_T length:N*K*fp16 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufC = [g_device newBufferWithLength:M*N*fp16 options:MTLResourceStorageModeShared];

        MPSMatrixDescriptor* dA = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:K rowBytes:K*fp16 dataType:MPSDataTypeFloat16];
        MPSMatrixDescriptor* dB = [MPSMatrixDescriptor matrixDescriptorWithRows:N columns:K rowBytes:K*fp16 dataType:MPSDataTypeFloat16];
        MPSMatrixDescriptor* dC = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N rowBytes:N*fp16 dataType:MPSDataTypeFloat16];

        MPSMatrix* mA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:dA];
        MPSMatrix* mB = [[MPSMatrix alloc] initWithBuffer:bufB descriptor:dB];
        MPSMatrix* mC = [[MPSMatrix alloc] initWithBuffer:bufC descriptor:dC];

        MPSMatrixMultiplication* mm = [[MPSMatrixMultiplication alloc] initWithDevice:g_device
            transposeLeft:NO transposeRight:YES resultRows:M resultColumns:N interiorColumns:K alpha:1.0 beta:0.0];

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        [mm encodeToCommandBuffer:cmd leftMatrix:mA rightMatrix:mB resultMatrix:mC];
        [cmd commit];
        [cmd waitUntilCompleted];

        memcpy(C, [bufC contents], M*N*fp16);
    }
}

void cactus_matmul_int4_mps(const __fp16* A, const int8_t* B_packed, const __fp16* B_scales,
                            __fp16* C, size_t M, size_t K, size_t N, size_t group_size) {
    cactus_mps_init();
    if (!g_device || !g_queue || !g_dequant_pso) return;
    if (N % 4 != 0 || K % group_size != 0) return;

    @autoreleasepool {
        const size_t fp16 = sizeof(__fp16);
        const size_t packed_bytes = (N / 4) * K * 2;
        const size_t num_groups = K / group_size;
        const size_t scales_bytes = (N / 4) * num_groups * 4 * fp16;

        id<MTLBuffer> bufA = [g_device newBufferWithBytes:A length:M*K*fp16 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufBp = [g_device newBufferWithBytes:B_packed length:packed_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufBs = [g_device newBufferWithBytes:B_scales length:scales_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufBd = [g_device newBufferWithLength:N*K*fp16 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufC = [g_device newBufferWithLength:M*N*fp16 options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];

        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_dequant_pso];
        [enc setBuffer:bufBp offset:0 atIndex:0];
        [enc setBuffer:bufBs offset:0 atIndex:1];
        [enc setBuffer:bufBd offset:0 atIndex:2];
        uint32_t Ku = (uint32_t)K;
        uint32_t Gu = (uint32_t)group_size;
        [enc setBytes:&Ku length:sizeof(Ku) atIndex:3];
        [enc setBytes:&Gu length:sizeof(Gu) atIndex:4];
        [enc dispatchThreads:MTLSizeMake(K, N, 1) threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
        [enc endEncoding];

        MPSMatrixDescriptor* dA = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:K rowBytes:K*fp16 dataType:MPSDataTypeFloat16];
        MPSMatrixDescriptor* dB = [MPSMatrixDescriptor matrixDescriptorWithRows:N columns:K rowBytes:K*fp16 dataType:MPSDataTypeFloat16];
        MPSMatrixDescriptor* dC = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N rowBytes:N*fp16 dataType:MPSDataTypeFloat16];
        MPSMatrix* mA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:dA];
        MPSMatrix* mB = [[MPSMatrix alloc] initWithBuffer:bufBd descriptor:dB];
        MPSMatrix* mC = [[MPSMatrix alloc] initWithBuffer:bufC descriptor:dC];
        MPSMatrixMultiplication* mm = [[MPSMatrixMultiplication alloc] initWithDevice:g_device
            transposeLeft:NO transposeRight:YES resultRows:M resultColumns:N interiorColumns:K alpha:1.0 beta:0.0];
        [mm encodeToCommandBuffer:cmd leftMatrix:mA rightMatrix:mB resultMatrix:mC];

        [cmd commit];
        [cmd waitUntilCompleted];
        memcpy(C, [bufC contents], M*N*fp16);
    }
}

#endif
