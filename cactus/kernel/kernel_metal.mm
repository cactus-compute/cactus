#ifdef __APPLE__
#ifdef CACTUS_USE_MPS

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <mutex>
#include <cstdint>
#include <cstddef>

class CactusMetalDevice {
public:
    static CactusMetalDevice& instance() {
        static CactusMetalDevice dev;
        return dev;
    }

    bool available() const { return device_ != nil; }
    id<MTLDevice> device() const { return device_; }
    id<MTLCommandQueue> queue() const { return queue_; }

private:
    CactusMetalDevice() {
        device_ = MTLCreateSystemDefaultDevice();
        if (device_) {
            queue_ = [device_ newCommandQueue];
        }
    }

    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> queue_ = nil;
};


bool cactus_metal_available() {
    return CactusMetalDevice::instance().available();
}

void cactus_matmul_f16_mps(const __fp16* a, const __fp16* b_transposed, __fp16* c,
                           size_t M, size_t K, size_t N) {
    auto& metal = CactusMetalDevice::instance();
    id<MTLDevice> device = metal.device();
    id<MTLCommandQueue> queue = metal.queue();

    const size_t a_bytes = M * K * sizeof(uint16_t);
    const size_t b_bytes = N * K * sizeof(uint16_t);
    const size_t c_bytes = M * N * sizeof(uint16_t);

    const size_t page_size = getpagesize();

    auto make_buffer = [&](void* ptr, size_t bytes, MTLResourceOptions opts) -> id<MTLBuffer> {
        uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
        size_t aligned_bytes = (bytes + page_size - 1) & ~(page_size - 1);
        if ((addr % page_size) == 0) {
            id<MTLBuffer> buf = [device newBufferWithBytesNoCopy:ptr
                                                         length:aligned_bytes
                                                        options:opts
                                                    deallocator:nil];
            if (buf) return buf;
        }
        return [device newBufferWithBytes:ptr length:bytes options:opts];
    };

    id<MTLBuffer> bufA = make_buffer((void*)a, a_bytes, MTLResourceStorageModeShared);
    id<MTLBuffer> bufB = make_buffer((void*)b_transposed, b_bytes, MTLResourceStorageModeShared);

    bool c_is_zero_copy = false;
    id<MTLBuffer> bufC;
    {
        uintptr_t addr = reinterpret_cast<uintptr_t>(c);
        size_t aligned_bytes = (c_bytes + page_size - 1) & ~(page_size - 1);
        if ((addr % page_size) == 0) {
            bufC = [device newBufferWithBytesNoCopy:(void*)c
                                            length:aligned_bytes
                                           options:MTLResourceStorageModeShared
                                       deallocator:nil];
            if (bufC) c_is_zero_copy = true;
        }
        if (!bufC) {
            bufC = [device newBufferWithLength:c_bytes options:MTLResourceStorageModeShared];
        }
    }

    MPSMatrixDescriptor* descA = [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                                                       columns:K
                                                                      rowBytes:K * sizeof(uint16_t)
                                                                      dataType:MPSDataTypeFloat16];

    MPSMatrixDescriptor* descB = [MPSMatrixDescriptor matrixDescriptorWithRows:N
                                                                       columns:K
                                                                      rowBytes:K * sizeof(uint16_t)
                                                                      dataType:MPSDataTypeFloat16];

    MPSMatrixDescriptor* descC = [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                                                       columns:N
                                                                      rowBytes:N * sizeof(uint16_t)
                                                                      dataType:MPSDataTypeFloat16];

    MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:descA];
    MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:bufB descriptor:descB];
    MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:bufC descriptor:descC];

    MPSMatrixMultiplication* gemm = [[MPSMatrixMultiplication alloc]
        initWithDevice:device
         transposeLeft:NO
        transposeRight:YES
            resultRows:M
         resultColumns:N
       interiorColumns:K
                 alpha:1.0
                  beta:0.0];

    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    [gemm encodeToCommandBuffer:cmdBuf leftMatrix:matA rightMatrix:matB resultMatrix:matC];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    if (!c_is_zero_copy) {
        memcpy(c, [bufC contents], c_bytes);
    }
}

#endif // CACTUS_USE_MPS
#endif // __APPLE__
