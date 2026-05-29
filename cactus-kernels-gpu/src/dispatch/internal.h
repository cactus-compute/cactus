/* Internal accessors shared between cactus-kernels-gpu translation units.
 * NOT exported in the public header. Apple-only — guard everything with
 * CACTUS_HAS_METAL. */
#ifndef CACTUS_GPU_INTERNAL_H
#define CACTUS_GPU_INTERNAL_H

#include "cactus_gpu.h"

#if CACTUS_HAS_METAL

#import <Metal/Metal.h>

namespace cactus {
namespace gpu {

// Forward-declared opaque struct definitions visible to the implementation.
struct Buffer {
    id<MTLBuffer> mtl_buffer;
    size_t        size_bytes;
    StorageMode   mode;
};

struct Pipeline {
    id<MTLComputePipelineState> pipeline_state;
    id<MTLFunction>             function;
    // Recommended threadgroup size for this pipeline (computed at creation).
    uint32_t                    tg_threads;
    uint32_t                    simdgroup_threads;
};

struct CommandBuffer {
    id<MTLCommandBuffer>          mtl_buffer;
    id<MTLComputeCommandEncoder>  encoder;
    bool                          encoder_open;
};

// Context internals accessed from sibling .mm files.
id<MTLDevice>       _internal_device(Context* ctx);
id<MTLCommandQueue> _internal_queue(Context* ctx);
id<MTLLibrary>      _internal_library(Context* ctx);

} // namespace gpu
} // namespace cactus

#endif  // CACTUS_HAS_METAL
#endif  // CACTUS_GPU_INTERNAL_H
