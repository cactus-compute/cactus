/* Cactus GPU command buffer + dispatch encoding. */
#import <Metal/Metal.h>
#include "cactus_gpu.h"
#include "internal.h"

#include <cstdio>

namespace cactus {
namespace gpu {

#if !CACTUS_HAS_METAL

CommandBuffer* command_buffer_begin(Context*) { return nullptr; }
void command_buffer_dispatch(CommandBuffer*, Pipeline*, const BufferBinding*, size_t,
                             uint32_t, uint32_t, uint32_t,
                             uint32_t, uint32_t, uint32_t) {}
void command_buffer_barrier(CommandBuffer*) {}
void command_buffer_commit(CommandBuffer*) {}
void command_buffer_wait(CommandBuffer*) {}
bool command_buffer_is_complete(const CommandBuffer*) { return true; }

#else

CommandBuffer* command_buffer_begin(Context* ctx) {
    if (!ctx) return nullptr;
    @autoreleasepool {
        id<MTLCommandQueue> q = _internal_queue(ctx);
        // commandBufferWithUnretainedReferences = MLX trick. We track the
        // resource lifetimes ourselves and skip Metal's tracking overhead.
        id<MTLCommandBuffer> cb = [q commandBufferWithUnretainedReferences];
        if (!cb) return nullptr;
        CommandBuffer* out = new CommandBuffer();
        out->mtl_buffer = cb;
        out->encoder = nil;
        out->encoder_open = false;
        return out;
    }
}

static id<MTLComputeCommandEncoder> _ensure_encoder(CommandBuffer* cb) {
    if (!cb->encoder_open) {
        cb->encoder = [cb->mtl_buffer computeCommandEncoder];
        cb->encoder_open = true;
    }
    return cb->encoder;
}

void command_buffer_dispatch(CommandBuffer* cb,
                             Pipeline* p,
                             const BufferBinding* buffers,
                             size_t num_buffers,
                             uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                             uint32_t tg_x,   uint32_t tg_y,   uint32_t tg_z) {
    if (!cb || !p) return;
    id<MTLComputeCommandEncoder> enc = _ensure_encoder(cb);
    [enc setComputePipelineState:p->pipeline_state];
    for (size_t i = 0; i < num_buffers; ++i) {
        const BufferBinding& bb = buffers[i];
        if (bb.buffer && bb.buffer->mtl_buffer) {
            [enc setBuffer:bb.buffer->mtl_buffer
                    offset:bb.offset
                   atIndex:i];
        }
    }
    // Caller passes `grid` as a count of THREADGROUPS (more intuitive for
    // kernels that use `tgid` to index into output tiles). We translate to
    // Metal's `dispatchThreadgroups:` directly. If a kernel needs Apple's
    // non-uniform dispatch (kernel handles the remainder lanes), we can
    // expose a separate `dispatch_threads_total` entry point.
    MTLSize tg_count = MTLSizeMake(grid_x, grid_y, grid_z);
    MTLSize tg_size  = MTLSizeMake(tg_x,   tg_y,   tg_z);
    [enc dispatchThreadgroups:tg_count threadsPerThreadgroup:tg_size];
}

void command_buffer_barrier(CommandBuffer* cb) {
    if (!cb || !cb->encoder_open) return;
    [cb->encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

void command_buffer_commit(CommandBuffer* cb) {
    if (!cb) return;
    if (cb->encoder_open) {
        [cb->encoder endEncoding];
        cb->encoder = nil;
        cb->encoder_open = false;
    }
    [cb->mtl_buffer commit];
}

void command_buffer_wait(CommandBuffer* cb) {
    if (!cb) return;
    [cb->mtl_buffer waitUntilCompleted];
    // Don't destroy here — caller owns the lifetime. They'll delete it.
}

bool command_buffer_is_complete(const CommandBuffer* cb) {
    if (!cb) return true;
    MTLCommandBufferStatus s = [cb->mtl_buffer status];
    return s == MTLCommandBufferStatusCompleted || s == MTLCommandBufferStatusError;
}

#endif

} // namespace gpu
} // namespace cactus
