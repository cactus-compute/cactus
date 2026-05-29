/* Cactus GPU buffer wrappers — MTLBuffer create / wrap / destroy. */
#import <Metal/Metal.h>
#include "cactus_gpu.h"
#include "internal.h"

#include <cstdio>

namespace cactus {
namespace gpu {

#if !CACTUS_HAS_METAL

Buffer* buffer_wrap_host_memory(Context*, void*, size_t) { return nullptr; }
Buffer* buffer_create(Context*, size_t, StorageMode) { return nullptr; }
void*   buffer_contents(Buffer*) { return nullptr; }
size_t  buffer_size(const Buffer*) { return 0; }
void    buffer_destroy(Buffer*) {}

#else

Buffer* buffer_wrap_host_memory(Context* ctx, void* host_ptr, size_t size) {
    if (!ctx || !host_ptr || size == 0) return nullptr;
    @autoreleasepool {
        id<MTLDevice> dev = _internal_device(ctx);
        // No-copy wrap. We do not assume ownership — caller keeps the memory alive.
        id<MTLBuffer> b = [dev newBufferWithBytesNoCopy:host_ptr
                                                  length:size
                                                 options:MTLResourceStorageModeShared
                                             deallocator:^(void*, NSUInteger){}];
        if (!b) {
            std::fprintf(stderr, "cactus_gpu: newBufferWithBytesNoCopy(%zu) returned nil\n", size);
            return nullptr;
        }
        Buffer* out = new Buffer();
        out->mtl_buffer = b;
        out->size_bytes = size;
        out->mode = StorageMode::SHARED;
        return out;
    }
}

Buffer* buffer_create(Context* ctx, size_t size, StorageMode mode) {
    if (!ctx || size == 0) return nullptr;
    @autoreleasepool {
        id<MTLDevice> dev = _internal_device(ctx);
        MTLResourceOptions opts = (mode == StorageMode::PRIVATE)
                                    ? MTLResourceStorageModePrivate
                                    : MTLResourceStorageModeShared;
        opts |= MTLResourceHazardTrackingModeUntracked;
        id<MTLBuffer> b = [dev newBufferWithLength:size options:opts];
        if (!b) {
            std::fprintf(stderr, "cactus_gpu: newBufferWithLength(%zu) returned nil\n", size);
            return nullptr;
        }
        Buffer* out = new Buffer();
        out->mtl_buffer = b;
        out->size_bytes = size;
        out->mode = mode;
        return out;
    }
}

void* buffer_contents(Buffer* buf) {
    if (!buf || buf->mode == StorageMode::PRIVATE) return nullptr;
    return [buf->mtl_buffer contents];
}

size_t buffer_size(const Buffer* buf) { return buf ? buf->size_bytes : 0; }

void buffer_destroy(Buffer* buf) {
    if (!buf) return;
    @autoreleasepool { buf->mtl_buffer = nil; }
    delete buf;
}

#endif

} // namespace gpu
} // namespace cactus
