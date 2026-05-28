/* Cactus GPU context implementation — Metal device + command queue +
 * compiled kernel library. */
#import <Metal/Metal.h>
#include "cactus_gpu.h"

#include <cstdio>
#include <cstring>
#include <string>

namespace cactus {
namespace gpu {

#if !CACTUS_HAS_METAL

// Non-Apple stubs.
Context* context_create(const char*) { return nullptr; }
void context_destroy(Context*) {}
bool context_has_metal(const Context*) { return false; }
uint64_t context_recommended_max_working_set(const Context*) { return 0; }

#else

struct Context {
    id<MTLDevice>       device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary>      library;
    // Heap reserved for activation scratch (~50 MB by default; resizable).
    id<MTLHeap>         scratch_heap;
    uint64_t            scratch_heap_size;
};

Context* context_create(const char* metallib_path) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            std::fprintf(stderr, "cactus_gpu: no Metal device available\n");
            return nullptr;
        }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        if (!q) {
            std::fprintf(stderr, "cactus_gpu: failed to create MTLCommandQueue\n");
            return nullptr;
        }

        // Library: caller can pass nullptr or empty to use the default device library.
        // If a path is given we try to load it as a precompiled .metallib.
        id<MTLLibrary> lib = nil;
        NSError* error = nil;
        if (metallib_path && metallib_path[0]) {
            NSString* path = [NSString stringWithUTF8String:metallib_path];
            NSURL* url = [NSURL fileURLWithPath:path];
            lib = [dev newLibraryWithURL:url error:&error];
            if (!lib) {
                std::fprintf(stderr, "cactus_gpu: newLibraryWithURL(%s) failed: %s\n",
                             metallib_path,
                             error ? [[error localizedDescription] UTF8String] : "?");
                return nullptr;
            }
        } else {
            lib = [dev newDefaultLibrary];
            if (!lib) {
                std::fprintf(stderr, "cactus_gpu: newDefaultLibrary() returned nil\n");
                return nullptr;
            }
        }

        // Scratch heap for activations. 64 MB default — sized for largest model
        // expected (Gemma 4 E2B prefill chunk of 128 tokens × hidden 4096 × fp16
        // = 1 MB per layer × 24 layers ≈ 25 MB worst case).
        MTLHeapDescriptor* hd = [[MTLHeapDescriptor alloc] init];
        hd.size = 64ULL * 1024 * 1024;
        hd.storageMode = MTLStorageModePrivate;
        hd.hazardTrackingMode = MTLHazardTrackingModeUntracked;
        id<MTLHeap> heap = [dev newHeapWithDescriptor:hd];

        Context* ctx = new Context();
        ctx->device = dev;
        ctx->queue  = q;
        ctx->library = lib;
        ctx->scratch_heap = heap;
        ctx->scratch_heap_size = hd.size;
        return ctx;
    }
}

void context_destroy(Context* ctx) {
    if (!ctx) return;
    @autoreleasepool {
        ctx->device = nil;
        ctx->queue = nil;
        ctx->library = nil;
        ctx->scratch_heap = nil;
    }
    delete ctx;
}

bool context_has_metal(const Context* ctx) { return ctx != nullptr; }

uint64_t context_recommended_max_working_set(const Context* ctx) {
    if (!ctx) return 0;
    return [ctx->device recommendedMaxWorkingSetSize];
}

// Library accessor for sibling translation units (pipeline.mm, etc.).
id<MTLDevice>       _internal_device(Context* ctx)  { return ctx ? ctx->device  : nil; }
id<MTLCommandQueue> _internal_queue(Context* ctx)   { return ctx ? ctx->queue   : nil; }
id<MTLLibrary>      _internal_library(Context* ctx) { return ctx ? ctx->library : nil; }

#endif  // CACTUS_HAS_METAL

} // namespace gpu
} // namespace cactus
