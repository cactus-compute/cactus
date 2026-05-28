/* GPUModel implementation skeleton.
 *
 * Status (M2 deliverable): plan parser + pipeline cache + buffer
 * allocation. The actual forward pass that walks the plan and dispatches
 * kernels is M3. This file documents the API surface and shows how the
 * cactus_gpu.h dispatch wrappers plug into the runtime.
 *
 * The dependency direction:
 *   gpu_model.mm  →  cactus_gpu.h  →  Metal.framework
 * Everything in cactus_gpu.h is C-callable and Metal-free at the type
 * layer, so this file is the *only* place in cactus-engine that touches
 * Metal headers (and even here only transitively). */
#import <Foundation/Foundation.h>
#include "gpu_model.h"
#include "../../../cactus-kernels-gpu/include/cactus_gpu.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace cactus {
namespace engine {

#if CACTUS_HAS_GPU

namespace {

// Minimal mmapped file helper — used to wrap weights.bin / scales.bin as
// MTLBuffers via cactus::gpu::buffer_wrap_host_memory.
struct MMapped {
    void*  ptr  = nullptr;
    size_t size = 0;
    int    fd   = -1;

    bool open(const std::string& path) {
        struct stat st;
        if (::stat(path.c_str(), &st) != 0) return false;
        size = (size_t)st.st_size;
        if (size == 0) return false;
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) return false;
        ptr = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (ptr == MAP_FAILED) { ::close(fd); fd = -1; ptr = nullptr; return false; }
        return true;
    }
    void close() {
        if (ptr) { ::munmap(ptr, size); ptr = nullptr; size = 0; }
        if (fd >= 0) { ::close(fd); fd = -1; }
    }
    ~MMapped() { close(); }
};

}  // namespace

struct GPUModel::Impl {
    cactus::gpu::Context* ctx = nullptr;
    MMapped weights, scales, embedding;
    cactus::gpu::Buffer* weights_buf = nullptr;
    cactus::gpu::Buffer* scales_buf  = nullptr;
    cactus::gpu::Buffer* embed_buf   = nullptr;

    // Pipelines keyed by op kind + shape tuple, built once per (K,N,head_dim,...).
    // M3 will populate this from the parsed plan.
    std::map<std::string, cactus::gpu::Pipeline*> pipelines;

    // KV cache: one MTLBuffer per layer × {K,V}. Sized at load() time.
    std::vector<cactus::gpu::Buffer*> kv_k;
    std::vector<cactus::gpu::Buffer*> kv_v;
    size_t kv_max_seq = 0;
    size_t kv_cur_len = 0;

    // The parsed plan (held in a simple string-keyed dict — full parsing
    // lands at M3).
    std::string plan_json;

    bool loaded = false;
};

GPUModel::GPUModel() : impl_(std::make_unique<Impl>()) {}
GPUModel::~GPUModel() {
    if (!impl_) return;
    for (auto& kv : impl_->pipelines) cactus::gpu::pipeline_destroy(kv.second);
    for (auto* b : impl_->kv_k) cactus::gpu::buffer_destroy(b);
    for (auto* b : impl_->kv_v) cactus::gpu::buffer_destroy(b);
    cactus::gpu::buffer_destroy(impl_->weights_buf);
    cactus::gpu::buffer_destroy(impl_->scales_buf);
    cactus::gpu::buffer_destroy(impl_->embed_buf);
    cactus::gpu::context_destroy(impl_->ctx);
}

bool GPUModel::load(const std::string& bundle_dir, size_t max_context_size) {
    @autoreleasepool {
        // Find the metallib path. We expect it shipped alongside the binary
        // (CMake install rule) or in a known relative location.
        // M3: make this configurable via an env var.
        NSString* exe_dir = [[NSBundle mainBundle] resourcePath];
        if (!exe_dir) exe_dir = @"";
        NSString* candidate = [exe_dir stringByAppendingPathComponent:@"cactus_kernels.metallib"];
        std::string metallib_path = [candidate UTF8String];
        // Fallbacks for dev builds (not installed):
        if (access(metallib_path.c_str(), R_OK) != 0) {
            const char* env_path = std::getenv("CACTUS_GPU_METALLIB");
            if (env_path) metallib_path = env_path;
        }

        impl_->ctx = cactus::gpu::context_create(metallib_path.c_str());
        if (!impl_->ctx) {
            std::fprintf(stderr, "GPUModel: failed to create gpu context (metallib=%s)\n",
                         metallib_path.c_str());
            return false;
        }

        // Read gpu_plan.json into memory.
        std::string plan_path = bundle_dir + "/components/gpu/gpu_plan.json";
        std::ifstream in(plan_path);
        if (!in) {
            std::fprintf(stderr, "GPUModel: cannot read %s\n", plan_path.c_str());
            return false;
        }
        std::stringstream ss; ss << in.rdbuf();
        impl_->plan_json = ss.str();

        // Mmap weights, scales, embedding.
        const std::string weights_path = bundle_dir + "/components/gpu/weights.bin";
        const std::string scales_path  = bundle_dir + "/components/gpu/scales.bin";
        const std::string embed_path   = bundle_dir + "/components/gpu/embedding.bin";
        if (!impl_->weights.open(weights_path)) {
            std::fprintf(stderr, "GPUModel: cannot mmap %s\n", weights_path.c_str());
            return false;
        }
        impl_->scales.open(scales_path);     // optional (fp16 paths have no scales)
        impl_->embedding.open(embed_path);   // optional (some bundles tie weights)

        impl_->weights_buf = cactus::gpu::buffer_wrap_host_memory(
            impl_->ctx, impl_->weights.ptr, impl_->weights.size);
        if (impl_->scales.ptr) {
            impl_->scales_buf = cactus::gpu::buffer_wrap_host_memory(
                impl_->ctx, impl_->scales.ptr, impl_->scales.size);
        }
        if (impl_->embedding.ptr) {
            impl_->embed_buf = cactus::gpu::buffer_wrap_host_memory(
                impl_->ctx, impl_->embedding.ptr, impl_->embedding.size);
        }

        // KV cache allocation — M3 will size these per the plan's num_layers
        // and head_dim. For now we just record max_context_size.
        impl_->kv_max_seq = max_context_size;
        impl_->kv_cur_len = 0;

        impl_->loaded = true;
        return true;
    }
}

bool GPUModel::is_loaded() const { return impl_ && impl_->loaded; }

uint32_t GPUModel::prefill_and_sample(const std::vector<uint32_t>& tokens,
                                       float temperature, float top_p, size_t top_k) {
    // M3: walk the plan, encode per-layer dispatches into one or more
    // command buffers, commit and pipeline. For now this is a stub that
    // signals "not implemented" by returning the BOS token (id 0).
    (void)tokens; (void)temperature; (void)top_p; (void)top_k;
    std::fprintf(stderr, "GPUModel::prefill_and_sample not implemented (M3 deliverable)\n");
    return 0;
}

uint32_t GPUModel::decode_one(uint32_t input_token,
                               float temperature, float top_p, size_t top_k) {
    (void)input_token; (void)temperature; (void)top_p; (void)top_k;
    std::fprintf(stderr, "GPUModel::decode_one not implemented (M3 deliverable)\n");
    return 0;
}

void GPUModel::reset_cache() {
    if (impl_) impl_->kv_cur_len = 0;
}

size_t GPUModel::kv_cache_length()   const { return impl_ ? impl_->kv_cur_len  : 0; }
size_t GPUModel::kv_cache_capacity() const { return impl_ ? impl_->kv_max_seq  : 0; }

#endif  // CACTUS_HAS_GPU

}  // namespace engine
}  // namespace cactus
