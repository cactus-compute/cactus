/* GPUModel — the GPU-resident text decoder.
 *
 * Parallel to `cactus::engine::Model` (the CPU graph executor) and
 * `cactus::npu::ANEEncoder` (the ANE encoder dispatcher). This class is
 * the third backend: full GPU prefill + decode for the text decoder.
 *
 * Lifecycle:
 *   1. Load: opens the bundle's `gpu_plan.json`, mmaps weights.bin /
 *      scales.bin / embedding.bin, wraps them as MTLBuffers (shared mode,
 *      zero copy). Allocates KV cache as PRIVATE MTLBuffers (one per
 *      layer × direction).  Builds Metal pipelines per the plan.
 *   2. Prefill: encode N tokens worth of work into a command buffer,
 *      commit, kick off the next chunk before waiting.
 *   3. Decode: per-token MTLCommandBuffer, pipelined ~4 deep. Sampling
 *      happens on GPU; only the sampled token id crosses to CPU.
 *
 * All weights stay GPU-resident. KV cache never copies back. Sampling on GPU.
 * That's the entire perf story; everything else is plumbing. */
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#if defined(__APPLE__)
#define CACTUS_HAS_GPU 1
#else
#define CACTUS_HAS_GPU 0
#endif

namespace cactus {
namespace gpu {
struct Context;
struct Buffer;
struct Pipeline;
struct CommandBuffer;
}  // namespace gpu

namespace engine {

#if CACTUS_HAS_GPU

class GPUModel {
public:
    GPUModel();
    ~GPUModel();

    GPUModel(const GPUModel&) = delete;
    GPUModel& operator=(const GPUModel&) = delete;

    // Load a GPU bundle from disk. `bundle_dir` is the bundle root (the
    // dir containing components/manifest.json). Reads
    // `components/gpu/gpu_plan.json` and the three `*.bin` files.
    bool load(const std::string& bundle_dir, size_t max_context_size);
    bool is_loaded() const;

    // Run prefill on N input tokens, building up the KV cache. Returns
    // the sampled first response token (top of distribution).
    // Internally pipelines command buffers so the GPU stays busy.
    uint32_t prefill_and_sample(const std::vector<uint32_t>& tokens,
                                float temperature,
                                float top_p,
                                size_t top_k);

    // Generate one token. Reads/writes KV cache in place. The previously
    // sampled token id (or initial position) is passed via input_token.
    uint32_t decode_one(uint32_t input_token,
                        float temperature,
                        float top_p,
                        size_t top_k);

    // Reset the KV cache to length zero. Cheap (no buffer deallocation).
    void reset_cache();

    // Plumbing for telemetry / health.
    size_t kv_cache_length() const;
    size_t kv_cache_capacity() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

#else  // CACTUS_HAS_GPU

// Non-Apple stub: every call fails / returns 0.
class GPUModel {
public:
    GPUModel() = default;
    ~GPUModel() = default;
    bool load(const std::string&, size_t) { return false; }
    bool is_loaded() const { return false; }
    uint32_t prefill_and_sample(const std::vector<uint32_t>&, float, float, size_t) { return 0; }
    uint32_t decode_one(uint32_t, float, float, size_t) { return 0; }
    void reset_cache() {}
    size_t kv_cache_length() const { return 0; }
    size_t kv_cache_capacity() const { return 0; }
};

#endif  // CACTUS_HAS_GPU

}  // namespace engine
}  // namespace cactus
