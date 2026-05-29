/* Smoke test: load the Gemma 4 GPU bundle, run decode_one(BOS).
 *
 * Validates that the entire M3 forward pass infrastructure executes
 * without crashing on a real model. Numerical correctness is a M3+
 * deliverable (needs Gemma 4 sliding-attention KV-share handling +
 * verified against HF reference). */
#include <cstdio>
#include <cstdlib>
#include <string>
#include "../src/gpu/gpu_model.h"

int main(int argc, char** argv) {
    using namespace cactus::engine;

    std::string bundle = (argc > 1) ? argv[1] : "weights/gemma-4-e2b-it";
    std::printf("test_gpu_decode: loading %s\n", bundle.c_str());

    GPUModel m;
    if (!m.load(bundle, 2048)) {
        std::fprintf(stderr, "GPUModel::load failed\n");
        return 1;
    }
    std::printf("loaded OK; KV cache cap = %zu\n", m.kv_cache_capacity());

    // Run BOS (Gemma 4 BOS = 2).
    std::printf("running decode_one(2)...\n");
    uint32_t out = m.decode_one(2, 0.0f, 1.0f, 1);
    std::printf("decode_one returned token id %u  (KV cache len = %zu)\n",
                out, m.kv_cache_length());

    // A second token to confirm the KV cache advances.
    std::printf("running decode_one(%u)...\n", out);
    uint32_t out2 = m.decode_one(out, 0.0f, 1.0f, 1);
    std::printf("decode_one returned token id %u  (KV cache len = %zu)\n",
                out2, m.kv_cache_length());

    return 0;
}
