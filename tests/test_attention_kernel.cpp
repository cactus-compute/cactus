// Test the attention kernel against a naive FP32 reference.
// Runs with various seq lengths and head dims to find discrepancies.
//
// Build: from cactus root, after `cactus build`:
//   clang++ -std=c++20 -O2 -march=armv8.2-a+fp16+simd+dotprod+i8mm \
//     tests/test_attention_kernel.cpp -I cactus -L cactus/build -lcactus -o tests/build/test_attn
// Run:
//   tests/build/test_attn

#include "kernel/kernel.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>

static float cos_sim(const float* a, const float* b, size_t n) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; i++) {
        dot += (double)a[i] * (double)b[i];
        na += (double)a[i] * (double)a[i];
        nb += (double)b[i] * (double)b[i];
    }
    return (float)(dot / (sqrt(na) * sqrt(nb) + 1e-12));
}

// Naive FP32 reference attention
static void reference_attention_f32(
    const float* Q, const float* K, const float* V, float* O,
    size_t batch, size_t seq_len, size_t kv_seq_len,
    size_t num_q_heads, size_t num_kv_heads, size_t head_dim,
    float scale, size_t position_offset, size_t window_size, bool is_causal
) {
    size_t group_size = num_q_heads / num_kv_heads;

    for (size_t b = 0; b < batch; b++) {
        for (size_t qh = 0; qh < num_q_heads; qh++) {
            size_t kvh = qh / group_size;
            for (size_t qi = 0; qi < seq_len; qi++) {
                size_t abs_q = position_offset + qi;

                // Compute scores
                std::vector<float> scores(kv_seq_len, -1e30f);
                float max_score = -1e30f;

                for (size_t ki = 0; ki < kv_seq_len; ki++) {
                    // Causal mask
                    if (is_causal && ki > abs_q) continue;
                    // Sliding window
                    if (window_size > 0 && ki < abs_q && (abs_q - ki) > window_size) continue;

                    float dot = 0;
                    const float* q_vec = Q + (b * seq_len * num_q_heads * head_dim) +
                                         (qi * num_q_heads * head_dim) + (qh * head_dim);
                    const float* k_vec = K + (b * kv_seq_len * num_kv_heads * head_dim) +
                                         (ki * num_kv_heads * head_dim) + (kvh * head_dim);

                    for (size_t d = 0; d < head_dim; d++)
                        dot += q_vec[d] * k_vec[d];

                    scores[ki] = dot * scale;
                    max_score = std::max(max_score, scores[ki]);
                }

                // Softmax
                float sum_exp = 0;
                for (size_t ki = 0; ki < kv_seq_len; ki++) {
                    if (scores[ki] > -1e20f) {
                        scores[ki] = expf(scores[ki] - max_score);
                        sum_exp += scores[ki];
                    } else {
                        scores[ki] = 0;
                    }
                }
                for (size_t ki = 0; ki < kv_seq_len; ki++)
                    scores[ki] /= sum_exp;

                // Weighted sum of V
                float* o_vec = O + (b * seq_len * num_q_heads * head_dim) +
                               (qi * num_q_heads * head_dim) + (qh * head_dim);
                for (size_t d = 0; d < head_dim; d++) {
                    float acc = 0;
                    for (size_t ki = 0; ki < kv_seq_len; ki++) {
                        const float* v_vec = V + (b * kv_seq_len * num_kv_heads * head_dim) +
                                             (ki * num_kv_heads * head_dim) + (kvh * head_dim);
                        acc += scores[ki] * v_vec[d];
                    }
                    o_vec[d] = acc;
                }
            }
        }
    }
}

struct TestConfig {
    size_t batch, seq_len, num_q_heads, num_kv_heads, head_dim;
    float scale;
    size_t position_offset, window_size;
    const char* name;
};

static void run_test(const TestConfig& cfg) {
    size_t q_size = cfg.batch * cfg.seq_len * cfg.num_q_heads * cfg.head_dim;
    size_t kv_size = cfg.batch * cfg.seq_len * cfg.num_kv_heads * cfg.head_dim;
    size_t o_size = q_size;

    // Generate random data
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // Use RMS-normed values (as Gemma 3n does) to be realistic
    auto make_normed = [&](size_t total, size_t num_rows, size_t dim) {
        std::vector<float> data(total);
        for (size_t i = 0; i < total; i++)
            data[i] = dist(rng);
        // Normalize each row
        for (size_t r = 0; r < num_rows; r++) {
            float ss = 0;
            for (size_t d = 0; d < dim; d++)
                ss += data[r * dim + d] * data[r * dim + d];
            float rms = sqrtf(ss / dim);
            for (size_t d = 0; d < dim; d++)
                data[r * dim + d] /= rms;
        }
        return data;
    };

    auto Q_f32 = make_normed(q_size, cfg.batch * cfg.seq_len * cfg.num_q_heads, cfg.head_dim);
    auto K_f32 = make_normed(kv_size, cfg.batch * cfg.seq_len * cfg.num_kv_heads, cfg.head_dim);
    auto V_f32 = make_normed(kv_size, cfg.batch * cfg.seq_len * cfg.num_kv_heads, cfg.head_dim);

    // Convert to FP16
    std::vector<__fp16> Q_fp16(q_size), K_fp16(kv_size), V_fp16(kv_size), O_fp16(o_size);
    for (size_t i = 0; i < q_size; i++) Q_fp16[i] = (__fp16)Q_f32[i];
    for (size_t i = 0; i < kv_size; i++) K_fp16[i] = (__fp16)K_f32[i];
    for (size_t i = 0; i < kv_size; i++) V_fp16[i] = (__fp16)V_f32[i];

    // Also convert Q/K/V back to f32 (to match the precision that cactus_attention sees)
    std::vector<float> Q_f32_from_fp16(q_size), K_f32_from_fp16(kv_size), V_f32_from_fp16(kv_size);
    for (size_t i = 0; i < q_size; i++) Q_f32_from_fp16[i] = (float)Q_fp16[i];
    for (size_t i = 0; i < kv_size; i++) K_f32_from_fp16[i] = (float)K_fp16[i];
    for (size_t i = 0; i < kv_size; i++) V_f32_from_fp16[i] = (float)V_fp16[i];

    // Run FP32 reference (using the fp16-rounded inputs for fair comparison)
    std::vector<float> O_ref(o_size);
    reference_attention_f32(
        Q_f32_from_fp16.data(), K_f32_from_fp16.data(), V_f32_from_fp16.data(), O_ref.data(),
        cfg.batch, cfg.seq_len, cfg.seq_len,
        cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim,
        cfg.scale, cfg.position_offset, cfg.window_size, true
    );

    // Run Cactus attention kernel
    cactus_attention_f16(
        Q_fp16.data(), K_fp16.data(), V_fp16.data(), O_fp16.data(),
        cfg.batch, cfg.seq_len, cfg.seq_len,
        cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim,
        cfg.scale, nullptr, cfg.position_offset, cfg.window_size,
        true, false, false
    );

    // Convert output to f32
    std::vector<float> O_cactus(o_size);
    for (size_t i = 0; i < o_size; i++) O_cactus[i] = (float)O_fp16[i];

    // Compare
    float overall_cos = cos_sim(O_cactus.data(), O_ref.data(), o_size);

    // Per-head comparison
    float min_head_cos = 1.0f;
    for (size_t b = 0; b < cfg.batch; b++) {
        for (size_t qh = 0; qh < cfg.num_q_heads; qh++) {
            for (size_t qi = 0; qi < cfg.seq_len; qi++) {
                size_t offset = (b * cfg.seq_len * cfg.num_q_heads * cfg.head_dim) +
                               (qi * cfg.num_q_heads * cfg.head_dim) + (qh * cfg.head_dim);
                float hcs = cos_sim(O_cactus.data() + offset, O_ref.data() + offset, cfg.head_dim);
                min_head_cos = std::min(min_head_cos, hcs);
            }
        }
    }

    // Max absolute difference
    float max_diff = 0;
    for (size_t i = 0; i < o_size; i++)
        max_diff = std::max(max_diff, std::abs(O_cactus[i] - O_ref[i]));

    printf("%-40s  cos=%.6f  min_head_cos=%.6f  max_diff=%.6f\n",
           cfg.name, overall_cos, min_head_cos, max_diff);
}

int main() {
    printf("Attention Kernel Test: Cactus FP16 vs FP32 Reference\n");
    printf("====================================================\n\n");

    // Gemma 3n config: 8 q_heads, 2 kv_heads, 256 head_dim, scale=1.0
    TestConfig tests[] = {
        {1, 1,  8, 2, 256, 1.0f, 0, 0, "gemma3n: seq=1, no cache"},
        {1, 11, 8, 2, 256, 1.0f, 0, 0, "gemma3n: seq=11 (prefill)"},
        {1, 11, 8, 2, 256, 1.0f, 0, 512, "gemma3n: seq=11, window=512"},
        {1, 32, 8, 2, 256, 1.0f, 0, 0, "gemma3n: seq=32"},
        {1, 64, 8, 2, 256, 1.0f, 0, 0, "gemma3n: seq=64"},
        {1, 128, 8, 2, 256, 1.0f, 0, 0, "gemma3n: seq=128"},
        // With 1/sqrt(head_dim) scaling (typical models)
        {1, 11, 8, 2, 256, 1.0f/sqrtf(256), 0, 0, "gemma3n: seq=11, scale=1/sqrt(256)"},
        // Smaller head_dim for comparison
        {1, 11, 8, 2, 128, 1.0f, 0, 0, "head_dim=128: seq=11, scale=1.0"},
        {1, 11, 8, 2, 64,  1.0f, 0, 0, "head_dim=64: seq=11, scale=1.0"},
        // Global layer (no window)
        {1, 11, 8, 2, 256, 1.0f, 0, 0, "gemma3n global: seq=11"},
    };

    for (const auto& t : tests)
        run_test(t);

    printf("\n");
    printf("NOTE: cos < 0.999 indicates significant kernel precision loss.\n");
    printf("      cos < 0.99 with scale=1.0 would explain the observed L0 divergence.\n");

    return 0;
}
