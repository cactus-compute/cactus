#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

constexpr uint32_t kGroupSize = 128;
constexpr uint32_t kSamples = 1000;

struct ErrStats {
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    double sum_rel_sq = 0.0;
    float max_abs = 0.0f;
    uint64_t count = 0;

    void add(float ref, float got) {
        const float d = std::abs(got - ref);
        sum_abs += d;
        sum_sq += static_cast<double>(d) * d;
        const float denom = std::max(std::abs(ref), 1e-8f);
        const double rel = static_cast<double>(d) / denom;
        sum_rel_sq += rel * rel;
        max_abs = std::max(max_abs, d);
        ++count;
    }

    void print(const char* name) const {
        const double inv_n = 1.0 / static_cast<double>(count);
        std::printf("%-28s mean_abs=%.9g rms=%.9g rel_rms=%.9g max_abs=%.9g\n",
                    name,
                    sum_abs * inv_n,
                    std::sqrt(sum_sq * inv_n),
                    std::sqrt(sum_rel_sq * inv_n),
                    static_cast<double>(max_abs));
    }
};

void fwht128(std::vector<float>& x) {
    for (uint32_t h = 1; h < kGroupSize; h <<= 1) {
        for (uint32_t i = 0; i < kGroupSize; i += (h << 1)) {
            for (uint32_t j = i; j < i + h; ++j) {
                const float a = x[j];
                const float b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
    }
    const float inv = 1.0f / std::sqrt(static_cast<float>(kGroupSize));
    for (float& v : x) v *= inv;
}

float quantize_sym_i8(const std::vector<float>& src, std::vector<int8_t>& q) {
    float max_abs = 0.0f;
    for (float v : src) max_abs = std::max(max_abs, std::abs(v));
    float scale = max_abs / 127.0f;
    if (scale < 1e-10f) scale = 1e-10f;
    const float inv = 1.0f / scale;
    for (uint32_t i = 0; i < kGroupSize; ++i) {
        const int qi = static_cast<int>(std::lrint(src[i] * inv));
        q[i] = static_cast<int8_t>(std::clamp(qi, -127, 127));
    }
    return scale;
}

struct AsymParams {
    float mean = 0.0f;
    float scale = 1e-10f;
};

AsymParams quantize_asym_centered_i8(const std::vector<float>& src, std::vector<int8_t>& q) {
    AsymParams p;
    for (float v : src) p.mean += v;
    p.mean /= static_cast<float>(kGroupSize);

    float max_abs = 0.0f;
    for (float v : src) max_abs = std::max(max_abs, std::abs(v - p.mean));
    p.scale = max_abs / 127.0f;
    if (p.scale < 1e-10f) p.scale = 1e-10f;
    const float inv = 1.0f / p.scale;
    for (uint32_t i = 0; i < kGroupSize; ++i) {
        const int qi = static_cast<int>(std::lrint((src[i] - p.mean) * inv));
        q[i] = static_cast<int8_t>(std::clamp(qi, -127, 127));
    }
    return p;
}

} // namespace

int main() {
    std::mt19937 rng(12345);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    std::vector<float> x(kGroupSize);
    std::vector<float> work(kGroupSize);
    std::vector<float> recon(kGroupSize);
    std::vector<int8_t> q(kGroupSize);

    ErrStats raw_sym;
    ErrStats had_sym_roundtrip;
    ErrStats raw_asym;
    ErrStats had_asym_roundtrip;

    double fp16_sum_abs = 0.0;
    double fp16_sum_sq = 0.0;
    double fp16_sum_rel_sq = 0.0;
    float fp16_max_abs = 0.0f;
    uint64_t fp16_count = 0;

    for (uint32_t sample = 0; sample < kSamples; ++sample) {
        const float amplitude = std::exp(normal(rng) * 0.6f);
        const float bias = normal(rng) * 0.1f;
        for (float& v : x) v = bias + amplitude * normal(rng);

        float scale = quantize_sym_i8(x, q);
        for (uint32_t i = 0; i < kGroupSize; ++i) {
            raw_sym.add(x[i], static_cast<float>(q[i]) * scale);
        }

        AsymParams ap = quantize_asym_centered_i8(x, q);
        for (uint32_t i = 0; i < kGroupSize; ++i) {
            raw_asym.add(x[i], ap.mean + static_cast<float>(q[i]) * ap.scale);
        }

        work = x;
        fwht128(work);
        scale = quantize_sym_i8(work, q);
        for (uint32_t i = 0; i < kGroupSize; ++i) recon[i] = static_cast<float>(q[i]) * scale;
        fwht128(recon);
        for (uint32_t i = 0; i < kGroupSize; ++i) had_sym_roundtrip.add(x[i], recon[i]);

        AsymParams hap = quantize_asym_centered_i8(work, q);
        for (uint32_t i = 0; i < kGroupSize; ++i) recon[i] = hap.mean + static_cast<float>(q[i]) * hap.scale;
        fwht128(recon);
        for (uint32_t i = 0; i < kGroupSize; ++i) had_asym_roundtrip.add(x[i], recon[i]);

        for (float v : work) {
            const float got = static_cast<float>(static_cast<__fp16>(v));
            const float d = std::abs(got - v);
            fp16_sum_abs += d;
            fp16_sum_sq += static_cast<double>(d) * d;
            const double rel = static_cast<double>(d) / std::max(std::abs(v), 1e-8f);
            fp16_sum_rel_sq += rel * rel;
            fp16_max_abs = std::max(fp16_max_abs, d);
            ++fp16_count;
        }
    }

    std::printf("TQ activation quantization precision (%u random groups of %u)\n",
                kSamples, kGroupSize);
    raw_sym.print("raw symmetric int8");
    had_sym_roundtrip.print("hadamard symmetric int8");
    raw_asym.print("raw centered int8");
    had_asym_roundtrip.print("hadamard centered int8");

    const double fp16_inv = 1.0 / static_cast<double>(fp16_count);
    std::printf("%-28s mean_abs=%.9g rms=%.9g rel_rms=%.9g max_abs=%.9g\n",
                "A_rot fp32->fp16",
                fp16_sum_abs * fp16_inv,
                std::sqrt(fp16_sum_sq * fp16_inv),
                std::sqrt(fp16_sum_rel_sq * fp16_inv),
                static_cast<double>(fp16_max_abs));

    return 0;
}
