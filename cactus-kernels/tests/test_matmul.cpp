#include "test_utils.h"
#include <vector>
#include <cmath>
#include <random>

using namespace TestUtils;

struct SyntheticCQ {
    uint32_t bits, K, N, group_size, num_groups;
    std::vector<__fp16> codebook;
    std::vector<__fp16> input_scale;
    std::vector<__fp16> input_scale_recip;
    std::vector<__fp16> norms;
    std::vector<int8_t> left_signs;
    std::vector<int8_t> right_signs;
    std::vector<uint32_t> permutation;
    std::vector<uint8_t> packed;

    SyntheticCQ(uint32_t b, uint32_t k, uint32_t n, uint32_t gs, uint32_t seed = 42)
        : bits(b), K(k), N(n), group_size(gs), num_groups(k / gs) {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(-1.f, 1.f);

        uint32_t cb_size = 1u << bits;
        codebook.resize(cb_size);
        for (auto& v : codebook) v = static_cast<__fp16>(dist(gen));

        input_scale.resize(K);
        input_scale_recip.resize(K);
        for (uint32_t i = 0; i < K; i++) {
            float s = 0.5f + std::abs(dist(gen));
            input_scale[i] = static_cast<__fp16>(s);
            input_scale_recip[i] = static_cast<__fp16>(1.f / s);
        }

        norms.resize(size_t(N) * num_groups);
        for (auto& v : norms) v = static_cast<__fp16>(dist(gen) * 0.1f);

        left_signs.resize(group_size);
        right_signs.resize(group_size);
        for (auto& v : left_signs) v = (gen() & 1) ? 1 : -1;
        for (auto& v : right_signs) v = (gen() & 1) ? 1 : -1;

        permutation.resize(group_size);
        for (uint32_t i = 0; i < group_size; i++) permutation[i] = i;

        size_t packed_bytes = size_t(N) * num_groups * cactus_quant_packed_group_bytes(bits, group_size);
        packed.resize(packed_bytes);
        for (auto& v : packed) v = static_cast<uint8_t>(gen() & 0xFF);
    }

    std::vector<int8_t> expanded_buf;
    std::vector<float> norm_f32_buf;

    void preexpand() {
        int8_t cb_i8[16] = {};
        float cb_max = 0.f;
        uint32_t cb_size = 1u << bits;
        for (uint32_t i = 0; i < cb_size; i++) {
            float v = std::abs(static_cast<float>(codebook[i]));
            if (v > cb_max) cb_max = v;
        }
        float cb_sc = cb_max / 127.f;
        if (cb_sc < 1e-10f) cb_sc = 1e-10f;
        for (uint32_t i = 0; i < cb_size; i++)
            cb_i8[i] = static_cast<int8_t>(std::round(static_cast<float>(codebook[i]) / cb_sc));
        int8x16_t cb_lut = vld1q_s8(cb_i8);

        size_t N_blocks = (N + 3) / 4;
        uint32_t pgb = cactus_quant_packed_group_bytes(bits, group_size);
        expanded_buf.resize(N_blocks * num_groups * group_size * 4);
        norm_f32_buf.resize(N_blocks * num_groups * 4);

        auto expand16 = [&](const uint8_t* p) -> int8x16_t {
            if (bits == 4) {
                uint8x8_t bytes = vld1_u8(p);
                return vqtbl1q_s8(cb_lut, vcombine_u8(vzip1_u8(vand_u8(bytes,vdup_n_u8(0x0F)),vshr_n_u8(bytes,4)),
                                                       vzip2_u8(vand_u8(bytes,vdup_n_u8(0x0F)),vshr_n_u8(bytes,4))));
            } else if (bits == 2) {
                uint8_t b0=p[0],b1=p[1],b2=p[2],b3=p[3];
                uint64_t lo=((uint64_t)(b0&3))|((uint64_t)((b0>>2)&3)<<8)|((uint64_t)((b0>>4)&3)<<16)|((uint64_t)((b0>>6)&3)<<24)|
                            ((uint64_t)(b1&3)<<32)|((uint64_t)((b1>>2)&3)<<40)|((uint64_t)((b1>>4)&3)<<48)|((uint64_t)((b1>>6)&3)<<56);
                uint64_t hi=((uint64_t)(b2&3))|((uint64_t)((b2>>2)&3)<<8)|((uint64_t)((b2>>4)&3)<<16)|((uint64_t)((b2>>6)&3)<<24)|
                            ((uint64_t)(b3&3)<<32)|((uint64_t)((b3>>2)&3)<<40)|((uint64_t)((b3>>4)&3)<<48)|((uint64_t)((b3>>6)&3)<<56);
                return vqtbl1q_s8(cb_lut, vcombine_u8(vcreate_u8(lo),vcreate_u8(hi)));
            } else if (bits == 1) {
                uint8_t b0=p[0],b1=p[1];
                uint64_t lo=((uint64_t)((b0>>0)&1))|((uint64_t)((b0>>1)&1)<<8)|((uint64_t)((b0>>2)&1)<<16)|((uint64_t)((b0>>3)&1)<<24)|
                            ((uint64_t)((b0>>4)&1)<<32)|((uint64_t)((b0>>5)&1)<<40)|((uint64_t)((b0>>6)&1)<<48)|((uint64_t)((b0>>7)&1)<<56);
                uint64_t hi=((uint64_t)((b1>>0)&1))|((uint64_t)((b1>>1)&1)<<8)|((uint64_t)((b1>>2)&1)<<16)|((uint64_t)((b1>>3)&1)<<24)|
                            ((uint64_t)((b1>>4)&1)<<32)|((uint64_t)((b1>>5)&1)<<40)|((uint64_t)((b1>>6)&1)<<48)|((uint64_t)((b1>>7)&1)<<56);
                return vqtbl1q_s8(cb_lut, vcombine_u8(vcreate_u8(lo),vcreate_u8(hi)));
            } else {
                uint64_t raw=0; std::memcpy(&raw,p,6);
                uint64_t lo=0,hi=0;
                for(int i=0;i<8;i++) lo|=((raw>>(i*3))&7ULL)<<(i*8);
                for(int i=0;i<8;i++) hi|=((raw>>((i+8)*3))&7ULL)<<(i*8);
                return vqtbl1q_s8(cb_lut, vcombine_u8(vcreate_u8(lo),vcreate_u8(hi)));
            }
        };

        for (size_t nb = 0; nb < N_blocks; ++nb) {
            size_t n_start = nb * 4;
            size_t valid_n = std::min(size_t(4), static_cast<size_t>(N) - n_start);
            for (uint32_t g = 0; g < num_groups; ++g) {
                int8x16_t exp4[4][16];
                uint32_t n_vecs = group_size / 16;
                for (size_t ni = 0; ni < valid_n; ++ni) {
                    const uint8_t* pk = packed.data() + (static_cast<size_t>(n_start+ni)*num_groups+g)*pgb;
                    for (uint32_t v = 0; v < n_vecs; ++v)
                        exp4[ni][v] = expand16(pk + (v*16*bits)/8);
                }
                for (size_t ni = valid_n; ni < 4; ++ni)
                    for (uint32_t v = 0; v < n_vecs; ++v) exp4[ni][v] = vdupq_n_s8(0);

                int8_t* dst = expanded_buf.data() + (nb*num_groups+g)*group_size*4;
                for (uint32_t v = 0; v < n_vecs; ++v) {
                    int32x4_t r0=vreinterpretq_s32_s8(exp4[0][v]),r1=vreinterpretq_s32_s8(exp4[1][v]);
                    int32x4_t r2=vreinterpretq_s32_s8(exp4[2][v]),r3=vreinterpretq_s32_s8(exp4[3][v]);
                    int32x4_t t01l=vzip1q_s32(r0,r1),t01h=vzip2q_s32(r0,r1);
                    int32x4_t t23l=vzip1q_s32(r2,r3),t23h=vzip2q_s32(r2,r3);
                    vst1q_s8(dst+v*64,    vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s32(t01l),vreinterpretq_s64_s32(t23l))));
                    vst1q_s8(dst+v*64+16, vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s32(t01l),vreinterpretq_s64_s32(t23l))));
                    vst1q_s8(dst+v*64+32, vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s32(t01h),vreinterpretq_s64_s32(t23h))));
                    vst1q_s8(dst+v*64+48, vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s32(t01h),vreinterpretq_s64_s32(t23h))));
                }
                float* nd = norm_f32_buf.data() + (nb*num_groups+g)*4;
                for (size_t ni = 0; ni < 4; ++ni)
                    nd[ni] = (n_start+ni < N) ? static_cast<float>(norms[(n_start+ni)*num_groups+g]) * cb_sc : 0.f;
            }
        }
    }

    CactusQuantMatrix matrix() const {
        return CactusQuantMatrix{
            .bits = bits, .K = K, .N = N,
            .group_size = group_size, .num_groups = num_groups,
            .flags = CACTUS_QUANT_FLAG_CODE_ORDERED_INDICES,
            .codebook = codebook.data(),
            .input_scale = input_scale.data(),
            .input_scale_recip = input_scale_recip.data(),
            .norms = norms.data(),
            .packed_indices = packed.data(),
            .left_signs = left_signs.data(),
            .right_signs = right_signs.data(),
            .permutation = permutation.data(),
            .expanded = expanded_buf.empty() ? nullptr : expanded_buf.data(),
            .norm_f32 = norm_f32_buf.empty() ? nullptr : norm_f32_buf.data(),
        };
    }
};

static void fwht_f32(float* x, uint32_t n) {
    for (uint32_t h = 1; h < n; h <<= 1)
        for (uint32_t i = 0; i < n; i += h << 1)
            for (uint32_t j = i; j < i + h; ++j) {
                float a = x[j], b = x[j + h];
                x[j] = a + b; x[j + h] = a - b;
            }
    float s = 1.f / std::sqrt(static_cast<float>(n));
    for (uint32_t i = 0; i < n; ++i) x[i] *= s;
}

static uint8_t unpack_index(const uint8_t* base, uint32_t bits, uint32_t k) {
    switch (bits) {
        case 1: return (base[k / 8] >> (k % 8)) & 0x1u;
        case 2: return (base[k / 4] >> ((k & 3u) * 2u)) & 0x3u;
        case 3: {
            uint32_t bit_offset = k * 3;
            uint32_t byte_idx = bit_offset / 8;
            uint32_t bit_idx = bit_offset % 8;
            uint32_t word = base[byte_idx] | (base[byte_idx + 1] << 8);
            return (word >> bit_idx) & 0x7u;
        }
        case 4: return (k & 1u) ? (base[k / 2] >> 4) : (base[k / 2] & 0x0Fu);
        default: return 0;
    }
}

static void cq_reference_gemv_f32(const SyntheticCQ& w, const float* x, float* y) {
    uint32_t pgb = cactus_quant_packed_group_bytes(w.bits, w.group_size);
    for (uint32_t n = 0; n < w.N; ++n) {
        for (uint32_t g = 0; g < w.num_groups; ++g) {
            uint32_t base_k = g * w.group_size;
            std::vector<float> z(w.group_size);
            for (uint32_t k = 0; k < w.group_size; ++k)
                z[k] = x[base_k + k] / static_cast<float>(w.input_scale[base_k + k])
                        * static_cast<float>(w.left_signs[k]);
            fwht_f32(z.data(), w.group_size);
            for (uint32_t k = 0; k < w.group_size; ++k)
                z[k] *= static_cast<float>(w.right_signs[k]);

            const uint8_t* packed_row = w.packed.data() + (size_t(n) * w.num_groups + g) * pgb;
            float gsum = 0.f;
            for (uint32_t k = 0; k < w.group_size; ++k) {
                uint8_t idx = unpack_index(packed_row, w.bits, k);
                gsum += z[k] * static_cast<float>(w.codebook[idx]);
            }
            y[n] += static_cast<float>(w.norms[size_t(n) * w.num_groups + g]) * gsum;
        }
    }
}

static double compute_mse(const float* ref, const __fp16* actual, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        double diff = static_cast<double>(ref[i]) - static_cast<double>(actual[i]);
        sum += diff * diff;
    }
    return sum / static_cast<double>(n);
}

// ══════════════════════════════════════════════════════════════════════════════
// Correctness tests
// ══════════════════════════════════════════════════════════════════════════════

bool test_matmul_f16() {
    const size_t M = 4, K = 1024, N = 64;
    std::vector<__fp16> a(M * K), b(N * K), c(M * N);
    fill_random_fp16(a, -0.5f, 0.5f);
    fill_random_fp16(b, -0.5f, 0.5f);
    cactus_matmul_f16(a.data(), b.data(), c.data(), M, K, N);
    for (size_t i = 0; i < M; i++)
        for (size_t j = 0; j < N; j++) {
            float ref = 0.0f;
            for (size_t k = 0; k < K; k++)
                ref += static_cast<float>(a[i * K + k]) * static_cast<float>(b[j * K + k]);
            if (std::abs(static_cast<float>(c[i * N + j]) - ref) > 1.0f) return false;
        }
    return true;
}

bool test_cq_correctness(uint32_t bits) {
    const uint32_t K = 1024, N = 64, gs = 128;
    SyntheticCQ cq(bits, K, N, gs, 123);
    CactusQuantMatrix mat = cq.matrix();

    std::mt19937 gen(77);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<float> x_f32(K);
    for (auto& v : x_f32) v = dist(gen);

    // FP32 reference
    std::vector<float> ref(N, 0.f);
    cq_reference_gemv_f32(cq, x_f32.data(), ref.data());

    // FP16 kernel
    std::vector<__fp16> x_f16(K), y_f16(N, static_cast<__fp16>(0));
    for (size_t i = 0; i < K; i++) x_f16[i] = static_cast<__fp16>(x_f32[i]);
    cactus_quant_matmul(&mat, x_f16.data(), 1, y_f16.data());

    double mse = compute_mse(ref.data(), y_f16.data(), N);
    
    double threshold = 0.1; 
    if (mse > threshold) {
        std::cerr << "  cq" << bits << " MSE=" << mse << " > " << threshold << "\n";
        return false;
    }
    return true;
}

bool run_benchmarks() {
    auto bench = [](const char* label, size_t M, size_t K, size_t N, auto fn) {
        fn();
        Timer t;
        for (int i = 0; i < 100; i++) fn();
        double ms = t.elapsed_ms() / 100.0;
        double gflops = (2.0 * M * K * N) / (ms * 1e6);
        std::cout << "  \u26A1 " << std::left << std::setw(28) << label
                  << std::fixed << std::setprecision(3) << ms << "ms  "
                  << std::setprecision(1) << gflops << " GFLOPS\n";
    };

    const size_t K = 1024, N = 1024;
    const size_t M_batch = 1024;
    const uint32_t gs = 128;

    // FP16
    {
        std::vector<__fp16> a(K), b(N * K), c(N);
        fill_random_fp16(a, -0.5f, 0.5f); fill_random_fp16(b, -0.5f, 0.5f);
        bench("matmul_f16 1x1024x1024", 1, K, N, [&]{ cactus_matmul_f16(a.data(), b.data(), c.data(), 1, K, N); });
    }
    {
        std::vector<__fp16> a(M_batch * K), b(N * K), c(M_batch * N);
        fill_random_fp16(a, -0.5f, 0.5f); fill_random_fp16(b, -0.5f, 0.5f);
        bench("matmul_f16 1024^3", M_batch, K, N, [&]{ cactus_matmul_f16(a.data(), b.data(), c.data(), M_batch, K, N); });
    }

    // TQ1
    {
        SyntheticCQ cq(1, K, N, gs);
        cq.preexpand();
        CactusQuantMatrix mat = cq.matrix();
        std::vector<__fp16> x(K), y(N);
        fill_random_fp16(x, -1.f, 1.f);
        bench("matmul_cq1 1x1024x1024", 1, K, N, [&]{ cactus_quant_matmul(&mat, x.data(), 1, y.data()); });
    }
    {
        SyntheticCQ cq(1, K, N, gs);
        cq.preexpand();
        CactusQuantMatrix mat = cq.matrix();
        std::vector<__fp16> A(M_batch * K), C(M_batch * N);
        fill_random_fp16(A, -1.f, 1.f);
        bench("matmul_cq1 1024^3", M_batch, K, N, [&]{ cactus_quant_matmul(&mat, A.data(), M_batch, C.data()); });
    }

    {
        SyntheticCQ cq(2, K, N, gs);
        cq.preexpand();
        CactusQuantMatrix mat = cq.matrix();
        std::vector<__fp16> x(K), y(N);
        fill_random_fp16(x, -1.f, 1.f);
        bench("matmul_cq2 1x1024x1024", 1, K, N, [&]{ cactus_quant_matmul(&mat, x.data(), 1, y.data()); });
    }
    {
        SyntheticCQ cq(2, K, N, gs);
        cq.preexpand();
        CactusQuantMatrix mat = cq.matrix();
        std::vector<__fp16> A(M_batch * K), C(M_batch * N);
        fill_random_fp16(A, -1.f, 1.f);
        bench("matmul_cq2 1024^3", M_batch, K, N, [&]{ cactus_quant_matmul(&mat, A.data(), M_batch, C.data()); });
    }

    {
        SyntheticCQ cq(3, K, N, gs);
        cq.preexpand();
        CactusQuantMatrix mat = cq.matrix();
        std::vector<__fp16> x(K), y(N);
        fill_random_fp16(x, -1.f, 1.f);
        bench("matmul_cq3 1x1024x1024", 1, K, N, [&]{ cactus_quant_matmul(&mat, x.data(), 1, y.data()); });
    }
    {
        SyntheticCQ cq(3, K, N, gs);
        cq.preexpand();
        CactusQuantMatrix mat = cq.matrix();
        std::vector<__fp16> A(M_batch * K), C(M_batch * N);
        fill_random_fp16(A, -1.f, 1.f);
        bench("matmul_cq3 1024^3", M_batch, K, N, [&]{ cactus_quant_matmul(&mat, A.data(), M_batch, C.data()); });
    }

    {
        SyntheticCQ cq(4, K, N, gs);
        cq.preexpand();
        CactusQuantMatrix mat = cq.matrix();
        std::vector<__fp16> x(K), y(N);
        fill_random_fp16(x, -1.f, 1.f);
        bench("matmul_cq4 1x1024x1024", 1, K, N, [&]{ cactus_quant_matmul(&mat, x.data(), 1, y.data()); });
    }
    {
        SyntheticCQ cq(4, K, N, gs);
        cq.preexpand();
        CactusQuantMatrix mat = cq.matrix();
        std::vector<__fp16> A(M_batch * K), C(M_batch * N);
        fill_random_fp16(A, -1.f, 1.f);
        bench("matmul_cq4 1024^3", M_batch, K, N, [&]{ cactus_quant_matmul(&mat, A.data(), M_batch, C.data()); });
    }

    auto bench2k = [](const char* label, size_t M, size_t K, size_t N, auto fn) {
        fn();
        Timer t;
        for (int i = 0; i < 10; i++) fn();
        double ms = t.elapsed_ms() / 10.0;
        double gflops = (2.0 * M * K * N) / (ms * 1e6);
        std::cout << "  \u26A1 " << std::left << std::setw(28) << label
                  << std::fixed << std::setprecision(3) << ms << "ms  "
                  << std::setprecision(1) << gflops << " GFLOPS\n";
    };

    const size_t K2 = 2048, N2 = 2048;
    const size_t M2 = 2048;
    const uint32_t gs2 = 128;

    {
        std::vector<__fp16> a(K2), b(N2 * K2), c(N2);
        fill_random_fp16(a, -0.5f, 0.5f); fill_random_fp16(b, -0.5f, 0.5f);
        bench2k("matmul_f16 1x2048x2048", 1, K2, N2, [&]{ cactus_matmul_f16(a.data(), b.data(), c.data(), 1, K2, N2); });
    }
    {
        std::vector<__fp16> a(M2 * K2), b(N2 * K2), c(M2 * N2);
        fill_random_fp16(a, -0.5f, 0.5f); fill_random_fp16(b, -0.5f, 0.5f);
        bench2k("matmul_f16 2048^3", M2, K2, N2, [&]{ cactus_matmul_f16(a.data(), b.data(), c.data(), M2, K2, N2); });
    }
    {
        SyntheticCQ cq(2, K2, N2, gs2);
        cq.preexpand();
        CactusQuantMatrix mat = cq.matrix();
        std::vector<__fp16> x(K2), y(N2);
        fill_random_fp16(x, -1.f, 1.f);
        bench2k("matmul_cq2 1x2048x2048", 1, K2, N2, [&]{ cactus_quant_matmul(&mat, x.data(), 1, y.data()); });
    }
    {
        SyntheticCQ cq(2, K2, N2, gs2);
        cq.preexpand();
        CactusQuantMatrix mat = cq.matrix();
        std::vector<__fp16> A(M2 * K2), C(M2 * N2);
        fill_random_fp16(A, -1.f, 1.f);
        bench2k("matmul_cq2 2048^3", M2, K2, N2, [&]{ cactus_quant_matmul(&mat, A.data(), M2, C.data()); });
    }
    {
        SyntheticCQ cq(4, K2, N2, gs2);
        cq.preexpand();
        CactusQuantMatrix mat = cq.matrix();
        std::vector<__fp16> x(K2), y(N2);
        fill_random_fp16(x, -1.f, 1.f);
        bench2k("matmul_cq4 1x2048x2048", 1, K2, N2, [&]{ cactus_quant_matmul(&mat, x.data(), 1, y.data()); });
    }
    {
        SyntheticCQ cq(4, K2, N2, gs2);
        cq.preexpand();
        CactusQuantMatrix mat = cq.matrix();
        std::vector<__fp16> A(M2 * K2), C(M2 * N2);
        fill_random_fp16(A, -1.f, 1.f);
        bench2k("matmul_cq4 2048^3", M2, K2, N2, [&]{ cactus_quant_matmul(&mat, A.data(), M2, C.data()); });
    }

    auto bench_model = [](const char* label, size_t M, size_t K, size_t N, auto fn) {
        fn();
        Timer t;
        for (int i = 0; i < 5; i++) fn();
        double ms = t.elapsed_ms() / 5.0;
        double gflops = (2.0 * M * K * N) / (ms * 1e6);
        std::cout << "  \u26A1 " << std::left << std::setw(28) << label
                  << std::fixed << std::setprecision(3) << ms << "ms  "
                  << std::setprecision(1) << gflops << " GFLOPS\n";
    };

    const size_t Km = 2304, Nm = 9216;
    const uint32_t gsm = 128;

    {
        std::vector<__fp16> a(Km), b(Nm * Km), c(Nm);
        fill_random_fp16(a, -0.5f, 0.5f); fill_random_fp16(b, -0.5f, 0.5f);
        bench_model("f16 1x2304x9216", 1, Km, Nm, [&]{ cactus_matmul_f16(a.data(), b.data(), c.data(), 1, Km, Nm); });
    }
    {
        SyntheticCQ cq(1, Km, Nm, gsm);
        cq.preexpand();
        CactusQuantMatrix mat = cq.matrix();
        std::vector<__fp16> x(Km), y(Nm);
        fill_random_fp16(x, -1.f, 1.f);
        bench_model("cq1 1x2304x9216", 1, Km, Nm, [&]{ cactus_quant_matmul(&mat, x.data(), 1, y.data()); });
    }
    {
        SyntheticCQ cq(2, Km, Nm, gsm);
        cq.preexpand();
        CactusQuantMatrix mat = cq.matrix();
        std::vector<__fp16> x(Km), y(Nm);
        fill_random_fp16(x, -1.f, 1.f);
        bench_model("cq2 1x2304x9216", 1, Km, Nm, [&]{ cactus_quant_matmul(&mat, x.data(), 1, y.data()); });
    }
    {
        SyntheticCQ cq(4, Km, Nm, gsm);
        cq.preexpand();
        CactusQuantMatrix mat = cq.matrix();
        std::vector<__fp16> x(Km), y(Nm);
        fill_random_fp16(x, -1.f, 1.f);
        bench_model("cq4 1x2304x9216", 1, Km, Nm, [&]{ cactus_quant_matmul(&mat, x.data(), 1, y.data()); });
    }

    return true;
}


void print_mse_report() {
    const uint32_t K = 1024, N = 256, gs = 128;

    std::mt19937 gen(99);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<float> x_f32(K);
    for (auto& v : x_f32) v = dist(gen);
    std::vector<__fp16> x_f16(K);
    for (size_t i = 0; i < K; i++) x_f16[i] = static_cast<__fp16>(x_f32[i]);

    std::cout << "── MSE vs FP32 reference ──────────────────────────────────────────────────────────\n";

    for (uint32_t bits : {1u, 2u, 3u, 4u}) {
        SyntheticCQ cq(bits, K, N, gs, 55 + bits);
        cq.preexpand();
        CactusQuantMatrix mat = cq.matrix();

        std::vector<float> ref(N, 0.f);
        cq_reference_gemv_f32(cq, x_f32.data(), ref.data());

        std::vector<__fp16> y(N, static_cast<__fp16>(0));
        cactus_quant_matmul(&mat, x_f16.data(), 1, y.data());

        double mse = compute_mse(ref.data(), y.data(), N);
        double max_err = 0.0;
        for (size_t i = 0; i < N; i++) {
            double err = std::abs(static_cast<double>(ref[i]) - static_cast<double>(y[i]));
            max_err = std::max(max_err, err);
        }

        std::cout << "  TQ" << bits << " │ MSE=" << std::scientific << std::setprecision(4) << mse
                  << "  max_err=" << std::fixed << std::setprecision(5) << max_err << "\n";
    }
}

int main() {
    TestRunner runner("Matrix Multiplication");
    runner.run_test("matmul_f16", test_matmul_f16());
    runner.run_test("matmul_cq1", test_cq_correctness(1));
    runner.run_test("matmul_cq2", test_cq_correctness(2));
    runner.run_test("matmul_cq3", test_cq_correctness(3));
    runner.run_test("matmul_cq4", test_cq_correctness(4));
    runner.print_benchmarks_header();
    runner.run_bench("benchmarks", run_benchmarks());
    print_mse_report();
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
