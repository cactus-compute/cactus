#include "test_utils.h"
#include <vector>
#include <cmath>
#include <cstring>
#include <random>
#include <utility>

using namespace TestUtils;

static uint8_t unpack_index(const uint8_t* base, uint32_t bits, uint32_t k);

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
    std::vector<uint8_t> packed_panels_buf;
    std::vector<float> norm_panels_buf;

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

        // Independent reference packer for the panel file format (built straight from the
        // original packed indices): packed_panels_buf[SB64][num_groups][nkg][128B] — nibble j of
        // a kg-block = byte j of the int8 panel [4 vectors][16 ch][4 K] (vector v = channels
        // 16v..16v+15, low nibble first). norm_panels_buf[SB64][num_groups][64] (cb_scale
        // folded; padded channels stay 0).
        const uint32_t nkg = group_size / 4;
        const size_t SB64 = (size_t(N) + 63) / 64;
        packed_panels_buf.assign(SB64 * num_groups * nkg * 128, 0);
        norm_panels_buf.assign(SB64 * num_groups * 64, 0.f);
        for (size_t sb = 0; sb < SB64; ++sb)
            for (uint32_t g = 0; g < num_groups; ++g) {
                uint8_t* wbase = packed_panels_buf.data() + (sb * num_groups + g) * nkg * 128;
                float* nbase = norm_panels_buf.data() + (sb * num_groups + g) * 64;
                for (uint32_t kg = 0; kg < nkg; ++kg)
                    for (uint32_t b = 0; b < 256; ++b) {        // expanded byte index in the kg-panel
                        const uint32_t v = b / 64, ch = (b % 64) / 4, kc = b % 4;
                        const size_t n = sb * 64 + v * 16 + ch;
                        if (n >= N) continue;
                        const uint8_t* row = packed.data() + (n * num_groups + g) * pgb;
                        const uint8_t idx = unpack_index(row, bits, kg * 4 + kc);
                        uint8_t& dst = wbase[kg * 128 + b / 2];
                        dst = (b & 1u) ? static_cast<uint8_t>(dst | (idx << 4))
                                       : static_cast<uint8_t>(dst | idx);
                    }
                for (uint32_t c = 0; c < 64; ++c) {
                    const size_t n = sb * 64 + c;
                    if (n < N) nbase[c] = static_cast<float>(norms[n * num_groups + g]) * cb_sc;
                }
            }
    }

    // INTERLEAVED_4ROW encoding (CQ4): exact inverse of the shipped decoder
    // (tq_preexpand_weights_interleaved / cactus_quant_4bit_gemv_interleaved). Panel per (nb,g) of
    // 4*pgb bytes; 16-byte columns; column word r = row r; per 8-byte half: low nibbles = k 0-3,
    // high nibbles = k 4-7 of the half's K-range. norms_il[(nb*ng+g)*4 + r] = norms[n][g].
    std::vector<uint8_t> packed_il;
    std::vector<__fp16> norms_il;
    std::vector<uint8_t> panels_il_buf;
    std::vector<float> npanels_il_buf;

    void make_interleaved() {
        if (bits != 4 || (N % 4) != 0) return;
        const uint32_t pgb = cactus_quant_packed_group_bytes(4, group_size);
        const size_t NB = N / 4;
        packed_il.assign(NB * num_groups * 4 * (size_t)pgb, 0);
        norms_il.resize(NB * num_groups * 4);
        for (size_t nb = 0; nb < NB; ++nb)
            for (uint32_t g = 0; g < num_groups; ++g) {
                uint8_t* panel = packed_il.data() + (nb * num_groups + g) * 4 * (size_t)pgb;
                for (uint32_t r = 0; r < 4; ++r) {
                    const size_t n = nb * 4 + r;
                    const uint8_t* row = packed.data() + (n * num_groups + g) * pgb;
                    for (uint32_t v = 0; v < group_size / 16; ++v)
                        for (uint32_t b = 0; b < 4; ++b) {
                            auto idx = [&](uint32_t k) { return unpack_index(row, 4, k); };
                            panel[(2 * v) * 16 + r * 4 + b] =
                                (uint8_t)(idx(16 * v + b) | (idx(16 * v + 4 + b) << 4));
                            panel[(2 * v + 1) * 16 + r * 4 + b] =
                                (uint8_t)(idx(16 * v + 8 + b) | (idx(16 * v + 12 + b) << 4));
                        }
                    norms_il[(nb * num_groups + g) * 4 + r] = norms[n * num_groups + g];
                }
            }
    }

    CactusQuantMatrix matrix_interleaved() {
        if (packed_il.empty()) make_interleaved();
        return CactusQuantMatrix{
            .bits = bits, .K = K, .N = N,
            .group_size = group_size, .num_groups = num_groups,
            .flags = CACTUS_QUANT_FLAG_INTERLEAVED_4ROW,
            .codebook = codebook.data(),
            .input_scale = input_scale.data(),
            .input_scale_recip = input_scale_recip.data(),
            .norms = norms_il.data(),
            .packed_indices = packed_il.data(),
            .left_signs = left_signs.data(),
            .right_signs = right_signs.data(),
            .permutation = permutation.data(),
            .rotation = nullptr,
            .expanded = nullptr,
            .norm_f32 = nullptr,
            .packed_panels = panels_il_buf.empty() ? nullptr : panels_il_buf.data(),
            .norm_panels = npanels_il_buf.empty() ? nullptr : npanels_il_buf.data(),
        };
    }

    // Build the panel layout from the INTERLEAVED matrix through the reference encoder.
    void preexpand_il() {
        CactusQuantMatrix W = matrix_interleaved();
        const size_t SB64 = ((size_t)N + 63) / 64;
        panels_il_buf.resize(SB64 * num_groups * (size_t)(group_size / 4) * 128);
        npanels_il_buf.resize(SB64 * num_groups * 64);
        cactus_quant_build_panels(&W, panels_il_buf.data(), npanels_il_buf.data());
    }

    CactusQuantMatrix matrix() const {
        return CactusQuantMatrix{
            .bits = bits, .K = K, .N = N,
            .group_size = group_size, .num_groups = num_groups,
            .flags = 0,
            .codebook = codebook.data(),
            .input_scale = input_scale.data(),
            .input_scale_recip = input_scale_recip.data(),
            .norms = norms.data(),
            .packed_indices = packed.data(),
            .left_signs = left_signs.data(),
            .right_signs = right_signs.data(),
            .permutation = permutation.data(),
            .rotation = nullptr,
            .expanded = expanded_buf.empty() ? nullptr : expanded_buf.data(),
            .norm_f32 = norm_f32_buf.empty() ? nullptr : norm_f32_buf.data(),
            .packed_panels = packed_panels_buf.empty() ? nullptr : packed_panels_buf.data(),
            .norm_panels = norm_panels_buf.empty() ? nullptr : norm_panels_buf.data(),
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
            uint32_t word = static_cast<uint32_t>(base[byte_idx]) >> bit_idx;
            if (bit_idx > 5) {
                word |= static_cast<uint32_t>(base[byte_idx + 1]) << (8 - bit_idx);
            }
            return word & 0x7u;
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

// ── Panel-format GEMV validation ─────────────────────────────────────────────────────────────
// With packed panels present (built here by the independent test packer), the real
// cactus_quant_matmul dispatch takes the panel kernels; validate them against the SAME FP32
// reference oracle the legacy kernels are gated on. backend pins the variant: 1 = NEON panel
// kernels, 2 = forced SME2 leaves (k_sme >= 1 so the streaming path is test-covered).
struct BackendGuard {
    explicit BackendGuard(int b) { cactus_quant_set_backend(b); }
    ~BackendGuard() { cactus_quant_set_backend(0); }
};

static bool test_cq_panel(uint32_t bits, int backend, double& mse_out) {
    const uint32_t K = 1024, N = 64, gs = 128;
    SyntheticCQ cq(bits, K, N, gs, 123);
    cq.preexpand();
    CactusQuantMatrix mat = cq.matrix();

    std::mt19937 gen(77);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<float> x_f32(K);
    for (auto& v : x_f32) v = dist(gen);
    std::vector<float> ref(N, 0.f);
    cq_reference_gemv_f32(cq, x_f32.data(), ref.data());

    std::vector<__fp16> x_f16(K), y_f16(N, static_cast<__fp16>(0));
    for (size_t i = 0; i < K; i++) x_f16[i] = static_cast<__fp16>(x_f32[i]);

    {
        BackendGuard bg(backend);
        cactus_quant_matmul(&mat, x_f16.data(), 1, y_f16.data());
    }

    mse_out = compute_mse(ref.data(), y_f16.data(), N);
    return mse_out <= 0.1;
}

// GEMM (M>1) variant: validates the panel GEMM (M and N tails included) vs a per-row FP32
// reference.
static bool test_cq_panel_gemm(uint32_t bits, uint32_t M, uint32_t N, int backend, double& mse_out) {
    const uint32_t K = 512, gs = 128;
    SyntheticCQ cq(bits, K, N, gs, 321);
    cq.preexpand();
    CactusQuantMatrix mat = cq.matrix();

    std::mt19937 gen(91);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<float> X(static_cast<size_t>(M) * K);
    for (auto& v : X) v = dist(gen);

    std::vector<float> ref(static_cast<size_t>(M) * N, 0.f);
    for (uint32_t m = 0; m < M; m++)
        cq_reference_gemv_f32(cq, X.data() + static_cast<size_t>(m) * K, ref.data() + static_cast<size_t>(m) * N);

    std::vector<__fp16> Xf(static_cast<size_t>(M) * K), Yf(static_cast<size_t>(M) * N, static_cast<__fp16>(0));
    for (size_t i = 0; i < static_cast<size_t>(M) * K; i++) Xf[i] = static_cast<__fp16>(X[i]);

    {
        BackendGuard bg(backend);
        cactus_quant_matmul(&mat, Xf.data(), M, Yf.data());
    }

    mse_out = compute_mse(ref.data(), Yf.data(), static_cast<size_t>(M) * N);
    return mse_out <= 0.1;
}

// NEON and SME2 leaves must agree on the SAME panel fixture: both accumulate identical int32
// partials before the fp32 rescale, so outputs differ only by fp16 store rounding.
static bool test_panel_neon_sme_agreement() {
    const uint32_t K = 1024, gs = 128;
    for (uint32_t M : {1u, 20u}) {
        const uint32_t N = (M == 1) ? 192 : 72;
        SyntheticCQ cq(4, K, N, gs, 909);
        cq.preexpand();
        CactusQuantMatrix mat = cq.matrix();
        std::vector<__fp16> X(static_cast<size_t>(M) * K);
        fill_random_fp16(X, -1.f, 1.f);
        std::vector<__fp16> y_neon(static_cast<size_t>(M) * N), y_sme(static_cast<size_t>(M) * N);
        {
            BackendGuard bg(1);
            cactus_quant_matmul(&mat, X.data(), M, y_neon.data());
        }
        {
            BackendGuard bg(2);
            cactus_quant_matmul(&mat, X.data(), M, y_sme.data());
        }
        double mx = 0;
        for (size_t i = 0; i < y_neon.size(); ++i) {
            const double d = std::abs(static_cast<double>(y_neon[i]) - static_cast<double>(y_sme[i])) /
                             std::max(1.0, std::abs(static_cast<double>(y_neon[i])));
            mx = std::max(mx, d);
        }
        if (mx > 2e-3) {
            std::cerr << "  neon/sme agreement M=" << M << " rel_err=" << mx << "\n";
            return false;
        }
    }
    return true;
}

// ── Orthogonal-rotation CQ4 (the Gemma lm_head format) ──────────────────────────────────────
// Oracle: y[n] = norm[n] * sum_i cb[idx(n,i)] * (a_scaled @ R)[i], fp32. Gates the incumbent and
// the panel virtual-group driver against the same reference, plus head-to-head agreement.
static bool test_orth_panel(int backend, double& mse_inc, double& mse_panel) {
    // PRODUCTION lm_head format: orthogonal-rotation CQ4 in the INTERLEAVED_4ROW packed layout
    // (gs == K, ng == 1) for legacy bundles; panel files store it with virtual 128-wide groups.
    // Row-major orthogonal bundles were a transpiler bug and take the per-row fallback.
    const uint32_t K = 1536, N = 1024, VGS = 128, VNG = K / VGS;
    std::mt19937 gen(2024);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<__fp16> codebook(16), isr(K), norms(N), rot((size_t)K * K), rot_t((size_t)K * K);
    for (auto& v : codebook) v = (__fp16)dist(gen);
    for (auto& v : isr) v = (__fp16)(0.7f + 0.3f * dist(gen));
    for (auto& v : norms) v = (__fp16)(dist(gen) * 0.05f);
    for (size_t i = 0; i < rot.size(); i++) rot[i] = (__fp16)(dist(gen) * 0.04f);
    for (uint32_t k = 0; k < K; k++)
        for (uint32_t i = 0; i < K; i++) rot_t[(size_t)i * K + k] = rot[(size_t)k * K + i];
    std::vector<uint8_t> packed((size_t)N * K / 2);
    for (auto& v : packed) v = (uint8_t)(gen() & 0xFF);
    std::vector<__fp16> x(K);
    for (auto& v : x) v = (__fp16)dist(gen);

    // Encode INTERLEAVED_4ROW panels from the row-major nibbles (exact inverse of the shipped
    // decoder): per 4-row block nb, chunk c covers k = 8c..8c+7; byte at [c*16 + r4*4 + b] holds
    // low nibble = idx(row, 8c+b), high nibble = idx(row, 8c+4+b).
    std::vector<uint8_t> packed_il((size_t)N * K / 2);
    for (uint32_t nb = 0; nb < N / 4; nb++) {
        uint8_t* panel = packed_il.data() + (size_t)nb * 4 * (K / 2);
        for (uint32_t c = 0; c < K / 8; c++)
            for (uint32_t r4 = 0; r4 < 4; r4++) {
                const uint8_t* row = packed.data() + (size_t)(nb * 4 + r4) * K / 2;
                for (uint32_t b = 0; b < 4; b++) {
                    uint8_t lo = unpack_index(row, 4, c * 8 + b);
                    uint8_t hi = unpack_index(row, 4, c * 8 + 4 + b);
                    panel[c * 16 + r4 * 4 + b] = (uint8_t)(lo | (hi << 4));
                }
            }
    }

    // fp32 oracle (weights identical in both packings)
    std::vector<float> ar(K, 0.f), ref(N);
    for (uint32_t k = 0; k < K; k++) {
        float a = (float)x[k] * (float)isr[k];
        for (uint32_t i = 0; i < K; i++) ar[i] += a * (float)rot[(size_t)k * K + i];
    }
    for (uint32_t n = 0; n < N; n++) {
        const uint8_t* row = packed.data() + (size_t)n * K / 2;
        float acc = 0.f;
        for (uint32_t i = 0; i < K; i++)
            acc += (float)codebook[unpack_index(row, 4, i)] * ar[i];
        ref[n] = acc * (float)norms[n];
    }

    // incumbent (production NEON interleaved lm_head path)
    CactusQuantMatrix Wo{ .bits = 4, .K = K, .N = N, .group_size = K, .num_groups = 1,
        .flags = CACTUS_QUANT_FLAG_ORTHOGONAL | CACTUS_QUANT_FLAG_INTERLEAVED_4ROW,
        .codebook = codebook.data(), .input_scale = nullptr,
        .input_scale_recip = isr.data(), .norms = norms.data(), .packed_indices = packed_il.data(),
        .left_signs = nullptr, .right_signs = nullptr, .permutation = nullptr,
        .rotation = rot.data(), .rotation_t = nullptr, .expanded = nullptr, .norm_f32 = nullptr,
        .packed_panels = nullptr, .norm_panels = nullptr };
    std::vector<__fp16> y_inc(N);
    cactus_quant_orthogonal_matmul(&Wo, x.data(), 1, y_inc.data());
    mse_inc = compute_mse(ref.data(), y_inc.data(), N);

    // Panel-file view: virtual 128-wide groups already applied (byte-exact regrouping), norms
    // replicated per virtual group, panels from the reference encoder, rotation_t alongside —
    // exactly what the loader hands the kernels for an orthogonal panel file. Routed through
    // the real cactus_quant_orthogonal_matmul dispatch.
    std::vector<__fp16> norms_rep((size_t)N * VNG);
    for (uint32_t nb = 0; nb < N / 4; nb++)
        for (uint32_t g = 0; g < VNG; g++)
            for (uint32_t ni = 0; ni < 4; ni++)
                norms_rep[((size_t)nb * VNG + g) * 4 + ni] = norms[nb * 4 + ni];
    CactusQuantMatrix W2 = Wo;
    W2.flags = CACTUS_QUANT_FLAG_INTERLEAVED_4ROW;
    W2.group_size = VGS; W2.num_groups = VNG; W2.norms = norms_rep.data();
    const size_t SB64 = (N + 63) / 64;
    std::vector<uint8_t> panels(SB64 * VNG * (VGS / 4) * 128);
    std::vector<float> npanels(SB64 * VNG * 64);
    cactus_quant_build_panels(&W2, panels.data(), npanels.data());
    W2.flags = CACTUS_QUANT_FLAG_ORTHOGONAL;       // loader view: panel files carry no IL flag
    W2.packed_indices = nullptr;                   // panel files carry no legacy packed region
    W2.packed_panels = panels.data(); W2.norm_panels = npanels.data();
    W2.rotation_t = rot_t.data();
    std::vector<__fp16> y_panel(N);
    {
        BackendGuard bg(backend);
        cactus_quant_orthogonal_matmul(&W2, x.data(), 1, y_panel.data());
    }
    mse_panel = compute_mse(ref.data(), y_panel.data(), N);
    return mse_inc <= 0.1 && mse_panel <= 0.1 && mse_panel <= mse_inc * 4 + 1e-4;
}

// ── Interleaved-4row (legacy PRODUCTION format) tests ────────────────────────────────────────
// CQ4 interleaved vs the FP32 oracle, through the real dispatch. use_panels=false exercises the
// legacy interleaved NEON kernel (old bundles); use_panels=true builds the panel layout from the
// SAME interleaved fixture through the reference encoder and exercises the panel GEMV
// (multi-super-block stealing included: N=192 = 3 super-blocks).
static bool test_cq4_interleaved(bool use_panels, int backend, double& mse_out,
                                 uint32_t K = 1024, uint32_t N = 192, uint32_t gs = 128) {
    SyntheticCQ cq(4, K, N, gs, 777);
    if (use_panels) cq.preexpand_il();
    CactusQuantMatrix mat = cq.matrix_interleaved();

    std::mt19937 gen(31);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<float> x_f32(K);
    for (auto& v : x_f32) v = dist(gen);
    std::vector<float> ref(N, 0.f);
    cq_reference_gemv_f32(cq, x_f32.data(), ref.data());

    std::vector<__fp16> x_f16(K), y(N, (__fp16)0);
    for (size_t i = 0; i < K; i++) x_f16[i] = (__fp16)x_f32[i];
    {
        BackendGuard bg(backend);
        cactus_quant_matmul(&mat, x_f16.data(), 1, y.data());
    }
    mse_out = compute_mse(ref.data(), y.data(), N);
    return mse_out <= 0.1;
}

// Batched orthogonal embedding-row dequant must match the per-row reference for both packed
// layouts (row-major nibbles + interleaved-4row). Any random byte stream is a valid nibble
// stream, so the fixture is random packed data + random codebook/norms/rotation.
static bool test_orth_embed_rows() {
    const uint32_t K = 256, vocab = 64;
    std::mt19937 gen(99);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<uint8_t> packed(static_cast<size_t>(vocab) * K / 2);
    for (auto& b : packed) b = static_cast<uint8_t>(gen() & 0xFF);
    std::vector<__fp16> codebook(16), rotation(static_cast<size_t>(K) * K), scale_recip(K);
    std::vector<__fp16> norms(vocab);
    for (auto& v : codebook) v = static_cast<__fp16>(dist(gen));
    for (auto& v : rotation) v = static_cast<__fp16>(dist(gen) * 0.06f);
    for (auto& v : scale_recip) v = static_cast<__fp16>(0.9f + 0.2f * std::fabs(dist(gen)));
    for (auto& v : norms) v = static_cast<__fp16>(0.5f + std::fabs(dist(gen)));
    const uint32_t rows[5] = {5, 9, 63, 0, 17};
    // Legacy batched fn is INTERLEAVED_4ROW-only (the pre-panel production format); the per-row
    // reference covers both layouts and serves as the oracle here.
    for (uint32_t flags : {(uint32_t)CACTUS_QUANT_FLAG_INTERLEAVED_4ROW}) {
        std::vector<__fp16> ref(5 * K), got(5 * K);
        for (int u = 0; u < 5; ++u)
            cactus_quant_dequantize_orthogonal_embedding_row(
                4, K, rows[u], packed.data(), codebook.data(), norms.data(),
                scale_recip.data(), rotation.data(), flags, ref.data() + (size_t)u * K);
        cactus_quant_dequantize_orthogonal_embedding_rows(
            4, K, rows, 5, packed.data(), codebook.data(), norms.data(),
            scale_recip.data(), rotation.data(), flags, got.data());
        double mx = 0;
        for (size_t i = 0; i < ref.size(); ++i)
            mx = std::max(mx, (double)std::fabs((float)ref[i] - (float)got[i]));
        if (mx > 1e-2) {
            std::cerr << "  orth_embed_rows flags=" << flags << " max_err=" << mx << "\n";
            return false;
        }
    }
    // Panel-format batched variant: encode the SAME weights as panels (row-major source through
    // the reference encoder with the virtual 128-wide groups a panel file stores), then check
    // the panel row decoder against the row-major per-row reference.
    {
        const uint32_t VGS = 128, VNG = K / VGS;
        std::vector<__fp16> norms_rep((size_t)vocab * VNG);
        for (uint32_t n = 0; n < vocab; ++n)
            for (uint32_t g = 0; g < VNG; ++g) norms_rep[(size_t)n * VNG + g] = norms[n];
        CactusQuantMatrix Wv{ .bits = 4, .K = K, .N = vocab, .group_size = VGS, .num_groups = VNG,
            .flags = 0, .codebook = codebook.data(), .input_scale = nullptr,
            .input_scale_recip = scale_recip.data(), .norms = norms_rep.data(),
            .packed_indices = packed.data(), .left_signs = nullptr, .right_signs = nullptr,
            .permutation = nullptr, .rotation = rotation.data(), .rotation_t = nullptr,
            .expanded = nullptr, .norm_f32 = nullptr,
            .packed_panels = nullptr, .norm_panels = nullptr };
        const size_t SB64 = (vocab + 63) / 64;
        std::vector<uint8_t> panels(SB64 * VNG * (VGS / 4) * 128);
        std::vector<float> npanels(SB64 * VNG * 64);
        cactus_quant_build_panels(&Wv, panels.data(), npanels.data());

        std::vector<__fp16> ref(5 * K), got(5 * K);
        for (int u = 0; u < 5; ++u)
            cactus_quant_dequantize_orthogonal_embedding_row(
                4, K, rows[u], packed.data(), codebook.data(), norms.data(),
                scale_recip.data(), rotation.data(), 0, ref.data() + (size_t)u * K);
        cactus_quant_dequantize_orthogonal_embedding_rows_panels(
            K, VGS, VNG, rows, 5, panels.data(), codebook.data(), norms_rep.data(),
            scale_recip.data(), rotation.data(), got.data());
        double mx = 0;
        for (size_t i = 0; i < ref.size(); ++i)
            mx = std::max(mx, (double)std::fabs((float)ref[i] - (float)got[i]));
        if (mx > 1e-2) {
            std::cerr << "  orth_embed_rows[panel] max_err=" << mx << "\n";
            return false;
        }
    }
    return true;
}

// The panel encoding must be byte-identical whether built from row-major or interleaved
// weights — the reference encoder is layout-invariant (and the transpiler's panel writer is
// gated against it). The independent test packer is NOT byte-compared: it keeps the original
// indices while the encoder canonicalizes equal-value codebook indices (both dequantize
// identically; the panel kernel tests cover that equivalence).
static bool test_panel_layout_invariance() {
    const uint32_t K = 512, N = 128, gs = 128;
    SyntheticCQ cq(4, K, N, gs, 555);
    CactusQuantMatrix w_rm = cq.matrix();
    cq.preexpand_il();     // reference encoder over the interleaved fixture
    const size_t SB64 = (N + 63) / 64;
    std::vector<uint8_t> panels_rm(SB64 * cq.num_groups * (gs / 4) * 128);
    std::vector<float> npanels_rm(SB64 * cq.num_groups * 64);
    cactus_quant_build_panels(&w_rm, panels_rm.data(), npanels_rm.data());
    bool ok = panels_rm == cq.panels_il_buf && npanels_rm == cq.npanels_il_buf;
    if (!ok) std::cerr << "  panel layout invariance FAILED\n";
    return ok;
}

// Production-format bench: interleaved fixtures at Gemma 4 E2B shapes; legacy file-layout NEON
// kernel vs the panel NEON GEMV over the same weights. Alternating rounds + best-of: run-to-run
// noise (E-core scheduling, thermal/clock drift) exceeds the kernel deltas, so a single timed
// run cannot rank the formats.
static void print_panel_comparison() {
    std::cout << "── CQ4 GEMV: panel NEON vs legacy file-layout NEON (gemma shapes) ──────────────────\n";
    struct Shape { const char* name; uint32_t K, N; int iters; };
    Shape shapes[] = {
        {"cq4 1x1536x6144 (ffn)", 1536, 6144, 30},
        {"cq4 1x1536x12288 (gate_up)", 1536, 12288, 20},
        {"cq4 1x2048x1536 (o_proj)", 2048, 1536, 30},
        {"cq4 1x1536x2048 (q_proj)", 1536, 2048, 30},
        {"cq4 1x6144x1536 (down)", 6144, 1536, 30},
        {"cq4 1x1536x262144 (lm_head)", 1536, 262144, 6},
        {"cq4 1x1536x512 (kv_proj)", 1536, 512, 60},
    };
    for (auto& sh : shapes) {
        SyntheticCQ cq(4, sh.K, sh.N, 128);
        CactusQuantMatrix mat_file = cq.matrix_interleaved();   // no panels: legacy IL kernel
        cq.preexpand_il();
        CactusQuantMatrix mat_panel = cq.matrix_interleaved();  // panels set: panel GEMV
        std::vector<__fp16> x(sh.K), y(sh.N);
        fill_random_fp16(x, -1.f, 1.f);
        auto run_ms = [&](CactusQuantMatrix* m, int backend) {
            BackendGuard bg(backend);
            cactus_quant_matmul(m, x.data(), 1, y.data());      // warmup
            Timer t;
            for (int i = 0; i < sh.iters; i++) cactus_quant_matmul(m, x.data(), 1, y.data());
            return t.elapsed_ms() / sh.iters;
        };
        const bool sme = cactus_quant_sme_available() != 0;
        double ms_f = 1e30, ms_p = 1e30, ms_s = 1e30;
        for (int round = 0; round < 5; round++) {
            ms_f = std::min(ms_f, run_ms(&mat_file, 1));
            ms_p = std::min(ms_p, run_ms(&mat_panel, 1));
            if (sme) ms_s = std::min(ms_s, run_ms(&mat_panel, 0));
        }
        const double fl = 2.0 * sh.K * sh.N;
        double gf = fl / (ms_f * 1e6), gp = fl / (ms_p * 1e6);
        std::cout << "  " << std::left << std::setw(31) << sh.name
                  << " file-NEON " << std::fixed << std::setprecision(1) << gf
                  << " | panel " << gp << " GF";
        if (sme) std::cout << " | panel-auto(sme) " << fl / (ms_s * 1e6) << " GF";
        std::cout << " | " << std::setprecision(2) << (gp / gf)
                  << "x  (best of 5x" << sh.iters << ")\n";
    }
    // GEMM (M>1): panel GEMM vs the legacy expand-from-packed NEON GEMM (the real-model M>1
    // path for old bundles).
    for (uint32_t M : {4u, 16u, 32u, 64u, 128u, 256u}) {
        const uint32_t K = 1024, N = 1024, gs = 128;
        SyntheticCQ cq(4, K, N, gs);
        CactusQuantMatrix mat_legacy = cq.matrix();   // no panels: per-call expand NEON GEMM
        cq.preexpand();
        CactusQuantMatrix mat_panel = cq.matrix();    // panels set: panel NEON GEMM
        std::vector<__fp16> A(static_cast<size_t>(M) * K), Cc(static_cast<size_t>(M) * N);
        fill_random_fp16(A, -1.f, 1.f);
        auto bench = [&](CactusQuantMatrix* m, int backend) -> double {
            BackendGuard bg(backend);
            cactus_quant_matmul(m, A.data(), M, Cc.data());     // warmup
            const int iters = 8;
            Timer t;
            for (int i = 0; i < iters; i++) cactus_quant_matmul(m, A.data(), M, Cc.data());
            return t.elapsed_ms() / iters;
        };
        const bool sme = cactus_quant_sme_available() != 0;
        double ml = 1e30, mp = 1e30, ms = 1e30;
        for (int round = 0; round < 3; round++) {
            ml = std::min(ml, bench(&mat_legacy, 1));
            mp = std::min(mp, bench(&mat_panel, 1));
            if (sme) ms = std::min(ms, bench(&mat_panel, 2));
        }
        double gl = (2.0 * M * K * N) / (ml * 1e6), gp = (2.0 * M * K * N) / (mp * 1e6);
        std::string label = "cq4 M" + std::to_string(M) + " 1024x1024";
        std::cout << "  " << std::left << std::setw(20) << label
                  << " legacy " << std::fixed << std::setprecision(1) << gl << " GFLOPS"
                  << " | panel " << gp << " GFLOPS";
        if (sme) std::cout << " | panel-sme " << (2.0 * M * K * N) / (ms * 1e6) << " GFLOPS";
        std::cout << " | " << std::setprecision(2) << (gp / gl) << "x  (best of 3x8)\n";
    }
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

    // ── Panel-format registry: validate the panel kernels through the real dispatch.
    // backend 1 pins the NEON panel kernels; backend 2 (when SME2 silicon is present) forces the
    // streaming LUTI4/SMOPA leaves so they stay test-covered.
    struct GemmCase { uint32_t M, N; };
    const GemmCase gemm_cases[] = {GemmCase{5, 64}, GemmCase{20, 64}, GemmCase{20, 72}, GemmCase{64, 256}};
    std::vector<std::pair<int, const char*>> backends{{1, "[panel]"}};
    if (cactus_quant_sme_available()) backends.push_back({2, "[sme2]"});
    for (auto [backend, tag] : backends) {
        for (uint32_t bits : {1u, 2u, 3u, 4u}) {
            double mse_p = 0.0;
            bool ok = test_cq_panel(bits, backend, mse_p);
            runner.run_test(std::string("matmul_cq") + std::to_string(bits) + tag, ok);
            if (!ok) std::cerr << "    cq" << bits << tag << " MSE=" << mse_p << "\n";
        }
        // GEMM (M>1): M tail (20 -> 16+4) and N tail (72 -> 4 super-blocks + 8 valid) for CQ4.
        for (GemmCase gc : gemm_cases) {
            double mse_g = 0.0;
            bool ok = test_cq_panel_gemm(4, gc.M, gc.N, backend, mse_g);
            runner.run_test(std::string("matmul_cq4_M") + std::to_string(gc.M) +
                            "_N" + std::to_string(gc.N) + tag, ok);
            if (!ok) std::cerr << "    cq4 M=" << gc.M << " N=" << gc.N << tag << " MSE=" << mse_g << "\n";
        }
        {
            double mi = 0, mp = 0;
            bool orth_ok = test_orth_panel(backend, mi, mp);
            runner.run_test(std::string("matmul_cq4_orth") + tag, orth_ok);
            if (!orth_ok) std::cerr << "    orth" << tag << " mse_incumbent=" << mi << " mse_panel=" << mp << "\n";
        }
        {
            double m2 = 0;
            runner.run_test(std::string("matmul_cq4_il") + tag, test_cq4_interleaved(true, backend, m2));
        }
        // The panel GEMM is bits-agnostic (panel nibbles are codebook indices) — validate CQ1-3 too.
        for (uint32_t b : {1u, 2u, 3u}) {
            double mse_g = 0.0;
            bool ok = test_cq_panel_gemm(b, 20, 64, backend, mse_g);
            runner.run_test(std::string("matmul_cq") + std::to_string(b) + "_gemm" + tag, ok);
            if (!ok) std::cerr << "    cq" << b << " gemm" << tag << " MSE=" << mse_g << "\n";
        }
    }
    {
        double m1 = 0;
        runner.run_test("matmul_cq4_il[file]", test_cq4_interleaved(false, 1, m1));
        // N=4164: 1041 IL blocks -> 66 chunks (16-block + 1-block tail) -> multi-thread fused
        // driver (phase-A stealing, spin barrier, 4-chunk grabs); N=192 stays on the serial path.
        double m_mt = 0;
        runner.run_test("matmul_cq4_il_mt[file]", test_cq4_interleaved(false, 1, m_mt, 1024, 4164, 128));
        runner.run_test("panel_layout_invariance", test_panel_layout_invariance());
        runner.run_test("orth_embed_rows_batched", test_orth_embed_rows());
    }
    if (cactus_quant_sme_available()) {
        runner.run_test("panel_neon_sme_agreement", test_panel_neon_sme_agreement());
    } else {
        std::cout << "  (SME2 unavailable on this CPU — [sme2] variants skipped)\n";
    }

    runner.print_benchmarks_header();
    runner.run_bench("benchmarks", run_benchmarks());
    print_mse_report();
    print_panel_comparison();
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
