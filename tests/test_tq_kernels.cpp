#include "cactus_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

enum : uint32_t {
    OP_GEMV = 1,
    OP_GEMM = 2,
    SOURCE_FLAG_CODE_ORDERED_INDICES = 1u << 0,
    SOURCE_FLAG_PANEL_MAJOR = 1u << 1,
};

struct TQHeader {
    uint32_t flags;
    uint64_t dim0;
    uint64_t dim1;
    uint32_t precision;
    uint64_t indices_bytes;
    uint64_t scales_bytes;
    uint32_t group_size;
    uint32_t num_groups;
    uint32_t bits;
    uint64_t off_codebook;
    uint64_t off_input_scale;
    uint64_t off_rotation;
    uint64_t off_scales;
    uint64_t off_indices;
    uint32_t rotation_family;
    uint32_t has_input_scale;
};

struct WeightCase {
    uint32_t bits;
    uint32_t M;
    uint32_t K;
    uint32_t N;
    uint32_t group_size;
    uint32_t num_groups;
    uint32_t flags;
    std::vector<__fp16> codebook;
    std::vector<__fp16> input_scale;
    std::vector<__fp16> input_scale_recip;
    std::vector<int8_t> left_signs;
    std::vector<int8_t> right_signs;
    std::vector<uint32_t> permutation;
    std::vector<__fp16> norms;
    std::vector<uint8_t> packed;
    std::vector<__fp16> A;
};

bool read_exact(std::ifstream& in, uint64_t offset, void* dst, size_t bytes) {
    in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!in) return false;
    in.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(bytes));
    return static_cast<size_t>(in.gcount()) == bytes;
}

uint32_t u32_at(const uint8_t* p) {
    uint32_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

uint64_t u64_at(const uint8_t* p) {
    uint64_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

bool read_header(const std::string& path, TQHeader& h) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    uint8_t header[136];
    if (!read_exact(in, 0, header, sizeof(header))) return false;
    if (std::memcmp(header, "CACT", 4) != 0) return false;
    const uint32_t ndim = u32_at(header + 12);
    if (ndim != 2) return false;
    h.flags = u32_at(header + 4);
    h.dim0 = u64_at(header + 16);
    h.dim1 = u64_at(header + 24);
    h.precision = u32_at(header + 48);
    h.indices_bytes = u64_at(header + 52);
    h.scales_bytes = u64_at(header + 60);
    h.group_size = u32_at(header + 68);
    h.num_groups = u32_at(header + 72);
    h.bits = u32_at(header + 76);
    h.off_codebook = u64_at(header + 80);
    h.off_input_scale = u64_at(header + 88);
    h.off_rotation = u64_at(header + 96);
    h.off_scales = u64_at(header + 104);
    h.off_indices = u64_at(header + 112);
    h.rotation_family = u32_at(header + 128);
    h.has_input_scale = u32_at(header + 132);
    return true;
}

uint32_t map_flags(uint32_t source_flags) {
    uint32_t out = 0;
    if (source_flags & SOURCE_FLAG_PANEL_MAJOR) out |= CACTUS_TQ_FLAG_PANEL_MAJOR;
    if (source_flags & SOURCE_FLAG_CODE_ORDERED_INDICES) out |= CACTUS_TQ_FLAG_CODE_ORDERED_INDICES;
    return out;
}

uint32_t packed_group_bytes(uint32_t bits, uint32_t group_size) {
    return (group_size * bits) / 8;
}

uint8_t packed_index_at(const WeightCase& tc, uint32_t row, uint32_t group, uint32_t k) {
    const uint32_t pgb = packed_group_bytes(tc.bits, tc.group_size);
    const uint8_t* base = tc.packed.data() + (static_cast<size_t>(row) * tc.num_groups + group) * pgb;
    if (tc.bits == 2) {
        const uint8_t byte = base[k >> 2];
        return static_cast<uint8_t>((byte >> ((k & 3u) * 2u)) & 0x3u);
    }
    const uint8_t byte = base[k >> 1];
    return static_cast<uint8_t>((k & 1u) ? (byte >> 4) : (byte & 0x0Fu));
}

void fwht(std::vector<__fp16>& x) {
    const size_t n = x.size();
    for (size_t h = 1; h < n; h <<= 1) {
        for (size_t i = 0; i < n; i += h << 1) {
            for (size_t j = i; j < i + h; ++j) {
                const __fp16 a = x[j];
                const __fp16 b = x[j + h];
                x[j] = static_cast<__fp16>(a + b);
                x[j + h] = static_cast<__fp16>(a - b);
            }
        }
    }
    const __fp16 inv = static_cast<__fp16>(1.0f / std::sqrt(static_cast<float>(n)));
    for (__fp16& v : x) v = static_cast<__fp16>(v * inv);
}

void transform_activation_group(const WeightCase& tc, const __fp16* x, uint32_t group, __fp16* out) {
    const uint32_t gs = tc.group_size;
    std::vector<__fp16> work(gs);
    const uint32_t base = group * gs;
    for (uint32_t k = 0; k < gs; ++k) {
        work[k] = static_cast<__fp16>(x[k] * tc.input_scale_recip[base + k]
                                      * static_cast<__fp16>(tc.left_signs[k]));
    }
    fwht(work);
    for (uint32_t k = 0; k < gs; ++k) {
        work[k] = static_cast<__fp16>(work[k] * static_cast<__fp16>(tc.right_signs[k]));
    }
    if (tc.flags & CACTUS_TQ_FLAG_CODE_ORDERED_INDICES) {
        std::copy(work.begin(), work.end(), out);
    } else {
        for (uint32_t k = 0; k < gs; ++k) out[k] = work[tc.permutation[k]];
    }
}

std::vector<__fp16> reference_output(const WeightCase& tc) {
    std::vector<__fp16> code_basis(static_cast<size_t>(tc.M) * tc.K);
    for (uint32_t m = 0; m < tc.M; ++m) {
        for (uint32_t g = 0; g < tc.num_groups; ++g) {
            transform_activation_group(
                tc,
                tc.A.data() + static_cast<size_t>(m) * tc.K + static_cast<size_t>(g) * tc.group_size,
                g,
                code_basis.data() + static_cast<size_t>(m) * tc.K + static_cast<size_t>(g) * tc.group_size);
        }
    }

    std::vector<__fp16> out(static_cast<size_t>(tc.M) * tc.N);
    for (uint32_t m = 0; m < tc.M; ++m) {
        for (uint32_t n = 0; n < tc.N; ++n) {
            float sum = 0.0f;
            for (uint32_t g = 0; g < tc.num_groups; ++g) {
                float group_sum = 0.0f;
                const __fp16* z = code_basis.data() + static_cast<size_t>(m) * tc.K
                    + static_cast<size_t>(g) * tc.group_size;
                for (uint32_t k = 0; k < tc.group_size; ++k) {
                    const uint8_t idx = packed_index_at(tc, n, g, k);
                    group_sum += static_cast<float>(z[k]) * static_cast<float>(tc.codebook[idx]);
                }
                sum += static_cast<float>(tc.norms[static_cast<size_t>(n) * tc.num_groups + g]) * group_sum;
            }
            out[static_cast<size_t>(m) * tc.N + n] = static_cast<__fp16>(sum);
        }
    }
    return out;
}

bool load_weight_case(const std::string& path, uint32_t expected_bits, uint32_t op,
                      uint32_t rows, uint32_t M, WeightCase& tc) {
    TQHeader h{};
    if (!read_header(path, h)) {
        std::cerr << "skip missing/unreadable weight " << path << "\n";
        return false;
    }
    if (h.bits != expected_bits || h.rotation_family != 0 || h.has_input_scale == 0) {
        std::cerr << path << ": unsupported TQ header\n";
        return false;
    }
    if (h.flags & SOURCE_FLAG_PANEL_MAJOR) {
        std::cerr << path << ": panel-major weights are not used by this direct-weight test\n";
        return false;
    }

    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    tc.bits = expected_bits;
    tc.M = M;
    tc.K = static_cast<uint32_t>(h.dim1);
    tc.N = std::min<uint32_t>(rows, static_cast<uint32_t>(h.dim0));
    tc.group_size = h.group_size;
    tc.num_groups = h.num_groups;
    tc.flags = map_flags(h.flags);

    std::vector<float> cb32(1u << expected_bits);
    if (!read_exact(in, h.off_codebook, cb32.data(), cb32.size() * sizeof(float))) return false;
    tc.codebook.resize(cb32.size());
    for (size_t i = 0; i < cb32.size(); ++i) tc.codebook[i] = static_cast<__fp16>(cb32[i]);

    tc.input_scale.resize(tc.K);
    if (!read_exact(in, h.off_input_scale, tc.input_scale.data(), tc.input_scale.size() * sizeof(__fp16))) return false;
    tc.input_scale_recip.resize(tc.K);
    for (uint32_t i = 0; i < tc.K; ++i) {
        tc.input_scale_recip[i] = static_cast<__fp16>(1.0f / static_cast<float>(tc.input_scale[i]));
    }

    tc.left_signs.resize(tc.group_size);
    tc.right_signs.resize(tc.group_size);
    tc.permutation.resize(tc.group_size);
    if (!read_exact(in, h.off_rotation, tc.left_signs.data(), tc.group_size)) return false;
    if (!read_exact(in, h.off_rotation + tc.group_size, tc.right_signs.data(), tc.group_size)) return false;
    if (!read_exact(in, h.off_rotation + 2u * tc.group_size,
                    tc.permutation.data(), tc.group_size * sizeof(uint32_t))) return false;

    tc.norms.resize(static_cast<size_t>(tc.N) * tc.num_groups);
    if (!read_exact(in, h.off_scales, tc.norms.data(), tc.norms.size() * sizeof(__fp16))) return false;

    const size_t packed_bytes = static_cast<size_t>(tc.N) * tc.num_groups
        * packed_group_bytes(tc.bits, tc.group_size);
    tc.packed.resize(packed_bytes);
    if (!read_exact(in, h.off_indices, tc.packed.data(), packed_bytes)) return false;

    (void)op;
    return true;
}

std::string asset_root() {
    if (const char* env = std::getenv("CACTUS_TEST_ASSETS")) {
        return std::string(env) + "/tq_kernels";
    }
    return "tests/assets/tq_kernels";
}

std::string weight_root() {
    if (const char* env = std::getenv("CACTUS_TQ_WEIGHTS_ROOT")) return env;
    return "weights/gemma-4-e2b-it-tqh-u4-codeorder";
}

bool load_activation_asset(const std::string& name, uint32_t M, uint32_t K,
                           std::vector<__fp16>& out) {
    const std::string path = asset_root() + "/" + name;
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "missing activation asset " << path << "\n";
        return false;
    }

    uint8_t header[16];
    if (!read_exact(in, 0, header, sizeof(header))) return false;
    if (std::memcmp(header, "TQAC", 4) != 0 || u32_at(header + 4) != 1) {
        std::cerr << path << ": bad activation header\n";
        return false;
    }
    const uint32_t file_m = u32_at(header + 8);
    const uint32_t file_k = u32_at(header + 12);
    if (file_m != M || file_k != K) {
        std::cerr << path << ": activation shape mismatch, expected M=" << M
                  << " K=" << K << " got M=" << file_m << " K=" << file_k << "\n";
        return false;
    }

    out.resize(static_cast<size_t>(M) * K);
    return read_exact(in, sizeof(header), out.data(), out.size() * sizeof(__fp16));
}

bool close_enough(__fp16 actual, __fp16 expected) {
    const float a = static_cast<float>(actual);
    const float e = static_cast<float>(expected);
    return std::abs(a - e) <= 0.25f + 0.02f * std::abs(e);
}

bool run_case(const char* label, const std::string& path, uint32_t bits, uint32_t op,
              uint32_t rows, uint32_t M, const char* activation_name) {
    WeightCase tc{};
    if (!load_weight_case(path, bits, op, rows, M, tc)) return false;
    if (!load_activation_asset(activation_name, tc.M, tc.K, tc.A)) return false;
    CactusTQMatrix W{
        tc.bits, tc.K, tc.N, tc.group_size, tc.num_groups, tc.flags,
        tc.codebook.data(), tc.input_scale.data(), tc.input_scale_recip.data(),
        tc.norms.data(), tc.packed.data(), tc.left_signs.data(), tc.right_signs.data(),
        tc.permutation.data(),
    };

    std::vector<__fp16> actual(static_cast<size_t>(tc.M) * tc.N);
    if (bits == 4 && op == OP_GEMV) cactus_tq4_gemv(&W, tc.A.data(), actual.data());
    else if (bits == 4 && op == OP_GEMM) cactus_tq4_gemm(&W, tc.A.data(), tc.M, actual.data());
    else if (bits == 2 && op == OP_GEMV) cactus_tq2_gemv(&W, tc.A.data(), actual.data());
    else if (bits == 2 && op == OP_GEMM) cactus_tq2_gemm(&W, tc.A.data(), tc.M, actual.data());
    else return false;

    const std::vector<__fp16> expected = reference_output(tc);
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!close_enough(actual[i], expected[i])) {
            std::cerr << label << ": output[" << i << "] expected "
                      << static_cast<float>(expected[i]) << " got "
                      << static_cast<float>(actual[i]) << "\n";
            return false;
        }
    }
    std::cout << "PASS " << label << " from " << path
              << " M=" << tc.M << " N=" << tc.N << " K=" << tc.K << "\n";
    return true;
}

}  // namespace

int main() {
    const std::string root = weight_root();
    bool ok = true;
    ok = run_case("tq4_gemv", root + "/layer_0_ffn_gate.weights", 4, OP_GEMV, 256, 1,
                  "tq4_gemv_activations.bin") && ok;
    ok = run_case("tq4_gemm", root + "/layer_0_ffn_gate.weights", 4, OP_GEMM, 256, 8,
                  "tq4_gemm_activations.bin") && ok;
    ok = run_case("tq2_gemv", root + "/embed_tokens_per_layer.weights", 2, OP_GEMV, 64, 1,
                  "tq2_gemv_activations.bin") && ok;
    ok = run_case("tq2_gemm", root + "/embed_tokens_per_layer.weights", 2, OP_GEMM, 64, 4,
                  "tq2_gemm_activations.bin") && ok;
    return ok ? 0 : 1;
}
