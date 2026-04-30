#pragma once

#include "../cactus_kernels.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <vector>

struct TQWeightData {
    std::vector<__fp16>   codebook;
    std::vector<__fp16>   input_scale;
    std::vector<__fp16>   input_scale_recip;
    std::vector<int8_t>   left_signs;
    std::vector<int8_t>   right_signs;
    std::vector<uint32_t> permutation;
    std::vector<__fp16>   norms;
    std::vector<uint8_t>  packed;

    uint32_t bits = 0;
    uint32_t K = 0;
    uint32_t N = 0;
    uint32_t group_size = 0;
    uint32_t num_groups = 0;
    uint32_t flags = 0;
    CactusTQMatrix matrix() const {
        return CactusTQMatrix{
            .bits = bits,
            .K = K,
            .N = N,
            .group_size = group_size,
            .num_groups = num_groups,
            .flags = flags,
            .codebook = codebook.data(),
            .input_scale = input_scale.data(),
            .input_scale_recip = input_scale_recip.data(),
            .norms = norms.data(),
            .packed_indices = packed.data(),
            .left_signs = left_signs.data(),
            .right_signs = right_signs.data(),
            .permutation = permutation.data(),
        };
    }
};

inline bool tq_load_weights(const char* path, uint32_t max_rows, TQWeightData& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    auto r32 = [&](uint64_t off) {
        uint32_t v; f.seekg(off); f.read(reinterpret_cast<char*>(&v), 4); return v;
    };
    auto r64 = [&](uint64_t off) {
        uint64_t v; f.seekg(off); f.read(reinterpret_cast<char*>(&v), 8); return v;
    };
    auto read_at = [&](uint64_t off, void* dst, size_t n) {
        f.seekg(off); f.read(reinterpret_cast<char*>(dst), n);
        return size_t(f.gcount()) == n;
    };

    char magic[4]; f.read(magic, 4);
    if (std::memcmp(magic, "CACT", 4) != 0) return false;

    uint32_t source_flags = r32(4);
    if (r32(12) != 2) return false;
    uint64_t dim0 = r64(16);
    uint64_t dim1 = r64(24);
    uint32_t bits = r32(76);
    uint32_t group_size = r32(68);
    uint32_t num_groups = r32(72);
    if (r32(128) != 0) return false;
    if (r32(132) == 0) return false;

    uint64_t off_cb  = r64(80);
    uint64_t off_is  = r64(88);
    uint64_t off_rot = r64(96);
    uint64_t off_sc  = r64(104);
    uint64_t off_ix  = r64(112);

    constexpr uint32_t FLAG_CODE_ORDERED = 1u << 0;
    constexpr uint32_t FLAG_PANEL_MAJOR  = 1u << 1;
    if (source_flags & FLAG_PANEL_MAJOR) return false;

    uint32_t flags = 0;
    if (source_flags & FLAG_CODE_ORDERED) flags |= CACTUS_TQ_FLAG_CODE_ORDERED_INDICES;

    uint32_t N = uint32_t(std::min<uint64_t>(dim0, max_rows));

    uint32_t cb_size = 1u << bits;
    std::vector<float> cb32(cb_size);
    if (!read_at(off_cb, cb32.data(), cb_size * 4)) return false;
    out.codebook.resize(cb_size);
    for (uint32_t i = 0; i < cb_size; ++i) out.codebook[i] = __fp16(cb32[i]);

    out.input_scale.resize(dim1);
    if (!read_at(off_is, out.input_scale.data(), dim1 * 2)) return false;
    out.input_scale_recip.resize(dim1);
    for (uint64_t i = 0; i < dim1; ++i)
        out.input_scale_recip[i] = __fp16(1.f / float(out.input_scale[i]));

    out.left_signs.resize(group_size);
    out.right_signs.resize(group_size);
    out.permutation.resize(group_size);
    if (!read_at(off_rot,                    out.left_signs.data(),  group_size))   return false;
    if (!read_at(off_rot + group_size,       out.right_signs.data(), group_size))   return false;
    if (!read_at(off_rot + 2 * group_size,   out.permutation.data(), group_size * 4)) return false;

    out.norms.resize(size_t(N) * num_groups);
    if (!read_at(off_sc, out.norms.data(), out.norms.size() * 2)) return false;

    uint32_t pgb = (group_size * bits) / 8;
    out.packed.resize(size_t(N) * num_groups * pgb);
    if (!read_at(off_ix, out.packed.data(), out.packed.size())) return false;

    out.bits = bits;
    out.K = uint32_t(dim1);
    out.N = N;
    out.group_size = group_size;
    out.num_groups = num_groups;
    out.flags = flags;
    return true;
}
