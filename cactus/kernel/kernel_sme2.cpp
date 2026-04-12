#include <arm_sve.h>
#include <cstdint>
#ifndef __ARM_FEATURE_SME2
#error "kernel_sme2.cpp must be compiled with SME2 enabled (e.g. -march=armv9.2-a+sme2)"
#endif

#include "kernel.h"
#include "kernel_utils.h"
#include <arm_sme.h>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>
#include <memory>
#include <unordered_map>

#if defined(__clang__)
#define CACTUS_UNROLL4 _Pragma("clang loop unroll_count(4)")
#else
#define CACTUS_UNROLL4
#endif

namespace {

struct Sme2PackedBKey {
    const __fp16* ptr;
    size_t K;
    size_t N;
    size_t tile_rows;
    size_t tile_pairs;

    bool operator==(const Sme2PackedBKey& other) const {
        return ptr == other.ptr &&
               K == other.K &&
               N == other.N &&
               tile_rows == other.tile_rows &&
               tile_pairs == other.tile_pairs;
    }
};

struct Sme2PackedBKeyHash {
    size_t operator()(const Sme2PackedBKey& key) const {
        size_t h = std::hash<const void*>{}(key.ptr);
        h ^= std::hash<size_t>{}(key.K) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h ^= std::hash<size_t>{}(key.N) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h ^= std::hash<size_t>{}(key.tile_rows) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h ^= std::hash<size_t>{}(key.tile_pairs) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        return h;
    }
};

uint64_t cactus_fp16_sample_fingerprint(const __fp16* data, size_t count) {
    if (!data || count == 0) return 0;

    constexpr size_t SAMPLE_COUNT = 16;
    const size_t step = std::max<size_t>(1, count / SAMPLE_COUNT);
    uint64_t hash = 1469598103934665603ull ^ static_cast<uint64_t>(count);

    auto mix_index = [&](size_t idx) {
        uint16_t bits = 0;
        std::memcpy(&bits, data + idx, sizeof(bits));
        hash ^= static_cast<uint64_t>(bits) + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
    };

    size_t idx = 0;
    for (size_t sample = 0; sample < SAMPLE_COUNT && idx < count; ++sample, idx += step) {
        mix_index(idx);
    }
    mix_index(count - 1);
    return hash;
}

uint64_t cactus_u8_sample_fingerprint(const uint8_t* data, size_t count) {
    if (!data || count == 0) return 0;

    constexpr size_t SAMPLE_COUNT = 32;
    const size_t step = std::max<size_t>(1, count / SAMPLE_COUNT);
    uint64_t hash = 1469598103934665603ull ^ static_cast<uint64_t>(count);

    for (size_t idx = 0, sample = 0; sample < SAMPLE_COUNT && idx < count; ++sample, idx += step) {
        hash ^= static_cast<uint64_t>(data[idx]) + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
    }
    hash ^= static_cast<uint64_t>(data[count - 1]) + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
    return hash;
}

uint64_t cactus_mix_fingerprint(uint64_t seed, uint64_t value) {
    seed ^= value + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2);
    return seed;
}

class Sme2PackedBCache {
public:
    static Sme2PackedBCache& instance() {
        static Sme2PackedBCache cache;
        return cache;
    }

    template <typename Builder>
    std::shared_ptr<std::vector<__fp16>> get_or_create(
        const Sme2PackedBKey& key,
        size_t elements,
        uint64_t fingerprint,
        Builder&& builder
    ) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = entries_.find(key);
            if (it != entries_.end() && it->second.fingerprint == fingerprint) {
                it->second.last_use = ++clock_;
                return it->second.data;
            }
        }

        auto packed = std::make_shared<std::vector<__fp16>>(elements);
        builder(packed->data());
        const size_t bytes = packed->size() * sizeof(__fp16);

        std::lock_guard<std::mutex> lock(mutex_);
        auto it = entries_.find(key);
        if (it != entries_.end() && it->second.fingerprint == fingerprint) {
            it->second.last_use = ++clock_;
            return it->second.data;
        }

        Entry& entry = entries_[key];
        if (entry.data && current_bytes_ >= entry.bytes) {
            current_bytes_ -= entry.bytes;
        }

        entry.data = packed;
        entry.bytes = bytes;
        entry.fingerprint = fingerprint;
        entry.last_use = ++clock_;
        current_bytes_ += bytes;
        evict_locked();
        return entry.data;
    }

private:
    struct Entry {
        std::shared_ptr<std::vector<__fp16>> data;
        size_t bytes = 0;
        uint64_t fingerprint = 0;
        uint64_t last_use = 0;
    };

    void evict_locked() {
        constexpr size_t MAX_CACHE_BYTES = 256ull * 1024ull * 1024ull;
        while (current_bytes_ > MAX_CACHE_BYTES && entries_.size() > 1) {
            auto oldest = entries_.end();
            for (auto it = entries_.begin(); it != entries_.end(); ++it) {
                if (oldest == entries_.end() || it->second.last_use < oldest->second.last_use) {
                    oldest = it;
                }
            }

            if (oldest == entries_.end()) break;
            if (current_bytes_ >= oldest->second.bytes) {
                current_bytes_ -= oldest->second.bytes;
            } else {
                current_bytes_ = 0;
            }
            entries_.erase(oldest);
        }
    }

    std::mutex mutex_;
    std::unordered_map<Sme2PackedBKey, Entry, Sme2PackedBKeyHash> entries_;
    size_t current_bytes_ = 0;
    uint64_t clock_ = 0;
};

struct Sme2PackedInt4Key {
    const int8_t* weights_ptr;
    const __fp16* scales_ptr;
    size_t K;
    size_t N;
    size_t group_size;
    size_t tile_rows;
    size_t tile_bytes;

    bool operator==(const Sme2PackedInt4Key& other) const {
        return weights_ptr == other.weights_ptr &&
               scales_ptr == other.scales_ptr &&
               K == other.K &&
               N == other.N &&
               group_size == other.group_size &&
               tile_rows == other.tile_rows &&
               tile_bytes == other.tile_bytes;
    }
};

struct Sme2PackedInt4KeyHash {
    size_t operator()(const Sme2PackedInt4Key& key) const {
        size_t h = std::hash<const void*>{}(key.weights_ptr);
        h ^= std::hash<const void*>{}(key.scales_ptr) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h ^= std::hash<size_t>{}(key.K) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h ^= std::hash<size_t>{}(key.N) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h ^= std::hash<size_t>{}(key.group_size) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h ^= std::hash<size_t>{}(key.tile_rows) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h ^= std::hash<size_t>{}(key.tile_bytes) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        return h;
    }
};

struct Sme2PackedInt4Panels {
    std::vector<int8_t> weights;
    std::vector<float> scales;
};

class Sme2PackedInt4Cache {
public:
    static Sme2PackedInt4Cache& instance() {
        static Sme2PackedInt4Cache cache;
        return cache;
    }

    template <typename Builder>
    std::shared_ptr<Sme2PackedInt4Panels> get_or_create(
        const Sme2PackedInt4Key& key,
        size_t weight_elements,
        size_t scale_elements,
        uint64_t fingerprint,
        Builder&& builder
    ) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = entries_.find(key);
            if (it != entries_.end() && it->second.fingerprint == fingerprint) {
                it->second.last_use = ++clock_;
                return it->second.data;
            }
        }

        auto packed = std::make_shared<Sme2PackedInt4Panels>();
        packed->weights.resize(weight_elements);
        packed->scales.resize(scale_elements);
        builder(*packed);
        const size_t bytes = packed->weights.size() * sizeof(int8_t) +
                             packed->scales.size() * sizeof(float);

        std::lock_guard<std::mutex> lock(mutex_);
        auto it = entries_.find(key);
        if (it != entries_.end() && it->second.fingerprint == fingerprint) {
            it->second.last_use = ++clock_;
            return it->second.data;
        }

        Entry& entry = entries_[key];
        if (entry.data && current_bytes_ >= entry.bytes) {
            current_bytes_ -= entry.bytes;
        }

        entry.data = packed;
        entry.bytes = bytes;
        entry.fingerprint = fingerprint;
        entry.last_use = ++clock_;
        current_bytes_ += bytes;
        evict_locked();
        return entry.data;
    }

private:
    struct Entry {
        std::shared_ptr<Sme2PackedInt4Panels> data;
        size_t bytes = 0;
        uint64_t fingerprint = 0;
        uint64_t last_use = 0;
    };

    void evict_locked() {
        constexpr size_t MAX_CACHE_BYTES = 256ull * 1024ull * 1024ull;
        while (current_bytes_ > MAX_CACHE_BYTES && entries_.size() > 1) {
            auto oldest = entries_.end();
            for (auto it = entries_.begin(); it != entries_.end(); ++it) {
                if (oldest == entries_.end() || it->second.last_use < oldest->second.last_use) {
                    oldest = it;
                }
            }

            if (oldest == entries_.end()) break;
            if (current_bytes_ >= oldest->second.bytes) {
                current_bytes_ -= oldest->second.bytes;
            } else {
                current_bytes_ = 0;
            }
            entries_.erase(oldest);
        }
    }

    std::mutex mutex_;
    std::unordered_map<Sme2PackedInt4Key, Entry, Sme2PackedInt4KeyHash> entries_;
    size_t current_bytes_ = 0;
    uint64_t clock_ = 0;
};

}  // namespace

static inline void cactus_pack_a_f16_row_block(
    const __fp16* __restrict a,
    __fp16* __restrict a_packed,
    size_t K,
    size_t rb,
    size_t row0,
    size_t active_r,
    size_t tile_rows,
    size_t tile_pairs,
    size_t k_pairs
) {
    const size_t block_stride = k_pairs * tile_pairs;
    const bool even_k = ((K & 1u) == 0);
    for (size_t kp = 0; kp < k_pairs; ++kp) {
        const size_t k0 = kp * 2;
        const size_t k1 = k0 + 1;
        __fp16* dst = a_packed + rb * block_stride + kp * tile_pairs;
        const __fp16* src_col = a + row0 * K + k0;

        if (even_k) {
            CACTUS_UNROLL4
            for (size_t r = 0; r < active_r; ++r) {
                dst[2 * r] = src_col[0];
                dst[2 * r + 1] = src_col[1];
                src_col += K;
            }
        } else {
            const bool has_k1 = (k1 < K);
            CACTUS_UNROLL4
            for (size_t r = 0; r < active_r; ++r) {
                dst[2 * r] = src_col[0];
                dst[2 * r + 1] = has_k1 ? src_col[1] : static_cast<__fp16>(0);
                src_col += K;
            }
        }
        CACTUS_UNROLL4
        for (size_t r = active_r; r < tile_rows; ++r) {
            dst[2 * r] = static_cast<__fp16>(0);
            dst[2 * r + 1] = static_cast<__fp16>(0);
        }
    }
}

static inline void cactus_pack_a_int8_row_block(
    const int8_t* __restrict a,
    int8_t* __restrict a_packed,
    size_t K,
    size_t rb,
    size_t row0,
    size_t active_r,
    size_t tile_rows,
    size_t tile_bytes,
    size_t k_quads
) {
    const size_t block_stride = k_quads * tile_bytes;
    for (size_t kq = 0; kq < k_quads; ++kq) {
        const size_t k0 = kq * 4;
        int8_t* dst = a_packed + rb * block_stride + kq * tile_bytes;

        for (size_t r = 0; r < active_r; ++r) {
            const int8_t* src = a + (row0 + r) * K + k0;
            dst[4 * r + 0] = (k0 + 0 < K) ? src[0] : 0;
            dst[4 * r + 1] = (k0 + 1 < K) ? src[1] : 0;
            dst[4 * r + 2] = (k0 + 2 < K) ? src[2] : 0;
            dst[4 * r + 3] = (k0 + 3 < K) ? src[3] : 0;
        }

        for (size_t r = active_r; r < tile_rows; ++r) {
            dst[4 * r + 0] = 0;
            dst[4 * r + 1] = 0;
            dst[4 * r + 2] = 0;
            dst[4 * r + 3] = 0;
        }
    }
}

static inline int8_t cactus_load_interleaved_int8_weight(
    const int8_t* __restrict b_interleaved,
    size_t K,
    size_t n,
    size_t k
) {
    const size_t n_block = n / 4;
    const size_t ni = n % 4;
    const size_t k_block = k / 4;
    const size_t ki = k % 4;
    return b_interleaved[(n_block * (K / 4) + k_block) * 16 + ni * 4 + ki];
}

static void cactus_pack_b_int4_for_sme2(
    const int8_t* __restrict b_packed_raw,
    const __fp16* __restrict b_scales,
    Sme2PackedInt4Panels& packed,
    size_t K,
    size_t N,
    size_t group_size,
    size_t tile_rows,
    size_t tile_bytes
) {
    const size_t num_groups = K / group_size;
    const size_t group_quads = group_size / 4;
    const size_t col_blocks = (N + tile_rows - 1) / tile_rows;
    const size_t n_storage = ((N + 3) / 4) * 4;

    std::vector<int8_t> b_interleaved(n_storage * K);
    cactus_unpack_int4_to_int8(reinterpret_cast<const uint8_t*>(b_packed_raw), b_interleaved.data(), b_interleaved.size());

    CactusThreading::parallel_for(col_blocks * num_groups, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
        [&](size_t start, size_t end) {
            for (size_t idx = start; idx < end; ++idx) {
                const size_t cb = idx / num_groups;
                const size_t g = idx % num_groups;
                const size_t col0 = cb * tile_rows;
                const size_t active_c = std::min(tile_rows, N - col0);
                const size_t k_base = g * group_size;

                float* scale_dst = packed.scales.data() + (cb * num_groups + g) * tile_rows;
                for (size_t c = 0; c < active_c; ++c) {
                    const size_t col = col0 + c;
                    scale_dst[c] = static_cast<float>(b_scales[((col / 4) * num_groups + g) * 4 + (col % 4)]);
                }
                for (size_t c = active_c; c < tile_rows; ++c) {
                    scale_dst[c] = 0.0f;
                }

                int8_t* weight_dst = packed.weights.data() + (cb * num_groups + g) * group_quads * tile_bytes;
                for (size_t kq = 0; kq < group_quads; ++kq) {
                    int8_t* quad_dst = weight_dst + kq * tile_bytes;
                    const size_t k0 = k_base + kq * 4;

                    for (size_t c = 0; c < active_c; ++c) {
                        const size_t col = col0 + c;
                        quad_dst[4 * c + 0] = cactus_load_interleaved_int8_weight(b_interleaved.data(), K, col, k0 + 0);
                        quad_dst[4 * c + 1] = cactus_load_interleaved_int8_weight(b_interleaved.data(), K, col, k0 + 1);
                        quad_dst[4 * c + 2] = cactus_load_interleaved_int8_weight(b_interleaved.data(), K, col, k0 + 2);
                        quad_dst[4 * c + 3] = cactus_load_interleaved_int8_weight(b_interleaved.data(), K, col, k0 + 3);
                    }

                    for (size_t c = active_c; c < tile_rows; ++c) {
                        quad_dst[4 * c + 0] = 0;
                        quad_dst[4 * c + 1] = 0;
                        quad_dst[4 * c + 2] = 0;
                        quad_dst[4 * c + 3] = 0;
                    }
                }
            }
        });
}

static void cactus_pack_b_f16_from_bt(
    const __fp16* __restrict b_transposed,
    __fp16* __restrict b_packed,
    size_t K,
    size_t N,
    size_t tile_cols,
    size_t tile_pairs
) {
    const size_t k_pairs = (K + 1) / 2;
    const size_t col_blocks = (N + tile_cols - 1) / tile_cols;
    const size_t full_col_blocks = N / tile_cols;
    const size_t cb4_tiles = (full_col_blocks / 4) * 4;
    const size_t cb2_tiles = ((full_col_blocks - cb4_tiles) / 2) * 2;
    const size_t cb1_tiles = col_blocks - cb4_tiles - cb2_tiles;
    const size_t cb4_groups = cb4_tiles / 4;
    const size_t cb2_groups = cb2_tiles / 2;

    const size_t off_cb4 = 0;
    const size_t off_cb2 = off_cb4 + cb4_groups * k_pairs * 4 * tile_pairs;
    const size_t off_cb1 = off_cb2 + cb2_groups * k_pairs * 2 * tile_pairs;

    const bool even_k = ((K & 1u) == 0);

    CactusThreading::parallel_for(cb4_groups * k_pairs, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
        [=](size_t start, size_t end) {
            for (size_t idx = start; idx < end; ++idx) {
                const size_t g4 = idx / k_pairs;
                const size_t kp = idx % k_pairs;
                const size_t col0 = g4 * 4 * tile_cols;

                const size_t k0 = kp * 2;
                const size_t k1 = k0 + 1;
                __fp16* dst = b_packed + off_cb4 + (g4 * k_pairs + kp) * (4 * tile_pairs);

                for (size_t t = 0; t < 4; ++t) {
                    __fp16* dst_t = dst + t * tile_pairs;
                    const size_t col_t = col0 + t * tile_cols;
                    const __fp16* src_col = b_transposed + col_t * K + k0;
                    if (even_k) {
                        CACTUS_UNROLL4
                        for (size_t c = 0; c < tile_cols; ++c) {
                            dst_t[2 * c] = src_col[0];
                            dst_t[2 * c + 1] = src_col[1];
                            src_col += K;
                        }
                    } else {
                        const bool has_k1 = (k1 < K);
                        CACTUS_UNROLL4
                        for (size_t c = 0; c < tile_cols; ++c) {
                            dst_t[2 * c] = src_col[0];
                            dst_t[2 * c + 1] = has_k1 ? src_col[1] : static_cast<__fp16>(0);
                            src_col += K;
                        }
                    }
                }
            }
        });

    CactusThreading::parallel_for(cb2_groups * k_pairs, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
        [=](size_t start, size_t end) {
            for (size_t idx = start; idx < end; ++idx) {
                const size_t g2 = idx / k_pairs;
                const size_t kp = idx % k_pairs;
                const size_t col0 = cb4_tiles * tile_cols + g2 * 2 * tile_cols;

                const size_t k0 = kp * 2;
                const size_t k1 = k0 + 1;
                __fp16* dst = b_packed + off_cb2 + (g2 * k_pairs + kp) * (2 * tile_pairs);

                for (size_t t = 0; t < 2; ++t) {
                    __fp16* dst_t = dst + t * tile_pairs;
                    const size_t col_t = col0 + t * tile_cols;
                    const __fp16* src_col = b_transposed + col_t * K + k0;
                    if (even_k) {
                        CACTUS_UNROLL4
                        for (size_t c = 0; c < tile_cols; ++c) {
                            dst_t[2 * c] = src_col[0];
                            dst_t[2 * c + 1] = src_col[1];
                            src_col += K;
                        }
                    } else {
                        const bool has_k1 = (k1 < K);
                        CACTUS_UNROLL4
                        for (size_t c = 0; c < tile_cols; ++c) {
                            dst_t[2 * c] = src_col[0];
                            dst_t[2 * c + 1] = has_k1 ? src_col[1] : static_cast<__fp16>(0);
                            src_col += K;
                        }
                    }
                }
            }
        });

    CactusThreading::parallel_for(cb1_tiles * k_pairs, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
        [=](size_t start, size_t end) {
            for (size_t idx = start; idx < end; ++idx) {
                const size_t g1 = idx / k_pairs;
                const size_t kp = idx % k_pairs;
                const size_t cb = cb4_tiles + cb2_tiles + g1;
                const size_t col0 = cb * tile_cols;
                const size_t active_c = (col0 < N) ? std::min(tile_cols, N - col0) : 0;

                const size_t k0 = kp * 2;
                const size_t k1 = k0 + 1;
                __fp16* dst = b_packed + off_cb1 + (g1 * k_pairs + kp) * tile_pairs;
                const __fp16* src_col = b_transposed + col0 * K + k0;
                if (even_k) {
                    CACTUS_UNROLL4
                    for (size_t c = 0; c < active_c; ++c) {
                        dst[2 * c] = src_col[0];
                        dst[2 * c + 1] = src_col[1];
                        src_col += K;
                    }
                } else {
                    const bool has_k1 = (k1 < K);
                    CACTUS_UNROLL4
                    for (size_t c = 0; c < active_c; ++c) {
                        dst[2 * c] = src_col[0];
                        dst[2 * c + 1] = has_k1 ? src_col[1] : static_cast<__fp16>(0);
                        src_col += K;
                    }
                }
                CACTUS_UNROLL4
                for (size_t c = active_c; c < tile_cols; ++c) {
                    dst[2 * c] = static_cast<__fp16>(0);
                    dst[2 * c + 1] = static_cast<__fp16>(0);
                }
            }
        });
}

static void cactus_matmul_f16_sme2_worker(
    const __fp16* a_packed,
    const __fp16* b_packed,
    __fp16* c,
    size_t M,
    size_t K,
    size_t N,
    size_t start_row,
    size_t end_row,
    size_t tile_rows,
    size_t tile_pairs
) __arm_streaming __arm_inout("za") {
    if (start_row >= end_row) return;
    (void)M;

    const size_t k_pairs = (K + 1) / 2;
    const size_t col_blocks = (N + tile_rows - 1) / tile_rows;
    const size_t full_col_blocks = N / tile_rows;
    const size_t cb4_tiles = (full_col_blocks / 4) * 4;
    const size_t cb2_tiles = ((full_col_blocks - cb4_tiles) / 2) * 2;
    const size_t cb4_groups = cb4_tiles / 4;
    const size_t cb2_groups = cb2_tiles / 2;
    const size_t cb1_off = cb4_groups * k_pairs * 4 * tile_pairs + cb2_groups * k_pairs * 2 * tile_pairs;
    const size_t a_row_block_stride = k_pairs * tile_pairs;

    const svcount_t pNh_full_c = svptrue_c16();
    const svbool_t pNh_full = svptrue_b16();
    const svbool_t pN32_full = svwhilelt_b32(static_cast<uint64_t>(0), static_cast<uint64_t>(tile_rows));
    const svcount_t pOut16_full_c = svwhilelt_c16(static_cast<uint64_t>(0), static_cast<uint64_t>(tile_rows), 2);
    const svfloat32_t z0_f32 = svdup_n_f32(0.0f);
#define CACTUS_STORE_FULL_TILE_ROW(dst_idx, out32_vec)                                   \
    do {                                                                                  \
        svfloat16_t out16_local = svcvt_f16_f32_z(pN32_full, (out32_vec));               \
        out16_local = svuzp1_f16(out16_local, out16_local);                              \
        svst1_f16_x2(pOut16_full_c, &c[(dst_idx)], svcreate2(out16_local, out16_local)); \
    } while (0)

    for (size_t row = start_row; row < end_row; row += tile_rows) {
        const size_t rb = row / tile_rows;
        const size_t active_r = std::min(tile_rows, end_row - row);
        const bool full_r = (active_r == tile_rows);
        const svbool_t pMh = full_r
            ? svptrue_b16()
            : svwhilelt_b16(static_cast<uint64_t>(0), static_cast<uint64_t>(active_r * 2));

        size_t cb = 0;

        for (; cb < cb4_tiles; cb += 4) {
            svzero_za();
            const size_t g4 = cb / 4;
            const __fp16* b_g4_base = b_packed + g4 * k_pairs * 4 * tile_pairs;

            size_t kp = 0;
            for (; kp + 3 < k_pairs; kp += 4) {
                const __fp16* a_ptr0 = a_packed + rb * a_row_block_stride + kp * tile_pairs;
                const __fp16* b_ptr0 = b_g4_base + kp * (4 * tile_pairs);
                const __fp16* a_ptr1 = a_ptr0 + tile_pairs;
                const __fp16* b_ptr1 = b_ptr0 + 4 * tile_pairs;
                const __fp16* a_ptr2 = a_ptr1 + tile_pairs;
                const __fp16* b_ptr2 = b_ptr1 + 4 * tile_pairs;
                const __fp16* a_ptr3 = a_ptr2 + tile_pairs;
                const __fp16* b_ptr3 = b_ptr2 + 4 * tile_pairs;

                const svfloat16_t zA0 = svld1(pMh, a_ptr0);
                const svfloat16x4_t zB40 = svld1_f16_x4(pNh_full_c, b_ptr0);
                svmopa_za32_f16_m(0, pMh, pNh_full, zA0, svget4(zB40, 0));
                svmopa_za32_f16_m(1, pMh, pNh_full, zA0, svget4(zB40, 1));
                svmopa_za32_f16_m(2, pMh, pNh_full, zA0, svget4(zB40, 2));
                svmopa_za32_f16_m(3, pMh, pNh_full, zA0, svget4(zB40, 3));

                const svfloat16_t zA1 = svld1(pMh, a_ptr1);
                const svfloat16x4_t zB41 = svld1_f16_x4(pNh_full_c, b_ptr1);
                svmopa_za32_f16_m(0, pMh, pNh_full, zA1, svget4(zB41, 0));
                svmopa_za32_f16_m(1, pMh, pNh_full, zA1, svget4(zB41, 1));
                svmopa_za32_f16_m(2, pMh, pNh_full, zA1, svget4(zB41, 2));
                svmopa_za32_f16_m(3, pMh, pNh_full, zA1, svget4(zB41, 3));

                const svfloat16_t zA2 = svld1(pMh, a_ptr2);
                const svfloat16x4_t zB42 = svld1_f16_x4(pNh_full_c, b_ptr2);
                svmopa_za32_f16_m(0, pMh, pNh_full, zA2, svget4(zB42, 0));
                svmopa_za32_f16_m(1, pMh, pNh_full, zA2, svget4(zB42, 1));
                svmopa_za32_f16_m(2, pMh, pNh_full, zA2, svget4(zB42, 2));
                svmopa_za32_f16_m(3, pMh, pNh_full, zA2, svget4(zB42, 3));

                const svfloat16_t zA3 = svld1(pMh, a_ptr3);
                const svfloat16x4_t zB43 = svld1_f16_x4(pNh_full_c, b_ptr3);
                svmopa_za32_f16_m(0, pMh, pNh_full, zA3, svget4(zB43, 0));
                svmopa_za32_f16_m(1, pMh, pNh_full, zA3, svget4(zB43, 1));
                svmopa_za32_f16_m(2, pMh, pNh_full, zA3, svget4(zB43, 2));
                svmopa_za32_f16_m(3, pMh, pNh_full, zA3, svget4(zB43, 3));
            }

            for (; kp < k_pairs; ++kp) {
                const __fp16* a_ptr = a_packed + rb * a_row_block_stride + kp * tile_pairs;
                const __fp16* b_ptr = b_g4_base + kp * (4 * tile_pairs);

                const svfloat16_t zA = svld1(pMh, a_ptr);
                const svfloat16x4_t zB4 = svld1_f16_x4(pNh_full_c, b_ptr);
                svmopa_za32_f16_m(0, pMh, pNh_full, zA, svget4(zB4, 0));
                svmopa_za32_f16_m(1, pMh, pNh_full, zA, svget4(zB4, 1));
                svmopa_za32_f16_m(2, pMh, pNh_full, zA, svget4(zB4, 2));
                svmopa_za32_f16_m(3, pMh, pNh_full, zA, svget4(zB4, 3));
            }

            const size_t col = cb * tile_rows;
            if (full_r) {
                for (size_t trow = 0; trow < tile_rows; trow += 4) {
                    const svfloat32x4_t zT0 = svread_hor_za32_f32_vg4(0, static_cast<uint32_t>(trow));
                    const svfloat32x4_t zT1 = svread_hor_za32_f32_vg4(1, static_cast<uint32_t>(trow));
                    const svfloat32x4_t zT2 = svread_hor_za32_f32_vg4(2, static_cast<uint32_t>(trow));
                    const svfloat32x4_t zT3 = svread_hor_za32_f32_vg4(3, static_cast<uint32_t>(trow));
                    const size_t dst0 = (row + trow + 0) * N + col;
                    CACTUS_STORE_FULL_TILE_ROW(dst0, svget4(zT0, 0));
                    CACTUS_STORE_FULL_TILE_ROW(dst0 + tile_rows, svget4(zT1, 0));
                    CACTUS_STORE_FULL_TILE_ROW(dst0 + 2 * tile_rows, svget4(zT2, 0));
                    CACTUS_STORE_FULL_TILE_ROW(dst0 + 3 * tile_rows, svget4(zT3, 0));
                    const size_t dst1 = (row + trow + 1) * N + col;
                    CACTUS_STORE_FULL_TILE_ROW(dst1, svget4(zT0, 1));
                    CACTUS_STORE_FULL_TILE_ROW(dst1 + tile_rows, svget4(zT1, 1));
                    CACTUS_STORE_FULL_TILE_ROW(dst1 + 2 * tile_rows, svget4(zT2, 1));
                    CACTUS_STORE_FULL_TILE_ROW(dst1 + 3 * tile_rows, svget4(zT3, 1));
                    const size_t dst2 = (row + trow + 2) * N + col;
                    CACTUS_STORE_FULL_TILE_ROW(dst2, svget4(zT0, 2));
                    CACTUS_STORE_FULL_TILE_ROW(dst2 + tile_rows, svget4(zT1, 2));
                    CACTUS_STORE_FULL_TILE_ROW(dst2 + 2 * tile_rows, svget4(zT2, 2));
                    CACTUS_STORE_FULL_TILE_ROW(dst2 + 3 * tile_rows, svget4(zT3, 2));
                    const size_t dst3 = (row + trow + 3) * N + col;
                    CACTUS_STORE_FULL_TILE_ROW(dst3, svget4(zT0, 3));
                    CACTUS_STORE_FULL_TILE_ROW(dst3 + tile_rows, svget4(zT1, 3));
                    CACTUS_STORE_FULL_TILE_ROW(dst3 + 2 * tile_rows, svget4(zT2, 3));
                    CACTUS_STORE_FULL_TILE_ROW(dst3 + 3 * tile_rows, svget4(zT3, 3));
                }
            } else {
                size_t trow = 0;
                for (; trow + 3 < active_r; trow += 4) {
                    const svfloat32x4_t zT0 = svread_hor_za32_f32_vg4(0, static_cast<uint32_t>(trow));
                    const svfloat32x4_t zT1 = svread_hor_za32_f32_vg4(1, static_cast<uint32_t>(trow));
                    const svfloat32x4_t zT2 = svread_hor_za32_f32_vg4(2, static_cast<uint32_t>(trow));
                    const svfloat32x4_t zT3 = svread_hor_za32_f32_vg4(3, static_cast<uint32_t>(trow));
                    const size_t dst0 = (row + trow + 0) * N + col;
                    CACTUS_STORE_FULL_TILE_ROW(dst0, svget4(zT0, 0));
                    CACTUS_STORE_FULL_TILE_ROW(dst0 + tile_rows, svget4(zT1, 0));
                    CACTUS_STORE_FULL_TILE_ROW(dst0 + 2 * tile_rows, svget4(zT2, 0));
                    CACTUS_STORE_FULL_TILE_ROW(dst0 + 3 * tile_rows, svget4(zT3, 0));
                    const size_t dst1 = (row + trow + 1) * N + col;
                    CACTUS_STORE_FULL_TILE_ROW(dst1, svget4(zT0, 1));
                    CACTUS_STORE_FULL_TILE_ROW(dst1 + tile_rows, svget4(zT1, 1));
                    CACTUS_STORE_FULL_TILE_ROW(dst1 + 2 * tile_rows, svget4(zT2, 1));
                    CACTUS_STORE_FULL_TILE_ROW(dst1 + 3 * tile_rows, svget4(zT3, 1));
                    const size_t dst2 = (row + trow + 2) * N + col;
                    CACTUS_STORE_FULL_TILE_ROW(dst2, svget4(zT0, 2));
                    CACTUS_STORE_FULL_TILE_ROW(dst2 + tile_rows, svget4(zT1, 2));
                    CACTUS_STORE_FULL_TILE_ROW(dst2 + 2 * tile_rows, svget4(zT2, 2));
                    CACTUS_STORE_FULL_TILE_ROW(dst2 + 3 * tile_rows, svget4(zT3, 2));
                    const size_t dst3 = (row + trow + 3) * N + col;
                    CACTUS_STORE_FULL_TILE_ROW(dst3, svget4(zT0, 3));
                    CACTUS_STORE_FULL_TILE_ROW(dst3 + tile_rows, svget4(zT1, 3));
                    CACTUS_STORE_FULL_TILE_ROW(dst3 + 2 * tile_rows, svget4(zT2, 3));
                    CACTUS_STORE_FULL_TILE_ROW(dst3 + 3 * tile_rows, svget4(zT3, 3));
                }
                for (; trow + 1 < active_r; trow += 2) {
                    const svfloat32x2_t zT0 = svread_hor_za32_f32_vg2(0, static_cast<uint32_t>(trow));
                    const svfloat32x2_t zT1 = svread_hor_za32_f32_vg2(1, static_cast<uint32_t>(trow));
                    const svfloat32x2_t zT2 = svread_hor_za32_f32_vg2(2, static_cast<uint32_t>(trow));
                    const svfloat32x2_t zT3 = svread_hor_za32_f32_vg2(3, static_cast<uint32_t>(trow));
                    const size_t dst0 = (row + trow + 0) * N + col;
                    CACTUS_STORE_FULL_TILE_ROW(dst0, svget2(zT0, 0));
                    CACTUS_STORE_FULL_TILE_ROW(dst0 + tile_rows, svget2(zT1, 0));
                    CACTUS_STORE_FULL_TILE_ROW(dst0 + 2 * tile_rows, svget2(zT2, 0));
                    CACTUS_STORE_FULL_TILE_ROW(dst0 + 3 * tile_rows, svget2(zT3, 0));
                    const size_t dst1 = (row + trow + 1) * N + col;
                    CACTUS_STORE_FULL_TILE_ROW(dst1, svget2(zT0, 1));
                    CACTUS_STORE_FULL_TILE_ROW(dst1 + tile_rows, svget2(zT1, 1));
                    CACTUS_STORE_FULL_TILE_ROW(dst1 + 2 * tile_rows, svget2(zT2, 1));
                    CACTUS_STORE_FULL_TILE_ROW(dst1 + 3 * tile_rows, svget2(zT3, 1));
                }
                for (; trow < active_r; ++trow) {
                    const size_t dst = (row + trow) * N + col;
                    CACTUS_STORE_FULL_TILE_ROW(dst, svread_hor_za32_f32_m(z0_f32, pN32_full, 0, static_cast<uint32_t>(trow)));
                    CACTUS_STORE_FULL_TILE_ROW(dst + tile_rows, svread_hor_za32_f32_m(z0_f32, pN32_full, 1, static_cast<uint32_t>(trow)));
                    CACTUS_STORE_FULL_TILE_ROW(dst + 2 * tile_rows, svread_hor_za32_f32_m(z0_f32, pN32_full, 2, static_cast<uint32_t>(trow)));
                    CACTUS_STORE_FULL_TILE_ROW(dst + 3 * tile_rows, svread_hor_za32_f32_m(z0_f32, pN32_full, 3, static_cast<uint32_t>(trow)));
                }
            }
        }

        for (; cb < cb4_tiles + cb2_tiles; cb += 2) {
            svzero_za();
            const size_t g2 = (cb - cb4_tiles) / 2;
            const __fp16* b_g2_base = b_packed + cb4_groups * k_pairs * 4 * tile_pairs + g2 * k_pairs * 2 * tile_pairs;

            size_t kp = 0;
            for (; kp < k_pairs; ++kp) {
                const __fp16* a_ptr = a_packed + rb * a_row_block_stride + kp * tile_pairs;
                const __fp16* b_ptr = b_g2_base + kp * (2 * tile_pairs);

                const svfloat16_t zA = svld1(pMh, a_ptr);
                const svfloat16x2_t zB2 = svld1_f16_x2(pNh_full_c, b_ptr);
                svmopa_za32_f16_m(0, pMh, pNh_full, zA, svget2(zB2, 0));
                svmopa_za32_f16_m(1, pMh, pNh_full, zA, svget2(zB2, 1));
            }

            const size_t col = cb * tile_rows;
            if (full_r) {
                for (size_t trow = 0; trow < tile_rows; trow += 4) {
                    const svfloat32x4_t zT0 = svread_hor_za32_f32_vg4(0, static_cast<uint32_t>(trow));
                    const svfloat32x4_t zT1 = svread_hor_za32_f32_vg4(1, static_cast<uint32_t>(trow));
                    const size_t dst0 = (row + trow + 0) * N + col;
                    CACTUS_STORE_FULL_TILE_ROW(dst0, svget4(zT0, 0));
                    CACTUS_STORE_FULL_TILE_ROW(dst0 + tile_rows, svget4(zT1, 0));
                    const size_t dst1 = (row + trow + 1) * N + col;
                    CACTUS_STORE_FULL_TILE_ROW(dst1, svget4(zT0, 1));
                    CACTUS_STORE_FULL_TILE_ROW(dst1 + tile_rows, svget4(zT1, 1));
                    const size_t dst2 = (row + trow + 2) * N + col;
                    CACTUS_STORE_FULL_TILE_ROW(dst2, svget4(zT0, 2));
                    CACTUS_STORE_FULL_TILE_ROW(dst2 + tile_rows, svget4(zT1, 2));
                    const size_t dst3 = (row + trow + 3) * N + col;
                    CACTUS_STORE_FULL_TILE_ROW(dst3, svget4(zT0, 3));
                    CACTUS_STORE_FULL_TILE_ROW(dst3 + tile_rows, svget4(zT1, 3));
                }
            } else {
                size_t trow = 0;
                for (; trow + 3 < active_r; trow += 4) {
                    const svfloat32x4_t zT0 = svread_hor_za32_f32_vg4(0, static_cast<uint32_t>(trow));
                    const svfloat32x4_t zT1 = svread_hor_za32_f32_vg4(1, static_cast<uint32_t>(trow));
                    const size_t dst0 = (row + trow + 0) * N + col;
                    CACTUS_STORE_FULL_TILE_ROW(dst0, svget4(zT0, 0));
                    CACTUS_STORE_FULL_TILE_ROW(dst0 + tile_rows, svget4(zT1, 0));
                    const size_t dst1 = (row + trow + 1) * N + col;
                    CACTUS_STORE_FULL_TILE_ROW(dst1, svget4(zT0, 1));
                    CACTUS_STORE_FULL_TILE_ROW(dst1 + tile_rows, svget4(zT1, 1));
                    const size_t dst2 = (row + trow + 2) * N + col;
                    CACTUS_STORE_FULL_TILE_ROW(dst2, svget4(zT0, 2));
                    CACTUS_STORE_FULL_TILE_ROW(dst2 + tile_rows, svget4(zT1, 2));
                    const size_t dst3 = (row + trow + 3) * N + col;
                    CACTUS_STORE_FULL_TILE_ROW(dst3, svget4(zT0, 3));
                    CACTUS_STORE_FULL_TILE_ROW(dst3 + tile_rows, svget4(zT1, 3));
                }
                for (; trow + 1 < active_r; trow += 2) {
                    const svfloat32x2_t zT0 = svread_hor_za32_f32_vg2(0, static_cast<uint32_t>(trow));
                    const svfloat32x2_t zT1 = svread_hor_za32_f32_vg2(1, static_cast<uint32_t>(trow));
                    const size_t dst0 = (row + trow + 0) * N + col;
                    CACTUS_STORE_FULL_TILE_ROW(dst0, svget2(zT0, 0));
                    CACTUS_STORE_FULL_TILE_ROW(dst0 + tile_rows, svget2(zT1, 0));
                    const size_t dst1 = (row + trow + 1) * N + col;
                    CACTUS_STORE_FULL_TILE_ROW(dst1, svget2(zT0, 1));
                    CACTUS_STORE_FULL_TILE_ROW(dst1 + tile_rows, svget2(zT1, 1));
                }
                for (; trow < active_r; ++trow) {
                    const size_t dst = (row + trow) * N + col;
                    CACTUS_STORE_FULL_TILE_ROW(dst, svread_hor_za32_f32_m(z0_f32, pN32_full, 0, static_cast<uint32_t>(trow)));
                    CACTUS_STORE_FULL_TILE_ROW(dst + tile_rows, svread_hor_za32_f32_m(z0_f32, pN32_full, 1, static_cast<uint32_t>(trow)));
                }
            }
        }

        for (; cb < col_blocks; ++cb) {
            const size_t col = cb * tile_rows;
            const size_t active_c = std::min(tile_rows, N - col);
            const svbool_t pNh = svwhilelt_b16(static_cast<uint64_t>(0), static_cast<uint64_t>(active_c * 2));
            const svbool_t pN32 = svwhilelt_b32(static_cast<uint64_t>(0), static_cast<uint64_t>(active_c));
            const svbool_t pOut16 = svwhilelt_b16(static_cast<uint64_t>(0), static_cast<uint64_t>(active_c));
            const size_t g1 = cb - cb4_tiles - cb2_tiles;
            const __fp16* b_g1_base = b_packed + cb1_off + g1 * k_pairs * tile_pairs;

            svzero_za();

            size_t kp = 0;
            for (; kp + 1 < k_pairs; kp += 2) {
                const __fp16* a_ptr0 = a_packed + rb * a_row_block_stride + kp * tile_pairs;
                const __fp16* b_ptr0 = b_g1_base + kp * tile_pairs;
                const __fp16* a_ptr1 = a_ptr0 + tile_pairs;
                const __fp16* b_ptr1 = b_ptr0 + tile_pairs;

                const svfloat16_t zA0 = svld1(pMh, a_ptr0);
                const svfloat16_t zB0 = svld1(pNh, b_ptr0);
                svmopa_za32_f16_m(0, pMh, pNh, zA0, zB0);

                const svfloat16_t zA1 = svld1(pMh, a_ptr1);
                const svfloat16_t zB1 = svld1(pNh, b_ptr1);
                svmopa_za32_f16_m(0, pMh, pNh, zA1, zB1);
            }

            for (; kp < k_pairs; ++kp) {
                const __fp16* a_ptr = a_packed + rb * a_row_block_stride + kp * tile_pairs;
                const __fp16* b_ptr = b_g1_base + kp * tile_pairs;

                const svfloat16_t zA = svld1(pMh, a_ptr);
                const svfloat16_t zB = svld1(pNh, b_ptr);
                svmopa_za32_f16_m(0, pMh, pNh, zA, zB);
            }

            for (size_t trow = 0; trow < active_r; ++trow) {
                svfloat32_t out32 = svread_hor_za32_f32_m(z0_f32, pN32, 0, static_cast<uint32_t>(trow));
                svfloat16_t out16 = svcvt_f16_f32_z(pN32, out32);
                out16 = svuzp1_f16(out16, out16);
                svst1(pOut16, &c[(row + trow) * N + col], out16);
            }
        }
    }
#undef CACTUS_STORE_FULL_TILE_ROW
}

__arm_new("za") __arm_locally_streaming
static void cactus_matmul_f16_sme2_thread_entry(
    const __fp16* a,
    __fp16* a_packed,
    const __fp16* b_packed,
    __fp16* c,
    size_t M,
    size_t K,
    size_t N,
    size_t row_block_size,
    size_t start_block,
    size_t end_block
) {
    const size_t tile_rows = svcntsw();
    const size_t tile_pairs = svcnth();
    const size_t k_pairs = (K + 1) / 2;

    for (size_t block_idx = start_block; block_idx < end_block; ++block_idx) {
        const size_t start_row = block_idx * row_block_size;
        const size_t end_row = std::min(start_row + row_block_size, M);

        for (size_t row = start_row; row < end_row; row += tile_rows) {
            const size_t rb = row / tile_rows;
            const size_t active_r = std::min(tile_rows, end_row - row);
            cactus_pack_a_f16_row_block(
                a,
                a_packed,
                K,
                rb,
                row,
                active_r,
                tile_rows,
                tile_pairs,
                k_pairs
            );
        }

        cactus_matmul_f16_sme2_worker(
            a_packed,
            b_packed,
            c,
            M,
            K,
            N,
            start_row,
            end_row,
            tile_rows,
            tile_pairs
        );
    }
}

__arm_new("za") __arm_locally_streaming
void cactus_matmul_f16_sme2_caller(
    const __fp16* a,
    const __fp16* b_transposed,
    __fp16* c,
    size_t M,
    size_t K,
    size_t N
) {
    const size_t tile_rows = svcntsw();
    const size_t tile_pairs = svcnth();
    constexpr size_t SME2_TILES_PER_THREAD = 3;

    const size_t row_blocks = (M + tile_rows - 1) / tile_rows;
    const size_t k_pairs = (K + 1) / 2;
    const size_t col_blocks = (N + tile_rows - 1) / tile_rows;
    const size_t packed_b_elements = k_pairs * col_blocks * tile_pairs;

    std::unique_ptr<__fp16[]> a_packed(new __fp16[row_blocks * k_pairs * tile_pairs]);
    __fp16* a_packed_ptr = a_packed.get();
    const Sme2PackedBKey cache_key{b_transposed, K, N, tile_rows, tile_pairs};
    const uint64_t rhs_fingerprint = cactus_fp16_sample_fingerprint(b_transposed, K * N);
    auto b_packed = Sme2PackedBCache::instance().get_or_create(
        cache_key,
        packed_b_elements,
        rhs_fingerprint,
        [&]( __fp16* packed_dst) {
            cactus_pack_b_f16_from_bt(b_transposed, packed_dst, K, N, tile_rows, tile_pairs);
        });
    const __fp16* b_packed_ptr = b_packed->data();

    const size_t row_block_size = SME2_TILES_PER_THREAD * tile_rows;
    const size_t num_row_blocks = (M + row_block_size - 1) / row_block_size;

    auto& pool = CactusThreading::get_thread_pool();
    const size_t num_workers = std::min(pool.num_workers(), num_row_blocks);
    if (num_workers <= 1) {
        cactus_matmul_f16_sme2_thread_entry(
            a,
            a_packed_ptr,
            b_packed_ptr,
            c,
            M,
            K,
            N,
            row_block_size,
            0,
            num_row_blocks
        );
        return;
    }

    std::atomic<size_t> next_block{0};
    pool.enqueue_n_threads(num_workers, num_workers, [&](size_t, size_t) {
        while (true) {
            const size_t block_idx = next_block.fetch_add(1, std::memory_order_relaxed);
            if (block_idx >= num_row_blocks) break;
            cactus_matmul_f16_sme2_thread_entry(
                a,
                a_packed_ptr,
                b_packed_ptr,
                c,
                M,
                K,
                N,
                row_block_size,
                block_idx,
                block_idx + 1
            );
        }
    });
    pool.wait_all();
}

static void cactus_matmul_int4_sme2_worker(
    const int8_t* a_packed,
    const int8_t* b_packed,
    const float* b_scales_packed,
    const float* a_scales,
    float* tile_accum,
    __fp16* c,
    size_t K,
    size_t N,
    size_t group_size,
    size_t start_row,
    size_t end_row,
    size_t tile_rows,
    size_t tile_bytes
) __arm_streaming __arm_inout("za") {
    if (start_row >= end_row) return;

    const size_t num_groups = K / group_size;
    const size_t group_quads = group_size / 4;
    const size_t col_blocks = (N + tile_rows - 1) / tile_rows;
    const size_t a_row_block_stride = (K / 4) * tile_bytes;

    const svint32_t z0_s32 = svdup_n_s32(0);

    for (size_t row = start_row; row < end_row; row += tile_rows) {
        const size_t rb = row / tile_rows;
        const size_t active_r = std::min(tile_rows, end_row - row);
        const svbool_t pMh = (active_r == tile_rows)
            ? svptrue_b8()
            : svwhilelt_b8(static_cast<uint64_t>(0), static_cast<uint64_t>(active_r * 4));

        for (size_t cb = 0; cb < col_blocks; ++cb) {
            const size_t col = cb * tile_rows;
            const size_t active_c = std::min(tile_rows, N - col);
            const svbool_t pNh = (active_c == tile_rows)
                ? svptrue_b8()
                : svwhilelt_b8(static_cast<uint64_t>(0), static_cast<uint64_t>(active_c * 4));
            const svbool_t pN32 = (active_c == tile_rows)
                ? svptrue_b32()
                : svwhilelt_b32(static_cast<uint64_t>(0), static_cast<uint64_t>(active_c));
            // Each FP32 accumulator lane becomes exactly one FP16 output value after
            // compaction. The store predicate must track the logical output columns,
            // not the full FP16 vector width, or full-tile stores will overrun rows.
            const svbool_t pOut16 = svwhilelt_b16(static_cast<uint64_t>(0), static_cast<uint64_t>(active_c));

            std::fill(tile_accum, tile_accum + tile_rows * tile_rows, 0.0f);

            for (size_t g = 0; g < num_groups; ++g) {
                const int8_t* b_group = b_packed + (cb * num_groups + g) * group_quads * tile_bytes;
                const float* scale_ptr = b_scales_packed + (cb * num_groups + g) * tile_rows;
                const svfloat32_t b_scale = svld1(pN32, scale_ptr);

                svzero_za();

                for (size_t kq = 0; kq < group_quads; ++kq) {
                    const int8_t* a_ptr = a_packed + rb * a_row_block_stride + (g * group_quads + kq) * tile_bytes;
                    const int8_t* b_ptr = b_group + kq * tile_bytes;
                    const svint8_t zA = svld1_s8(pMh, a_ptr);
                    const svint8_t zB = svld1_s8(pNh, b_ptr);
                    svmopa_za32_s8_m(0, pMh, pNh, zA, zB);
                }

                for (size_t r = 0; r < active_r; ++r) {
                    float* acc_ptr = tile_accum + r * tile_rows;
                    svfloat32_t acc = svld1(pN32, acc_ptr);
                    const svint32_t group_sum = svread_hor_za32_s32_m(z0_s32, pN32, 0, static_cast<uint32_t>(r));
                    svfloat32_t scaled = svcvt_f32_s32_x(pN32, group_sum);
                    scaled = svmul_f32_x(pN32, scaled, b_scale);
                    scaled = svmul_n_f32_x(pN32, scaled, a_scales[row + r]);
                    acc = svadd_f32_x(pN32, acc, scaled);
                    svst1(pN32, acc_ptr, acc);
                }
            }

            for (size_t r = 0; r < active_r; ++r) {
                const float* acc_ptr = tile_accum + r * tile_rows;
                svfloat32_t acc = svld1(pN32, acc_ptr);
                svfloat16_t out16 = svcvt_f16_f32_z(pN32, acc);
                out16 = svuzp1_f16(out16, out16);
                svst1(pOut16, &c[(row + r) * N + col], out16);
            }
        }
    }
}

__arm_new("za") __arm_locally_streaming
static void cactus_matmul_int4_sme2_thread_entry(
    const int8_t* a,
    const float* a_scales,
    int8_t* a_packed,
    const int8_t* b_packed,
    const float* b_scales_packed,
    __fp16* c,
    size_t M,
    size_t K,
    size_t N,
    size_t group_size,
    size_t row_block_size,
    size_t start_block,
    size_t end_block
) {
    const size_t tile_rows = svcntsw();
    const size_t tile_bytes = svcntsb();
    const size_t k_quads = K / 4;
    std::unique_ptr<float[]> tile_accum(new float[tile_rows * tile_rows]);

    for (size_t block_idx = start_block; block_idx < end_block; ++block_idx) {
        const size_t start_row = block_idx * row_block_size;
        const size_t end_row = std::min(start_row + row_block_size, M);

        for (size_t row = start_row; row < end_row; row += tile_rows) {
            const size_t rb = row / tile_rows;
            const size_t active_r = std::min(tile_rows, end_row - row);
            cactus_pack_a_int8_row_block(
                a,
                a_packed,
                K,
                rb,
                row,
                active_r,
                tile_rows,
                tile_bytes,
                k_quads
            );
        }

        cactus_matmul_int4_sme2_worker(
            a_packed,
            b_packed,
            b_scales_packed,
            a_scales,
            tile_accum.get(),
            c,
            K,
            N,
            group_size,
            start_row,
            end_row,
            tile_rows,
            tile_bytes
        );
    }
}

__arm_new("za") __arm_locally_streaming
void cactus_matmul_int4_sme2_caller(
    const int8_t* a,
    const float* a_scales,
    const int8_t* b_packed,
    const __fp16* b_scales,
    __fp16* c,
    size_t M,
    size_t K,
    size_t N,
    size_t group_size
) {
    if (M == 0 || K == 0 || N == 0) return;
    if ((group_size % 4) != 0 || group_size == 0 || (K % group_size) != 0 || (K % 4) != 0) return;

    const size_t tile_rows = svcntsw();
    const size_t tile_bytes = svcntsb();
    constexpr size_t SME2_TILES_PER_THREAD = 3;

    const size_t row_blocks = (M + tile_rows - 1) / tile_rows;
    const size_t col_blocks = (N + tile_rows - 1) / tile_rows;
    const size_t num_groups = K / group_size;
    const size_t group_quads = group_size / 4;
    const size_t k_quads = K / 4;

    std::unique_ptr<int8_t[]> a_packed(new int8_t[row_blocks * k_quads * tile_bytes]);
    int8_t* a_packed_ptr = a_packed.get();

    const size_t packed_weight_elements = col_blocks * num_groups * group_quads * tile_bytes;
    const size_t packed_scale_elements = col_blocks * num_groups * tile_rows;
    const size_t stored_n = ((N + 3) / 4) * 4;
    const size_t packed_bytes = stored_n * K / 2;

    const Sme2PackedInt4Key cache_key{b_packed, b_scales, K, N, group_size, tile_rows, tile_bytes};
    uint64_t fingerprint = cactus_u8_sample_fingerprint(reinterpret_cast<const uint8_t*>(b_packed), packed_bytes);
    fingerprint = cactus_mix_fingerprint(fingerprint, cactus_fp16_sample_fingerprint(b_scales, stored_n * num_groups));

    auto packed = Sme2PackedInt4Cache::instance().get_or_create(
        cache_key,
        packed_weight_elements,
        packed_scale_elements,
        fingerprint,
        [&](Sme2PackedInt4Panels& packed_panels) {
            cactus_pack_b_int4_for_sme2(
                b_packed,
                b_scales,
                packed_panels,
                K,
                N,
                group_size,
                tile_rows,
                tile_bytes
            );
        });

    const size_t row_block_size = SME2_TILES_PER_THREAD * tile_rows;
    const size_t num_row_blocks = (M + row_block_size - 1) / row_block_size;

    auto& pool = CactusThreading::get_thread_pool();
    const size_t num_workers = std::min(pool.num_workers(), num_row_blocks);
    if (num_workers <= 1) {
        cactus_matmul_int4_sme2_thread_entry(
            a,
            a_scales,
            a_packed_ptr,
            packed->weights.data(),
            packed->scales.data(),
            c,
            M,
            K,
            N,
            group_size,
            row_block_size,
            0,
            num_row_blocks
        );
        return;
    }

    std::atomic<size_t> next_block{0};
    pool.enqueue_n_threads(num_workers, num_workers, [&](size_t, size_t) {
        while (true) {
            const size_t block_idx = next_block.fetch_add(1, std::memory_order_relaxed);
            if (block_idx >= num_row_blocks) break;
            cactus_matmul_int4_sme2_thread_entry(
                a,
                a_scales,
                a_packed_ptr,
                packed->weights.data(),
                packed->scales.data(),
                c,
                M,
                K,
                N,
                group_size,
                row_block_size,
                block_idx,
                block_idx + 1
            );
        }
    });
    pool.wait_all();
}

#undef CACTUS_UNROLL4
