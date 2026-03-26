#include "test_utils.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <random>
#include <algorithm>
#include <limits>

namespace {

std::vector<int8_t> interleave_int8_rows_4(const std::vector<int8_t>& rowmajor, size_t rows, size_t cols) {
    const size_t block = 4;
    const size_t row_blocks = rows / block;
    const size_t col_blocks = cols / block;
    std::vector<int8_t> interleaved(rows * cols);
    for (size_t row_block = 0; row_block < row_blocks; ++row_block) {
        for (size_t col_block = 0; col_block < col_blocks; ++col_block) {
            for (size_t lane = 0; lane < block; ++lane) {
                for (size_t col_lane = 0; col_lane < block; ++col_lane) {
                    size_t src_row = row_block * block + lane;
                    size_t src_col = col_block * block + col_lane;
                    size_t dst_idx = (row_block * col_blocks + col_block) * block * block
                                     + lane * block + col_lane;
                    interleaved[dst_idx] = rowmajor[src_row * cols + src_col];
                }
            }
        }
    }
    return interleaved;
}

std::vector<uint8_t> pack_ternary_2bit(const std::vector<int8_t>& signs) {
    std::vector<uint8_t> packed((signs.size() + 3) / 4, 0);
    for (size_t i = 0; i < signs.size(); ++i) {
        uint8_t code = 0;
        if (signs[i] > 0) code = 1;
        else if (signs[i] < 0) code = 2;
        packed[i / 4] |= static_cast<uint8_t>(code << ((i % 4) * 2));
    }
    return packed;
}

std::vector<int8_t> make_ternary_signs(size_t rows, size_t cols) {
    std::vector<int8_t> signs(rows * cols);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            const int selector = static_cast<int>((row * 7 + col * 5) % 3);
            signs[row * cols + col] = selector == 0 ? -1 : (selector == 1 ? 0 : 1);
        }
    }
    return signs;
}

std::vector<__fp16> ternary_reference_output(
    const std::vector<__fp16>& lhs_fp16,
    size_t M,
    size_t K,
    const std::vector<int8_t>& rhs_rowmajor,
    size_t N,
    TernaryScaleMode mode,
    const std::vector<__fp16>& scales) {
    std::vector<__fp16> transformed_lhs = lhs_fp16;
    std::vector<__fp16> rhs_row_scales(N, static_cast<__fp16>(1.0f));

    switch (mode) {
        case TernaryScaleMode::TENSOR:
            std::fill(rhs_row_scales.begin(), rhs_row_scales.end(), scales[0]);
            break;
        case TernaryScaleMode::ROW:
            rhs_row_scales = scales;
            break;
        case TernaryScaleMode::COLUMN:
            for (size_t row = 0; row < M; ++row) {
                for (size_t col = 0; col < K; ++col) {
                    transformed_lhs[row * K + col] = static_cast<__fp16>(
                        static_cast<float>(lhs_fp16[row * K + col]) * static_cast<float>(scales[col]));
                }
            }
            break;
        case TernaryScaleMode::NONE:
            break;
    }

    std::vector<int8_t> lhs_quantized(M * K);
    std::vector<float> lhs_scales(M, 1.0f);
    for (size_t row = 0; row < M; ++row) {
        float scale = cactus_fp16_max_abs(transformed_lhs.data() + row * K, K) / 127.0f;
        if (scale < 1e-10f) scale = 1e-10f;
        lhs_scales[row] = scale;
        cactus_fp16_to_int8(transformed_lhs.data() + row * K, lhs_quantized.data() + row * K, K, scale);
    }

    std::vector<__fp16> expected(M * N, static_cast<__fp16>(0.0f));
    for (size_t row = 0; row < M; ++row) {
        for (size_t out = 0; out < N; ++out) {
            int32_t dot = 0;
            for (size_t col = 0; col < K; ++col) {
                dot += static_cast<int32_t>(lhs_quantized[row * K + col]) *
                       static_cast<int32_t>(rhs_rowmajor[out * K + col]);
            }
            expected[row * N + out] = static_cast<__fp16>(
                static_cast<float>(dot) * lhs_scales[row] * static_cast<float>(rhs_row_scales[out]));
        }
    }

    return expected;
}

std::vector<float> make_test_rotation_matrix(size_t head_dim, uint64_t seed) {
    std::vector<float> rotation(head_dim * head_dim, 0.0f);
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    std::vector<float> col(head_dim);
    for (size_t j = 0; j < head_dim; ++j) {
        for (size_t i = 0; i < head_dim; ++i) {
            col[i] = normal(rng);
        }

        for (size_t k = 0; k < j; ++k) {
            float dot = 0.0f;
            for (size_t i = 0; i < head_dim; ++i) {
                dot += col[i] * rotation[i * head_dim + k];
            }
            for (size_t i = 0; i < head_dim; ++i) {
                col[i] -= dot * rotation[i * head_dim + k];
            }
        }

        float norm_sq = 0.0f;
        for (float v : col) norm_sq += v * v;
        float norm = std::sqrt(std::max(norm_sq, 1e-12f));
        float sign = (col[j] < 0.0f) ? -1.0f : 1.0f;
        float inv = sign / norm;
        for (size_t i = 0; i < head_dim; ++i) {
            rotation[i * head_dim + j] = col[i] * inv;
        }
    }

    return rotation;
}

std::vector<float> make_inverse_rotation_scaled(const std::vector<float>& rotation, size_t head_dim) {
    std::vector<float> inv(head_dim * head_dim, 0.0f);
    const float inv_sqrt_dim = 1.0f / std::sqrt(static_cast<float>(head_dim));
    for (size_t row = 0; row < head_dim; ++row) {
        for (size_t col = 0; col < head_dim; ++col) {
            inv[row * head_dim + col] = rotation[row * head_dim + col] * inv_sqrt_dim;
        }
    }
    return inv;
}

std::vector<float> dequantize_kv_key_reference(
    const int8_t* cached_keys,
    const float* key_scales,
    size_t token_idx,
    size_t kv_heads,
    size_t head_dim,
    size_t group_size = KV_QUANT_GROUP_SIZE) {
    const size_t num_groups = (head_dim + group_size - 1) / group_size;
    std::vector<float> result(kv_heads * head_dim, 0.0f);
    for (size_t head = 0; head < kv_heads; ++head) {
        const int8_t* src = cached_keys + token_idx * kv_heads * head_dim + head * head_dim;
        const float* scales = key_scales + (token_idx * kv_heads + head) * num_groups;
        for (size_t dim = 0; dim < head_dim; ++dim) {
            result[head * head_dim + dim] = static_cast<float>(src[dim]) * scales[dim / group_size];
        }
    }
    return result;
}

std::vector<float> dequantize_theoretical_v_reference(
    const uint8_t* packed_values,
    const float* magnitudes,
    const float* inverse_rotation_scaled,
    size_t token_idx,
    size_t kv_heads,
    size_t head_dim) {
    const size_t packed_head_dim = (head_dim + 7) / 8;
    std::vector<float> result(kv_heads * head_dim, 0.0f);
    for (size_t head = 0; head < kv_heads; ++head) {
        const uint8_t* bits = packed_values + token_idx * kv_heads * packed_head_dim + head * packed_head_dim;
        const float magnitude = magnitudes[token_idx * kv_heads + head];
        for (size_t out_dim = 0; out_dim < head_dim; ++out_dim) {
            float sum = 0.0f;
            const float* inv_row = inverse_rotation_scaled + out_dim * head_dim;
            for (size_t rot_dim = 0; rot_dim < head_dim; ++rot_dim) {
                const uint8_t packed = bits[rot_dim >> 3];
                const float sign = ((packed >> (rot_dim & 7)) & 1u) ? 1.0f : -1.0f;
                sum += sign * inv_row[rot_dim];
            }
            result[head * head_dim + out_dim] = magnitude * sum;
        }
    }
    return result;
}

}  // namespace

bool test_neon_add_fp16_correctness() {
    const size_t size = 16;
    std::vector<__fp16> a(size), b(size), result(size), expected(size);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (size_t i = 0; i < size; ++i) {
        a[i] = static_cast<__fp16>(dis(gen));
        b[i] = static_cast<__fp16>(dis(gen));
        expected[i] = a[i] + b[i];
    }

    cactus_add_f16(a.data(), b.data(), result.data(), size);

    for (size_t i = 0; i < size; ++i) {
        if (std::abs(static_cast<float>(result[i] - expected[i])) > 1e-3f) {
            return false;
        }
    }
    return true;
}

bool test_neon_subtract_fp16_correctness() {
    const size_t size = 16;
    std::vector<__fp16> a(size), b(size), result(size), expected(size);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (size_t i = 0; i < size; ++i) {
        a[i] = static_cast<__fp16>(dis(gen));
        b[i] = static_cast<__fp16>(dis(gen));
        expected[i] = a[i] - b[i];
    }

    cactus_subtract_f16(a.data(), b.data(), result.data(), size);

    for (size_t i = 0; i < size; ++i) {
        if (std::abs(static_cast<float>(result[i] - expected[i])) > 1e-3f) {
            return false;
        }
    }
    return true;
}

bool test_neon_multiply_fp16_correctness() {
    const size_t size = 16;
    std::vector<__fp16> a(size), b(size), result(size), expected(size);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (size_t i = 0; i < size; ++i) {
        a[i] = static_cast<__fp16>(dis(gen));
        b[i] = static_cast<__fp16>(dis(gen));
        expected[i] = a[i] * b[i];
    }

    cactus_multiply_f16(a.data(), b.data(), result.data(), size);

    for (size_t i = 0; i < size; ++i) {
        if (std::abs(static_cast<float>(result[i] - expected[i])) > 1e-3f) {
            return false;
        }
    }
    return true;
}

bool test_neon_scalar_operations_fp16_correctness() {
    const size_t size = 8;
    std::vector<__fp16> input = {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, -3.0f, -4.0f};
    std::vector<__fp16> result(size);
    const float scalar = 2.0f;

    std::vector<__fp16> expected_add(size);
    for (size_t i = 0; i < size; ++i) {
        expected_add[i] = static_cast<__fp16>(static_cast<float>(input[i]) + scalar);
    }

    cactus_scalar_op_f16(input.data(), result.data(), size, scalar, ScalarOpType::ADD);

    if (!TestUtils::compare_arrays(result.data(), expected_add.data(), size)) {
        return false;
    }

    std::vector<__fp16> expected_mul(size);
    for (size_t i = 0; i < size; ++i) {
        expected_mul[i] = static_cast<__fp16>(static_cast<float>(input[i]) * scalar);
    }

    cactus_scalar_op_f16(input.data(), result.data(), size, scalar, ScalarOpType::MULTIPLY);

    return TestUtils::compare_arrays(result.data(), expected_mul.data(), size);
}

bool test_neon_reduction_correctness() {
    std::vector<__fp16> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

    double sum_result = cactus_sum_all_f16(input.data(), input.size());
    double expected_sum = 36.0;

    if (std::abs(sum_result - expected_sum) > 1e-2) {
        return false;
    }

    double mean_result = cactus_mean_all_f16(input.data(), input.size());
    double expected_mean = 4.5;

    if (std::abs(mean_result - expected_mean) > 1e-2) {
        return false;
    }

    return true;
}

bool test_neon_transpose_fp16_correctness() {
    const size_t M = 3, N = 4;
    std::vector<__fp16> input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<__fp16> result(M * N);
    std::vector<__fp16> expected = {1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12};

    size_t shape[] = {M, N};
    size_t perm[] = {1, 0};

    cactus_transpose_f16(input.data(), result.data(), shape, perm, 2, 0, M);

    return TestUtils::compare_arrays(result.data(), expected.data(), M * N);
}

bool test_neon_softmax_correctness() {
    const size_t batch_size = 1, seq_len = 4, vocab_size = 3;
    std::vector<__fp16> input = {1.0f, 2.0f, 3.0f,
                                 2.0f, 3.0f, 4.0f,
                                 3.0f, 4.0f, 5.0f,
                                 4.0f, 5.0f, 6.0f};
    std::vector<__fp16> result(input.size());

    cactus_softmax_f16(input.data(), result.data(), batch_size, seq_len, vocab_size);

    for (size_t i = 0; i < seq_len; ++i) {
        float row_sum = 0.0f;
        for (size_t j = 0; j < vocab_size; ++j) {
            row_sum += static_cast<float>(result[i * vocab_size + j]);
        }
        if (std::abs(row_sum - 1.0f) > 1e-2f) {
            return false;
        }
    }

    return true;
}


bool test_neon_rope_correctness() {
    const size_t batch_size = 1, seq_len = 2, num_heads = 1, head_dim = 4;
    const size_t start_pos = 0;
    const float theta = 10000.0f;
    const size_t total_elements = batch_size * seq_len * num_heads * head_dim;

    std::vector<__fp16> input(total_elements);
    std::vector<__fp16> result(total_elements);

    TestUtils::fill_random_fp16(input);

    cactus_rope_f16(input.data(), result.data(),
                    batch_size, seq_len, num_heads, head_dim, start_pos, theta);

    bool different_from_input = false;
    for (size_t i = 0; i < total_elements; ++i) {
        if (std::abs(static_cast<float>(result[i]) - static_cast<float>(input[i])) > 1e-3f) {
            different_from_input = true;
            break;
        }
    }

    return different_from_input;
}

bool test_neon_attention_fp16_correctness() {
    const size_t batch_size = 1, seq_len = 2, num_heads = 1, head_dim = 8;
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    const size_t total_elements = batch_size * seq_len * num_heads * head_dim;

    std::vector<__fp16> queries(total_elements);
    std::vector<__fp16> keys(total_elements);
    std::vector<__fp16> values(total_elements);
    std::vector<__fp16> result(total_elements);

    TestUtils::fill_random_fp16(queries);
    TestUtils::fill_random_fp16(keys);
    TestUtils::fill_random_fp16(values);

    cactus_attention_f16(queries.data(), keys.data(), values.data(), result.data(),
                         batch_size, seq_len, seq_len, num_heads, num_heads, head_dim, scale, nullptr);

    bool has_non_zero = false;
    for (size_t i = 0; i < total_elements; ++i) {
        if (static_cast<float>(result[i]) != 0) {
            has_non_zero = true;
            break;
        }
    }

    return has_non_zero;
}

bool test_theoretical_1bit_hybrid_attention_correctness() {
    const size_t batch_size = 1;
    const size_t seq_len = 1;
    const size_t cache_len = 5;
    const size_t new_len = 1;
    const size_t num_q_heads = 4;
    const size_t num_kv_heads = 2;
    const size_t head_dim = 32;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const size_t num_groups = head_dim / KV_QUANT_GROUP_SIZE;
    const size_t packed_head_dim = (head_dim + 7) / 8;
    const size_t gqa_group_size = num_q_heads / num_kv_heads;

    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dis(-0.5f, 0.5f);

    std::vector<__fp16> query(batch_size * seq_len * num_q_heads * head_dim);
    std::vector<__fp16> keys_cached_fp16(cache_len * num_kv_heads * head_dim);
    std::vector<__fp16> values_cached_fp16(cache_len * num_kv_heads * head_dim);
    std::vector<__fp16> keys_new(new_len * num_kv_heads * head_dim);
    std::vector<__fp16> values_new(new_len * num_kv_heads * head_dim);
    for (auto* vec : {&query, &keys_cached_fp16, &values_cached_fp16, &keys_new, &values_new}) {
        for (auto& v : *vec) v = static_cast<__fp16>(dis(gen));
    }

    std::vector<int8_t> keys_cached_q(cache_len * num_kv_heads * head_dim);
    std::vector<float> key_scales(cache_len * num_kv_heads * num_groups, 1.0f);
    cactus_quantize_kv_fp16_to_int8(
        keys_cached_fp16.data(),
        keys_cached_q.data(),
        key_scales.data(),
        cache_len,
        num_kv_heads,
        head_dim);

    const std::vector<float> rotation = make_test_rotation_matrix(head_dim, 1234);
    const std::vector<float> inverse_rotation_scaled = make_inverse_rotation_scaled(rotation, head_dim);

    std::vector<uint8_t> values_cached_bits(cache_len * num_kv_heads * packed_head_dim, 0);
    std::vector<float> value_magnitudes(cache_len * num_kv_heads, 0.0f);
    cactus_quantize_kv_fp16_to_theoretical_1bit(
        values_cached_fp16.data(),
        values_cached_bits.data(),
        value_magnitudes.data(),
        rotation.data(),
        cache_len,
        num_kv_heads,
        head_dim);

    std::vector<__fp16> output(batch_size * seq_len * num_q_heads * head_dim, static_cast<__fp16>(0.0f));
    cactus_attention_hybrid_int8_fp16(
        query.data(),
        keys_cached_q.data(),
        reinterpret_cast<const int8_t*>(values_cached_bits.data()),
        key_scales.data(),
        value_magnitudes.data(),
        keys_new.data(),
        values_new.data(),
        output.data(),
        batch_size,
        seq_len,
        cache_len,
        new_len,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale,
        cache_len,
        true,
        0,
        KV_QUANT_GROUP_SIZE,
        head_dim,
        inverse_rotation_scaled.data(),
        true);

    float max_abs_err = 0.0f;
    for (size_t q_head_idx = 0; q_head_idx < num_q_heads; ++q_head_idx) {
        const size_t kv_head_idx = q_head_idx / gqa_group_size;
        std::vector<float> scores(cache_len + new_len, 0.0f);

        for (size_t kv_pos = 0; kv_pos < cache_len; ++kv_pos) {
            auto key = dequantize_kv_key_reference(keys_cached_q.data(), key_scales.data(), kv_pos, num_kv_heads, head_dim);
            float dot = 0.0f;
            const size_t q_offset = q_head_idx * head_dim;
            const size_t k_offset = kv_head_idx * head_dim;
            for (size_t dim = 0; dim < head_dim; ++dim) {
                dot += static_cast<float>(query[q_offset + dim]) * key[k_offset + dim];
            }
            scores[kv_pos] = dot * scale;
        }
        {
            float dot = 0.0f;
            const size_t q_offset = q_head_idx * head_dim;
            const size_t k_offset = kv_head_idx * head_dim;
            for (size_t dim = 0; dim < head_dim; ++dim) {
                dot += static_cast<float>(query[q_offset + dim]) * static_cast<float>(keys_new[k_offset + dim]);
            }
            scores[cache_len] = dot * scale;
        }

        const float max_score = *std::max_element(scores.begin(), scores.end());
        float sum_exp = 0.0f;
        for (float& s : scores) {
            s = std::exp(s - max_score);
            sum_exp += s;
        }
        for (float& s : scores) s /= sum_exp;

        std::vector<float> expected(head_dim, 0.0f);
        for (size_t kv_pos = 0; kv_pos < cache_len; ++kv_pos) {
            auto value = dequantize_theoretical_v_reference(
                values_cached_bits.data(),
                value_magnitudes.data(),
                inverse_rotation_scaled.data(),
                kv_pos,
                num_kv_heads,
                head_dim);
            const size_t v_offset = kv_head_idx * head_dim;
            for (size_t dim = 0; dim < head_dim; ++dim) {
                expected[dim] += scores[kv_pos] * value[v_offset + dim];
            }
        }
        {
            const size_t v_offset = kv_head_idx * head_dim;
            for (size_t dim = 0; dim < head_dim; ++dim) {
                expected[dim] += scores[cache_len] * static_cast<float>(values_new[v_offset + dim]);
            }
        }

        const size_t out_offset = q_head_idx * head_dim;
        for (size_t dim = 0; dim < head_dim; ++dim) {
            max_abs_err = std::max(max_abs_err, std::abs(static_cast<float>(output[out_offset + dim]) - expected[dim]));
        }
    }
    if (max_abs_err >= 1e-2f) {
        std::cerr << "Theoretical 1-bit hybrid attention max abs error: " << max_abs_err << std::endl;
    }
    return max_abs_err < 1e-2f;
}

bool test_matmul_int8_grouped_correctness() {
    const size_t M = 2, K = 128, N = 4;
    const size_t group_size = 32;
    const size_t num_groups = K / group_size;
    const size_t BLOCK_SIZE = 4;

    std::vector<__fp16> A(M * K);
    for (size_t i = 0; i < M * K; ++i) {
        A[i] = static_cast<__fp16>((static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f) * 0.5f);
    }

    std::vector<int8_t> B_rowmajor(N * K);
    for (size_t i = 0; i < N * K; ++i) {
        B_rowmajor[i] = static_cast<int8_t>((rand() % 128) - 64);
    }

    std::vector<__fp16> B_scales_rowmajor(N * num_groups);
    for (size_t n = 0; n < N; ++n) {
        for (size_t g = 0; g < num_groups; ++g) {
            float max_abs = 0.0f;
            for (size_t k = 0; k < group_size; ++k) {
                float val = std::abs(static_cast<float>(B_rowmajor[n * K + g * group_size + k]));
                if (val > max_abs) max_abs = val;
            }
            float scale = max_abs / 127.0f;
            if (scale < 1e-6f) scale = 1e-6f;
            B_scales_rowmajor[n * num_groups + g] = static_cast<__fp16>(scale);
        }
    }

    std::vector<int8_t> B_interleaved(N * K);
    size_t N_blocks = N / BLOCK_SIZE;
    size_t K_blocks = K / BLOCK_SIZE;
    for (size_t n_blk = 0; n_blk < N_blocks; ++n_blk) {
        for (size_t k_blk = 0; k_blk < K_blocks; ++k_blk) {
            for (size_t ni = 0; ni < BLOCK_SIZE; ++ni) {
                for (size_t ki = 0; ki < BLOCK_SIZE; ++ki) {
                    size_t src_n = n_blk * BLOCK_SIZE + ni;
                    size_t src_k = k_blk * BLOCK_SIZE + ki;
                    size_t dst_idx = (n_blk * K_blocks + k_blk) * (BLOCK_SIZE * BLOCK_SIZE) +
                                     ni * BLOCK_SIZE + ki;
                    B_interleaved[dst_idx] = B_rowmajor[src_n * K + src_k];
                }
            }
        }
    }

    std::vector<__fp16> B_scales_interleaved(N * num_groups);
    for (size_t n_blk = 0; n_blk < N_blocks; ++n_blk) {
        for (size_t g = 0; g < num_groups; ++g) {
            for (size_t ni = 0; ni < BLOCK_SIZE; ++ni) {
                size_t src_n = n_blk * BLOCK_SIZE + ni;
                size_t dst_idx = (n_blk * num_groups + g) * BLOCK_SIZE + ni;
                B_scales_interleaved[dst_idx] = B_scales_rowmajor[src_n * num_groups + g];
            }
        }
    }

    std::vector<int8_t> A_quant(M * K);
    std::vector<float> A_scales(M);
    for (size_t m = 0; m < M; ++m) {
        float max_abs = cactus_fp16_max_abs(A.data() + m * K, K);
        float scale = max_abs / 127.0f;
        if (scale < 1e-10f) scale = 1e-10f;
        A_scales[m] = scale;
        cactus_fp16_to_int8(A.data() + m * K, A_quant.data() + m * K, K, scale);
    }

    std::vector<__fp16> C(M * N);

    cactus_matmul_int8(A_quant.data(), A_scales.data(), B_interleaved.data(),
                       B_scales_interleaved.data(), C.data(), M, K, N, group_size);

    std::vector<float> C_ref(M * N, 0.0f);
    for (size_t m = 0; m < M; ++m) {
        float a_max_abs = 0.0f;
        for (size_t k = 0; k < K; ++k) {
            float val = std::abs(static_cast<float>(A[m * K + k]));
            if (val > a_max_abs) a_max_abs = val;
        }
        float a_scale = a_max_abs / 127.0f;
        if (a_scale < 1e-10f) a_scale = 1e-10f;

        std::vector<int8_t> A_quant_ref(K);
        for (size_t k = 0; k < K; ++k) {
            float val = static_cast<float>(A[m * K + k]) / a_scale;
            int32_t q = static_cast<int32_t>(std::round(val));
            q = std::max(-128, std::min(127, q));
            A_quant_ref[k] = static_cast<int8_t>(q);
        }

        for (size_t n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (size_t g = 0; g < num_groups; ++g) {
                float b_scale = static_cast<float>(B_scales_rowmajor[n * num_groups + g]);
                float combined_scale = a_scale * b_scale;

                int32_t group_sum = 0;
                for (size_t k = 0; k < group_size; ++k) {
                    size_t k_idx = g * group_size + k;
                    group_sum += static_cast<int32_t>(A_quant_ref[k_idx]) *
                                 static_cast<int32_t>(B_rowmajor[n * K + k_idx]);
                }
                acc += static_cast<float>(group_sum) * combined_scale;
            }
            C_ref[m * N + n] = acc;
        }
    }

    float max_abs_error = 0.0f;
    for (size_t i = 0; i < M * N; ++i) {
        float error = std::abs(static_cast<float>(C[i]) - C_ref[i]);
        if (error > max_abs_error) max_abs_error = error;
    }

    return max_abs_error < 0.1f;
}

bool test_int4_matmul_correctness() {
    const size_t K = 128, N = 8, group_size = 32;
    const size_t num_groups = K / group_size;
    const size_t BS = 4;

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    std::vector<float> B_fp32(N * K);
    for (size_t i = 0; i < N * K; ++i) B_fp32[i] = dis(gen);

    std::vector<int8_t> B_raw(N * K);
    std::vector<float> B_scales(N * num_groups);
    for (size_t n = 0; n < N; ++n) {
        for (size_t g = 0; g < num_groups; ++g) {
            float max_abs = 0.0f;
            for (size_t k = 0; k < group_size; ++k) {
                float val = std::abs(B_fp32[n * K + g * group_size + k]);
                if (val > max_abs) max_abs = val;
            }
            float scale = std::max(max_abs / 7.0f, 1e-10f);
            B_scales[n * num_groups + g] = scale;
            for (size_t k = 0; k < group_size; ++k) {
                int32_t q = static_cast<int32_t>(std::round(B_fp32[n * K + g * group_size + k] / scale));
                B_raw[n * K + g * group_size + k] = static_cast<int8_t>(std::clamp(q, -8, 7));
            }
        }
    }

    std::vector<int8_t> B_interleaved(N * K);
    for (size_t n_blk = 0; n_blk < N / BS; ++n_blk)
        for (size_t k_blk = 0; k_blk < K / BS; ++k_blk)
            for (size_t ni = 0; ni < BS; ++ni)
                for (size_t ki = 0; ki < BS; ++ki)
                    B_interleaved[(n_blk * (K / BS) + k_blk) * BS * BS + ni * BS + ki] =
                        B_raw[(n_blk * BS + ni) * K + k_blk * BS + ki];

    std::vector<uint8_t> B_packed(N * K / 2);
    for (size_t i = 0; i < N * K; i += 32) {
        for (size_t j = 0; j < 16; ++j) {
            uint8_t lo = static_cast<uint8_t>(B_interleaved[i + j] & 0x0F);
            uint8_t hi = static_cast<uint8_t>((B_interleaved[i + 16 + j] & 0x0F) << 4);
            B_packed[i / 2 + j] = lo | hi;
        }
    }

    std::vector<__fp16> B_scales_interleaved(N * num_groups);
    for (size_t n_blk = 0; n_blk < N / BS; ++n_blk)
        for (size_t g = 0; g < num_groups; ++g)
            for (size_t ni = 0; ni < BS; ++ni)
                B_scales_interleaved[(n_blk * num_groups + g) * BS + ni] =
                    static_cast<__fp16>(B_scales[(n_blk * BS + ni) * num_groups + g]);

    for (size_t M : {1, 5}) {
        std::vector<__fp16> A_fp16(M * K);
        for (size_t i = 0; i < M * K; ++i) A_fp16[i] = static_cast<__fp16>(dis(gen));

        std::vector<int8_t> A_quant(M * K);
        std::vector<float> A_scales(M);
        for (size_t m = 0; m < M; ++m) {
            A_scales[m] = std::max(cactus_fp16_max_abs(A_fp16.data() + m * K, K) / 127.0f, 1e-10f);
            cactus_fp16_to_int8(A_fp16.data() + m * K, A_quant.data() + m * K, K, A_scales[m]);
        }

        std::vector<__fp16> C(M * N);
        cactus_matmul_int4(A_quant.data(), A_scales.data(),
                           reinterpret_cast<const int8_t*>(B_packed.data()),
                           B_scales_interleaved.data(), C.data(), M, K, N, group_size);

        std::vector<float> C_ref(M * N, 0.0f);
        for (size_t m = 0; m < M; ++m) {
            for (size_t n = 0; n < N; ++n) {
                float acc = 0.0f;
                for (size_t g = 0; g < num_groups; ++g) {
                    int32_t group_sum = 0;
                    for (size_t k = 0; k < group_size; ++k) {
                        size_t k_idx = g * group_size + k;
                        group_sum += static_cast<int32_t>(A_quant[m * K + k_idx]) *
                                     static_cast<int32_t>(B_raw[n * K + k_idx]);
                    }
                    acc += static_cast<float>(group_sum) * A_scales[m] * B_scales[n * num_groups + g];
                }
                C_ref[m * N + n] = acc;
            }
        }

        float max_err = 0.0f;
        for (size_t i = 0; i < M * N; ++i) {
            float err = std::abs(static_cast<float>(C[i]) - C_ref[i]);
            if (err > max_err) max_err = err;
        }

        std::cout << "  INT4 " << (M == 1 ? "GEMV" : "GEMM") << " max abs error: " << max_err << std::endl;
        if (max_err >= 0.1f) return false;
    }

    return true;
}

bool test_ternary_matmul_correctness() {
    const size_t M = 3;
    const size_t K = 128;
    const size_t N = 8;
    constexpr size_t group_size = 32;

    std::vector<__fp16> lhs(M * K);
    for (size_t i = 0; i < lhs.size(); ++i) {
        lhs[i] = static_cast<__fp16>((static_cast<float>((i * 13) % 29) - 14.0f) / 11.0f);
    }

    const auto rhs_rowmajor = make_ternary_signs(N, K);
    const auto rhs_interleaved = interleave_int8_rows_4(rhs_rowmajor, N, K);
    const auto rhs_packed = pack_ternary_2bit(rhs_interleaved);

    struct TernaryCase {
        const char* label;
        TernaryScaleMode mode;
        std::vector<__fp16> scales;
    };

    std::vector<__fp16> column_scales(K);
    for (size_t col = 0; col < K; ++col) {
        column_scales[col] = static_cast<__fp16>(0.15f + 0.01f * static_cast<float>(col % 9));
    }

    const std::vector<TernaryCase> cases = {
        {"tensor", TernaryScaleMode::TENSOR, {static_cast<__fp16>(0.30f)}},
        {"row", TernaryScaleMode::ROW, {static_cast<__fp16>(0.10f), static_cast<__fp16>(0.15f), static_cast<__fp16>(0.20f), static_cast<__fp16>(0.25f),
                                         static_cast<__fp16>(0.30f), static_cast<__fp16>(0.35f), static_cast<__fp16>(0.40f), static_cast<__fp16>(0.45f)}},
        {"column", TernaryScaleMode::COLUMN, column_scales},
    };

    for (const auto& test_case : cases) {
        std::vector<__fp16> effective_lhs = lhs;

        if (test_case.mode == TernaryScaleMode::COLUMN) {
            for (size_t row = 0; row < M; ++row) {
                for (size_t col = 0; col < K; ++col) {
                    effective_lhs[row * K + col] = static_cast<__fp16>(
                        static_cast<float>(lhs[row * K + col]) * static_cast<float>(test_case.scales[col]));
                }
            }
        }

        std::vector<int8_t> lhs_quantized(M * K);
        std::vector<float> lhs_scales(M, 1.0f);
        for (size_t row = 0; row < M; ++row) {
            float scale = cactus_fp16_max_abs(effective_lhs.data() + row * K, K) / 127.0f;
            if (scale < 1e-10f) scale = 1e-10f;
            lhs_scales[row] = scale;
            cactus_fp16_to_int8(effective_lhs.data() + row * K, lhs_quantized.data() + row * K, K, scale);
        }

        std::vector<__fp16> output(M * N, static_cast<__fp16>(0.0f));
        cactus_matmul_ternary_packed2(lhs_quantized.data(), lhs_scales.data(),
                                      rhs_packed.data(), test_case.scales.data(), test_case.mode,
                                      output.data(), M, K, N, group_size);

        const auto expected = ternary_reference_output(
            lhs, M, K, rhs_rowmajor, N, test_case.mode, test_case.scales);

        float max_err = 0.0f;
        for (size_t i = 0; i < output.size(); ++i) {
            float err = std::abs(static_cast<float>(output[i]) - static_cast<float>(expected[i]));
            if (err > max_err) max_err = err;
        }

        std::cout << "  TERNARY " << test_case.label << " max abs error: " << max_err << std::endl;
        if (max_err >= 0.1f) {
            return false;
        }
    }

    return true;
}

bool test_stft_kernel_correctness() {
    const size_t N = 2, C_in = 1, L = 8, K = 4, stride = 2, num_fft_bins = 2;
    const size_t C_out = 2 * num_fft_bins;
    const size_t out_len = (L - K) / stride + 1;

    const __fp16 bin0_re[] = {(__fp16) 1, (__fp16) 1, (__fp16) 1, (__fp16) 1};
    const __fp16 bin1_re[] = {(__fp16) 1, (__fp16) 0, (__fp16)-1, (__fp16) 0};
    const __fp16 bin0_im[] = {(__fp16) 0, (__fp16) 0, (__fp16) 0, (__fp16) 0};
    const __fp16 bin1_im[] = {(__fp16) 0, (__fp16)-1, (__fp16) 0, (__fp16) 1};
    std::vector<__fp16> weight;
    for (const __fp16* row : {bin0_re, bin1_re, bin0_im, bin1_im})
        weight.insert(weight.end(), row, row + K);

    const __fp16 ramp[]   = {(__fp16)1, (__fp16)2, (__fp16)3, (__fp16)4,
                              (__fp16)5, (__fp16)6, (__fp16)7, (__fp16)8};
    const __fp16 cosine[] = {(__fp16)0, (__fp16)1, (__fp16) 0, (__fp16)-1,
                              (__fp16)0, (__fp16)1, (__fp16) 0, (__fp16)-1};
    std::vector<__fp16> input;
    input.insert(input.end(), ramp,   ramp   + L);
    input.insert(input.end(), cosine, cosine + L);

    struct Cplx { float r, i; };
    const Cplx expected[2][2][3] = {
        { { {10,0},{18,0},{26,0} }, { {-2,2},{-2,2},{-2,2} } },
        { { { 0,0},{ 0,0},{ 0,0} }, { { 0,-2},{ 0,2},{ 0,-2} } },
    };

    std::vector<__fp16> cplx(N * C_out * out_len, (__fp16)0);
    cactus_stft_f16(input.data(), weight.data(), cplx.data(),
                            N, L, C_in, C_out, K, stride, num_fft_bins);

    const size_t out_bs = C_out * out_len;
    const float tol = 0.1f;
    for (size_t n = 0; n < N; ++n) {
        for (size_t b = 0; b < num_fft_bins; ++b) {
            for (size_t t = 0; t < out_len; ++t) {
                float r  = (float)cplx[n * out_bs + b * out_len + t];
                float im = (float)cplx[n * out_bs + (b + num_fft_bins) * out_len + t];
                if (std::abs(r  - expected[n][b][t].r) > tol) return false;
                if (std::abs(im - expected[n][b][t].i) > tol) return false;
            }
        }
    }

    return true;
}

int main() {
    TestUtils::TestRunner runner("Kernel Backend Tests");

    runner.run_test("Kernel Add FP16 Correctness", test_neon_add_fp16_correctness());
    runner.run_test("Kernel Subtract FP16 Correctness", test_neon_subtract_fp16_correctness());
    runner.run_test("Kernel Multiply FP16 Correctness", test_neon_multiply_fp16_correctness());
    runner.run_test("Kernel Scalar Ops FP16 Correctness", test_neon_scalar_operations_fp16_correctness());
    runner.run_test("Kernel Reduction Correctness", test_neon_reduction_correctness());
    runner.run_test("Kernel Transpose FP16 Correctness", test_neon_transpose_fp16_correctness());
    runner.run_test("Kernel Softmax Correctness", test_neon_softmax_correctness());
    runner.run_test("Kernel RoPE Correctness", test_neon_rope_correctness());
    runner.run_test("Kernel Attention FP16 Correctness", test_neon_attention_fp16_correctness());
    runner.run_test("Kernel Theoretical 1-bit Hybrid Attention Correctness", test_theoretical_1bit_hybrid_attention_correctness());
    runner.run_test("Kernel Grouped INT8 MatMul Correctness", test_matmul_int8_grouped_correctness());
    runner.run_test("Kernel INT4 MatMul Correctness", test_int4_matmul_correctness());
    runner.run_test("Kernel Ternary MatMul Correctness", test_ternary_matmul_correctness());
    runner.run_test("Kernel STFT Complex Correctness", test_stft_kernel_correctness());

    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
