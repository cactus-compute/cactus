#include "test_utils.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <random>

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

bool test_matmul_int8_grouped_correctness() {
    const size_t M = 2, K = 128, N = 4;
    const size_t group_size = 32;
    const size_t num_groups = K / group_size;
    const size_t BLOCK_SIZE = 4;

    std::vector<__fp16> A(M * K);
    for (size_t i = 0; i < M * K; ++i) {
        A[i] = static_cast<__fp16>((static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.5f);
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
    runner.run_test("Kernel Grouped INT8 MatMul Correctness", test_matmul_int8_grouped_correctness());

    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
