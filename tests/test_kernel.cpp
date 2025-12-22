#include "test_utils.h"
#include <vector>
#include <cmath>
#include <iostream>

bool test_neon_add_correctness() {
    const size_t size = 16;
    std::vector<int8_t> a(size), b(size), result(size), expected(size);
    
    TestUtils::fill_random_int8(a);
    TestUtils::fill_random_int8(b);
    
    for (size_t i = 0; i < size; ++i) {
        expected[i] = static_cast<int8_t>(std::max(-128, std::min(127, static_cast<int>(a[i]) + static_cast<int>(b[i]))));
    }
    
    cactus_add_int8(a.data(), b.data(), result.data(), size);
    
    return TestUtils::compare_arrays(result.data(), expected.data(), size);
}

bool test_neon_subtract_correctness() {
    const size_t size = 16;
    std::vector<int8_t> a(size), b(size), result(size), expected(size);
    
    TestUtils::fill_random_int8(a);
    TestUtils::fill_random_int8(b);
    
    for (size_t i = 0; i < size; ++i) {
        expected[i] = static_cast<int8_t>(std::max(-128, std::min(127, static_cast<int>(a[i]) - static_cast<int>(b[i]))));
    }
    
    cactus_subtract_int8(a.data(), b.data(), result.data(), size);
    
    return TestUtils::compare_arrays(result.data(), expected.data(), size);
}

bool test_neon_hadamard_correctness() {
    const size_t size = 16;
    std::vector<int8_t> a(size), b(size), result(size), expected(size);
    
    TestUtils::fill_random_int8(a);
    TestUtils::fill_random_int8(b);
    
    for (size_t i = 0; i < size; ++i) {
        expected[i] = static_cast<int8_t>(std::max(-128, std::min(127, static_cast<int>(a[i]) * static_cast<int>(b[i]))));
    }
    
    cactus_multiply_int8(a.data(), b.data(), result.data(), size);
    
    return TestUtils::compare_arrays(result.data(), expected.data(), size);
}

bool test_neon_scalar_operations_correctness() {
    const size_t size = 8;
    std::vector<int8_t> input = {1, 2, 3, 4, -1, -2, -3, -4};
    std::vector<int8_t> result(size);
    const float scalar = 2.0f;
    
    std::vector<int8_t> expected_add(size);
    for (size_t i = 0; i < size; ++i) {
        expected_add[i] = static_cast<int8_t>(std::max(-128, std::min(127, static_cast<int>(input[i] + scalar))));
    }
    
    cactus_scalar_op_int8(input.data(), result.data(), size, scalar, ScalarOpType::ADD);
    
    if (!TestUtils::compare_arrays(result.data(), expected_add.data(), size)) {
        return false;
    }
    
    std::vector<int8_t> expected_mul(size);
    for (size_t i = 0; i < size; ++i) {
        expected_mul[i] = static_cast<int8_t>(std::max(-128, std::min(127, static_cast<int>(input[i] * scalar))));
    }
    
    cactus_scalar_op_int8(input.data(), result.data(), size, scalar, ScalarOpType::MULTIPLY);
    
    return TestUtils::compare_arrays(result.data(), expected_mul.data(), size);
}

bool test_neon_matrix_multiply_correctness() {
    const size_t M = 4, K = 3, N = 2;
    std::vector<int8_t> a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<int8_t> b = {1, 2, 3, 4, 5, 6};
    std::vector<int8_t> b_transposed = {1, 3, 5, 2, 4, 6};
    std::vector<int8_t> result(M * N, 0);
    
    std::vector<int8_t> expected = {22, 28, 49, 64, 76, 100, 103, 127};
    
    cactus_matmul_int8(a.data(), b_transposed.data(), result.data(), M, K, N, 1.0f, 1.0f, 1.0f);
    
    for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] = static_cast<int8_t>(std::max(-128, std::min(127, static_cast<int>(expected[i]))));
    }
    
    return TestUtils::compare_arrays(result.data(), expected.data(), M * N);
}

bool test_neon_reduction_correctness() {
    // Test INT8 sum
    std::vector<int8_t> input = {1, 2, 3, 4, 5, 6, 7, 8};
    
    int64_t sum_result = cactus_sum_all_int8(input.data(), input.size());
    int64_t expected_sum = 36; 
    
    if (sum_result != expected_sum) {
        return false;
    }
    
    double mean_result = cactus_mean_all_int8(input.data(), input.size());
    double expected_mean = 4.5; 
    
    if (std::abs(mean_result - expected_mean) > 1e-6) {
        return false;
    }
    
    // Test FP16 sum
    std::vector<__fp16> input_f16 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    double sum_result_f16 = cactus_sum_all_f16(input_f16.data(), input_f16.size());
    double expected_sum_f16 = 36.0;
    
    if (std::abs(sum_result_f16 - expected_sum_f16) > 1e-3) {
        return false;
    }
    
    // Test FP16 mean (which uses sum internally)
    double mean_result_f16 = cactus_mean_all_f16(input_f16.data(), input_f16.size());
    double expected_mean_f16 = 4.5;
    
    if (std::abs(mean_result_f16 - expected_mean_f16) > 1e-3) {
        return false;
    }
    
    return true;
}

bool test_neon_transpose_correctness() {
    const size_t M = 3, N = 4;
    std::vector<int8_t> input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<int8_t> result(M * N);
    std::vector<int8_t> expected = {1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12};
    
    size_t shape[] = {M, N};
    size_t perm[] = {1, 0};
    
    cactus_transpose_int8(input.data(), result.data(), shape, perm, 2, 0, M);
    
    return TestUtils::compare_arrays(result.data(), expected.data(), M * N);
}

bool test_neon_softmax_correctness() {
    const size_t batch_size = 1, seq_len = 4, vocab_size = 3;
    std::vector<float> input = {1.0f, 2.0f, 3.0f,
                               2.0f, 3.0f, 4.0f,
                               3.0f, 4.0f, 5.0f,
                               4.0f, 5.0f, 6.0f};
    std::vector<float> result(input.size());
    
    cactus_softmax_f32(input.data(), result.data(), batch_size, seq_len, vocab_size);
    
    for (size_t i = 0; i < seq_len; ++i) {
        float row_sum = 0.0f;
        for (size_t j = 0; j < vocab_size; ++j) {
            row_sum += result[i * vocab_size + j];
        }
        if (std::abs(row_sum - 1.0f) > 1e-5f) {
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
    
    std::vector<float> input(total_elements);
    std::vector<float> result(total_elements);
    
    TestUtils::fill_random_float(input);
    
    cactus_rope_f32(input.data(), result.data(), 
                   batch_size, seq_len, num_heads, head_dim, start_pos, theta);
    
    bool different_from_input = false;
    for (size_t i = 0; i < total_elements; ++i) {
        if (std::abs(result[i] - input[i]) > 1e-6f) {
            different_from_input = true;
            break;
        }
    }
    
    return different_from_input;
}

bool test_neon_attention_correctness() {
    const size_t batch_size = 1, seq_len = 2, num_heads = 1, head_dim = 4;
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    const size_t total_elements = batch_size * seq_len * num_heads * head_dim;
    
    std::vector<int8_t> queries(total_elements);
    std::vector<int8_t> keys(total_elements);
    std::vector<int8_t> values(total_elements);
    std::vector<int8_t> result(total_elements);
    
    TestUtils::fill_random_int8(queries);
    TestUtils::fill_random_int8(keys);
    TestUtils::fill_random_int8(values);
    
    cactus_attention_int8(queries.data(), keys.data(), values.data(), result.data(),
                         batch_size, seq_len, seq_len, num_heads, num_heads, head_dim, scale, nullptr,
                         1.0f, 1.0f, 1.0f, 1.0f);
    
    bool has_non_zero = false;
    for (size_t i = 0; i < total_elements; ++i) {
        if (result[i] != 0) {
            has_non_zero = true;
            break;
        }
    }
    
    return has_non_zero;
}


int main() {
    TestUtils::TestRunner runner("Kernel Backend Tests");
    
    runner.run_test("Kernel Add Correctness", test_neon_add_correctness());
    runner.run_test("Kernel Subtract Correctness", test_neon_subtract_correctness());
    runner.run_test("Kernel Multiply Correctness", test_neon_hadamard_correctness());
    runner.run_test("Kernel Scalar Operations Correctness", test_neon_scalar_operations_correctness());
    runner.run_test("Kernel Matrix Multiply Correctness", test_neon_matrix_multiply_correctness());
    runner.run_test("Kernel Reduction Correctness", test_neon_reduction_correctness());
    runner.run_test("Kernel Transpose Correctness", test_neon_transpose_correctness());
    runner.run_test("Kernel Softmax Correctness", test_neon_softmax_correctness());
    runner.run_test("Kernel RoPE Correctness", test_neon_rope_correctness());
    runner.run_test("Kernel Attention Correctness", test_neon_attention_correctness());
    
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}