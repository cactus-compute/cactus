#include "test_utils.h"
#include <cassert>
#include <memory>
#include <fstream>
#include <cstring>
#include <cmath>
#include <iostream>
#include <cstdio>

bool test_basic_operations() {
    TestUtils::FP16TestFixture fixture("Basic Operations");

    size_t input_a = fixture.create_input({2, 3});
    size_t input_b = fixture.create_input({2, 3});
    size_t add_result = fixture.graph().add(input_a, input_b);
    size_t mul_result = fixture.graph().multiply(add_result, input_a);
    size_t scalar_result = fixture.graph().scalar_multiply(mul_result, 2.0f);

    std::vector<__fp16> data_a = {1, 2, 3, 4, 5, 6};
    std::vector<__fp16> data_b = {2, 3, 4, 5, 6, 7};

    fixture.set_input_data(input_a, data_a);
    fixture.set_input_data(input_b, data_b);
    fixture.execute();

    std::vector<__fp16> expected(6);
    for (int i = 0; i < 6; i++) {
        float result = ((static_cast<float>(data_a[i]) + static_cast<float>(data_b[i])) * static_cast<float>(data_a[i])) * 2.0f;
        expected[i] = static_cast<__fp16>(result);
    }

    return fixture.verify_output(scalar_result, expected);
}

bool test_basic_addition() {
    return TestUtils::test_basic_operation(
        "Addition",
        [](CactusGraph& graph, size_t a, size_t b) { return graph.add(a, b); },
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {6, 8, 10, 12}
    );
}

bool test_basic_subtraction() {
    return TestUtils::test_basic_operation(
        "Subtraction",
        [](CactusGraph& graph, size_t a, size_t b) { return graph.subtract(a, b); },
        {10, 8, 6, 4},
        {2, 3, 1, 2},
        {8, 5, 5, 2}
    );
}

bool test_basic_multiplication() {
    return TestUtils::test_basic_operation(
        "Multiplication",
        [](CactusGraph& graph, size_t a, size_t b) { return graph.multiply(a, b); },
        {2, 3, 4, 5},
        {3, 4, 2, 2},
        {6, 12, 8, 10}
    );
}

bool test_basic_division() {
    return TestUtils::test_basic_operation(
        "Division",
        [](CactusGraph& graph, size_t a, size_t b) { return graph.divide(a, b); },
        {12, 15, 8, 9},
        {3, 5, 2, 3},
        {4, 3, 4, 3}
    );
}

bool test_matrix_multiplication() {
    TestUtils::FP16TestFixture fixture("Matrix Multiplication");

    size_t input_a = fixture.create_input({2, 3});
    size_t input_b = fixture.create_input({3, 2});
    size_t matmul_result = fixture.graph().matmul(input_a, input_b, false);

    std::vector<__fp16> data_a = {1, 2, 3, 4, 5, 6};
    std::vector<__fp16> data_b = {1, 2, 3, 4, 5, 6};
    fixture.set_input_data(input_a, data_a);
    fixture.set_input_data(input_b, data_b);
    fixture.execute();

    std::vector<__fp16> expected = {22, 28, 49, 64};
    return fixture.verify_output(matmul_result, expected);
}

bool test_transpose() {
    TestUtils::FP16TestFixture fixture("Transpose");

    size_t input_a = fixture.create_input({2, 3});
    size_t transpose_result = fixture.graph().transpose(input_a);

    std::vector<__fp16> data_a = {1, 2, 3, 4, 5, 6};
    fixture.set_input_data(input_a, data_a);
    fixture.execute();

    std::vector<__fp16> expected = {1, 4, 2, 5, 3, 6};
    return fixture.verify_output(transpose_result, expected);
}

bool test_reshape() {
    TestUtils::FP16TestFixture fixture("Reshape");

    size_t input_a = fixture.create_input({2, 3});
    size_t reshape_result = fixture.graph().reshape(input_a, {3, 2});

    std::vector<__fp16> data_a = {1, 2, 3, 4, 5, 6};
    fixture.set_input_data(input_a, data_a);
    fixture.execute();

    return fixture.verify_output(reshape_result, data_a);
}


bool test_scalar_operations() {
    TestUtils::FP16TestFixture fixture("Scalar Operations");

    size_t input_a = fixture.create_input({4});
    size_t add_result = fixture.graph().scalar_add(input_a, 5.0f);
    size_t mul_result = fixture.graph().scalar_multiply(add_result, 2.0f);

    std::vector<__fp16> data_a = {1, 2, 3, 4};
    fixture.set_input_data(input_a, data_a);
    fixture.execute();

    std::vector<__fp16> expected = {12, 14, 16, 18};
    return fixture.verify_output(mul_result, expected);
}

bool test_scalar_subtract_divide() {
    TestUtils::FP16TestFixture fixture("Scalar Subtract/Divide");

    size_t input_a = fixture.create_input({4});
    size_t sub_result = fixture.graph().scalar_subtract(input_a, 2.0f);
    size_t div_result = fixture.graph().scalar_divide(input_a, 2.0f);

    std::vector<__fp16> data_a = {10, 8, 6, 4};
    fixture.set_input_data(input_a, data_a);
    fixture.execute();

    std::vector<__fp16> expected_sub = {8, 6, 4, 2};
    std::vector<__fp16> expected_div = {5, 4, 3, 2};
    return fixture.verify_output(sub_result, expected_sub) &&
           fixture.verify_output(div_result, expected_div);
}

bool test_scalar_math_functions() {
    TestUtils::FP16TestFixture fixture("Scalar Math Functions");

    size_t input_a = fixture.create_input({3});
    size_t exp_result = fixture.graph().scalar_exp(input_a);
    size_t sqrt_result = fixture.graph().scalar_sqrt(input_a);
    size_t cos_result = fixture.graph().scalar_cos(input_a);
    size_t sin_result = fixture.graph().scalar_sin(input_a);

    std::vector<__fp16> input_data = {0.0f, 1.0f, 4.0f};
    fixture.set_input_data(input_a, input_data);
    fixture.execute();

    std::vector<__fp16> exp_expected = {1.0f, 2.71828f, 54.5982f};
    std::vector<__fp16> sqrt_expected = {0.0f, 1.0f, 2.0f};
    std::vector<__fp16> cos_expected = {1.0f, 0.54030f, -0.65364f};
    std::vector<__fp16> sin_expected = {0.0f, 0.84147f, -0.75680f};

    return fixture.verify_output(exp_result, exp_expected, 0.01f) &&
           fixture.verify_output(sqrt_result, sqrt_expected, 0.01f) &&
           fixture.verify_output(cos_result, cos_expected, 0.01f) &&
           fixture.verify_output(sin_result, sin_expected, 0.01f);
}

bool test_rms_norm() {
    TestUtils::FP16TestFixture fixture("RMS Norm");

    size_t input_a = fixture.create_input({1, 8});
    size_t weight = fixture.create_input({8});
    size_t norm_result = fixture.graph().rms_norm(input_a, weight);

    std::vector<__fp16> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<__fp16> weight_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

    fixture.set_input_data(input_a, input_data);
    fixture.set_input_data(weight, weight_data);
    fixture.execute();

    float sum_squares = 0.0f;
    for (auto val : input_data) {
        float v = static_cast<float>(val);
        sum_squares += v * v;
    }
    float rms = sqrtf(sum_squares / 8.0f + 1e-5f);
    float inv_rms = 1.0f / rms;

    std::vector<__fp16> expected;
    for (size_t i = 0; i < input_data.size(); i++) {
        expected.push_back(static_cast<__fp16>(static_cast<float>(input_data[i]) * inv_rms * static_cast<float>(weight_data[i])));
    }

    return fixture.verify_output(norm_result, expected, 0.01f);
}

bool test_softmax() {
    TestUtils::FP16TestFixture fixture("Softmax");

    size_t input_a = fixture.create_input({2, 3});
    size_t softmax_result = fixture.graph().softmax(input_a, -1);

    std::vector<__fp16> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    fixture.set_input_data(input_a, input_data);
    fixture.execute();

    std::vector<__fp16> expected = {0.09003f, 0.24473f, 0.66524f, 0.09003f, 0.24473f, 0.66524f};
    return fixture.verify_output(softmax_result, expected, 0.01f);
}

bool test_attention() {
    TestUtils::FP16TestFixture fixture("Attention");

    size_t query = fixture.create_input({1, 2, 1, 4});
    size_t key = fixture.create_input({1, 2, 1, 4});
    size_t value = fixture.create_input({1, 2, 1, 4});

    size_t attention_result = fixture.graph().attention(query, key, value, 0.5f);
    (void)attention_result;

    std::vector<__fp16> q_data = {1, 0, 0, 0, 0, 1, 0, 0};
    std::vector<__fp16> k_data = {1, 0, 0, 0, 0, 1, 0, 0};
    std::vector<__fp16> v_data = {1, 2, 3, 4, 5, 6, 7, 8};

    fixture.set_input_data(query, q_data);
    fixture.set_input_data(key, k_data);
    fixture.set_input_data(value, v_data);
    fixture.execute();

    return true;
}

bool test_reduction_operations() {
    TestUtils::FP16TestFixture fixture("Reduction Operations");

    size_t input_a = fixture.create_input({2, 3});
    size_t sum_all = fixture.graph().sum(input_a, -1);
    size_t sum_axis0 = fixture.graph().sum(input_a, 0);
    size_t sum_axis1 = fixture.graph().sum(input_a, 1);

    std::vector<__fp16> data_a = {1, 2, 3, 4, 5, 6};
    fixture.set_input_data(input_a, data_a);
    fixture.execute();

    std::vector<__fp16> expected_all = {21};
    std::vector<__fp16> expected_axis0 = {5, 7, 9};
    std::vector<__fp16> expected_axis1 = {6, 15};

    return fixture.verify_output(sum_all, expected_all) &&
           fixture.verify_output(sum_axis0, expected_axis0) &&
           fixture.verify_output(sum_axis1, expected_axis1);
}

bool test_fp16_reduction_operations() {
    CactusGraph graph;

    size_t input_a = graph.input({2, 3}, Precision::FP16);
    size_t sum_all = graph.sum(input_a, -1);

    std::vector<__fp16> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    graph.set_input(input_a, input_data.data(), Precision::FP16);
    graph.execute();

    __fp16* output = static_cast<__fp16*>(graph.get_output(sum_all));
    double result = static_cast<double>(output[0]);
    double expected = 21.0;

    bool success = std::abs(result - expected) < 0.1f; // FP16 has lower precision

    graph.hard_reset();
    return success;
}

bool test_mean_operations() {
    TestUtils::FP16TestFixture fixture("Mean Operations");

    size_t input_a = fixture.create_input({2, 4});
    size_t mean_all = fixture.graph().mean(input_a, -1);
    size_t mean_axis0 = fixture.graph().mean(input_a, 0);
    size_t mean_axis1 = fixture.graph().mean(input_a, 1);

    std::vector<__fp16> data_a = {2, 4, 6, 8, 10, 12, 14, 16};
    fixture.set_input_data(input_a, data_a);
    fixture.execute();

    std::vector<__fp16> expected_all = {9};
    std::vector<__fp16> expected_axis0 = {6, 8, 10, 12};
    std::vector<__fp16> expected_axis1 = {5, 13};

    return fixture.verify_output(mean_all, expected_all) &&
           fixture.verify_output(mean_axis0, expected_axis0) &&
           fixture.verify_output(mean_axis1, expected_axis1);
}

bool test_variance_operations() {
    TestUtils::FP16TestFixture fixture("Variance Operations");

    size_t input_a = fixture.create_input({1, 4});
    size_t var_axis1 = fixture.graph().variance(input_a, 1);

    std::vector<__fp16> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
    fixture.set_input_data(input_a, input_data);
    fixture.execute();

    std::vector<__fp16> expected = {1.25f};
    return fixture.verify_output(var_axis1, expected, 0.01f);
}

bool test_min_max_operations() {
    TestUtils::FP16TestFixture fixture("Min/Max Operations");

    size_t input_a = fixture.create_input({2, 3});
    size_t min_axis0 = fixture.graph().min(input_a, 0);
    size_t max_axis0 = fixture.graph().max(input_a, 0);
    size_t min_axis1 = fixture.graph().min(input_a, 1);
    size_t max_axis1 = fixture.graph().max(input_a, 1);

    std::vector<__fp16> data_a = {6, 2, 8, 1, 5, 3};
    fixture.set_input_data(input_a, data_a);
    fixture.execute();

    std::vector<__fp16> expected_min_axis0 = {1, 2, 3};
    std::vector<__fp16> expected_max_axis0 = {6, 5, 8};
    std::vector<__fp16> expected_min_axis1 = {2, 1};
    std::vector<__fp16> expected_max_axis1 = {8, 5};

    return fixture.verify_output(min_axis0, expected_min_axis0) &&
           fixture.verify_output(max_axis0, expected_max_axis0) &&
           fixture.verify_output(min_axis1, expected_min_axis1) &&
           fixture.verify_output(max_axis1, expected_max_axis1);
}

bool test_fp16_precision() {
    TestUtils::FP16TestFixture fixture("FP16 Precision");

    size_t input_a = fixture.create_input({3});
    size_t input_b = fixture.create_input({3});
    size_t result_id = fixture.graph().add(input_a, input_b);

    std::vector<__fp16> data_a = {1.5f, 2.5f, 3.5f};
    std::vector<__fp16> data_b = {0.5f, 1.5f, 2.5f};
    fixture.set_input_data(input_a, data_a);
    fixture.set_input_data(input_b, data_b);
    fixture.execute();

    std::vector<__fp16> expected = {2.0f, 4.0f, 6.0f};
    return fixture.verify_output(result_id, expected);
}

bool test_broadcast_shape_compatibility() {
    TestUtils::FP16TestFixture fixture("Broadcast Shape Compatibility");

    size_t a_id = fixture.create_input({2, 3});
    size_t b_id = fixture.create_input({2, 1});

    std::vector<__fp16> data_a = {1, 2, 3, 4, 5, 6};
    std::vector<__fp16> data_b = {10, 20};
    fixture.set_input_data(a_id, data_a);
    fixture.set_input_data(b_id, data_b);

    size_t result_id = fixture.graph().add(a_id, b_id);
    fixture.execute();

    std::vector<__fp16> expected = {11, 12, 13, 24, 25, 26};
    return fixture.verify_output(result_id, expected);
}

bool test_broadcast_scalar_tensor() {
    TestUtils::FP16TestFixture fixture("Broadcast Scalar Tensor");

    size_t a_id = fixture.create_input({2, 2});
    size_t b_id = fixture.create_input({1});

    std::vector<__fp16> data_a = {1, 2, 3, 4};
    std::vector<__fp16> data_b = {5};
    fixture.set_input_data(a_id, data_a);
    fixture.set_input_data(b_id, data_b);

    size_t result_id = fixture.graph().add(a_id, b_id);
    fixture.execute();

    std::vector<__fp16> expected = {6, 7, 8, 9};
    return fixture.verify_output(result_id, expected);
}

bool test_broadcast_different_ranks() {
    TestUtils::FP16TestFixture fixture("Broadcast Different Ranks");

    size_t a_id = fixture.create_input({2, 2, 3});
    size_t b_id = fixture.create_input({2, 3});

    std::vector<__fp16> data_a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<__fp16> data_b = {1, 1, 1, 2, 2, 2};
    fixture.set_input_data(a_id, data_a);
    fixture.set_input_data(b_id, data_b);

    size_t result_id = fixture.graph().add(a_id, b_id);
    fixture.execute();

    std::vector<__fp16> expected = {2, 3, 4, 6, 7, 8, 8, 9, 10, 12, 13, 14};
    return fixture.verify_output(result_id, expected);
}

bool test_broadcast_fp16_precision() {
    TestUtils::FP16TestFixture fixture("Broadcast FP16 Precision");

    size_t a_id = fixture.create_input({2, 2});
    size_t b_id = fixture.create_input({1});

    std::vector<__fp16> data_a = {1.5f, 2.5f, 3.5f, 4.5f};
    std::vector<__fp16> data_b = {0.5f};
    fixture.set_input_data(a_id, data_a);
    fixture.set_input_data(b_id, data_b);

    size_t result_id = fixture.graph().add(a_id, b_id);
    fixture.execute();

    std::vector<__fp16> expected = {2.0f, 3.0f, 4.0f, 5.0f};
    return fixture.verify_output(result_id, expected);
}

bool test_precision_traits() {
    assert(PrecisionTraits::size_of(Precision::INT8) == 1);
    assert(PrecisionTraits::size_of(Precision::FP32) == 4);
    return true;
}

bool test_graph_precision_construction() {
    TestUtils::FP16TestFixture fixture("Graph Precision Construction");

    size_t fp16_id = fixture.create_input({2, 3}, Precision::FP16);
    size_t fp32_id = fixture.create_input({3, 4}, Precision::FP32);

    const auto& fp16_buffer = fixture.graph().get_output_buffer(fp16_id);
    const auto& fp32_buffer = fixture.graph().get_output_buffer(fp32_id);

    assert(fp16_buffer.precision == Precision::FP16);
    assert(fp16_buffer.shape[0] == 2);
    assert(fp16_buffer.shape[1] == 3);
    assert(fp16_buffer.byte_size == 12);  // 6 elements * 2 bytes

    assert(fp32_buffer.precision == Precision::FP32);
    assert(fp32_buffer.shape[0] == 3);
    assert(fp32_buffer.shape[1] == 4);
    assert(fp32_buffer.byte_size == 48);

    return true;
}

bool test_precision_conversion() {
    TestUtils::FP16TestFixture fixture("Precision Conversion");

    size_t fp16_id = fixture.create_input({2, 2}, Precision::FP16);
    std::vector<__fp16> data = {1, 2, 3, 4};
    fixture.set_input_data(fp16_id, data);

    size_t fp32_converted_id = fixture.graph().precision_cast(fp16_id, Precision::FP32);
    fixture.execute();

    float* fp32_data = static_cast<float*>(fixture.graph().get_output(fp32_converted_id));

    for (size_t i = 0; i < 4; ++i) {
        assert(std::abs(fp32_data[i] - static_cast<float>(data[i])) < 1e-3f);
    }

    return true;
}

bool test_graph_save_load() {
    try {
        CactusGraph graph;

        size_t input_a = graph.input({2, 3}, Precision::FP16);
        size_t input_b = graph.input({2, 3}, Precision::FP16);
        size_t result_id = graph.add(input_a, input_b);

        std::vector<__fp16> data_a = {1, 2, 3, 4, 5, 6};
        std::vector<__fp16> data_b = {10, 20, 30, 40, 50, 60};

        graph.set_input(input_a, const_cast<void*>(static_cast<const void*>(data_a.data())), Precision::FP16);
        graph.set_input(input_b, const_cast<void*>(static_cast<const void*>(data_b.data())), Precision::FP16);
        graph.execute();

        std::string filename = "test_graph_save_load.bin";
        GraphFile::save_node(graph, result_id, filename);

        CactusGraph new_graph;
        size_t loaded_id = new_graph.mmap_weights(filename);
        new_graph.execute();

        __fp16* original_data = static_cast<__fp16*>(graph.get_output(result_id));
        __fp16* loaded_data = static_cast<__fp16*>(new_graph.get_output(loaded_id));

        for (size_t i = 0; i < 6; ++i) {
            if (std::abs(static_cast<float>(original_data[i]) - static_cast<float>(loaded_data[i])) > 1e-3f) {
                graph.hard_reset();
                new_graph.hard_reset();
                std::remove(filename.c_str());
                return false;
            }
        }

        const auto& buf = new_graph.get_output_buffer(loaded_id);
        bool result = (buf.shape == std::vector<size_t>{2, 3}) &&
                     (buf.precision == Precision::FP16) &&
                     (buf.byte_size == 12);

        graph.hard_reset();
        new_graph.hard_reset();
        std::remove(filename.c_str());
        return result;
    } catch (const std::exception& e) {
        return false;
    }
}

bool test_complex_graph_structure() {
    TestUtils::FP16TestFixture fixture("Complex Graph Structure");

    size_t input_a = fixture.create_input({2, 2});
    size_t input_b = fixture.create_input({2, 2});
    size_t input_c = fixture.create_input({2, 2});

    size_t add_ab = fixture.graph().add(input_a, input_b);
    size_t mul_result = fixture.graph().multiply(add_ab, input_c);
    size_t scalar_result = fixture.graph().scalar_add(mul_result, 1.0f);

    std::vector<__fp16> data_a = {1, 2, 3, 4};
    std::vector<__fp16> data_b = {2, 3, 4, 5};
    std::vector<__fp16> data_c = {2, 2, 2, 2};
    fixture.set_input_data(input_a, data_a);
    fixture.set_input_data(input_b, data_b);
    fixture.set_input_data(input_c, data_c);

    fixture.execute();

    std::vector<__fp16> expected = {7, 11, 15, 19};
    return fixture.verify_output(scalar_result, expected);
}

bool test_multiple_outputs() {
    TestUtils::FP16TestFixture fixture("Multiple Outputs");

    size_t input_a = fixture.create_input({3});
    size_t add_result = fixture.graph().scalar_add(input_a, 10.0f);
    size_t mul_result = fixture.graph().scalar_multiply(input_a, 2.0f);
    size_t combine_result = fixture.graph().add(add_result, mul_result);

    std::vector<__fp16> data_a = {1, 2, 3};
    fixture.set_input_data(input_a, data_a);
    fixture.execute();

    std::vector<__fp16> expected_add = {11, 12, 13};
    std::vector<__fp16> expected_mul = {2, 4, 6};
    std::vector<__fp16> expected_combine = {13, 16, 19};

    return fixture.verify_output(add_result, expected_add) &&
           fixture.verify_output(mul_result, expected_mul) &&
           fixture.verify_output(combine_result, expected_combine);
}

bool test_graph_reset() {
    CactusGraph graph;

    size_t input_a = graph.input({2}, Precision::FP16);
    size_t result_id = graph.scalar_add(input_a, 5.0f);

    std::vector<__fp16> data_a = {1, 2};
    graph.set_input(input_a, data_a.data(), Precision::FP16);
    graph.execute();

    __fp16* output1 = static_cast<__fp16*>(graph.get_output(result_id));
    if (std::abs(static_cast<float>(output1[0]) - 6.0f) > 1e-2f ||
        std::abs(static_cast<float>(output1[1]) - 7.0f) > 1e-2f) return false;

    graph.hard_reset();
    if (graph.get_node_count() != 0) return false;

    size_t new_input = graph.input({2}, Precision::FP16);
    size_t new_result = graph.scalar_add(new_input, 5.0f);

    std::vector<__fp16> data_b = {10, 20};
    graph.set_input(new_input, data_b.data(), Precision::FP16);
    graph.execute();

    __fp16* output2 = static_cast<__fp16*>(graph.get_output(new_result));
    return (std::abs(static_cast<float>(output2[0]) - 15.0f) < 1e-2f &&
            std::abs(static_cast<float>(output2[1]) - 25.0f) < 1e-2f);
}

bool test_gather_operation() {
    // Gather uses INT8 indices but can work with FP16 data
    CactusGraph graph;

    size_t embeddings = graph.input({5, 3}, Precision::FP16);
    size_t indices = graph.input({2, 2}, Precision::INT8);
    size_t gathered = graph.gather(embeddings, indices);

    std::vector<__fp16> emb_data = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12,
        13, 14, 15
    };
    std::vector<int8_t> idx_data = {0, 2, 4, 1};

    graph.set_input(embeddings, emb_data.data(), Precision::FP16);
    graph.set_input(indices, idx_data.data(), Precision::INT8);
    graph.execute();

    __fp16* output = static_cast<__fp16*>(graph.get_output(gathered));
    std::vector<__fp16> expected = {
        1, 2, 3,
        7, 8, 9,
        13, 14, 15,
        4, 5, 6
    };

    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::abs(static_cast<float>(output[i]) - static_cast<float>(expected[i])) > 1e-3f) {
            graph.hard_reset();
            return false;
        }
    }

    graph.hard_reset();
    return true;
}

bool test_gather_1d_tensor() {
    CactusGraph graph;

    size_t tensor = graph.input({8}, Precision::FP16);
    size_t indices = graph.input({3}, Precision::INT8);
    size_t gathered = graph.gather(tensor, indices);

    std::vector<__fp16> tensor_data = {10, 20, 30, 40, 50, 60, 70, 80};
    std::vector<int8_t> idx_data = {7, 2, 0};

    graph.set_input(tensor, tensor_data.data(), Precision::FP16);
    graph.set_input(indices, idx_data.data(), Precision::INT8);
    graph.execute();

    __fp16* output = static_cast<__fp16*>(graph.get_output(gathered));
    std::vector<__fp16> expected = {80, 30, 10};

    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::abs(static_cast<float>(output[i]) - static_cast<float>(expected[i])) > 1e-3f) {
            graph.hard_reset();
            return false;
        }
    }

    graph.hard_reset();
    return true;
}

bool test_gather_3d_tensor() {
    TestUtils::FP16TestFixture fixture("Gather 3D Tensor");

    size_t tensor = fixture.create_input({3, 2, 4});
    size_t indices = fixture.graph().input({2}, Precision::INT8);
    size_t gathered = fixture.graph().gather(tensor, indices);

    std::vector<__fp16> tensor_data = {
        // First 2x4 slice
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        // Second 2x4 slice
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f,
        // Third 2x4 slice
        17.0f, 18.0f, 19.0f, 20.0f,
        21.0f, 22.0f, 23.0f, 24.0f
    };
    std::vector<int8_t> idx_data = {2, 0};

    fixture.set_input_data(tensor, tensor_data);
    fixture.graph().set_input(indices, idx_data.data(), Precision::INT8);
    fixture.execute();

    std::vector<__fp16> expected = {
        // Third slice (index 2)
        17.0f, 18.0f, 19.0f, 20.0f,
        21.0f, 22.0f, 23.0f, 24.0f,
        // First slice (index 0)
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f
    };

    return fixture.verify_output(gathered, expected);
}

bool test_gather_fp16() {
    TestUtils::FP16TestFixture fixture("Gather FP16");

    size_t embeddings = fixture.create_input({4, 2});
    CactusGraph& graph = fixture.graph();
    size_t indices = graph.input({3}, Precision::INT8);
    size_t gathered = graph.gather(embeddings, indices);

    std::vector<__fp16> emb_data = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
        7.0f, 8.0f
    };
    std::vector<int8_t> idx_data = {2, 0, 3};

    fixture.set_input_data(embeddings, emb_data);
    graph.set_input(indices, idx_data.data(), Precision::INT8);
    fixture.execute();

    std::vector<__fp16> expected = {
        5.0f, 6.0f,
        1.0f, 2.0f,
        7.0f, 8.0f
    };

    return fixture.verify_output(gathered, expected);
}

bool test_mmap_gather() {
    CactusGraph graph;

    std::vector<__fp16> embeddings_data = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
        10.0f, 11.0f, 12.0f
    };

    size_t temp_embeddings = graph.input({4, 3}, Precision::FP16);
    graph.set_input(temp_embeddings, embeddings_data.data(), Precision::FP16);

    const std::string temp_file = "test_embeddings.bin";
    GraphFile::save_node(graph, temp_embeddings, temp_file);

    graph.hard_reset();

    size_t mmap_embeddings = graph.mmap_embeddings(temp_file);
    size_t indices = graph.input({3}, Precision::INT8);
    size_t gathered = graph.gather(mmap_embeddings, indices);

    std::vector<int8_t> idx_data = {2, 0, 3};
    graph.set_input(indices, idx_data.data(), Precision::INT8);
    graph.execute();

    std::vector<__fp16> expected = {
        7.0f, 8.0f, 9.0f,
        1.0f, 2.0f, 3.0f,
        10.0f, 11.0f, 12.0f
    };

    __fp16* output = static_cast<__fp16*>(graph.get_output(gathered));
    bool passed = true;
    for (size_t i = 0; i < expected.size(); i++) {
        if (std::abs(static_cast<float>(output[i]) - static_cast<float>(expected[i])) > 0.01f) {
            passed = false;
            break;
        }
    }

    std::remove(temp_file.c_str());

    return passed;
}

bool test_embedding_operation() {
    CactusGraph graph;

    const size_t vocab_size = 4;
    const size_t hidden_dim = 8;
    const size_t group_size = 8;
    const size_t num_groups = hidden_dim / group_size;
    const size_t BLOCK_SIZE = 4;

    std::vector<int8_t> emb_rowmajor(vocab_size * hidden_dim);
    for (size_t row = 0; row < vocab_size; ++row) {
        for (size_t k = 0; k < hidden_dim; ++k) {
            emb_rowmajor[row * hidden_dim + k] = static_cast<int8_t>((row + 1) * 10 + k);
        }
    }

    std::vector<int8_t> emb_interleaved(vocab_size * hidden_dim);
    size_t N_blocks = vocab_size / BLOCK_SIZE;
    size_t K_groups = hidden_dim / BLOCK_SIZE;

    for (size_t n_blk = 0; n_blk < N_blocks; ++n_blk) {
        for (size_t k_grp = 0; k_grp < K_groups; ++k_grp) {
            for (size_t lane = 0; lane < BLOCK_SIZE; ++lane) {
                for (size_t k_within = 0; k_within < BLOCK_SIZE; ++k_within) {
                    size_t src_row = n_blk * BLOCK_SIZE + lane;
                    size_t src_k = k_grp * BLOCK_SIZE + k_within;
                    size_t dst_idx = (n_blk * K_groups + k_grp) * 16 + lane * 4 + k_within;
                    emb_interleaved[dst_idx] = emb_rowmajor[src_row * hidden_dim + src_k];
                }
            }
        }
    }

    std::vector<__fp16> scales_rowmajor(vocab_size * num_groups);
    for (size_t i = 0; i < scales_rowmajor.size(); ++i) {
        scales_rowmajor[i] = static_cast<__fp16>(1.0f);
    }

    std::vector<__fp16> scales_interleaved(vocab_size * num_groups);
    for (size_t n_blk = 0; n_blk < N_blocks; ++n_blk) {
        for (size_t g = 0; g < num_groups; ++g) {
            for (size_t lane = 0; lane < BLOCK_SIZE; ++lane) {
                size_t src_row = n_blk * BLOCK_SIZE + lane;
                size_t dst_idx = (n_blk * num_groups + g) * BLOCK_SIZE + lane;
                scales_interleaved[dst_idx] = scales_rowmajor[src_row * num_groups + g];
            }
        }
    }

    size_t embeddings = graph.input({vocab_size, hidden_dim}, Precision::INT8);
    graph.set_input(embeddings, emb_interleaved.data(), Precision::INT8);

    graph.set_grouped_scales(embeddings, group_size, num_groups, scales_interleaved.data());
    graph.set_interleaved(embeddings, true, vocab_size);

    size_t indices = graph.input({4}, Precision::INT8);
    size_t embedded = graph.embedding(embeddings, indices);

    std::vector<int8_t> idx_data = {0, 2, 3, 1};  
    graph.set_input(indices, idx_data.data(), Precision::INT8);
    graph.execute();

    __fp16* output = static_cast<__fp16*>(graph.get_output(embedded));

    std::vector<float> expected = {
        10, 11, 12, 13, 14, 15, 16, 17,  // idx 0
        30, 31, 32, 33, 34, 35, 36, 37,  // idx 2
        40, 41, 42, 43, 44, 45, 46, 47,  // idx 3
        20, 21, 22, 23, 24, 25, 26, 27   // idx 1
    };

    for (size_t i = 0; i < expected.size(); ++i) {
        float out_val = static_cast<float>(output[i]);
        if (std::abs(out_val - expected[i]) > 0.5f) {
            std::cerr << "Embedding mismatch at " << i << ": got " << out_val
                      << ", expected " << expected[i] << std::endl;
            return false;
        }
    }

    return true;
}

bool test_embedding_from_file() {
    CactusGraph graph;

    std::vector<__fp16> embeddings_data = {
        1.0f, 5.0f, 9.0f,
        2.0f, 6.0f, 10.0f,
        3.0f, 7.0f, 11.0f,
        4.0f, 8.0f, 12.0f
    };

    size_t temp_embeddings = graph.input({4, 3}, Precision::FP16);
    graph.set_input(temp_embeddings, embeddings_data.data(), Precision::FP16);

    const std::string temp_file = "test_embedding.bin";
    GraphFile::save_node(graph, temp_embeddings, temp_file);

    graph.hard_reset();

    size_t indices = graph.input({3}, Precision::INT8);
    size_t embedded = graph.embedding(temp_file, indices);

    std::vector<int8_t> idx_data = {2, 0, 3};
    graph.set_input(indices, idx_data.data(), Precision::INT8);
    graph.execute();

    std::vector<__fp16> expected = {
        3.0f, 7.0f, 11.0f,
        1.0f, 5.0f, 9.0f,
        4.0f, 8.0f, 12.0f
    };

    __fp16* output = static_cast<__fp16*>(graph.get_output(embedded));
    bool passed = true;
    for (size_t i = 0; i < expected.size(); i++) {
        if (std::abs(static_cast<float>(output[i]) - static_cast<float>(expected[i])) > 0.01f) {
            passed = false;
            break;
        }
    }

    std::remove(temp_file.c_str());

    return passed;
}

int main() {
    TestUtils::TestRunner runner("Graph Operations Tests");

    runner.run_test("Basic Operations", test_basic_operations());
    runner.run_test("Basic Addition", test_basic_addition());
    runner.run_test("Basic Subtraction", test_basic_subtraction());
    runner.run_test("Basic Multiplication", test_basic_multiplication());
    runner.run_test("Basic Division", test_basic_division());
    runner.run_test("Matrix Multiplication", test_matrix_multiplication());
    runner.run_test("Transpose", test_transpose());
    runner.run_test("Reshape", test_reshape());
    runner.run_test("Scalar Operations", test_scalar_operations());
    runner.run_test("Scalar Subtract/Divide", test_scalar_subtract_divide());
    runner.run_test("Scalar Math Functions", test_scalar_math_functions());
    runner.run_test("Reduction Operations", test_reduction_operations());
    runner.run_test("FP16 Reduction Operations", test_fp16_reduction_operations());
    runner.run_test("Mean Operations", test_mean_operations());
    runner.run_test("Variance Operations", test_variance_operations());
    runner.run_test("Min/Max Operations", test_min_max_operations());
    runner.run_test("RMS Norm", test_rms_norm());
    runner.run_test("Softmax", test_softmax());
    runner.run_test("Attention", test_attention());
    runner.run_test("FP16 Precision", test_fp16_precision());
    runner.run_test("Broadcast Shape Compatibility", test_broadcast_shape_compatibility());
    runner.run_test("Broadcast Scalar Tensor", test_broadcast_scalar_tensor());
    runner.run_test("Broadcast Different Ranks", test_broadcast_different_ranks());
    runner.run_test("Broadcast FP16 Precision", test_broadcast_fp16_precision());
    runner.run_test("Precision Traits", test_precision_traits());
    runner.run_test("Graph Precision Construction", test_graph_precision_construction());
    runner.run_test("Precision Conversion", test_precision_conversion());
    runner.run_test("Graph Save/Load", test_graph_save_load());
    runner.run_test("Complex Graph Structure", test_complex_graph_structure());
    runner.run_test("Multiple Outputs", test_multiple_outputs());
    runner.run_test("Graph Reset", test_graph_reset());
    runner.run_test("Gather Operation", test_gather_operation());
    runner.run_test("Gather 1D Tensor", test_gather_1d_tensor());
    runner.run_test("Gather 3D Tensor", test_gather_3d_tensor());
    runner.run_test("Gather FP16", test_gather_fp16());
    runner.run_test("Memory-Mapped Gather", test_mmap_gather());
    runner.run_test("Embedding Operation", test_embedding_operation());
    runner.run_test("Embedding from File", test_embedding_from_file());

    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
