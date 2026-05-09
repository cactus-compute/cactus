#include "test_utils.h"
#include <cassert>
#include <memory>
#include <fstream>
#include <cstring>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <cstdio>

using namespace TestUtils;

bool test_abs() {
    TestUtils::FP16TestFixture fixture("absval");
    size_t input = fixture.create_input({2, 3});

    size_t abs_result = fixture.graph().abs(input);
    std::vector<__fp16> data = {1, -2, 3, 4, -5, -6};

    fixture.set_input_data(input, data);
    fixture.execute();

    std::vector<__fp16> expected = {1, 2, 3, 4, 5, 6};
    return fixture.verify_output(abs_result, expected);
}

bool test_concat() {
    TestUtils::FP16TestFixture fixture("Concat");

    size_t input_a = fixture.create_input({2, 3});
    size_t input_b = fixture.create_input({2, 5});
    size_t concat_result = fixture.graph().concat(input_a, input_b, 1);

    std::vector<__fp16> data_a = {1, 2, 3, 4, 5, 6};
    std::vector<__fp16> data_b = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<__fp16> expected = {1, 2, 3, 1, 2, 3, 4, 5, 4, 5, 6, 6, 7, 8, 9, 10};
    fixture.set_input_data(input_a, data_a);
    fixture.set_input_data(input_b, data_b);
    fixture.execute();

    return fixture.verify_output(concat_result, expected);
}

bool test_cat() {
    TestUtils::FP16TestFixture fixture("Cat (multiple input tensors)");

    size_t input_a = fixture.create_input({2, 3});
    size_t input_b = fixture.create_input({2, 5});
    size_t input_c = fixture.create_input({2, 2});

    size_t cat_result = fixture.graph().cat({input_a, input_b, input_c}, 1);

    std::vector<__fp16> data_a = {1, 2, 3,
                                  4, 5, 6};

    std::vector<__fp16> data_b = {1, 2, 3, 4, 5,
                                  6, 7, 8, 9, 10};

    std::vector<__fp16> data_c = {-1, -2,
                                  -1, -2};

    std::vector<__fp16> expected = {
        1, 2, 3, 1, 2, 3, 4, 5, -1, -2,
        4, 5, 6, 6, 7, 8, 9, 10, -1, -2
    };

    fixture.set_input_data(input_a, data_a);
    fixture.set_input_data(input_b, data_b);
    fixture.set_input_data(input_c, data_c);

    fixture.execute();

    return fixture.verify_output(cat_result, expected);
}

bool test_view() {
    TestUtils::FP16TestFixture fixture("View");

    size_t input = fixture.create_input({2, 3});
    size_t view_result = fixture.graph().view(input, {3, 2});

    std::vector<__fp16> data = {1, 2, 3, 4, 5, 6};
    std::vector<__fp16> expected = {1, 2, 3, 4, 5, 6};

    fixture.set_input_data(input, data);
    fixture.execute();

    return fixture.verify_output(view_result, expected);
}

bool test_flatten() {
    TestUtils::FP16TestFixture fixture("Flatten");

    size_t input = fixture.create_input({2, 3});
    size_t flatten_result = fixture.graph().flatten(input);

    std::vector<__fp16> data = {1, 2, 3, 4, 5, 6};
    std::vector<__fp16> expected = {1, 2, 3, 4, 5, 6};

    fixture.set_input_data(input, data);
    fixture.execute();

    return fixture.verify_output(flatten_result, expected);
}

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
    size_t log_result = fixture.graph().scalar_log(input_a);

    std::vector<__fp16> input_data = {0.5f, 1.0f, 4.0f};
    fixture.set_input_data(input_a, input_data);
    fixture.execute();

    std::vector<__fp16> exp_expected = {1.64872f, 2.71828f, 54.5982f};
    std::vector<__fp16> sqrt_expected = {0.70711f, 1.0f, 2.0f};
    std::vector<__fp16> cos_expected = {0.87758f, 0.54030f, -0.65364f};
    std::vector<__fp16> sin_expected = {0.47943f, 0.84147f, -0.75680f};
    std::vector<__fp16> log_expected = {-0.69315f, 0.0f, 1.38629f};

    return fixture.verify_output(exp_result, exp_expected, 0.01f) &&
           fixture.verify_output(sqrt_result, sqrt_expected, 0.01f) &&
           fixture.verify_output(cos_result, cos_expected, 0.01f) &&
           fixture.verify_output(sin_result, sin_expected, 0.01f) &&
           fixture.verify_output(log_result, log_expected, 0.01f);
}

bool test_scalar_operations_with_pow() {
    TestUtils::FP16TestFixture fixture("Scalar Operations with Pow");

    size_t input_a = fixture.create_input({4});
    size_t add_result = fixture.graph().scalar_add(input_a, 5.0f);
    size_t mul_result = fixture.graph().scalar_multiply(add_result, 2.0f);
    size_t pow_result = fixture.graph().pow(mul_result, 2.0f);

    std::vector<__fp16> data_a = {1, 2, 3, 4};
    fixture.set_input_data(input_a, data_a);
    fixture.execute();

    std::vector<__fp16> expected = {144, 196, 256, 324};
    return fixture.verify_output(pow_result, expected);
}

bool test_gather_operation() {
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
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f,
        17.0f, 18.0f, 19.0f, 20.0f,
        21.0f, 22.0f, 23.0f, 24.0f
    };
    std::vector<int8_t> idx_data = {2, 0};

    fixture.set_input_data(tensor, tensor_data);
    fixture.graph().set_input(indices, idx_data.data(), Precision::INT8);
    fixture.execute();

    std::vector<__fp16> expected = {
        17.0f, 18.0f, 19.0f, 20.0f,
        21.0f, 22.0f, 23.0f, 24.0f,
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

    std::vector<__fp16> emb_data(vocab_size * hidden_dim);
    for (size_t row = 0; row < vocab_size; ++row) {
        for (size_t k = 0; k < hidden_dim; ++k) {
            emb_data[row * hidden_dim + k] = static_cast<__fp16>((row + 1) * 10 + k);
        }
    }

    size_t embeddings = graph.input({vocab_size, hidden_dim}, Precision::FP16);
    graph.set_input(embeddings, emb_data.data(), Precision::FP16);

    size_t indices = graph.input({4}, Precision::INT8);
    size_t embedded = graph.embedding(embeddings, indices);

    std::vector<int8_t> idx_data = {0, 2, 3, 1};
    graph.set_input(indices, idx_data.data(), Precision::INT8);
    graph.execute();

    __fp16* output = static_cast<__fp16*>(graph.get_output(embedded));

    std::vector<float> expected = {
        10, 11, 12, 13, 14, 15, 16, 17,
        30, 31, 32, 33, 34, 35, 36, 37,
        40, 41, 42, 43, 44, 45, 46, 47,
        20, 21, 22, 23, 24, 25, 26, 27
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

bool test_cq_embedding_operation() {
    const uint32_t bits = 2;
    const uint32_t vocab_size = 4;
    const uint32_t hidden_dim = 16;
    const uint32_t group_size = 16;  
    const uint32_t num_groups = hidden_dim / group_size;
    const uint32_t cb_size = 1u << bits; 

    std::vector<__fp16> codebook(cb_size);
    codebook[0] = static_cast<__fp16>(-1.0f);
    codebook[1] = static_cast<__fp16>(-0.5f);
    codebook[2] = static_cast<__fp16>(0.5f);
    codebook[3] = static_cast<__fp16>(1.0f);

    std::vector<__fp16> input_scale_recip(hidden_dim, static_cast<__fp16>(1.0f));

    std::vector<__fp16> norms(vocab_size * num_groups);
    for (uint32_t r = 0; r < vocab_size; r++)
        for (uint32_t g = 0; g < num_groups; g++)
            norms[r * num_groups + g] = static_cast<__fp16>(1.0f);

    std::vector<int8_t> left_signs(group_size, 1);
    std::vector<int8_t> right_signs(group_size, 1);

    std::vector<uint32_t> permutation(group_size);
    for (uint32_t i = 0; i < group_size; i++) permutation[i] = i;
    uint32_t pgb = cactus_quant_packed_group_bytes(bits, group_size);
    std::vector<uint8_t> packed(vocab_size * num_groups * pgb, 0);
    for (uint32_t row = 0; row < vocab_size; row++) {
        uint8_t idx = static_cast<uint8_t>(row % cb_size);
        for (uint32_t g = 0; g < num_groups; g++) {
            uint8_t* p = packed.data() + (row * num_groups + g) * pgb;
            // Pack: each byte holds 4 x 2-bit indices
            for (uint32_t byte_i = 0; byte_i < pgb; byte_i++) {
                p[byte_i] = static_cast<uint8_t>(idx | (idx << 2) | (idx << 4) | (idx << 6));
            }
        }
    }

    std::vector<__fp16> expected(vocab_size * hidden_dim);
    for (uint32_t row = 0; row < vocab_size; row++) {
        cactus_quant_dequantize_hadamard_embedding_row(
            bits, hidden_dim, group_size, num_groups, row,
            packed.data(), codebook.data(), norms.data(),
            input_scale_recip.data(), left_signs.data(), right_signs.data(),
            permutation.data(), expected.data() + row * hidden_dim);
    }

    CactusGraph graph;

    size_t emb_node = graph.input({vocab_size, hidden_dim}, Precision::CQ2);
    graph.set_input(emb_node, packed.data(), Precision::CQ2);

    auto& buffer = graph.nodes_[graph.node_index_map_[emb_node]]->output_buffer;
    buffer.group_size = group_size;
    buffer.num_groups = num_groups;
    buffer.cq_codebook = codebook.data();
    buffer.cq_input_scale = nullptr;
    buffer.cq_input_scale_recip = input_scale_recip.data();
    buffer.cq_norms = norms.data();
    buffer.cq_left_signs = left_signs.data();
    buffer.cq_right_signs = right_signs.data();
    buffer.cq_permutation = permutation.data();
    buffer.cq_rotation = nullptr;
    buffer.cq_flags = 0;

    size_t indices = graph.input({3}, Precision::INT8);
    size_t embedded = graph.embedding(emb_node, indices);

    std::vector<int8_t> idx_data = {0, 2, 3};
    graph.set_input(indices, idx_data.data(), Precision::INT8);
    graph.execute();

    __fp16* output = static_cast<__fp16*>(graph.get_output(embedded));

    size_t lookup[] = {0, 2, 3};
    for (size_t i = 0; i < 3; i++) {
        const __fp16* exp_row = expected.data() + lookup[i] * hidden_dim;
        for (size_t k = 0; k < hidden_dim; k++) {
            float out_val = static_cast<float>(output[i * hidden_dim + k]);
            float exp_val = static_cast<float>(exp_row[k]);
            if (std::abs(out_val - exp_val) > 0.01f) {
                std::cerr << "CQ Embedding mismatch at [" << i << "," << k << "]: got "
                          << out_val << ", expected " << exp_val << std::endl;
                return false;
            }
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

bool run_benchmarks() {
    const size_t N = 1024 * 1024;
    std::vector<__fp16> data_a(N), data_b(N);
    TestUtils::fill_random_fp16(data_a);
    TestUtils::fill_random_fp16(data_b);

    auto bench = [](const char* label, auto fn) {
        fn();
        TestUtils::Timer t;
        for (int i = 0; i < 100; i++) fn();
        double ms = t.elapsed_ms() / 100.0;
        std::cout << "  ⚡ " << std::left << std::setw(30) << label
                  << std::fixed << std::setprecision(3) << ms << " ms\n";
    };

    bench("add 1M (graph)", [&]{
        CactusGraph g;
        size_t a = g.input({N}, Precision::FP16);
        size_t b = g.input({N}, Precision::FP16);
        g.add(a, b);
        g.set_input(a, data_a.data(), Precision::FP16);
        g.set_input(b, data_b.data(), Precision::FP16);
        g.execute();
    });

    bench("scalar_multiply 1M (graph)", [&]{
        CactusGraph g;
        size_t a = g.input({N}, Precision::FP16);
        g.scalar_multiply(a, 2.0f);
        g.set_input(a, data_a.data(), Precision::FP16);
        g.execute();
    });

    return true;
}

int main() {
    TestUtils::TestRunner runner("Ops Tests");

    runner.run_test("Abs Operation", test_abs());
    runner.run_test("Concat Operation", test_concat());
    runner.run_test("Cat Operation", test_cat());
    runner.run_test("View Operation", test_view());
    runner.run_test("Flatten Operation", test_flatten());
    runner.run_test("Basic Operations", test_basic_operations());
    runner.run_test("Basic Addition", test_basic_addition());
    runner.run_test("Basic Subtraction", test_basic_subtraction());
    runner.run_test("Basic Multiplication", test_basic_multiplication());
    runner.run_test("Basic Division", test_basic_division());
    runner.run_test("Scalar Operations", test_scalar_operations());
    runner.run_test("Scalar Subtract/Divide", test_scalar_subtract_divide());
    runner.run_test("Scalar Math Functions", test_scalar_math_functions());
    runner.run_test("Scalar Operations with Pow", test_scalar_operations_with_pow());
    runner.run_test("Gather Operation", test_gather_operation());
    runner.run_test("Gather 1D Tensor", test_gather_1d_tensor());
    runner.run_test("Gather 3D Tensor", test_gather_3d_tensor());
    runner.run_test("Gather FP16", test_gather_fp16());
    runner.run_test("Memory-Mapped Gather", test_mmap_gather());
    runner.run_test("Embedding Operation", test_embedding_operation());
    runner.run_test("CQ Embedding Operation", test_cq_embedding_operation());
    runner.run_test("Embedding from File", test_embedding_from_file());
    runner.print_benchmarks_header();
    runner.run_bench("benchmarks", run_benchmarks());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
