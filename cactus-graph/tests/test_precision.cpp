#include "test_utils.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <iomanip>

using namespace TestUtils;

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

bool test_broadcast_5d_expand_shape() {
    TestUtils::FP16TestFixture fixture("Broadcast 5D Expand Shape");

    size_t a_id = fixture.create_input({1, 1, 1, 1, 256});
    size_t b_id = fixture.create_input({1, 1, 4, 1, 256});

    std::vector<__fp16> data_a(256);
    std::vector<__fp16> data_b(4 * 256);
    std::vector<__fp16> expected(4 * 256);
    for (size_t i = 0; i < data_a.size(); ++i) {
        data_a[i] = static_cast<__fp16>(i % 17);
    }
    for (size_t i = 0; i < data_b.size(); ++i) {
        data_b[i] = static_cast<__fp16>(1);
        expected[i] = static_cast<__fp16>(static_cast<float>(data_a[i % 256]) + 1.0f);
    }
    fixture.set_input_data(a_id, data_a);
    fixture.set_input_data(b_id, data_b);

    size_t result_id = fixture.graph().add(a_id, b_id);
    fixture.execute();

    return fixture.verify_output(result_id, expected);
}

bool test_expand_5d_shape() {
    TestUtils::FP16TestFixture fixture("Expand 5D Shape");

    size_t input_id = fixture.create_input({1, 1, 1, 1, 256});

    std::vector<__fp16> data(256);
    std::vector<__fp16> expected(4 * 256);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<__fp16>(i % 13);
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] = data[i % 256];
    }
    fixture.set_input_data(input_id, data);

    size_t result_id = fixture.graph().expand(input_id, {1, 1, 4, 1, 256});
    fixture.execute();

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

    if (fixture.graph().get_output_buffer(fp16_id).precision != Precision::FP16) return false;
    if (fixture.graph().get_output_buffer(fp16_id).shape[0] != 2) return false;
    if (fixture.graph().get_output_buffer(fp16_id).shape[1] != 3) return false;
    if (fixture.graph().get_output_buffer(fp16_id).byte_size != 12) return false;

    if (fixture.graph().get_output_buffer(fp32_id).precision != Precision::FP32) return false;
    if (fixture.graph().get_output_buffer(fp32_id).shape[0] != 3) return false;
    if (fixture.graph().get_output_buffer(fp32_id).shape[1] != 4) return false;
    if (fixture.graph().get_output_buffer(fp32_id).byte_size != 48) return false;

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
        if (std::abs(fp32_data[i] - static_cast<float>(data[i])) >= 1e-3f) return false;
    }

    return true;
}

bool run_benchmarks() {
    const size_t N = 1024 * 1024;
    std::vector<__fp16> fp16_data(N);
    std::vector<__fp16> bcast_a(1024 * 1024), bcast_b(1024);
    TestUtils::fill_random_fp16(fp16_data);
    TestUtils::fill_random_fp16(bcast_a);
    TestUtils::fill_random_fp16(bcast_b);

    auto bench = [](const char* label, auto fn) {
        fn();
        TestUtils::Timer t;
        for (int i = 0; i < 100; i++) fn();
        double ms = t.elapsed_ms() / 100.0;
        std::cout << "  ⚡ " << std::left << std::setw(30) << label
                  << std::fixed << std::setprecision(3) << ms << " ms\n";
    };

    bench("precision_cast FP16->FP32 1M", [&]{
        CactusGraph g;
        size_t a = g.input({N}, Precision::FP16);
        g.precision_cast(a, Precision::FP32);
        g.set_input(a, fp16_data.data(), Precision::FP16);
        g.execute();
    });

    bench("broadcast_add 1024x1024", [&]{
        CactusGraph g;
        size_t a = g.input({1024, 1024}, Precision::FP16);
        size_t b = g.input({1, 1024}, Precision::FP16);
        g.add(a, b);
        g.set_input(a, bcast_a.data(), Precision::FP16);
        g.set_input(b, bcast_b.data(), Precision::FP16);
        g.execute();
    });

    return true;
}

int main() {
    TestUtils::TestRunner runner("Precision & Broadcast Tests");

    runner.run_test("FP16 Precision", test_fp16_precision());
    runner.run_test("Broadcast Shape Compatibility", test_broadcast_shape_compatibility());
    runner.run_test("Broadcast Scalar Tensor", test_broadcast_scalar_tensor());
    runner.run_test("Broadcast Different Ranks", test_broadcast_different_ranks());
    runner.run_test("Broadcast FP16 Precision", test_broadcast_fp16_precision());
    runner.run_test("Broadcast 5D Expand Shape", test_broadcast_5d_expand_shape());
    runner.run_test("Expand 5D Shape", test_expand_5d_shape());
    runner.run_test("Precision Traits", test_precision_traits());
    runner.run_test("Graph Precision Construction", test_graph_precision_construction());
    runner.run_test("Precision Conversion", test_precision_conversion());
    runner.print_benchmarks_header();
    runner.run_bench("benchmarks", run_benchmarks());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
