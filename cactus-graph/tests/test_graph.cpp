#include "test_utils.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <iomanip>

using namespace TestUtils;

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

    fixture.graph().retain_outputs({static_cast<int>(add_result), static_cast<int>(mul_result)});

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

bool run_benchmarks() {
    std::vector<__fp16> data(4, static_cast<__fp16>(1.0f));

    auto build_chain = [&](CactusGraph& g) {
        size_t a = g.input({4}, Precision::FP16);
        size_t b = g.input({4}, Precision::FP16);
        g.set_input(a, data.data(), Precision::FP16);
        g.set_input(b, data.data(), Precision::FP16);
        size_t node = a;
        for (int n = 0; n < 100; ++n) node = g.add(node, b);
        return node;
    };

    auto bench = [](const char* label, auto fn) {
        fn();
        TestUtils::Timer t;
        for (int i = 0; i < 100; i++) fn();
        double ms = t.elapsed_ms() / 100.0;
        std::cout << "  ⚡ " << std::left << std::setw(30) << label
                  << std::fixed << std::setprecision(3) << ms << " ms\n";
    };

    bench("graph_construct 100 nodes", [&]{
        CactusGraph g;
        build_chain(g);
    });

    bench("graph_execute 100 nodes", [&]{
        CactusGraph g;
        build_chain(g);
        g.execute();
    });

    bench("graph_reset", [&]{
        CactusGraph g;
        build_chain(g);
        g.execute();
        g.hard_reset();
    });

    return true;
}

int main() {
    TestUtils::TestRunner runner("Graph Structure Tests");

    runner.run_test("Complex Graph Structure", test_complex_graph_structure());
    runner.run_test("Multiple Outputs", test_multiple_outputs());
    runner.run_test("Graph Reset", test_graph_reset());
    runner.print_benchmarks_header();
    runner.run_bench("benchmarks", run_benchmarks());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
