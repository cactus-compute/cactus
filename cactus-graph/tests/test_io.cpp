#include "test_utils.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <cstdio>

using namespace TestUtils;

namespace {
template <typename T>
void print_vector_inline(const std::vector<T>& values) {
    std::cout << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << values[i];
    }
    std::cout << "]";
}
}

bool test_node_save_load() {
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

bool test_graph_save_load() {
    try {
        const std::string filename = "test_graph_save_load.cg";

        CactusGraph graph;
        size_t input_a = graph.input({2, 3}, Precision::FP16);
        size_t input_b = graph.input({2, 3}, Precision::FP16);
        size_t sum_id = graph.add(input_a, input_b);
        size_t pow_id = graph.pow(sum_id, 2.0f);

        graph.save(filename);

        GraphFile::SerializedGraph sg = GraphFile::load_graph(filename);

        if (sg.header.node_count != 4) {
            std::cout << "[graph_save_load] unexpected node_count: "
                      << sg.header.node_count << " expected 4" << std::endl;
            std::remove(filename.c_str());
            return false;
        }

        if (sg.graph_inputs.size() != 2 || sg.graph_inputs[0] != input_a ||
  sg.graph_inputs[1] != input_b) {
            std::cout << "[graph_save_load] unexpected graph_inputs:";
            for (uint32_t idx : sg.graph_inputs) {
                std::cout << " " << idx;
            }
            std::cout << std::endl;
            std::remove(filename.c_str());
            return false;
        }

        if (sg.graph_outputs.size() != 1 || sg.graph_outputs[0] != pow_id) {
            std::cout << "[graph_save_load] unexpected graph_outputs:";
            for (uint32_t idx : sg.graph_outputs) {
                std::cout << " " << idx;
            }
            std::cout << std::endl;
            std::remove(filename.c_str());
            return false;
        }

        CactusGraph loaded = CactusGraph::load(filename);

        std::vector<__fp16> data_a = {1, 2, 3, 4, 5, 6};
        std::vector<__fp16> data_b = {10, 20, 30, 40, 50, 60};

        const size_t loaded_input_a = runtime_id_from_serialized_index(sg.graph_inputs[0]);
        const size_t loaded_input_b = runtime_id_from_serialized_index(sg.graph_inputs[1]);
        const size_t loaded_output_id = runtime_id_from_serialized_index(sg.graph_outputs[0]);

        loaded.set_input(loaded_input_a, data_a.data(), Precision::FP16);
        loaded.set_input(loaded_input_b, data_b.data(), Precision::FP16);
        loaded.execute();

        __fp16* output = static_cast<__fp16*>(loaded.get_output(loaded_output_id));
        std::vector<float> expected = {
            121.0f, 484.0f, 1089.0f,
            1936.0f, 3025.0f, 4356.0f
        };

        for (size_t i = 0; i < expected.size(); ++i) {
            float got = static_cast<float>(output[i]);
            if (std::abs(got - expected[i]) > 1.5f) {
                std::cout << "[graph_save_load] mismatch at index " << i
                          << ": got=" << got
                          << " expected=" << expected[i] << std::endl;
                std::remove(filename.c_str());
                return false;
            }
        }

        std::remove(filename.c_str());
        return true;
    } catch (const std::exception& e) {
        std::cout << "[graph_save_load] exception: " << e.what() << std::endl;
        return false;
    }
}

bool test_graph_save_load_roundtrip_execution() {
    try {
        const std::string filename = "test_graph_save_load_roundtrip.cg";

        CactusGraph original;
        size_t input_a = original.input({2, 3}, Precision::FP16);
        size_t input_b = original.input({2, 3}, Precision::FP16);
        size_t sum_id = original.add(input_a, input_b);
        size_t pow_id = original.pow(sum_id, 2.0f);

        std::vector<__fp16> data_a = {1, 2, 3, 4, 5, 6};
        std::vector<__fp16> data_b = {10, 20, 30, 40, 50, 60};

        original.set_input(input_a, data_a.data(), Precision::FP16);
        original.set_input(input_b, data_b.data(), Precision::FP16);
        original.execute();

        __fp16* original_output = static_cast<__fp16*>(original.get_output(pow_id));
        std::vector<float> expected(6);
        for (size_t i = 0; i < expected.size(); ++i) {
            expected[i] = static_cast<float>(original_output[i]);
        }

        original.save(filename);

        CactusGraph loaded = CactusGraph::load(filename);
        GraphFile::SerializedGraph loaded_graph = GraphFile::load_graph(filename);
        const size_t loaded_input_a = runtime_id_from_serialized_index(loaded_graph.graph_inputs[0]);
        const size_t loaded_input_b = runtime_id_from_serialized_index(loaded_graph.graph_inputs[1]);
        const size_t loaded_output_id = runtime_id_from_serialized_index(loaded_graph.graph_outputs[0]);

        loaded.set_input(loaded_input_a, data_a.data(), Precision::FP16);
        loaded.set_input(loaded_input_b, data_b.data(), Precision::FP16);
        loaded.execute();

        __fp16* loaded_output = static_cast<__fp16*>(loaded.get_output(loaded_output_id));
        for (size_t i = 0; i < expected.size(); ++i) {
            float got = static_cast<float>(loaded_output[i]);
            if (std::abs(got - expected[i]) > 1e-3f) {
                std::cout << "[graph_save_load_roundtrip] mismatch at index " << i
                          << ": got=" << got
                          << " expected(original)=" << expected[i] << std::endl;
                std::remove(filename.c_str());
                return false;
            }
        }

        std::remove(filename.c_str());
        return true;
    } catch (const std::exception& e) {
        std::cout << "[graph_save_load_roundtrip] exception: " << e.what() << std::endl;
        return false;
    }
}

bool test_graph_save_load_supported_ops_roundtrip() {
    try {
        const std::string filename = "test_graph_supported_roundtrip.cg";

        CactusGraph original;

        size_t a = original.input({2, 3}, Precision::FP16);
        size_t b = original.input({2, 3}, Precision::FP16);
        size_t m1 = original.input({2, 3}, Precision::FP16);
        size_t m2 = original.input({3, 2}, Precision::FP16);

        size_t add_id = original.add(a, b);
        size_t add_clipped_id = original.add_clipped(a, b);
        size_t sub_id = original.subtract(a, b);
        size_t mul_id = original.multiply(a, b);
        size_t div_id = original.divide(a, b);

        size_t abs_id = original.abs(sub_id);
        size_t relu_id = original.relu(sub_id);
        size_t silu_id = original.silu(add_id);
        size_t gelu_id = original.gelu(add_id);
        size_t gelu_erf_id = original.gelu_erf(add_id);
        size_t sigmoid_id = original.sigmoid(sub_id);
        size_t tanh_id = original.tanh(sub_id);

        size_t pow_id = original.pow(abs_id, 2.0f);
        size_t scalar_add_id = original.scalar_add(add_id, 1.5f);
        size_t scalar_sub_id = original.scalar_subtract(add_id, 0.5f);
        size_t scalar_mul_id = original.scalar_multiply(add_id, 2.0f);
        size_t scalar_div_id = original.scalar_divide(add_id, 2.0f);

        size_t view_id = original.view(add_id, {3, 2});
        size_t reshape_id = original.reshape(add_id, {3, 2});
        size_t flatten_id = original.flatten(add_id);
        size_t concat_id = original.concat(a, b, 1);
        size_t cat_id = original.cat({a, b}, 1);

        size_t slice_id = original.slice(add_id, 1, 1, 2);
        size_t index_id = original.index(add_id, 1, 0);

        size_t sum_id = original.sum(add_id, -1);
        size_t mean_id = original.mean(add_id, 1);
        size_t variance_id = original.variance(add_id, 1);
        size_t min_id = original.min(add_id, 1);
        size_t max_id = original.max(add_id, 1);
        size_t softmax_id = original.softmax(add_id, 1);

        size_t matmul_id = original.matmul(m1, m2, false);
        size_t persistent_id = original.persistent(add_id);

        std::vector<__fp16> data_a = {1, 2, 3, 4, 5, 6};
        std::vector<__fp16> data_b = {6, 5, 4, 3, 2, 1};
        std::vector<__fp16> data_m1 = {1, 2, 3, 4, 5, 6};
        std::vector<__fp16> data_m2 = {1, 2, 3, 4, 5, 6};

        original.set_input(a, data_a.data(), Precision::FP16);
        original.set_input(b, data_b.data(), Precision::FP16);
        original.set_input(m1, data_m1.data(), Precision::FP16);
        original.set_input(m2, data_m2.data(), Precision::FP16);
        original.execute();

        std::vector<size_t> check_nodes = {
            add_id, add_clipped_id, sub_id, mul_id, div_id,
            abs_id, relu_id, silu_id, gelu_id, gelu_erf_id, sigmoid_id, tanh_id,
            pow_id, scalar_add_id, scalar_sub_id, scalar_mul_id, scalar_div_id,
            view_id, reshape_id, flatten_id, concat_id, cat_id,
            slice_id, index_id,
            sum_id, mean_id, variance_id, min_id, max_id, softmax_id,
            matmul_id, persistent_id
        };

        std::vector<std::vector<float>> expected_outputs;
        std::vector<std::vector<size_t>> expected_shapes;
        expected_outputs.reserve(check_nodes.size());
        expected_shapes.reserve(check_nodes.size());
        for (size_t node_id : check_nodes) {
            const auto& buf = original.get_output_buffer(node_id);
            __fp16* out = static_cast<__fp16*>(original.get_output(node_id));
            std::vector<float> values(buf.total_size);
            for (size_t i = 0; i < buf.total_size; ++i) {
                values[i] = static_cast<float>(out[i]);
            }
            expected_outputs.push_back(std::move(values));
            expected_shapes.push_back(buf.shape);
        }

        original.save(filename);

        CactusGraph loaded = CactusGraph::load(filename);
        GraphFile::SerializedGraph loaded_graph = GraphFile::load_graph(filename);
        if (loaded_graph.graph_inputs.size() != 4 || loaded_graph.graph_outputs.empty()) {
            std::remove(filename.c_str());
            return false;
        }

        loaded.set_input(runtime_id_from_serialized_index(loaded_graph.graph_inputs[0]), data_a.data(), Precision::FP16);
        loaded.set_input(runtime_id_from_serialized_index(loaded_graph.graph_inputs[1]), data_b.data(), Precision::FP16);
        loaded.set_input(runtime_id_from_serialized_index(loaded_graph.graph_inputs[2]), data_m1.data(), Precision::FP16);
        loaded.set_input(runtime_id_from_serialized_index(loaded_graph.graph_inputs[3]), data_m2.data(), Precision::FP16);
        loaded.execute();

        for (size_t node_idx = 0; node_idx < check_nodes.size(); ++node_idx) {
            size_t node_id = check_nodes[node_idx];
            const auto& loaded_buf = loaded.get_output_buffer(node_id);
            __fp16* loaded_out = static_cast<__fp16*>(loaded.get_output(node_id));

            if (loaded_buf.shape != expected_shapes[node_idx]) {
                std::cout << "[supported_roundtrip] shape mismatch for node " << node_id
                          << ": got=";
                print_vector_inline(loaded_buf.shape);
                std::cout << " expected=";
                print_vector_inline(expected_shapes[node_idx]);
                std::cout << std::endl;
                std::remove(filename.c_str());
                return false;
            }

            if (loaded_buf.total_size != expected_outputs[node_idx].size()) {
                std::cout << "[supported_roundtrip] size mismatch for node " << node_id
                          << ": got=" << loaded_buf.total_size
                          << " expected=" << expected_outputs[node_idx].size() << std::endl;
                std::remove(filename.c_str());
                return false;
            }

            for (size_t i = 0; i < loaded_buf.total_size; ++i) {
                float got = static_cast<float>(loaded_out[i]);
                float expected = expected_outputs[node_idx][i];
                if (std::abs(got - expected) > 1e-3f) {
                    std::cout << "[supported_roundtrip] mismatch for node " << node_id
                              << " at index " << i
                              << ": got=" << got
                              << " expected=" << expected << std::endl;
                    std::remove(filename.c_str());
                    return false;
                }
            }
        }

        std::remove(filename.c_str());
        return true;
    } catch (const std::exception& e) {
        std::cout << "[supported_roundtrip] exception: " << e.what() << std::endl;
        return false;
    }
}

bool test_graph_save_for_inspection() {
    try {
        const std::string filename = "test_graph_inspect.cg";

        CactusGraph graph;
        size_t input_a = graph.input({2, 3}, Precision::FP16);
        size_t input_b = graph.input({2, 3}, Precision::FP16);
        size_t sum_id = graph.add(input_a, input_b);
        size_t pow_id = graph.pow(sum_id, 2.0f);

        (void)input_a;
        (void)input_b;
        (void)sum_id;
        (void)pow_id;

        graph.save(filename);
        return true;
    } catch (const std::exception& e) {
        std::cout << "[graph_save_for_inspection] exception: " << e.what() << std::endl;
        return false;
    }
}

bool test_save_load_preserves_recurrent_cache_persistence() {
    try {
        const std::string filename = "test_recurrent_save_load.cg";

        CactusGraph graph;
        const std::vector<size_t> shape{2, 4};
        size_t cache = graph.recurrent_cache_state(shape, Precision::FP16);
        (void)cache;
        graph.save(filename);

        CactusGraph loaded = CactusGraph::load(filename);
        loaded.execute();

        size_t loaded_cache_id = 0;
        bool found = false;
        for (size_t i = 0; i < loaded.get_node_count(); ++i) {
            const auto& node = loaded.nodes_[i];
            if (node->op_type == OpType::RECURRENT_CACHE_STATE) {
                loaded_cache_id = node->id;
                found = true;
                break;
            }
        }
        if (!found) {
            std::remove(filename.c_str());
            return false;
        }

        const __fp16* data = static_cast<const __fp16*>(loaded.get_output(loaded_cache_id));
        if (!data) {
            std::remove(filename.c_str());
            return false;
        }
        for (size_t i = 0; i < 8; ++i) {
            if (static_cast<float>(data[i]) != 0.0f) {
                std::remove(filename.c_str());
                return false;
            }
        }

        loaded.release_runtime_buffers();
        const __fp16* still_there = static_cast<const __fp16*>(loaded.get_output(loaded_cache_id));
        bool persisted = still_there != nullptr;

        std::remove(filename.c_str());
        return persisted;
    } catch (const std::exception& e) {
        std::cout << "[recurrent_save_load] exception: " << e.what() << std::endl;
        return false;
    }
}

bool test_save_load_preserves_kv_cache_num_slots() {
    try {
        const std::string filename = "test_kv_slots_save_load.cg";

        CactusGraph graph;
        size_t cache = graph.kv_cache_state(128, 4, 64, 0, 4, 8);  // num_slots = 8
        (void)cache;
        graph.save(filename);

        CactusGraph loaded = CactusGraph::load(filename);
        size_t loaded_cache_id = 0;
        bool found = false;
        for (size_t i = 0; i < loaded.get_node_count(); ++i) {
            const auto& node = loaded.nodes_[i];
            if (node->op_type == OpType::KV_CACHE_STATE) {
                loaded_cache_id = node->id;
                found = true;
                break;
            }
        }
        std::remove(filename.c_str());
        if (!found) return false;
        return loaded.get_node_cache_num_slots(loaded_cache_id) == 8;
    } catch (const std::exception& e) {
        std::cout << "[kv_slots_save_load] exception: " << e.what() << std::endl;
        return false;
    }
}

bool run_benchmarks() {
    const int ITERS = 100;
    const std::string temp_file = "bench_io_50nodes.cg";

    // graph_save 50 nodes
    {
        double total = 0.0;
        for (int i = 0; i < ITERS; ++i) {
            CactusGraph graph;
            size_t node = graph.input({4}, Precision::FP16);
            size_t other = graph.input({4}, Precision::FP16);
            for (int n = 0; n < 50; ++n) {
                node = graph.add(node, other);
            }
            TestUtils::Timer t;
            graph.save(temp_file);
            total += t.elapsed_ms();
            graph.hard_reset();
        }
        std::remove(temp_file.c_str());
        double ms = total / ITERS;
        std::cout << "  \u26a1 " << std::left << std::setw(30) << "graph_save 50 nodes"
                  << std::fixed << std::setprecision(3) << ms << " ms\n";
    }

    // graph_load 50 nodes — build and save once, then load repeatedly
    {
        {
            CactusGraph graph;
            size_t node = graph.input({4}, Precision::FP16);
            size_t other = graph.input({4}, Precision::FP16);
            for (int n = 0; n < 50; ++n) {
                node = graph.add(node, other);
            }
            graph.save(temp_file);
        }
        double total = 0.0;
        for (int i = 0; i < ITERS; ++i) {
            TestUtils::Timer t;
            CactusGraph loaded = CactusGraph::load(temp_file);
            total += t.elapsed_ms();
            loaded.hard_reset();
        }
        std::remove(temp_file.c_str());
        double ms = total / ITERS;
        std::cout << "  \u26a1 " << std::left << std::setw(30) << "graph_load 50 nodes"
                  << std::fixed << std::setprecision(3) << ms << " ms\n";
    }

    return true;
}

int main() {
    TestUtils::TestRunner runner("IO / Serialization Tests");

    runner.run_test("Node Save/Load", test_node_save_load());
    runner.run_test("Graph Save/Load", test_graph_save_load());
    runner.run_test("Graph Save/Load Roundtrip Execution", test_graph_save_load_roundtrip_execution());
    runner.run_test("Graph Save/Load Supported Ops Roundtrip", test_graph_save_load_supported_ops_roundtrip());
    runner.run_test("Graph Save For Inspection", test_graph_save_for_inspection());
    runner.run_test("Save/Load Preserves Recurrent Cache Persistence",
                    test_save_load_preserves_recurrent_cache_persistence());
    runner.run_test("Save/Load Preserves KV Cache Num Slots",
                    test_save_load_preserves_kv_cache_num_slots());
    runner.print_benchmarks_header();
    runner.run_bench("benchmarks", run_benchmarks());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
