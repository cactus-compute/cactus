#include "../cactus/cactus.h"
#include "test_utils.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstring>
#include <cassert>

using namespace cactus::engine;
using namespace std;

bool test_sliding_window_basic() {
    const size_t num_layers = 2;
    const size_t max_seq = 2048;
    const size_t num_kv_heads = 8;
    const size_t head_dim = 64;
    const size_t window_size = 16;
    const size_t sink_size = 4;

    KVCache cache;
    cache.init(num_layers, max_seq, num_kv_heads, head_dim, Precision::FP32);
    cache.set_window_size(window_size, sink_size);

    CactusGraph graph;

    {
        size_t seq_len = 10;
        vector<size_t> k_nodes, v_nodes;

        for (size_t layer = 0; layer < num_layers; layer++) {
            size_t k_node = graph.input({seq_len, num_kv_heads, head_dim}, Precision::FP32);
            size_t v_node = graph.input({seq_len, num_kv_heads, head_dim}, Precision::FP32);

            vector<float> k_data(seq_len * num_kv_heads * head_dim, layer + 1.0f);
            vector<float> v_data(seq_len * num_kv_heads * head_dim, layer + 2.0f);
            graph.set_input(k_node, k_data.data(), Precision::FP32);
            graph.set_input(v_node, v_data.data(), Precision::FP32);

            k_nodes.push_back(k_node);
            v_nodes.push_back(v_node);
        }

        graph.execute();
        cache.update_from_graph(&graph, k_nodes, v_nodes, seq_len, num_layers, num_kv_heads, head_dim);

        assert(cache.get_effective_seq_len() == seq_len);
    }

    {
        size_t additional_seq = 6;
        vector<size_t> k_nodes, v_nodes;

        size_t total_seq = cache.get_effective_seq_len() + additional_seq;

        for (size_t layer = 0; layer < num_layers; layer++) {
            size_t k_node = graph.input({total_seq, num_kv_heads, head_dim}, Precision::FP32);
            size_t v_node = graph.input({total_seq, num_kv_heads, head_dim}, Precision::FP32);

            vector<float> k_data(total_seq * num_kv_heads * head_dim, layer + 10.0f);
            vector<float> v_data(total_seq * num_kv_heads * head_dim, layer + 20.0f);
            graph.set_input(k_node, k_data.data(), Precision::FP32);
            graph.set_input(v_node, v_data.data(), Precision::FP32);

            k_nodes.push_back(k_node);
            v_nodes.push_back(v_node);
        }

        graph.execute();
        cache.update_from_graph(&graph, k_nodes, v_nodes, additional_seq, num_layers, num_kv_heads, head_dim);

        assert(cache.get_effective_seq_len() == window_size);
    }

    {
        size_t additional_seq = 10; 
        vector<size_t> k_nodes, v_nodes;

        size_t total_seq = cache.get_effective_seq_len() + additional_seq;

        for (size_t layer = 0; layer < num_layers; layer++) {
            size_t k_node = graph.input({total_seq, num_kv_heads, head_dim}, Precision::FP32);
            size_t v_node = graph.input({total_seq, num_kv_heads, head_dim}, Precision::FP32);

            vector<float> k_data(total_seq * num_kv_heads * head_dim);
            vector<float> v_data(total_seq * num_kv_heads * head_dim);

            for (size_t i = 0; i < sink_size * num_kv_heads * head_dim; i++) {
                k_data[i] = 100.0f + layer;  
                v_data[i] = 200.0f + layer;
            }
            for (size_t i = sink_size * num_kv_heads * head_dim; i < k_data.size(); i++) {
                k_data[i] = 300.0f + layer;
                v_data[i] = 400.0f + layer;
            }

            graph.set_input(k_node, k_data.data(), Precision::FP32);
            graph.set_input(v_node, v_data.data(), Precision::FP32);

            k_nodes.push_back(k_node);
            v_nodes.push_back(v_node);
        }

        graph.execute();
        cache.update_from_graph(&graph, k_nodes, v_nodes, additional_seq, num_layers, num_kv_heads, head_dim);

        assert(cache.get_effective_seq_len() == window_size);

        float* key_data = static_cast<float*>(cache.get_key_ptr(0));
        if (key_data) {
            bool sink_preserved = true;
            for (size_t i = 0; i < sink_size * num_kv_heads * head_dim; i++) {
                if (key_data[i] != 100.0f) {
                    sink_preserved = false;
                    break;
                }
            }
            assert(sink_preserved);
        }

        assert(cache.get_total_seq_len() == 26);
    }

    return true;
}

bool test_incremental_updates() {
    const size_t num_layers = 1;
    const size_t num_kv_heads = 4;
    const size_t head_dim = 32;
    const size_t window_size = 8;
    const size_t sink_size = 2;

    KVCache cache;
    cache.init(num_layers, 2048, num_kv_heads, head_dim, Precision::FP32);
    cache.set_window_size(window_size, sink_size);

    CactusGraph graph;

    for (size_t token = 0; token < 12; token++) {
        vector<size_t> k_nodes, v_nodes;

        size_t seq_len = 1;
        size_t k_node = graph.input({seq_len, num_kv_heads, head_dim}, Precision::FP32);
        size_t v_node = graph.input({seq_len, num_kv_heads, head_dim}, Precision::FP32);

        vector<float> k_data(seq_len * num_kv_heads * head_dim, float(token));
        vector<float> v_data(seq_len * num_kv_heads * head_dim, float(token + 100));

        graph.set_input(k_node, k_data.data(), Precision::FP32);
        graph.set_input(v_node, v_data.data(), Precision::FP32);

        k_nodes.push_back(k_node);
        v_nodes.push_back(v_node);

        graph.execute();
        cache.update_from_graph(&graph, k_nodes, v_nodes, seq_len, num_layers, num_kv_heads, head_dim);

        size_t expected_len = min(token + 1, window_size);
        assert(cache.get_effective_seq_len() == expected_len);
    }

    assert(cache.get_effective_seq_len() == window_size);
    assert(cache.get_total_seq_len() == 12);

    float* key_data = static_cast<float*>(cache.get_key_ptr(0));
    if (key_data) {
        for (size_t i = 0; i < sink_size; i++) {
            float expected = float(i);
            float actual = key_data[i * num_kv_heads * head_dim];
            assert(actual == expected);
        }
    }

    return true;
}

bool test_reset_functionality() {
    KVCache cache;
    cache.init(2, 1024, 8, 64, Precision::FP32);
    cache.set_window_size(16, 4);

    CactusGraph graph;
    vector<size_t> k_nodes, v_nodes;

    for (size_t layer = 0; layer < 2; layer++) {
        size_t k_node = graph.input({10, 8, 64}, Precision::FP32);
        size_t v_node = graph.input({10, 8, 64}, Precision::FP32);

        vector<float> data(10 * 8 * 64, 1.0f);
        graph.set_input(k_node, data.data(), Precision::FP32);
        graph.set_input(v_node, data.data(), Precision::FP32);

        k_nodes.push_back(k_node);
        v_nodes.push_back(v_node);
    }

    graph.execute();
    cache.update_from_graph(&graph, k_nodes, v_nodes, 10, 2, 8, 64);

    assert(cache.get_effective_seq_len() == 10);
    assert(cache.get_total_seq_len() == 10);

    cache.reset();

    assert(cache.get_effective_seq_len() == 0);
    assert(cache.get_total_seq_len() == 0);
    assert(cache.get_key_ptr(0) == nullptr);
    assert(cache.get_value_ptr(0) == nullptr);

    return true;
}

bool test_large_window() {
    const size_t num_layers = 4;
    const size_t num_kv_heads = 8;
    const size_t head_dim = 64;
    const size_t window_size = 512; 

    KVCache cache;
    cache.init(num_layers, 2048, num_kv_heads, head_dim, Precision::FP32);
    cache.set_window_size(window_size, 4);

    CactusGraph graph;

    size_t seq_len = 600;
    vector<size_t> k_nodes, v_nodes;

    for (size_t layer = 0; layer < num_layers; layer++) {
        size_t k_node = graph.input({seq_len, num_kv_heads, head_dim}, Precision::FP32);
        size_t v_node = graph.input({seq_len, num_kv_heads, head_dim}, Precision::FP32);

        vector<float> k_data(seq_len * num_kv_heads * head_dim, float(layer));
        vector<float> v_data(seq_len * num_kv_heads * head_dim, float(layer + 100));

        graph.set_input(k_node, k_data.data(), Precision::FP32);
        graph.set_input(v_node, v_data.data(), Precision::FP32);

        k_nodes.push_back(k_node);
        v_nodes.push_back(v_node);
    }

    graph.execute();
    cache.update_from_graph(&graph, k_nodes, v_nodes, seq_len, num_layers, num_kv_heads, head_dim);

    assert(cache.get_effective_seq_len() == window_size);
    assert(cache.get_total_seq_len() == seq_len);

    return true;
}

int main() {
    TestUtils::TestRunner runner("KV Cache Sliding Window Tests");
    runner.run_test("Basic Sliding Window", test_sliding_window_basic());
    runner.run_test("Incremental Updates", test_incremental_updates());
    runner.run_test("Reset Functionality", test_reset_functionality());
    runner.run_test("Large Window (512 tokens)", test_large_window());

    cout << "────────────────────────────────────────────────────────────────────────────────────────\n";
    runner.print_summary();

    return runner.all_passed() ? 0 : 1;
}