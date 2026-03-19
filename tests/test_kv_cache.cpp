#include "../cactus/cactus.h"
#include "test_utils.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstring>
#include <cassert>
#include <cmath>

using namespace cactus::engine;
using namespace std;

void fill_fp16(vector<uint8_t>& buffer, size_t count, float value) {
    buffer.resize(count * sizeof(__fp16));
    __fp16* ptr = reinterpret_cast<__fp16*>(buffer.data());
    for (size_t i = 0; i < count; i++) {
        ptr[i] = static_cast<__fp16>(value);
    }
}

bool test_sliding_window_basic() {
    const size_t num_layers = 2;
    const size_t max_seq = 2048;
    const size_t num_kv_heads = 8;
    const size_t head_dim = 64;
    const size_t window_size = 16;
    const size_t sink_size = 4;

    KVCache cache;
    cache.init(num_layers, max_seq, num_kv_heads, std::vector<size_t>(num_layers, head_dim), Precision::INT8);
    cache.set_window_size(window_size, sink_size);

    CactusGraph graph;
    graph.input({1}, Precision::FP32);

    {
        size_t seq_len = 10;
        vector<size_t> k_nodes, v_nodes;

        for (size_t layer = 0; layer < num_layers; layer++) {
            size_t k_node = graph.input({seq_len, num_kv_heads, head_dim}, Precision::FP16);
            size_t v_node = graph.input({seq_len, num_kv_heads, head_dim}, Precision::FP16);

            vector<uint8_t> k_data, v_data;
            fill_fp16(k_data, seq_len * num_kv_heads * head_dim, layer + 1.0f);
            fill_fp16(v_data, seq_len * num_kv_heads * head_dim, layer + 2.0f);
            graph.set_input(k_node, k_data.data(), Precision::FP16);
            graph.set_input(v_node, v_data.data(), Precision::FP16);

            k_nodes.push_back(k_node);
            v_nodes.push_back(v_node);
        }

        graph.execute();
        cache.update_from_graph(&graph, k_nodes, v_nodes, seq_len, num_layers, num_kv_heads);

        assert(cache.get_effective_seq_len() == seq_len);
    }

    {
        size_t additional_seq = 6;
        vector<size_t> k_nodes, v_nodes;

        for (size_t layer = 0; layer < num_layers; layer++) {
            size_t k_node = graph.input({additional_seq, num_kv_heads, head_dim}, Precision::FP16);
            size_t v_node = graph.input({additional_seq, num_kv_heads, head_dim}, Precision::FP16);

            vector<uint8_t> k_data, v_data;
            fill_fp16(k_data, additional_seq * num_kv_heads * head_dim, layer + 10.0f);
            fill_fp16(v_data, additional_seq * num_kv_heads * head_dim, layer + 20.0f);
            graph.set_input(k_node, k_data.data(), Precision::FP16);
            graph.set_input(v_node, v_data.data(), Precision::FP16);

            k_nodes.push_back(k_node);
            v_nodes.push_back(v_node);
        }

        graph.execute();
        cache.update_from_graph(&graph, k_nodes, v_nodes, additional_seq, num_layers, num_kv_heads);

        assert(cache.get_effective_seq_len() == window_size);
    }

    {
        size_t additional_seq = 10;
        vector<size_t> k_nodes, v_nodes;

        float new_token_value = 100.0f;

        for (size_t layer = 0; layer < num_layers; layer++) {
            size_t k_node = graph.input({additional_seq, num_kv_heads, head_dim}, Precision::FP16);
            size_t v_node = graph.input({additional_seq, num_kv_heads, head_dim}, Precision::FP16);

            vector<uint8_t> k_data, v_data;
            fill_fp16(k_data, additional_seq * num_kv_heads * head_dim, new_token_value);
            fill_fp16(v_data, additional_seq * num_kv_heads * head_dim, new_token_value);
            graph.set_input(k_node, k_data.data(), Precision::FP16);
            graph.set_input(v_node, v_data.data(), Precision::FP16);

            k_nodes.push_back(k_node);
            v_nodes.push_back(v_node);
        }

        graph.execute();
        cache.update_from_graph(&graph, k_nodes, v_nodes, additional_seq, num_layers, num_kv_heads);

        assert(cache.get_effective_seq_len() == window_size);

        const int8_t* key_data = cache.get_keys_int8(0);
        const float* key_scales = cache.get_key_scales(0);
        if (key_data && key_scales) {
            bool sink_preserved = true;

            float expected_sink_value = 1.0f;
            float sink_scale = key_scales[0];
            int8_t expected_sink_q = static_cast<int8_t>(std::round(expected_sink_value / sink_scale));

            for (size_t i = 0; i < min((size_t)8, sink_size * num_kv_heads * head_dim); i++) {
                if (abs(key_data[i] - expected_sink_q) > 2) {
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
    cache.init(num_layers, 2048, num_kv_heads, std::vector<size_t>(num_layers, head_dim), Precision::FP16);
    cache.set_window_size(window_size, sink_size);

    CactusGraph graph;
    graph.input({1}, Precision::FP32);

    for (size_t token = 0; token < 12; token++) {
        vector<size_t> k_nodes, v_nodes;

        size_t seq_len = 1;
        size_t k_node = graph.input({seq_len, num_kv_heads, head_dim}, Precision::FP16);
        size_t v_node = graph.input({seq_len, num_kv_heads, head_dim}, Precision::FP16);

        vector<uint8_t> k_data, v_data;
        fill_fp16(k_data, seq_len * num_kv_heads * head_dim, float(token + 1)); 
        fill_fp16(v_data, seq_len * num_kv_heads * head_dim, float(token + 101));

        graph.set_input(k_node, k_data.data(), Precision::FP16);
        graph.set_input(v_node, v_data.data(), Precision::FP16);

        k_nodes.push_back(k_node);
        v_nodes.push_back(v_node);

        graph.execute();
        cache.update_from_graph(&graph, k_nodes, v_nodes, seq_len, num_layers, num_kv_heads);

        size_t expected_len = min(token + 1, window_size);
        assert(cache.get_effective_seq_len() == expected_len);
    }

    assert(cache.get_effective_seq_len() == window_size);
    assert(cache.get_total_seq_len() == 12);

    const int8_t* key_data = cache.get_keys_int8(0);
    const float* key_scales = cache.get_key_scales(0);
    assert(key_data != nullptr);
    assert(key_scales != nullptr);

    return true;
}

bool test_reset_functionality() {
    KVCache cache;
    cache.init(2, 1024, 8, std::vector<size_t>(2, 64), Precision::FP16);
    cache.set_window_size(16, 4);

    CactusGraph graph;
    graph.input({1}, Precision::FP32);
    vector<size_t> k_nodes, v_nodes;

    for (size_t layer = 0; layer < 2; layer++) {
        size_t k_node = graph.input({10, 8, 64}, Precision::FP16);
        size_t v_node = graph.input({10, 8, 64}, Precision::FP16);

        vector<uint8_t> data;
        fill_fp16(data, 10 * 8 * 64, 1.0f);
        graph.set_input(k_node, data.data(), Precision::FP16);
        graph.set_input(v_node, data.data(), Precision::FP16);

        k_nodes.push_back(k_node);
        v_nodes.push_back(v_node);
    }

    graph.execute();
    cache.update_from_graph(&graph, k_nodes, v_nodes, 10, 2, 8);

    assert(cache.get_effective_seq_len() == 10);
    assert(cache.get_total_seq_len() == 10);

    cache.reset();

    assert(cache.get_effective_seq_len() == 0);
    assert(cache.get_total_seq_len() == 0);
    assert(cache.get_keys_int8(0) == nullptr);
    assert(cache.get_values_int8(0) == nullptr);

    return true;
}

bool test_large_window() {
    const size_t num_layers = 4;
    const size_t num_kv_heads = 8;
    const size_t head_dim = 64;
    const size_t window_size = 512;

    KVCache cache;
    cache.init(num_layers, 2048, num_kv_heads, std::vector<size_t>(num_layers, head_dim), Precision::FP16);
    cache.set_window_size(window_size, 4);

    CactusGraph graph;
    graph.input({1}, Precision::FP32);

    size_t seq_len = 600;
    vector<size_t> k_nodes, v_nodes;

    for (size_t layer = 0; layer < num_layers; layer++) {
        size_t k_node = graph.input({seq_len, num_kv_heads, head_dim}, Precision::FP16);
        size_t v_node = graph.input({seq_len, num_kv_heads, head_dim}, Precision::FP16);

        vector<uint8_t> k_data, v_data;
        fill_fp16(k_data, seq_len * num_kv_heads * head_dim, float(layer + 1)); 
        fill_fp16(v_data, seq_len * num_kv_heads * head_dim, float(layer + 101));

        graph.set_input(k_node, k_data.data(), Precision::FP16);
        graph.set_input(v_node, v_data.data(), Precision::FP16);

        k_nodes.push_back(k_node);
        v_nodes.push_back(v_node);
    }

    graph.execute();
    cache.update_from_graph(&graph, k_nodes, v_nodes, seq_len, num_layers, num_kv_heads);

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