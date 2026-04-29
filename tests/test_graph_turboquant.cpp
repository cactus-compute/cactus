#include "test_utils.h"
#include "../cactus/graph/graph.h"
#include "../cactus/kernel/kernel.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <chrono>
#include <memory>
#include <functional>

using namespace cactus;

bool test_graph_performance_comparison(size_t cache_len) {
    size_t batch_size = 1;
    size_t seq_len = 1;
    size_t num_q_heads = 32;
    size_t num_kv_heads = 8;
    size_t head_dim = 128;
    size_t key_angle_bits = 2;
    size_t value_angle_bits = 4;
    float scale = 1.0f / std::sqrt(head_dim);
    int iterations = (cache_len > 100000) ? 2 : ((cache_len > 8192) ? 5 : 20);

    struct BenchData {
        std::vector<__fp16> q_data;
        std::vector<__fp16> k_data, v_data;
        std::vector<int8_t> k_cache_int8, v_cache_int8;
        std::vector<float> k_scales, v_scales;
        std::vector<uint8_t> rot_signs;
        std::vector<float> k_radii, v_radii;
        std::vector<uint8_t> k_angles, v_angles;
    };

    auto run_bench = [&](const std::string& name, std::function<void(CactusGraph&, BenchData&)> setup) {
        CactusGraph graph;
        BenchData data;
        setup(graph, data);

        graph.execute();

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            graph.execute();
        }
        auto end = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
        std::cout << "  - " << name << ": " << ms << " ms" << std::endl;
        return ms;
    };

    std::cout << "\nGraph Attention Nodes (cache_len=" << cache_len << "):" << std::endl;

    run_bench("FP16", [&](CactusGraph& g, BenchData& d) {
        size_t q = g.input({batch_size, seq_len, num_q_heads, head_dim}, Precision::FP16);
        size_t k = g.input({batch_size, cache_len, num_kv_heads, head_dim}, Precision::FP16);
        size_t v = g.input({batch_size, cache_len, num_kv_heads, head_dim}, Precision::FP16);
        g.attention(q, k, v, scale);

        d.q_data.assign(batch_size * seq_len * num_q_heads * head_dim, 0.1234f);
        d.k_data.assign(batch_size * cache_len * num_kv_heads * head_dim, 0.5678f);
        d.v_data.assign(batch_size * cache_len * num_kv_heads * head_dim, 0.9012f);
        g.set_input(q, d.q_data.data(), Precision::FP16);
        g.set_input(k, d.k_data.data(), Precision::FP16);
        g.set_input(v, d.v_data.data(), Precision::FP16);
    });

    run_bench("INT8 Hybrid", [&](CactusGraph& g, BenchData& d) {
        size_t q = g.input({batch_size, seq_len, num_q_heads, head_dim}, Precision::FP16);
        size_t k_new = g.input({batch_size, 0, num_kv_heads, head_dim}, Precision::FP16);
        size_t v_new = g.input({batch_size, 0, num_kv_heads, head_dim}, Precision::FP16);

        size_t k_scales_cnt = (cache_len * num_kv_heads * head_dim + 31) / 32;
        d.k_cache_int8.assign(cache_len * num_kv_heads * head_dim, 0);
        d.v_cache_int8.assign(cache_len * num_kv_heads * head_dim, 0);
        d.k_scales.assign(k_scales_cnt, 1.0f);
        d.v_scales.assign(k_scales_cnt, 1.0f);

        g.attention_int8_hybrid(q, k_new, v_new, scale, 0,
                               d.k_cache_int8.data(), d.v_cache_int8.data(), d.k_scales.data(), d.v_scales.data(),
                               cache_len, num_kv_heads, head_dim);

        d.q_data.assign(batch_size * seq_len * num_q_heads * head_dim, 0.1234f);
        g.set_input(q, d.q_data.data(), Precision::FP16);
    });

    run_bench("TurboQuant K2V4", [&](CactusGraph& g, BenchData& d) {
        size_t q = g.input({batch_size, seq_len, num_q_heads, head_dim}, Precision::FP16);
        size_t k_new = g.input({batch_size, 0, num_kv_heads, head_dim}, Precision::FP16);
        size_t v_new = g.input({batch_size, 0, num_kv_heads, head_dim}, Precision::FP16);

        d.rot_signs.resize(turboquant_rotation_signs_bytes(head_dim));
        cactus_turboquant_init(d.rot_signs.data(), head_dim, 42);

        size_t num_vectors = cache_len * num_kv_heads;
        d.k_radii.assign(num_vectors, 1.0f);
        d.v_radii.assign(num_vectors, 1.0f);
        d.k_angles.assign(num_vectors * turboquant_angles_bytes_per_head(head_dim, key_angle_bits), 0);
        d.v_angles.assign(num_vectors * turboquant_angles_bytes_per_head(head_dim, value_angle_bits), 0);

        g.attention_turboquant_hybrid(
            q, k_new, v_new, scale, 0,
            d.k_radii.data(), d.k_angles.data(),
            d.v_radii.data(), d.v_angles.data(),
            d.rot_signs.data(),
            key_angle_bits, value_angle_bits, cache_len, num_kv_heads, head_dim
        );

        d.q_data.assign(batch_size * seq_len * num_q_heads * head_dim, 0);
        g.set_input(q, d.q_data.data(), Precision::FP16);
    });

    return true;
}

int main() {
    std::cout << "TurboQuant Graph Benchmarks" << std::endl;
    test_graph_performance_comparison(2048);
    test_graph_performance_comparison(8192);
    test_graph_performance_comparison(32768);
    test_graph_performance_comparison(131072);
    return 0;
}
