#include "test_utils.h"

#include <cstdint>
#include <vector>

using namespace TestUtils;

static uint64_t current_seq(CactusGraph& graph, size_t cache_node) {
    auto* raw = static_cast<uint8_t*>(graph.get_output(cache_node));
    return *reinterpret_cast<uint64_t*>(raw);
}

static bool append_tokens(CactusGraph& graph, size_t cache_node, size_t tokens, size_t kv_heads, size_t head_dim, float value) {
    const size_t elements = tokens * kv_heads * head_dim;
    size_t input = graph.input({elements}, Precision::FP16);
    std::vector<__fp16> data(elements, static_cast<__fp16>(value));
    graph.set_input(input, data.data(), Precision::FP16);
    graph.kv_cache_append(input, cache_node);
    graph.execute();
    return current_seq(graph, cache_node) > 0;
}

bool test_cache_transaction_rollback_restores_sequence_length() {
    CactusGraph graph;
    const size_t kv_heads = 2;
    const size_t head_dim = 16;
    size_t cache = graph.kv_cache_state(64, kv_heads, head_dim);
    graph.execute();

    if (!append_tokens(graph, cache, 4, kv_heads, head_dim, 1.0f)) return false;
    if (current_seq(graph, cache) != 4) return false;

    auto txn = graph.begin_kv_cache_transaction({cache});

    graph.soft_reset();
    if (!append_tokens(graph, cache, 3, kv_heads, head_dim, 2.0f)) return false;
    if (current_seq(graph, cache) != 7) return false;

    txn.rollback();
    graph.execute();

    return current_seq(graph, cache) == 4;
}

bool test_cache_transaction_commit_all_keeps_draft_tokens() {
    CactusGraph graph;
    const size_t kv_heads = 2;
    const size_t head_dim = 16;
    size_t cache = graph.kv_cache_state(64, kv_heads, head_dim);
    graph.execute();

    if (!append_tokens(graph, cache, 4, kv_heads, head_dim, 1.0f)) return false;

    auto txn = graph.begin_kv_cache_transaction({cache});

    graph.soft_reset();
    if (!append_tokens(graph, cache, 3, kv_heads, head_dim, 2.0f)) return false;

    txn.commit_all();
    graph.execute();

    return current_seq(graph, cache) == 7;
}

bool test_cache_transaction_commit_prefix_crops_rejected_suffix() {
    CactusGraph graph;
    const size_t kv_heads = 2;
    const size_t head_dim = 16;
    size_t cache = graph.kv_cache_state(64, kv_heads, head_dim);
    graph.execute();

    if (!append_tokens(graph, cache, 4, kv_heads, head_dim, 1.0f)) return false;

    auto txn = graph.begin_kv_cache_transaction({cache});

    graph.soft_reset();
    if (!append_tokens(graph, cache, 6, kv_heads, head_dim, 2.0f)) return false;

    txn.commit_prefix(2);
    graph.execute();

    return current_seq(graph, cache) == 6;
}

bool test_cache_transaction_tracks_multiple_layer_caches() {
    CactusGraph graph;
    const size_t kv_heads = 2;
    const size_t head_dim = 16;
    size_t k_cache = graph.kv_cache_state(64, kv_heads, head_dim);
    size_t v_cache = graph.kv_cache_state(64, kv_heads, head_dim);
    graph.execute();

    if (!append_tokens(graph, k_cache, 5, kv_heads, head_dim, 1.0f)) return false;
    graph.soft_reset();
    if (!append_tokens(graph, v_cache, 5, kv_heads, head_dim, 1.0f)) return false;

    auto txn = graph.begin_kv_cache_transaction({k_cache, v_cache});

    graph.soft_reset();
    if (!append_tokens(graph, k_cache, 4, kv_heads, head_dim, 2.0f)) return false;
    graph.soft_reset();
    if (!append_tokens(graph, v_cache, 4, kv_heads, head_dim, 2.0f)) return false;

    txn.rollback();
    graph.execute();

    return current_seq(graph, k_cache) == 5
        && current_seq(graph, v_cache) == 5;
}

bool test_cache_transaction_sliding_window_commit_prefix() {
    CactusGraph graph;
    const size_t kv_heads = 1;
    const size_t head_dim = 16;
    const size_t window = 8;
    const size_t sink = 2;
    size_t cache = graph.kv_cache_state(window, kv_heads, head_dim, window, sink);
    graph.execute();

    if (!append_tokens(graph, cache, 6, kv_heads, head_dim, 1.0f)) return false;

    auto txn = graph.begin_kv_cache_transaction({cache});

    graph.soft_reset();
    size_t input = graph.input({4 * kv_heads * head_dim}, Precision::FP16);
    std::vector<__fp16> data(4 * kv_heads * head_dim, static_cast<__fp16>(2.0f));
    graph.set_input(input, data.data(), Precision::FP16);
    graph.kv_cache_append(input, cache, window, sink);
    graph.execute();

    txn.commit_prefix(1);
    graph.execute();

    return current_seq(graph, cache) == 7;
}

int main() {
    TestRunner runner("KV Cache Transaction Tests");

    runner.run_test("rollback_restores_length", test_cache_transaction_rollback_restores_sequence_length());
    runner.run_test("commit_all_keeps_tokens", test_cache_transaction_commit_all_keeps_draft_tokens());
    runner.run_test("commit_prefix_crops_suffix", test_cache_transaction_commit_prefix_crops_rejected_suffix());
    runner.run_test("multiple_layer_caches", test_cache_transaction_tracks_multiple_layer_caches());
    runner.run_test("sliding_window_prefix", test_cache_transaction_sliding_window_commit_prefix());

    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
