#include "test_utils.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <cstring>

using namespace TestUtils;

bool test_kv_cache_state_init() {
    CactusGraph g;

    const size_t max_seq = 64, kv_heads = 4, head_dim = 16;
    size_t cache_node = g.kv_cache_state(max_seq, kv_heads, head_dim);
    g.execute();

    auto* raw = static_cast<uint8_t*>(g.get_output(cache_node));
    if (!raw) return false;

    uint64_t current_seq = *reinterpret_cast<uint64_t*>(raw + 0);
    uint64_t stored_max  = *reinterpret_cast<uint64_t*>(raw + 8);
    uint64_t stored_kv   = *reinterpret_cast<uint64_t*>(raw + 16);
    uint64_t stored_hdim = *reinterpret_cast<uint64_t*>(raw + 24);

    if (current_seq != 0) return false;
    if (stored_max != max_seq) return false;
    if (stored_kv != kv_heads) return false;
    if (stored_hdim != head_dim) return false;

    return true;
}

bool test_kv_cache_state_persistent() {
    CactusGraph g;

    size_t cache_node = g.kv_cache_state(32, 2, 16);
    g.execute();

    auto* raw = static_cast<uint8_t*>(g.get_output(cache_node));
    if (!raw) return false;

    g.soft_reset();

    if (!g.is_populated(cache_node)) return false;

    return true;
}

bool test_kv_cache_append_basic() {
    CactusGraph g;

    const size_t max_seq = 64, kv_heads = 2, head_dim = 16;
    const size_t new_tokens = 3;
    const size_t kv_elements = new_tokens * kv_heads * head_dim;

    size_t cache_node = g.kv_cache_state(max_seq, kv_heads, head_dim);

    size_t kv_input = g.input({kv_elements}, Precision::FP16);
    std::vector<__fp16> kv_data(kv_elements);
    for (size_t i = 0; i < kv_elements; i++) {
        kv_data[i] = static_cast<__fp16>(static_cast<float>(i) * 0.1f);
    }
    g.set_input(kv_input, kv_data.data(), Precision::FP16);

    size_t append_result = g.kv_cache_append(kv_input, cache_node);
    g.execute();

    float* result = static_cast<float*>(g.get_output(append_result));
    if (!result) return false;
    if (static_cast<size_t>(*result) != new_tokens) return false;

    auto* raw = static_cast<uint8_t*>(g.get_output(cache_node));
    uint64_t current_seq = *reinterpret_cast<uint64_t*>(raw + 0);
    if (current_seq != new_tokens) return false;

    return true;
}

bool test_kv_cache_append_multiple() {
    CactusGraph g;

    const size_t max_seq = 64, kv_heads = 2, head_dim = 16;
    size_t cache_node = g.kv_cache_state(max_seq, kv_heads, head_dim);

    {
        const size_t tokens = 2;
        const size_t elements = tokens * kv_heads * head_dim;
        size_t kv_input = g.input({elements}, Precision::FP16);
        std::vector<__fp16> data(elements, static_cast<__fp16>(1.0f));
        g.set_input(kv_input, data.data(), Precision::FP16);
        g.kv_cache_append(kv_input, cache_node);
        g.execute();
    }

    auto* raw = static_cast<uint8_t*>(g.get_output(cache_node));
    if (*reinterpret_cast<uint64_t*>(raw) != 2) return false;

    g.soft_reset();
    {
        const size_t tokens = 3;
        const size_t elements = tokens * kv_heads * head_dim;
        size_t kv_input = g.input({elements}, Precision::FP16);
        std::vector<__fp16> data(elements, static_cast<__fp16>(2.0f));
        g.set_input(kv_input, data.data(), Precision::FP16);
        g.kv_cache_append(kv_input, cache_node);
        g.execute();
    }

    raw = static_cast<uint8_t*>(g.get_output(cache_node));
    uint64_t total_seq = *reinterpret_cast<uint64_t*>(raw);
    if (total_seq != 5) return false;

    return true;
}

bool test_kv_cache_append_eviction() {
    CactusGraph g;

    const size_t kv_heads = 1, head_dim = 16;
    const size_t window = 8, sink = 2;

    size_t cache_node = g.kv_cache_state(window, kv_heads, head_dim, window, sink);

    {
        const size_t tokens = 8;
        const size_t elements = tokens * kv_heads * head_dim;
        size_t kv_input = g.input({elements}, Precision::FP16);
        std::vector<__fp16> data(elements);
        for (size_t t = 0; t < tokens; t++) {
            for (size_t j = 0; j < kv_heads * head_dim; j++) {
                data[t * kv_heads * head_dim + j] = static_cast<__fp16>(static_cast<float>(t));
            }
        }
        g.set_input(kv_input, data.data(), Precision::FP16);
        g.kv_cache_append(kv_input, cache_node, window, sink);
        g.execute();
    }

    auto* raw = static_cast<uint8_t*>(g.get_output(cache_node));
    uint64_t seq = *reinterpret_cast<uint64_t*>(raw);
    if (seq != 8) return false;

    g.soft_reset();
    {
        const size_t tokens = 2;
        const size_t elements = tokens * kv_heads * head_dim;
        size_t kv_input = g.input({elements}, Precision::FP16);
        std::vector<__fp16> data(elements);
        for (size_t t = 0; t < tokens; t++) {
            for (size_t j = 0; j < kv_heads * head_dim; j++) {
                data[t * kv_heads * head_dim + j] = static_cast<__fp16>(100.0f + static_cast<float>(t));
            }
        }
        g.set_input(kv_input, data.data(), Precision::FP16);
        g.kv_cache_append(kv_input, cache_node, window, sink);
        g.execute();
    }

    raw = static_cast<uint8_t*>(g.get_output(cache_node));
    seq = *reinterpret_cast<uint64_t*>(raw);
    if (seq != window) return false;

    return true;
}

bool test_attention_cached_basic() {
    const size_t b = 1, s = 1, h = 2, kv = 2, d = 16;
    const size_t max_seq = 64;

    CactusGraph g;

    size_t k_cache = g.kv_cache_state(max_seq, kv, d);
    size_t v_cache = g.kv_cache_state(max_seq, kv, d);

    size_t iq = g.input({b, s, h, d}, Precision::FP16);
    size_t ik = g.input({b, s, kv, d}, Precision::FP16);
    size_t iv = g.input({b, s, kv, d}, Precision::FP16);

    std::vector<__fp16> q(b * s * h * d), k_new(b * s * kv * d), v_new(b * s * kv * d);
    fill_random_fp16(q);
    fill_random_fp16(k_new);
    fill_random_fp16(v_new);

    g.set_input(iq, q.data(), Precision::FP16);
    g.set_input(ik, k_new.data(), Precision::FP16);
    g.set_input(iv, v_new.data(), Precision::FP16);

    g.kv_cache_append(ik, k_cache);
    g.kv_cache_append(iv, v_cache);

    float scale = 1.0f / std::sqrt(static_cast<float>(d));
    size_t attn = g.attention_cached(iq, ik, iv, k_cache, v_cache, scale, 0);

    g.execute();

    __fp16* result = static_cast<__fp16*>(g.get_output(attn));
    size_t out_size = b * s * h * d;

    bool has_nonzero = false;
    for (size_t i = 0; i < out_size; i++) {
        if (!std::isfinite(static_cast<float>(result[i]))) return false;
        if (std::abs(static_cast<float>(result[i])) > 1e-6f) has_nonzero = true;
    }
    return has_nonzero;
}

bool test_attention_cached_multistep() {
    const size_t b = 1, h = 2, kv = 2, d = 16;
    const size_t max_seq = 64;

    CactusGraph g;

    size_t k_cache = g.kv_cache_state(max_seq, kv, d);
    size_t v_cache = g.kv_cache_state(max_seq, kv, d);

    {
        const size_t s = 4;
        size_t iq = g.input({b, s, h, d}, Precision::FP16);
        size_t ik = g.input({b, s, kv, d}, Precision::FP16);
        size_t iv = g.input({b, s, kv, d}, Precision::FP16);

        std::vector<__fp16> q(b*s*h*d), k(b*s*kv*d), v(b*s*kv*d);
        fill_random_fp16(q);
        fill_random_fp16(k);
        fill_random_fp16(v);

        g.set_input(iq, q.data(), Precision::FP16);
        g.set_input(ik, k.data(), Precision::FP16);
        g.set_input(iv, v.data(), Precision::FP16);

        g.kv_cache_append(ik, k_cache);
        g.kv_cache_append(iv, v_cache);
        g.attention_cached(iq, ik, iv, k_cache, v_cache,
                           1.0f / std::sqrt(static_cast<float>(d)), 0);
        g.execute();
    }

    auto* raw = static_cast<uint8_t*>(g.get_output(k_cache));
    if (*reinterpret_cast<uint64_t*>(raw) != 4) return false;

    g.soft_reset();
    {
        const size_t s = 1;
        size_t iq = g.input({b, s, h, d}, Precision::FP16);
        size_t ik = g.input({b, s, kv, d}, Precision::FP16);
        size_t iv = g.input({b, s, kv, d}, Precision::FP16);

        std::vector<__fp16> q(b*s*h*d), k(b*s*kv*d), v(b*s*kv*d);
        fill_random_fp16(q);
        fill_random_fp16(k);
        fill_random_fp16(v);

        g.set_input(iq, q.data(), Precision::FP16);
        g.set_input(ik, k.data(), Precision::FP16);
        g.set_input(iv, v.data(), Precision::FP16);

        g.kv_cache_append(ik, k_cache);
        g.kv_cache_append(iv, v_cache);
        size_t attn = g.attention_cached(iq, ik, iv, k_cache, v_cache,
                                          1.0f / std::sqrt(static_cast<float>(d)), 4);
        g.execute();

        __fp16* result = static_cast<__fp16*>(g.get_output(attn));
        for (size_t i = 0; i < b*s*h*d; i++) {
            if (!std::isfinite(static_cast<float>(result[i]))) return false;
        }
    }

    raw = static_cast<uint8_t*>(g.get_output(k_cache));
    if (*reinterpret_cast<uint64_t*>(raw) != 5) return false;

    return true;
}

bool test_kv_cache_invalidate() {
    CactusGraph g;

    const size_t max_seq = 32, kv_heads = 2, head_dim = 16;
    size_t cache_node = g.kv_cache_state(max_seq, kv_heads, head_dim);

    {
        const size_t tokens = 4;
        const size_t elements = tokens * kv_heads * head_dim;
        size_t kv_input = g.input({elements}, Precision::FP16);
        std::vector<__fp16> data(elements, static_cast<__fp16>(1.0f));
        g.set_input(kv_input, data.data(), Precision::FP16);
        g.kv_cache_append(kv_input, cache_node);
        g.execute();
    }

    if (!g.is_populated(cache_node)) return false;

    g.invalidate_persistent(cache_node);
    if (g.is_populated(cache_node)) return false;

    g.soft_reset();
    size_t new_cache = g.kv_cache_state(max_seq, kv_heads, head_dim);
    g.execute();

    auto* raw = static_cast<uint8_t*>(g.get_output(new_cache));
    uint64_t seq = *reinterpret_cast<uint64_t*>(raw);
    if (seq != 0) return false;

    return true;
}

bool test_conv_cache_state_init() {
    CactusGraph g;

    const size_t ws = 8, hd = 32;
    size_t cache = g.conv_cache_state(ws, hd);
    g.execute();

    auto* raw = static_cast<uint8_t*>(g.get_output(cache));
    if (!raw) return false;

    uint64_t head = *reinterpret_cast<uint64_t*>(raw + 0);
    uint64_t count = *reinterpret_cast<uint64_t*>(raw + 8);
    uint64_t stored_ws = *reinterpret_cast<uint64_t*>(raw + 16);
    uint64_t stored_hd = *reinterpret_cast<uint64_t*>(raw + 24);

    if (head != 0 || count != 0) return false;
    if (stored_ws != ws || stored_hd != hd) return false;

    return true;
}

bool test_conv_cache_append_basic() {
    CactusGraph g;

    const size_t ws = 4, hd = 8;
    size_t cache = g.conv_cache_state(ws, hd);

    size_t inp = g.input({2, hd}, Precision::FP16);
    std::vector<__fp16> data(2 * hd);
    for (size_t i = 0; i < 2 * hd; i++) data[i] = static_cast<__fp16>(static_cast<float>(i));
    g.set_input(inp, data.data(), Precision::FP16);

    size_t window_out = g.conv_cache_append(inp, cache);
    g.execute();

    const auto& buf = g.get_output_buffer(window_out);
    if (buf.shape[0] != ws || buf.shape[1] != hd) return false;

    __fp16* out = static_cast<__fp16*>(g.get_output(window_out));

    for (size_t i = 0; i < 2 * hd; i++) {
        if (std::abs(static_cast<float>(out[i]) - static_cast<float>(data[i])) > 1e-3f) return false;
    }

    return true;
}

bool test_conv_cache_append_circular() {
    CactusGraph g;

    const size_t ws = 3, hd = 4;
    size_t cache = g.conv_cache_state(ws, hd);

    for (int step = 0; step < 5; step++) {
        if (step > 0) g.soft_reset();
        size_t inp = g.input({1, hd}, Precision::FP16);
        std::vector<__fp16> data(hd);
        for (size_t j = 0; j < hd; j++) data[j] = static_cast<__fp16>(static_cast<float>(step * 10 + j));
        g.set_input(inp, data.data(), Precision::FP16);
        g.conv_cache_append(inp, cache);
        g.execute();
    }

    g.soft_reset();
    size_t inp = g.input({1, hd}, Precision::FP16);
    std::vector<__fp16> data(hd, static_cast<__fp16>(99.0f));
    g.set_input(inp, data.data(), Precision::FP16);
    size_t window_out = g.conv_cache_append(inp, cache);
    g.execute();

    __fp16* out = static_cast<__fp16*>(g.get_output(window_out));
    const auto& buf = g.get_output_buffer(window_out);
    if (buf.shape[0] != ws) return false;

    for (size_t j = 0; j < hd; j++) {
        if (std::abs(static_cast<float>(out[(ws - 1) * hd + j]) - 99.0f) > 1e-2f) return false;
    }

    return true;
}

bool test_conv_cache_persistent() {
    CactusGraph g;

    size_t cache = g.conv_cache_state(4, 8);
    g.execute();

    g.soft_reset();
    if (!g.is_populated(cache)) return false;

    g.invalidate_persistent(cache);
    if (g.is_populated(cache)) return false;

    return true;
}

bool run_benchmarks() {
    auto bench = [](const char* label, auto setup, auto run) {
        setup();
        run();
        TestUtils::Timer t;
        for (int i = 0; i < 100; i++) run();
        double ms = t.elapsed_ms() / 100.0;
        std::cout << "  ⚡ " << std::left << std::setw(30) << label
                  << std::fixed << std::setprecision(3) << ms << " ms\n";
    };

    {
        const size_t kv = 8, d = 128, max_seq = 1024;

        CactusGraph g;
        size_t k_cache = g.kv_cache_state(max_seq, kv, d);

        size_t prefill_elements = 512 * kv * d;
        size_t prefill_input = g.input({prefill_elements}, Precision::FP16);
        std::vector<__fp16> prefill_data(prefill_elements);
        fill_random_fp16(prefill_data);
        g.set_input(prefill_input, prefill_data.data(), Precision::FP16);
        g.kv_cache_append(prefill_input, k_cache);
        g.execute();

        size_t append_elements = 1 * kv * d;
        std::vector<__fp16> append_data(append_elements);
        fill_random_fp16(append_data);

        bench("kv_cache_append 1tok@512", []{}, [&]{
            g.soft_reset_keep_pool();
            size_t inp = g.input({append_elements}, Precision::FP16);
            g.set_input(inp, append_data.data(), Precision::FP16);
            g.kv_cache_append(inp, k_cache);
            g.execute();
        });
    }

    {
        const size_t b = 1, s = 1, h = 16, kv = 8, d = 128, max_seq = 1024;
        float scale = 1.0f / std::sqrt(static_cast<float>(d));

        CactusGraph g;
        size_t k_cache = g.kv_cache_state(max_seq, kv, d);
        size_t v_cache = g.kv_cache_state(max_seq, kv, d);

        size_t prefill_elements = 512 * kv * d;
        std::vector<__fp16> prefill_data(prefill_elements);
        fill_random_fp16(prefill_data);

        size_t pk = g.input({prefill_elements}, Precision::FP16);
        size_t pv = g.input({prefill_elements}, Precision::FP16);
        g.set_input(pk, prefill_data.data(), Precision::FP16);
        g.set_input(pv, prefill_data.data(), Precision::FP16);
        g.kv_cache_append(pk, k_cache);
        g.kv_cache_append(pv, v_cache);
        g.execute();

        std::vector<__fp16> q(b*s*h*d), k_new(b*s*kv*d), v_new(b*s*kv*d);
        fill_random_fp16(q);
        fill_random_fp16(k_new);
        fill_random_fp16(v_new);

        bench("attention_cached 1tok@512", []{}, [&]{
            g.soft_reset_keep_pool();
            size_t iq = g.input({b, s, h, d}, Precision::FP16);
            size_t ik = g.input({b, s, kv, d}, Precision::FP16);
            size_t iv = g.input({b, s, kv, d}, Precision::FP16);
            g.set_input(iq, q.data(), Precision::FP16);
            g.set_input(ik, k_new.data(), Precision::FP16);
            g.set_input(iv, v_new.data(), Precision::FP16);
            g.kv_cache_append(ik, k_cache);
            g.kv_cache_append(iv, v_cache);
            g.attention_cached(iq, ik, iv, k_cache, v_cache, scale, 512);
            g.execute();
        });
    }

    return true;
}

int main() {
    TestUtils::TestRunner runner("Cache Tests");

    runner.run_test("KV Cache State Init", test_kv_cache_state_init());
    runner.run_test("KV Cache Persistent", test_kv_cache_state_persistent());
    runner.run_test("KV Cache Append Basic", test_kv_cache_append_basic());
    runner.run_test("KV Cache Append Multiple", test_kv_cache_append_multiple());
    runner.run_test("KV Cache Append Eviction", test_kv_cache_append_eviction());
    runner.run_test("Attention Cached Basic", test_attention_cached_basic());
    runner.run_test("Attention Cached Multistep", test_attention_cached_multistep());
    runner.run_test("KV Cache Invalidate", test_kv_cache_invalidate());
    runner.run_test("Conv Cache State Init", test_conv_cache_state_init());
    runner.run_test("Conv Cache Append Basic", test_conv_cache_append_basic());
    runner.run_test("Conv Cache Circular", test_conv_cache_append_circular());
    runner.run_test("Conv Cache Persistent", test_conv_cache_persistent());
    runner.print_benchmarks_header();
    runner.run_bench("benchmarks", run_benchmarks());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
