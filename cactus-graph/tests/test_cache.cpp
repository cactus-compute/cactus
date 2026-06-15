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

bool test_kv_cache_append_full_window_eviction() {
    CactusGraph g;

    const size_t kv_heads = 1, head_dim = 16;
    const size_t window = 8, sink = 2;

    size_t cache_node = g.kv_cache_state(window, kv_heads, head_dim, window, sink);

    for (int pass = 0; pass < 2; pass++) {
        if (pass > 0) g.soft_reset();
        const size_t tokens = window;
        const size_t elements = tokens * kv_heads * head_dim;
        size_t kv_input = g.input({elements}, Precision::FP16);
        std::vector<__fp16> data(elements, static_cast<__fp16>(static_cast<float>(pass + 1)));
        g.set_input(kv_input, data.data(), Precision::FP16);
        g.kv_cache_append(kv_input, cache_node, window, sink);
        g.execute();

        auto* raw = static_cast<uint8_t*>(g.get_output(cache_node));
        uint64_t seq = *reinterpret_cast<uint64_t*>(raw);
        if (seq != window) return false;
    }

    return true;
}

static bool padded_rollback_matches_exact_append(size_t prefix, size_t real, size_t pads) {
    const size_t kv_heads = 1, head_dim = 16, max_seq = 64, window = 16, sink = 2;
    const size_t stride = kv_heads * head_dim;

    auto append = [&](CactusGraph& g, size_t cache_node, size_t real_tokens, size_t pad_tokens, float base) {
        const size_t tokens = real_tokens + pad_tokens;
        const size_t elements = tokens * stride;
        size_t kv_input = g.input({elements}, Precision::FP16);
        std::vector<__fp16> data(elements);
        for (size_t t = 0; t < tokens; t++) {
            float value = t < real_tokens ? base + static_cast<float>(t) : 9999.0f;
            for (size_t j = 0; j < stride; j++) data[t * stride + j] = static_cast<__fp16>(value);
        }
        g.set_input(kv_input, data.data(), Precision::FP16);
        g.kv_cache_append(kv_input, cache_node, window, sink);
        g.execute();
        g.soft_reset();
    };

    CactusGraph g_exact, g_padded;
    size_t exact_node = g_exact.kv_cache_state(max_seq, kv_heads, head_dim, window, sink);
    size_t padded_node = g_padded.kv_cache_state(max_seq, kv_heads, head_dim, window, sink);
    if (prefix > 0) {
        append(g_exact, exact_node, prefix, 0, 1.0f);
        append(g_padded, padded_node, prefix, 0, 1.0f);
    }
    append(g_exact, exact_node, real, 0, 100.0f);
    auto backup = g_padded.snapshot_cache_padded_append(padded_node, real, pads);
    append(g_padded, padded_node, real, pads, 100.0f);
    g_padded.rollback_cache_padded_append(padded_node, real, pads, backup);

    auto* exact_raw = static_cast<uint8_t*>(g_exact.get_output(exact_node));
    auto* padded_raw = static_cast<uint8_t*>(g_padded.get_output(padded_node));
    uint64_t exact_len = *reinterpret_cast<uint64_t*>(exact_raw);
    uint64_t padded_len = *reinterpret_cast<uint64_t*>(padded_raw);
    if (exact_len != padded_len) {
        std::cerr << "  current_seq_len mismatch: " << exact_len << " != " << padded_len << "\n";
        return false;
    }
    uint64_t buffer_max_seq = *reinterpret_cast<uint64_t*>(exact_raw + 8);
    const size_t meta_bytes = 64;
    if (std::memcmp(exact_raw + meta_bytes, padded_raw + meta_bytes, exact_len * stride) != 0) {
        std::cerr << "  cache data rows differ after rollback\n";
        return false;
    }
    const size_t num_groups = (head_dim + KV_QUANT_GROUP_SIZE - 1) / KV_QUANT_GROUP_SIZE;
    const size_t scales_offset = meta_bytes + buffer_max_seq * stride;
    if (std::memcmp(exact_raw + scales_offset, padded_raw + scales_offset,
                    exact_len * kv_heads * num_groups * sizeof(float)) != 0) {
        std::cerr << "  cache scale rows differ after rollback\n";
        return false;
    }
    return true;
}

bool test_padded_rollback_no_eviction() {
    return padded_rollback_matches_exact_append(4, 3, 5);
}

bool test_padded_rollback_eviction_overshoot() {
    return padded_rollback_matches_exact_append(16, 3, 6);
}

bool test_padded_rollback_only_pads_evict() {
    return padded_rollback_matches_exact_append(10, 2, 8);
}

bool test_padded_rollback_empty_cache() {
    return padded_rollback_matches_exact_append(0, 5, 11);
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

    const size_t pad_rows = ws - 2;
    for (size_t i = 0; i < pad_rows * hd; i++) {
        if (std::abs(static_cast<float>(out[i])) > 1e-3f) return false;
    }
    for (size_t i = 0; i < 2 * hd; i++) {
        if (std::abs(static_cast<float>(out[pad_rows * hd + i]) - static_cast<float>(data[i])) > 1e-3f) return false;
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

bool test_conv_cache_initialize_populates_trailing_rows() {
    CactusGraph g;

    constexpr size_t ws = 4;
    constexpr size_t hd = 8;
    constexpr size_t num_rows = 6;

    size_t cache = g.conv_cache_state(ws, hd);

    size_t rows_input = g.input({num_rows, hd}, Precision::FP16);
    std::vector<__fp16> data(num_rows * hd);
    for (size_t r = 0; r < num_rows; ++r) {
        for (size_t c = 0; c < hd; ++c) {
            data[r * hd + c] = static_cast<__fp16>(static_cast<float>(r + 1));
        }
    }
    g.set_input(rows_input, data.data(), Precision::FP16);

    g.conv_cache_initialize(rows_input, cache);
    g.execute();

    g.soft_reset_keep_pool();

    size_t append_input = g.input({1, hd}, Precision::FP16);
    std::vector<__fp16> append_data(hd, static_cast<__fp16>(99.0f));
    g.set_input(append_input, append_data.data(), Precision::FP16);

    size_t window = g.conv_cache_append(append_input, cache);
    g.execute();

    const __fp16* w = static_cast<const __fp16*>(g.get_output(window));
    if (!w) return false;
    const float expected[ws] = {4.0f, 5.0f, 6.0f, 99.0f};
    for (size_t r = 0; r < ws; ++r) {
        for (size_t c = 0; c < hd; ++c) {
            if (static_cast<float>(w[r * hd + c]) != expected[r]) return false;
        }
    }
    return true;
}

bool test_conv_cache_initialize_rejects_non_state_input() {
    CactusGraph g;
    size_t plain_input = g.input({4, 8}, Precision::FP16);
    std::vector<__fp16> zeros(32, static_cast<__fp16>(0.0f));
    g.set_input(plain_input, zeros.data(), Precision::FP16);
    size_t rows_input = g.input({2, 8}, Precision::FP16);
    g.set_input(rows_input, zeros.data(), Precision::FP16);
    try {
        g.conv_cache_initialize(rows_input, plain_input);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

bool test_conv_cache_initialize_output_is_zero_byte() {
    CactusGraph g;
    size_t cache = g.conv_cache_state(/*window=*/4, /*hidden_dim=*/8);
    size_t rows = g.input({2, 8}, Precision::FP16);
    std::vector<__fp16> data(16, static_cast<__fp16>(1.0f));
    g.set_input(rows, data.data(), Precision::FP16);
    size_t out_node = g.conv_cache_initialize(rows, cache);
    const auto& desc = g.get_output_buffer(out_node);
    return desc.byte_size == 0;
}

bool test_conv_cache_initialize_resets_dirty_cache() {
    CactusGraph g;
    constexpr size_t ws = 4;
    constexpr size_t hd = 8;
    size_t cache = g.conv_cache_state(ws, hd);

    size_t dirty_input = g.input({1, hd}, Precision::FP16);
    std::vector<__fp16> dirty(hd, static_cast<__fp16>(77.0f));
    g.set_input(dirty_input, dirty.data(), Precision::FP16);
    g.conv_cache_append(dirty_input, cache);
    g.execute();

    g.soft_reset_keep_pool();
    size_t rows_input = g.input({ws, hd}, Precision::FP16);
    std::vector<__fp16> data(ws * hd);
    for (size_t r = 0; r < ws; ++r) {
        for (size_t c = 0; c < hd; ++c) {
            data[r * hd + c] = static_cast<__fp16>(static_cast<float>(r + 1));
        }
    }
    g.set_input(rows_input, data.data(), Precision::FP16);
    g.conv_cache_initialize(rows_input, cache);
    g.execute();

    g.soft_reset_keep_pool();
    size_t append_input = g.input({1, hd}, Precision::FP16);
    std::vector<__fp16> append_data(hd, static_cast<__fp16>(99.0f));
    g.set_input(append_input, append_data.data(), Precision::FP16);
    size_t window = g.conv_cache_append(append_input, cache);
    g.execute();

    const __fp16* w = static_cast<const __fp16*>(g.get_output(window));
    if (!w) return false;
    const float expected[ws] = {2.0f, 3.0f, 4.0f, 99.0f};
    for (size_t r = 0; r < ws; ++r) {
        for (size_t c = 0; c < hd; ++c) {
            if (static_cast<float>(w[r * hd + c]) != expected[r]) return false;
        }
    }
    return true;
}

bool test_recurrent_cache_write_carries_state_across_executions() {
    CactusGraph g;

    const std::vector<size_t> shape{2, 4};
    const size_t elements = 2 * 4;

    size_t cache_state = g.recurrent_cache_state(shape, Precision::FP16);

    size_t new_value = g.input(shape, Precision::FP16);
    std::vector<__fp16> first(elements);
    for (size_t i = 0; i < elements; ++i) first[i] = static_cast<__fp16>(static_cast<float>(i + 1));
    g.set_input(new_value, first.data(), Precision::FP16);

    g.recurrent_cache_write(new_value, cache_state);
    g.execute();

    const __fp16* cache_after = static_cast<const __fp16*>(g.get_output(cache_state));
    if (!cache_after) return false;
    for (size_t i = 0; i < elements; ++i) {
        if (static_cast<float>(cache_after[i]) != static_cast<float>(first[i])) return false;
    }

    g.soft_reset_keep_pool();
    size_t second_input = g.input(shape, Precision::FP16);
    std::vector<__fp16> second(elements);
    for (size_t i = 0; i < elements; ++i) second[i] = static_cast<__fp16>(static_cast<float>(i + 1) * 10.0f);
    g.set_input(second_input, second.data(), Precision::FP16);
    g.recurrent_cache_write(second_input, cache_state);
    g.execute();

    const __fp16* cache_after2 = static_cast<const __fp16*>(g.get_output(cache_state));
    if (!cache_after2) return false;
    for (size_t i = 0; i < elements; ++i) {
        if (static_cast<float>(cache_after2[i]) != static_cast<float>(second[i])) return false;
    }

    return true;
}

bool test_recurrent_cache_state_initializes_to_zero() {
    CactusGraph g;
    const std::vector<size_t> shape{3, 5};
    const size_t elements = 3 * 5;
    size_t cache_state = g.recurrent_cache_state(shape, Precision::FP16);
    g.execute();
    const __fp16* data = static_cast<const __fp16*>(g.get_output(cache_state));
    if (!data) return false;
    for (size_t i = 0; i < elements; ++i) {
        if (static_cast<float>(data[i]) != 0.0f) return false;
    }
    return true;
}

bool test_steal_cache_buffer_preserves_source_descriptor() {
    CactusGraph src;
    CactusGraph dst;
    const std::vector<size_t> shape{1, 16, 4, 32};
    const size_t elements = 1 * 16 * 4 * 32;
    size_t src_state = src.recurrent_cache_state(shape, Precision::FP16);
    size_t dst_state = dst.recurrent_cache_state(shape, Precision::FP16);
    src.execute();
    dst.steal_cache_buffer(dst_state, src, src_state);
    if (!dst.get_output(dst_state)) return false;
    if (src.get_output_buffer(src_state).shape != shape) return false;
    src.execute();
    const BufferDesc& buf = src.get_output_buffer(src_state);
    const __fp16* data = static_cast<const __fp16*>(buf.get_data());
    if (!data) return false;
    if (buf.byte_size != elements * sizeof(__fp16)) return false;
    for (size_t i = 0; i < elements; ++i) {
        if (static_cast<float>(data[i]) != 0.0f) return false;
    }
    return true;
}

bool test_recurrent_cache_write_rejects_non_state_input() {
    CactusGraph g;
    size_t plain_input = g.input({2, 4}, Precision::FP16);
    std::vector<__fp16> zeros(8, static_cast<__fp16>(0.0f));
    g.set_input(plain_input, zeros.data(), Precision::FP16);
    size_t new_value = g.input({2, 4}, Precision::FP16);
    g.set_input(new_value, zeros.data(), Precision::FP16);
    try {
        g.recurrent_cache_write(new_value, plain_input);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

bool test_recurrent_cache_write_rejects_shape_mismatch() {
    CactusGraph g;

    size_t cache_state = g.recurrent_cache_state({2, 4}, Precision::FP16);

    size_t new_value = g.input({2, 8}, Precision::FP16);
    std::vector<__fp16> data(16, static_cast<__fp16>(0.0f));
    g.set_input(new_value, data.data(), Precision::FP16);

    try {
        g.recurrent_cache_write(new_value, cache_state);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

bool test_allocate_buffers_preserves_recurrent_state_init() {
    CactusGraph g;
    const std::vector<size_t> shape{3, 5};
    const size_t elements = 3 * 5;
    size_t cache_state = g.recurrent_cache_state(shape, Precision::FP16);
    g.allocate_buffers();
    g.execute();
    const __fp16* data = static_cast<const __fp16*>(g.get_output(cache_state));
    if (!data) return false;
    for (size_t i = 0; i < elements; ++i) {
        if (static_cast<float>(data[i]) != 0.0f) return false;
    }
    return true;
}

bool test_conv_cache_initialize_rejects_hidden_dim_mismatch() {
    CactusGraph g;
    size_t cache = g.conv_cache_state(/*window=*/4, /*hidden_dim=*/8);
    size_t rows = g.input({2, 7}, Precision::FP16);
    std::vector<__fp16> data(14, static_cast<__fp16>(0.0f));
    g.set_input(rows, data.data(), Precision::FP16);
    try {
        g.conv_cache_initialize(rows, cache);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

bool test_conv_cache_initialize_exact_window() {
    CactusGraph g;
    constexpr size_t ws = 4, hd = 8;
    size_t cache = g.conv_cache_state(ws, hd);

    size_t rows = g.input({ws, hd}, Precision::FP16);
    std::vector<__fp16> data(ws * hd);
    for (size_t r = 0; r < ws; ++r) {
        for (size_t c = 0; c < hd; ++c) {
            data[r * hd + c] = static_cast<__fp16>(static_cast<float>(r + 1));
        }
    }
    g.set_input(rows, data.data(), Precision::FP16);
    g.conv_cache_initialize(rows, cache);
    g.execute();

    g.soft_reset_keep_pool();
    size_t append_input = g.input({1, hd}, Precision::FP16);
    std::vector<__fp16> append_data(hd, static_cast<__fp16>(99.0f));
    g.set_input(append_input, append_data.data(), Precision::FP16);
    size_t window = g.conv_cache_append(append_input, cache);
    g.execute();

    const __fp16* w = static_cast<const __fp16*>(g.get_output(window));
    if (!w) return false;
    const float expected[ws] = {2.0f, 3.0f, 4.0f, 99.0f};
    for (size_t r = 0; r < ws; ++r) {
        for (size_t c = 0; c < hd; ++c) {
            if (static_cast<float>(w[r * hd + c]) != expected[r]) return false;
        }
    }
    return true;
}

bool test_conv_cache_initialize_single_row() {
    CactusGraph g;
    constexpr size_t ws = 4, hd = 8;
    size_t cache = g.conv_cache_state(ws, hd);

    size_t rows = g.input({1, hd}, Precision::FP16);
    std::vector<__fp16> data(hd, static_cast<__fp16>(42.0f));
    g.set_input(rows, data.data(), Precision::FP16);
    g.conv_cache_initialize(rows, cache);
    g.execute();

    g.soft_reset_keep_pool();
    size_t append_input = g.input({1, hd}, Precision::FP16);
    std::vector<__fp16> append_data(hd, static_cast<__fp16>(7.0f));
    g.set_input(append_input, append_data.data(), Precision::FP16);
    size_t window = g.conv_cache_append(append_input, cache);
    g.execute();

    const __fp16* w = static_cast<const __fp16*>(g.get_output(window));
    if (!w) return false;
    const float expected[ws] = {0.0f, 0.0f, 42.0f, 7.0f};
    for (size_t r = 0; r < ws; ++r) {
        for (size_t c = 0; c < hd; ++c) {
            if (static_cast<float>(w[r * hd + c]) != expected[r]) return false;
        }
    }
    return true;
}

bool test_allocate_buffers_idempotent() {
    CactusGraph g;
    const std::vector<size_t> shape{2, 4};
    size_t cache_state = g.recurrent_cache_state(shape, Precision::FP16);
    g.allocate_buffers();
    g.allocate_buffers();
    g.execute();
    const __fp16* data = static_cast<const __fp16*>(g.get_output(cache_state));
    if (!data) return false;
    for (size_t i = 0; i < 8; ++i) {
        if (static_cast<float>(data[i]) != 0.0f) return false;
    }
    return true;
}

bool test_recurrent_cache_write_idempotent() {
    CactusGraph g;
    const std::vector<size_t> shape{2, 4};
    size_t cache_state = g.recurrent_cache_state(shape, Precision::FP16);

    size_t new_value = g.input(shape, Precision::FP16);
    std::vector<__fp16> data(8);
    for (size_t i = 0; i < 8; ++i) data[i] = static_cast<__fp16>(static_cast<float>(i + 1));
    g.set_input(new_value, data.data(), Precision::FP16);

    g.recurrent_cache_write(new_value, cache_state);
    g.execute();

    g.soft_reset_keep_pool();
    size_t new_value2 = g.input(shape, Precision::FP16);
    g.set_input(new_value2, data.data(), Precision::FP16);
    g.recurrent_cache_write(new_value2, cache_state);
    g.execute();

    const __fp16* read = static_cast<const __fp16*>(g.get_output(cache_state));
    if (!read) return false;
    for (size_t i = 0; i < 8; ++i) {
        if (static_cast<float>(read[i]) != static_cast<float>(data[i])) return false;
    }
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
    runner.run_test("KV Cache Append Full Window Eviction", test_kv_cache_append_full_window_eviction());
    runner.run_test("Padded Rollback No Eviction", test_padded_rollback_no_eviction());
    runner.run_test("Padded Rollback Eviction Overshoot", test_padded_rollback_eviction_overshoot());
    runner.run_test("Padded Rollback Only Pads Evict", test_padded_rollback_only_pads_evict());
    runner.run_test("Padded Rollback Empty Cache", test_padded_rollback_empty_cache());
    runner.run_test("Attention Cached Basic", test_attention_cached_basic());
    runner.run_test("Attention Cached Multistep", test_attention_cached_multistep());
    runner.run_test("KV Cache Invalidate", test_kv_cache_invalidate());
    runner.run_test("Conv Cache State Init", test_conv_cache_state_init());
    runner.run_test("Conv Cache Append Basic", test_conv_cache_append_basic());
    runner.run_test("Conv Cache Circular", test_conv_cache_append_circular());
    runner.run_test("Conv Cache Persistent", test_conv_cache_persistent());
    runner.run_test("Conv Cache Initialize Populates Trailing", test_conv_cache_initialize_populates_trailing_rows());
    runner.run_test("Conv Cache Initialize Rejects Non-State Input", test_conv_cache_initialize_rejects_non_state_input());
    runner.run_test("Conv Cache Initialize Resets Dirty Cache", test_conv_cache_initialize_resets_dirty_cache());
    runner.run_test("Conv Cache Initialize Output Zero Byte", test_conv_cache_initialize_output_is_zero_byte());
    runner.run_test("Recurrent Cache State Initializes to Zero", test_recurrent_cache_state_initializes_to_zero());
    runner.run_test("Steal Cache Buffer Preserves Source Descriptor", test_steal_cache_buffer_preserves_source_descriptor());
    runner.run_test("Recurrent Cache Write Carries State", test_recurrent_cache_write_carries_state_across_executions());
    runner.run_test("Recurrent Cache Write Rejects Non-State Input", test_recurrent_cache_write_rejects_non_state_input());
    runner.run_test("Recurrent Cache Write Rejects Mismatch", test_recurrent_cache_write_rejects_shape_mismatch());
    runner.run_test("Allocate Buffers Preserves Recurrent Init", test_allocate_buffers_preserves_recurrent_state_init());
    runner.run_test("Conv Cache Initialize Rejects Hidden Dim Mismatch", test_conv_cache_initialize_rejects_hidden_dim_mismatch());
    runner.run_test("Conv Cache Initialize Exact Window", test_conv_cache_initialize_exact_window());
    runner.run_test("Conv Cache Initialize Single Row", test_conv_cache_initialize_single_row());
    runner.run_test("Allocate Buffers Idempotent", test_allocate_buffers_idempotent());
    runner.run_test("Recurrent Cache Write Idempotent", test_recurrent_cache_write_idempotent());
    runner.print_benchmarks_header();
    runner.run_bench("benchmarks", run_benchmarks());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
