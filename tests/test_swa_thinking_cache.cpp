#include "../cactus/cactus.h"
#include "test_utils.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>

using namespace cactus::engine;
using namespace std;

namespace {

constexpr size_t WINDOW = 512;
constexpr size_t RING = 2 * WINDOW;
constexpr size_t KV_HEADS = 1;
constexpr size_t HEAD_DIM = 32;
constexpr size_t ELEMS_PER_TOKEN = KV_HEADS * HEAD_DIM;

vector<__fp16> make_token_fp16(float marker) {
    vector<__fp16> t(ELEMS_PER_TOKEN);
    for (size_t i = 0; i < ELEMS_PER_TOKEN; i++)
        t[i] = static_cast<__fp16>(marker + static_cast<float>(i) * 0.01f);
    return t;
}

float read_slot_marker(const KVCache& cache, size_t layer_idx, uint32_t slot) {
    const auto& c = cache.layer_caches[layer_idx];
    size_t elements_per_slot = c.kv_heads * c.head_dim;
    if (cache.precision == Precision::INT8) {
        const int8_t* q = reinterpret_cast<const int8_t*>(c.keys.data()) + slot * elements_per_slot;
        const float* scales = c.key_scales.data() + slot * c.kv_heads * ((c.head_dim + KV_QUANT_GROUP_SIZE - 1) / KV_QUANT_GROUP_SIZE);
        return static_cast<float>(q[0]) * scales[0];
    }
    const __fp16* p = reinterpret_cast<const __fp16*>(c.keys.data()) + slot * elements_per_slot;
    return static_cast<float>(p[0]);
}

KVCache make_cache(Precision precision) {
    KVCache cache;
    cache.init(1, 8192, {HEAD_DIM}, {KV_HEADS}, precision);
    cache.configure_swa_layers({WINDOW});
    return cache;
}

void append(KVCache& cache, float marker) {
    auto tok = make_token_fp16(marker);
    cache.append_swa_token(0, tok.data(), tok.data());
    cache.commit_token();
}

bool test_basic_non_thinking_grows_then_caps() {
    KVCache cache = make_cache(Precision::FP16);

    for (size_t i = 0; i < 100; i++) append(cache, 100.0f + i);
    if (cache.get_swa_count(0) != 100) return false;
    if (cache.get_swa_head(0) != 100) return false;
    if (cache.get_total_seq_len() != 100) return false;

    for (size_t i = 100; i < RING; i++) append(cache, 100.0f + i);
    if (cache.get_swa_count(0) != RING) return false;
    if (cache.get_swa_head(0) != 0) return false;
    if (cache.get_total_seq_len() != RING) return false;

    append(cache, 1234.0f);
    if (cache.get_swa_count(0) != RING) return false;
    if (cache.get_swa_head(0) != 1) return false;
    if (cache.get_total_seq_len() != RING + 1) return false;

    if (std::abs(read_slot_marker(cache, 0, 0) - 1234.0f) > 2.0f) return false;

    return true;
}

bool test_enter_thinking_snapshots_state() {
    KVCache cache = make_cache(Precision::FP16);
    for (size_t i = 0; i < 300; i++) append(cache, static_cast<float>(i));

    cache.enter_thinking();
    if (!cache.in_thinking) return false;
    if (cache.thinking_count != 0) return false;
    if (cache.get_swa_think_anchor(0) != 300) return false;
    if (cache.get_swa_head(0) != 300) return false;
    if (cache.get_swa_count(0) != 300) return false;
    return true;
}

bool test_thinking_small_segment_exit_rewind() {
    KVCache cache = make_cache(Precision::FP16);
    for (size_t i = 0; i < 300; i++) append(cache, static_cast<float>(i));
    size_t total_before = cache.get_total_seq_len();

    cache.enter_thinking();
    uint32_t anchor = cache.get_swa_think_anchor(0);

    for (size_t i = 0; i < 50; i++) append(cache, 1000.0f + i);
    if (cache.thinking_count != 50) return false;
    if (cache.get_swa_head(0) != anchor + 50) return false;
    if (cache.get_swa_count(0) != 350) return false;
    if (cache.get_total_seq_len() != total_before + 50) return false;

    cache.exit_thinking();
    if (cache.in_thinking) return false;
    if (cache.thinking_count != 0) return false;
    if (cache.get_swa_head(0) != anchor) return false;
    if (cache.get_swa_count(0) != 300) return false;
    if (cache.get_total_seq_len() != total_before) return false;

    return true;
}

bool test_thinking_exactly_window() {
    KVCache cache = make_cache(Precision::FP16);
    for (size_t i = 0; i < 100; i++) append(cache, static_cast<float>(i));

    cache.enter_thinking();
    uint32_t anchor = cache.get_swa_think_anchor(0);

    for (size_t i = 0; i < WINDOW; i++) append(cache, 2000.0f + i);
    if (cache.thinking_count != WINDOW) return false;
    // head after 512 thinking writes: (anchor + 512%512) % ring = anchor
    if (cache.get_swa_head(0) != anchor) return false;
    if (cache.get_swa_count(0) != 100 + WINDOW) return false;

    // Last-written thinking (marker 2000+511) sits at slot (anchor + 511) % ring.
    float last = read_slot_marker(cache, 0, (anchor + WINDOW - 1) % RING);
    if (std::abs(last - (2000.0f + (WINDOW - 1))) > 1.0f) return false;

    cache.exit_thinking();
    if (cache.get_swa_head(0) != anchor) return false;
    if (cache.get_swa_count(0) != 100) return false;

    return true;
}

bool test_thinking_sub_ring_rollover() {
    KVCache cache = make_cache(Precision::FP16);
    for (size_t i = 0; i < 100; i++) append(cache, static_cast<float>(i));

    cache.enter_thinking();
    uint32_t anchor = cache.get_swa_think_anchor(0);

    size_t N = WINDOW + 200;  // 712 thinking writes, 200 past rollover
    for (size_t i = 0; i < N; i++) append(cache, 3000.0f + i);

    if (cache.thinking_count != N) return false;
    if (cache.get_swa_head(0) != (anchor + (N % WINDOW)) % RING) return false;

    // Thinking count contribution to cache count is capped at WINDOW.
    if (cache.get_swa_count(0) != 100 + WINDOW) return false;

    // Latest thinking token (marker 3000 + N - 1) is at slot (anchor + (N-1) % WINDOW) % ring.
    uint32_t latest_slot = (anchor + (N - 1) % WINDOW) % RING;
    float latest = read_slot_marker(cache, 0, latest_slot);
    if (std::abs(latest - (3000.0f + (N - 1))) > 2.0f) return false;

    // First thinking token (marker 3000) was at slot anchor; should have been overwritten
    // after the sub-ring wrapped. At N = WINDOW + 200 the token at slot anchor is the
    // 513th thinking token (marker 3000 + WINDOW).
    float at_anchor = read_slot_marker(cache, 0, anchor);
    if (std::abs(at_anchor - (3000.0f + WINDOW)) > 2.0f) return false;

    cache.exit_thinking();
    if (cache.get_swa_head(0) != anchor) return false;
    // count reduced by min(thinking_count, WINDOW) == WINDOW.
    if (cache.get_swa_count(0) != 100) return false;

    return true;
}

bool test_thinking_with_full_prior_cache_overwrites_oldest() {
    KVCache cache = make_cache(Precision::FP16);
    // Fill cache past ring capacity.
    size_t pre_count = 900;
    for (size_t i = 0; i < pre_count; i++) append(cache, static_cast<float>(i));
    // count clamped at RING once we pass 1024.

    uint32_t head_before = cache.get_swa_head(0);
    uint32_t count_before = cache.get_swa_count(0);

    cache.enter_thinking();

    size_t thinking_writes = 600;
    for (size_t i = 0; i < thinking_writes; i++) append(cache, 4000.0f + i);

    if (cache.get_swa_count(0) != RING) return false;
    if (cache.thinking_count != thinking_writes) return false;

    cache.exit_thinking();

    // Exit formula: count -= min(thinking_count, WINDOW) from in-thinking count (RING).
    // => count = RING - WINDOW = WINDOW.
    if (cache.get_swa_head(0) != head_before) return false;
    if (cache.get_swa_count(0) != RING - WINDOW) return false;
    (void)count_before;

    return true;
}

bool test_resumed_non_thinking_after_exit() {
    KVCache cache = make_cache(Precision::FP16);
    for (size_t i = 0; i < 300; i++) append(cache, static_cast<float>(i));
    size_t total_pre = cache.get_total_seq_len();

    cache.enter_thinking();
    uint32_t anchor = cache.get_swa_think_anchor(0);

    for (size_t i = 0; i < 100; i++) append(cache, 5000.0f + i);
    cache.exit_thinking();

    if (cache.get_swa_head(0) != anchor) return false;
    if (cache.get_swa_count(0) != 300) return false;
    if (cache.get_total_seq_len() != total_pre) return false;

    append(cache, 777.0f);
    if (cache.get_swa_head(0) != anchor + 1) return false;
    if (cache.get_swa_count(0) != 301) return false;
    if (cache.get_total_seq_len() != total_pre + 1) return false;

    float written = read_slot_marker(cache, 0, anchor);
    if (std::abs(written - 777.0f) > 1.0f) return false;

    return true;
}

bool test_int8_precision_round_trip() {
    KVCache cache = make_cache(Precision::INT8);

    for (size_t i = 0; i < 50; i++) append(cache, 10.0f + static_cast<float>(i) * 0.5f);
    if (cache.get_swa_count(0) != 50) return false;
    if (cache.get_swa_head(0) != 50) return false;

    cache.enter_thinking();
    for (size_t i = 0; i < WINDOW + 5; i++) append(cache, 100.0f + i);
    if (cache.get_swa_count(0) != 50 + WINDOW) return false;
    if (cache.thinking_count != WINDOW + 5) return false;

    cache.exit_thinking();
    if (cache.get_swa_head(0) != 50) return false;
    if (cache.get_swa_count(0) != 50) return false;

    return true;
}

} // namespace

int main() {
    TestUtils::TestRunner runner("swa_thinking_cache");
    runner.run_test("basic_non_thinking_grows_then_caps", test_basic_non_thinking_grows_then_caps());
    runner.run_test("enter_thinking_snapshots_state", test_enter_thinking_snapshots_state());
    runner.run_test("thinking_small_segment_exit_rewind", test_thinking_small_segment_exit_rewind());
    runner.run_test("thinking_exactly_window", test_thinking_exactly_window());
    runner.run_test("thinking_sub_ring_rollover", test_thinking_sub_ring_rollover());
    runner.run_test("thinking_with_full_prior_cache_overwrites_oldest", test_thinking_with_full_prior_cache_overwrites_oldest());
    runner.run_test("resumed_non_thinking_after_exit", test_resumed_non_thinking_after_exit());
    runner.run_test("int8_precision_round_trip", test_int8_precision_round_trip());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
