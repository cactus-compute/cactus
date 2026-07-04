#include "test_utils.h"
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

using namespace TestUtils;

namespace {

std::vector<__fp16> random_fp16(size_t n, float lo = -1.0f, float hi = 1.0f, uint32_t seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(lo, hi);
    std::vector<__fp16> out(n);
    for (auto& v : out) v = static_cast<__fp16>(dis(gen));
    return out;
}

struct ParityCase {
    std::vector<std::vector<size_t>> input_shapes;
    std::vector<std::vector<__fp16>> input_data;
    std::vector<std::vector<float>> input_data_f32;
    std::function<size_t(CactusGraph&, const std::vector<size_t>&)> build;
    float tolerance = 5e-2f;

    size_t add_input(const std::vector<size_t>& shape, float lo = -1.0f, float hi = 1.0f) {
        size_t n = 1;
        for (size_t d : shape) n *= d;
        input_shapes.push_back(shape);
        input_data.push_back(random_fp16(n, lo, hi, 42 + (uint32_t)input_shapes.size()));
        input_data_f32.push_back({});
        return input_shapes.size() - 1;
    }

    size_t add_input_data(const std::vector<size_t>& shape, std::vector<__fp16> data) {
        input_shapes.push_back(shape);
        input_data.push_back(std::move(data));
        input_data_f32.push_back({});
        return input_shapes.size() - 1;
    }

    size_t add_input_f32(const std::vector<size_t>& shape, const std::vector<float>& data) {
        input_shapes.push_back(shape);
        input_data.push_back({});
        input_data_f32.push_back(data);
        return input_shapes.size() - 1;
    }

    std::vector<float> run(const char* backend) {
        if (cactus_backend_select(backend) != 0) return {};
        CactusGraph graph;
        std::vector<size_t> ids;
        for (size_t i = 0; i < input_shapes.size(); ++i) {
            bool f32 = !input_data_f32[i].empty();
            ids.push_back(graph.input(input_shapes[i], f32 ? Precision::FP32 : Precision::FP16));
        }
        size_t out_id = build(graph, ids);
        for (size_t i = 0; i < ids.size(); ++i) {
            bool f32 = !input_data_f32[i].empty();
            graph.set_input(ids[i],
                f32 ? static_cast<void*>(input_data_f32[i].data())
                    : static_cast<void*>(input_data[i].data()),
                f32 ? Precision::FP32 : Precision::FP16);
        }
        graph.execute();
        const auto& desc = graph.get_output_buffer(out_id);
        std::vector<float> result(desc.total_size);
        if (desc.precision == Precision::FP16) {
            const __fp16* p = static_cast<const __fp16*>(graph.get_output(out_id));
            for (size_t i = 0; i < result.size(); ++i) result[i] = static_cast<float>(p[i]);
        } else {
            const float* p = static_cast<const float*>(graph.get_output(out_id));
            for (size_t i = 0; i < result.size(); ++i) result[i] = p[i];
        }
        graph.hard_reset();
        cactus_backend_select("auto");
        return result;
    }

    bool check() {
        std::vector<float> cpu = run("cpu");
        std::vector<float> metal = run("metal");
        if (cpu.size() != metal.size() || cpu.empty()) return false;
        for (size_t i = 0; i < cpu.size(); ++i) {
            float scale = std::max(1.0f, std::fabs(cpu[i]));
            if (std::fabs(cpu[i] - metal[i]) > tolerance * scale) {
                std::cout << "    mismatch at " << i << ": cpu=" << cpu[i]
                          << " metal=" << metal[i] << "\n";
                return false;
            }
        }
        return true;
    }
};

bool metal_present() {
    bool ok = cactus_backend_select("metal") == 0;
    cactus_backend_select("auto");
    return ok;
}

bool parity_unary(size_t (CactusGraph::*fn)(size_t), float lo = -1.0f, float hi = 1.0f) {
    ParityCase c;
    c.add_input({4, 33}, lo, hi);
    c.build = [fn](CactusGraph& g, const std::vector<size_t>& in) { return (g.*fn)(in[0]); };
    return c.check();
}

bool parity_scalar(size_t (CactusGraph::*fn)(size_t, float), float p, float lo = -1.0f, float hi = 1.0f) {
    ParityCase c;
    c.add_input({4, 33}, lo, hi);
    c.build = [fn, p](CactusGraph& g, const std::vector<size_t>& in) { return (g.*fn)(in[0], p); };
    return c.check();
}

bool parity_reduce(size_t (CactusGraph::*fn)(size_t, int)) {
    for (int axis = 0; axis < 3; ++axis) {
        ParityCase c;
        c.add_input({3, 5, 17});
        c.build = [fn, axis](CactusGraph& g, const std::vector<size_t>& in) { return (g.*fn)(in[0], axis); };
        if (!c.check()) return false;
    }
    return true;
}

bool parity_not_equal() {
    ParityCase c;
    c.add_input({2, 31});
    c.add_input({2, 31});
    c.build = [](CactusGraph& g, const std::vector<size_t>& in) { return g.not_equal(in[0], in[1]); };
    return c.check();
}

bool parity_broadcast_binary() {
    {
        ParityCase c;
        c.add_input({4, 33});
        c.add_input({1, 33});
        c.build = [](CactusGraph& g, const std::vector<size_t>& in) { return g.multiply(in[0], in[1]); };
        if (!c.check()) return false;
    }
    {
        ParityCase c;
        c.add_input({4, 33});
        c.add_input({1, 33});
        c.build = [](CactusGraph& g, const std::vector<size_t>& in) { return g.not_equal(in[0], in[1]); };
        if (!c.check()) return false;
    }
    return true;
}

bool parity_concat() {
    for (int axis = 0; axis < 2; ++axis) {
        ParityCase c;
        c.add_input({3, 7});
        c.add_input({axis == 0 ? 5u : 3u, axis == 0 ? 7u : 11u});
        c.build = [axis](CactusGraph& g, const std::vector<size_t>& in) { return g.concat(in[0], in[1], axis); };
        if (!c.check()) return false;
    }
    return true;
}

bool parity_gather() {
    ParityCase c;
    c.add_input({16, 24});
    c.add_input_f32({6}, {3.0f, 0.0f, 15.0f, 7.0f, 7.0f, 2.0f});
    c.build = [](CactusGraph& g, const std::vector<size_t>& in) { return g.gather(in[0], in[1]); };
    return c.check();
}

bool parity_rope(bool gptj) {
    ParityCase c;
    c.add_input({1, 6, 4, 32});
    c.build = [gptj](CactusGraph& g, const std::vector<size_t>& in) {
        return gptj ? g.rope_gptj(in[0], 10000.0f, 3, 16) : g.rope(in[0], 10000.0f, 3);
    };
    return c.check();
}

bool parity_maxpool1d() {
    ParityCase c;
    c.add_input({2, 5, 29});
    c.build = [](CactusGraph& g, const std::vector<size_t>& in) { return g.maxpool1d(in[0], 3, 2); };
    return c.check();
}

bool parity_bilinear() {
    ParityCase c;
    c.add_input({64, 12});
    c.build = [](CactusGraph& g, const std::vector<size_t>& in) {
        return g.bilinear_interpolation(in[0], 13, 11, false);
    };
    return c.check();
}

bool parity_conv1d() {
    ParityCase c;
    c.add_input({2, 6, 31});
    c.add_input({4, 6, 5});
    c.build = [](CactusGraph& g, const std::vector<size_t>& in) { return g.conv1d(in[0], in[1], 2); };
    return c.check();
}

bool parity_conv1d_k7s3() {
    ParityCase c;
    c.add_input({1, 8, 40});
    c.add_input({8, 7, 16});
    c.add_input({16});
    c.build = [](CactusGraph& g, const std::vector<size_t>& in) { return g.conv1d_k7s3(in[0], in[1], in[2]); };
    return c.check();
}

bool parity_conv1d_causal() {
    for (size_t dil : {1u, 2u}) {
        ParityCase c;
        c.add_input({1, 21, 8});
        c.add_input({8, 1, 4});
        c.build = [dil](CactusGraph& g, const std::vector<size_t>& in) {
            return g.conv1d_causal(in[0], in[1], 4, dil);
        };
        if (!c.check()) return false;
    }
    return true;
}

bool parity_conv1d_dw_k9() {
    ParityCase c;
    c.add_input({1, 25, 6});
    c.add_input({6, 1, 9});
    c.add_input({6});
    c.build = [](CactusGraph& g, const std::vector<size_t>& in) {
        return g.conv1d_same_depthwise_k9(in[0], in[1], in[2]);
    };
    return c.check();
}

bool parity_conv1d_pointwise() {
    ParityCase c;
    c.add_input({1, 13, 16});
    c.add_input({24, 16});
    c.add_input({24});
    c.build = [](CactusGraph& g, const std::vector<size_t>& in) {
        return g.conv1d_pointwise(in[0], in[1], in[2]);
    };
    return c.check();
}

bool parity_conv2d(size_t (CactusGraph::*fn)(size_t, size_t, size_t), bool depthwise, bool pointwise) {
    ParityCase c;
    c.add_input({1, 4, 11, 13});
    if (depthwise) c.add_input({4, 1, 3, 3});
    else if (pointwise) c.add_input({6, 4, 1, 1});
    else c.add_input({6, 4, 3, 3});
    c.add_input({pointwise || depthwise ? (depthwise ? 4u : 6u) : 6u});
    c.build = [fn](CactusGraph& g, const std::vector<size_t>& in) { return (g.*fn)(in[0], in[1], in[2]); };
    return c.check();
}

bool parity_batchnorm() {
    ParityCase c;
    c.add_input({2, 5, 9});
    c.add_input({5});
    c.add_input({5});
    c.add_input({5});
    c.add_input({5}, 0.5f, 1.5f);
    c.build = [](CactusGraph& g, const std::vector<size_t>& in) {
        return g.batchnorm(in[0], in[1], in[2], in[3], in[4], 1);
    };
    return c.check();
}

bool parity_groupnorm() {
    ParityCase c;
    c.add_input({2, 8, 6});
    c.add_input({8});
    c.add_input({8});
    c.build = [](CactusGraph& g, const std::vector<size_t>& in) {
        return g.groupnorm(in[0], in[1], in[2], 4);
    };
    return c.check();
}

bool parity_cumsum() {
    for (int axis = 0; axis < 2; ++axis) {
        ParityCase c;
        c.add_input({4, 19});
        c.build = [axis](CactusGraph& g, const std::vector<size_t>& in) { return g.cumsum(in[0], axis); };
        if (!c.check()) return false;
    }
    return true;
}

bool parity_softmax() {
    ParityCase c;
    c.add_input({5, 333}, -4.0f, 4.0f);
    c.tolerance = 5e-3f;
    c.build = [](CactusGraph& g, const std::vector<size_t>& in) { return g.softmax(in[0], -1); };
    return c.check();
}

bool parity_rms_norm() {
    for (size_t rows : {1u, 7u}) {
        ParityCase c;
        c.add_input({rows, 256});
        c.add_input({256}, 0.5f, 1.5f);
        c.tolerance = 1e-2f;
        c.build = [](CactusGraph& g, const std::vector<size_t>& in) {
            return g.rms_norm(in[0], in[1], 1e-5f);
        };
        if (!c.check()) return false;
    }
    return true;
}

bool parity_residual_rms_norm() {
    for (size_t rows : {1u, 6u}) {
        ParityCase c;
        c.add_input({rows, 192});
        c.add_input({rows, 192});
        c.add_input({192}, 0.5f, 1.5f);
        c.tolerance = 1e-2f;
        c.build = [](CactusGraph& g, const std::vector<size_t>& in) {
            return g.rms_norm(g.add(in[0], in[1]), in[2], 1e-5f);
        };
        if (!c.check()) return false;
    }
    return true;
}

bool parity_elemwise_chain() {
    ParityCase c;
    c.add_input({4, 96});
    c.add_input({4, 96});
    c.add_input({4, 96});
    c.tolerance = 1e-2f;
    c.build = [](CactusGraph& g, const std::vector<size_t>& in) {
        return g.add(g.multiply(g.silu(in[0]), in[1]), g.scalar_multiply(in[2], 0.5f));
    };
    return c.check();
}

bool parity_flash_attention() {
    for (size_t T : {64u, 89u, 121u}) {
        ParityCase c;
        const size_t H = 2, D = 64, S = 729;
        c.add_input({1, T, H, D});
        c.add_input({1, S, H, D});
        c.add_input({1, S, H, D});
        c.tolerance = 2e-2f;
        c.build = [D](CactusGraph& g, const std::vector<size_t>& in) {
            return g.attention(in[0], in[1], in[2], 1.0f / std::sqrt((float)D), false);
        };
        if (!c.check()) return false;
    }
    return true;
}

bool parity_attention_causal() {
    ParityCase c;
    const size_t T = 37, H = 4, HKV = 2, D = 32;
    c.add_input({1, T, H, D});
    c.add_input({1, T, HKV, D});
    c.add_input({1, T, HKV, D});
    c.tolerance = 2e-2f;
    c.build = [D](CactusGraph& g, const std::vector<size_t>& in) {
        return g.attention(in[0], in[1], in[2], 1.0f / std::sqrt((float)D), true);
    };
    return c.check();
}

bool parity_equality_exact() {
    const size_t n = 2 * 31;
    std::vector<__fp16> a = random_fp16(n, -1.0f, 1.0f, 7);
    std::vector<__fp16> b = random_fp16(n, -1.0f, 1.0f, 8);
    for (size_t i = 0; i < n; i += 2) b[i] = a[i];
    {
        ParityCase c;
        c.add_input_data({2, 31}, a);
        c.add_input_data({2, 31}, b);
        c.tolerance = 1e-3f;
        c.build = [](CactusGraph& g, const std::vector<size_t>& in) { return g.not_equal(in[0], in[1]); };
        if (!c.check()) return false;
    }
    {
        std::vector<__fp16> z = a;
        for (size_t i = 0; i < n; i += 2) z[i] = (__fp16)0.0f;
        ParityCase c;
        c.add_input_data({2, 31}, z);
        c.tolerance = 1e-3f;
        c.build = [](CactusGraph& g, const std::vector<size_t>& in) { return g.scalar_not_equal(in[0], 0.0f); };
        if (!c.check()) return false;
    }
    return true;
}

struct CacheStepData {
    std::vector<__fp16> q, k, v;
    size_t tokens;
};

std::vector<float> run_cached_attention(const char* backend, size_t ceiling, size_t window, size_t sink,
                                        size_t h, size_t kv, size_t d,
                                        const std::vector<CacheStepData>& steps, size_t collect_from) {
    if (cactus_backend_select(backend) != 0) return {};
    const size_t sentinel = std::numeric_limits<size_t>::max();
    const float scale = 1.0f / std::sqrt((float)d);
    std::vector<float> collected;
    CactusGraph g;
    size_t kc = g.kv_cache_state(ceiling, kv, d, window, sink);
    size_t vc = g.kv_cache_state(ceiling, kv, d, window, sink);
    for (size_t si = 0; si < steps.size(); ++si) {
        const auto& data = steps[si];
        size_t s = data.tokens;
        size_t iq = g.input({1, s, h, d}, Precision::FP16);
        size_t ik = g.input({1, s, kv, d}, Precision::FP16);
        size_t iv = g.input({1, s, kv, d}, Precision::FP16);
        g.set_input(iq, const_cast<__fp16*>(data.q.data()), Precision::FP16);
        g.set_input(ik, const_cast<__fp16*>(data.k.data()), Precision::FP16);
        g.set_input(iv, const_cast<__fp16*>(data.v.data()), Precision::FP16);
        g.kv_cache_append(ik, kc, window, sink);
        g.kv_cache_append(iv, vc, window, sink);
        size_t attn = g.attention_cached(iq, ik, iv, kc, vc, scale, sentinel, window);
        g.execute();
        if (si >= collect_from) {
            const __fp16* p = static_cast<const __fp16*>(g.get_output(attn));
            for (size_t i = 0; i < s * h * d; ++i) collected.push_back((float)p[i]);
        }
        g.soft_reset();
    }
    g.hard_reset();
    cactus_backend_select("auto");
    return collected;
}

bool check_cached_attention(size_t ceiling, size_t window, size_t sink,
                            const std::vector<size_t>& chunk_sizes, size_t decode_steps,
                            float tolerance, size_t collect_from = 0) {
    const size_t h = 4, kv = 2, d = 32;
    std::vector<CacheStepData> steps;
    uint32_t seed = 100;
    for (size_t s : chunk_sizes) {
        steps.push_back({random_fp16(s * h * d, -1.0f, 1.0f, seed),
                         random_fp16(s * kv * d, -1.0f, 1.0f, seed + 1),
                         random_fp16(s * kv * d, -1.0f, 1.0f, seed + 2), s});
        seed += 3;
    }
    for (size_t i = 0; i < decode_steps; ++i) {
        steps.push_back({random_fp16(h * d, -1.0f, 1.0f, seed),
                         random_fp16(kv * d, -1.0f, 1.0f, seed + 1),
                         random_fp16(kv * d, -1.0f, 1.0f, seed + 2), 1});
        seed += 3;
    }
    std::vector<float> cpu = run_cached_attention("cpu", ceiling, window, sink, h, kv, d, steps, collect_from);
    std::vector<float> metal = run_cached_attention("metal", ceiling, window, sink, h, kv, d, steps, collect_from);
    if (cpu.size() != metal.size() || cpu.empty()) return false;
    for (size_t i = 0; i < cpu.size(); ++i) {
        float scale = std::max(1.0f, std::fabs(cpu[i]));
        if (std::fabs(cpu[i] - metal[i]) > tolerance * scale) {
            std::cout << "    mismatch at " << i << " (token " << i / (4 * 32) << "): cpu=" << cpu[i]
                      << " metal=" << metal[i] << "\n";
            return false;
        }
    }
    return true;
}

bool parity_sliding_window_cache() {
    if (!check_cached_attention(/*ceiling=*/64, /*window=*/8, /*sink=*/2,
                                /*chunks=*/{5}, /*decode_steps=*/12, 5e-2f)) return false;
    return check_cached_attention(/*ceiling=*/64, /*window=*/8, /*sink=*/2,
                                  /*chunks=*/{4, 8}, /*decode_steps=*/6, 5e-2f, /*collect_from=*/2);
}

bool parity_cache_growth() {
    return check_cached_attention(/*ceiling=*/2048, /*window=*/0, /*sink=*/2,
                                  /*chunks=*/{600}, /*decode_steps=*/3, 5e-2f);
}

}

int main() {
    TestRunner runner("Metal Parity (CPU vs Metal per-op)");
    if (!metal_present()) {
#if defined(__APPLE__)
        if (std::getenv("CACTUS_PARITY_ALLOW_SKIP")) {
            std::cout << "Metal backend unavailable; skipping parity suite (CACTUS_PARITY_ALLOW_SKIP).\n";
            return 0;
        }
        std::cout << "FAIL: Metal backend unavailable on Apple hardware "
                     "(set CACTUS_PARITY_ALLOW_SKIP=1 to skip).\n";
        return 1;
#else
        std::cout << "Metal backend unavailable; skipping parity suite.\n";
        return 0;
#endif
    }

    runner.run_test("abs", parity_unary(&CactusGraph::abs));
    runner.run_test("scalar_exp", parity_unary(&CactusGraph::scalar_exp));
    runner.run_test("scalar_sqrt", parity_unary(&CactusGraph::scalar_sqrt, 0.1f, 2.0f));
    runner.run_test("scalar_cos", parity_unary(&CactusGraph::scalar_cos));
    runner.run_test("scalar_sin", parity_unary(&CactusGraph::scalar_sin));
    runner.run_test("scalar_log", parity_unary(&CactusGraph::scalar_log, 0.1f, 3.0f));
    runner.run_test("gelu_erf", parity_unary(&CactusGraph::gelu_erf));
    runner.run_test("sigmoid", parity_unary(&CactusGraph::sigmoid));
    runner.run_test("pow", parity_scalar(&CactusGraph::pow, 2.0f, 0.1f, 2.0f));
    runner.run_test("leaky_relu", parity_scalar(&CactusGraph::leaky_relu, 0.1f));
    runner.run_test("scalar_not_equal", parity_scalar(&CactusGraph::scalar_not_equal, 0.0f));
    runner.run_test("not_equal", parity_not_equal());
    runner.run_test("broadcast_binary", parity_broadcast_binary());
    runner.run_test("sum", parity_reduce(&CactusGraph::sum));
    runner.run_test("mean", parity_reduce(&CactusGraph::mean));
    runner.run_test("variance", parity_reduce(&CactusGraph::variance));
    runner.run_test("min", parity_reduce(&CactusGraph::min));
    runner.run_test("max", parity_reduce(&CactusGraph::max));
    runner.run_test("cumsum", parity_cumsum());
    runner.run_test("concat", parity_concat());
    runner.run_test("gather", parity_gather());
    runner.run_test("rope", parity_rope(false));
    runner.run_test("rope_gptj", parity_rope(true));
    runner.run_test("maxpool1d", parity_maxpool1d());
    runner.run_test("bilinear_interpolation", parity_bilinear());
    runner.run_test("conv1d", parity_conv1d());
    runner.run_test("conv1d_k7s3", parity_conv1d_k7s3());
    runner.run_test("conv1d_causal", parity_conv1d_causal());
    runner.run_test("conv1d_same_depthwise_k9", parity_conv1d_dw_k9());
    runner.run_test("conv1d_pointwise", parity_conv1d_pointwise());
    runner.run_test("conv2d_k3s2p1", parity_conv2d(&CactusGraph::conv2d_k3s2p1, false, false));
    runner.run_test("conv2d_k3s1p1", parity_conv2d(&CactusGraph::conv2d_k3s1p1, false, false));
    runner.run_test("conv2d_depthwise_k3s2p1", parity_conv2d(&CactusGraph::conv2d_depthwise_k3s2p1, true, false));
    runner.run_test("conv2d_pointwise_1x1", parity_conv2d(&CactusGraph::conv2d_pointwise_1x1, false, true));
    runner.run_test("batchnorm", parity_batchnorm());
    runner.run_test("groupnorm", parity_groupnorm());
    runner.run_test("softmax", parity_softmax());
    runner.run_test("rms_norm", parity_rms_norm());
    runner.run_test("residual_rms_norm (fused)", parity_residual_rms_norm());
    runner.run_test("elemwise_chain (fused)", parity_elemwise_chain());
    runner.run_test("flash_attention (T%64!=0)", parity_flash_attention());
    runner.run_test("attention_causal", parity_attention_causal());
    runner.run_test("equality_exact", parity_equality_exact());
    runner.run_test("sliding_window_cache (ring wrap)", parity_sliding_window_cache());
    runner.run_test("cache_growth", parity_cache_growth());

    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
