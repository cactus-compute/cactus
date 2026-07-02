#include "test_utils.h"
#include <cmath>
#include <functional>
#include <iostream>
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
        if (metal.empty()) return true;
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

}

int main() {
    TestRunner runner("Metal Parity (CPU vs Metal per-op)");
    if (!metal_present()) {
        std::cout << "Metal backend unavailable; skipping parity suite.\n";
        return 0;
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

    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
