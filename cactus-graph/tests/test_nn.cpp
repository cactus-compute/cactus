#include "test_utils.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>

using namespace TestUtils;

bool test_matrix_multiplication() {
    TestUtils::FP16TestFixture fixture("Matrix Multiplication");

    size_t input_a = fixture.create_input({2, 3});
    size_t input_b = fixture.create_input({3, 2});
    size_t matmul_result = fixture.graph().matmul(input_a, input_b, false);

    std::vector<__fp16> data_a = {1, 2, 3, 4, 5, 6};
    std::vector<__fp16> data_b = {1, 2, 3, 4, 5, 6};
    fixture.set_input_data(input_a, data_a);
    fixture.set_input_data(input_b, data_b);
    fixture.execute();

    std::vector<__fp16> expected = {22, 28, 49, 64};
    return fixture.verify_output(matmul_result, expected);
}

bool test_matmul_cq() {
    // Test the graph-level CQ matmul dispatch
    // Creates a CQ4 weight matrix and verifies the graph executes correctly
    const size_t M = 2, K = 128, N = 8;
    const size_t gs = 128;
    const size_t ng = K / gs;
    const uint32_t bits = 4;
    const uint32_t cb_size = 1u << bits;

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    // Activations
    std::vector<__fp16> A(M * K);
    for (auto& v : A) v = static_cast<__fp16>(dist(gen));

    // CQ weight data: packed indices
    uint32_t pgb = cactus_quant_packed_group_bytes(bits, gs);
    std::vector<uint8_t> packed(N * ng * pgb);
    for (auto& v : packed) v = static_cast<uint8_t>(gen() & 0xFF);

    // CQ metadata
    std::vector<__fp16> codebook(cb_size), input_scale(K), input_scale_recip(K), norms(N * ng);
    std::vector<int8_t> left_signs(gs), right_signs(gs);
    std::vector<uint32_t> permutation(gs);

    for (auto& v : codebook) v = static_cast<__fp16>(dist(gen));
    for (size_t i = 0; i < K; i++) {
        float s = 0.5f + std::abs(dist(gen));
        input_scale[i] = static_cast<__fp16>(s);
        input_scale_recip[i] = static_cast<__fp16>(1.f / s);
    }
    for (auto& v : norms) v = static_cast<__fp16>(dist(gen) * 0.1f);
    for (auto& v : left_signs) v = (gen() & 1) ? 1 : -1;
    for (auto& v : right_signs) v = (gen() & 1) ? 1 : -1;
    for (uint32_t i = 0; i < gs; i++) permutation[i] = i;

    // Build CQ matrix and call cactus_quant_matmul directly
    CactusQuantMatrix mat{
        .bits = bits, .K = static_cast<uint32_t>(K), .N = static_cast<uint32_t>(N),
        .group_size = static_cast<uint32_t>(gs), .num_groups = static_cast<uint32_t>(ng),
        .flags = CACTUS_QUANT_FLAG_CODE_ORDERED_INDICES,
        .codebook = codebook.data(),
        .input_scale = input_scale.data(),
        .input_scale_recip = input_scale_recip.data(),
        .norms = norms.data(),
        .packed_indices = packed.data(),
        .left_signs = left_signs.data(),
        .right_signs = right_signs.data(),
        .permutation = permutation.data(),
        .expanded = nullptr,
        .norm_f32 = nullptr,
    };

    std::vector<__fp16> C(M * N, static_cast<__fp16>(0));
    cactus_quant_matmul(&mat, A.data(), static_cast<uint32_t>(M), C.data());

    // Verify results are finite and non-zero
    for (size_t i = 0; i < M * N; i++) {
        if (!std::isfinite(static_cast<float>(C[i]))) return false;
    }
    bool has_nonzero = false;
    for (size_t i = 0; i < M * N; i++) {
        if (std::abs(static_cast<float>(C[i])) > 1e-6f) has_nonzero = true;
    }
    return has_nonzero;
}

bool test_attention_int8_hybrid() {
    const size_t b = 1, s = 1, h = 2, kv = 2, d = 16;
    const size_t cache_len = 4;
    const size_t num_groups = (d + KV_QUANT_GROUP_SIZE - 1) / KV_QUANT_GROUP_SIZE;

    std::vector<__fp16> q(b * s * h * d), k_new(b * s * kv * d), v_new(b * s * kv * d);
    std::vector<int8_t> k_cached(cache_len * kv * d, 10);
    std::vector<int8_t> v_cached(cache_len * kv * d, 5);
    std::vector<float> k_scales(cache_len * kv * num_groups, 0.01f);
    std::vector<float> v_scales(cache_len * kv * num_groups, 0.01f);

    fill_random_fp16(q);
    fill_random_fp16(k_new);
    fill_random_fp16(v_new);

    float scale = 1.0f / std::sqrt(static_cast<float>(d));

    CactusGraph g;
    size_t iq = g.input({b, s, h, d}, Precision::FP16);
    size_t ik = g.input({b, s, kv, d}, Precision::FP16);
    size_t iv = g.input({b, s, kv, d}, Precision::FP16);
    size_t out = g.attention_int8_hybrid(iq, ik, iv, scale, 0,
        k_cached.data(), v_cached.data(),
        k_scales.data(), v_scales.data(),
        cache_len, kv, d);
    g.set_input(iq, q.data(), Precision::FP16);
    g.set_input(ik, k_new.data(), Precision::FP16);
    g.set_input(iv, v_new.data(), Precision::FP16);
    g.execute();

    __fp16* result = static_cast<__fp16*>(g.get_output(out));
    size_t out_size = b * s * h * d;
    for (size_t i = 0; i < out_size; i++) {
        if (!std::isfinite(static_cast<float>(result[i]))) return false;
    }

    bool has_nonzero = false;
    for (size_t i = 0; i < out_size; i++) {
        if (std::abs(static_cast<float>(result[i])) > 1e-6f) has_nonzero = true;
    }
    return has_nonzero;
}

bool test_transpose() {
    TestUtils::FP16TestFixture fixture("Transpose");

    size_t input_a = fixture.create_input({2, 3});
    size_t transpose_result = fixture.graph().transpose(input_a);

    std::vector<__fp16> data_a = {1, 2, 3, 4, 5, 6};
    fixture.set_input_data(input_a, data_a);
    fixture.execute();

    std::vector<__fp16> expected = {1, 4, 2, 5, 3, 6};
    return fixture.verify_output(transpose_result, expected);
}

bool test_reshape() {
    TestUtils::FP16TestFixture fixture("Reshape");

    size_t input_a = fixture.create_input({2, 3});
    size_t reshape_result = fixture.graph().reshape(input_a, {3, 2});

    std::vector<__fp16> data_a = {1, 2, 3, 4, 5, 6};
    fixture.set_input_data(input_a, data_a);
    fixture.execute();

    return fixture.verify_output(reshape_result, data_a);
}

bool test_rms_norm() {
    TestUtils::FP16TestFixture fixture("RMS Norm");

    size_t input_a = fixture.create_input({1, 8});
    size_t weight = fixture.create_input({8});
    size_t norm_result = fixture.graph().rms_norm(input_a, weight);

    std::vector<__fp16> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<__fp16> weight_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

    fixture.set_input_data(input_a, input_data);
    fixture.set_input_data(weight, weight_data);
    fixture.execute();

    float sum_squares = 0.0f;
    for (auto val : input_data) {
        float v = static_cast<float>(val);
        sum_squares += v * v;
    }
    float rms = sqrtf(sum_squares / 8.0f + 1e-5f);
    float inv_rms = 1.0f / rms;

    std::vector<__fp16> expected;
    for (size_t i = 0; i < input_data.size(); i++) {
        expected.push_back(static_cast<__fp16>(static_cast<float>(input_data[i]) * inv_rms * static_cast<float>(weight_data[i])));
    }

    return fixture.verify_output(norm_result, expected, 0.01f);
}

bool test_softmax() {
    TestUtils::FP16TestFixture fixture("Softmax");

    size_t input_a = fixture.create_input({2, 3});
    size_t softmax_result = fixture.graph().softmax(input_a, -1);

    std::vector<__fp16> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    fixture.set_input_data(input_a, input_data);
    fixture.execute();

    std::vector<__fp16> expected = {0.09003f, 0.24473f, 0.66524f, 0.09003f, 0.24473f, 0.66524f};
    return fixture.verify_output(softmax_result, expected, 0.01f);
}

bool test_attention() {
    TestUtils::FP16TestFixture fixture("Attention");

    size_t query = fixture.create_input({1, 2, 1, 4});
    size_t key = fixture.create_input({1, 2, 1, 4});
    size_t value = fixture.create_input({1, 2, 1, 4});

    size_t attention_result = fixture.graph().attention(query, key, value, 0.5f);
    (void)attention_result;

    std::vector<__fp16> q_data = {1, 0, 0, 0, 0, 1, 0, 0};
    std::vector<__fp16> k_data = {1, 0, 0, 0, 0, 1, 0, 0};
    std::vector<__fp16> v_data = {1, 2, 3, 4, 5, 6, 7, 8};

    fixture.set_input_data(query, q_data);
    fixture.set_input_data(key, k_data);
    fixture.set_input_data(value, v_data);
    fixture.execute();

    return true;
}

bool test_reduction_operations() {
    TestUtils::FP16TestFixture fixture("Reduction Operations");

    size_t input_a = fixture.create_input({2, 3});
    size_t sum_all = fixture.graph().sum(input_a, -1);
    size_t sum_axis0 = fixture.graph().sum(input_a, 0);
    size_t sum_axis1 = fixture.graph().sum(input_a, 1);

    std::vector<__fp16> data_a = {1, 2, 3, 4, 5, 6};
    fixture.set_input_data(input_a, data_a);
    fixture.execute();

    std::vector<__fp16> expected_all = {21};
    std::vector<__fp16> expected_axis0 = {5, 7, 9};
    std::vector<__fp16> expected_axis1 = {6, 15};

    return fixture.verify_output(sum_all, expected_all) &&
           fixture.verify_output(sum_axis0, expected_axis0) &&
           fixture.verify_output(sum_axis1, expected_axis1);
}

bool test_fp16_reduction_operations() {
    CactusGraph graph;

    size_t input_a = graph.input({2, 3}, Precision::FP16);
    size_t sum_all = graph.sum(input_a, -1);

    std::vector<__fp16> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    graph.set_input(input_a, input_data.data(), Precision::FP16);
    graph.execute();

    __fp16* output = static_cast<__fp16*>(graph.get_output(sum_all));
    double result = static_cast<double>(output[0]);
    double expected = 21.0;

    bool success = std::abs(result - expected) < 0.1f;

    graph.hard_reset();
    return success;
}

bool test_mean_operations() {
    TestUtils::FP16TestFixture fixture("Mean Operations");

    size_t input_a = fixture.create_input({2, 4});
    size_t mean_all = fixture.graph().mean(input_a, -1);
    size_t mean_axis0 = fixture.graph().mean(input_a, 0);
    size_t mean_axis1 = fixture.graph().mean(input_a, 1);

    std::vector<__fp16> data_a = {2, 4, 6, 8, 10, 12, 14, 16};
    fixture.set_input_data(input_a, data_a);
    fixture.execute();

    std::vector<__fp16> expected_all = {9};
    std::vector<__fp16> expected_axis0 = {6, 8, 10, 12};
    std::vector<__fp16> expected_axis1 = {5, 13};

    return fixture.verify_output(mean_all, expected_all) &&
           fixture.verify_output(mean_axis0, expected_axis0) &&
           fixture.verify_output(mean_axis1, expected_axis1);
}

bool test_variance_operations() {
    TestUtils::FP16TestFixture fixture("Variance Operations");

    size_t input_a = fixture.create_input({1, 4});
    size_t var_axis1 = fixture.graph().variance(input_a, 1);

    std::vector<__fp16> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
    fixture.set_input_data(input_a, input_data);
    fixture.execute();

    std::vector<__fp16> expected = {1.25f};
    return fixture.verify_output(var_axis1, expected, 0.01f);
}

bool test_min_max_operations() {
    TestUtils::FP16TestFixture fixture("Min/Max Operations");

    size_t input_a = fixture.create_input({2, 3});
    size_t min_axis0 = fixture.graph().min(input_a, 0);
    size_t max_axis0 = fixture.graph().max(input_a, 0);
    size_t min_axis1 = fixture.graph().min(input_a, 1);
    size_t max_axis1 = fixture.graph().max(input_a, 1);

    std::vector<__fp16> data_a = {6, 2, 8, 1, 5, 3};
    fixture.set_input_data(input_a, data_a);
    fixture.execute();

    std::vector<__fp16> expected_min_axis0 = {1, 2, 3};
    std::vector<__fp16> expected_max_axis0 = {6, 5, 8};
    std::vector<__fp16> expected_min_axis1 = {2, 1};
    std::vector<__fp16> expected_max_axis1 = {8, 5};

    return fixture.verify_output(min_axis0, expected_min_axis0) &&
           fixture.verify_output(max_axis0, expected_max_axis0) &&
           fixture.verify_output(min_axis1, expected_min_axis1) &&
           fixture.verify_output(max_axis1, expected_max_axis1);
}

bool test_stft() {
    const size_t N = 2, C_in = 1, L = 8, K = 4, stride = 2, num_fft_bins = 2;
    const size_t C_out = 2 * num_fft_bins;
    const size_t out_len = (L - K) / stride + 1;

    std::vector<__fp16> weight_data = {
        (__fp16) 1, (__fp16) 1, (__fp16) 1, (__fp16) 1,
        (__fp16) 1, (__fp16) 0, (__fp16)-1, (__fp16) 0,
        (__fp16) 0, (__fp16) 0, (__fp16) 0, (__fp16) 0,
        (__fp16) 0, (__fp16)-1, (__fp16) 0, (__fp16) 1,
    };
    std::vector<__fp16> input_data = {
        (__fp16)1, (__fp16)2, (__fp16)3, (__fp16)4, (__fp16)5, (__fp16)6, (__fp16)7, (__fp16)8,
        (__fp16)0, (__fp16)1, (__fp16)0, (__fp16)-1, (__fp16)0, (__fp16)1, (__fp16)0, (__fp16)-1,
    };

    TestUtils::FP16TestFixture fx;
    size_t inp = fx.create_input({N, C_in, L});
    size_t wt  = fx.create_input({C_out, C_in, K});
    size_t out = fx.graph().stft(inp, wt, stride, num_fft_bins);

    if (fx.graph().get_output_buffer(out).shape != std::vector<size_t>{N, C_out, out_len}) return false;

    fx.set_input_data(inp, input_data);
    fx.set_input_data(wt, weight_data);
    fx.execute();

    const __fp16* cplx = fx.get_output(out);
    const size_t out_bs = C_out * out_len;
    const float tol = 0.1f;

    for (size_t t = 0; t < out_len; ++t) {
        if (std::abs((float)cplx[1 * out_len + t] - (-2.0f)) > tol) return false;
        if (std::abs((float)cplx[(1 + num_fft_bins) * out_len + t] - 2.0f) > tol) return false;
    }

    const float batch1_bin1_imag[3] = {-2.0f, 2.0f, -2.0f};
    for (size_t t = 0; t < out_len; ++t) {
        if (std::abs((float)cplx[out_bs + 1 * out_len + t] - 0.0f) > tol) return false;
        if (std::abs((float)cplx[out_bs + (1 + num_fft_bins) * out_len + t] - batch1_bin1_imag[t]) > tol) return false;
    }

    return true;
}

template<typename T>
static bool run_layernorm_case(
    size_t batch, size_t feat, bool with_bias, float epsilon,
    float weight_scale, float bias_val)
{
    const size_t total = batch * feat;

    std::vector<float> input_f(total), weight_f(feat), bias_f(feat);
    for (size_t b = 0; b < batch; ++b)
        for (size_t j = 0; j < feat; ++j)
            input_f[b * feat + j] = static_cast<float>(j + 1);
    for (size_t j = 0; j < feat; ++j) {
        weight_f[j] = weight_scale;
        bias_f[j]   = bias_val;
    }

    std::vector<T> inp_data(total), w_data(feat), b_data(feat);
    for (size_t i = 0; i < total; ++i) inp_data[i] = static_cast<T>(input_f[i]);
    for (size_t j = 0; j < feat;  ++j) {
        w_data[j] = static_cast<T>(weight_f[j]);
        b_data[j] = static_cast<T>(bias_f[j]);
    }

    TestUtils::TestFixture<T> fx;
    size_t inp_id = fx.create_input({batch, feat});
    size_t w_id   = fx.create_input({feat});
    fx.set_input_data(inp_id, inp_data);
    fx.set_input_data(w_id,   w_data);

    size_t out_id;
    if (with_bias) {
        size_t b_id = fx.create_input({feat});
        fx.set_input_data(b_id, b_data);
        out_id = fx.graph().layernorm(inp_id, w_id, b_id, epsilon);
    } else {
        out_id = fx.graph().layernorm(inp_id, w_id, epsilon);
    }

    fx.execute();

    std::vector<T> expected(total);
    for (size_t b = 0; b < batch; ++b) {
        float mean = 0.0f;
        for (size_t j = 0; j < feat; ++j) mean += input_f[b * feat + j];
        mean /= static_cast<float>(feat);
        float var = 0.0f;
        for (size_t j = 0; j < feat; ++j) {
            float d = input_f[b * feat + j] - mean;
            var += d * d;
        }
        var /= static_cast<float>(feat);
        float inv_std = 1.0f / std::sqrt(var + epsilon);
        for (size_t j = 0; j < feat; ++j) {
            float val = (input_f[b * feat + j] - mean) * inv_std * weight_f[j];
            if (with_bias) val += bias_f[j];
            expected[b * feat + j] = static_cast<T>(val);
        }
    }

    return fx.verify_output(out_id, expected, TestUtils::default_tolerance<T>());
}

bool test_layernorm() {
    struct Case { size_t batch, feat; bool fp32, with_bias; float epsilon, weight_scale, bias_val; };
    const std::vector<Case> cases = {
        {1,  1,  false, false, 1e-5f, 1.0f, 0.0f},
        {1,  7,  false, false, 1e-5f, 1.0f, 0.0f},
        {1,  8,  false, false, 1e-5f, 1.0f, 0.0f},
        {4,  8,  false, false, 1e-5f, 1.0f, 0.0f},
        {4,  8,  false, true,  1e-5f, 1.0f, 0.0f},
        {4,  8,  true,  false, 1e-5f, 1.0f, 0.0f},
        {4,  8,  true,  true,  1e-5f, 1.0f, 0.0f},
        {1,  8,  false, false, 1.0f,  1.0f, 0.0f},
        {2, 16,  false, true,  1e-5f, 0.5f, 0.3f},
    };

    for (const auto& c : cases) {
        bool ok = c.fp32
            ? run_layernorm_case<float>(c.batch, c.feat, c.with_bias, c.epsilon, c.weight_scale, c.bias_val)
            : run_layernorm_case<__fp16>(c.batch, c.feat, c.with_bias, c.epsilon, c.weight_scale, c.bias_val);
        if (!ok) return false;
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
        const size_t M = 1024, K = 1024, N = 1024;
        std::vector<__fp16> a(M * K), b(N * K);
        TestUtils::fill_random_fp16(a);
        TestUtils::fill_random_fp16(b);
        CactusGraph g;
        size_t ia = g.input({M, K}, Precision::FP16);
        size_t ib = g.input({N, K}, Precision::FP16);
        g.matmul(ia, ib, true);
        g.set_input(ia, a.data(), Precision::FP16);
        g.set_input(ib, b.data(), Precision::FP16);
        bench("matmul_f16 1024^3", []{}, [&]{ g.execute(); });
    }
    {
        // CQ4 matmul benchmark via cactus_quant_matmul (graph-level equivalent)
        const size_t M = 1024, K = 1024, N = 1024, gs = 128;
        const size_t ng = K / gs;
        const uint32_t bits = 4, cb_size = 16;
        std::mt19937 bgen(77);
        std::uniform_real_distribution<float> bdist(-1.f, 1.f);

        std::vector<__fp16> A(M * K), codebook(cb_size), input_sc(K), input_sc_r(K), norms_v(N * ng);
        std::vector<int8_t> lsigns(gs), rsigns(gs);
        std::vector<uint32_t> perm(gs);
        for (auto& v : A) v = static_cast<__fp16>(bdist(bgen));
        for (auto& v : codebook) v = static_cast<__fp16>(bdist(bgen));
        for (size_t i = 0; i < K; i++) { float s = 0.5f+std::abs(bdist(bgen)); input_sc[i]=(__fp16)s; input_sc_r[i]=(__fp16)(1.f/s); }
        for (auto& v : norms_v) v = static_cast<__fp16>(bdist(bgen) * 0.1f);
        for (auto& v : lsigns) v = (bgen()&1)?1:-1;
        for (auto& v : rsigns) v = (bgen()&1)?1:-1;
        for (uint32_t i = 0; i < gs; i++) perm[i] = i;
        uint32_t pgb = cactus_quant_packed_group_bytes(bits, gs);
        std::vector<uint8_t> packed(N * ng * pgb);
        for (auto& v : packed) v = static_cast<uint8_t>(bgen() & 0xFF);

        CactusQuantMatrix mat{bits, (uint32_t)K, (uint32_t)N, (uint32_t)gs, (uint32_t)ng,
            CACTUS_QUANT_FLAG_CODE_ORDERED_INDICES,
            codebook.data(), input_sc.data(), input_sc_r.data(), norms_v.data(),
            packed.data(), lsigns.data(), rsigns.data(), perm.data(), nullptr, nullptr};

        std::vector<__fp16> C(M * N);
        bench("matmul_cq4 1024^3", []{}, [&]{ cactus_quant_matmul(&mat, A.data(), M, C.data()); });
    }
    {
        const size_t b = 1, s = 256, h = 16, kv = 8, d = 128;
        std::vector<__fp16> q(b*s*h*d), k(b*s*kv*d), v(b*s*kv*d);
        TestUtils::fill_random_fp16(q);
        TestUtils::fill_random_fp16(k);
        TestUtils::fill_random_fp16(v);
        float scale = 1.0f / std::sqrt(static_cast<float>(d));
        CactusGraph g;
        size_t iq = g.input({b, s, h, d}, Precision::FP16);
        size_t ik = g.input({b, s, kv, d}, Precision::FP16);
        size_t iv = g.input({b, s, kv, d}, Precision::FP16);
        g.attention(iq, ik, iv, scale);
        g.set_input(iq, q.data(), Precision::FP16);
        g.set_input(ik, k.data(), Precision::FP16);
        g.set_input(iv, v.data(), Precision::FP16);
        bench("attention_f16 seq=256", []{}, [&]{ g.execute(); });
    }
    {
        const size_t b = 1, s = 1, h = 16, kv = 8, d = 128;
        const size_t cache_len = 512;
        std::vector<__fp16> q(b*s*h*d), k(b*s*kv*d), v(b*s*kv*d);
        std::vector<int8_t> ck(cache_len*kv*d, 1), cv(cache_len*kv*d, 1);
        size_t ng = (d + KV_QUANT_GROUP_SIZE - 1) / KV_QUANT_GROUP_SIZE;
        std::vector<float> ks(cache_len*kv*ng, 0.01f), vs(cache_len*kv*ng, 0.01f);
        TestUtils::fill_random_fp16(q);
        TestUtils::fill_random_fp16(k);
        TestUtils::fill_random_fp16(v);
        float scale = 1.0f / std::sqrt(static_cast<float>(d));
        CactusGraph g;
        size_t iq = g.input({b, s, h, d}, Precision::FP16);
        size_t ik = g.input({b, s, kv, d}, Precision::FP16);
        size_t iv = g.input({b, s, kv, d}, Precision::FP16);
        g.attention_int8_hybrid(iq, ik, iv, scale, 0,
            ck.data(), cv.data(), ks.data(), vs.data(), cache_len, kv, d);
        g.set_input(iq, q.data(), Precision::FP16);
        g.set_input(ik, k.data(), Precision::FP16);
        g.set_input(iv, v.data(), Precision::FP16);
        bench("attention_int8 cache=512", []{}, [&]{ g.execute(); });
    }
    {
        const size_t batch = 1024, dim = 1024;
        std::vector<__fp16> in(batch * dim), w(dim);
        TestUtils::fill_random_fp16(in);
        for (size_t i = 0; i < dim; i++) w[i] = static_cast<__fp16>(1.0f);
        CactusGraph g;
        size_t ii = g.input({batch, dim}, Precision::FP16);
        size_t iw = g.input({dim}, Precision::FP16);
        g.rms_norm(ii, iw, 1e-6f);
        g.set_input(ii, in.data(), Precision::FP16);
        g.set_input(iw, w.data(), Precision::FP16);
        bench("rms_norm 1024x1024", []{}, [&]{ g.execute(); });
    }
    {
        const size_t rows = 1024, cols = 1024;
        std::vector<__fp16> in(rows * cols);
        TestUtils::fill_random_fp16(in);
        CactusGraph g;
        size_t ii = g.input({rows, cols}, Precision::FP16);
        g.softmax(ii);
        g.set_input(ii, in.data(), Precision::FP16);
        bench("softmax 1024x1024", []{}, [&]{ g.execute(); });
    }
    return true;
}

int main() {
    TestUtils::TestRunner runner("Neural Network Ops Tests");

    runner.run_test("Matrix Multiplication", test_matrix_multiplication());
    runner.run_test("MatMul CQ", test_matmul_cq());
    runner.run_test("Transpose", test_transpose());
    runner.run_test("Reshape", test_reshape());
    runner.run_test("RMS Norm", test_rms_norm());
    runner.run_test("Softmax", test_softmax());
    runner.run_test("Attention", test_attention());
    runner.run_test("Attention INT8 Hybrid", test_attention_int8_hybrid());
    runner.run_test("Reduction Operations", test_reduction_operations());
    runner.run_test("FP16 Reduction Operations", test_fp16_reduction_operations());
    runner.run_test("Mean Operations", test_mean_operations());
    runner.run_test("Variance Operations", test_variance_operations());
    runner.run_test("Min/Max Operations", test_min_max_operations());
    runner.run_test("STFT Complex", test_stft());
    runner.run_test("LayerNorm", test_layernorm());
    runner.print_benchmarks_header();
    runner.run_bench("benchmarks", run_benchmarks());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
