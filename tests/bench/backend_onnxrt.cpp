#include "bench_driver.h"

#ifdef WITH_ONNXRT

#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <thread>
#include <unordered_map>

namespace {

struct PBuf {
    std::vector<uint8_t> d;
    void varint(uint64_t v) {
        while (v > 0x7F) { d.push_back(static_cast<uint8_t>((v & 0x7F) | 0x80)); v >>= 7; }
        d.push_back(static_cast<uint8_t>(v));
    }
    void fld_vi(int f, uint64_t v) {
        varint(static_cast<uint64_t>(f) << 3); varint(v);
    }
    void fld_ld(int f, const PBuf& sub) {
        varint(static_cast<uint64_t>(f) << 3 | 2);
        varint(sub.d.size());
        d.insert(d.end(), sub.d.begin(), sub.d.end());
    }
    void fld_str(int f, const char* s) {
        size_t n = std::strlen(s);
        varint(static_cast<uint64_t>(f) << 3 | 2);
        varint(n);
        d.insert(d.end(), s, s + n);
    }
};

static PBuf make_dim_param(const char* name) { PBuf d; d.fld_str(2, name); return d; }
static PBuf make_dim_value(size_t v) { PBuf d; d.fld_vi(1, v); return d; }

template<typename... Dims>
static PBuf make_value_info(const char* name, int elem_type, const Dims&... dims) {
    PBuf shape; (shape.fld_ld(1, dims), ...);
    PBuf tensor; tensor.fld_vi(1, elem_type); tensor.fld_ld(2, shape);
    PBuf type; type.fld_ld(1, tensor);
    PBuf vi; vi.fld_str(1, name); vi.fld_ld(2, type);
    return vi;
}

static PBuf make_attr_int(const char* name, int64_t val) {
    PBuf a;
    a.fld_str(1, name);
    a.fld_vi(3, static_cast<uint64_t>(val));
    a.fld_vi(20, 2);
    return a;
}

static Ort::Env& get_env() {
    static Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "bench");
    return env;
}

static Ort::MemoryInfo& get_cpu_mem() {
    static Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    return mem;
}

static std::vector<uint8_t> build_model(size_t K, size_t N) {
    const size_t n_blocks = K / bench::kGroupSize;
    const size_t b_last_dim = bench::kGroupSize / 2;
    const size_t zp_last_dim = (n_blocks + 1) / 2;

    PBuf node;
    node.fld_str(1, "A");
    node.fld_str(1, "B");
    node.fld_str(1, "scales");
    node.fld_str(1, "zero_points");
    node.fld_str(2, "Y");
    node.fld_str(4, "MatMulNBits");
    node.fld_str(7, "com.microsoft");
    node.fld_ld(5, make_attr_int("K", static_cast<int64_t>(K)));
    node.fld_ld(5, make_attr_int("N", static_cast<int64_t>(N)));
    node.fld_ld(5, make_attr_int("bits", 4));
    node.fld_ld(5, make_attr_int("block_size", static_cast<int64_t>(bench::kGroupSize)));
    node.fld_ld(5, make_attr_int("accuracy_level", 4));

    auto a_vi  = make_value_info("A", 1, make_dim_param("M"), make_dim_value(K));
    auto b_vi  = make_value_info("B", 2, make_dim_value(N), make_dim_value(n_blocks),
                                 make_dim_value(b_last_dim));
    auto s_vi  = make_value_info("scales", 1, make_dim_value(N), make_dim_value(n_blocks));
    auto zp_vi = make_value_info("zero_points", 2,
                                 make_dim_value(N), make_dim_value(zp_last_dim));
    auto y_vi  = make_value_info("Y", 1, make_dim_param("M"), make_dim_value(N));

    PBuf graph;
    graph.fld_ld(1, node);
    graph.fld_str(2, "bench_q4");
    graph.fld_ld(11, a_vi);
    graph.fld_ld(11, b_vi);
    graph.fld_ld(11, s_vi);
    graph.fld_ld(11, zp_vi);
    graph.fld_ld(12, y_vi);

    PBuf opset1; opset1.fld_str(1, ""); opset1.fld_vi(2, 13);
    PBuf opset2; opset2.fld_str(1, "com.microsoft"); opset2.fld_vi(2, 1);

    PBuf model;
    model.fld_vi(1, 7);
    model.fld_ld(8, opset1);
    model.fld_ld(8, opset2);
    model.fld_ld(7, graph);
    return model.d;
}

static constexpr int kGemvThreads = 2;
static constexpr int kGemmThreads = 3;

struct OrtWeights {
    size_t K = 0, N = 0;
    std::vector<uint8_t> B_packed;
    std::vector<float> scales;
    std::vector<uint8_t> zero_points;
};

struct OrtActivations {
    std::vector<float> fp32;
};

// Cache ORT sessions by (M, K, N). Session creation costs ~tens of ms each
// (model parse + graph optimization); without caching, NM matrices ⇒ NM
// sessions per config — quickly dominates wall time at small dims.
struct SessionKey { size_t M, K, N; };
struct SessionKeyHash {
    size_t operator()(const SessionKey& k) const noexcept {
        // Avoid XOR-with-shift's high collision rate on power-of-two M, K, N.
        uint64_t packed = (static_cast<uint64_t>(k.M) * 0x9e3779b97f4a7c15ull)
                        ^ (static_cast<uint64_t>(k.K) * 0xbf58476d1ce4e5b9ull)
                        ^ (static_cast<uint64_t>(k.N) * 0x94d049bb133111ebull);
        return std::hash<uint64_t>{}(packed);
    }
};
struct SessionKeyEq {
    bool operator()(const SessionKey& a, const SessionKey& b) const noexcept {
        return a.M == b.M && a.K == b.K && a.N == b.N;
    }
};

// Function-local static so its destruction is sequenced AFTER any
// function-local Ort::Env (constructed in get_env()). Reverse order of
// construction → reverse order of destruction; Env outlives sessions.
static auto& matmul_sessions() {
    static std::unordered_map<SessionKey, std::shared_ptr<Ort::Session>,
                               SessionKeyHash, SessionKeyEq> m;
    return m;
}

static std::shared_ptr<Ort::Session> get_matmul_session(size_t M, size_t K, size_t N) {
    auto& cache = matmul_sessions();
    SessionKey key{M, K, N};
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;

    try {
        auto bytes = build_model(K, N);
        int threads = bench::get_effective_threads((M == 1) ? kGemvThreads : kGemmThreads);
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(threads);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        auto session = std::make_shared<Ort::Session>(get_env(), bytes.data(), bytes.size(), opts);
        cache.emplace(key, session);
        return session;
    } catch (const Ort::Exception& e) {
        fprintf(stderr, "[onnxrt matmul] session creation failed (M=%zu K=%zu N=%zu): %s\n",
                M, K, N, e.what());
        return nullptr;
    }
}

void* prepare_act(const float* fp32, size_t M, size_t K, void* /*raw_weights*/) {
    auto* a = new OrtActivations();
    a->fp32.assign(fp32, fp32 + M * K);
    return a;
}

void run_kernel(size_t M, size_t K, size_t N,
                void* weights, void* activations,
                const int8_t*, const float*,
                float* output, float*) {
    auto* w = static_cast<OrtWeights*>(weights);
    auto* a = static_cast<OrtActivations*>(activations);
    if (!w || !a) return;

    auto session = get_matmul_session(M, K, N);
    if (!session) return;
    auto& mem = get_cpu_mem();
    const size_t n_blocks = K / bench::kGroupSize;
    const size_t b_last_dim = bench::kGroupSize / 2;
    const size_t b_total = N * n_blocks * b_last_dim;
    const size_t zp_cols = (n_blocks + 1) / 2;

    int64_t a_shape[]  = {(int64_t)M, (int64_t)K};
    int64_t b_shape[]  = {(int64_t)N, (int64_t)n_blocks, (int64_t)b_last_dim};
    int64_t s_shape[]  = {(int64_t)N, (int64_t)n_blocks};
    int64_t zp_shape[] = {(int64_t)N, (int64_t)zp_cols};

    Ort::Value inputs[4] = {
        Ort::Value::CreateTensor<float>(mem, a->fp32.data(), M * K, a_shape, 2),
        Ort::Value::CreateTensor<uint8_t>(mem, w->B_packed.data(), b_total, b_shape, 3),
        Ort::Value::CreateTensor<float>(mem, w->scales.data(), N * n_blocks, s_shape, 2),
        Ort::Value::CreateTensor<uint8_t>(mem, w->zero_points.data(), N * zp_cols, zp_shape, 2),
    };

    static const char* in_names[]  = {"A", "B", "scales", "zero_points"};
    static const char* out_names[] = {"Y"};

    Ort::RunOptions run_opts;
    auto out = session->Run(run_opts, in_names, inputs, 4, out_names, 1);
    if (output) {
        const float* y = out[0].GetTensorData<float>();
        std::memcpy(output, y, M * w->N * sizeof(float));
    }
}

void cleanup(void* weights, void* activations) {
    delete static_cast<OrtWeights*>(weights);
    if (activations) delete static_cast<OrtActivations*>(activations);
}

static void pack_q4(OrtWeights* w, const float* fp32) {
    size_t N = w->N, K = w->K;
    const size_t n_blocks = K / bench::kGroupSize;
    const size_t packed_block = bench::kGroupSize / 2;
    w->B_packed.assign(N * n_blocks * packed_block, 0);
    w->scales.resize(N * n_blocks);
    w->zero_points.resize(N * ((n_blocks + 1) / 2), 0x88);

    for (size_t n = 0; n < N; ++n) {
        for (size_t g = 0; g < n_blocks; ++g) {
            const size_t src_base = n * K + g * bench::kGroupSize;
            float max_abs = 0.0f;
            for (size_t k = 0; k < bench::kGroupSize; ++k)
                max_abs = std::max(max_abs, std::abs(fp32[src_base + k]));
            const float scale = std::max(max_abs / 7.0f, 1e-10f);
            w->scales[n * n_blocks + g] = scale;
            for (size_t k = 0; k < bench::kGroupSize; k += 2) {
                int q0 = static_cast<int>(std::round(fp32[src_base + k] / scale));
                int q1 = static_cast<int>(std::round(fp32[src_base + k + 1] / scale));
                q0 = std::max(-8, std::min(7, q0));
                q1 = std::max(-8, std::min(7, q1));
                w->B_packed[(n * n_blocks + g) * packed_block + k / 2] =
                    static_cast<uint8_t>((q0 + 8) | ((q1 + 8) << 4));
            }
        }
    }
}

void* q4_prepare(const float* fp32, size_t N, size_t K) {
    auto* w = new OrtWeights();
    w->K = K; w->N = N;
    pack_q4(w, fp32);
    return w;
}

// GQA is the only ORT contrib attention op with FP16 support on CPU EP
// (MultiHeadAttention and DecoderMaskedMultiHeadAttention CPU registrations
// are FP32-only). We use GQA with num_heads==kv_num_heads. Decode runs FP16
// KV, not INT8 — ORT has no quantized-KV attention op.

namespace gqa {

constexpr int kElemFloat16 = 10;
constexpr int kElemInt32   = 6;
constexpr int kAttrTypeInt   = 2;
constexpr int kAttrTypeFloat = 1;

static PBuf attr_int(const char* name, int64_t val) {
    PBuf a;
    a.fld_str(1, name);
    a.fld_vi(3, static_cast<uint64_t>(val));
    a.fld_vi(20, kAttrTypeInt);
    return a;
}

static PBuf attr_float(const char* name, float val) {
    PBuf a;
    a.fld_str(1, name);
    uint32_t bits;
    std::memcpy(&bits, &val, 4);
    a.varint(static_cast<uint64_t>(2) << 3 | 5);
    a.d.push_back(static_cast<uint8_t>(bits & 0xFF));
    a.d.push_back(static_cast<uint8_t>((bits >> 8) & 0xFF));
    a.d.push_back(static_cast<uint8_t>((bits >> 16) & 0xFF));
    a.d.push_back(static_cast<uint8_t>((bits >> 24) & 0xFF));
    a.fld_vi(20, kAttrTypeFloat);
    return a;
}

static PBuf make_scalar_value_info(const char* name, int elem_type) {
    PBuf shape;
    PBuf tensor; tensor.fld_vi(1, elem_type); tensor.fld_ld(2, shape);
    PBuf type;   type.fld_ld(1, tensor);
    PBuf vi;     vi.fld_str(1, name); vi.fld_ld(2, type);
    return vi;
}

static std::vector<uint8_t> build_gqa_model(size_t q_seq, size_t past_seq, size_t total_seq,
                                             size_t num_heads, size_t head_dim) {
    const size_t hidden = num_heads * head_dim;
    const size_t new_kv_seq = total_seq - past_seq;
    const bool with_past = (past_seq > 0);
    // The CPU compute path expects K/V as 4D BNSH (gqa_attention_base.h:341).
    // For prefill (no past), we omit past_key/past_value entirely — empty
    // strings in the node's input list and no matching value_info. Passing
    // shape-[1,h,0,d] tensors instead silently corrupts output on the prompt
    // path (nrmse=0.87).
    PBuf node;
    node.fld_str(1, "Q");
    node.fld_str(1, "K");
    node.fld_str(1, "V");
    node.fld_str(1, with_past ? "past_key"   : "");
    node.fld_str(1, with_past ? "past_value" : "");
    node.fld_str(1, "seqlens_k");
    node.fld_str(1, "total_seq_len");
    node.fld_str(2, "Y");
    // GQA schema requires 3+ outputs and disallows empty-string outputs, so
    // present_key/present_value are always declared (even for prefill).
    node.fld_str(2, "present_key");
    node.fld_str(2, "present_value");
    node.fld_str(4, "GroupQueryAttention");
    node.fld_str(7, "com.microsoft");
    node.fld_ld(5, attr_int  ("num_heads",         static_cast<int64_t>(num_heads)));
    node.fld_ld(5, attr_int  ("kv_num_heads",      static_cast<int64_t>(num_heads)));
    node.fld_ld(5, attr_int  ("do_rotary",         0));
    node.fld_ld(5, attr_int  ("rotary_interleaved",0));
    node.fld_ld(5, attr_int  ("local_window_size", -1));
    node.fld_ld(5, attr_float("scale",             1.0f / std::sqrt(static_cast<float>(head_dim))));
    node.fld_ld(5, attr_float("softcap",           0.0f));

    auto q_vi  = make_value_info("Q", kElemFloat16,
                                  make_dim_value(1), make_dim_value(q_seq), make_dim_value(hidden));
    auto k_vi  = make_value_info("K", kElemFloat16,
                                  make_dim_value(1), make_dim_value(new_kv_seq), make_dim_value(hidden));
    auto v_vi  = make_value_info("V", kElemFloat16,
                                  make_dim_value(1), make_dim_value(new_kv_seq), make_dim_value(hidden));
    auto sk_vi = make_value_info("seqlens_k", kElemInt32, make_dim_value(1));
    auto ts_vi = make_scalar_value_info("total_seq_len", kElemInt32);
    auto y_vi  = make_value_info("Y", kElemFloat16,
                                  make_dim_value(1), make_dim_value(q_seq), make_dim_value(hidden));

    PBuf graph;
    graph.fld_ld(1, node);
    graph.fld_str(2, "bench_gqa");
    graph.fld_ld(11, q_vi);
    graph.fld_ld(11, k_vi);
    graph.fld_ld(11, v_vi);
    if (with_past) {
        auto pk_vi = make_value_info("past_key", kElemFloat16,
                                      make_dim_value(1), make_dim_value(num_heads),
                                      make_dim_value(past_seq), make_dim_value(head_dim));
        auto pv_vi = make_value_info("past_value", kElemFloat16,
                                      make_dim_value(1), make_dim_value(num_heads),
                                      make_dim_value(past_seq), make_dim_value(head_dim));
        graph.fld_ld(11, pk_vi);
        graph.fld_ld(11, pv_vi);
    }
    graph.fld_ld(11, sk_vi);
    graph.fld_ld(11, ts_vi);
    graph.fld_ld(12, y_vi);
    {
        auto presk_vi = make_value_info("present_key", kElemFloat16,
                                         make_dim_value(1), make_dim_value(num_heads),
                                         make_dim_value(total_seq), make_dim_value(head_dim));
        auto presv_vi = make_value_info("present_value", kElemFloat16,
                                         make_dim_value(1), make_dim_value(num_heads),
                                         make_dim_value(total_seq), make_dim_value(head_dim));
        graph.fld_ld(12, presk_vi);
        graph.fld_ld(12, presv_vi);
    }

    PBuf opset1; opset1.fld_str(1, ""); opset1.fld_vi(2, 17);
    PBuf opset2; opset2.fld_str(1, "com.microsoft"); opset2.fld_vi(2, 1);

    PBuf model;
    model.fld_vi(1, 7);
    model.fld_ld(8, opset1);
    model.fld_ld(8, opset2);
    model.fld_ld(7, graph);
    return model.d;
}

static uint16_t fp32_to_fp16_bits(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t  exp  = static_cast<int32_t>((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (x >> 13) & 0x3FF;
    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        mant = ((x & 0x7FFFFF) | 0x800000) >> (1 - exp + 13);
        return static_cast<uint16_t>(sign | (mant & 0x3FF));
    }
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | mant);
}
static float fp16_bits_to_fp32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) { out = sign; }
        else {
            while ((mant & 0x400) == 0) { mant <<= 1; exp--; }
            exp++; mant &= 0x3FF;
            out = sign | ((exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7F800000 | (mant << 13);
    } else {
        out = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float f; std::memcpy(&f, &out, 4); return f;
}

// Driver [head, seq, head_dim] → ORT BSH [1, seq, head*head_dim] FP16.
static void pack_bsh(const float* src, uint16_t* dst,
                      size_t num_heads, size_t seq, size_t head_dim) {
    for (size_t s = 0; s < seq; ++s)
        for (size_t h = 0; h < num_heads; ++h) {
            const float* in = src + h * seq * head_dim + s * head_dim;
            uint16_t* out = dst + s * num_heads * head_dim + h * head_dim;
            for (size_t d = 0; d < head_dim; ++d)
                out[d] = fp32_to_fp16_bits(in[d]);
        }
}

// Driver [head, seq, head_dim] and ORT BNSH [1, head, seq, head_dim] share
// the same physical buffer order, so this is just FP32 → FP16.
static void pack_bnsh(const float* src, uint16_t* dst,
                       size_t num_heads, size_t seq, size_t head_dim) {
    const size_t total = num_heads * seq * head_dim;
    for (size_t i = 0; i < total; ++i)
        dst[i] = fp32_to_fp16_bits(src[i]);
}

static void unpack_bsh(const uint16_t* src, float* dst,
                        size_t num_heads, size_t seq, size_t head_dim) {
    for (size_t h = 0; h < num_heads; ++h)
        for (size_t s = 0; s < seq; ++s) {
            const uint16_t* in = src + s * num_heads * head_dim + h * head_dim;
            float* out = dst + h * seq * head_dim + s * head_dim;
            for (size_t d = 0; d < head_dim; ++d)
                out[d] = fp16_bits_to_fp32(in[d]);
        }
}

struct GqaState {
    bench::AttnDims dims;
    size_t q_seq;
    size_t past_seq;
    size_t new_kv_seq;
    size_t total_seq;
    size_t hidden;

    std::vector<uint16_t> q_fp16;
    std::vector<uint16_t> k_new_fp16;
    std::vector<uint16_t> v_new_fp16;
    std::vector<uint16_t> past_key_fp16;
    std::vector<uint16_t> past_value_fp16;
    std::vector<uint16_t> present_key_buf;
    std::vector<uint16_t> present_value_buf;
    int32_t seqlens_k_val = 0;
    int32_t total_seq_val = 0;

    std::unique_ptr<Ort::Session> session;
    Ort::RunOptions run_opts;
};

static void* gqa_prepare_impl(const bench::AttnDims& dims, size_t seq_len, size_t cache_len,
                               const float* fp32_q, const float* fp32_k, const float* fp32_v,
                               bench::AttnMode mode) {
    if (dims.num_q_heads != dims.num_kv_heads) {
        fprintf(stderr, "[onnxrt gqa] this bench wires GQA with h_q==h_kv only\n");
        return nullptr;
    }

    auto* s = new GqaState();
    s->dims = dims;
    s->hidden = dims.num_q_heads * dims.head_dim;

    // GQA's `seqlens_k` is "index of the last valid token in the K cache
    // AFTER processing", i.e., total_sequence_length - 1. The past_seq
    // interpretation happens to match for decode but is wrong for prefill.
    if (mode == bench::AttnMode::PREFILL) {
        s->q_seq      = seq_len;
        s->past_seq   = 0;
        s->new_kv_seq = seq_len;
        s->total_seq  = seq_len;
        s->seqlens_k_val = static_cast<int32_t>(seq_len - 1);
        s->total_seq_val = static_cast<int32_t>(seq_len);
    } else {
        s->q_seq      = 1;
        s->past_seq   = cache_len;
        s->new_kv_seq = 1;
        s->total_seq  = cache_len + 1;
        s->seqlens_k_val = static_cast<int32_t>(cache_len);
        s->total_seq_val = static_cast<int32_t>(cache_len + 1);
    }

    s->q_fp16.resize(s->q_seq * s->hidden);
    pack_bsh(fp32_q, s->q_fp16.data(), dims.num_q_heads, s->q_seq, dims.head_dim);

    s->past_key_fp16.resize(dims.num_kv_heads * s->past_seq * dims.head_dim);
    s->past_value_fp16.resize(dims.num_kv_heads * s->past_seq * dims.head_dim);
    s->k_new_fp16.resize(s->new_kv_seq * s->hidden);
    s->v_new_fp16.resize(s->new_kv_seq * s->hidden);

    if (mode == bench::AttnMode::PREFILL) {
        pack_bsh(fp32_k, s->k_new_fp16.data(), dims.num_kv_heads, s->new_kv_seq, dims.head_dim);
        pack_bsh(fp32_v, s->v_new_fp16.data(), dims.num_kv_heads, s->new_kv_seq, dims.head_dim);
    } else {
        auto split_pack = [&](const float* src, uint16_t* past_dst, uint16_t* new_dst) {
            std::vector<float> past_only(dims.num_kv_heads * s->past_seq * dims.head_dim);
            std::vector<float> new_only(dims.num_kv_heads * s->new_kv_seq * dims.head_dim);
            for (size_t h = 0; h < dims.num_kv_heads; ++h) {
                const float* head_base = src + h * s->total_seq * dims.head_dim;
                std::memcpy(past_only.data() + h * s->past_seq * dims.head_dim,
                            head_base, s->past_seq * dims.head_dim * sizeof(float));
                std::memcpy(new_only.data() + h * s->new_kv_seq * dims.head_dim,
                            head_base + s->past_seq * dims.head_dim,
                            s->new_kv_seq * dims.head_dim * sizeof(float));
            }
            pack_bnsh(past_only.data(), past_dst, dims.num_kv_heads, s->past_seq, dims.head_dim);
            pack_bsh(new_only.data(),   new_dst,  dims.num_kv_heads, s->new_kv_seq, dims.head_dim);
        };
        split_pack(fp32_k, s->past_key_fp16.data(),   s->k_new_fp16.data());
        split_pack(fp32_v, s->past_value_fp16.data(), s->v_new_fp16.data());
    }

    s->present_key_buf.resize(dims.num_kv_heads * s->total_seq * dims.head_dim);
    s->present_value_buf.resize(dims.num_kv_heads * s->total_seq * dims.head_dim);

    auto bytes = build_gqa_model(s->q_seq, s->past_seq, s->total_seq,
                                  dims.num_q_heads, dims.head_dim);

    int threads = bench::get_effective_threads(static_cast<int>(std::thread::hardware_concurrency()));
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(threads);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    try {
        s->session = std::make_unique<Ort::Session>(get_env(), bytes.data(), bytes.size(), opts);
    } catch (const Ort::Exception& e) {
        fprintf(stderr, "[onnxrt gqa] session creation failed: %s\n", e.what());
        delete s;
        return nullptr;
    }
    return s;
}

void* gqa_prefill(const bench::AttnDims& d, size_t sl, size_t cl,
                  const float* q, const float* k, const float* v) {
    return gqa_prepare_impl(d, sl, cl, q, k, v, bench::AttnMode::PREFILL);
}
void* gqa_decode_fp16kv(const bench::AttnDims& d, size_t sl, size_t cl,
                         const float* q, const float* k, const float* v) {
    return gqa_prepare_impl(d, sl, cl, q, k, v, bench::AttnMode::DECODE);
}

void gqa_run(void* state, float* output) {
    auto* s = static_cast<GqaState*>(state);
    if (!s || !s->session) return;

    auto& mem = get_cpu_mem();
    const bool with_past = (s->past_seq > 0);

    int64_t q_shape[]  = {1, (int64_t)s->q_seq, (int64_t)s->hidden};
    int64_t kv_shape[] = {1, (int64_t)s->new_kv_seq, (int64_t)s->hidden};
    int64_t pkv_shape[] = {1, (int64_t)s->dims.num_kv_heads,
                            (int64_t)s->past_seq, (int64_t)s->dims.head_dim};
    int64_t sk_shape[] = {1};
    // 0-dim scalar tensor: pass nullptr shape with dim_count=0 (a zero-sized
    // array is a gcc/clang extension, not portable).
    const int64_t* ts_shape = nullptr;

    if (with_past) {
        Ort::Value inputs[7] = {
            Ort::Value::CreateTensor(mem, s->q_fp16.data(), s->q_fp16.size() * sizeof(uint16_t),
                                      q_shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
            Ort::Value::CreateTensor(mem, s->k_new_fp16.data(), s->k_new_fp16.size() * sizeof(uint16_t),
                                      kv_shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
            Ort::Value::CreateTensor(mem, s->v_new_fp16.data(), s->v_new_fp16.size() * sizeof(uint16_t),
                                      kv_shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
            Ort::Value::CreateTensor(mem, s->past_key_fp16.data(), s->past_key_fp16.size() * sizeof(uint16_t),
                                      pkv_shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
            Ort::Value::CreateTensor(mem, s->past_value_fp16.data(), s->past_value_fp16.size() * sizeof(uint16_t),
                                      pkv_shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
            Ort::Value::CreateTensor<int32_t>(mem, &s->seqlens_k_val, 1, sk_shape, 1),
            Ort::Value::CreateTensor<int32_t>(mem, &s->total_seq_val, 1, ts_shape, 0),
        };
        static const char* in_names[]  = {"Q","K","V","past_key","past_value","seqlens_k","total_seq_len"};
        static const char* out_names[] = {"Y","present_key","present_value"};
        auto out = s->session->Run(s->run_opts, in_names, inputs, 7, out_names, 3);
        if (output) {
            const uint16_t* y = out[0].GetTensorData<uint16_t>();
            unpack_bsh(y, output, s->dims.num_q_heads, s->q_seq, s->dims.head_dim);
        }
    } else {
        Ort::Value inputs[5] = {
            Ort::Value::CreateTensor(mem, s->q_fp16.data(), s->q_fp16.size() * sizeof(uint16_t),
                                      q_shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
            Ort::Value::CreateTensor(mem, s->k_new_fp16.data(), s->k_new_fp16.size() * sizeof(uint16_t),
                                      kv_shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
            Ort::Value::CreateTensor(mem, s->v_new_fp16.data(), s->v_new_fp16.size() * sizeof(uint16_t),
                                      kv_shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
            Ort::Value::CreateTensor<int32_t>(mem, &s->seqlens_k_val, 1, sk_shape, 1),
            Ort::Value::CreateTensor<int32_t>(mem, &s->total_seq_val, 1, ts_shape, 0),
        };
        static const char* in_names[]  = {"Q","K","V","seqlens_k","total_seq_len"};
        static const char* out_names[] = {"Y","present_key","present_value"};
        auto out = s->session->Run(s->run_opts, in_names, inputs, 5, out_names, 3);
        if (output) {
            const uint16_t* y = out[0].GetTensorData<uint16_t>();
            unpack_bsh(y, output, s->dims.num_q_heads, s->q_seq, s->dims.head_dim);
        }
    }
}

void gqa_cleanup(void* state) { delete static_cast<GqaState*>(state); }

} // namespace gqa

static int reg = [] {
    bench::register_matmul_backend({
        "onnxrt_q4", "onnxrt",
        q4_prepare, prepare_act, run_kernel, cleanup
    });
    bench::register_attn_backend({
        "onnxrt_gqa_prefill", "onnxrt", bench::AttnMode::PREFILL,
        gqa::gqa_prefill, gqa::gqa_run, gqa::gqa_cleanup
    });
    // FP16 KV (not INT8) — comparison vs cactus's hybrid_int8 is
    // precision-asymmetric; ORT has no INT8-KV-cache attention op.
    bench::register_attn_backend({
        "onnxrt_gqa_decode_fp16kv", "onnxrt", bench::AttnMode::DECODE,
        gqa::gqa_decode_fp16kv, gqa::gqa_run, gqa::gqa_cleanup
    });
    return 0;
}();

} // namespace

#else
namespace { [[maybe_unused]] static int reg = []{ return 0; }(); }
#endif
