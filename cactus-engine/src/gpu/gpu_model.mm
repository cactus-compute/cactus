/* GPUModel — full forward pass implementation.
 *
 * Walks the gpu_plan.json layer-by-layer, dispatches the matching Metal
 * kernel per op, keeps KV cache GPU-resident, samples on GPU.
 *
 * Activation memory model — simple per-buffer scratch:
 *   residual    : hidden_dim fp16, the accumulating hidden state
 *   norm_out    : RMSNorm output (input to next matmul)
 *   q, k, v     : projection outputs (Q persists for RoPE; K/V → cache)
 *   attn_out    : flash-attention output
 *   mlp_gate    : gate proj output
 *   mlp_up      : up proj output
 *   mlp_silu    : swiglu(gate, up) output
 *   mlp_down    : down proj output
 *   logits      : final LM head output, vocab_size fp16
 *   sampled_id  : single int32 the GPU writes
 *
 * KV cache is per-layer × {K, V}, PRIVATE storage, never copied back.
 */
#import <Foundation/Foundation.h>
#include "gpu_model.h"
#include "../../../cactus-kernels-gpu/include/cactus_gpu.h"
#include "picojson.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace cactus {
namespace engine {

#if CACTUS_HAS_GPU

namespace {

// ---- mmap helper ------------------------------------------------------------
struct MMapped {
    void*  ptr  = nullptr;
    size_t size = 0;
    int    fd   = -1;
    bool open(const std::string& path) {
        struct stat st;
        if (::stat(path.c_str(), &st) != 0) return false;
        size = (size_t)st.st_size;
        if (size == 0) return false;
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) return false;
        ptr = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (ptr == MAP_FAILED) { ::close(fd); fd = -1; ptr = nullptr; return false; }
        return true;
    }
    void close() {
        if (ptr) { ::munmap(ptr, size); ptr = nullptr; size = 0; }
        if (fd >= 0) { ::close(fd); fd = -1; }
    }
    ~MMapped() { close(); }
};

// ---- plan structures --------------------------------------------------------
struct OpSpec {
    std::string kind;
    std::string tag;
    int K = 0, N = 0;
    int head_dim_q = 0, head_dim_v = 0, num_query_groups = 1;
    int head_dim = 0;
    int num_q_heads = 0, num_kv_heads = 0;
    int axis_size = 0;
    int hidden_dim = 0;
    int M_tile = 1;
    bool is_neox = true;
    bool causal = true;
    bool has_softcap = false;
    float theta = 10000.0f;
    int kv_source_layer = -1;    // for flash_attn under sliding-attention
    long long weight_offset_bytes = -1;
    long long weight_nbytes = 0;
    long long scale_offset_bytes = -1;
    long long scale_nbytes = 0;
};

struct LayerSpec {
    int layer_idx;
    std::string attention_kind = "full";   // "full" | "sliding"
    int kv_source_layer = -1;
    std::vector<OpSpec> ops;
};

struct PlanSpec {
    int version = 1;
    std::string model_family;
    int hidden_dim = 0, head_dim = 0;
    int num_q_heads = 0, num_kv_heads = 0;
    int num_layers = 0, vocab_size = 0;
    float rope_theta = 10000.0f;
    bool rope_neox = true;
    std::string weight_format = "cq4_group128";
    std::vector<LayerSpec> layers;
    long long final_norm_offset = -1;
    int final_norm_axis = 0;
    long long lm_head_offset = -1;
    int lm_head_K = 0, lm_head_N = 0;
};

// ---- picojson getters -------------------------------------------------------
static long long _get_int_or_neg1(const picojson::object& o, const char* k) {
    auto it = o.find(k);
    if (it == o.end()) return -1;
    if (it->second.is<picojson::null>()) return -1;
    if (it->second.is<double>()) return (long long)it->second.get<double>();
    return -1;
}
static int _get_int(const picojson::object& o, const char* k, int def = 0) {
    auto it = o.find(k);
    if (it == o.end() || !it->second.is<double>()) return def;
    return (int)it->second.get<double>();
}
static std::string _get_str(const picojson::object& o, const char* k, const char* def = "") {
    auto it = o.find(k);
    if (it == o.end() || !it->second.is<std::string>()) return def;
    return it->second.get<std::string>();
}
static bool _get_bool(const picojson::object& o, const char* k, bool def = false) {
    auto it = o.find(k);
    if (it == o.end() || !it->second.is<bool>()) return def;
    return it->second.get<bool>();
}
static float _get_float(const picojson::object& o, const char* k, float def = 0.0f) {
    auto it = o.find(k);
    if (it == o.end() || !it->second.is<double>()) return def;
    return (float)it->second.get<double>();
}

static bool parse_plan(const std::string& json_text, PlanSpec& plan) {
    picojson::value root;
    std::string err = picojson::parse(root, json_text);
    if (!err.empty() || !root.is<picojson::object>()) {
        std::fprintf(stderr, "GPUModel: plan parse error: %s\n", err.c_str());
        return false;
    }
    const auto& obj = root.get<picojson::object>();
    plan.version       = _get_int(obj, "version", 1);
    plan.model_family  = _get_str(obj, "model_family");
    plan.hidden_dim    = _get_int(obj, "hidden_dim");
    plan.head_dim      = _get_int(obj, "head_dim");
    plan.num_q_heads   = _get_int(obj, "num_q_heads");
    plan.num_kv_heads  = _get_int(obj, "num_kv_heads");
    plan.num_layers    = _get_int(obj, "num_layers");
    plan.vocab_size    = _get_int(obj, "vocab_size");
    plan.rope_theta    = _get_float(obj, "rope_theta", 10000.0f);
    plan.rope_neox     = _get_bool(obj, "rope_neox", true);
    plan.weight_format = _get_str(obj, "weight_format", "cq4_group128");

    auto layers_it = obj.find("layers");
    if (layers_it == obj.end() || !layers_it->second.is<picojson::array>()) return false;
    for (const auto& lv : layers_it->second.get<picojson::array>()) {
        if (!lv.is<picojson::object>()) continue;
        const auto& lo = lv.get<picojson::object>();
        LayerSpec ls;
        ls.layer_idx       = _get_int(lo, "layer_idx");
        ls.attention_kind  = _get_str(lo, "attention_kind", "full");
        ls.kv_source_layer = _get_int(lo, "kv_source_layer", -1);
        auto ops_it = lo.find("ops");
        if (ops_it == lo.end() || !ops_it->second.is<picojson::array>()) continue;
        for (const auto& ov : ops_it->second.get<picojson::array>()) {
            if (!ov.is<picojson::object>()) continue;
            const auto& oo = ov.get<picojson::object>();
            OpSpec op;
            op.kind = _get_str(oo, "kind");
            op.tag  = _get_str(oo, "tag");
            op.K    = _get_int(oo, "K");
            op.N    = _get_int(oo, "N");
            op.head_dim_q       = _get_int(oo, "head_dim_q");
            op.head_dim_v       = _get_int(oo, "head_dim_v");
            op.num_query_groups = _get_int(oo, "num_query_groups", 1);
            op.head_dim    = _get_int(oo, "head_dim");
            op.num_q_heads  = _get_int(oo, "num_q_heads");
            op.num_kv_heads = _get_int(oo, "num_kv_heads");
            op.axis_size   = _get_int(oo, "axis_size");
            op.hidden_dim  = _get_int(oo, "hidden_dim");
            op.M_tile      = _get_int(oo, "M_tile", 1);
            op.is_neox     = _get_bool(oo, "is_neox", true);
            op.causal      = _get_bool(oo, "causal", true);
            op.has_softcap = _get_bool(oo, "has_softcap", false);
            op.theta       = _get_float(oo, "theta", 10000.0f);
            op.kv_source_layer = _get_int(oo, "kv_source_layer", -1);
            op.weight_offset_bytes = _get_int_or_neg1(oo, "weight_offset_bytes");
            op.weight_nbytes       = _get_int(oo, "weight_nbytes", 0);
            op.scale_offset_bytes  = _get_int_or_neg1(oo, "scale_offset_bytes");
            op.scale_nbytes        = _get_int(oo, "scale_nbytes", 0);
            ls.ops.push_back(op);
        }
        plan.layers.push_back(std::move(ls));
    }
    auto fn_it = obj.find("final_norm");
    if (fn_it != obj.end() && fn_it->second.is<picojson::object>()) {
        const auto& fo = fn_it->second.get<picojson::object>();
        plan.final_norm_offset = _get_int_or_neg1(fo, "weight_offset_bytes");
        plan.final_norm_axis   = _get_int(fo, "axis_size");
    }
    auto lh_it = obj.find("lm_head");
    if (lh_it != obj.end() && lh_it->second.is<picojson::object>()) {
        const auto& lo = lh_it->second.get<picojson::object>();
        plan.lm_head_offset = _get_int_or_neg1(lo, "weight_offset_bytes");
        plan.lm_head_K      = _get_int(lo, "K");
        plan.lm_head_N      = _get_int(lo, "N");
    }
    return true;
}

}  // namespace

// ============================================================================
// GPUModel::Impl
// ============================================================================
struct GPUModel::Impl {
    cactus::gpu::Context* ctx = nullptr;
    MMapped weights, scales, embedding;
    cactus::gpu::Buffer* weights_buf = nullptr;
    cactus::gpu::Buffer* scales_buf  = nullptr;
    cactus::gpu::Buffer* embed_buf   = nullptr;

    // Scratch buffers
    cactus::gpu::Buffer* residual    = nullptr;
    cactus::gpu::Buffer* norm_out    = nullptr;
    cactus::gpu::Buffer* q_buf       = nullptr;
    cactus::gpu::Buffer* k_buf       = nullptr;
    cactus::gpu::Buffer* v_buf       = nullptr;
    cactus::gpu::Buffer* attn_buf    = nullptr;
    cactus::gpu::Buffer* mlp_gate    = nullptr;
    cactus::gpu::Buffer* mlp_up      = nullptr;
    cactus::gpu::Buffer* mlp_silu    = nullptr;
    cactus::gpu::Buffer* mlp_down    = nullptr;
    cactus::gpu::Buffer* logits      = nullptr;
    cactus::gpu::Buffer* sampled_id  = nullptr;
    cactus::gpu::Buffer* token_buf   = nullptr;
    cactus::gpu::Buffer* position_buf = nullptr;

    // KV cache
    std::vector<cactus::gpu::Buffer*> kv_k;
    std::vector<cactus::gpu::Buffer*> kv_v;
    size_t kv_max_seq = 0;
    size_t kv_cur_len = 0;

    PlanSpec plan;
    bool loaded = false;

    // Pipeline cache keyed by string.
    std::map<std::string, cactus::gpu::Pipeline*> pipeline_cache;

    cactus::gpu::Pipeline* cache(const std::string& key, cactus::gpu::Pipeline* built) {
        if (built) pipeline_cache[key] = built;
        return built;
    }
    cactus::gpu::Pipeline* lookup(const std::string& key) {
        auto it = pipeline_cache.find(key);
        return it == pipeline_cache.end() ? nullptr : it->second;
    }
};

GPUModel::GPUModel() : impl_(std::make_unique<Impl>()) {}
GPUModel::~GPUModel() {
    if (!impl_) return;
    for (auto& kv : impl_->pipeline_cache) cactus::gpu::pipeline_destroy(kv.second);
    for (auto* b : impl_->kv_k) cactus::gpu::buffer_destroy(b);
    for (auto* b : impl_->kv_v) cactus::gpu::buffer_destroy(b);
    cactus::gpu::buffer_destroy(impl_->weights_buf);
    cactus::gpu::buffer_destroy(impl_->scales_buf);
    cactus::gpu::buffer_destroy(impl_->embed_buf);
    cactus::gpu::buffer_destroy(impl_->residual);
    cactus::gpu::buffer_destroy(impl_->norm_out);
    cactus::gpu::buffer_destroy(impl_->q_buf);
    cactus::gpu::buffer_destroy(impl_->k_buf);
    cactus::gpu::buffer_destroy(impl_->v_buf);
    cactus::gpu::buffer_destroy(impl_->attn_buf);
    cactus::gpu::buffer_destroy(impl_->mlp_gate);
    cactus::gpu::buffer_destroy(impl_->mlp_up);
    cactus::gpu::buffer_destroy(impl_->mlp_silu);
    cactus::gpu::buffer_destroy(impl_->mlp_down);
    cactus::gpu::buffer_destroy(impl_->logits);
    cactus::gpu::buffer_destroy(impl_->sampled_id);
    cactus::gpu::buffer_destroy(impl_->token_buf);
    cactus::gpu::buffer_destroy(impl_->position_buf);
    cactus::gpu::context_destroy(impl_->ctx);
}

bool GPUModel::load(const std::string& bundle_dir, size_t max_context_size) {
    @autoreleasepool {
        std::string metallib_path;
        const char* env = std::getenv("CACTUS_GPU_METALLIB");
        if (env && env[0]) metallib_path = env;
        if (metallib_path.empty()) {
            NSString* res = [[NSBundle mainBundle] resourcePath];
            if (res) {
                metallib_path = std::string([res UTF8String]) + "/cactus_kernels.metallib";
                if (access(metallib_path.c_str(), R_OK) != 0) metallib_path.clear();
            }
        }
        impl_->ctx = cactus::gpu::context_create(metallib_path.c_str());
        if (!impl_->ctx) {
            std::fprintf(stderr, "GPUModel: failed to create Metal context (metallib=%s)\n",
                         metallib_path.c_str());
            return false;
        }

        std::ifstream in(bundle_dir + "/components/gpu/gpu_plan.json");
        if (!in) {
            std::fprintf(stderr, "GPUModel: missing gpu_plan.json under %s\n", bundle_dir.c_str());
            return false;
        }
        std::stringstream ss; ss << in.rdbuf();
        if (!parse_plan(ss.str(), impl_->plan)) return false;

        if (!impl_->weights.open(bundle_dir + "/components/gpu/weights.bin")) {
            std::fprintf(stderr, "GPUModel: cannot mmap weights.bin\n");
            return false;
        }
        impl_->scales.open(bundle_dir + "/components/gpu/scales.bin");
        impl_->embedding.open(bundle_dir + "/components/gpu/embedding.bin");

        impl_->weights_buf = cactus::gpu::buffer_wrap_host_memory(
            impl_->ctx, impl_->weights.ptr, impl_->weights.size);
        if (impl_->scales.ptr)
            impl_->scales_buf = cactus::gpu::buffer_wrap_host_memory(
                impl_->ctx, impl_->scales.ptr, impl_->scales.size);
        if (impl_->embedding.ptr)
            impl_->embed_buf = cactus::gpu::buffer_wrap_host_memory(
                impl_->ctx, impl_->embedding.ptr, impl_->embedding.size);

        const auto& p = impl_->plan;
        const size_t HD = (size_t)p.hidden_dim;
        const size_t QHD = (size_t)p.num_q_heads * (size_t)p.head_dim;
        const size_t KHD = (size_t)p.num_kv_heads * (size_t)p.head_dim;
        size_t MLP_DIM = HD * 4;
        for (const auto& l : p.layers) {
            for (const auto& op : l.ops) {
                if (op.kind == "swiglu_fwd" && op.hidden_dim > 0) { MLP_DIM = (size_t)op.hidden_dim; break; }
            }
            if (MLP_DIM != HD * 4) break;
        }

        auto make = [&](size_t bytes, cactus::gpu::StorageMode m = cactus::gpu::StorageMode::SHARED) {
            return cactus::gpu::buffer_create(impl_->ctx, bytes, m);
        };
        impl_->residual    = make(HD * 2);
        impl_->norm_out    = make(HD * 2);
        impl_->q_buf       = make(QHD * 2);
        impl_->k_buf       = make(KHD * 2);
        impl_->v_buf       = make(KHD * 2);
        impl_->attn_buf    = make(QHD * 2);
        impl_->mlp_gate    = make(MLP_DIM * 2);
        impl_->mlp_up      = make(MLP_DIM * 2);
        impl_->mlp_silu    = make(MLP_DIM * 2);
        impl_->mlp_down    = make(HD * 2);
        impl_->logits      = make((size_t)p.vocab_size * 2);
        impl_->sampled_id  = make(4);
        impl_->token_buf   = make(4);
        impl_->position_buf = make(4);

        impl_->kv_max_seq = max_context_size;
        impl_->kv_cur_len = 0;
        impl_->kv_k.reserve(p.num_layers);
        impl_->kv_v.reserve(p.num_layers);
        for (int li = 0; li < p.num_layers; ++li) {
            const size_t cache_bytes = (size_t)p.num_kv_heads * max_context_size *
                                       (size_t)p.head_dim * 2;
            impl_->kv_k.push_back(cactus::gpu::buffer_create(
                impl_->ctx, cache_bytes, cactus::gpu::StorageMode::PRIVATE));
            impl_->kv_v.push_back(cactus::gpu::buffer_create(
                impl_->ctx, cache_bytes, cactus::gpu::StorageMode::PRIVATE));
        }

        impl_->loaded = true;
        std::fprintf(stderr, "GPUModel: loaded plan (layers=%d hidden=%d head_dim=%d "
                             "qh=%d kvh=%d vocab=%d) + %zu MB weights\n",
                     p.num_layers, p.hidden_dim, p.head_dim, p.num_q_heads,
                     p.num_kv_heads, p.vocab_size, impl_->weights.size / (1024*1024));
        return true;
    }
}

bool GPUModel::is_loaded() const { return impl_ && impl_->loaded; }

// ============================================================================
// decode_one
// ============================================================================
uint32_t GPUModel::decode_one(uint32_t input_token,
                               float temperature, float top_p, size_t top_k) {
    (void)temperature; (void)top_p; (void)top_k;
    if (!impl_ || !impl_->loaded) return 0;
    @autoreleasepool {
        using namespace cactus::gpu;
        auto& I = *impl_;
        const auto& P = I.plan;

        // Stage 1: feed input_token + position.
        {
            int32_t tok = (int32_t)input_token;
            std::memcpy(buffer_contents(I.token_buf), &tok, 4);
            int32_t pos = (int32_t)I.kv_cur_len;
            std::memcpy(buffer_contents(I.position_buf), &pos, 4);
        }

        // Stage 2: embed lookup.
        auto* p_embed = I.lookup("embed");
        if (!p_embed) p_embed = I.cache("embed",
            pipeline_embed_lookup(I.ctx, (uint32_t)P.hidden_dim));

        CommandBuffer* cb = command_buffer_begin(I.ctx);

        // We use commandBufferWithUnretainedReferences (MLX pattern) which
        // means every Buffer* bound during encoding MUST stay alive until
        // command_buffer_wait returns. Collect ephemeral buffers + destroy
        // them after the wait.
        std::vector<Buffer*> ephemeral;
        ephemeral.reserve(64);

        {
            BufferBinding bb[3] = {{I.embed_buf, 0}, {I.token_buf, 0}, {I.residual, 0}};
            const uint32_t tg = 64;
            const uint32_t tgx = (P.hidden_dim + tg - 1) / tg;
            command_buffer_dispatch(cb, p_embed, bb, 3, tgx, 1, 1, tg, 1, 1);
        }
        command_buffer_barrier(cb);

        // PROBE: commit just the embed lookup and check the residual buffer.
        if (std::getenv("CACTUS_GPU_PROBE_EMBED")) {
            command_buffer_commit(cb);
            command_buffer_wait(cb);
            command_buffer_destroy(cb);
            uint16_t* r = (uint16_t*)buffer_contents(I.residual);
            std::fprintf(stderr, "  [probe-embed] residual[0..7]:");
            for (int i = 0; i < 8; ++i) std::fprintf(stderr, " 0x%04x", r[i]);
            std::fprintf(stderr, "\n");
            // Also print embed_table[input_token * hidden + 0..7] for comparison
            uint16_t* e = (uint16_t*)I.embedding.ptr;
            size_t off = (size_t)input_token * (size_t)P.hidden_dim;
            std::fprintf(stderr, "  [probe-embed] embed[%u][0..7]:", input_token);
            for (int i = 0; i < 8; ++i) std::fprintf(stderr, " 0x%04x", e[off + i]);
            std::fprintf(stderr, "\n");
            I.kv_cur_len++;
            return 0;
        }

        // Stage 3: per-layer loop.
        for (size_t li = 0; li < P.layers.size(); ++li) {
            const auto& layer = P.layers[li];
            for (const auto& op : layer.ops) {
                if (op.kind == "rms_norm") {
                    char key[64]; std::snprintf(key, sizeof(key), "rmsnorm:%d", op.axis_size);
                    auto* pp = I.lookup(key);
                    if (!pp) pp = I.cache(key, pipeline_rms_norm_fp16(I.ctx, (uint32_t)op.axis_size));
                    if (!pp) continue;
                    float eps = 1e-6f;
                    Buffer* eps_buf = buffer_create(I.ctx, 4, StorageMode::SHARED);
                    std::memcpy(buffer_contents(eps_buf), &eps, 4);
                    Buffer* w = nullptr;
                    if (op.weight_offset_bytes >= 0) {
                        w = buffer_wrap_host_memory(I.ctx,
                            (char*)I.weights.ptr + op.weight_offset_bytes,
                            op.weight_nbytes);
                    }
                    BufferBinding bb[4] = {{I.residual, 0}, {w, 0}, {I.norm_out, 0}, {eps_buf, 0}};
                    const uint32_t tg = (uint32_t)(op.axis_size / 4);
                    command_buffer_dispatch(cb, pp, bb, 4, 1, 1, 1, tg, 1, 1);
                    command_buffer_barrier(cb);
                    ephemeral.push_back(eps_buf);
                    if (w) ephemeral.push_back(w);
                }
                else if (op.kind == "mul_mv_int4_fp16") {
                    if (op.weight_offset_bytes < 0) continue;  // missing weights (e.g. KV-shared sliding layer)
                    char key[96]; std::snprintf(key, sizeof(key), "mvi4:%d:%d", op.K, op.N);
                    auto* pp = I.lookup(key);
                    if (!pp) pp = I.cache(key,
                        pipeline_mul_mv_int4_fp16(I.ctx, (uint32_t)op.K, (uint32_t)op.N));
                    if (!pp) continue;
                    auto* qs = buffer_wrap_host_memory(I.ctx,
                        (char*)I.weights.ptr + op.weight_offset_bytes, op.weight_nbytes);
                    auto* sc = buffer_wrap_host_memory(I.ctx,
                        (char*)I.scales.ptr + op.scale_offset_bytes, op.scale_nbytes);
                    Buffer* in_buf  = I.norm_out;
                    Buffer* out_buf = nullptr;
                    if      (op.tag == "q_proj")    out_buf = I.q_buf;
                    else if (op.tag == "k_proj")    out_buf = I.k_buf;
                    else if (op.tag == "v_proj")    out_buf = I.v_buf;
                    else if (op.tag == "out_proj")  { in_buf = I.attn_buf;  out_buf = I.mlp_down; }
                    else if (op.tag == "gate_proj") out_buf = I.mlp_gate;
                    else if (op.tag == "up_proj")   out_buf = I.mlp_up;
                    else if (op.tag == "down_proj") { in_buf = I.mlp_silu; out_buf = I.mlp_down; }
                    else { ephemeral.push_back(qs); ephemeral.push_back(sc); continue; }
                    BufferBinding bb[4] = {{qs, 0}, {sc, 0}, {in_buf, 0}, {out_buf, 0}};
                    command_buffer_dispatch(cb, pp, bb, 4, (uint32_t)(op.N / 4), 1, 1, 32, 1, 1);
                    command_buffer_barrier(cb);
                    ephemeral.push_back(qs); ephemeral.push_back(sc);
                }
                else if (op.kind == "rope_apply") {
                    char key[96]; std::snprintf(key, sizeof(key), "rope:%d:%d:%g",
                                                op.head_dim, op.is_neox ? 1 : 0, op.theta);
                    auto* pp = I.lookup(key);
                    if (!pp) pp = I.cache(key,
                        pipeline_rope_apply(I.ctx, (uint32_t)op.head_dim,
                                                       op.is_neox, op.theta));
                    if (!pp) continue;
                    uint32_t nq  = (uint32_t)op.num_q_heads;
                    uint32_t nkv = (uint32_t)op.num_kv_heads;
                    // Kernel reads num_q_heads / num_kv_heads via buffer(3)/(4).
                    auto* nq_buf  = buffer_create(I.ctx, 4, StorageMode::SHARED);
                    auto* nkv_buf = buffer_create(I.ctx, 4, StorageMode::SHARED);
                    std::memcpy(buffer_contents(nq_buf),  &nq,  4);
                    std::memcpy(buffer_contents(nkv_buf), &nkv, 4);
                    BufferBinding bb[5] = {
                        {I.q_buf, 0}, {I.k_buf, 0}, {I.position_buf, 0},
                        {nq_buf, 0}, {nkv_buf, 0}};
                    uint32_t head_max = std::max(nq, std::max(nkv, 1u));
                    command_buffer_dispatch(cb, pp, bb, 5, 1, head_max, 1,
                                            (uint32_t)(op.head_dim / 2), 1, 1);
                    command_buffer_barrier(cb);
                    ephemeral.push_back(nq_buf);
                    ephemeral.push_back(nkv_buf);
                }
                else if (op.kind == "kv_cache_append") {
                    char key[64]; std::snprintf(key, sizeof(key), "kvapp:%d:%d",
                                                op.num_kv_heads, op.head_dim);
                    auto* pp = I.lookup(key);
                    if (!pp) pp = I.cache(key,
                        pipeline_kv_cache_append(I.ctx, (uint32_t)op.num_kv_heads,
                                                            (uint32_t)op.head_dim));
                    if (!pp) continue;
                    uint32_t offset = (uint32_t)I.kv_cur_len;
                    uint32_t maxseq = (uint32_t)I.kv_max_seq;
                    auto* off_buf = buffer_create(I.ctx, 4, StorageMode::SHARED);
                    auto* max_buf = buffer_create(I.ctx, 4, StorageMode::SHARED);
                    std::memcpy(buffer_contents(off_buf), &offset, 4);
                    std::memcpy(buffer_contents(max_buf), &maxseq, 4);
                    BufferBinding bb[6] = {
                        {I.k_buf, 0}, {I.v_buf, 0},
                        {I.kv_k[li], 0}, {I.kv_v[li], 0},
                        {off_buf, 0}, {max_buf, 0}};
                    command_buffer_dispatch(cb, pp, bb, 6,
                                            (uint32_t)op.head_dim,
                                            (uint32_t)op.num_kv_heads, 1,
                                            1, 1, 1);
                    command_buffer_barrier(cb);
                    ephemeral.push_back(off_buf);
                    ephemeral.push_back(max_buf);
                }
                else if (op.kind == "flash_attn") {
                    char key[96]; std::snprintf(key, sizeof(key), "fa:%d:%d:%d",
                                                op.head_dim_q, op.head_dim_v,
                                                op.num_query_groups);
                    auto* pp = I.lookup(key);
                    if (!pp) pp = I.cache(key, pipeline_flash_attn(
                        I.ctx, (uint32_t)op.head_dim_q, (uint32_t)op.head_dim_v,
                        (uint32_t)op.num_query_groups, op.causal, op.has_softcap));
                    if (!pp) continue;
                    float scale = 1.0f / std::sqrt((float)op.head_dim_q);
                    uint32_t seqk = (uint32_t)(I.kv_cur_len + 1);
                    uint32_t maxseq = (uint32_t)I.kv_max_seq;
                    float softcap = 0.0f;
                    auto* s_buf  = buffer_create(I.ctx, 4, StorageMode::SHARED);
                    auto* sk_buf = buffer_create(I.ctx, 4, StorageMode::SHARED);
                    auto* mx_buf = buffer_create(I.ctx, 4, StorageMode::SHARED);
                    auto* sc_buf = buffer_create(I.ctx, 4, StorageMode::SHARED);
                    std::memcpy(buffer_contents(s_buf),  &scale,   4);
                    std::memcpy(buffer_contents(sk_buf), &seqk,    4);
                    std::memcpy(buffer_contents(mx_buf), &maxseq,  4);
                    std::memcpy(buffer_contents(sc_buf), &softcap, 4);
                    // Pick the right KV cache: sliding-attention layers reuse
                    // K/V from `kv_source_layer`; full layers use their own (li).
                    const size_t kv_li = (op.kv_source_layer >= 0)
                        ? (size_t)op.kv_source_layer : li;
                    BufferBinding bb[9] = {
                        {I.q_buf, 0}, {I.kv_k[kv_li], 0}, {I.kv_v[kv_li], 0},
                        {nullptr, 0},
                        {I.attn_buf, 0},
                        {s_buf, 0}, {sk_buf, 0}, {mx_buf, 0}, {sc_buf, 0}};
                    command_buffer_dispatch(cb, pp, bb, 9,
                                            (uint32_t)P.num_q_heads, 1, 1,
                                            32, 1, 1);
                    command_buffer_barrier(cb);
                    ephemeral.push_back(s_buf);  ephemeral.push_back(sk_buf);
                    ephemeral.push_back(mx_buf); ephemeral.push_back(sc_buf);
                }
                else if (op.kind == "swiglu_fwd") {
                    char key[64]; std::snprintf(key, sizeof(key), "swiglu:%d", op.hidden_dim);
                    auto* pp = I.lookup(key);
                    if (!pp) pp = I.cache(key,
                        pipeline_swiglu(I.ctx, (uint32_t)op.hidden_dim));
                    if (!pp) continue;
                    BufferBinding bb[3] = {{I.mlp_gate, 0}, {I.mlp_up, 0}, {I.mlp_silu, 0}};
                    const uint32_t tg = 64;
                    const uint32_t tgx = (op.hidden_dim + tg - 1) / tg;
                    command_buffer_dispatch(cb, pp, bb, 3, tgx, 1, 1, tg, 1, 1);
                    command_buffer_barrier(cb);
                }
                else if (op.kind == "residual_add") {
                    char key[64]; std::snprintf(key, sizeof(key), "resadd:%d", op.axis_size);
                    auto* pp = I.lookup(key);
                    if (!pp) pp = I.cache(key,
                        pipeline_residual_add(I.ctx, (uint32_t)op.axis_size));
                    if (!pp) continue;
                    // The contribution from the just-completed sub-block sits
                    // in mlp_down (both out_proj and down_proj wrote there).
                    BufferBinding bb[2] = {{I.residual, 0}, {I.mlp_down, 0}};
                    const uint32_t tg = 64;
                    const uint32_t tgx = (op.axis_size + tg - 1) / tg;
                    command_buffer_dispatch(cb, pp, bb, 2, tgx, 1, 1, tg, 1, 1);
                    command_buffer_barrier(cb);
                }
            }
        }

        // Stage 4: final norm → LM head → sample.
        // Final RMSNorm into norm_out.
        if (P.final_norm_offset >= 0 && P.final_norm_axis > 0) {
            char key[64]; std::snprintf(key, sizeof(key), "rmsnorm:%d", P.final_norm_axis);
            auto* pp = I.lookup(key);
            if (!pp) pp = I.cache(key,
                pipeline_rms_norm_fp16(I.ctx, (uint32_t)P.final_norm_axis));
            if (pp) {
                float eps = 1e-6f;
                Buffer* eps_buf = buffer_create(I.ctx, 4, StorageMode::SHARED);
                std::memcpy(buffer_contents(eps_buf), &eps, 4);
                Buffer* w = buffer_wrap_host_memory(I.ctx,
                    (char*)I.weights.ptr + P.final_norm_offset, P.final_norm_axis * 2);
                BufferBinding bb[4] = {{I.residual, 0}, {w, 0}, {I.norm_out, 0}, {eps_buf, 0}};
                command_buffer_dispatch(cb, pp, bb, 4, 1, 1, 1,
                                        (uint32_t)(P.final_norm_axis / 4), 1, 1);
                command_buffer_barrier(cb);
                ephemeral.push_back(eps_buf);
                ephemeral.push_back(w);
            }
        }

        // LM head: fp16 mat-vec norm_out → logits.
        if (P.lm_head_offset >= 0 && P.lm_head_K > 0 && P.lm_head_N > 0) {
            char key[96]; std::snprintf(key, sizeof(key), "mvfp16:%d:%d",
                                        P.lm_head_K, P.lm_head_N);
            auto* pp = I.lookup(key);
            if (!pp) pp = I.cache(key,
                pipeline_mul_mv_fp16(I.ctx, (uint32_t)P.lm_head_K, (uint32_t)P.lm_head_N));
            if (pp) {
                Buffer* lmw = buffer_wrap_host_memory(I.ctx,
                    (char*)I.weights.ptr + P.lm_head_offset,
                    (size_t)P.lm_head_K * (size_t)P.lm_head_N * 2);
                BufferBinding bb[3] = {{lmw, 0}, {I.norm_out, 0}, {I.logits, 0}};
                command_buffer_dispatch(cb, pp, bb, 3,
                                        (uint32_t)(P.lm_head_N / 4), 1, 1,
                                        32, 1, 1);
                command_buffer_barrier(cb);
                ephemeral.push_back(lmw);
            }
        }

        // Sample argmax over the vocab.
        {
            auto* pp = I.lookup("argmax");
            if (!pp) pp = I.cache("argmax",
                pipeline_sample_argmax(I.ctx, (uint32_t)P.vocab_size));
            if (pp) {
                BufferBinding bb[2] = {{I.logits, 0}, {I.sampled_id, 0}};
                // Single threadgroup of 1024 threads — argmax kernel handles
                // the per-thread stride internally.
                command_buffer_dispatch(cb, pp, bb, 2, 1, 1, 1, 1024, 1, 1);
                command_buffer_barrier(cb);
            }
        }

        command_buffer_commit(cb);
        command_buffer_wait(cb);
        command_buffer_destroy(cb);

        // Free all the per-token ephemeral wrappers now that GPU work
        // is complete and the buffers are no longer in use.
        for (auto* eb : ephemeral) buffer_destroy(eb);

        // ---- DEBUG: peek at residual + norm_out + logits to see where ----
        // ---- the forward pass is producing meaningful values vs zeros. ----
        if (std::getenv("CACTUS_GPU_DEBUG")) {
            auto peek = [&](Buffer* b, const char* name, int n) {
                if (!b || !buffer_contents(b)) return;
                uint16_t* h = (uint16_t*)buffer_contents(b);
                std::fprintf(stderr, "  [debug] %s[0..%d] (as fp16 raw):", name, n - 1);
                for (int i = 0; i < n; ++i) std::fprintf(stderr, " 0x%04x", h[i]);
                std::fprintf(stderr, "\n");
            };
            peek(I.residual, "residual", 8);
            peek(I.norm_out, "norm_out", 8);
            peek(I.logits,   "logits  ", 8);
            int32_t s; std::memcpy(&s, buffer_contents(I.sampled_id), 4);
            std::fprintf(stderr, "  [debug] sampled_id = %d\n", s);
        }

        // Advance KV cache.
        I.kv_cur_len++;

        // Read sampled token.
        int32_t out = 0;
        std::memcpy(&out, buffer_contents(I.sampled_id), 4);
        return (uint32_t)out;
    }
}

uint32_t GPUModel::prefill_and_sample(const std::vector<uint32_t>& tokens,
                                       float temperature, float top_p, size_t top_k) {
    if (tokens.empty()) return 0;
    uint32_t last = 0;
    for (uint32_t t : tokens) {
        last = decode_one(t, temperature, top_p, top_k);
    }
    return last;
}

void GPUModel::reset_cache() {
    if (impl_) impl_->kv_cur_len = 0;
}
size_t GPUModel::kv_cache_length()   const { return impl_ ? impl_->kv_cur_len  : 0; }
size_t GPUModel::kv_cache_capacity() const { return impl_ ? impl_->kv_max_seq  : 0; }

#endif  // CACTUS_HAS_GPU

}  // namespace engine
}  // namespace cactus
