#include "../cactus_graph.h"
#include "cactus_kernels.h"
#include "metal_backend.h"

#include <cstring>
#include <vector>
#include <cstdint>

namespace {

static bool g_metal_argmax_valid = false;
static const float* g_metal_argmax_buf = nullptr;

}

static bool g_last_step_adjusted = false;
static bool g_last_amax_biased = false;
static bool g_prefill_consistent = false;

void cactus_graph_mark_unadjusted() {
    g_last_step_adjusted = false;
    g_last_amax_biased = false;
    g_metal_argmax_valid = false;
}

void cactus_graph_set_prefill_consistent(bool on) { g_prefill_consistent = on; }

bool cactus_graph_prefill_consistent() { return g_prefill_consistent; }

struct GSampling {
    bool active = false;
    float rep_penalty = 1.0f;
    std::vector<uint32_t> recent;
    const float* bias_dense = nullptr;
    size_t bias_len = 0;
    long long suppressed = -1;
};
static GSampling g_samp;
static float* g_bias_dense = nullptr;
static size_t g_bias_dense_n = 0;

void cactus_graph_set_sampling(const uint32_t* recent, int n_recent, float rep_penalty,
                               const float* bias_dense, size_t bias_len,
                               long long suppressed) {
    g_samp.active = true;
    g_samp.rep_penalty = rep_penalty;
    g_samp.recent.assign(recent, recent + (n_recent > 0 ? n_recent : 0));
    g_samp.bias_dense = bias_dense;
    g_samp.bias_len = bias_dense ? bias_len : 0;
    g_samp.suppressed = suppressed;
}
void cactus_graph_clear_sampling() { g_samp = GSampling(); }
bool cactus_graph_metal_adjusted() { return g_last_step_adjusted; }
bool cactus_graph_metal_argmax_biased() { return g_last_amax_biased; }

static void* g_plan_amax = nullptr;
static bool g_plan_tail_pending = false;
static bool g_plan_tail_adjusted = false;
static bool g_plan_tail_biased = false;


static const float* update_bias_dense(size_t vocab) {
    if (!g_samp.bias_dense || g_samp.bias_len == 0) return nullptr;
    if (!g_bias_dense || g_bias_dense_n < vocab) {
        if (g_bias_dense) cactus_metal_free_shared(g_bias_dense);
        g_bias_dense = (float*)cactus_metal_alloc_shared(vocab * sizeof(float));
        if (!g_bias_dense) { g_bias_dense_n = 0; return nullptr; }
        g_bias_dense_n = vocab;
    }
    size_t n = g_samp.bias_len < vocab ? g_samp.bias_len : vocab;
    std::memcpy(g_bias_dense, g_samp.bias_dense, n * sizeof(float));
    if (n < vocab) std::memset(g_bias_dense + n, 0, (vocab - n) * sizeof(float));
    return g_bias_dense;
}

bool cactus_graph_metal_tail(void* logits, size_t vocab) {
    g_plan_tail_pending = false;
    bool adjusted = false;
    if (g_samp.active &&
        ((g_samp.rep_penalty != 1.0f && !g_samp.recent.empty()) || g_samp.suppressed >= 0)) {
        adjusted = cactus_metal_encode_adjust_logits(logits, vocab,
            g_samp.recent.data(), (uint32_t)g_samp.recent.size(),
            g_samp.suppressed, g_samp.rep_penalty);
        if (!adjusted) return false;
    }
    if (!g_plan_amax) {
        g_plan_amax = cactus_metal_alloc_shared(3 * sizeof(float));
        if (!g_plan_amax) return false;
    }
    const float* bias = g_samp.active ? update_bias_dense(vocab) : nullptr;
    if (!cactus_metal_encode_argmax(logits, (uint32_t)vocab, g_plan_amax, bias)) return false;
    g_plan_tail_pending = true;
    g_plan_tail_adjusted = adjusted;
    g_plan_tail_biased = (bias != nullptr);
    return true;
}

void cactus_graph_metal_tail_commit() {
    if (!g_plan_tail_pending) return;
    g_plan_tail_pending = false;
    g_metal_argmax_buf = (const float*)g_plan_amax;
    g_metal_argmax_valid = true;
    g_last_step_adjusted = g_plan_tail_adjusted;
    g_last_amax_biased = g_plan_tail_biased;
}


bool cactus_graph_metal_argmax(uint32_t* idx, float* best, float* second) {
    if (!g_metal_argmax_valid || !g_metal_argmax_buf) return false;
    g_metal_argmax_valid = false;
    *best = g_metal_argmax_buf[0]; *second = g_metal_argmax_buf[1];
    *idx = (uint32_t)g_metal_argmax_buf[2];
    return true;
}

static FusedEmbedCtx g_fe;
void cactus_graph_set_fused_embed(const FusedEmbedCtx* ctx) {
    if (ctx && ctx->ok) g_fe = *ctx; else g_fe.ok = false;
}
const FusedEmbedCtx* cactus_graph_fused_embed() { return g_fe.ok ? &g_fe : nullptr; }

bool cactus_graph_metal_fold_prologue(void* h_buf, void* ple_buf, void* pos_buf,
                                      const CactusQuantMatrix* lm_head, size_t nl, size_t ple_dim) {
    if (!g_fe.ok || !lm_head) return false;
    static void* pe = nullptr;
    static void* pa = nullptr;
    static void* pj = nullptr;
    static void* pjs = nullptr;
    static size_t cap = 0;
    static size_t cap_pe = 0;
    const size_t PK = g_fe.proj.N;
    const size_t EK = g_fe.ple.K;
    if (cap_pe < EK) {
        if (pe) cactus_metal_free_shared(pe);
        pe = cactus_metal_alloc_shared(EK * 2);
        cap_pe = pe ? EK : 0;
    }
    if (cap < PK) {
        if (pa) cactus_metal_free_shared(pa);
        if (pj) cactus_metal_free_shared(pj);
        if (pjs) cactus_metal_free_shared(pjs);
        pa = cactus_metal_alloc_shared(PK * 2);
        pj = cactus_metal_alloc_shared(PK * 2);
        pjs = cactus_metal_alloc_shared(PK * 2);
        cap = (pa && pj && pjs) ? PK : 0;
    }
    if (!pe || !pa || !pj || !pjs) return false;
    uint32_t tok = (uint32_t)g_fe.token_id;
    if (!cactus_metal_encode_embedding_ortho(h_buf, tok, lm_head, g_fe.emb_scale)) return false;
    if (!cactus_metal_encode_embedding_hadamard(pe, tok, &g_fe.ple)) return false;
    cactus_metal_encode_scalar(2, pa, pe, PK, g_fe.ple_scale);
    if (!cactus_metal_encode_transform_gemv(pj, h_buf, &g_fe.proj, nullptr)
        && !cactus_metal_encode_quant_matmul(pj, h_buf, &g_fe.proj)) return false;
    cactus_metal_encode_scalar(2, pjs, pj, PK, g_fe.proj_scale);
    if (!cactus_metal_encode_rms_norm_add_scale(ple_buf, pjs, g_fe.rms_weight, pa, nl, ple_dim, g_fe.rms_eps, g_fe.final_scale)) {
        if (!cactus_metal_encode_rms_norm_add(pa, pjs, g_fe.rms_weight, pa, nl, ple_dim, g_fe.rms_eps)) return false;
        cactus_metal_encode_scalar(2, ple_buf, pa, PK, g_fe.final_scale);
    }
    if (pos_buf) *(float*)pos_buf = (float)g_fe.position;
    return true;
}

bool CactusGraph::extract_ple_pathway(FusedEmbedCtx& ctx) const {

    if (nodes_.size() != 23) return false;
    auto op = [&](size_t i){ return nodes_[i]->op_type; };
    if (op(3) != OpType::INPUT || op(4) != OpType::INPUT || op(5) != OpType::INPUT) return false;
    if (op(8) != OpType::EMBEDDING || op(12) != OpType::MATMUL || op(19) != OpType::RMS_NORM) return false;
    if (op(7) != OpType::SCALAR_MULTIPLY || op(9) != OpType::SCALAR_MULTIPLY ||
        op(14) != OpType::SCALAR_MULTIPLY || op(22) != OpType::SCALAR_MULTIPLY) return false;
    const BufferDesc& pleW = nodes_[3]->output_buffer;
    const BufferDesc& projW = nodes_[5]->output_buffer;
    if (!pleW.is_cq() || (pleW.cq_flags & CACTUS_QUANT_FLAG_ORTHOGONAL) || !projW.is_cq()) return false;
    ctx.ple = pleW.to_cq_matrix();
    ctx.proj = projW.to_cq_matrix();
    ctx.rms_weight = nodes_[4]->output_buffer.get_data();
    ctx.emb_scale = nodes_[7]->params.scalar;
    ctx.ple_scale = nodes_[9]->params.scalar;
    ctx.proj_scale = nodes_[14]->params.scalar;
    ctx.final_scale = nodes_[22]->params.scalar;
    ctx.rms_eps = nodes_[19]->params.epsilon;
    if (!ctx.rms_weight || ctx.emb_scale == 0.0f || ctx.proj.N == 0) return false;
    ctx.ok = true;
    return true;
}

void cactus_graph_on_destroy(const void* graph) {
    (void)graph;
    g_samp = GSampling();
    cactus_metal_invalidate_host_wraps();
}

CactusGraph::~CactusGraph() {
    cactus_graph_on_destroy(this);
    invalidate_metal_state();
}
