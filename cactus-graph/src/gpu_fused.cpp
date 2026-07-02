

#include "../cactus_graph.h"
#include "cactus_kernels.h"
#include "metal_backend.h"
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <cstdint>
#include <chrono>

namespace {

constexpr int NL = 35, HID = 1536, NQH = 8, PLE_DIM = 256;
constexpr float RMS_EPS = 1e-6f, GATE_SCALE = 0.015625f, SOFTCAP = 30.0f;

struct LayerW {
    CactusQuantMatrix q, o, ple_down, ple_up, gate, up, down;
    CactusQuantMatrix k, v;
    const void *in_norm, *q_norm, *k_norm, *post_attn, *pre_ffn, *post_ffn, *post_ple, *scalar;
    bool shared, global;
    int hd, mlp;
    void *kc, *vc;
};

struct Arena {
    std::vector<std::pair<void*,size_t>> bufs;
    size_t cur = 0;
    void reset() { cur = 0; }
    void* fresh(size_t bytes) {
        if (cur >= bufs.size()) {
            void* p = cactus_metal_alloc_shared(bytes);
            if (p) std::memset(p, 0, bytes);
            bufs.push_back({p, bytes});
        }
        return bufs[cur++].first;
    }
};

struct GpuModel {
    bool init = false;
    LayerW L[NL];
    size_t kc_idx[15], vc_idx[15];
    const void *vnorm, *vnorm_g;
    const void *cosL, *sinL, *cosG, *sinG;
    const float *position;
    const void *hidden;
    const void *ple;
    const void *final_norm;
    CactusQuantMatrix lm_head;
    void *cos_s, *sin_s, *cos_g, *sin_g;
    void *argmax_buf = nullptr;
    Arena arena;
};
static GpuModel G;

static bool g_gpu_argmax_valid = false;
static const float* g_gpu_argmax_buf = nullptr;

}

bool cactus_graph_gpu_argmax(uint32_t* idx, float* best, float* second) {
    if (!g_gpu_argmax_valid || !g_gpu_argmax_buf) return false;
    g_gpu_argmax_valid = false;
    *best = g_gpu_argmax_buf[0]; *second = g_gpu_argmax_buf[1];
    *idx = (uint32_t)g_gpu_argmax_buf[2];
    return true;
}

static FusedEmbedCtx g_fe;
void cactus_graph_set_fused_embed(const FusedEmbedCtx* ctx) {
    if (ctx && ctx->ok) g_fe = *ctx; else g_fe.ok = false;
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

bool CactusGraph::execute_gpu_fused() {
    if (!cactus_metal_available()) return false;
    const size_t n = nodes_.size();
    if ((int)n == 15510 && std::getenv("CACTUS_NO_G26_FUSED") == nullptr)
        if (execute_gpu_fused_g26()) return true;
    if (n < 3861) return false;
    auto B = [&](size_t idx) -> BufferDesc& { return nodes_[idx]->output_buffer; };
    auto P = [&](size_t idx) -> void* { return B(idx).get_data(); };
    if (B(0).total_size != HID) return false;
    if (nodes_[0]->op_type != OpType::INPUT || B(0).precision != Precision::FP16) return false;
    g_gpu_argmax_valid = false;

    if (!G.init) {

        auto cq = [&](size_t idx){ return B(idx).to_cq_matrix(); };
        for (int l = 0; l < NL; ++l) {
            LayerW& w = G.L[l];
            w.global = (l % 5 == 4);
            w.hd = w.global ? 512 : 256;
            w.shared = (l >= 15);
            if (!w.shared) {
                size_t b1 = 3 + 10*l, b2 = 334 + 6*l;
                w.in_norm=P(b1); w.q_norm=P(b1+1); w.k_norm=P(b1+2); w.post_attn=P(b1+3);
                w.pre_ffn=P(b1+4); w.gate=cq(b1+5); w.up=cq(b1+6); w.down=cq(b1+7);
                w.post_ffn=P(b1+8); w.post_ple=P(b1+9);
                w.q=cq(b2); w.k=cq(b2+1); w.v=cq(b2+2); w.o=cq(b2+3); w.ple_down=cq(b2+4); w.ple_up=cq(b2+5);
                w.mlp = 6144;
            } else {
                int j = l - 15; size_t b1 = 153 + 9*j, b2 = 424 + 4*j;
                w.in_norm=P(b1); w.q_norm=P(b1+1); w.post_attn=P(b1+2); w.pre_ffn=P(b1+3);
                w.gate=cq(b1+4); w.up=cq(b1+5); w.down=cq(b1+6); w.post_ffn=P(b1+7); w.post_ple=P(b1+8);
                w.q=cq(b2); w.o=cq(b2+1); w.ple_down=cq(b2+2); w.ple_up=cq(b2+3);
                w.mlp = 12288;
            }
            w.scalar = P(505 + l);
        }
        G.vnorm = P(540); G.vnorm_g = P(541);
        G.cosL = P(542); G.sinL = P(543); G.cosG = P(544); G.sinG = P(545);
        G.final_norm = P(333); G.lm_head = B(504).to_cq_matrix();

        std::vector<size_t> cidx;
        for (size_t i = 0; i < n && cidx.size() < 30; ++i)
            if (nodes_[i]->op_type == OpType::KV_CACHE_STATE) cidx.push_back(i);
        if (cidx.size() < 30) return false;
        for (int l = 0; l < 15; ++l) { G.kc_idx[l] = cidx[2*l]; G.vc_idx[l] = cidx[2*l+1]; }

        auto sh = [&](size_t bytes){ void* p = cactus_metal_alloc_shared(bytes); if (p) std::memset(p,0,bytes); return p; };
        G.cos_s = sh(256*2); G.sin_s = sh(256*2); G.cos_g = sh(512*2); G.sin_g = sh(512*2);
        G.argmax_buf = sh(3*sizeof(float));
        // wrap frequently-bound weight pointers so encoding doesn't memcpy them per token
        for (int l = 0; l < NL; ++l) {
            LayerW& w = G.L[l];
            const void* ptrs[] = { w.in_norm, w.q_norm, w.k_norm, w.post_attn, w.pre_ffn, w.post_ffn, w.post_ple };
            size_t sizes[]     = { (size_t)HID*2, (size_t)w.hd*2, (size_t)w.hd*2, (size_t)HID*2, (size_t)HID*2, (size_t)HID*2, (size_t)HID*2 };
            for (int i = 0; i < 7; ++i) if (ptrs[i]) cactus_metal_register_readonly(ptrs[i], sizes[i]);
        }
        cactus_metal_register_readonly(G.vnorm, 256*2);
        cactus_metal_register_readonly(G.vnorm_g, 512*2);
        cactus_metal_register_readonly(G.final_norm, (size_t)HID*2);
        G.init = true;
    }

    const bool fold = g_fe.ok;

    int pos;
    if (fold) {
        pos = g_fe.position;
    } else {
        G.hidden = P(0); G.ple = P(1); G.position = static_cast<float*>(P(2));
        if (!G.hidden || !G.ple || !G.position) return false;
        pos = (int)G.position[0];
    }
    for (int l = 0; l < 15; ++l) { G.L[l].kc = P(G.kc_idx[l]); G.L[l].vc = P(G.vc_idx[l]); }
    for (int l = 15; l < NL; ++l) { int src = G.L[l].global ? 14 : 13; G.L[l].kc = G.L[src].kc; G.L[l].vc = G.L[src].vc; }
    for (int l = 0; l < NL; ++l) if (!G.L[l].kc || !G.L[l].vc) return false;

    for (int l = 0; l < 15; ++l) {
        const uint64_t* km = (const uint64_t*)G.L[l].kc;
        if (G.L[l].global && km[0] + 1 > km[1]) return false;
    }

    G.arena.reset();
    auto fresh = [&](size_t elems){ return G.arena.fresh(elems * 2); };
    cactus_metal_session_begin();
    cactus_metal_set_active(true);
    cactus_metal_manual_begin();
    auto bail = [&](){ cactus_metal_manual_end(); cactus_metal_set_active(false); cactus_metal_session_end(); return false; };
    // CACTUS_ABLATE: comma list of {attn,qkv,o,mlp,ple,lmhead,nobar} — perf
    // attribution only, output is garbage while set
    static const char* abl = std::getenv("CACTUS_ABLATE");
    static const auto has = [](const char* f){ return abl && std::strstr(abl, f); };
    static const bool ab_attn=has("attn"), ab_qkv=has("qkv"), ab_o=has("oproj");
    static const bool ab_mlp=has("mlp"), ab_ple=has("ple"), ab_head=has("lmhead"), ab_nobar=has("nobar");
    static const bool ab_gu=ab_mlp||has("gup"), ab_down=ab_mlp||has("down");
    static const char* rd = std::getenv("CACTUS_RESID_ONLY");
    static const bool r_qkv = !rd || std::strstr(rd, "qkv");
    static const bool r_gu  = !rd || std::strstr(rd, "gu");
    static const bool r_pd  = !rd || std::strstr(rd, "pd");
    auto sync_point = [&](){ if (!ab_nobar) cactus_metal_barrier(); };

    void* h = fresh(HID);
    const void* pleBase;
    if (fold) {
        const uint32_t tok = (uint32_t)g_fe.token_id;
        const int PK = (int)g_fe.proj.N;

        // h = emb(tok)*emb_scale and pe = ple_emb(tok) are independent
        if (!cactus_metal_encode_embedding_ortho(h, tok, &G.lm_head, g_fe.emb_scale)) return bail();
        void* pe = fresh(g_fe.ple.K);
        if (!cactus_metal_encode_embedding_hadamard(pe, tok, &g_fe.ple)) return bail();
        sync_point();
        void* pa = fresh(PK); cactus_metal_encode_scalar(2, pa, pe, PK, g_fe.ple_scale);
        void* pj = fresh(PK);
        if (!cactus_metal_encode_quant_matmul(pj, h, &g_fe.proj)) return bail();
        sync_point();
        void* pjs= fresh(PK); cactus_metal_encode_scalar(2, pjs, pj, PK, g_fe.proj_scale);
        sync_point();
        void* psum=fresh(PK); cactus_metal_encode_rms_norm_add(psum, pjs, g_fe.rms_weight, pa, NL, PLE_DIM, g_fe.rms_eps, g_fe.final_scale);
        pleBase = psum;
        sync_point();
    } else {
        std::memcpy(h, G.hidden, HID*2);
        pleBase = G.ple;
    }

    auto gather = [&](const void* tbl, void* dst, int hd){ std::memcpy(dst, (const char*)tbl + (size_t)pos*hd*2, hd*2); };
    gather(G.cosL, G.cos_s, 256); gather(G.sinL, G.sin_s, 256);
    gather(G.cosG, G.cos_g, 512); gather(G.sinG, G.sin_g, 512);

    // xn for layer 0; later layers fold the previous layer's residual close
    // and their own pre-norm into the q/k/v transform dispatch
    void* xn = fresh(HID);
    cactus_metal_encode_rms_norm(xn, h, G.L[0].in_norm, 1, HID, RMS_EPS);
    sync_point();

    // carried from the previous layer for the fused residual close
    void* prev_pu = nullptr;   // ple up output of layer l-1
    void* prev_h2 = nullptr;   // post-ffn residual of layer l-1
    const void* prev_w1 = nullptr;
    float prev_ls = 1.0f;

    for (int l = 0; l < NL; ++l) {

        LayerW& w = G.L[l];
        const int hd = w.hd, QD = NQH*hd;
        const void* rc = w.global ? G.cos_g : G.cos_s;
        const void* rs = w.global ? G.sin_g : G.sin_s;
        uint64_t* km = (uint64_t*)w.kc; uint64_t* vm = (uint64_t*)w.vc;
        size_t mx = km[1], ng = (hd+31)/32;

        const uint32_t Wn = w.global ? 0u : (uint32_t)(km[1] - km[4] - 1);
        const uint32_t Sn = w.global ? 0u : (uint32_t)km[4];
        const uint32_t Rn = (Wn > Sn) ? (Wn - Sn) : 1u;
        void* q  = fresh(QD);
        void* k  = nullptr;
        void* v  = nullptr;
        size_t kv_start = 0, kv_end;
        uint32_t use_local, local_slot;

        // q/k/v projections; for l>0 the previous layer's residual close
        // (h = clamp(h2 + rms(pu)*post_ple)*ls) and this layer's pre-norm run
        // inside the transform dispatch
        {
            uint32_t cnt = w.shared ? 1u : 3u;
            if (!w.shared) { k = fresh(hd); v = fresh(hd); }
            void* outs[3] = { q, k, v };
            const CactusQuantMatrix* ws[3] = { &w.q, &w.k, &w.v };
            bool enc = false;
            if (!ab_qkv) {
                if (l > 0 && r_qkv) {
                    void* hn = fresh(HID);
                    enc = cactus_metal_encode_quant_matmul_many_resid(outs, ws, cnt,
                            prev_pu, prev_w1, prev_h2, w.in_norm, hn, prev_ls, RMS_EPS);
                    if (enc) h = hn;
                }
                if (!enc) {
                    if (l > 0) {
                        void* hn = fresh(HID);
                        void* xnn = fresh(HID);
                        cactus_metal_encode_rms_norm_add_rms(hn, xnn, prev_pu, prev_w1, prev_h2, w.in_norm, 1, HID, RMS_EPS, prev_ls);
                        h = hn; xn = xnn;
                        sync_point();
                    }
                    if (cnt == 1 || !cactus_metal_encode_quant_matmul_many(outs, xn, ws, cnt)) {
                        for (uint32_t i = 0; i < cnt; ++i)
                            cactus_metal_encode_quant_matmul(outs[i], xn, ws[i]);
                    }
                }
            }
            sync_point();
        }
        if (!w.shared) {
            size_t clen = km[0];
            bool ring_wrapped = (!w.global && clen >= (size_t)Wn);
            use_local = 1u;
            local_slot = ring_wrapped ? (uint32_t)(Sn + ((clen - Sn) % Rn)) : (uint32_t)clen;
            km[0] = clen + 1; vm[0] = clen + 1;
            kv_end = ring_wrapped ? (size_t)Wn : std::min(clen + 1, (size_t)pos + 1);
        } else {
            use_local = 0u; local_slot = 0u;
            size_t clen = km[0];
            kv_end = (!w.global && clen > (size_t)Wn) ? (size_t)Wn
                                                      : std::min(clen, (size_t)pos + 1);
        }
        void* attn = fresh(QD);
        if (!ab_attn) cactus_metal_encode_attention_fused_i8(attn, q, k, v,
            w.q_norm, w.k_norm, w.global ? G.vnorm_g : G.vnorm, rc, rs,
            (char*)w.kc+64, (char*)w.vc+64, (char*)w.kc+64+mx*hd, (char*)w.vc+64+mx*hd,
            NQH, (uint32_t)hd, 1.0f, RMS_EPS,
            (uint32_t)kv_start, (uint32_t)kv_end, use_local, local_slot,
            mx*hd, mx*hd, mx*ng*sizeof(float), mx*ng*sizeof(float));
        sync_point();
        void* o  = fresh(HID);
        if (!ab_o) cactus_metal_encode_quant_matmul(o, attn, &w.o);
        sync_point();

        // gate/up with the post-attention residual close + pre-ffn norm fused
        // into their transform dispatch; h becomes h1
        const int M = w.mlp;
        void* gate= fresh(M);
        void* up  = fresh(M);
        if (!ab_gu) {
            void* gu_outs[2] = { gate, up };
            const CactusQuantMatrix* gu_ws[2] = { &w.gate, &w.up };
            void* h1 = fresh(HID);
            if (r_gu && cactus_metal_encode_quant_matmul_many_resid(gu_outs, gu_ws, 2,
                    o, w.post_attn, h, w.pre_ffn, h1, 1.0f, RMS_EPS)) {
                h = h1;
            } else {
                void* xn2 = fresh(HID);
                cactus_metal_encode_rms_norm_add_rms(h1, xn2, o, w.post_attn, h, w.pre_ffn, 1, HID, RMS_EPS, 1.0f);
                h = h1;
                sync_point();
                if (!cactus_metal_encode_quant_matmul_many(gu_outs, xn2, gu_ws, 2)) {
                    cactus_metal_encode_quant_matmul(gate, xn2, &w.gate);
                    cactus_metal_encode_quant_matmul(up, xn2, &w.up);
                }
            }
        }
        sync_point();
        void* mo  = fresh(HID);
        if (ab_down) { /* skip down proj */ }
        else if (!cactus_metal_encode_quant_matmul_swiglu(mo, gate, up, GATE_SCALE, &w.down)) {
            void* g3 = fresh(M);
            cactus_metal_encode_swiglu(g3, gate, up, M, GATE_SCALE);
            sync_point();
            cactus_metal_encode_quant_matmul(mo, g3, &w.down);
        }
        sync_point();

        // ple down with the post-ffn residual close fused in; h becomes h2
        void* ps  = fresh(PLE_DIM);
        void* h2  = fresh(HID);
        if (!ab_ple) {
            if (r_pd && cactus_metal_encode_quant_matmul_resid(ps, mo, w.post_ffn, h, h2, RMS_EPS, &w.ple_down)) {
                h = h2;
            } else {
                cactus_metal_encode_rms_norm_add(h2, mo, w.post_ffn, h, 1, HID, RMS_EPS); h = h2;
                sync_point();
                cactus_metal_encode_quant_matmul(ps, h, &w.ple_down);
            }
        } else {
            cactus_metal_encode_rms_norm_add(h2, mo, w.post_ffn, h, 1, HID, RMS_EPS); h = h2;
        }
        sync_point();
        void* pu  = fresh(HID);
        if (ab_ple) { /* skip ple up proj */ }
        else if (!cactus_metal_encode_quant_matmul_swiglu(pu, ps, (const char*)pleBase + (size_t)l*PLE_DIM*2, 1.0f, &w.ple_up)) {
            void* pm = fresh(PLE_DIM);
            cactus_metal_encode_swiglu(pm, ps, (const char*)pleBase + (size_t)l*PLE_DIM*2, PLE_DIM, 1.0f);
            sync_point();
            cactus_metal_encode_quant_matmul(pu, pm, &w.ple_up);
        }
        sync_point();

        prev_pu = pu; prev_h2 = h; prev_w1 = w.post_ple;
        prev_ls = (float)(*(const __fp16*)w.scalar);
        // let the GPU start on the first half while the CPU encodes the rest
        if (l % 6 == 5) cactus_metal_session_flush();
    }

    // final residual close + final norm
    {
        void* hf = fresh(HID);
        void* fnb = fresh(HID);
        cactus_metal_encode_rms_norm_add_rms(hf, fnb, prev_pu, prev_w1, prev_h2, G.final_norm, 1, HID, RMS_EPS, prev_ls);
        xn = fnb;
        sync_point();
    }

    size_t V = B(3860).total_size;
    void* fn   = xn;  // final rms-norm output
    void* code = fresh(HID);
    void* lg   = fresh(V);
    if (ab_head) {
        B(3860).set_external(lg);
        cactus_metal_manual_end();
        cactus_metal_set_active(false);
        cactus_metal_session_end();
        return true;
    }
    if (cactus_metal_encode_quant_matmul_ortho(lg, fn, code, &G.lm_head)) {
        sync_point();
        cactus_metal_encode_softcap(lg, lg, V, SOFTCAP);
        B(3860).set_external(lg);
        if (G.argmax_buf) {
            sync_point();
            if (cactus_metal_encode_argmax(lg, (uint32_t)V, G.argmax_buf)) {
                g_gpu_argmax_buf = (const float*)G.argmax_buf;
                g_gpu_argmax_valid = true;
            }
        }
    } else {
        cactus_metal_session_sync();
        cactus_quant_matmul(&G.lm_head, (const __fp16*)fn, 1u, (__fp16*)lg);
        __fp16* L=(__fp16*)lg; for (size_t i=0;i<V;++i){ float v=(float)L[i]; L[i]=(__fp16)(SOFTCAP*std::tanh(v/SOFTCAP)); }
        B(3860).set_external(lg);
    }

    cactus_metal_manual_end();
    cactus_metal_set_active(false);
    cactus_metal_session_end();
    return true;
}

// ---------------------------------------------------------------------------
// Hand-fused decode path for gemma4-26B-A4B (MoE, cq2). Layer/node indices
// come from gpu_fused_g26_plan.inc (generated from a graph dump and verified
// against the live graph before first use). Norms/matmuls/rope/residuals are
// encoded directly with fused kernels; attention, kv-append and the MoE layer
// are delegated to encode_node_gpu() so their (cache-metadata dependent)
// logic stays single-sourced.
#include "gpu_fused_g26_plan.inc"

namespace {

struct G26Layer {
    CactusQuantMatrix q, k, v, o, gate, up, down;
    const void *in_norm, *q_norm, *k_norm, *v_norm, *post_attn, *pre_ffn, *post_ffn;
    const void *post_moe, *post_moe2;
    const void *router_w;                 // fp16 [128, 2816]
    void *router_norm_f = nullptr;        // fp16, folded * ROUTER_IN_SCALE * 16
    void *moe_h_norm_f = nullptr;         // fp16, folded * 16
    float ls = 1.0f;
    int hd = 0, win = 0, pad_n = 0;
};

struct G26Model {
    bool init = false;
    bool bad = false;
    G26Layer L[30];
    const void *cosL, *sinL, *cosG, *sinG;
    CactusQuantMatrix lm;
    // persistent activation buffers
    void *h, *xn, *q, *k, *v, *qr, *kr, *vn, *attn_o, *o, *h1;
    void *xf, *r_in, *moe_h, *gate, *up, *gu_pad, *lg16, *probs, *topk;
    void *dn, *moe_out, *f_rms, *m_rms, *s1, *fn, *lg;
    void *cos_s, *sin_s, *cos_g, *sin_g;
    void *argmax_buf;
};
static G26Model M26;

}

bool CactusGraph::execute_gpu_fused_g26() {
    if (!cactus_metal_available()) return false;
    if (M26.bad) return false;
    const size_t n = nodes_.size();
    if ((int)n != G26_NODES) return false;
    auto B = [&](size_t idx) -> BufferDesc& { return nodes_[idx]->output_buffer; };
    auto P = [&](size_t idx) -> void* { return B(idx).get_data(); };
    auto OP = [&](size_t idx) { return nodes_[idx]->op_type; };
    g_gpu_argmax_valid = false;

    constexpr int HID = 2816, NQH = 16, NKVH = 8, NE = 704;

    if (!M26.init) {
        auto bad = [&](const char* why){
            CACTUS_LOG_WARN("g26", "fused plan rejected: " << why);
            M26.bad = true; return false;
        };
        // structural verification against the generated plan
        if (OP(G26_H_IN) != OpType::INPUT || OP(G26_POS_IN) != OpType::INPUT) return bad("inputs");
        if (OP(G26_OUT) != OpType::SCALAR_MULTIPLY) return bad("out");
        for (int l = 0; l < 30; ++l) {
            const G26LayerPlan& p = G26_L[l];
            if (OP(p.attn) != OpType::ATTENTION_CACHED || OP(p.moe) != OpType::MOE_LAYER ||
                OP(p.k_app) != OpType::KV_CACHE_APPEND || OP(p.v_app) != OpType::KV_CACHE_APPEND ||
                OP(p.kc) != OpType::KV_CACHE_STATE || OP(p.vc) != OpType::KV_CACHE_STATE)
                return bad("layer anchors");
            for (int w : {p.q_w, p.k_w, p.v_w, p.o_w, p.gate_w, p.up_w, p.down_w, p.in_norm_w, p.ls_w})
                if (OP(w) != OpType::INPUT) return bad("weights");
        }
        if (B(G26_H_IN).total_size != (size_t)HID) return bad("hidden size");
        for (int l = 0; l < 30; ++l) {
            const G26LayerPlan& p = G26_L[l];
            G26Layer& w = M26.L[l];
            auto cq = [&](int idx){ return B(idx).to_cq_matrix(); };
            w.q = cq(p.q_w); w.k = cq(p.k_w); w.v = cq(p.v_w); w.o = cq(p.o_w);
            w.gate = cq(p.gate_w); w.up = cq(p.up_w); w.down = cq(p.down_w);
            if (w.q.bits != 2 || w.q.group_size != 128) return bad("q not cq2/gs128");
            w.in_norm = P(p.in_norm_w); w.q_norm = P(p.q_norm_w); w.k_norm = P(p.k_norm_w);
            w.v_norm = P(p.v_norm_w); w.post_attn = P(p.post_attn_w); w.pre_ffn = P(p.pre_ffn_w);
            w.post_ffn = P(p.post_ffn_w); w.post_moe = P(p.post_moe_w); w.post_moe2 = P(p.post_moe2_w);
            w.router_w = P(p.router_w);
            if (B(p.router_w).precision != Precision::FP16) return bad("router prec");
            if (B(p.ls_w).precision != Precision::FP16) return bad("ls prec");
            w.ls = (float)*(const __fp16*)P(p.ls_w);
            w.hd = p.hd; w.win = p.win; w.pad_n = p.pad_n;
            // folded norm weights (constant): router_norm * IN_SCALE * 16, moe_h_norm * 16
            if (B(p.router_norm_w).precision != Precision::FP16 ||
                B(p.moe_h_norm_w).precision != Precision::FP16) return bad("norm prec");
            w.router_norm_f = cactus_metal_alloc_shared((size_t)HID*2);
            w.moe_h_norm_f  = cactus_metal_alloc_shared((size_t)HID*2);
            const __fp16* rn = (const __fp16*)P(p.router_norm_w);
            const __fp16* mn = (const __fp16*)P(p.moe_h_norm_w);
            __fp16* rf = (__fp16*)w.router_norm_f; __fp16* mf = (__fp16*)w.moe_h_norm_f;
            for (int i = 0; i < HID; ++i) {
                rf[i] = (__fp16)((float)rn[i] * G26_ROUTER_IN_SCALE * 16.0f);
                mf[i] = (__fp16)((float)mn[i] * 16.0f);
            }
            const void* ro[] = { w.in_norm, w.q_norm, w.k_norm, w.v_norm, w.post_attn,
                                 w.pre_ffn, w.post_ffn, w.post_moe, w.post_moe2, w.router_w };
            size_t rs[] = { (size_t)HID*2, (size_t)w.hd*2, (size_t)w.hd*2, (size_t)w.hd*2, (size_t)HID*2,
                            (size_t)HID*2, (size_t)HID*2, (size_t)HID*2, (size_t)HID*2, (size_t)128*HID*2 };
            for (int i = 0; i < 10; ++i) if (ro[i]) cactus_metal_register_readonly(ro[i], rs[i]);
        }
        M26.cosL = P(G26_COS_L); M26.sinL = P(G26_SIN_L);
        M26.cosG = P(G26_COS_G); M26.sinG = P(G26_SIN_G);
        M26.lm = B(G26_LM_W).to_cq_matrix();
        if (!(M26.lm.flags & CACTUS_QUANT_FLAG_ORTHOGONAL)) return bad("lm not ortho");
        size_t V = B(G26_OUT).total_size;
        auto sh = [&](size_t bytes){ void* pp = cactus_metal_alloc_shared(bytes); if (pp) std::memset(pp, 0, bytes); return pp; };
        M26.h = sh(HID*2); M26.xn = sh(HID*2);
        M26.q = sh((size_t)NQH*512*2); M26.k = sh((size_t)NKVH*512*2); M26.v = sh((size_t)NKVH*512*2);
        M26.qr = sh((size_t)NQH*512*2); M26.kr = sh((size_t)NKVH*512*2); M26.vn = sh((size_t)NKVH*512*2);
        M26.attn_o = sh((size_t)NQH*512*2); M26.o = sh(HID*2); M26.h1 = sh(HID*2);
        M26.xf = sh(HID*2); M26.r_in = sh(HID*2); M26.moe_h = sh(HID*2);
        M26.gate = sh(2112*2); M26.up = sh(2112*2); M26.gu_pad = sh(2176*2);
        M26.lg16 = sh(128*2); M26.probs = sh(128*2); M26.topk = sh(16*sizeof(float));
        M26.dn = sh(HID*2); M26.moe_out = sh(HID*2); M26.f_rms = sh(HID*2); M26.m_rms = sh(HID*2);
        M26.s1 = sh(HID*2); M26.fn = sh(HID*2); M26.lg = sh(V*2);
        M26.cos_s = sh(256*2); M26.sin_s = sh(256*2); M26.cos_g = sh(512*2); M26.sin_g = sh(512*2);
        M26.argmax_buf = sh(3*sizeof(float));
        M26.init = true;
    }

    // caches must exist (first decode token goes through the node loop)
    for (int l = 0; l < 30; ++l)
        if (!P(G26_L[l].kc) || !P(G26_L[l].vc)) return false;

    const float* posp = (const float*)P(G26_POS_IN);
    const void* hin = P(G26_H_IN);
    if (!posp || !hin) return false;
    int pos = (int)posp[0];

    // wire delegated nodes' inputs/outputs to our persistent buffers
    // (cheap; re-done every call because the node loop may reassign them)
    auto ext = [&](size_t node_id, void* buf) {
        auto it = node_index_map_.find(node_id);
        if (it == node_index_map_.end()) return false;
        nodes_[it->second]->output_buffer.set_external(buf);
        return true;
    };
    for (int l = 0; l < 30; ++l) {
        const G26LayerPlan& p = G26_L[l];
        GraphNode& an = *nodes_[p.attn];
        if (!ext(an.input_ids[0], M26.qr) || !ext(an.input_ids[1], M26.kr) || !ext(an.input_ids[2], M26.vn)) return false;
        B(p.attn).set_external(M26.attn_o);
        GraphNode& ka = *nodes_[p.k_app];
        GraphNode& va = *nodes_[p.v_app];
        if (!ext(ka.input_ids[0], M26.kr) || !ext(va.input_ids[0], M26.vn)) return false;
        static float appout[2];
        B(p.k_app).set_external(&appout[0]);
        B(p.v_app).set_external(&appout[1]);
        GraphNode& mo = *nodes_[p.moe];
        if (!ext(mo.input_ids[0], M26.moe_h) || !ext(mo.input_ids[1], M26.probs) || !ext(mo.input_ids[2], M26.topk)) return false;
        B(p.moe).set_external(M26.moe_out);
    }
    B(G26_OUT).set_external(M26.lg);

    cactus_metal_session_begin();
    cactus_metal_set_active(true);
    cactus_metal_manual_begin();
    auto barrier = [&](){ cactus_metal_barrier(); };
    auto bail = [&](){
        cactus_metal_manual_end(); cactus_metal_set_active(false); cactus_metal_session_end();
        M26.bad = true;
        return false;
    };

    std::memcpy(M26.h, hin, HID*2);
    std::memcpy(M26.cos_s, (const char*)M26.cosL + (size_t)pos*256*2, 256*2);
    std::memcpy(M26.sin_s, (const char*)M26.sinL + (size_t)pos*256*2, 256*2);
    std::memcpy(M26.cos_g, (const char*)M26.cosG + (size_t)pos*512*2, 512*2);
    std::memcpy(M26.sin_g, (const char*)M26.sinG + (size_t)pos*512*2, 512*2);

    void* h = M26.h;
    size_t V = B(G26_OUT).total_size;

    for (int l = 0; l < 30; ++l) {
        const G26LayerPlan& p = G26_L[l];
        G26Layer& w = M26.L[l];
        const void* rc = (w.hd == 512) ? M26.cos_g : M26.cos_s;
        const void* rs = (w.hd == 512) ? M26.sin_g : M26.sin_s;

        // S1: pre-attention norm
        cactus_metal_encode_rms_norm(M26.xn, h, w.in_norm, 1, HID, 1e-6f);
        barrier();
        // S2: q/k/v projections (single-dispatch cq2 each, concurrent)
        cactus_metal_encode_quant_matmul(M26.q, M26.xn, &w.q);
        cactus_metal_encode_quant_matmul(M26.k, M26.xn, &w.k);
        cactus_metal_encode_quant_matmul(M26.v, M26.xn, &w.v);
        barrier();
        // S3: per-head norms + rope
        cactus_metal_encode_rms_rope(M26.qr, M26.q, w.q_norm, rc, rs, NQH, w.hd, 1e-6f);
        cactus_metal_encode_rms_rope(M26.kr, M26.k, w.k_norm, rc, rs, NKVH, w.hd, 1e-6f);
        cactus_metal_encode_rms_norm(M26.vn, M26.v, w.v_norm, NKVH, w.hd, 1e-6f);
        barrier();
        // S4: kv appends (update cache metadata on CPU), then attention
        if (!encode_node_gpu(p.k_app) || !encode_node_gpu(p.v_app)) return bail();
        barrier();
        if (!encode_node_gpu(p.attn)) return bail();
        barrier();
        // S5: output projection
        cactus_metal_encode_quant_matmul(M26.o, M26.attn_o, &w.o);
        barrier();
        // S6: h1 = clip(h + rms(o)), with the dense-ffn pre-norm fused in
        cactus_metal_encode_rms_norm_add_rms(M26.h1, M26.xf, M26.o, w.post_attn, h, w.pre_ffn, 1, HID, 1e-6f, 1.0f);
        barrier();
        // S7: router + moe-hidden norms of h1 (scales folded into weights)
        cactus_metal_encode_rms_norm(M26.r_in, M26.h1, w.router_norm_f, 1, HID, 1e-6f);
        cactus_metal_encode_rms_norm(M26.moe_h, M26.h1, w.moe_h_norm_f, 1, HID, 1e-6f);
        barrier();
        // S8: dense gate/up + router logits (x16 folded into r_in)
        cactus_metal_encode_quant_matmul(M26.gate, M26.xf, &w.gate);
        cactus_metal_encode_quant_matmul(M26.up, M26.xf, &w.up);
        cactus_metal_encode_matmul_f16_gemv(M26.lg16, M26.r_in, w.router_w, HID, 128);
        barrier();
        // S9: swiglu into the zero-padded down input; router softmax + topk
        cactus_metal_encode_swiglu(M26.gu_pad, M26.gate, M26.up, 2112, 1.0f);
        cactus_metal_encode_softmax_rows(M26.probs, M26.lg16, 1, 128);
        cactus_metal_encode_topk_row(M26.topk, M26.lg16, 128, 8);
        barrier();
        // S10: dense down projection + the whole MoE block
        cactus_metal_encode_quant_matmul(M26.dn, M26.gu_pad, &w.down);
        if (!encode_node_gpu(p.moe)) return bail();
        barrier();
        // S11: s1 = clip( rms(dn)*post_ffn + rms(moe_out)*post_moe )  (one dispatch)
        cactus_metal_encode_rms2_add_clip(M26.s1, M26.dn, w.post_ffn, M26.moe_out, w.post_moe, HID, 1e-6f);
        barrier();
        // S13: h = clip(h1 + rms(s1)) * layer_scalar   (single fused dispatch)
        void* hn = (h == M26.h) ? M26.h1 : M26.h;  // ping-pong? keep distinct from h1!
        hn = M26.xn;  // xn is free after S2; reuse as next-h
        cactus_metal_encode_rms_norm_add(hn, M26.s1, w.post_moe2, M26.h1, 1, HID, 1e-6f, w.ls);
        h = hn;
        // swap roles: next layer writes xn — use M26.h as the scratch instead
        { void* t = M26.h; M26.h = M26.xn; M26.xn = t; }
        barrier();
        if (l % 6 == 5) cactus_metal_session_flush();
    }

    // final norm + lm head + softcap + argmax
    cactus_metal_encode_rms_norm(M26.fn, h, P(G26_FINAL_NORM_W), 1, HID, 1e-6f);
    barrier();
    {
        static void* code = nullptr;
        if (!code) code = cactus_metal_alloc_shared((size_t)HID*2);
        if (!cactus_metal_encode_quant_matmul_ortho(M26.lg, M26.fn, code, &M26.lm)) return bail();
    }
    barrier();
    cactus_metal_encode_softcap(M26.lg, M26.lg, V, G26_SOFTCAP);
    barrier();
    if (cactus_metal_encode_argmax(M26.lg, (uint32_t)V, M26.argmax_buf)) {
        g_gpu_argmax_buf = (const float*)M26.argmax_buf;
        g_gpu_argmax_valid = true;
    }

    cactus_metal_manual_end();
    cactus_metal_set_active(false);
    cactus_metal_session_end();
    return true;
}
