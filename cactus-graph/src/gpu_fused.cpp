

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

bool g_cactus_force_cpu = false;

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
    static const bool dbg = std::getenv("CACTUS_FUSED_DEBUG") != nullptr;
    if (!cactus_metal_available()) return false;
    if (g_cactus_force_cpu) return false;
    const size_t n = nodes_.size();
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
        if (G.L[l].global && km[0] + 1 > km[1]) {
            static const bool dbg2 = (std::getenv("CACTUS_FUSED_TIMER") != nullptr);
            if (dbg2) { static int n=0; if (n++<3) fprintf(stderr,"[fused-bail] L%d clen=%llu max=%llu hd=%d\n",
                l,(unsigned long long)km[0],(unsigned long long)km[1],G.L[l].hd); }
            return false;
        }
    }

    G.arena.reset();
    auto fresh = [&](size_t elems){ return G.arena.fresh(elems * 2); };
    cactus_metal_session_begin();
    auto g_t0 = std::chrono::high_resolution_clock::now();
    cactus_metal_set_active(true);

    void* h = fresh(HID);
    const void* pleBase;
    if (fold) {
        auto bail = [&](){ cactus_metal_set_active(false); cactus_metal_session_end(); return false; };
        const uint32_t tok = (uint32_t)g_fe.token_id;
        const int PK = (int)g_fe.proj.N;

        void* hr = fresh(HID);
        if (!cactus_metal_encode_embedding_ortho(hr, tok, &G.lm_head)) return bail();
        cactus_metal_encode_scalar(2, h, hr, HID, g_fe.emb_scale);

        void* pe = fresh(g_fe.ple.K);
        if (!cactus_metal_encode_embedding_hadamard(pe, tok, &g_fe.ple)) return bail();
        void* pa = fresh(PK); cactus_metal_encode_scalar(2, pa, pe, PK, g_fe.ple_scale);
        void* pj = fresh(PK);
        if (!cactus_metal_encode_quant_matmul(pj, h, &g_fe.proj)) return bail();
        void* pjs= fresh(PK); cactus_metal_encode_scalar(2, pjs, pj, PK, g_fe.proj_scale);
        void* psum=fresh(PK); cactus_metal_encode_rms_norm_add(psum, pjs, g_fe.rms_weight, pa, NL, PLE_DIM, g_fe.rms_eps);
        void* plt= fresh(PK); cactus_metal_encode_scalar(2, plt, psum, PK, g_fe.final_scale);
        pleBase = plt;
    } else {
        std::memcpy(h, G.hidden, HID*2);
        pleBase = G.ple;
    }
    if (dbg) { cactus_metal_session_sync(); const __fp16* hh=(const __fp16*)h; const __fp16* pp=(const __fp16*)pleBase;
        fprintf(stderr,"[embed %s] tok=%d h[0..3]= %.5f %.5f %.5f %.5f | ple[0..3]= %.5f %.5f %.5f %.5f\n",
            fold?"FOLD":"node", fold?g_fe.token_id:-1, (float)hh[0],(float)hh[1],(float)hh[2],(float)hh[3],
            (float)pp[0],(float)pp[1],(float)pp[2],(float)pp[3]); }

    auto gather = [&](const void* tbl, void* dst, int hd){ std::memcpy(dst, (const char*)tbl + (size_t)pos*hd*2, hd*2); };
    gather(G.cosL, G.cos_s, 256); gather(G.sinL, G.sin_s, 256);
    gather(G.cosG, G.cos_g, 512); gather(G.sinG, G.sin_g, 512);

    for (int l = 0; l < NL; ++l) {

        LayerW& w = G.L[l];
        const int hd = w.hd, QD = NQH*hd;
        const void* rc = w.global ? G.cos_g : G.cos_s;
        const void* rs = w.global ? G.sin_g : G.sin_s;
        uint64_t* km = (uint64_t*)w.kc; uint64_t* vm = (uint64_t*)w.vc;
        size_t mx = km[1], ng = (hd+31)/32;

        void* xn = fresh(HID);  cactus_metal_encode_rms_norm(xn, h, w.in_norm, 1, HID, RMS_EPS);
        void* q  = fresh(QD);   cactus_metal_encode_quant_matmul(q, xn, &w.q);
        void* qn = fresh(QD);   cactus_metal_encode_rms_norm(qn, q, w.q_norm, NQH, hd, RMS_EPS);
        void* qr = fresh(QD);   cactus_metal_encode_rope(qr, qn, rc, rs, NQH, hd);

        size_t hist, total; const void *knewp, *vnewp; size_t kv_start = 0, kv_end;
        const uint32_t Wn = w.global ? 0u : (uint32_t)(km[1] - km[4] - 1);
        const uint32_t Sn = w.global ? 0u : (uint32_t)km[4];
        const uint32_t Rn = (Wn > Sn) ? (Wn - Sn) : 1u;
        if (!w.shared) {
            void* k  = fresh(hd); cactus_metal_encode_quant_matmul(k, xn, &w.k);
            void* v  = fresh(hd); cactus_metal_encode_quant_matmul(v, xn, &w.v);
            void* kn = fresh(hd); cactus_metal_encode_rms_norm(kn, k, w.k_norm, 1, hd, RMS_EPS);
            void* vn = fresh(hd); cactus_metal_encode_rms_norm(vn, v, w.global ? G.vnorm_g : G.vnorm, 1, hd, RMS_EPS);
            void* kr = fresh(hd); cactus_metal_encode_rope(kr, kn, rc, rs, 1, hd);
            size_t clen = km[0];
            size_t slot = (!w.global && clen >= (size_t)Wn) ? (size_t)(Sn + ((clen - Sn) % Rn)) : clen;
            cactus_metal_encode_kv_append_i8(kr, (char*)w.kc+64, (char*)w.kc+64+mx*hd,
                1, hd, (uint32_t)slot, 32, hd*2, mx*hd, mx*ng*sizeof(float));
            cactus_metal_encode_kv_append_i8(vn, (char*)w.vc+64, (char*)w.vc+64+mx*hd,
                1, hd, (uint32_t)slot, 32, hd*2, mx*hd, mx*ng*sizeof(float));
            km[0] = clen + 1; vm[0] = clen + 1;
            if (!w.global && clen >= (size_t)Wn) {
                hist = Wn; total = Wn; knewp = nullptr; vnewp = nullptr; kv_end = Wn;
            } else {
                hist = clen; total = clen + 1; knewp = kr; vnewp = vn;
                kv_end = std::min(total, (size_t)pos + 1);
            }
        } else {
            size_t clen = km[0];
            if (!w.global && clen > (size_t)Wn) {
                hist = Wn; total = Wn; knewp = nullptr; vnewp = nullptr; kv_end = Wn;
            } else {
                hist = clen; total = clen; knewp = qr; vnewp = qr;
                kv_end = std::min(clen, (size_t)pos + 1);
            }
        }
        void* attn = fresh(QD);
        cactus_metal_encode_attention_i8(attn, qr, knewp, vnewp,
            (char*)w.kc+64, (char*)w.vc+64, (char*)w.kc+64+mx*hd, (char*)w.vc+64+mx*hd,
            NQH, 1, hd, hd, (uint32_t)hist, (uint32_t)total, (uint32_t)kv_start, (uint32_t)kv_end, 1.0f,
            hist*hd, hist*hd, hist*ng*sizeof(float), hist*ng*sizeof(float));
        void* o  = fresh(HID);  cactus_metal_encode_quant_matmul(o, attn, &w.o);
        void* h1 = fresh(HID);  cactus_metal_encode_rms_norm_add(h1, o, w.post_attn, h, 1, HID, RMS_EPS); h = h1;

        const int M = w.mlp;
        void* xn2 = fresh(HID); cactus_metal_encode_rms_norm(xn2, h, w.pre_ffn, 1, HID, RMS_EPS);
        void* gate= fresh(M);   cactus_metal_encode_quant_matmul(gate, xn2, &w.gate);
        void* up  = fresh(M);   cactus_metal_encode_quant_matmul(up, xn2, &w.up);
        void* g3  = fresh(M);   cactus_metal_encode_swiglu(g3, gate, up, M, GATE_SCALE);
        void* mo  = fresh(HID); cactus_metal_encode_quant_matmul(mo, g3, &w.down);
        void* h2  = fresh(HID); cactus_metal_encode_rms_norm_add(h2, mo, w.post_ffn, h, 1, HID, RMS_EPS); h = h2;

        void* ps  = fresh(PLE_DIM); cactus_metal_encode_quant_matmul(ps, h, &w.ple_down);
        void* pm  = fresh(PLE_DIM); cactus_metal_encode_swiglu(pm, ps, (const char*)pleBase + (size_t)l*PLE_DIM*2, PLE_DIM, 1.0f);
        void* pu  = fresh(HID); cactus_metal_encode_quant_matmul(pu, pm, &w.ple_up);
        void* h3  = fresh(HID); cactus_metal_encode_rms_norm_add(h3, pu, w.post_ple, h, 1, HID, RMS_EPS); h = h3;

        float ls = (float)(*(const __fp16*)w.scalar);
        void* h4 = fresh(HID); cactus_metal_encode_scalar(2, h4, h, HID, ls); h = h4;

        if (dbg) { cactus_metal_session_sync(); const __fp16* p=(const __fp16*)h;
            fprintf(stderr,"[fused] L%d out: %.5f %.5f %.5f %.5f (ls=%.6f)\n",l,(float)p[0],(float)p[1],(float)p[2],(float)p[3],ls); }
    }

    size_t V = B(3860).total_size;
    void* fn   = fresh(HID); cactus_metal_encode_rms_norm(fn, h, G.final_norm, 1, HID, RMS_EPS);
    void* code = fresh(HID);
    void* lg   = fresh(V);
    if (cactus_metal_encode_quant_matmul_ortho(lg, fn, code, &G.lm_head)) {
        cactus_metal_encode_scalar(3, lg, lg, V, SOFTCAP);
        cactus_metal_encode_unary(1, lg, lg, V);
        cactus_metal_encode_scalar(2, lg, lg, V, SOFTCAP);
        B(3860).set_external(lg);
        if (G.argmax_buf && cactus_metal_encode_argmax(lg, (uint32_t)V, G.argmax_buf)) {
            g_gpu_argmax_buf = (const float*)G.argmax_buf;
            g_gpu_argmax_valid = true;
        }
    } else {
        cactus_metal_session_sync();
        cactus_quant_matmul(&G.lm_head, (const __fp16*)fn, 1u, (__fp16*)lg);
        __fp16* L=(__fp16*)lg; for (size_t i=0;i<V;++i){ float v=(float)L[i]; L[i]=(__fp16)(SOFTCAP*std::tanh(v/SOFTCAP)); }
        B(3860).set_external(lg);
    }

    cactus_metal_set_active(false);
    static const bool g_tmr = (std::getenv("CACTUS_FUSED_TIMER") != nullptr);
    if (g_tmr) {
        auto t1 = std::chrono::high_resolution_clock::now();
        cactus_metal_session_end();
        auto t2 = std::chrono::high_resolution_clock::now();
        double enc = std::chrono::duration<double,std::milli>(t1 - g_t0).count();
        double syn = std::chrono::duration<double,std::milli>(t2 - t1).count();
        fprintf(stderr, "[fused-timer] pos=%d encode_cpu=%.2fms commit+wait=%.2fms total=%.2fms\n",
                pos, enc, syn, enc+syn);
    } else {
        cactus_metal_session_end();
    }
    return true;
}
