#include "../cactus_graph.h"
#include "cactus_kernels.h"
#include "metal_backend.h"
#include <cmath>
#include <cstring>
#include <algorithm>

bool cactus_kv_cache_grow(BufferDesc&, size_t, size_t);

namespace {

bool is_alias_op(OpType op) {
    return op == OpType::VIEW || op == OpType::RESHAPE || op == OpType::FLATTEN;
}

bool is_noop_transpose(const GraphNode& nd) {
    if (nd.op_type != OpType::TRANSPOSE) return false;
    size_t real = 0;
    for (size_t d : nd.output_buffer.shape) if (d != 1) ++real;
    return real <= 1;
}

bool is_same_cast(const GraphNode& nd, const std::vector<std::unique_ptr<GraphNode>>& nodes,
                  const std::unordered_map<size_t, size_t>& map) {
    if (nd.op_type != OpType::PRECISION_CAST || nd.input_ids.empty()) return false;
    auto it = map.find(nd.input_ids[0]);
    return it != map.end() && nodes[it->second]->output_buffer.precision == nd.output_buffer.precision;
}

}

struct MetalCluster {
    int rule = 0;
    size_t a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0;
    size_t b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0;
    float f0 = 0.0f, f1 = 0.0f;
    uint32_t u0 = 0, u1 = 0;
    void* s0 = nullptr;
    void* s1 = nullptr;
    void* sc[3] = {nullptr, nullptr, nullptr};
};

struct MetalFusePlan {
    std::vector<int32_t> action;
    std::vector<MetalCluster> clusters;
    long fold_h = -1, fold_ple = -1, fold_pos = -1, fold_w = -1;
    size_t fold_nl = 0, fold_pd = 0;
    std::vector<uint32_t> exec_list;
    std::vector<long> arena_off;
    char* arena_base = nullptr;
    std::vector<std::vector<float>> blobs;
};

struct EwChainStep {
    int32_t kind;
    int32_t code;
    float p0;
    float p1;
};

MetalFusePlan* cactus_metal_plan_build(
    const std::vector<std::unique_ptr<GraphNode>>& nodes,
    const std::unordered_map<size_t, size_t>& map,
    const std::unordered_set<size_t>& pinned_ids,
    const std::vector<uint8_t>& retype,
    const std::unordered_set<size_t>* banned) {
    const size_t n = nodes.size();
    auto plan = new MetalFusePlan();
    plan->action.assign(n, -1);
    std::vector<uint8_t> pinned(n, 0);
    for (size_t i = 0; i < n; ++i)
        if (pinned_ids.count(nodes[i]->id)) pinned[i] = 1;
    std::vector<uint8_t> cpu_only(n, 0);
    for (size_t i = 0; i < n; ++i)
        if (nodes[i]->params.backend != ComputeBackend::METAL) { cpu_only[i] = 1; pinned[i] = 1; }

    auto idxof = [&](size_t id) -> long {
        auto it = map.find(id);
        return it == map.end() ? -1 : (long)it->second;
    };
    std::vector<std::vector<size_t>> cons(n);
    for (size_t i = 0; i < n; ++i)
        for (size_t id : nodes[i]->input_ids) { long j = idxof(id); if (j >= 0) cons[(size_t)j].push_back(i); }

    auto retyped = [&](size_t idx) { return idx < retype.size() && retype[idx] != 0; };
    auto passthrough = [&](const GraphNode& nd) {
        return is_alias_op(nd.op_type) || is_same_cast(nd, nodes, map) || is_noop_transpose(nd);
    };
    auto up = [&](long i) -> long {
        while (i >= 0) {
            const GraphNode& nd = *nodes[(size_t)i];
            if (passthrough(nd) && !nd.input_ids.empty()) {
                i = idxof(nd.input_ids[0]);
                continue;
            }
            return i;
        }
        return -1;
    };
    auto up_in = [&](size_t i, size_t k) -> long {
        if (k >= nodes[i]->input_ids.size()) return -1;
        return up(idxof(nodes[i]->input_ids[k]));
    };
    auto deep_f16_source = [&](long i) -> long {
        long cur = i;
        while (cur >= 0) {
            const GraphNode& nd = *nodes[(size_t)cur];
            if ((passthrough(nd) || nd.op_type == OpType::PRECISION_CAST) && !nd.input_ids.empty()) {
                cur = idxof(nd.input_ids[0]);
                continue;
            }
            break;
        }
        if (cur >= 0 && nodes[(size_t)cur]->output_buffer.precision == Precision::FP16) return cur;
        return i;
    };
    auto collect_chain = [&](size_t from, long to, std::vector<size_t>& out_chain) {
        long i = (long)from;
        while (i >= 0 && i != to) {
            const GraphNode& nd = *nodes[(size_t)i];
            if (!passthrough(nd)) break;
            out_chain.push_back((size_t)i);
            i = nd.input_ids.empty() ? -1 : idxof(nd.input_ids[0]);
        }
    };
    auto sole_use = [&](long i, const std::vector<size_t>& allowed) -> bool {
        if (i < 0) return false;
        for (size_t c : cons[(size_t)i]) {
            bool ok = false;
            for (size_t a : allowed) if (a == c) { ok = true; break; }
            if (!ok) {
                const GraphNode& cn = *nodes[c];
                if (is_alias_op(cn.op_type) || is_same_cast(cn, nodes, map) || is_noop_transpose(cn)) {
                    bool subok = true;
                    for (size_t cc : cons[c]) {
                        bool found = false;
                        for (size_t a : allowed) if (a == cc) { found = true; break; }
                        if (!found) { subok = false; break; }
                    }
                    if (!subok) return false;
                } else return false;
            }
        }
        return true;
    };

    auto match_rope = [&](long add_i, long* x_out, long* cos_out, long* sin_out,
                          std::vector<size_t>& cover) -> bool {
        if (add_i < 0 || nodes[(size_t)add_i]->op_type != OpType::ADD) return false;
        long m0 = up_in((size_t)add_i, 0), m1 = up_in((size_t)add_i, 1);
        if (m0 < 0 || m1 < 0) return false;
        if (nodes[(size_t)m0]->op_type != OpType::MULTIPLY) std::swap(m0, m1);
        if (m0 < 0 || m1 < 0) return false;
        if (nodes[(size_t)m0]->op_type != OpType::MULTIPLY ||
            nodes[(size_t)m1]->op_type != OpType::MULTIPLY) return false;
        long rot = up_in((size_t)m1, 0);
        long sinb = up_in((size_t)m1, 1);
        if (rot < 0 || nodes[(size_t)rot]->op_type != OpType::CAT) { std::swap(rot, sinb); }
        if (rot < 0 || sinb < 0 || nodes[(size_t)rot]->op_type != OpType::CAT) {
            std::swap(m0, m1);
            rot = up_in((size_t)m1, 0); sinb = up_in((size_t)m1, 1);
            if (rot < 0 || nodes[(size_t)rot]->op_type != OpType::CAT) { std::swap(rot, sinb); }
            if (rot < 0 || sinb < 0 || nodes[(size_t)rot]->op_type != OpType::CAT) return false;
        }
        long x = up_in((size_t)m0, 0), cosb = up_in((size_t)m0, 1);
        if (x < 0 || cosb < 0) return false;
        if (nodes[(size_t)rot]->input_ids.size() != 2) return false;
        long neg = up_in((size_t)rot, 0), lo = up_in((size_t)rot, 1);
        if (neg < 0 || lo < 0) return false;
        if (nodes[(size_t)neg]->op_type != OpType::SCALAR_MULTIPLY ||
            nodes[(size_t)neg]->params.scalar != -1.0f) return false;
        long hi = up_in((size_t)neg, 0);
        if (hi < 0 || nodes[(size_t)hi]->op_type != OpType::SLICE ||
            nodes[(size_t)lo]->op_type != OpType::SLICE) return false;
        long xs1 = up_in((size_t)hi, 0), xs2 = up_in((size_t)lo, 0);
        if (xs1 != x || xs2 != x) {
            long xa = up_in((size_t)m0, 1);
            if (xs1 == xa && xs2 == xa) { x = xa; cosb = up_in((size_t)m0, 0); }
            else return false;
        }
        *x_out = x; *cos_out = cosb; *sin_out = sinb;
        cover.push_back((size_t)add_i); cover.push_back((size_t)m0); cover.push_back((size_t)m1);
        cover.push_back((size_t)rot); cover.push_back((size_t)neg);
        cover.push_back((size_t)hi); cover.push_back((size_t)lo);
        return true;
    };

    auto alias_base = [&](long i, long* off_out) -> long {
        long off = 0;
        while (i >= 0) {
            const GraphNode& nd = *nodes[(size_t)i];
            if (passthrough(nd) && !nd.input_ids.empty()) {
                i = idxof(nd.input_ids[0]);
                continue;
            }
            if (nd.op_type == OpType::INDEX && nd.params.axis == 0 && !nd.input_ids.empty()) {
                long blk = 1;
                for (size_t d : nd.output_buffer.shape) blk *= (long)d;
                off += (long)nd.params.index_value * blk;
                i = idxof(nd.input_ids[0]);
                continue;
            }
            if (nd.op_type == OpType::SLICE && !nd.input_ids.empty()) {
                long j = idxof(nd.input_ids[0]);
                if (j < 0) break;
                const auto& ish = nodes[(size_t)j]->output_buffer.shape;
                size_t ax = (size_t)nd.params.axis;
                if (ax >= ish.size()) break;
                size_t outer = 1;
                for (size_t d = 0; d < ax; ++d) outer *= ish[d];
                if (outer != 1) break;
                long inner = 1;
                for (size_t d = ax + 1; d < ish.size(); ++d) inner *= (long)ish[d];
                off += (long)nd.params.slice_start * inner;
                i = j;
                continue;
            }
            break;
        }
        *off_out = off;
        return i;
    };

    auto release_scratch = [&](MetalCluster& c) {
        if (c.s0) { cactus_metal_free_shared(c.s0); c.s0 = nullptr; }
        if (c.s1) { cactus_metal_free_shared(c.s1); c.s1 = nullptr; }
        for (auto& sp : c.sc) if (sp) { cactus_metal_free_shared(sp); sp = nullptr; }
    };
    auto add_cluster = [&](MetalCluster c, size_t anchor, const std::vector<size_t>& cover) -> bool {
        if (banned && banned->count(anchor)) { release_scratch(c); return false; }
        if (cpu_only[anchor]) { release_scratch(c); return false; }
        for (size_t v : cover) if (v != anchor && pinned[v]) { release_scratch(c); return false; }
        int32_t cid = (int32_t)plan->clusters.size();
        plan->clusters.push_back(c);
        for (size_t v : cover) if (v != anchor) plan->action[v] = -2;
        plan->action[anchor] = cid;
        return true;
    };

    struct AttnCand {
        MetalCluster c;
        size_t anchor;
        std::vector<size_t> cover;
        size_t kcache;
        long kapp, vapp;
    };
    std::vector<AttnCand> cands;

    for (size_t i = 0; i < n; ++i) {
        if (plan->action[i] != -1) continue;
        GraphNode& nd = *nodes[i];

        if (nd.op_type == OpType::ATTENTION_CACHED && nd.input_ids.size() >= 5) {
            long qidx = idxof(nd.input_ids[0]);
            if (qidx < 0) continue;
            const BufferDesc& qb = nodes[(size_t)qidx]->output_buffer;
            if (qb.shape.size() < 4 || qb.shape[0] != 1 || qb.shape[1] != 1) continue;
            uint32_t nqh = (uint32_t)qb.shape[2], hd = (uint32_t)qb.shape[3];
            long qk = idxof(nd.input_ids[1]);
            long vk = idxof(nd.input_ids[2]);
            long kcache = idxof(nd.input_ids[3]);
            long vcache = idxof(nd.input_ids[4]);
            if (qk < 0 || vk < 0 || kcache < 0 || vcache < 0) continue;
            if (nodes[(size_t)kcache]->params.num_kv_heads != 1) continue;

            std::vector<size_t> cover;
            long q_rope = up_in(i, 0);
            long qx = -1, qcos = -1, qsin = -1;
            if (!match_rope(q_rope, &qx, &qcos, &qsin, cover)) continue;
            long qnorm = qx;
            if (qnorm < 0 || nodes[(size_t)qnorm]->op_type != OpType::RMS_NORM) continue;
            long qraw = up_in((size_t)qnorm, 0);
            long qw = up_in((size_t)qnorm, 1);
            if (qraw < 0 || qw < 0) continue;
            cover.push_back((size_t)qnorm);

            long k_rope = up(qk);
            long kx = -1, kcos = -1, ksin = -1;
            std::vector<size_t> kcover;
            long knorm = -1, kraw = -1, kw = -1, vnorm = -1, vraw = -1, vw = -1;
            long kapp = -1, vapp = -1;
            bool has_new = false;
            if (match_rope(k_rope, &kx, &kcos, &ksin, kcover)) {
                knorm = kx;
                if (knorm >= 0 && nodes[(size_t)knorm]->op_type == OpType::RMS_NORM) {
                    kraw = up_in((size_t)knorm, 0);
                    kw = up_in((size_t)knorm, 1);
                    long vn = up(vk);
                    if (kraw >= 0 && kw >= 0 && vn >= 0 && nodes[(size_t)vn]->op_type == OpType::RMS_NORM) {
                        vnorm = vn;
                        vraw = up_in((size_t)vn, 0);
                        vw = up_in((size_t)vn, 1);
                        for (size_t c = 0; c < n; ++c) {
                            if (nodes[c]->op_type != OpType::KV_CACHE_APPEND || nodes[c]->input_ids.size() < 2) continue;
                            long src = up(idxof(nodes[c]->input_ids[0]));
                            long cache = idxof(nodes[c]->input_ids[1]);
                            if (src == k_rope && cache == kcache) kapp = (long)c;
                            if (src == (long)vnorm && cache == vcache) vapp = (long)c;
                        }
                        if (kapp >= 0 && vapp >= 0 && vraw >= 0 && vw >= 0) has_new = true;
                    }
                }
            }
            bool shared = false;
            if (has_new) {
                for (auto& pc : cands)
                    if (pc.kapp == kapp || pc.kcache == (size_t)kcache) { shared = true; break; }
            }

            AttnCand cand;
            cand.anchor = i;
            cand.kcache = (size_t)kcache;
            cand.kapp = kapp; cand.vapp = vapp;
            MetalCluster& c = cand.c;
            c.rule = 1;
            c.b0 = (size_t)qcos; c.b1 = (size_t)qsin;
            c.b2 = (size_t)kcache; c.b3 = (size_t)vcache;
            c.b4 = i;
            c.a0 = (size_t)qraw; c.a3 = (size_t)deep_f16_source(qw);
            c.f0 = nodes[(size_t)qnorm]->params.epsilon;
            c.f1 = nd.params.scale != 0.0f ? nd.params.scale : 1.0f / std::sqrt((float)hd);
            c.u0 = nqh; c.u1 = hd;
            cand.cover = cover;
            collect_chain((size_t)idxof(nd.input_ids[0]), q_rope, cand.cover);
            if (has_new && !shared) {
                c.a1 = (size_t)kraw; c.a2 = (size_t)vraw;
                c.a4 = (size_t)deep_f16_source(kw); c.a5 = (size_t)deep_f16_source(vw);
                c.u1 |= 0x10000u;
                cand.cover.insert(cand.cover.end(), kcover.begin(), kcover.end());
                cand.cover.push_back((size_t)knorm);
                cand.cover.push_back((size_t)vnorm);
                cand.cover.push_back((size_t)kapp);
                cand.cover.push_back((size_t)vapp);
                collect_chain((size_t)idxof(nd.input_ids[1]), k_rope, cand.cover);
                collect_chain((size_t)idxof(nd.input_ids[2]), vnorm, cand.cover);
                collect_chain((size_t)idxof(nodes[(size_t)kapp]->input_ids[0]), k_rope, cand.cover);
                collect_chain((size_t)idxof(nodes[(size_t)vapp]->input_ids[0]), vnorm, cand.cover);
            } else if (!shared) {
                continue;
            }
            cands.push_back(std::move(cand));
            continue;
        }

        if (nd.op_type == OpType::MATMUL && nd.input_ids.size() >= 2) {
            long r = idxof(nd.input_ids[1]);
            const BufferDesc* rb = r >= 0 ? &nodes[(size_t)r]->output_buffer : nullptr;
            size_t M = 1;
            for (size_t d = 0; d + 1 < nd.output_buffer.shape.size(); ++d) M *= nd.output_buffer.shape[d];
            if (rb && M == 1 && PrecisionTraits::is_cq(rb->precision) && rb->group_size > 0
                && PrecisionTraits::cq_bits(rb->precision) == 4
                && !(rb->cq_flags & CACTUS_QUANT_FLAG_ORTHOGONAL)) {
                long mul = up_in(i, 0);
                if (mul >= 0 && nodes[(size_t)mul]->op_type == OpType::MULTIPLY && sole_use(mul, {i})) {
                    long g0 = up_in((size_t)mul, 0), u0v = up_in((size_t)mul, 1);
                    for (int side = 0; side < 2; ++side) {
                        long gel = side == 0 ? g0 : u0v;
                        long other = side == 0 ? u0v : g0;
                        long sm = -1;
                        float sc_val = 1.0f;
                        if (gel >= 0 && nodes[(size_t)gel]->op_type == OpType::SCALAR_MULTIPLY
                            && sole_use(gel, {(size_t)mul})) {
                            sm = gel;
                            sc_val = nodes[(size_t)gel]->params.scalar;
                            gel = up_in((size_t)gel, 0);
                        }
                        if (gel >= 0 && other >= 0 && nodes[(size_t)gel]->op_type == OpType::GELU
                            && sole_use(gel, {sm >= 0 ? (size_t)sm : (size_t)mul})) {
                            long gate = up_in((size_t)gel, 0);
                            if (gate < 0) break;
                            if (sc_val == 1.0f && nodes[(size_t)gate]->op_type == OpType::MATMUL
                                && nodes[(size_t)gate]->input_ids.size() >= 2
                                && sole_use(gate, {(size_t)gel})) {
                                long gr = idxof(nodes[(size_t)gate]->input_ids[1]);
                                const BufferDesc* grb = gr >= 0 ? &nodes[(size_t)gr]->output_buffer : nullptr;
                                size_t gM = 1;
                                for (size_t d = 0; d + 1 < nodes[(size_t)gate]->output_buffer.shape.size(); ++d)
                                    gM *= nodes[(size_t)gate]->output_buffer.shape[d];
                                if (grb && gM == 1 && PrecisionTraits::is_cq(grb->precision)
                                    && grb->group_size == 128 && PrecisionTraits::cq_bits(grb->precision) == 4
                                    && !(grb->cq_flags & CACTUS_QUANT_FLAG_ORTHOGONAL)
                                    && cactus_metal_transform_gemv_fits((uint32_t)(
                                           grb->shape.size() >= 2 ? grb->shape[1] : grb->shape[0]))) {
                                    long gx = up_in((size_t)gate, 0);
                                    if (gx >= 0) {
                                        MetalCluster c7;
                                        c7.rule = 7;
                                        c7.a0 = (size_t)gx;
                                        c7.a1 = (size_t)other;
                                        c7.b0 = (size_t)gate;
                                        c7.b4 = i;
                                        c7.s0 = cactus_metal_alloc_shared(nodes[(size_t)gate]->output_buffer.total_size * 2);
                                        std::vector<size_t> cover;
                                        cover.push_back((size_t)mul);
                                        cover.push_back((size_t)gel);
                                        cover.push_back((size_t)gate);
                                        collect_chain((size_t)idxof(nd.input_ids[0]), mul, cover);
                                        collect_chain((size_t)idxof(nodes[(size_t)gel]->input_ids[0]), gate, cover);
                                        const GraphNode& rn = *nodes[(size_t)other];
                                        if (rn.op_type == OpType::INDEX && !rn.input_ids.empty()
                                            && sole_use(other, {(size_t)mul})) {
                                            long ib = idxof(rn.input_ids[0]);
                                            if (ib >= 0) {
                                                size_t rowlen = rn.output_buffer.total_size;
                                                c7.a1 = (size_t)ib;
                                                c7.u1 = (uint32_t)(rn.params.index_value * rowlen * 2);
                                                c7.u0 = 1u;
                                                cover.push_back((size_t)other);
                                            }
                                        }
                                        add_cluster(c7, i, cover);
                                        break;
                                    }
                                }
                            }
                            MetalCluster c;
                            c.rule = 4;
                            c.a0 = (size_t)gate;
                            c.a1 = (size_t)other;
                            c.b4 = i;
                            c.f0 = sc_val;
                            const BufferDesc& gb = nodes[(size_t)gate]->output_buffer;
                            c.s0 = cactus_metal_alloc_shared(gb.total_size * 2);
                            std::vector<size_t> cover;
                            cover.push_back((size_t)mul);
                            cover.push_back((size_t)gel);
                            if (sm >= 0) cover.push_back((size_t)sm);
                            collect_chain((size_t)idxof(nd.input_ids[0]), mul, cover);
                            add_cluster(c, i, cover);
                            break;
                        }
                    }
                    if (plan->action[i] >= 0) continue;
                }
            }
        }

        if (nd.op_type == OpType::SCALAR_MULTIPLY && !nd.output_buffer.shape.empty()
            && nd.output_buffer.shape.back() >= 32768
            && nd.output_buffer.total_size == nd.output_buffer.shape.back()) {
            long th = up_in(i, 0);
            if (th >= 0 && nodes[(size_t)th]->op_type == OpType::TANH) {
                long dv = up_in((size_t)th, 0);
                if (dv >= 0 && nodes[(size_t)dv]->op_type == OpType::SCALAR_DIVIDE
                    && nodes[(size_t)dv]->params.scalar == nd.params.scalar) {
                    long mm = up_in((size_t)dv, 0);
                    if (mm >= 0 && nodes[(size_t)mm]->op_type == OpType::MATMUL
                        && nodes[(size_t)mm]->input_ids.size() >= 2) {
                        long wnode = idxof(nodes[(size_t)mm]->input_ids[1]);
                        long xn = up_in((size_t)mm, 0);
                        if (wnode >= 0 && xn >= 0) {
                            const BufferDesc& wb = nodes[(size_t)wnode]->output_buffer;
                            if (PrecisionTraits::is_cq(wb.precision) && wb.group_size > 0
                                && (wb.cq_flags & CACTUS_QUANT_FLAG_ORTHOGONAL)
                                && sole_use(mm, {(size_t)dv}) && sole_use(dv, {(size_t)th})
                                && sole_use(th, {i})) {
                                MetalCluster c;
                                c.rule = 5;
                                c.a0 = (size_t)xn; c.a1 = (size_t)wnode; c.b4 = i;
                                c.f0 = nd.params.scalar;
                                c.s0 = cactus_metal_alloc_shared(wb.shape.empty() ? 0 : wb.to_cq_matrix().K * 2);
                                std::vector<size_t> cover;
                                cover.push_back((size_t)mm);
                                cover.push_back((size_t)dv);
                                cover.push_back((size_t)th);
                                collect_chain((size_t)idxof(nodes[(size_t)dv]->input_ids[0]), mm, cover);
                                collect_chain((size_t)idxof(nd.input_ids[0]), th, cover);
                                add_cluster(c, i, cover);
                                continue;
                            }
                        }
                    }
                }
            }
        }

        if (nd.op_type == OpType::CAT && nd.input_ids.size() >= 4 && !nd.output_buffer.shape.empty()) {
            size_t cnt = nd.input_ids.size();
            bool blocky = true;
            {
                size_t ax = (size_t)nd.params.axis;
                const auto& osh = nd.output_buffer.shape;
                if (retyped(i)) blocky = false;
                if (ax >= osh.size()) blocky = false;
                for (size_t d = 0; blocky && d < ax; ++d) if (osh[d] != 1) blocky = false;
                long fi = idxof(nd.input_ids[0]);
                if (blocky && fi >= 0) {
                    const auto& ish = nodes[(size_t)fi]->output_buffer.shape;
                    size_t ip = 1, op2 = 1;
                    for (size_t d : ish) ip *= d;
                    for (size_t d : osh) op2 *= d;
                    if (ip * cnt != op2) blocky = false;
                } else blocky = false;
            }
            if (blocky) {
            long m0 = -1;
            {
                long t = up_in(i, 0);
                while (t >= 0 && nodes[(size_t)t]->op_type == OpType::PRECISION_CAST
                       && !nodes[(size_t)t]->input_ids.empty())
                    t = up(idxof(nodes[(size_t)t]->input_ids[0]));
                m0 = t;
            }
            if (m0 >= 0 && nodes[(size_t)m0]->op_type == OpType::MATMUL
                && nodes[(size_t)m0]->output_buffer.shape.size() == 2
                && nodes[(size_t)m0]->output_buffer.precision == Precision::FP16) {
                size_t M = nodes[(size_t)m0]->output_buffer.shape[0];
                size_t N = nodes[(size_t)m0]->output_buffer.shape[1];
                long ka = idxof(nodes[(size_t)m0]->input_ids[0]);
                size_t K = ka >= 0 ? nodes[(size_t)ka]->output_buffer.shape.back() : 0;
                bool f32out = nd.output_buffer.precision == Precision::FP32;
                long parentA = -1, parentB = -1, offA0 = 0, offB0 = 0;
                bool ok = K > 0;
                bool f32a = false;
                std::vector<size_t> cover;
                for (size_t j = 0; j < cnt && ok; ++j) {
                    long t = idxof(nd.input_ids[j]);
                    std::vector<size_t> chain;
                    while (t >= 0 && (passthrough(*nodes[(size_t)t])
                           || (nodes[(size_t)t]->op_type == OpType::PRECISION_CAST))) {
                        if (cons[(size_t)t].size() != 1) { t = -1; break; }
                        chain.push_back((size_t)t);
                        t = idxof(nodes[(size_t)t]->input_ids[0]);
                    }
                    if (t < 0 || nodes[(size_t)t]->op_type != OpType::MATMUL
                        || plan->action[(size_t)t] != -1
                        || cons[(size_t)t].size() != 1
                        || nodes[(size_t)t]->output_buffer.shape.size() != 2
                        || nodes[(size_t)t]->output_buffer.shape[0] != M
                        || nodes[(size_t)t]->output_buffer.shape[1] != N
                        || nodes[(size_t)t]->params.pretransposed_rhs) { ok = false; break; }
                    long offA, offB;
                    long lin = idxof(nodes[(size_t)t]->input_ids[0]);
                    bool jf32 = false;
                    long lcast = -1;
                    if (lin >= 0 && nodes[(size_t)lin]->op_type == OpType::PRECISION_CAST
                        && nodes[(size_t)lin]->output_buffer.precision == Precision::FP16
                        && cons[(size_t)lin].size() == 1
                        && !nodes[(size_t)lin]->input_ids.empty()) {
                        long ci = idxof(nodes[(size_t)lin]->input_ids[0]);
                        if (ci >= 0 && nodes[(size_t)ci]->output_buffer.precision == Precision::FP32) {
                            jf32 = true; lcast = lin; lin = ci;
                        }
                    }
                    long pa = alias_base(lin, &offA);
                    long pb = alias_base(idxof(nodes[(size_t)t]->input_ids[1]), &offB);
                    Precision wantA = jf32 ? Precision::FP32 : Precision::FP16;
                    if (pa < 0 || pb < 0
                        || nodes[(size_t)pa]->output_buffer.precision != wantA
                        || (jf32 && retyped((size_t)pa))
                        || nodes[(size_t)pb]->output_buffer.precision != Precision::FP16) { ok = false; break; }
                    if (j == 0) { parentA = pa; parentB = pb; offA0 = offA; offB0 = offB; f32a = jf32; }
                    else if (jf32 != f32a) { ok = false; break; }
                    else if (pa != parentA || pb != parentB
                             || offA != offA0 + (long)(j * M * K)
                             || offB != offB0 + (long)(j * K * N)) { ok = false; break; }
                    cover.push_back((size_t)t);
                    if (lcast >= 0) cover.push_back((size_t)lcast);
                    for (size_t cnode : chain) cover.push_back(cnode);
                }
                if (ok) {
                    MetalCluster c;
                    c.rule = 8;
                    c.a0 = (size_t)parentA; c.a1 = (size_t)parentB;
                    c.a2 = (size_t)offA0; c.a3 = (size_t)offB0;
                    c.u0 = (uint32_t)cnt; c.u1 = (uint32_t)M;
                    c.b0 = K; c.b1 = N; c.b2 = f32out ? 1 : 0;
                    c.b3 = f32a ? 1 : 0;
                    c.b4 = i;
                    add_cluster(c, i, cover);
                    continue;
                }
            }
            }
            bool samesz = true;
            {
                long fi = idxof(nd.input_ids[0]);
                if (fi < 0) samesz = false;
                else {
                    size_t ip = nodes[(size_t)fi]->output_buffer.total_size;
                    samesz = ip * cnt == nd.output_buffer.total_size;
                }
            }
            long tp0 = samesz ? up_in(i, 0) : -1;
            if (tp0 >= 0 && nodes[(size_t)tp0]->op_type == OpType::TRANSPOSE
                && nodes[(size_t)tp0]->output_buffer.precision == Precision::FP16
                && nd.output_buffer.precision == Precision::FP16
                && nodes[(size_t)tp0]->output_buffer.shape.size() + 1 <= 8
                && !is_noop_transpose(*nodes[(size_t)tp0])) {
                const auto& perm0 = nodes[(size_t)tp0]->params.permutation;
                const auto& tsh0 = nodes[(size_t)tp0]->output_buffer.shape;
                long parentP = -1, offP0 = 0, strideP = -1;
                size_t blk = nodes[(size_t)tp0]->output_buffer.total_size;
                bool ok = !perm0.empty();
                std::vector<size_t> cover;
                for (size_t j = 0; j < cnt && ok; ++j) {
                    long t = idxof(nd.input_ids[j]);
                    std::vector<size_t> chain;
                    while (t >= 0 && passthrough(*nodes[(size_t)t])) {
                        if (cons[(size_t)t].size() != 1) { t = -1; break; }
                        chain.push_back((size_t)t);
                        t = idxof(nodes[(size_t)t]->input_ids[0]);
                    }
                    if (t < 0 || nodes[(size_t)t]->op_type != OpType::TRANSPOSE
                        || plan->action[(size_t)t] != -1
                        || cons[(size_t)t].size() != 1
                        || nodes[(size_t)t]->params.permutation != perm0
                        || nodes[(size_t)t]->output_buffer.shape != tsh0
                        || nodes[(size_t)t]->output_buffer.precision != Precision::FP16) { ok = false; break; }
                    long offP;
                    long pp = alias_base(idxof(nodes[(size_t)t]->input_ids[0]), &offP);
                    if (pp < 0 || nodes[(size_t)pp]->output_buffer.precision != Precision::FP16) { ok = false; break; }
                    long ii = idxof(nodes[(size_t)t]->input_ids[0]);
                    if (ii < 0 || nodes[(size_t)ii]->output_buffer.total_size != blk) { ok = false; break; }
                    if (j == 0) { parentP = pp; offP0 = offP; }
                    else if (j == 1 && pp == parentP && offP > offP0) { strideP = offP - offP0; }
                    else if (j == 1 || pp != parentP || offP != offP0 + (long)j * strideP) { ok = false; break; }
                    cover.push_back((size_t)t);
                    for (size_t cnode : chain) cover.push_back(cnode);
                }
                if (ok && (strideP < 0 || cnt < 2)) ok = false;
                if (ok) {
                    size_t ax = (size_t)nd.params.axis;
                    long fi = idxof(nd.input_ids[0]);
                    const auto& fsh = nodes[(size_t)fi]->output_buffer.shape;
                    if (ax >= fsh.size()) ok = false;
                    else {
                        size_t pre = 1;
                        for (size_t d = 0; d < ax && d < fsh.size(); ++d) pre *= fsh[d];
                        size_t fpre = 1;
                        const auto& csh = nd.output_buffer.shape;
                        for (size_t d = 0; d < ax && d < csh.size(); ++d) fpre *= csh[d];
                        if (pre != fpre) ok = false;
                    }
                }
                if (ok) {
                    MetalCluster c;
                    c.rule = 10;
                    c.a0 = (size_t)parentP; c.a1 = (size_t)up_in(i, 0);
                    c.a2 = (size_t)offP0;
                    c.u0 = (uint32_t)cnt;
                    c.u1 = (uint32_t)nd.params.axis;
                    c.b0 = (size_t)strideP;
                    c.b4 = i;
                    add_cluster(c, i, cover);
                    continue;
                }
            }

            long c0 = up_in(i, 0);
            if (!blocky) c0 = -1;
            if (c0 >= 0 && nodes[(size_t)c0]->op_type == OpType::CONV1D
                && nodes[(size_t)c0]->output_buffer.precision == Precision::FP16) {
                const auto& osh = nodes[(size_t)c0]->output_buffer.shape;
                size_t Lout = osh.empty() ? 0 : osh.back();
                size_t stride = nodes[(size_t)c0]->params.stride ? nodes[(size_t)c0]->params.stride : 1;
                long parentX = -1, parentW = -1, offX0 = 0, offW0 = 0;
                size_t L = 0, Kk = 0;
                bool ok = Lout > 0;
                std::vector<size_t> cover;
                for (size_t j = 0; j < cnt && ok; ++j) {
                    long t = up(idxof(nd.input_ids[j]));
                    if (t < 0 || nodes[(size_t)t]->op_type != OpType::CONV1D
                        || plan->action[(size_t)t] != -1
                        || cons[(size_t)t].size() != 1
                        || nodes[(size_t)t]->input_ids.size() != 2
                        || nodes[(size_t)t]->output_buffer.shape.back() != Lout) { ok = false; break; }
                    long offX, offW;
                    long px = alias_base(idxof(nodes[(size_t)t]->input_ids[0]), &offX);
                    long pw = alias_base(idxof(nodes[(size_t)t]->input_ids[1]), &offW);
                    if (px < 0 || pw < 0
                        || nodes[(size_t)px]->output_buffer.precision != Precision::FP16
                        || nodes[(size_t)pw]->output_buffer.precision != Precision::FP16) { ok = false; break; }
                    long xin = idxof(nodes[(size_t)t]->input_ids[0]);
                    long win = idxof(nodes[(size_t)t]->input_ids[1]);
                    size_t Lj = nodes[(size_t)xin]->output_buffer.shape.back();
                    size_t Kj = nodes[(size_t)win]->output_buffer.shape.back();
                    if (j == 0) {
                        parentX = px; parentW = pw; offX0 = offX; offW0 = offW; L = Lj; Kk = Kj;
                        if ((L - Kk) / stride + 1 != Lout) { ok = false; break; }
                    } else if (px != parentX || pw != parentW || Lj != L || Kj != Kk
                               || offX != offX0 + (long)(j * L)
                               || offW != offW0 + (long)(j * Kk)) { ok = false; break; }
                    cover.push_back((size_t)t);
                }
                if (ok) {
                    MetalCluster c;
                    c.rule = 9;
                    c.a0 = (size_t)parentX; c.a1 = (size_t)parentW;
                    c.a2 = (size_t)offX0; c.a3 = (size_t)offW0;
                    c.u0 = (uint32_t)cnt; c.u1 = (uint32_t)L;
                    c.b0 = Lout; c.b1 = Kk; c.b2 = stride;
                    c.b4 = i;
                    add_cluster(c, i, cover);
                    continue;
                }
            }
        }

        if (nd.op_type == OpType::ADD_CLIPPED && nd.input_ids.size() == 2) {
            long n0 = up_in(i, 0), n1 = up_in(i, 1);
            auto rms2_side = [&](long nrm, long* src, long* w) -> bool {
                if (nrm < 0 || nodes[(size_t)nrm]->op_type != OpType::RMS_NORM
                    || nodes[(size_t)nrm]->input_ids.size() < 2
                    || plan->action[(size_t)nrm] != -1 || pinned[(size_t)nrm] || retyped((size_t)nrm)
                    || nodes[(size_t)nrm]->output_buffer.precision != Precision::FP16
                    || !sole_use(nrm, {i})) return false;
                long s0 = up_in((size_t)nrm, 0);
                long w0 = up_in((size_t)nrm, 1);
                if (w0 >= 0) w0 = deep_f16_source(w0);
                if (s0 < 0 || w0 < 0
                    || nodes[(size_t)s0]->output_buffer.precision != Precision::FP16
                    || nodes[(size_t)w0]->output_buffer.precision != Precision::FP16
                    || nodes[(size_t)s0]->output_buffer.shape.empty()
                    || nodes[(size_t)s0]->output_buffer.total_size !=
                       nodes[(size_t)s0]->output_buffer.shape.back()) return false;
                *src = s0; *w = w0;
                return true;
            };
            size_t D2 = nd.output_buffer.shape.empty() ? 0 : nd.output_buffer.shape.back();
            long srcA = -1, wA = -1, srcB = -1, wB = -1;
            if (n0 != n1 && D2 > 0 && nd.output_buffer.total_size == D2
                && nd.output_buffer.precision == Precision::FP16 && !retyped(i)
                && rms2_side(n0, &srcA, &wA) && rms2_side(n1, &srcB, &wB)
                && nodes[(size_t)srcA]->output_buffer.total_size == D2
                && nodes[(size_t)srcB]->output_buffer.total_size == D2
                && nodes[(size_t)wA]->output_buffer.total_size == D2
                && nodes[(size_t)wB]->output_buffer.total_size == D2) {
                MetalCluster c;
                c.rule = 18;
                c.a0 = (size_t)srcA; c.a1 = (size_t)wA;
                c.a2 = (size_t)srcB; c.a3 = (size_t)wB;
                c.u1 = (uint32_t)D2;
                c.f0 = nodes[(size_t)n0]->params.epsilon;
                c.f1 = nodes[(size_t)n1]->params.epsilon;
                c.b4 = i;
                std::vector<size_t> cover{(size_t)n0, (size_t)n1};
                if (add_cluster(c, i, cover)) continue;
            }
            long norm = -1, res = -1;
            if (n0 >= 0 && nodes[(size_t)n0]->op_type == OpType::RMS_NORM) { norm = n0; res = n1; }
            else if (n1 >= 0 && nodes[(size_t)n1]->op_type == OpType::RMS_NORM) { norm = n1; res = n0; }
            if (norm >= 0 && res >= 0 && sole_use(norm, {i})) {
                long src = up_in((size_t)norm, 0);
                long w = up_in((size_t)norm, 1);
                auto f16 = [&](long j) {
                    return j >= 0 && nodes[(size_t)j]->output_buffer.precision == Precision::FP16;
                };
                if (w >= 0) w = deep_f16_source(w);
                bool one_row = src >= 0 && !nodes[(size_t)src]->output_buffer.shape.empty()
                    && nodes[(size_t)src]->output_buffer.total_size ==
                       nodes[(size_t)src]->output_buffer.shape.back();
                if (src >= 0 && w >= 0 && one_row && f16((long)i) && f16(src) && f16(res) && f16(w) && f16(norm)) {
                    MetalCluster c;
                    c.rule = 3;
                    c.a0 = (size_t)src; c.a1 = (size_t)w; c.a2 = (size_t)res;
                    c.b4 = i;
                    c.f0 = nodes[(size_t)norm]->params.epsilon;
                    std::vector<size_t> cover;
                    cover.push_back((size_t)norm);
                    size_t h_node = i;
                    if (cons[i].size() == 1) {
                        long mt = (long)cons[i][0];
                        while (mt >= 0 && passthrough(*nodes[(size_t)mt]) && cons[(size_t)mt].size() == 1)
                            mt = (long)cons[(size_t)mt][0];
                        if (mt >= 0 && nodes[(size_t)mt]->op_type == OpType::MULTIPLY
                            && nodes[(size_t)mt]->input_ids.size() == 2
                            && nodes[(size_t)mt]->output_buffer.precision == Precision::FP16) {
                            long o0 = up_in((size_t)mt, 0), o1 = up_in((size_t)mt, 1);
                            long scn = (o0 == (long)i || (o0 >= 0 && up(o0) == (long)i)) ? o1 : o0;
                            long base = scn == o1 ? o0 : o1;
                            if (base >= 0 && scn >= 0
                                && nodes[(size_t)scn]->op_type == OpType::INPUT
                                && nodes[(size_t)scn]->output_buffer.total_size == 1
                                && nodes[(size_t)scn]->output_buffer.precision == Precision::FP16) {
                                c.b1 = (size_t)scn;
                                c.u1 = 1u;
                                collect_chain(cons[i][0], (long)i, cover);
                                cover.push_back((size_t)mt);
                                h_node = (size_t)mt;
                                c.b4 = h_node;
                                cover.push_back(i);
                            }
                        }
                    }
                    long next_norm = -1;
                    for (size_t cc : cons[h_node]) {
                        long t = (long)cc;
                        while (t >= 0 && passthrough(*nodes[(size_t)t])) {
                            if (cons[(size_t)t].size() != 1) { t = -1; break; }
                            t = (long)cons[(size_t)t][0];
                        }
                        if (t >= 0 && nodes[(size_t)t]->op_type == OpType::RMS_NORM
                            && up_in((size_t)t, 0) == (long)h_node
                            && nodes[(size_t)t]->output_buffer.precision == Precision::FP16) {
                            next_norm = t;
                            break;
                        }
                    }
                    if (next_norm >= 0) {
                        long nw = deep_f16_source(up_in((size_t)next_norm, 1));
                        size_t ndim = nodes[(size_t)next_norm]->output_buffer.shape.empty() ? 0
                                      : nodes[(size_t)next_norm]->output_buffer.shape.back();
                        size_t adim = nodes[i]->output_buffer.shape.empty() ? 0
                                      : nodes[i]->output_buffer.shape.back();
                        if (nw >= 0 && ndim == adim && f16(nw)) {
                            c.u0 = 1u;
                            c.a4 = (size_t)nw;
                            c.a5 = (size_t)next_norm;
                            c.s1 = cactus_metal_alloc_shared(nodes[(size_t)next_norm]->output_buffer.total_size * 2);
                            cover.push_back((size_t)next_norm);
                            long ch = idxof(nodes[(size_t)next_norm]->input_ids[0]);
                            collect_chain((size_t)ch, (long)h_node, cover);
                        }
                    }
                    add_cluster(c, h_node, cover);
                    continue;
                }
            }
        }
    }
    {
        std::unordered_map<long, std::vector<size_t>> groups;
        std::vector<long> order;
        for (size_t i = 0; i < n; ++i) {
            if (plan->action[i] != -1) continue;
            const GraphNode& mn = *nodes[i];
            if (mn.op_type != OpType::MATMUL || mn.input_ids.size() < 2) continue;
            long r = idxof(mn.input_ids[1]);
            if (r < 0) continue;
            const BufferDesc& rb = nodes[(size_t)r]->output_buffer;
            size_t M = 1;
            for (size_t d = 0; d + 1 < mn.output_buffer.shape.size(); ++d) M *= mn.output_buffer.shape[d];
            if (M != 1 || !PrecisionTraits::is_cq(rb.precision) || rb.group_size != 128
                || PrecisionTraits::cq_bits(rb.precision) != 4
                || (rb.cq_flags & CACTUS_QUANT_FLAG_ORTHOGONAL)) continue;
            long src = up_in(i, 0);
            if (src < 0) continue;
            auto it = groups.find(src);
            if (it == groups.end()) { groups[src] = {i}; order.push_back(src); }
            else it->second.push_back(i);
        }
        for (long src : order) {
            auto& mms = groups[src];
            if (mms.size() < 2) continue;
            for (size_t base = 0; base + 1 < mms.size(); base += 3) {
                size_t cnt = std::min<size_t>(3, mms.size() - base);
                if (cnt < 2) break;
                size_t anchor_i = mms[base];
                MetalCluster c;
                c.rule = 2;
                c.a0 = (size_t)src;
                c.b0 = mms[base];
                c.b1 = mms[base + 1];
                c.b2 = cnt > 2 ? mms[base + 2] : 0;
                c.u0 = (uint32_t)cnt;
                c.b4 = anchor_i;
                size_t maxK = 0;
                for (size_t mi = 0; mi < cnt; ++mi) {
                    const BufferDesc& rb = nodes[(size_t)idxof(nodes[mms[base + mi]]->input_ids[1])]->output_buffer;
                    CactusQuantMatrix W = rb.to_cq_matrix();
                    if (W.K > maxK) maxK = W.K;
                }
                for (uint32_t bi = 0; bi < c.u0; ++bi)
                    if (!c.sc[bi]) c.sc[bi] = cactus_metal_alloc_shared(maxK * 2);
                std::vector<size_t> cover;
                if (add_cluster(c, anchor_i, cover))
                    for (size_t mi = 1; mi < cnt; ++mi) plan->action[mms[base + mi]] = -3;
            }
        }
    }

    cands.erase(std::remove_if(cands.begin(), cands.end(),
        [&](const AttnCand& cd) { return cpu_only[cd.anchor] != 0; }), cands.end());
    if (!cands.empty()) {
        std::vector<uint8_t> mark(n, 0);
        for (auto& cd : cands) {
            mark[cd.anchor] = 2;
            for (size_t v : cd.cover) if (mark[v] == 0) mark[v] = 1;
        }
        for (size_t ri = n; ri-- > 0;) {
            if (mark[ri]) continue;
            const GraphNode& ndp = *nodes[ri];
            if (!(is_alias_op(ndp.op_type) || is_same_cast(ndp, nodes, map) || is_noop_transpose(ndp))) continue;
            if (cons[ri].empty()) { mark[ri] = 3; continue; }
            bool all_in = true;
            for (size_t cc : cons[ri]) if (!mark[cc]) { all_in = false; break; }
            if (all_in) mark[ri] = 3;
        }
        bool valid = true;
        for (size_t v = 0; v < n && valid; ++v) {
            if (mark[v] != 1 && mark[v] != 3) continue;
            for (size_t cc : cons[v]) {
                if (!mark[cc]) {
                    valid = false;
                    break;
                }
            }
        }
        for (size_t v = 0; v < n && valid; ++v)
            if ((mark[v] == 1 || mark[v] == 3) && pinned[v]) valid = false;
        if (valid) {
            for (size_t v = 0; v < n; ++v)
                if ((mark[v] == 1 || mark[v] == 3) && plan->action[v] == -1) plan->action[v] = -2;
            for (auto& cd : cands) {
                int32_t cid = (int32_t)plan->clusters.size();
                plan->clusters.push_back(cd.c);
                plan->action[cd.anchor] = cid;
            }
        }
    }

    for (auto& cl : plan->clusters) {
        if (cl.rule != 5) continue;
        const auto& sh0 = nodes[0]->output_buffer;
        const auto& sh1 = n > 1 ? nodes[1]->output_buffer : nodes[0]->output_buffer;
        const auto& sh2 = n > 2 ? nodes[2]->output_buffer : nodes[0]->output_buffer;
        if (n > 2 && nodes[0]->op_type == OpType::INPUT && nodes[1]->op_type == OpType::INPUT
            && nodes[2]->op_type == OpType::INPUT
            && sh0.precision == Precision::FP16 && sh0.total_size > 0
            && sh1.precision == Precision::FP16 && sh1.shape.size() >= 2
            && sh2.total_size == 1) {
            plan->fold_h = 0;
            plan->fold_ple = 1;
            plan->fold_pos = 2;
            plan->fold_w = (long)cl.a1;
            plan->fold_nl = sh1.shape[sh1.shape.size() - 2];
            plan->fold_pd = sh1.shape.back();
        }
        break;
    }

    {
        auto ew_kind = [&](const GraphNode& nd, int* kind, int* code, float* p0, float* p1) -> bool {
            *p0 = 0; *p1 = 0;
            switch (nd.op_type) {
                case OpType::GELU: *kind = 0; *code = 0; return true;
                case OpType::TANH: *kind = 0; *code = 1; return true;
                case OpType::SILU: *kind = 0; *code = 2; return true;
                case OpType::RELU: *kind = 0; *code = 3; return true;
                case OpType::GELU_ERF: *kind = 0; *code = 4; return true;
                case OpType::SIGMOID: *kind = 0; *code = 5; return true;
                case OpType::SCALAR_ADD: *kind = 1; *code = 0; *p0 = nd.params.scalar; return true;
                case OpType::SCALAR_SUBTRACT: *kind = 1; *code = 1; *p0 = nd.params.scalar; return true;
                case OpType::SCALAR_MULTIPLY: *kind = 1; *code = 2; *p0 = nd.params.scalar; return true;
                case OpType::SCALAR_DIVIDE: *kind = 1; *code = 3; *p0 = nd.params.scalar; return true;
                case OpType::SCALAR_EXP: *kind = 1; *code = 4; return true;
                case OpType::SCALAR_SQRT: *kind = 1; *code = 5; return true;
                case OpType::SCALAR_COS: *kind = 1; *code = 6; return true;
                case OpType::SCALAR_SIN: *kind = 1; *code = 7; return true;
                case OpType::SCALAR_LOG: *kind = 1; *code = 8; return true;
                case OpType::ABS: *kind = 1; *code = 9; return true;
                case OpType::POW: *kind = 1; *code = 10; *p0 = nd.params.scalar; return true;
                case OpType::SCALAR_NOT_EQUAL: *kind = 1; *code = 11; *p0 = nd.params.scalar; return true;
                case OpType::LEAKY_RELU: *kind = 1; *code = 12; *p0 = nd.params.scalar; return true;
                case OpType::ADD: *kind = 2; *code = 0; return true;
                case OpType::ADD_CLIPPED: *kind = 2; *code = 1; return true;
                case OpType::SUBTRACT: *kind = 2; *code = 2; return true;
                case OpType::MULTIPLY: *kind = 2; *code = 3; return true;
                case OpType::DIVIDE: *kind = 2; *code = 4; return true;
                case OpType::NOT_EQUAL: *kind = 2; *code = 5; return true;
                case OpType::CLAMP: *kind = 3; *code = 0; *p0 = nd.params.scalar; *p1 = nd.params.scale; return true;
                case OpType::PRECISION_CAST: *kind = 4;
                    *code = nd.output_buffer.precision == Precision::FP32 ? 1 : 0;
                    return nd.output_buffer.precision == Precision::FP16
                        || nd.output_buffer.precision == Precision::FP32;
                default: return false;
            }
        };
        auto prec_ok = [](Precision p) { return p == Precision::FP16 || p == Precision::FP32; };
        for (size_t i = 0; i < n; ++i) {
            if (plan->action[i] != -1) continue;
            GraphNode& nd0 = *nodes[i];
            if (nd0.op_type != OpType::RMS_NORM || nd0.input_ids.size() < 2) continue;
            if (nd0.output_buffer.precision != Precision::FP16 || retyped(i)) continue;
            const auto& osh = nd0.output_buffer.shape;
            if (osh.empty()) continue;
            size_t dim = osh.back();
            size_t rows = nd0.output_buffer.total_size / dim;
            if (rows < 2 || dim < 32) continue;
            long ai = idxof(nd0.input_ids[0]);
            long wi = idxof(nd0.input_ids[1]);
            if (ai < 0 || wi < 0 || plan->action[(size_t)ai] != -1 || retyped((size_t)ai)) continue;
            const GraphNode& an = *nodes[(size_t)ai];
            if ((an.op_type != OpType::ADD && an.op_type != OpType::ADD_CLIPPED)
                || an.input_ids.size() != 2
                || an.output_buffer.precision != Precision::FP16
                || an.output_buffer.total_size != nd0.output_buffer.total_size
                || pinned[(size_t)ai]) continue;
            long x0 = idxof(an.input_ids[0]);
            long x1 = idxof(an.input_ids[1]);
            if (x0 < 0 || x1 < 0
                || nodes[(size_t)x0]->output_buffer.precision != Precision::FP16
                || nodes[(size_t)x1]->output_buffer.precision != Precision::FP16
                || nodes[(size_t)x0]->output_buffer.total_size != an.output_buffer.total_size
                || nodes[(size_t)x1]->output_buffer.total_size != an.output_buffer.total_size) continue;
            long ww = deep_f16_source(wi);
            if (ww < 0 || nodes[(size_t)ww]->output_buffer.total_size != dim) continue;
            bool ordered = true;
            for (size_t cc : cons[(size_t)ai])
                if (cc < i) { ordered = false; break; }
            if (!ordered) continue;
            MetalCluster c;
            c.rule = 12;
            c.a0 = (size_t)x0; c.a1 = (size_t)x1; c.a2 = (size_t)ai; c.a3 = (size_t)ww;
            c.u0 = (uint32_t)rows; c.u1 = (uint32_t)dim;
            c.f0 = nd0.params.epsilon;
            c.b0 = an.op_type == OpType::ADD_CLIPPED ? 1 : 0;
            c.b4 = i;
            std::vector<size_t> cover;
            if (add_cluster(c, i, cover)) plan->action[(size_t)ai] = -3;
        }
        for (size_t i = 0; i < n; ++i) {
            if (plan->action[i] != -1) continue;
            GraphNode& nd0 = *nodes[i];
            if (nd0.op_type != OpType::ADD || nd0.input_ids.size() != 2) continue;
            if (nd0.output_buffer.precision != Precision::FP16 || retyped(i)) continue;
            long i0 = idxof(nd0.input_ids[0]);
            long i1 = idxof(nd0.input_ids[1]);
            if (i0 < 0 || i1 < 0 || i0 == i1) continue;
            long mm = -1, bs = -1;
            if (nodes[(size_t)i0]->op_type == OpType::MATMUL) { mm = i0; bs = i1; }
            else if (nodes[(size_t)i1]->op_type == OpType::MATMUL) { mm = i1; bs = i0; }
            if (mm < 0 || plan->action[(size_t)mm] != -1 || pinned[(size_t)mm]) continue;
            if (plan->action[(size_t)bs] == -2) continue;
            if (retyped((size_t)mm) || retyped((size_t)bs)) continue;
            if (cons[(size_t)mm].size() != 1) continue;
            GraphNode& m = *nodes[(size_t)mm];
            if (m.input_ids.size() < 2) continue;
            long li = idxof(m.input_ids[0]);
            long ri = idxof(m.input_ids[1]);
            if (li < 0 || ri < 0 || retyped((size_t)li) || retyped((size_t)ri)) continue;
            if (plan->action[(size_t)li] == -2 || plan->action[(size_t)ri] == -2) continue;
            const BufferDesc& lb = nodes[(size_t)li]->output_buffer;
            const BufferDesc& rb = nodes[(size_t)ri]->output_buffer;
            const BufferDesc& bb = nodes[(size_t)bs]->output_buffer;
            if (lb.precision != Precision::FP16 || rb.precision != Precision::FP16
                || bb.precision != Precision::FP16) continue;
            if (rb.shape.size() != 2 || lb.shape.empty()) continue;
            size_t K = lb.shape.back();
            if (K == 0 || lb.total_size != K) continue;
            size_t N = m.params.pretransposed_rhs ? rb.shape[0] : rb.shape[1];
            size_t rk = m.params.pretransposed_rhs ? rb.shape[1] : rb.shape[0];
            if (rk != K || N == 0 || m.output_buffer.total_size != N) continue;
            if (bb.total_size != N || nd0.output_buffer.total_size != N) continue;
            MetalCluster c;
            c.rule = 13;
            c.a0 = (size_t)li; c.a1 = (size_t)ri; c.a2 = (size_t)bs;
            c.u0 = (uint32_t)K; c.u1 = (uint32_t)N;
            c.b0 = m.params.pretransposed_rhs ? 1 : 0;
            c.b4 = i;
            std::vector<size_t> cover;
            cover.push_back((size_t)mm);
            add_cluster(c, i, cover);
        }
        for (size_t i = 0; i < n; ++i) {
            if (plan->action[i] != -1) continue;
            GraphNode& nd0 = *nodes[i];
            if (nd0.op_type != OpType::ADD || nd0.input_ids.size() != 2) continue;
            if (nd0.output_buffer.precision != Precision::FP16 || retyped(i)) continue;
            const auto& osh = nd0.output_buffer.shape;
            if (osh.size() < 2) continue;
            size_t D = osh.back();
            if (D == 0 || (D % 2) != 0) continue;
            size_t H = nd0.output_buffer.total_size / D;
            if (H == 0 || H * D != nd0.output_buffer.total_size) continue;
            long mA = up_in(i, 0), mB = up_in(i, 1);
            if (mA < 0 || mB < 0 || mA == mB) continue;
            auto is_mul = [&](long j) {
                return nodes[(size_t)j]->op_type == OpType::MULTIPLY
                    && nodes[(size_t)j]->input_ids.size() == 2
                    && plan->action[(size_t)j] == -1 && !pinned[(size_t)j] && !retyped((size_t)j)
                    && nodes[(size_t)j]->output_buffer.precision == Precision::FP16
                    && nodes[(size_t)j]->output_buffer.total_size == H * D;
            };
            if (!is_mul(mA) || !is_mul(mB)) continue;
            auto split_mul = [&](long m, long* big, long* small) -> bool {
                long i0 = up_in((size_t)m, 0), i1 = up_in((size_t)m, 1);
                if (i0 < 0 || i1 < 0) return false;
                size_t t0 = nodes[(size_t)i0]->output_buffer.total_size;
                size_t t1 = nodes[(size_t)i1]->output_buffer.total_size;
                if (t0 == H * D && t1 == D) { *big = i0; *small = i1; return true; }
                if (t1 == H * D && t0 == D) { *big = i1; *small = i0; return true; }
                return false;
            };
            long xA = -1, cA = -1, xB = -1, sB = -1;
            if (!split_mul(mA, &xA, &cA) || !split_mul(mB, &xB, &sB)) continue;
            long mulCos = -1, mulSin = -1, xi = -1, ci = -1, si = -1, cat = -1;
            if (nodes[(size_t)xB]->op_type == OpType::CAT) {
                mulCos = mA; mulSin = mB; xi = xA; ci = cA; si = sB; cat = xB;
            } else if (nodes[(size_t)xA]->op_type == OpType::CAT) {
                mulCos = mB; mulSin = mA; xi = xB; ci = sB; si = cA; cat = xA;
            } else continue;
            GraphNode& cn = *nodes[(size_t)cat];
            if (cn.input_ids.size() != 2 || plan->action[(size_t)cat] != -1
                || pinned[(size_t)cat] || retyped((size_t)cat)) continue;
            if (cn.output_buffer.total_size != H * D) continue;
            size_t cat_ax = (size_t)cn.params.axis;
            if (cat_ax != cn.output_buffer.shape.size() - 1) continue;
            long neg = up_in((size_t)cat, 0);
            long lo = up_in((size_t)cat, 1);
            if (neg < 0 || lo < 0) continue;
            GraphNode& ng = *nodes[(size_t)neg];
            if (ng.op_type != OpType::SCALAR_MULTIPLY || ng.params.scalar != -1.0f
                || plan->action[(size_t)neg] != -1 || pinned[(size_t)neg] || retyped((size_t)neg)) continue;
            long hi = up_in((size_t)neg, 0);
            if (hi < 0) continue;
            GraphNode& hs = *nodes[(size_t)hi];
            GraphNode& ls = *nodes[(size_t)lo];
            if (hs.op_type != OpType::SLICE || ls.op_type != OpType::SLICE) continue;
            if (plan->action[(size_t)hi] != -1 || plan->action[(size_t)lo] != -1
                || pinned[(size_t)hi] || pinned[(size_t)lo]) continue;
            if (hs.params.slice_start != D / 2 || ls.params.slice_start != 0) continue;
            size_t hlen = hs.params.slice_length ? hs.params.slice_length : D / 2;
            size_t llen = ls.params.slice_length ? ls.params.slice_length : D / 2;
            if (hlen != D / 2 || llen != D / 2) continue;
            if ((size_t)hs.params.axis != hs.output_buffer.shape.size() - 1
                || (size_t)ls.params.axis != ls.output_buffer.shape.size() - 1) continue;
            long xh = up_in((size_t)hi, 0), xl = up_in((size_t)lo, 0);
            if (xh != xi || xl != xi) continue;
            if (nodes[(size_t)xi]->output_buffer.precision != Precision::FP16
                || nodes[(size_t)ci]->output_buffer.precision != Precision::FP16
                || nodes[(size_t)si]->output_buffer.precision != Precision::FP16) continue;
            if (!sole_use(mulCos, {i}) || !sole_use(mulSin, {i})
                || !sole_use(cat, {(size_t)mulSin}) || !sole_use(neg, {(size_t)cat})
                || !sole_use(hi, {(size_t)neg}) || !sole_use(lo, {(size_t)cat})) continue;
            MetalCluster c;
            c.rule = 14;
            c.a0 = (size_t)xi; c.a1 = (size_t)ci; c.a2 = (size_t)si;
            c.u0 = (uint32_t)H; c.u1 = (uint32_t)D;
            c.b4 = i;
            std::vector<size_t> cover{(size_t)mulCos, (size_t)mulSin, (size_t)cat,
                                      (size_t)neg, (size_t)hi, (size_t)lo};
            collect_chain((size_t)idxof(nd0.input_ids[0]), mulCos, cover);
            collect_chain((size_t)idxof(nd0.input_ids[1]), mulSin, cover);
            GraphNode& xn = *nodes[(size_t)xi];
            if (D <= 1024 && xn.op_type == OpType::RMS_NORM && xn.input_ids.size() >= 2
                && plan->action[(size_t)xi] == -1 && !pinned[(size_t)xi] && !retyped((size_t)xi)
                && sole_use(xi, {(size_t)mulCos, (size_t)hi, (size_t)lo})) {
                long rsrc = up_in((size_t)xi, 0);
                long rw = up_in((size_t)xi, 1);
                if (rw >= 0) rw = deep_f16_source(rw);
                if (rsrc >= 0 && rw >= 0
                    && nodes[(size_t)rsrc]->output_buffer.precision == Precision::FP16
                    && nodes[(size_t)rsrc]->output_buffer.total_size == H * D
                    && nodes[(size_t)rw]->output_buffer.precision == Precision::FP16
                    && nodes[(size_t)rw]->output_buffer.total_size == D) {
                    c.rule = 17;
                    c.a0 = (size_t)rsrc; c.a3 = (size_t)rw;
                    c.f0 = xn.params.epsilon;
                    cover.push_back((size_t)xi);
                }
            }
            add_cluster(c, i, cover);
        }
        for (size_t i = 0; i < n; ++i) {
            if (plan->action[i] != -1) continue;
            GraphNode& nd0 = *nodes[i];
            if (nd0.op_type != OpType::SCALAR_MULTIPLY || nd0.input_ids.empty()) continue;
            if (nd0.output_buffer.precision != Precision::FP16 || retyped(i)) continue;
            const auto& osh = nd0.output_buffer.shape;
            if (osh.empty()) continue;
            size_t D = osh.back();
            size_t rows = nd0.output_buffer.total_size / D;
            if (D < 32 || rows == 0 || rows * D != nd0.output_buffer.total_size) continue;
            long rn = up_in(i, 0);
            if (rn < 0 || nodes[(size_t)rn]->op_type != OpType::RMS_NORM
                || nodes[(size_t)rn]->input_ids.size() < 2
                || plan->action[(size_t)rn] != -1 || pinned[(size_t)rn] || retyped((size_t)rn)
                || nodes[(size_t)rn]->output_buffer.precision != Precision::FP16
                || nodes[(size_t)rn]->output_buffer.total_size != rows * D) continue;
            if (!sole_use(rn, {i})) continue;
            long src = up_in((size_t)rn, 0);
            long w = up_in((size_t)rn, 1);
            if (w >= 0) w = deep_f16_source(w);
            if (src < 0 || w < 0
                || nodes[(size_t)src]->output_buffer.precision != Precision::FP16
                || nodes[(size_t)w]->output_buffer.precision != Precision::FP16
                || nodes[(size_t)w]->output_buffer.total_size != D) continue;
            MetalCluster c;
            c.rule = 15;
            c.a0 = (size_t)src; c.a1 = (size_t)w;
            c.u0 = (uint32_t)rows; c.u1 = (uint32_t)D;
            c.f0 = nodes[(size_t)rn]->params.epsilon;
            c.f1 = nd0.params.scalar;
            c.b4 = i;
            std::vector<size_t> cover{(size_t)rn};
            collect_chain((size_t)idxof(nd0.input_ids[0]), rn, cover);
            add_cluster(c, i, cover);
        }
        for (size_t i = 0; i < n; ++i) {
            if (plan->action[i] != -1) continue;
            GraphNode& nd0 = *nodes[i];
            if (nd0.op_type != OpType::TOPK || nd0.input_ids.empty()) continue;
            if (nd0.output_buffer.precision != Precision::FP32) continue;
            size_t k = nd0.params.top_k;
            if (k == 0 || k > 16) continue;
            long sm = up_in(i, 0);
            if (sm < 0 || nodes[(size_t)sm]->op_type != OpType::SCALAR_MULTIPLY
                || nodes[(size_t)sm]->input_ids.empty()
                || plan->action[(size_t)sm] != -1 || pinned[(size_t)sm] || retyped((size_t)sm)
                || nodes[(size_t)sm]->output_buffer.precision != Precision::FP16) continue;
            const auto& ssh = nodes[(size_t)sm]->output_buffer.shape;
            if (ssh.size() != 2) continue;
            size_t rows = ssh[0], E = ssh[1];
            if (rows == 0 || E == 0 || E > 4096) continue;
            if (nd0.output_buffer.total_size != 2 * rows * k) continue;
            long sfm = -1;
            for (size_t cc : cons[(size_t)sm]) {
                if (cc == i) continue;
                const GraphNode& cn = *nodes[cc];
                if (cn.op_type == OpType::SOFTMAX && plan->action[cc] == -1
                    && !pinned[cc] && !retyped(cc)
                    && cn.output_buffer.precision == Precision::FP16
                    && cn.output_buffer.total_size == rows * E) {
                    if (sfm < 0) sfm = (long)cc;
                    else { sfm = -1; break; }
                } else if (!is_alias_op(cn.op_type)) { sfm = -1; break; }
            }
            if (sfm < 0 || !sole_use(sm, {i, (size_t)sfm})) continue;
            if ((size_t)sfm > i) continue;
            bool ordered = true;
            for (size_t cc : cons[(size_t)sfm])
                if (cc < i) { ordered = false; break; }
            if (!ordered) continue;
            long lg = up_in((size_t)sm, 0);
            if (lg < 0 || nodes[(size_t)lg]->output_buffer.precision != Precision::FP16
                || nodes[(size_t)lg]->output_buffer.total_size != rows * E) continue;
            MetalCluster c;
            c.rule = 16;
            c.a0 = (size_t)lg; c.a1 = (size_t)sfm;
            c.u0 = (uint32_t)rows; c.u1 = (uint32_t)E;
            c.b0 = k;
            c.f0 = nodes[(size_t)sm]->params.scalar;
            c.b4 = i;
            std::vector<size_t> cover{(size_t)sm};
            collect_chain((size_t)idxof(nd0.input_ids[0]), sm, cover);
            if (add_cluster(c, i, cover)) plan->action[(size_t)sfm] = -3;
        }
        for (size_t i = 0; i < n; ++i) {
            if (plan->action[i] != -1) continue;
            int kind, code; float p0, p1;
            if (!ew_kind(*nodes[i], &kind, &code, &p0, &p1)) continue;
            if (!prec_ok(nodes[i]->output_buffer.precision) || retyped(i)) continue;
            size_t total = nodes[i]->output_buffer.total_size;
            std::vector<size_t> chain{i};
            size_t cur = i;
            while (chain.size() < 12) {
                if (cons[cur].size() != 1) break;
                size_t nxt = cons[cur][0];
                if (plan->action[nxt] != -1) break;
                int k2, c2; float q0, q1;
                if (!ew_kind(*nodes[nxt], &k2, &c2, &q0, &q1)) break;
                if (!prec_ok(nodes[nxt]->output_buffer.precision) || retyped(nxt)) break;
                if (nodes[nxt]->output_buffer.total_size != total) break;
                bool feeds = false;
                for (size_t a = 0; a < nodes[nxt]->input_ids.size() && a < 2; ++a)
                    if (idxof(nodes[nxt]->input_ids[a]) == (long)cur) feeds = true;
                if (!feeds) break;
                chain.push_back(nxt);
                cur = nxt;
            }
            size_t real_ops = 0;
            for (size_t ci = 0; ci < chain.size(); ++ci) {
                int k2, c2; float q0, q1;
                ew_kind(*nodes[chain[ci]], &k2, &c2, &q0, &q1);
                if (k2 != 4) ++real_ops;
            }
            if (chain.size() < 2 || real_ops == 0) continue;
            std::vector<float> blob;
            std::vector<size_t> sides;
            std::vector<int> side_f32;
            bool ok = true;
            long head_in = -1;
            Precision run = Precision::FP16;
            for (size_t ci = 0; ci < chain.size() && ok; ++ci) {
                const GraphNode& cn = *nodes[chain[ci]];
                int k2, c2; float q0, q1;
                ew_kind(cn, &k2, &c2, &q0, &q1);
                long prev = ci == 0 ? -1 : (long)chain[ci - 1];
                if (ci == 0) {
                    long hi2 = idxof(cn.input_ids[0]);
                    if (hi2 < 0 || !prec_ok(nodes[(size_t)hi2]->output_buffer.precision)
                        || retyped((size_t)hi2)
                        || nodes[(size_t)hi2]->output_buffer.total_size != total) { ok = false; break; }
                    head_in = hi2;
                    run = nodes[(size_t)hi2]->output_buffer.precision;
                }
                if (k2 == 2) {
                    long in0 = idxof(cn.input_ids[0]);
                    long in1 = idxof(cn.input_ids[1]);
                    long side, chain_in;
                    bool rhs;
                    if (ci > 0 && in0 == prev) { chain_in = in0; side = in1; rhs = false; }
                    else if (ci > 0 && in1 == prev) { chain_in = in1; side = in0; rhs = true; }
                    else { chain_in = in0; side = in1; rhs = false; }
                    if (side < 0 || chain_in < 0
                        || nodes[(size_t)side]->output_buffer.precision != run
                        || retyped((size_t)side)
                        || plan->action[(size_t)side] == -2) { ok = false; break; }
                    size_t inner = nodes[i]->output_buffer.shape.empty()
                        ? 1 : nodes[i]->output_buffer.shape.back();
                    size_t stotal = nodes[(size_t)side]->output_buffer.total_size;
                    int bmode;
                    if (stotal == total) bmode = 0;
                    else if (inner > 1 && stotal * inner == total
                             && !nodes[(size_t)side]->output_buffer.shape.empty()
                             && nodes[(size_t)side]->output_buffer.shape.back() == 1) bmode = 1;
                    else if (inner > 1 && stotal == inner) bmode = 2;
                    else { ok = false; break; }
                    size_t slot = sides.size();
                    for (size_t si = 0; si < sides.size(); ++si)
                        if (sides[si] == (size_t)side) slot = si;
                    if (slot == sides.size()) {
                        if (sides.size() >= 3) { ok = false; break; }
                        sides.push_back((size_t)side);
                        side_f32.push_back(run == Precision::FP32 ? 1 : 0);
                    } else if (side_f32[slot] != (run == Precision::FP32 ? 1 : 0)) { ok = false; break; }
                    c2 |= (rhs ? 16 : 0) | (run == Precision::FP32 ? 32 : 0) | ((int)slot << 6) | (bmode << 8);
                }
                if (k2 == 4) {
                    Precision from = ci == 0 ? run : nodes[(size_t)prev]->output_buffer.precision;
                    if (from == cn.output_buffer.precision) { ok = false; break; }
                    run = cn.output_buffer.precision;
                } else if (cn.output_buffer.precision != run) { ok = false; break; }
                EwChainStep st{k2, c2, q0, q1};
                float packed[4];
                std::memcpy(packed, &st, sizeof(st));
                blob.insert(blob.end(), packed, packed + 4);
            }
            if (!ok || head_in < 0) continue;
            Precision head_prec = nodes[(size_t)head_in]->output_buffer.precision;
            uint32_t flags = (head_prec == Precision::FP32 ? 1u : 0u)
                           | (nodes[chain.back()]->output_buffer.precision == Precision::FP32 ? 2u : 0u);
            for (size_t si = 0; si < sides.size(); ++si)
                if (side_f32[si]) flags |= (4u << si);
            MetalCluster c;
            c.rule = 11;
            c.a0 = (size_t)head_in;
            c.a1 = sides.size() > 0 ? sides[0] : (size_t)0;
            c.a2 = sides.size() > 1 ? sides[1] : (size_t)0;
            c.a3 = sides.size() > 2 ? sides[2] : (size_t)0;
            c.u0 = (uint32_t)chain.size();
            c.u1 = (uint32_t)sides.size();
            c.b0 = plan->blobs.size();
            c.b1 = flags;
            c.b4 = chain.back();
            plan->blobs.push_back(std::move(blob));
            std::vector<size_t> cover(chain.begin(), chain.end() - 1);
            add_cluster(c, chain.back(), cover);
        }
    }

    plan->exec_list.reserve(n);
    for (size_t i = 0; i < n; ++i)
        if (plan->action[i] != -2 && nodes[i]->op_type != OpType::INPUT)
            plan->exec_list.push_back((uint32_t)i);

    bool decode_like = false;
    for (auto& cl : plan->clusters) if (cl.rule == 1) { decode_like = true; break; }
    if (decode_like) {
        std::vector<size_t> last_read(n, 0);
        auto base_of = [&](size_t i) -> size_t {
            size_t cur = i;
            while (passthrough(*nodes[cur]) || plan->action[cur] == -2 || is_noop_transpose(*nodes[cur])
                   || nodes[cur]->op_type == OpType::SLICE || nodes[cur]->op_type == OpType::INDEX) {
                if (nodes[cur]->input_ids.empty()) break;
                long j = idxof(nodes[cur]->input_ids[0]);
                if (j < 0) break;
                if (plan->action[cur] != -2 && !passthrough(*nodes[cur])
                    && !is_noop_transpose(*nodes[cur])) {
                    bool aliasing = false;
                    const GraphNode& sn = *nodes[cur];
                    if (sn.op_type == OpType::INDEX && sn.params.axis == 0) aliasing = true;
                    if (sn.op_type == OpType::SLICE) {
                        size_t ax = (size_t)sn.params.axis;
                        long jj = idxof(sn.input_ids[0]);
                        if (jj >= 0 && ax < nodes[(size_t)jj]->output_buffer.shape.size()) {
                            size_t outer = 1;
                            for (size_t d = 0; d < ax; ++d) outer *= nodes[(size_t)jj]->output_buffer.shape[d];
                            if (outer == 1) aliasing = true;
                        }
                    }
                    if (!aliasing) break;
                }
                cur = (size_t)j;
            }
            return cur;
        };
        for (size_t i = 0; i < n; ++i) {
            for (size_t id : nodes[i]->input_ids) {
                long j = idxof(id);
                if (j >= 0) {
                    size_t b = base_of((size_t)j);
                    if (i > last_read[b]) last_read[b] = i;
                }
            }
        }
        for (auto& cl : plan->clusters) {
            size_t anchor = cl.b4 ? cl.b4 : (cl.rule == 2 ? std::max(cl.b0, std::max(cl.b1, cl.b2)) : 0);
            if (!anchor) continue;
            for (size_t arg : {cl.a0, cl.a1, cl.a2, cl.a3, cl.a4, cl.a5, cl.b0, cl.b1, cl.b2, cl.b3}) {
                if (arg == 0 || arg >= n) continue;
                size_t b = base_of(arg);
                if (anchor > last_read[b]) last_read[b] = anchor;
            }
        }
        plan->arena_off.assign(n, -1);
        struct Range { size_t off, size, free_at; };
        std::vector<Range> live;
        size_t cursor = 0, peak = 0;
        for (uint32_t ui : plan->exec_list) {
            size_t i = ui;
            const GraphNode& nd = *nodes[i];
            if (plan->action[i] == -2) continue;
            if (pinned[i]) continue;
            if (nd.op_type == OpType::KV_CACHE_STATE || nd.op_type == OpType::CONV_CACHE_STATE
                || nd.op_type == OpType::RECURRENT_CACHE_STATE || nd.op_type == OpType::KV_CACHE_APPEND
                || nd.op_type == OpType::PERSISTENT) continue;
            if (passthrough(nd) || is_noop_transpose(nd)) continue;
            if (nd.op_type == OpType::SLICE || nd.op_type == OpType::INDEX) continue;
            if (last_read[i] == 0) continue;
            size_t need = (nd.output_buffer.byte_size + 255) & ~size_t(255);
            if (need == 0 || need > (8u << 20)) continue;
            bool placed = false;
            for (auto& r : live) {
                if (r.free_at < i && r.size >= need) {
                    plan->arena_off[i] = (long)r.off;
                    r.free_at = last_read[i];
                    placed = true;
                    break;
                }
            }
            if (!placed) {
                plan->arena_off[i] = (long)cursor;
                live.push_back({cursor, need, last_read[i]});
                cursor += need;
                if (cursor > peak) peak = cursor;
            }
        }
        if (peak > 0 && peak < (256u << 20))
            plan->arena_base = (char*)cactus_metal_alloc_shared(peak);
        if (!plan->arena_base) plan->arena_off.assign(n, -1);
    }

    return plan;
}

static void release_cluster_buffers(MetalFusePlan* p) {
    for (auto& c : p->clusters) {
        if (c.s0) { cactus_metal_free_shared(c.s0); c.s0 = nullptr; }
        if (c.s1) { cactus_metal_free_shared(c.s1); c.s1 = nullptr; }
        for (auto& s : c.sc) if (s) { cactus_metal_free_shared(s); s = nullptr; }
    }
}

void cactus_metal_plan_free(MetalFusePlan* p) {
    if (!p) return;
    release_cluster_buffers(p);
    if (p->arena_base) { cactus_metal_free_shared(p->arena_base); p->arena_base = nullptr; }
    delete p;
}

bool cactus_metal_plan_has_arena(const MetalFusePlan* p) {
    if (!p) return false;
    if (p->arena_base) return true;
    for (const auto& c : p->clusters) if (c.rule == 1) return true;
    return false;
}

void cactus_metal_plan_extend_last_use(const MetalFusePlan* p, std::vector<size_t>& last_use) {
    if (!p) return;
    const size_t n = last_use.size();
    for (const auto& c : p->clusters) {
        size_t ai = c.b4;
        if (ai >= n) continue;
        for (size_t v : {c.a0, c.a1, c.a2, c.a3, c.a4, c.a5, c.b0, c.b1, c.b2, c.b3})
            if (v < n && last_use[v] < ai) last_use[v] = ai;
    }
}

int32_t cactus_metal_plan_action(const MetalFusePlan* p, size_t i) {
    return p && i < p->action.size() ? p->action[i] : -1;
}


const std::vector<uint32_t>* cactus_metal_plan_exec_list(const MetalFusePlan* p) {
    return p ? &p->exec_list : nullptr;
}

void* cactus_metal_plan_arena_ptr(const MetalFusePlan* p, size_t i) {
    if (!p || !p->arena_base || i >= p->arena_off.size() || p->arena_off[i] < 0) return nullptr;
    return p->arena_base + p->arena_off[i];
}

bool cactus_metal_plan_fold(MetalFusePlan* p,
                            const std::vector<std::unique_ptr<GraphNode>>& nodes) {
    if (!p || p->fold_h < 0 || !cactus_graph_fused_embed()) return false;
    void* h = nodes[(size_t)p->fold_h]->output_buffer.get_data();
    void* ple = nodes[(size_t)p->fold_ple]->output_buffer.get_data();
    void* pos = nodes[(size_t)p->fold_pos]->output_buffer.get_data();
    if (!h || !ple) return false;
    CactusQuantMatrix W = nodes[(size_t)p->fold_w]->output_buffer.to_cq_matrix();
    return cactus_graph_metal_fold_prologue(h, ple, pos, &W, p->fold_nl, p->fold_pd);
}

bool cactus_metal_plan_encode(MetalFusePlan* p, int32_t cid,
                              const std::vector<std::unique_ptr<GraphNode>>& nodes,
                              const std::unordered_map<size_t, size_t>& map) {
    MetalCluster& c = p->clusters[(size_t)cid];
    auto data = [&](size_t i) { return nodes[i]->output_buffer.get_data(); };
    switch (c.rule) {
        case 1: {
            GraphNode& anchor = *nodes[c.b4];
            BufferDesc& kc = nodes[c.b2]->output_buffer;
            BufferDesc& vc = nodes[c.b3]->output_buffer;
            uint64_t* km = reinterpret_cast<uint64_t*>(kc.get_data());
            uint64_t* vm = reinterpret_cast<uint64_t*>(vc.get_data());
            if (!km || !vm) return false;
            size_t clen = km[0], mx = km[1], sink = km[4];
            uint32_t hd = c.u1 & 0xFFFFu, nqh = c.u0;
            if (km[2] != 1 || vm[3] != km[3] || hd == 0 || hd > 512u) return false;
            bool has_new = (c.u1 & 0x10000u) != 0;
            size_t ng = (hd + 31) / 32;
            size_t win = anchor.params.window_size;
            bool sliding = win > 0;
            if (has_new && !sliding && clen >= mx) {
                size_t ceiling = nodes[c.b2]->params.max_cache_seq_len;
                if (!cactus_kv_cache_grow(kc, clen + 1, ceiling)
                    || !cactus_kv_cache_grow(vc, clen + 1, ceiling)) return false;
                km = reinterpret_cast<uint64_t*>(kc.get_data());
                vm = reinterpret_cast<uint64_t*>(vc.get_data());
                if (!km || !vm) return false;
                mx = km[1];
            }
            uint32_t Wn = sliding ? (uint32_t)(mx - sink - 1) : 0u;
            uint32_t Sn = sliding ? (uint32_t)sink : 0u;
            uint32_t Rn = (Wn > Sn) ? (Wn - Sn) : 1u;
            char* kb = (char*)kc.get_data();
            char* vb = (char*)vc.get_data();
            size_t slot = 0, kv_end;
            if (has_new) {
                bool wrap = sliding && clen >= (size_t)Wn;
                slot = wrap ? (size_t)(Sn + ((clen - Sn) % Rn)) : clen;
                kv_end = wrap ? (size_t)Wn : clen + 1;
            } else {
                kv_end = (sliding && clen > (size_t)Wn) ? (size_t)Wn : clen;
            }
            bool ok = cactus_metal_encode_attention_fused_i8(
                anchor.output_buffer.get_data(), data(c.a0),
                has_new ? data(c.a1) : nullptr, has_new ? data(c.a2) : nullptr,
                kb + 64, vb + 64, kb + 64 + mx * hd, vb + 64 + mx * hd,
                data(c.a3), has_new ? data(c.a4) : nullptr, has_new ? data(c.a5) : nullptr,
                data(c.b0), data(c.b1),
                nqh, hd, hd,
                0u, (uint32_t)kv_end, (uint32_t)slot, has_new ? 1u : 0u,
                c.f0, c.f1,
                mx * hd, mx * hd, mx * ng * sizeof(float), mx * ng * sizeof(float));
            if (!ok) return false;
            if (has_new) { km[0] = clen + 1; vm[0] = clen + 1; }
            return true;
        }
        case 2: {
            size_t mm[3] = { c.b0, c.b1, c.b2 };
            CactusQuantMatrix W[3];
            void* outs[3];
            const CactusQuantMatrix* Wp[3];
            for (uint32_t bi = 0; bi < c.u0; ++bi) {
                GraphNode& mn = *nodes[mm[bi]];
                auto it = map.find(mn.input_ids[1]);
                if (it == map.end()) return false;
                W[bi] = nodes[it->second]->output_buffer.to_cq_matrix();
                Wp[bi] = &W[bi];
                outs[bi] = mn.output_buffer.get_data();
                if (!outs[bi] || !c.sc[bi]) return false;
            }
            const void* x = nodes[c.a0]->output_buffer.get_data();
            if (!x) return false;
            void* codes[3] = { c.sc[0], c.sc[1], c.sc[2] };
            if (cactus_metal_encode_transform_batch(x, Wp, (int)c.u0, codes)) {
                const void* ccodes[3] = { c.sc[0], c.sc[1], c.sc[2] };
                if (!cactus_metal_encode_gemv_cat(outs, ccodes, Wp, (int)c.u0)) {
                    for (uint32_t bi = 0; bi < c.u0; ++bi)
                        if (!cactus_metal_encode_gemv_precoded(outs[bi], c.sc[bi], Wp[bi])) return false;
                }
                return true;
            }
            for (uint32_t bi = 0; bi < c.u0; ++bi)
                if (!cactus_metal_encode_quant_matmul(outs[bi], x, Wp[bi])) return false;
            return true;
        }
        case 7: {
            GraphNode& anchor = *nodes[c.b4];
            GraphNode& down = *nodes[c.b0];
            auto itd = map.find(down.input_ids[1]);
            auto itu = map.find(anchor.input_ids[1]);
            if (itd == map.end() || itu == map.end() || !c.s0) return false;
            CactusQuantMatrix Wd = nodes[itd->second]->output_buffer.to_cq_matrix();
            CactusQuantMatrix Wu = nodes[itu->second]->output_buffer.to_cq_matrix();
            const void* x = nodes[c.a0]->output_buffer.get_data();
            const void* row = nodes[c.a1]->output_buffer.get_data();
            if (row && c.u0 == 1u) row = (const char*)row + c.u1;
            void* out = anchor.output_buffer.get_data();
            if (!x || !row || !out) return false;
            if (!cactus_metal_encode_transform_gemv(c.s0, x, &Wd, row)) return false;
            if (!cactus_metal_encode_transform_gemv(out, c.s0, &Wu, nullptr)
                && !cactus_metal_encode_quant_matmul(out, c.s0, &Wu)) return false;
            return true;
        }
        case 4: {
            GraphNode& anchor = *nodes[c.b4];
            auto it = map.find(anchor.input_ids[1]);
            if (it == map.end() || !c.s0) return false;
            CactusQuantMatrix W = nodes[it->second]->output_buffer.to_cq_matrix();
            void* out = anchor.output_buffer.get_data();
            const void* gate = nodes[c.a0]->output_buffer.get_data();
            const void* up = nodes[c.a1]->output_buffer.get_data();
            if (!out || !gate || !up) return false;
            if (cactus_metal_encode_swiglu_transform(c.s0, gate, up, &W, c.f0)
                && cactus_metal_encode_gemv_precoded(out, c.s0, &W)) return true;
            size_t M = W.K;
            if (!cactus_metal_encode_swiglu(c.s0, gate, up, M, c.f0)) return false;
            return cactus_metal_encode_quant_matmul(out, c.s0, &W);
        }
        case 3: {
            GraphNode& anchor = *nodes[c.b4];
            const BufferDesc& src = nodes[c.a0]->output_buffer;
            size_t dim = anchor.output_buffer.shape.empty() ? 0 : anchor.output_buffer.shape.back();
            if (dim == 0) return false;
            size_t rows = src.total_size / dim;
            float out_scale = 1.0f;
            if (c.u1 == 1u) {
                const void* sp = nodes[c.b1]->output_buffer.get_data();
                if (!sp) return false;
                out_scale = (float)*(const __fp16*)sp;
            }
            if (c.u0 == 1u && rows == 1) {
                GraphNode& nn = *nodes[c.a5];
                if (!nn.output_buffer.get_data()) {
                    if (!c.s1) return false;
                    nn.output_buffer.set_external(c.s1);
                }
                if (cactus_metal_encode_rms_norm_add_rms(
                        anchor.output_buffer.get_data(), nn.output_buffer.get_data(),
                        src.get_data(), data(c.a1), data(c.a2), data(c.a4),
                        rows, dim, c.f0, out_scale)) return true;
            }
            if (c.u1 == 1u && c.u0 != 1u && rows == 1) {
                if (cactus_metal_encode_rms_norm_add_scale(
                        anchor.output_buffer.get_data(), src.get_data(), data(c.a1), data(c.a2),
                        rows, dim, c.f0, out_scale)) return true;
                return false;
            }
            if (c.u1 == 1u) return false;
            if (c.u0 == 1u) {
                GraphNode& nn = *nodes[c.a5];
                if (!nn.output_buffer.get_data()) {
                    if (!c.s1) return false;
                    nn.output_buffer.set_external(c.s1);
                }
                if (!cactus_metal_encode_rms_norm_add(
                        anchor.output_buffer.get_data(), src.get_data(), data(c.a1), data(c.a2),
                        rows, dim, c.f0)) return false;
                return cactus_metal_encode_rms_norm(
                    nn.output_buffer.get_data(), anchor.output_buffer.get_data(), data(c.a4),
                    rows, dim, c.f0);
            }
            return cactus_metal_encode_rms_norm_add(
                anchor.output_buffer.get_data(), src.get_data(), data(c.a1), data(c.a2),
                rows, dim, c.f0);
        }
        case 5: {
            GraphNode& anchor = *nodes[c.b4];
            const BufferDesc& wb = nodes[c.a1]->output_buffer;
            CactusQuantMatrix W = wb.to_cq_matrix();
            void* lg = anchor.output_buffer.get_data();
            if (!c.s0 || !cactus_metal_encode_quant_matmul_ortho(lg, data(c.a0), c.s0, &W)) return false;
            size_t V = anchor.output_buffer.total_size;
            if (!cactus_metal_encode_softcap(lg, lg, V, c.f0)) {
                cactus_metal_encode_scalar(3, lg, lg, V, c.f0);
                cactus_metal_encode_unary(1, lg, lg, V);
                cactus_metal_encode_scalar(2, lg, lg, V, c.f0);
            }
            return cactus_graph_metal_tail(lg, V);
        }
        case 8: {
            GraphNode& anchor = *nodes[c.b4];
            const char* pa = (const char*)nodes[c.a0]->output_buffer.get_data();
            const char* pb = (const char*)nodes[c.a1]->output_buffer.get_data();
            void* out = anchor.output_buffer.get_data();
            if (!pa || !pb || !out) return false;
            return cactus_metal_encode_gemm_batch(out, pa + c.a2 * (c.b3 ? 4 : 2), pb + c.a3 * 2,
                c.u1, (uint32_t)c.b0, (uint32_t)c.b1, c.u0, (int)c.b2, (int)c.b3);
        }
        case 9: {
            GraphNode& anchor = *nodes[c.b4];
            const char* px = (const char*)nodes[c.a0]->output_buffer.get_data();
            const char* pw = (const char*)nodes[c.a1]->output_buffer.get_data();
            void* out = anchor.output_buffer.get_data();
            if (!px || !pw || !out) return false;
            return cactus_metal_encode_conv1d_dw(out, px + c.a2 * 2, pw + c.a3 * 2,
                c.u0, c.u1, (uint32_t)c.b0, (uint32_t)c.b1, (uint32_t)c.b2);
        }
        case 10: {
            GraphNode& anchor = *nodes[c.b4];
            GraphNode& tn = *nodes[c.a1];
            const char* pp = (const char*)nodes[c.a0]->output_buffer.get_data();
            void* out = anchor.output_buffer.get_data();
            if (!pp || !out) return false;
            long ii = -1;
            {
                auto it = map.find(tn.input_ids[0]);
                if (it == map.end()) return false;
                ii = (long)it->second;
            }
            const auto& ish = nodes[(size_t)ii]->output_buffer.shape;
            const auto& osh = tn.output_buffer.shape;
            const auto& perm = tn.params.permutation;
            uint32_t ndim = (uint32_t)perm.size();
            uint32_t ax = c.u1;
            if (ndim == 0 || ndim + 1 > 8 || ish.size() != ndim || osh.size() != ndim || ax > ndim) return false;
            size_t istr[8];
            istr[ndim - 1] = 1;
            for (size_t d = ndim - 1; d > 0; --d) istr[d - 1] = istr[d] * ish[d];
            uint32_t oshape[8], sstride[8];
            for (uint32_t d = 0; d < ax; ++d) {
                oshape[d] = (uint32_t)osh[d];
                sstride[d] = (uint32_t)istr[perm[d]];
            }
            oshape[ax] = c.u0; sstride[ax] = (uint32_t)c.b0;
            for (uint32_t d = ax; d < ndim; ++d) {
                oshape[d + 1] = (uint32_t)osh[d];
                sstride[d + 1] = (uint32_t)istr[perm[d]];
            }
            size_t total = (size_t)c.u0 * tn.output_buffer.total_size;
            return cactus_metal_encode_strided_copy(out, pp + c.a2 * 2, oshape, sstride, ndim + 1,
                (uint32_t)total, 0, nodes[c.a0]->output_buffer.byte_size - c.a2 * 2, anchor.output_buffer.byte_size);
        }
        case 11: {
            GraphNode& anchor = *nodes[c.b4];
            const void* in = nodes[c.a0]->output_buffer.get_data();
            void* out = anchor.output_buffer.get_data();
            if (!in || !out) return false;
            const void* sides[3] = { nullptr, nullptr, nullptr };
            size_t side_elems[3] = { 0, 0, 0 };
            size_t side_idx[3] = { c.a1, c.a2, c.a3 };
            for (uint32_t si = 0; si < c.u1; ++si) {
                sides[si] = nodes[side_idx[si]]->output_buffer.get_data();
                side_elems[si] = nodes[side_idx[si]]->output_buffer.total_size;
                if (!sides[si]) return false;
            }
            const auto& blob = p->blobs[c.b0];
            size_t inner = anchor.output_buffer.shape.empty() ? 1 : anchor.output_buffer.shape.back();
            return cactus_metal_encode_elemwise_chain(out, in, blob.data(), c.u0,
                sides[0], sides[1], sides[2], side_elems, anchor.output_buffer.total_size,
                (uint32_t)c.b1, (uint32_t)inner);
        }
        case 12: {
            GraphNode& anchor = *nodes[c.b4];
            const void* x = nodes[c.a0]->output_buffer.get_data();
            const void* res = nodes[c.a1]->output_buffer.get_data();
            void* ysum = nodes[c.a2]->output_buffer.get_data();
            const void* w = nodes[c.a3]->output_buffer.get_data();
            void* ynorm = anchor.output_buffer.get_data();
            if (!x || !res || !ysum || !w || !ynorm) return false;
            return cactus_metal_encode_rms_norm_add_rows(ysum, ynorm, x, res, w,
                c.u0, c.u1, c.f0, (int)c.b0);
        }
        case 13: {
            GraphNode& anchor = *nodes[c.b4];
            const void* x = nodes[c.a0]->output_buffer.get_data();
            const void* w = nodes[c.a1]->output_buffer.get_data();
            const void* b = nodes[c.a2]->output_buffer.get_data();
            void* y = anchor.output_buffer.get_data();
            if (!x || !w || !b || !y) return false;
            return cactus_metal_encode_gemv_bias(y, x, w, b, c.u0, c.u1, (int)c.b0);
        }
        case 14: {
            GraphNode& anchor = *nodes[c.b4];
            const void* x = nodes[c.a0]->output_buffer.get_data();
            const void* cs = nodes[c.a1]->output_buffer.get_data();
            const void* sn = nodes[c.a2]->output_buffer.get_data();
            void* y = anchor.output_buffer.get_data();
            if (!x || !cs || !sn || !y) return false;
            return cactus_metal_encode_rope_pair(y, x, cs, sn, c.u0, c.u1);
        }
        case 15: {
            GraphNode& anchor = *nodes[c.b4];
            const void* x = nodes[c.a0]->output_buffer.get_data();
            const void* w = nodes[c.a1]->output_buffer.get_data();
            void* y = anchor.output_buffer.get_data();
            if (!x || !w || !y) return false;
            return cactus_metal_encode_rms_norm_scale(y, x, w, c.u0, c.u1, c.f0, c.f1);
        }
        case 16: {
            GraphNode& anchor = *nodes[c.b4];
            const void* lg = nodes[c.a0]->output_buffer.get_data();
            void* probs = nodes[c.a1]->output_buffer.get_data();
            void* tk = anchor.output_buffer.get_data();
            if (!lg || !probs || !tk) return false;
            return cactus_metal_encode_softmax_topk(probs, tk, lg, c.u0, c.u1, c.b0, c.f0);
        }
        case 18: {
            GraphNode& anchor = *nodes[c.b4];
            const void* a = nodes[c.a0]->output_buffer.get_data();
            const void* wa = nodes[c.a1]->output_buffer.get_data();
            const void* b = nodes[c.a2]->output_buffer.get_data();
            const void* wb = nodes[c.a3]->output_buffer.get_data();
            void* y = anchor.output_buffer.get_data();
            if (!a || !wa || !b || !wb || !y) return false;
            return cactus_metal_encode_rms2_add_clip(y, a, wa, b, wb, c.u1, c.f0, c.f1);
        }
        case 17: {
            GraphNode& anchor = *nodes[c.b4];
            const void* x = nodes[c.a0]->output_buffer.get_data();
            const void* w = nodes[c.a3]->output_buffer.get_data();
            const void* cs = nodes[c.a1]->output_buffer.get_data();
            const void* sn = nodes[c.a2]->output_buffer.get_data();
            void* y = anchor.output_buffer.get_data();
            if (!x || !w || !cs || !sn || !y) return false;
            return cactus_metal_encode_rope_pair_rms(y, x, w, cs, sn, c.u0, c.u1, c.f0);
        }
        default: return false;
    }
}
