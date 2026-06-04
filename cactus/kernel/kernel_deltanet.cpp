#include "kernel.h"
#include "kernel_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

inline float safe_exp_gate(float gate_log) {
    const float clamped = std::max(-20.0f, std::min(6.0f, gate_log));
    return std::exp(clamped);
}

inline void fold_state_scale(std::vector<float>& state, float factor) {
    const float32x4_t f4 = vdupq_n_f32(factor);
    size_t i = 0;
    for (; i + 8 <= state.size(); i += 8) {
        vst1q_f32(&state[i],     vmulq_f32(vld1q_f32(&state[i]),     f4));
        vst1q_f32(&state[i + 4], vmulq_f32(vld1q_f32(&state[i + 4]), f4));
    }
    for (; i < state.size(); ++i) state[i] *= factor;
}

struct GatedDeltaChunkScratch {
    std::vector<float> state;
    std::vector<float> m_chunk;
    std::vector<float> q_chunk;
    std::vector<float> k_chunk;
    std::vector<float> v_chunk;
    std::vector<float> g_chunk;
    std::vector<float> beta_chunk;
    std::vector<float> n0_proj;
    std::vector<float> delta_chunk;
    std::vector<float> gram;
    std::vector<float> p_prev;
    std::vector<float> p_curr;
    std::vector<float> inv_p_curr;
    std::vector<float> coeff;
    std::vector<float> out_acc;

    void ensure(size_t k_dim, size_t v_dim, size_t c_max) {
        const size_t kv = k_dim * v_dim;
        if (state.size() < kv) state.resize(kv);
        if (m_chunk.size() < kv) m_chunk.resize(kv);
        if (q_chunk.size() < c_max * k_dim) q_chunk.resize(c_max * k_dim);
        if (k_chunk.size() < c_max * k_dim) k_chunk.resize(c_max * k_dim);
        if (v_chunk.size() < c_max * v_dim) v_chunk.resize(c_max * v_dim);
        if (g_chunk.size() < c_max) g_chunk.resize(c_max);
        if (beta_chunk.size() < c_max) beta_chunk.resize(c_max);
        if (n0_proj.size() < c_max * v_dim) n0_proj.resize(c_max * v_dim);
        if (delta_chunk.size() < c_max * v_dim) delta_chunk.resize(c_max * v_dim);
        if (gram.size() < c_max * c_max) gram.resize(c_max * c_max);
        if (p_prev.size() < c_max) p_prev.resize(c_max);
        if (p_curr.size() < c_max) p_curr.resize(c_max);
        if (inv_p_curr.size() < c_max) inv_p_curr.resize(c_max);
        if (coeff.size() < c_max * c_max) coeff.resize(c_max * c_max);
        if (out_acc.size() < v_dim) out_acc.resize(v_dim);
    }
};

thread_local GatedDeltaChunkScratch g_gated_deltanet_chunk_scratch;

inline size_t tuned_gated_deltanet_chunk_size(size_t requested_chunk, size_t k_dim, size_t v_dim) {
    size_t chunk = requested_chunk == 0 ? 64 : requested_chunk;

    const char* env_chunk = std::getenv("CACTUS_GATED_DELTANET_CHUNK_SIZE");
    if (env_chunk != nullptr) {
        long parsed = std::strtol(env_chunk, nullptr, 10);
        if (parsed > 1) {
            chunk = static_cast<size_t>(parsed);
        }
    } else {
        (void)k_dim;
        (void)v_dim;
        chunk = std::min<size_t>(chunk, 16);
    }

    return std::max<size_t>(2, chunk);
}

void gated_deltanet_step(
    const __fp16* q_ptr,
    const __fp16* k_ptr,
    const __fp16* v_ptr,
    float gate_log,
    float beta,
    float scale,
    size_t k_dim,
    size_t v_dim,
    std::vector<float>& state,
    float& state_scale,
    std::vector<float>& proj,
    std::vector<float>& delta,
    __fp16* out_ptr) {
    const float gate_log_safe = std::isfinite(gate_log) ? gate_log : -20.0f;
    const float beta_safe = std::isfinite(beta) ? std::min(1.0f, std::max(0.0f, beta)) : 0.0f;

    if (proj.size() != v_dim) {
        proj.assign(v_dim, 0.0f);
    }
    if (delta.size() != v_dim) {
        delta.assign(v_dim, 0.0f);
    }

    const float prev_scale = state_scale;
    std::fill(proj.begin(), proj.end(), 0.0f);

    for (size_t kd = 0; kd < k_dim; ++kd) {
        if (kd + 2 < k_dim) __builtin_prefetch(state.data() + (kd + 2) * v_dim, 0, 3);
        const float k_val = static_cast<float>(k_ptr[kd]);
        const float* state_row = state.data() + kd * v_dim;
        size_t vd = 0;
        const float32x4_t k4 = vdupq_n_f32(k_val);
        for (; vd + 4 <= v_dim; vd += 4) {
            float32x4_t p = vld1q_f32(proj.data() + vd);
            float32x4_t s = vld1q_f32(state_row + vd);
            p = vfmaq_f32(p, s, k4);
            vst1q_f32(proj.data() + vd, p);
        }
        for (; vd < v_dim; ++vd) {
            proj[vd] += state_row[vd] * k_val;
        }
    }
    {
        size_t vd = 0;
        const float32x4_t scale4 = vdupq_n_f32(prev_scale);
        for (; vd + 4 <= v_dim; vd += 4) {
            float32x4_t p = vld1q_f32(proj.data() + vd);
            p = vmulq_f32(p, scale4);
            vst1q_f32(proj.data() + vd, p);
        }
        for (; vd < v_dim; ++vd) {
            proj[vd] *= prev_scale;
        }
    }

    for (size_t vd = 0; vd < v_dim; ++vd) {
        const float v_val = static_cast<float>(v_ptr[vd]);
        delta[vd] = (v_val - proj[vd]) * beta_safe;
    }

    state_scale = prev_scale * safe_exp_gate(gate_log_safe);
    if (!std::isfinite(state_scale) ||
        std::fabs(state_scale) < 1e-8f ||
        std::fabs(state_scale) > 1e8f) {
        fold_state_scale(state, state_scale);
        state_scale = 1.0f;
    }
    const float inv_scale = 1.0f / state_scale;

    for (size_t kd = 0; kd < k_dim; ++kd) {
        if (kd + 2 < k_dim) __builtin_prefetch(state.data() + (kd + 2) * v_dim, 1, 3);
        const float k_scaled = static_cast<float>(k_ptr[kd]) * inv_scale;
        float* state_row = state.data() + kd * v_dim;
        size_t vd = 0;
        const float32x4_t k4 = vdupq_n_f32(k_scaled);
        for (; vd + 4 <= v_dim; vd += 4) {
            float32x4_t s = vld1q_f32(state_row + vd);
            float32x4_t d = vld1q_f32(delta.data() + vd);
            s = vfmaq_f32(s, d, k4);
            vst1q_f32(state_row + vd, s);
        }
        for (; vd < v_dim; ++vd) {
            state_row[vd] += k_scaled * delta[vd];
        }
        for (size_t vd = 0; vd < v_dim; ++vd) {
            if (!std::isfinite(state_row[vd])) {
                state_row[vd] = 0.0f;
            }
        }
    }

    std::fill(proj.begin(), proj.end(), 0.0f);
    for (size_t kd = 0; kd < k_dim; ++kd) {
        if (kd + 2 < k_dim) __builtin_prefetch(state.data() + (kd + 2) * v_dim, 0, 3);
        const float q_val = static_cast<float>(q_ptr[kd]);
        const float* state_row = state.data() + kd * v_dim;
        size_t vd = 0;
        const float32x4_t q4 = vdupq_n_f32(q_val);
        for (; vd + 4 <= v_dim; vd += 4) {
            float32x4_t o = vld1q_f32(proj.data() + vd);
            float32x4_t s = vld1q_f32(state_row + vd);
            o = vfmaq_f32(o, s, q4);
            vst1q_f32(proj.data() + vd, o);
        }
        for (; vd < v_dim; ++vd) {
            proj[vd] += state_row[vd] * q_val;
        }
    }

    const float out_scale = state_scale * scale;
    for (size_t vd = 0; vd < v_dim; ++vd) {
        float acc = proj[vd];
        if (!std::isfinite(acc)) {
            acc = 0.0f;
        }
        out_ptr[vd] = static_cast<__fp16>(acc * out_scale);
    }
}

void gated_deltanet_prefill_old_f16(
    const __fp16* q_data,
    const __fp16* k_data,
    const __fp16* v_data,
    const __fp16* g_data,
    const __fp16* b_data,
    const __fp16* s_data,
    __fp16* out,
    size_t B,
    size_t T,
    size_t Hq,
    size_t Hv,
    size_t K,
    size_t V,
    float scale) {
    const size_t out_seq = T + K;
    const size_t qk_repeat = Hv / Hq;

    CactusThreading::parallel_for(B * Hv, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
        [&](size_t bh_start, size_t bh_end) {
            std::vector<float> state(K * V);
            std::vector<float> proj(V);
            std::vector<float> delta(V);

            for (size_t bh = bh_start; bh < bh_end; ++bh) {
                const size_t batch = bh / Hv;
                const size_t v_head = bh % Hv;
                const size_t qk_head = v_head / qk_repeat;

                for (size_t kd = 0; kd < K; ++kd) {
                    for (size_t vd = 0; vd < V; ++vd) {
                        const size_t s_idx = (((batch * K + kd) * Hv + v_head) * V + vd);
                        state[kd * V + vd] = static_cast<float>(s_data[s_idx]);
                    }
                }

                float state_scale = 1.0f;
                for (size_t t = 0; t < T; ++t) {
                    const size_t qkv_k_base = (((batch * T + t) * Hq + qk_head) * K);
                    const size_t qkv_v_base = (((batch * T + t) * Hv + v_head) * V);
                    const size_t gb_idx = ((batch * T + t) * Hv + v_head);

                    gated_deltanet_step(
                        q_data + qkv_k_base,
                        k_data + qkv_k_base,
                        v_data + qkv_v_base,
                        static_cast<float>(g_data[gb_idx]),
                        static_cast<float>(b_data[gb_idx]),
                        scale,
                        K,
                        V,
                        state,
                        state_scale,
                        proj,
                        delta,
                        out + (((batch * out_seq + t) * Hv + v_head) * V));
                }

                fold_state_scale(state, state_scale);
                for (size_t kd = 0; kd < K; ++kd) {
                    for (size_t vd = 0; vd < V; ++vd) {
                        const size_t out_idx = (((batch * out_seq + (T + kd)) * Hv + v_head) * V + vd);
                        out[out_idx] = static_cast<__fp16>(state[kd * V + vd]);
                    }
                }
            }
        });
}

void gated_deltanet_prefill_chunked_f16(
    const __fp16* q_data,
    const __fp16* k_data,
    const __fp16* v_data,
    const __fp16* g_data,
    const __fp16* b_data,
    const __fp16* s_data,
    __fp16* out,
    size_t B,
    size_t T,
    size_t Hq,
    size_t Hv,
    size_t K,
    size_t V,
    size_t chunk_size,
    float scale) {
    const size_t out_seq = T + K;
    const size_t qk_repeat = Hv / Hq;
    constexpr float kMinAbs = 1e-12f;

    CactusThreading::parallel_for(B * Hv, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
        [&](size_t bh_start, size_t bh_end) {
            auto& ws = g_gated_deltanet_chunk_scratch;
            ws.ensure(K, V, chunk_size);

            float* state = ws.state.data();
            float* m_chunk = ws.m_chunk.data();
            float* q_chunk = ws.q_chunk.data();
            float* k_chunk = ws.k_chunk.data();
            float* v_chunk = ws.v_chunk.data();
            float* g_chunk = ws.g_chunk.data();
            float* beta_chunk = ws.beta_chunk.data();
            float* n0_proj = ws.n0_proj.data();
            float* delta_chunk = ws.delta_chunk.data();
            float* gram = ws.gram.data();
            float* p_prev = ws.p_prev.data();
            float* p_curr = ws.p_curr.data();
            float* inv_p_curr = ws.inv_p_curr.data();
            float* coeff = ws.coeff.data();
            float* out_acc = ws.out_acc.data();

            for (size_t bh = bh_start; bh < bh_end; ++bh) {
                const size_t batch = bh / Hv;
                const size_t v_head = bh % Hv;
                const size_t qk_head = v_head / qk_repeat;

                for (size_t kd = 0; kd < K; ++kd) {
                    const size_t s_idx = (((batch * K + kd) * Hv + v_head) * V);
                    cactus_fp16_to_fp32(s_data + s_idx, state + kd * V, V);
                }

                for (size_t t0 = 0; t0 < T; t0 += chunk_size) {
                    const size_t C = std::min(chunk_size, T - t0);
                    const size_t CV = C * V;
                    const size_t CC = C * C;

                    std::fill(n0_proj, n0_proj + CV, 0.0f);
                    std::fill(delta_chunk, delta_chunk + CV, 0.0f);
                    std::fill(gram, gram + CC, 0.0f);
                    std::fill(coeff, coeff + CC, 0.0f);
                    std::fill(m_chunk, m_chunk + (K * V), 0.0f);

                    // Fused fp16→fp32 conversion + n0_proj: k_chunk stays in L1
                    for (size_t ct = 0; ct < C; ++ct) {
                        const size_t t = t0 + ct;
                        const size_t qkv_k_base = (((batch * T + t) * Hq + qk_head) * K);
                        const size_t qkv_v_base = (((batch * T + t) * Hv + v_head) * V);
                        const size_t gb_idx = ((batch * T + t) * Hv + v_head);

                        // Inline fp16→fp32 for q
                        {
                            const __fp16* src = q_data + qkv_k_base;
                            float* dst = q_chunk + ct * K;
                            size_t i = 0;
                            for (; i + 8 <= K; i += 8) {
                                float16x8_t in = vld1q_f16(src + i);
                                vst1q_f32(dst + i,     vcvt_f32_f16(vget_low_f16(in)));
                                vst1q_f32(dst + i + 4, vcvt_f32_f16(vget_high_f16(in)));
                            }
                            for (; i < K; ++i) dst[i] = static_cast<float>(src[i]);
                        }

                        // Inline fp16→fp32 for k
                        {
                            const __fp16* src = k_data + qkv_k_base;
                            float* dst = k_chunk + ct * K;
                            size_t i = 0;
                            for (; i + 8 <= K; i += 8) {
                                float16x8_t in = vld1q_f16(src + i);
                                vst1q_f32(dst + i,     vcvt_f32_f16(vget_low_f16(in)));
                                vst1q_f32(dst + i + 4, vcvt_f32_f16(vget_high_f16(in)));
                            }
                            for (; i < K; ++i) dst[i] = static_cast<float>(src[i]);
                        }

                        // Inline fp16→fp32 for v
                        {
                            const __fp16* src = v_data + qkv_v_base;
                            float* dst = v_chunk + ct * V;
                            size_t i = 0;
                            for (; i + 8 <= V; i += 8) {
                                float16x8_t in = vld1q_f16(src + i);
                                vst1q_f32(dst + i,     vcvt_f32_f16(vget_low_f16(in)));
                                vst1q_f32(dst + i + 4, vcvt_f32_f16(vget_high_f16(in)));
                            }
                            for (; i < V; ++i) dst[i] = static_cast<float>(src[i]);
                        }

                        const float gate_log = static_cast<float>(g_data[gb_idx]);
                        const float beta = static_cast<float>(b_data[gb_idx]);
                        g_chunk[ct] = safe_exp_gate(std::isfinite(gate_log) ? gate_log : -20.0f);
                        beta_chunk[ct] = std::isfinite(beta) ? std::min(1.0f, std::max(0.0f, beta)) : 0.0f;

                        // n0_proj fused here so k_chunk[ct] is still in L1
                        const float* k_row = k_chunk + ct * K;
                        float* p_row = n0_proj + ct * V;
                        for (size_t kd = 0; kd < K; ++kd) {
                            const float kval = k_row[kd];
                            const float* s_row = state + kd * V;
                            size_t vd = 0;
                            const float32x4_t kv4 = vdupq_n_f32(kval);
                            for (; vd + 8 <= V; vd += 8) {
                                float32x4_t p4 = vld1q_f32(p_row + vd);
                                const float32x4_t slo = vld1q_f32(s_row + vd);
                                const float32x4_t shi = vld1q_f32(s_row + vd + 4);
                                p4 = vfmaq_f32(p4, slo, kv4);
                                vst1q_f32(p_row + vd, p4);
                                p4 = vld1q_f32(p_row + vd + 4);
                                p4 = vfmaq_f32(p4, shi, kv4);
                                vst1q_f32(p_row + vd + 4, p4);
                            }
                            for (; vd < V; ++vd) {
                                p_row[vd] += kval * s_row[vd];
                            }
                        }
                    }

                    for (size_t t = 0; t < C; ++t) {
                        const float* kt = k_chunk + t * K;
                        for (size_t j = 0; j < t; ++j) {
                            const float* kj = k_chunk + j * K;
                            float dot = 0.0f;
                            size_t kd = 0;
                            float32x4_t acc4 = vdupq_n_f32(0.0f);
                            for (; kd + 8 <= K; kd += 8) {
                                const float32x4_t al = vld1q_f32(kt + kd);
                                const float32x4_t ah = vld1q_f32(kt + kd + 4);
                                const float32x4_t bl = vld1q_f32(kj + kd);
                                const float32x4_t bh = vld1q_f32(kj + kd + 4);
                                acc4 = vfmaq_f32(acc4, al, bl);
                                acc4 = vfmaq_f32(acc4, ah, bh);
                            }
                            dot += vaddvq_f32(acc4);
                            for (; kd < K; ++kd) {
                                dot += kt[kd] * kj[kd];
                            }
                            gram[t * C + j] = dot;
                        }
                    }

                    float running_p = 1.0f;
                    for (size_t t = 0; t < C; ++t) {
                        p_prev[t] = running_p;
                        running_p *= g_chunk[t];
                        p_curr[t] = running_p;
                        inv_p_curr[t] = (std::fabs(running_p) > kMinAbs) ? (1.0f / running_p) : 0.0f;
                    }

                    for (size_t t = 0; t < C; ++t) {
                        const float pp = p_prev[t];
                        float* coeff_row = coeff + t * C;
                        for (size_t j = 0; j < t; ++j) {
                            const float inv_pc = inv_p_curr[j];
                            coeff_row[j] = inv_pc == 0.0f ? 0.0f : (pp * inv_pc * gram[t * C + j]);
                        }
                    }

                    for (size_t t = 0; t < C; ++t) {
                        const float pp = p_prev[t];
                        const float bt = beta_chunk[t];
                        const float* n0_row = n0_proj + t * V;
                        const float* v_row = v_chunk + t * V;
                        float* d_row = delta_chunk + t * V;

                        size_t vd = 0;
                        const float32x4_t pp4 = vdupq_n_f32(pp);
                        const float32x4_t bt4 = vdupq_n_f32(bt);
                        for (; vd + 8 <= V; vd += 8) {
                            float32x4_t acc_lo = vmulq_f32(vld1q_f32(n0_row + vd), pp4);
                            float32x4_t acc_hi = vmulq_f32(vld1q_f32(n0_row + vd + 4), pp4);
                            for (size_t j = 0; j < t; ++j) {
                                const float c = coeff[t * C + j];
                                const float* dj = delta_chunk + j * V + vd;
                                acc_lo = vmlaq_n_f32(acc_lo, vld1q_f32(dj), c);
                                acc_hi = vmlaq_n_f32(acc_hi, vld1q_f32(dj + 4), c);
                            }
                            vst1q_f32(d_row + vd,     vmulq_f32(vsubq_f32(vld1q_f32(v_row + vd),     acc_lo), bt4));
                            vst1q_f32(d_row + vd + 4, vmulq_f32(vsubq_f32(vld1q_f32(v_row + vd + 4), acc_hi), bt4));
                        }
                        for (; vd < V; ++vd) {
                            float acc = pp * n0_row[vd];
                            for (size_t j = 0; j < t; ++j) {
                                acc += coeff[t * C + j] * delta_chunk[j * V + vd];
                            }
                            d_row[vd] = bt * (v_row[vd] - acc);
                        }
                    }

                    running_p = 1.0f;
                    for (size_t t = 0; t < C; ++t) {
                        const float gt = g_chunk[t];
                        const float* k_row = k_chunk + t * K;
                        const float* q_row = q_chunk + t * K;
                        const float* d_row = delta_chunk + t * V;

                        running_p *= gt;
                        const float p_run = running_p;

                        for (size_t kd = 0; kd < K; ++kd) {
                            const float kval = k_row[kd];
                            float* m_row = m_chunk + kd * V;
                            size_t vd = 0;
                            const float32x4_t gt4 = vdupq_n_f32(gt);
                            const float32x4_t kv4 = vdupq_n_f32(kval);
                            for (; vd + 8 <= V; vd += 8) {
                                float32x4_t ml = vld1q_f32(m_row + vd);
                                float32x4_t mh = vld1q_f32(m_row + vd + 4);
                                const float32x4_t dl = vld1q_f32(d_row + vd);
                                const float32x4_t dh = vld1q_f32(d_row + vd + 4);
                                ml = vmulq_f32(ml, gt4);
                                mh = vmulq_f32(mh, gt4);
                                ml = vfmaq_f32(ml, dl, kv4);
                                mh = vfmaq_f32(mh, dh, kv4);
                                vst1q_f32(m_row + vd, ml);
                                vst1q_f32(m_row + vd + 4, mh);
                            }
                            for (; vd < V; ++vd) {
                                m_row[vd] = gt * m_row[vd] + kval * d_row[vd];
                            }
                        }

                        const size_t out_base = (((batch * out_seq + (t0 + t)) * Hv + v_head) * V);
                        std::fill(out_acc, out_acc + V, 0.0f);
                        for (size_t kd = 0; kd < K; ++kd) {
                            const float qval = q_row[kd];
                            const float* s_row = state + kd * V;
                            const float* m_row = m_chunk + kd * V;
                            size_t vd = 0;
                            const float32x4_t q4 = vdupq_n_f32(qval);
                            const float32x4_t p4 = vdupq_n_f32(p_run);
                            for (; vd + 8 <= V; vd += 8) {
                                float32x4_t acc4 = vld1q_f32(out_acc + vd);
                                float32x4_t acc4h = vld1q_f32(out_acc + vd + 4);
                                const float32x4_t sl = vld1q_f32(s_row + vd);
                                const float32x4_t sh = vld1q_f32(s_row + vd + 4);
                                const float32x4_t ml = vld1q_f32(m_row + vd);
                                const float32x4_t mh = vld1q_f32(m_row + vd + 4);
                                const float32x4_t nl = vfmaq_f32(ml, sl, p4);
                                const float32x4_t nh = vfmaq_f32(mh, sh, p4);
                                acc4 = vfmaq_f32(acc4, nl, q4);
                                acc4h = vfmaq_f32(acc4h, nh, q4);
                                vst1q_f32(out_acc + vd, acc4);
                                vst1q_f32(out_acc + vd + 4, acc4h);
                            }
                            for (; vd < V; ++vd) {
                                out_acc[vd] += qval * (p_run * s_row[vd] + m_row[vd]);
                            }
                        }
                        for (size_t vd = 0; vd < V; ++vd) {
                            out[out_base + vd] = static_cast<__fp16>(out_acc[vd] * scale);
                        }
                    }

                    const float p_end = running_p;
                    for (size_t kd = 0; kd < K; ++kd) {
                        float* s_row = state + kd * V;
                        const float* m_row = m_chunk + kd * V;
                        size_t vd = 0;
                        const float32x4_t p4 = vdupq_n_f32(p_end);
                        for (; vd + 8 <= V; vd += 8) {
                            float32x4_t sl = vld1q_f32(s_row + vd);
                            float32x4_t sh = vld1q_f32(s_row + vd + 4);
                            const float32x4_t ml = vld1q_f32(m_row + vd);
                            const float32x4_t mh = vld1q_f32(m_row + vd + 4);
                            sl = vfmaq_f32(ml, sl, p4);
                            sh = vfmaq_f32(mh, sh, p4);
                            vst1q_f32(s_row + vd, sl);
                            vst1q_f32(s_row + vd + 4, sh);
                        }
                        for (; vd < V; ++vd) {
                            s_row[vd] = p_end * s_row[vd] + m_row[vd];
                        }
                    }
                }

                for (size_t kd = 0; kd < K; ++kd) {
                    const size_t out_idx = (((batch * out_seq + (T + kd)) * Hv + v_head) * V);
                    cactus_fp32_to_fp16(state + kd * V, out + out_idx, V);
                }
            }
        });
}

} // namespace

void cactus_gated_deltanet_decode_f16(
    const __fp16* q_data,
    const __fp16* k_data,
    const __fp16* v_data,
    const __fp16* g_data,
    const __fp16* b_data,
    const __fp16* s_data,
    __fp16* out,
    size_t B,
    size_t Hq,
    size_t Hv,
    size_t K,
    size_t V,
    float scale) {
    const size_t qk_repeat = Hv / Hq;
    const size_t out_seq = 1 + K;

    CactusThreading::parallel_for(B * Hv, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
        [&](size_t bh_start, size_t bh_end) {
            std::vector<float> state(K * V);
            std::vector<float> proj(V);
            std::vector<float> delta(V);

            for (size_t bh = bh_start; bh < bh_end; ++bh) {
                const size_t batch = bh / Hv;
                const size_t v_head = bh % Hv;
                const size_t qk_head = v_head / qk_repeat;

                for (size_t kd = 0; kd < K; ++kd) {
                    for (size_t vd = 0; vd < V; ++vd) {
                        const size_t s_idx = (((batch * K + kd) * Hv + v_head) * V + vd);
                        state[kd * V + vd] = static_cast<float>(s_data[s_idx]);
                    }
                }

                float state_scale = 1.0f;
                const size_t qk_base = ((batch * Hq + qk_head) * K);
                const size_t v_base = ((batch * Hv + v_head) * V);
                const float gate_log = static_cast<float>(g_data[batch * Hv + v_head]);
                const float beta = static_cast<float>(b_data[batch * Hv + v_head]);

                gated_deltanet_step(
                    q_data + qk_base,
                    k_data + qk_base,
                    v_data + v_base,
                    gate_log,
                    beta,
                    scale,
                    K,
                    V,
                    state,
                    state_scale,
                    proj,
                    delta,
                    out + (((batch * out_seq) * Hv + v_head) * V));

                fold_state_scale(state, state_scale);
                for (size_t kd = 0; kd < K; ++kd) {
                    for (size_t vd = 0; vd < V; ++vd) {
                        const size_t out_idx = (((batch * out_seq + (1 + kd)) * Hv + v_head) * V + vd);
                        out[out_idx] = static_cast<__fp16>(state[kd * V + vd]);
                    }
                }
            }
        });
}

void cactus_gated_deltanet_prefill_f16(
    const __fp16* q_data,
    const __fp16* k_data,
    const __fp16* v_data,
    const __fp16* g_data,
    const __fp16* b_data,
    const __fp16* s_data,
    __fp16* out,
    size_t B,
    size_t T,
    size_t Hq,
    size_t Hv,
    size_t K,
    size_t V,
    size_t requested_chunk_size,
    float scale) {
    const char* force_old = std::getenv("CACTUS_GATED_DELTANET_PREFILL_OLD");
    if (force_old != nullptr && std::atoi(force_old) != 0) {
        gated_deltanet_prefill_old_f16(q_data, k_data, v_data, g_data, b_data, s_data, out,
                                       B, T, Hq, Hv, K, V, scale);
        return;
    }

    if (requested_chunk_size <= 1) {
        gated_deltanet_prefill_old_f16(q_data, k_data, v_data, g_data, b_data, s_data, out,
                                       B, T, Hq, Hv, K, V, scale);
        return;
    }

    const size_t chunk_size = tuned_gated_deltanet_chunk_size(requested_chunk_size, K, V);
    gated_deltanet_prefill_chunked_f16(q_data, k_data, v_data, g_data, b_data, s_data, out,
                                       B, T, Hq, Hv, K, V, chunk_size, scale);
}
