uint lane = thread_index_in_simdgroup;
uint row = thread_position_in_grid.x / 32;
uint pair = thread_position_in_grid.y;
if (row >= N) return;
uint expert = EXPERTS ? indices[pair] : 0;
uint full_row = expert * N + row;
uint token = BROADCAST ? pair / TOPK : pair;
constexpr uint SLOTS = 1u << CQ_BS;
constexpr uint ODD = (1u << CQ_BO) * 128 * 8;
constexpr uint EVEN2 = ODD + (1u << CQ_B1) * 4;
device const float4* values = reinterpret_cast<device const float4*>(lut);
device const half4* x = reinterpret_cast<device const half4*>(permuted);
float sum = 0;
float ns = float(scales[full_row / SCALE_ROWS]);
for (uint g = lane / 8; g < K / 128; g += 4) {
    uint group = full_row * (K / 128) + g;
    float d = 0;
    for (uint stage = 0; stage < (CQ_RATE == 3 ? 2 : 1); ++stage) {
        uint cw = cq_code(codes, group, lane % 8, stage);
        uint i2 = cw & ((1u << CQ_B2) - 1), i1 = (cw >> CQ_B2) & ((1u << CQ_B1) - 1);
        uint s7 = (cw >> (CQ_B2 + CQ_B1)) & 127;
        uint o = (cw >> (CQ_B2 + CQ_B1 + 7)) & ((1u << CQ_BO) - 1);
        uint slot = cw >> (CQ_B2 + CQ_B1 + 7 + CQ_BO);
        uint odd = (o * 128 + s7) * 2;
        float4 v0 = values[odd], v1 = values[odd + 1];
        uint parity = parities[i1] ^ parities[(1u << CQ_B1) + i2];
        if (parity) v1.w = -v1.w;
        float4 v2 = values[ODD / 4 + i1], v3 = values[EVEN2 / 4 + i2];
        uint xp = ((token * (K / 16) + g * 8 + lane % 8) * SLOTS + slot) * 4;
        float product = dot(v0, float4(x[xp])) + dot(v1, float4(x[xp + 1])) + dot(v2, float4(x[xp + 2])) + dot(v3, float4(x[xp + 3]));
#if CQ_RATE == 3
        d += stage == 0 ? product / 0.93f : product * 0.28f;
#else
        d = product;
#endif
    }
    sum += d * cq_fp8(norms[group]) * ns;
}
sum = metal::simd_sum(sum) * cq_fp8(c[full_row]);
if (lane == 0) {
    OUT value = static_cast<OUT>(sum);
    if constexpr (RELU2) {
        float activated = max(float(value), 0.0f);
        value = static_cast<OUT>(activated * activated);
    }
    out[pair * N + row] = value;
}
