uint lane = thread_index_in_simdgroup;
uint row = thread_position_in_grid.x / 32;
uint pair = thread_position_in_grid.y;
if (row >= N) return;
uint expert = EXPERTS ? indices[pair] : 0;
uint full_row = expert * N + row;
uint token = BROADCAST ? pair / TOPK : pair;
constexpr uint SLOTS = 1u << CQ_BS;
device const uint2* tables = reinterpret_cast<device const uint2*>(raw);
device const half4* x = reinterpret_cast<device const half4*>(permuted);
float sum = 0;
float ns = float(scales[full_row / SCALE_ROWS]);
for (uint g = lane / 8; g < K / 128; g += 4) {
    uint group = full_row * (K / 128) + g;
    uint block = lane % 8;
    uint xp = (token * (K / 16) + g * 8 + block) * SLOTS * 4;
#if CQ_RATE == 3
    uint2 words = reinterpret_cast<device const uint2*>(codes)[group * 8 + block];
    uint c0 = (words.x & 65535u) | (words.y << 16);
    uint c1 = (words.x >> 16) | (words.y & 0xffff0000u);
    float d = cq_int_dot(c0, tables, x + xp) / 0.93f + 0.28f * cq_int_dot(c1, tables, x + xp);
#else
#if CQ_RATE == 0
    uint cw = reinterpret_cast<device const ushort*>(codes)[group * 8 + block];
#else
    uint cw = reinterpret_cast<device const uint*>(codes)[group * 8 + block];
#endif
    float d = cq_int_dot(cw, tables, x + xp);
#endif
    sum += d * cq_fp8(norms[group]);
}
sum = metal::simd_sum(sum) * (ns * CQ_UNIT) * cq_fp8(c[full_row]);
if (lane == 0) {
    OUT value = static_cast<OUT>(sum);
    if constexpr (RELU2) {
        float activated = max(float(value), 0.0f);
        value = static_cast<OUT>(activated * activated);
    }
    out[pair * N + row] = value;
}
