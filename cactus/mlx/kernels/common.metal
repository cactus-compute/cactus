// CQ dense little-endian code stream. This header contains no MLX-specific types.
inline float cq_fp8(uchar v) {
    uint e = (v >> 3) & 15, m = v & 7;
    float x = e == 0 ? float(m) / 512.0f : as_type<float>(((e + 120) << 23) | (m << 20));
    if (e == 15 && m == 7) x = NAN;
    return (v & 128) ? -x : x;
}

inline uint cq_code(device const uchar* codes, ulong group, uint block, uint stage) {
#if CQ_RATE == 3
    ulong p = group * 64 + block * 8 + stage * 2;
    return uint(codes[p]) | uint(codes[p + 1]) << 8 | uint(codes[p + 4]) << 16 | uint(codes[p + 5]) << 24;
#else
    constexpr uint bytes = CQ_RATE + 2;
    ulong p = (group * 8 + block) * bytes;
    uint c = uint(codes[p]) | uint(codes[p + 1]) << 8;
    if (bytes >= 3) c |= uint(codes[p + 2]) << 16;
    if (bytes >= 4) c |= uint(codes[p + 3]) << 24;
    return c;
#endif
}

inline float cq_value(uint cw, uint pos, device const int* tab, device const float* lv) {
    uint i2 = cw & ((1u << CQ_B2) - 1), i1 = (cw >> CQ_B2) & ((1u << CQ_B1) - 1);
    uint s7 = (cw >> (CQ_B2 + CQ_B1)) & 127;
    uint o = (cw >> (CQ_B2 + CQ_B1 + 7)) & ((1u << CQ_BO) - 1);
    uint s = cw >> (CQ_B2 + CQ_B1 + 7 + CQ_BO);
    uint j = tab[CQ_INV + s * 16 + pos];
    float v;
    if (j < 8) {
        uint neg = j < 7 ? ((s7 >> j) & 1) :
            ((tab[CQ_LO + o] + metal::popcount(s7) + (tab[CQ_L1 + i1] ^ tab[CQ_L2 + i2])) & 1);
        v = lv[tab[CQ_PAT + o * 8 + j]];
        if (neg) v = -v;
    } else if (j < 12) {
        v = lv[4 + tab[CQ_Q1 + i1 * 4 + j - 8]];
    } else {
        v = lv[4 + tab[CQ_Q2 + i2 * 4 + j - 12]];
    }
    return v;
}

inline float cq_decode(device const uchar* codes, ulong group, uint pos,
                       device const int* tab, device const float* lv) {
    float v = cq_value(cq_code(codes, group, pos / 16, 0), pos % 16, tab, lv);
#if CQ_RATE == 3
    float residual = cq_value(cq_code(codes, group, pos / 16, 1), pos % 16, tab, lv);
    v = v / 0.93f + 0.28f * residual;
#endif
    return v;
}

// Four adjacent values per lane; 32 lanes form a complete 128-point group.
inline float4 cq_fwht(float4 v, uint lane) {
    v = float4(v.x + v.y, v.x - v.y, v.z + v.w, v.z - v.w);
    v = float4(v.x + v.z, v.y + v.w, v.x - v.z, v.y - v.w);
    for (uint d = 1; d < 32; d *= 2) {
        float4 peer = metal::simd_shuffle_xor(v, d);
        v = (lane & d) ? peer - v : v + peer;
    }
    return v;
}
