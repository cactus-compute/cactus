inline float cq_int_dot(uint cw, device const uint2* tables, device const half4* x) {
    uint i2 = cw & ((1u << CQ_B2) - 1), i1 = (cw >> CQ_B2) & ((1u << CQ_B1) - 1);
    uint s7 = (cw >> (CQ_B2 + CQ_B1)) & 127;
    uint o = (cw >> (CQ_B2 + CQ_B1 + 7)) & ((1u << CQ_BO) - 1);
    uint slot = cw >> (CQ_B2 + CQ_B1 + 7 + CQ_BO);
    uint2 pat = tables[o];
    uint2 q1 = tables[(1u << CQ_BO) + i1];
    uint2 q2 = tables[(1u << CQ_BO) + (1u << CQ_B1) + i2];
    uint s8 = s7 | (((metal::popcount(s7) ^ q1.y ^ q2.y) & 1) << 7);
    float4 a = float4(as_type<char4>(pat.x)), b = float4(as_type<char4>(pat.y));
    a = as_type<float4>(as_type<uint4>(a) ^ (((uint4(s8) >> uint4(0, 1, 2, 3)) & 1) << 31));
    b = as_type<float4>(as_type<uint4>(b) ^ (((uint4(s8) >> uint4(4, 5, 6, 7)) & 1) << 31));
    float4 c = float4(as_type<char4>(q1.x)), d = float4(as_type<char4>(q2.x));
    x += slot * 4;
    return dot(a, float4(x[0])) + dot(b, float4(x[1])) + dot(c, float4(x[2])) + dot(d, float4(x[3]));
}
