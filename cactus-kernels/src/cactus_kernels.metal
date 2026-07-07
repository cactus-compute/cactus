#include <metal_stdlib>
using namespace metal;
#define ROWS 8u

inline uint cq_idx(device const uchar* packed, uint k, uint bits) {
    if (bits == 4u) return (packed[k >> 1] >> ((k & 1u) * 4u)) & 0xFu;
    if (bits == 2u) return (packed[k >> 2] >> ((k & 3u) * 2u)) & 0x3u;
    uint bit_pos = k * 3u;
    uint byte_idx = bit_pos >> 3, bit_idx = bit_pos & 7u;
    uint val = (uint)packed[byte_idx] >> bit_idx;
    if (bit_idx > 5u) val |= (uint)packed[byte_idx + 1u] << (8u - bit_idx);
    return val & 0x7u;
}

inline float gelu_tanh(float x) {
    float c = 0.7978845608028654f * (x + 0.044715f*x*x*x);
    return 0.5f * x * (1.0f + precise::tanh(c));
}

static inline void cq4_hada128(thread float& x0, thread float& x1,
                               thread float& x2, thread float& x3, uint lane) {
    float a0=x0+x1,a1=x0-x1,a2=x2+x3,a3=x2-x3;
    x0=a0+a2; x1=a1+a3; x2=a0-a2; x3=a1-a3;
    #pragma clang loop unroll(full)
    for (uint d=1u; d<=16u; d<<=1){
        bool hi=(lane&d)!=0u;
        float p0=simd_shuffle_xor(x0,d),p1=simd_shuffle_xor(x1,d),p2=simd_shuffle_xor(x2,d),p3=simd_shuffle_xor(x3,d);
        x0=hi?p0-x0:x0+p0; x1=hi?p1-x1:x1+p1; x2=hi?p2-x2:x2+p2; x3=hi?p3-x3:x3+p3;
    }
    float s=rsqrt(128.0f);
    x0*=s; x1*=s; x2*=s; x3*=s;
}

static inline float cq4_dot64(half4 c0, half4 c1, half4 c2, half4 c3,
                              ushort4 w, threadgroup const float* cb) {
    return (float)c0.x*cb[w[0]&0xF] + (float)c0.y*cb[(w[0]>>4)&0xF] + (float)c0.z*cb[(w[0]>>8)&0xF] + (float)c0.w*cb[(w[0]>>12)&0xF]
         + (float)c1.x*cb[w[1]&0xF] + (float)c1.y*cb[(w[1]>>4)&0xF] + (float)c1.z*cb[(w[1]>>8)&0xF] + (float)c1.w*cb[(w[1]>>12)&0xF]
         + (float)c2.x*cb[w[2]&0xF] + (float)c2.y*cb[(w[2]>>4)&0xF] + (float)c2.z*cb[(w[2]>>8)&0xF] + (float)c2.w*cb[(w[2]>>12)&0xF]
         + (float)c3.x*cb[w[3]&0xF] + (float)c3.y*cb[(w[3]>>4)&0xF] + (float)c3.z*cb[(w[3]>>8)&0xF] + (float)c3.w*cb[(w[3]>>12)&0xF];
}

static inline device const uchar4* cq4_il_base(device const uchar* packed, uint n, uint ng, uint g, uint pgb) {
    return (device const uchar4*)(packed + ((size_t)(n>>2)*ng+g)*4u*(size_t)pgb + (n&3u)*4u);
}

static inline float cq4_dot64_il(half4 c0, half4 c1, half4 c2, half4 c3,
                                 uchar4 A, uchar4 B, threadgroup const float* cb) {
    return (float)c0.x*cb[A.x&0xFu] + (float)c0.y*cb[A.y&0xFu] + (float)c0.z*cb[A.z&0xFu] + (float)c0.w*cb[A.w&0xFu]
         + (float)c1.x*cb[A.x>>4]   + (float)c1.y*cb[A.y>>4]   + (float)c1.z*cb[A.z>>4]   + (float)c1.w*cb[A.w>>4]
         + (float)c2.x*cb[B.x&0xFu] + (float)c2.y*cb[B.y&0xFu] + (float)c2.z*cb[B.z&0xFu] + (float)c2.w*cb[B.w&0xFu]
         + (float)c3.x*cb[B.x>>4]   + (float)c3.y*cb[B.y>>4]   + (float)c3.z*cb[B.z>>4]   + (float)c3.w*cb[B.w>>4];
}

static inline float cq4_norm_at(device const half* norms, uint n, uint ng, uint g, uint il) {
    return (float)(il != 0u ? norms[(((size_t)(n>>2)*ng+g)<<2) + (n&3u)]
                            : norms[(size_t)n*ng+g]);
}

static inline uint cq4_il_nib(device const uchar* packed, uint n, uint ng, uint g, uint pgb, uint e) {
    uint v = e>>4, j = e&15u, sub = j>>2, b = j&3u;
    device const uchar* panel = packed + ((size_t)(n>>2)*ng+g)*4u*(size_t)pgb;
    uchar by = panel[(2u*v + (sub>>1))*16u + (n&3u)*4u + b];
    return (sub&1u) ? (uint)(by>>4) : (uint)(by&0xFu);
}

static inline uint cq_il_idx(device const uchar* packed, uint n, uint ng, uint g, uint pgb, uint e, uint bits) {
    if (bits == 4u) return cq4_il_nib(packed, n, ng, g, pgb, e);
    device const uchar* panel = packed + ((size_t)(n>>2)*ng+g)*4u*(size_t)pgb;
    uint r = n&3u;
    if (bits == 2u) {
        uchar by = panel[(e>>4)*16u + ((e&15u)>>2)*4u + r];
        return ((uint)by >> (2u*(e&3u))) & 3u;
    }
    device const uchar* ch = panel + (e>>2)*6u;
    ulong word = (ulong)ch[0] | ((ulong)ch[1]<<8) | ((ulong)ch[2]<<16)
               | ((ulong)ch[3]<<24) | ((ulong)ch[4]<<32) | ((ulong)ch[5]<<40);
    return (uint)((word >> (r*12u + (e&3u)*3u)) & 7u);
}

static inline float cq4_dot64_ilw(half4 c0, half4 c1, half4 c2, half4 c3,
                                  uint A, uint B, threadgroup const float* cb) {
    return (float)c0.x*cb[A&0xFu]        + (float)c0.y*cb[(A>>8)&0xFu]  + (float)c0.z*cb[(A>>16)&0xFu] + (float)c0.w*cb[(A>>24)&0xFu]
         + (float)c1.x*cb[(A>>4)&0xFu]   + (float)c1.y*cb[(A>>12)&0xFu] + (float)c1.z*cb[(A>>20)&0xFu] + (float)c1.w*cb[A>>28]
         + (float)c2.x*cb[B&0xFu]        + (float)c2.y*cb[(B>>8)&0xFu]  + (float)c2.z*cb[(B>>16)&0xFu] + (float)c2.w*cb[(B>>24)&0xFu]
         + (float)c3.x*cb[(B>>4)&0xFu]   + (float)c3.y*cb[(B>>12)&0xFu] + (float)c3.z*cb[(B>>20)&0xFu] + (float)c3.w*cb[B>>28];
}

kernel void cq4_transform(
    device const half*  x        [[buffer(0)]],
    device const half*  recip    [[buffer(1)]],
    device const char*  lsign    [[buffer(2)]],
    device const char*  rsign    [[buffer(3)]],
    device const uint*  perm     [[buffer(4)]],
    device       half*  code     [[buffer(5)]],
    constant uint& gs            [[buffer(6)]],
    uint g  [[threadgroup_position_in_grid]],
    uint t  [[thread_position_in_threadgroup]],
    uint T  [[threads_per_threadgroup]],
    threadgroup float* z         [[threadgroup(0)]])
{
    for (uint k=t; k<gs; k+=T){ uint gk=g*gs+k; z[k]=(float)x[gk]*(float)recip[gk]*(float)lsign[k]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint h=1; h<gs; h<<=1) {
        for (uint k=t; k<gs; k+=T) if ((k&h)==0){ float a=z[k], b=z[k+h]; z[k]=a+b; z[k+h]=a-b; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint k=t; k<gs; k+=T) z[k] = z[k]*rsqrt((float)gs)*(float)rsign[k];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k=t; k<gs; k+=T) code[g*gs + k] = (half)z[perm[k]];
}

kernel void cq4_transform_simd(
    device const half*  x        [[buffer(0)]],
    device const half*  recip    [[buffer(1)]],
    device const char*  lsign    [[buffer(2)]],
    device const char*  rsign    [[buffer(3)]],
    device const uint*  perm     [[buffer(4)]],
    device       half*  code     [[buffer(5)]],
    constant uint& gs            [[buffer(6)]],
    uint g    [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    threadgroup float* zmem      [[threadgroup(0)]])
{
    uint b = g*128u + lane*4u;
    uint k = lane*4u;
    float x0=(float)x[b+0]*(float)recip[b+0]*(float)lsign[k+0];
    float x1=(float)x[b+1]*(float)recip[b+1]*(float)lsign[k+1];
    float x2=(float)x[b+2]*(float)recip[b+2]*(float)lsign[k+2];
    float x3=(float)x[b+3]*(float)recip[b+3]*(float)lsign[k+3];
    cq4_hada128(x0, x1, x2, x3, lane);
    zmem[k+0]=x0*(float)rsign[k+0]; zmem[k+1]=x1*(float)rsign[k+1];
    zmem[k+2]=x2*(float)rsign[k+2]; zmem[k+3]=x3*(float)rsign[k+3];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    code[b+0]=(half)zmem[perm[k+0]]; code[b+1]=(half)zmem[perm[k+1]];
    code[b+2]=(half)zmem[perm[k+2]]; code[b+3]=(half)zmem[perm[k+3]];
}

kernel void cq4_transform_batch(
    device const half*  x       [[buffer(0)]],
    device const half*  recip   [[buffer(1)]],
    device const char*  lsign0  [[buffer(2)]], device const char* lsign1 [[buffer(3)]], device const char* lsign2 [[buffer(4)]],
    device const char*  rsign0  [[buffer(5)]], device const char* rsign1 [[buffer(6)]], device const char* rsign2 [[buffer(7)]],
    device const uint*  perm0   [[buffer(8)]], device const uint* perm1  [[buffer(9)]], device const uint* perm2  [[buffer(10)]],
    device       half*  code0   [[buffer(11)]], device half* code1 [[buffer(12)]], device half* code2 [[buffer(13)]],
    constant uint& ng           [[buffer(14)]],
    uint tgpos [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    threadgroup float* zmem     [[threadgroup(0)]])
{
    uint g = tgpos % ng, b = tgpos / ng;
    device const char* lsign = (b==0u)?lsign0:((b==1u)?lsign1:lsign2);
    device const char* rsign = (b==0u)?rsign0:((b==1u)?rsign1:rsign2);
    device const uint* perm  = (b==0u)?perm0 :((b==1u)?perm1 :perm2 );
    device       half* code  = (b==0u)?code0 :((b==1u)?code1 :code2 );
    uint base = g*128u + lane*4u;
    uint k = lane*4u;
    float x0=(float)x[base+0]*(float)recip[base+0]*(float)lsign[k+0];
    float x1=(float)x[base+1]*(float)recip[base+1]*(float)lsign[k+1];
    float x2=(float)x[base+2]*(float)recip[base+2]*(float)lsign[k+2];
    float x3=(float)x[base+3]*(float)recip[base+3]*(float)lsign[k+3];
    cq4_hada128(x0, x1, x2, x3, lane);
    zmem[k+0]=x0*(float)rsign[k+0]; zmem[k+1]=x1*(float)rsign[k+1];
    zmem[k+2]=x2*(float)rsign[k+2]; zmem[k+3]=x3*(float)rsign[k+3];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    code[base+0]=(half)zmem[perm[k+0]]; code[base+1]=(half)zmem[perm[k+1]];
    code[base+2]=(half)zmem[perm[k+2]]; code[base+3]=(half)zmem[perm[k+3]];
}

#define CQ4_VPL 16u
kernel void cq4_gemv(
    device const half*  code     [[buffer(0)]],
    device const uchar* packed   [[buffer(1)]],
    device const half*  codebook [[buffer(2)]],
    device const half*  norms    [[buffer(3)]],
    device       half*  y        [[buffer(4)]],
    constant uint& gs            [[buffer(5)]],
    constant uint& num_groups    [[buffer(6)]],
    constant uint& pgb           [[buffer(7)]],
    constant uint& N             [[buffer(8)]],
    constant uint& il            [[buffer(9)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tl   [[thread_index_in_threadgroup]])
{
    threadgroup float cb[16];
    if (tl<16) cb[tl]=(float)codebook[tl];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint n = tg*ROWS + sgid;
    if (n>=N) return;
    uint K = num_groups*gs;
    float acc=0;
    for (uint base=lane*CQ4_VPL; base<K; base+=32u*CQ4_VPL){
        uint g = base/gs;
        uint off = base - g*gs;
        device const half4* cbase=(device const half4*)(code + g*gs + off);
        if (il != 0u) {
            device const uchar4* pp = cq4_il_base(packed, n, num_groups, g, pgb);
            uchar4 A = pp[(off>>4)*8u], B = pp[(off>>4)*8u+4u];
            float p = cq4_dot64_il(cbase[0], cbase[1], cbase[2], cbase[3], A, B, cb);
            acc += (float)norms[(((size_t)(n>>2)*num_groups+g)<<2) + (n&3u)]*p;
        } else {
            device const ushort4* pr=(device const ushort4*)(packed + ((size_t)n*num_groups+g)*pgb + off/2u);
            ushort4 w=pr[0];
            float p = 0;
            #pragma clang loop unroll(full)
            for (uint q=0;q<4;++q){
                half4 c=cbase[q];
                ushort ww = w[q];
                p += (float)c.x*cb[ww&0xF] + (float)c.y*cb[(ww>>4)&0xF]
                   + (float)c.z*cb[(ww>>8)&0xF] + (float)c.w*cb[(ww>>12)&0xF];
            }
            acc += (float)norms[(size_t)n*num_groups+g]*p;
        }
    }
    acc=simd_sum(acc);
    if (lane==0) y[n]=(half)acc;
}

#define CQ4_NR 2u
kernel void cq4_gemv_mr(
    device const half*  code     [[buffer(0)]],
    device const uchar* packed   [[buffer(1)]],
    device const half*  codebook [[buffer(2)]],
    device const half*  norms    [[buffer(3)]],
    device       half*  y        [[buffer(4)]],
    constant uint& gs            [[buffer(5)]],
    constant uint& num_groups    [[buffer(6)]],
    constant uint& pgb           [[buffer(7)]],
    constant uint& N             [[buffer(8)]],
    constant uint& il            [[buffer(9)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tl   [[thread_index_in_threadgroup]])
{
    threadgroup float cb[16];
    if (tl<16) cb[tl]=(float)codebook[tl];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint n0 = (tg*ROWS + sgid)*CQ4_NR;
    if (n0>=N) return;
    uint K = num_groups*gs;
    float acc[CQ4_NR];
    #pragma clang loop unroll(full)
    for (uint r=0;r<CQ4_NR;r++) acc[r]=0;
    for (uint base=lane*CQ4_VPL; base<K; base+=32u*CQ4_VPL){
        uint g = base/gs, off = base - g*gs;
        device const half4* cbase = (device const half4*)(code + g*gs + off);
        half4 c0=cbase[0], c1=cbase[1], c2=cbase[2], c3=cbase[3];
        if (il != 0u) {
            device const uchar* pan = packed + ((size_t)(n0>>2)*num_groups+g)*4u*(size_t)pgb
                                    + (n0&3u)*4u + (off>>4)*32u;
            uint2 Aw = ((device const uint2*)pan)[0];
            uint2 Bw = ((device const uint2*)(pan+16u))[0];
            #pragma clang loop unroll(full)
            for (uint r=0;r<CQ4_NR;r++){
                float p = cq4_dot64_ilw(c0, c1, c2, c3, r?Aw.y:Aw.x, r?Bw.y:Bw.x, cb);
                acc[r] += (float)norms[(((size_t)(n0>>2)*num_groups+g)<<2) + (n0&3u) + r]*p;
            }
        } else {
            #pragma clang loop unroll(full)
            for (uint r=0;r<CQ4_NR;r++){
                uint n = n0+r;
                device const ushort4* pr = (device const ushort4*)(packed + ((size_t)n*num_groups+g)*pgb + off/2u);
                ushort4 w = pr[0];
                float p = cq4_dot64(c0, c1, c2, c3, w, cb);
                acc[r] += (float)norms[(size_t)n*num_groups+g]*p;
            }
        }
    }
    #pragma clang loop unroll(full)
    for (uint r=0;r<CQ4_NR;r++){
        float a = simd_sum(acc[r]);
        if (lane==0) y[n0+r]=(half)a;
    }
}

struct TGU { uint K, ng, oswi; };
kernel void cq4_transform_gemv(
    device const half*  x        [[buffer(0)]],
    device const half*  recip    [[buffer(1)]],
    device const char*  lsign    [[buffer(2)]],
    device const char*  rsign    [[buffer(3)]],
    device const uint*  perm     [[buffer(4)]],
    device const half*  cbook    [[buffer(5)]],
    device const half*  norms    [[buffer(6)]],
    device const uchar* packed   [[buffer(7)]],
    device       half*  y        [[buffer(8)]],
    device const half*  osw      [[buffer(9)]],
    constant TGU& U              [[buffer(10)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint t    [[thread_position_in_threadgroup]],
    threadgroup half*  coded     [[threadgroup(0)]],
    threadgroup float* zwave     [[threadgroup(1)]])
{
    const uint K = U.K, ng = U.ng;
    threadgroup float cb[16];
    if (t < 16u) cb[t] = (float)cbook[t];

    uint k = lane*4u;
    for (uint g0 = 0u; g0 < ng; g0 += 8u) {
        uint g = g0 + sg;
        if (g < ng) {
            uint base = g*128u + k;
            float x0=(float)x[base+0]*(float)recip[base+0]*(float)lsign[k+0];
            float x1=(float)x[base+1]*(float)recip[base+1]*(float)lsign[k+1];
            float x2=(float)x[base+2]*(float)recip[base+2]*(float)lsign[k+2];
            float x3=(float)x[base+3]*(float)recip[base+3]*(float)lsign[k+3];
            cq4_hada128(x0, x1, x2, x3, lane);
            zwave[sg*128u+k+0]=x0*(float)rsign[k+0]; zwave[sg*128u+k+1]=x1*(float)rsign[k+1];
            zwave[sg*128u+k+2]=x2*(float)rsign[k+2]; zwave[sg*128u+k+3]=x3*(float)rsign[k+3];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (g < ng) {
            uint base = g*128u + k;
            coded[base+0]=(half)zwave[sg*128u+perm[k+0]]; coded[base+1]=(half)zwave[sg*128u+perm[k+1]];
            coded[base+2]=(half)zwave[sg*128u+perm[k+2]]; coded[base+3]=(half)zwave[sg*128u+perm[k+3]];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint n0 = tg*16u + sg*2u;
    float acc[2]; acc[0]=0; acc[1]=0;
    for (uint base=lane*16u; base<K; base+=512u){
        uint g = base/128u, off = base - g*128u;
        threadgroup const half4* cbase = (threadgroup const half4*)(coded + g*128u + off);
        half4 c0=cbase[0], c1=cbase[1], c2=cbase[2], c3=cbase[3];
        device const uchar* pan = packed + ((size_t)(n0>>2)*ng+g)*256u + (n0&3u)*4u + (off>>4)*32u;
        uint2 Aw = ((device const uint2*)pan)[0];
        uint2 Bw = ((device const uint2*)(pan+16u))[0];
        #pragma clang loop unroll(full)
        for (uint r=0u;r<2u;r++){
            float p = cq4_dot64_ilw(c0, c1, c2, c3, r?Aw.y:Aw.x, r?Bw.y:Bw.x, cb);
            acc[r] += (float)norms[(((size_t)(n0>>2)*ng+g)<<2) + (n0&3u) + r]*p;
        }
    }
    #pragma clang loop unroll(full)
    for (uint r=0u;r<2u;r++){
        float a = simd_sum(acc[r]);
        if (lane==0u){
            uint n = n0+r;
            if (U.oswi != 0u) {
                float gg = gelu_tanh((float)(half)a) * (float)osw[n];
                y[n] = (half)clamp(gg,-65504.0f,65504.0f);
            } else {
                y[n] = (half)a;
            }
        }
    }
}

kernel void cq4_swiglu_transform(
    device const half*  gate     [[buffer(0)]],
    device const half*  up       [[buffer(1)]],
    device const half*  recip    [[buffer(2)]],
    device const char*  lsign    [[buffer(3)]],
    device const char*  rsign    [[buffer(4)]],
    device const uint*  perm     [[buffer(5)]],
    device       half*  code     [[buffer(6)]],
    constant float& scale        [[buffer(7)]],
    uint g    [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    threadgroup float* zmem      [[threadgroup(0)]])
{
    uint b = g*128u + lane*4u;
    uint k = lane*4u;
    float x0,x1,x2,x3;
    {
        float g0 = gelu_tanh((float)gate[b+0]) * scale * (float)up[b+0];
        float g1 = gelu_tanh((float)gate[b+1]) * scale * (float)up[b+1];
        float g2 = gelu_tanh((float)gate[b+2]) * scale * (float)up[b+2];
        float g3 = gelu_tanh((float)gate[b+3]) * scale * (float)up[b+3];
        x0=(float)(half)clamp(g0,-65504.0f,65504.0f)*(float)recip[b+0]*(float)lsign[k+0];
        x1=(float)(half)clamp(g1,-65504.0f,65504.0f)*(float)recip[b+1]*(float)lsign[k+1];
        x2=(float)(half)clamp(g2,-65504.0f,65504.0f)*(float)recip[b+2]*(float)lsign[k+2];
        x3=(float)(half)clamp(g3,-65504.0f,65504.0f)*(float)recip[b+3]*(float)lsign[k+3];
    }
    cq4_hada128(x0, x1, x2, x3, lane);
    zmem[k+0]=x0*(float)rsign[k+0]; zmem[k+1]=x1*(float)rsign[k+1];
    zmem[k+2]=x2*(float)rsign[k+2]; zmem[k+3]=x3*(float)rsign[k+3];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    code[b+0]=(half)zmem[perm[k+0]]; code[b+1]=(half)zmem[perm[k+1]];
    code[b+2]=(half)zmem[perm[k+2]]; code[b+3]=(half)zmem[perm[k+3]];
}

struct CatU { uint ng, N0, N1, N2; };
kernel void cq4_gemv_cat(
    device const half*  code0  [[buffer(0)]], device const half* code1  [[buffer(1)]], device const half* code2  [[buffer(2)]],
    device const uchar* packed0[[buffer(3)]], device const uchar* packed1[[buffer(4)]], device const uchar* packed2[[buffer(5)]],
    device const half*  cbook0 [[buffer(6)]], device const half* cbook1 [[buffer(7)]], device const half* cbook2 [[buffer(8)]],
    device const half*  norms0 [[buffer(9)]], device const half* norms1 [[buffer(10)]], device const half* norms2 [[buffer(11)]],
    device       half*  y0     [[buffer(12)]], device half* y1 [[buffer(13)]], device half* y2 [[buffer(14)]],
    constant CatU& U            [[buffer(15)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tl   [[thread_index_in_threadgroup]])
{
    uint n0 = tg*16u;
    uint b, nbase;
    if (n0 < U.N0)              { b = 0u; nbase = 0u; }
    else if (n0 < U.N0 + U.N1)  { b = 1u; nbase = U.N0; }
    else                        { b = 2u; nbase = U.N0 + U.N1; }
    device const half*  code  = (b==0u)?code0 :((b==1u)?code1 :code2 );
    device const uchar* packed= (b==0u)?packed0:((b==1u)?packed1:packed2);
    device const half*  cbook = (b==0u)?cbook0:((b==1u)?cbook1:cbook2);
    device const half*  norms = (b==0u)?norms0:((b==1u)?norms1:norms2);
    device       half*  y     = (b==0u)?y0    :((b==1u)?y1    :y2    );
    threadgroup float cb[16];
    if (tl<16) cb[tl]=(float)cbook[tl];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint ng = U.ng;
    uint K = ng*128u;
    uint nl0 = (n0 - nbase) + sgid*2u;
    float acc[2]; acc[0]=0; acc[1]=0;
    for (uint base=lane*16u; base<K; base+=512u){
        uint g = base/128u, off = base - g*128u;
        device const half4* cbase = (device const half4*)(code + g*128u + off);
        half4 c0=cbase[0], c1=cbase[1], c2=cbase[2], c3=cbase[3];
        device const uchar* pan = packed + ((size_t)(nl0>>2)*ng+g)*256u + (nl0&3u)*4u + (off>>4)*32u;
        uint2 Aw = ((device const uint2*)pan)[0];
        uint2 Bw = ((device const uint2*)(pan+16u))[0];
        #pragma clang loop unroll(full)
        for (uint r=0;r<2;r++){
            float p = cq4_dot64_ilw(c0, c1, c2, c3, r?Aw.y:Aw.x, r?Bw.y:Bw.x, cb);
            acc[r] += (float)norms[(((size_t)(nl0>>2)*ng+g)<<2) + (nl0&3u) + r]*p;
        }
    }
    #pragma clang loop unroll(full)
    for (uint r=0;r<2;r++){
        float a = simd_sum(acc[r]);
        if (lane==0) y[nl0+r]=(half)a;
    }
}

kernel void cq_gemv_mr_lowbit(
    device const half*  code     [[buffer(0)]],
    device const uchar* packed   [[buffer(1)]],
    device const half*  codebook [[buffer(2)]],
    device const half*  norms    [[buffer(3)]],
    device       half*  y        [[buffer(4)]],
    constant uint& gs            [[buffer(5)]],
    constant uint& num_groups    [[buffer(6)]],
    constant uint& pgb           [[buffer(7)]],
    constant uint& N             [[buffer(8)]],
    constant uint& bits          [[buffer(9)]],
    constant uint& il            [[buffer(10)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tl   [[thread_index_in_threadgroup]])
{
    threadgroup float cb[8];
    uint cbn = 1u << bits;
    if (tl < cbn) cb[tl] = (float)codebook[tl];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint n0 = (tg*ROWS + sgid)*CQ4_NR;
    if (n0 >= N) return;
    uint K = num_groups*gs;
    float acc[CQ4_NR];
    #pragma clang loop unroll(full)
    for (uint r=0;r<CQ4_NR;r++) acc[r]=0;
    for (uint base=lane*16u; base<K; base+=512u){
        uint g = base/gs, off = base - g*gs;
        device const half4* cbase = (device const half4*)(code + g*gs + off);
        half4 c0=cbase[0], c1=cbase[1], c2=cbase[2], c3=cbase[3];
        uint byte_off = (off*bits) >> 3;
        #pragma clang loop unroll(full)
        for (uint r=0;r<CQ4_NR;r++){
            uint n = n0+r;
            float w0,w1,w2,w3,w4,w5,w6,w7,w8,w9,wa,wb,wc,wd,we,wf;
            if (bits == 2u) {
                uint w;
                if (il != 0u) {
                    device const uchar* pan = packed + ((size_t)(n>>2)*num_groups+g)*4u*(size_t)pgb + byte_off*4u;
                    uint rr = n&3u;
                    w = (uint)pan[rr] | ((uint)pan[4u+rr]<<8) | ((uint)pan[8u+rr]<<16) | ((uint)pan[12u+rr]<<24);
                } else {
                    w = *(device const uint*)(packed + ((size_t)n*num_groups+g)*pgb + byte_off);
                }
                w0=cb[w&3u];        w1=cb[(w>>2)&3u];  w2=cb[(w>>4)&3u];  w3=cb[(w>>6)&3u];
                w4=cb[(w>>8)&3u];   w5=cb[(w>>10)&3u]; w6=cb[(w>>12)&3u]; w7=cb[(w>>14)&3u];
                w8=cb[(w>>16)&3u];  w9=cb[(w>>18)&3u]; wa=cb[(w>>20)&3u]; wb=cb[(w>>22)&3u];
                wc=cb[(w>>24)&3u];  wd=cb[(w>>26)&3u]; we=cb[(w>>28)&3u]; wf=cb[(w>>30)&3u];
            } else {
                ulong v;
                if (il != 0u) {
                    device const uchar* pan = packed + ((size_t)(n>>2)*num_groups+g)*4u*(size_t)pgb + byte_off*4u;
                    uint rr = n&3u;
                    v = 0;
                    #pragma clang loop unroll(full)
                    for (uint q=0;q<4u;q++){
                        device const uchar* ch = pan + q*6u;
                        ulong word = (ulong)ch[0] | ((ulong)ch[1]<<8) | ((ulong)ch[2]<<16)
                                   | ((ulong)ch[3]<<24) | ((ulong)ch[4]<<32) | ((ulong)ch[5]<<40);
                        v |= ((word >> (rr*12u)) & 0xFFFul) << (12u*q);
                    }
                } else {
                    device const ushort* ps = (device const ushort*)(packed + ((size_t)n*num_groups+g)*pgb + byte_off);
                    v = (ulong)ps[0] | ((ulong)ps[1] << 16) | ((ulong)ps[2] << 32);
                }
                w0=cb[v&7u];        w1=cb[(v>>3)&7u];  w2=cb[(v>>6)&7u];  w3=cb[(v>>9)&7u];
                w4=cb[(v>>12)&7u];  w5=cb[(v>>15)&7u]; w6=cb[(v>>18)&7u]; w7=cb[(v>>21)&7u];
                w8=cb[(v>>24)&7u];  w9=cb[(v>>27)&7u]; wa=cb[(v>>30)&7u]; wb=cb[(v>>33)&7u];
                wc=cb[(v>>36)&7u];  wd=cb[(v>>39)&7u]; we=cb[(v>>42)&7u]; wf=cb[(v>>45)&7u];
            }
            float pacc = (float)c0.x*w0 + (float)c0.y*w1 + (float)c0.z*w2 + (float)c0.w*w3
                       + (float)c1.x*w4 + (float)c1.y*w5 + (float)c1.z*w6 + (float)c1.w*w7
                       + (float)c2.x*w8 + (float)c2.y*w9 + (float)c2.z*wa + (float)c2.w*wb
                       + (float)c3.x*wc + (float)c3.y*wd + (float)c3.z*we + (float)c3.w*wf;
            acc[r] += cq4_norm_at(norms, n, num_groups, g, il)*pacc;
        }
    }
    #pragma clang loop unroll(full)
    for (uint r=0;r<CQ4_NR;r++){
        float a = simd_sum(acc[r]);
        if (lane==0) y[n0+r]=(half)a;
    }
}

kernel void cq_act_quant_i8(
    device const half*  code       [[buffer(0)]],
    device       char*  act_i8     [[buffer(1)]],
    device       float* act_scales [[buffer(2)]],
    constant uint& gs              [[buffer(3)]],
    uint g    [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    device const half* z = code + (size_t)g*gs;
    float m = 0.f;
    for (uint k=lane; k<gs; k+=32u) m = max(m, fabs((float)z[k]));
    m = simd_max(m);
    float scale = m / 127.f;
    if (scale < 1e-10f) scale = 1e-10f;
    float inv = 1.f / scale;
    device char* a = act_i8 + (size_t)g*gs;
    for (uint k=lane; k<gs; k+=32u)
        a[k] = (char)clamp(rint((float)z[k] * inv), -128.f, 127.f);
    if (lane == 0u) act_scales[g] = scale;
}

kernel void cq2_gemv_i8(
    device const char*  act_i8     [[buffer(0)]],
    device const float* act_scales [[buffer(1)]],
    device const uchar* packed     [[buffer(2)]],
    device const char*  cb_i8     [[buffer(3)]],
    device const half*  norms      [[buffer(4)]],
    device       half*  y          [[buffer(5)]],
    constant uint& num_groups      [[buffer(6)]],
    constant uint& N               [[buffer(7)]],
    constant float& cb_scale       [[buffer(8)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tl   [[thread_index_in_threadgroup]])
{
    threadgroup int cb[4];
    if (tl < 4u) cb[tl] = (int)cb_i8[tl];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint n = tg*8u + sgid;
    if (n >= N) return;
    float acc = 0.f;
    for (uint g = 0u; g < num_groups; ++g) {
        uchar b = packed[((size_t)(n>>2)*num_groups + g)*128u + (lane>>2)*16u + (lane&3u)*4u + (n&3u)];
        device const char* a = act_i8 + (size_t)g*128u + lane*4u;
        int partial = (int)a[0]*cb[b & 3u] + (int)a[1]*cb[(b >> 2) & 3u]
                    + (int)a[2]*cb[(b >> 4) & 3u] + (int)a[3]*cb[(b >> 6) & 3u];
        int idot = simd_sum(partial);
        if (lane == 0u) {
            float ns = (float)norms[(((size_t)(n>>2)*num_groups+g)<<2) + (n&3u)] * cb_scale;
            float sc = ns * act_scales[g];
            acc = fma((float)idot, sc, acc);
        }
    }
    if (lane == 0u) y[n] = (half)acc;
}

kernel void cq4_transform_m(
    device const half*  x        [[buffer(0)]],
    device const half*  recip    [[buffer(1)]],
    device const char*  lsign    [[buffer(2)]],
    device const char*  rsign    [[buffer(3)]],
    device const uint*  perm     [[buffer(4)]],
    device       half*  code     [[buffer(5)]],
    constant uint& gs            [[buffer(6)]],
    constant uint& K             [[buffer(7)]],
    uint pos [[threadgroup_position_in_grid]],
    uint t   [[thread_position_in_threadgroup]],
    uint T   [[threads_per_threadgroup]],
    threadgroup float* z         [[threadgroup(0)]])
{
    uint ng = K/gs;
    uint g = pos % ng, row = pos / ng;
    size_t xb = (size_t)row*K + (size_t)g*gs;
    for (uint k=t; k<gs; k+=T) z[k] = (float)x[xb+k]*(float)recip[g*gs+k]*(float)lsign[k];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint h=1; h<gs; h<<=1) {
        for (uint k=t; k<gs; k+=T) if ((k&h)==0){ float a=z[k], b=z[k+h]; z[k]=a+b; z[k+h]=a-b; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint k=t; k<gs; k+=T) z[k] = z[k]*rsqrt((float)gs)*(float)rsign[k];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k=t; k<gs; k+=T) code[xb + k] = (half)z[perm[k]];
}

kernel void cq4_gemm_mma(
    device const half*  code     [[buffer(0)]],
    device const uchar* packed   [[buffer(1)]],
    device const half*  codebook [[buffer(2)]],
    device const half*  norms    [[buffer(3)]],
    device       half*  y        [[buffer(4)]],
    constant uint& gs            [[buffer(5)]],
    constant uint& num_groups    [[buffer(6)]],
    constant uint& pgb           [[buffer(7)]],
    constant uint& N             [[buffer(8)]],
    constant uint& M             [[buffer(9)]],
    uint2 tg  [[threadgroup_position_in_grid]],
    uint  tl  [[thread_index_in_threadgroup]],
    uint  sg  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float pool[2048];
    threadgroup float cb[16];
    threadgroup half* sa = (threadgroup half*)pool;
    threadgroup half* sb = sa + 2048u;
    threadgroup float* Cs = pool;
    if (tl<16u) cb[tl]=(float)codebook[tl];
    const uint NK=32u, NL0=2u, NL1=4u;
    uint K = num_groups*gs;
    uint r0 = tg.y*64u;
    uint r1 = tg.x*32u;
    uint lr0 = tl/NL0, lr1 = tl/NL1, il0 = tl%NL0;
    uint nrow = r0+lr0;
    uint iy = 8u*(tl%NL1);
    device const half* yrow = code + (size_t)(r1+lr1)*K;
    simdgroup_matrix<half,8,8> ma[4], mb[2];
    simdgroup_matrix<float,8,8> mc[8];
    for (uint i=0;i<8u;i++) mc[i]=make_filled_simdgroup_matrix<float,8,8>(0.f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint loop_k=0; loop_k<K; loop_k+=NK){
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint g=loop_k/gs, e0=loop_k-g*gs;
        float nrm=(nrow<N)?(float)norms[(((size_t)(nrow>>2)*num_groups+g)<<2) + (nrow&3u)]:0.f;
        float wv[16];
        if (nrow<N) {
            device const uchar4* pp = cq4_il_base(packed, nrow, num_groups, g, pgb);
            uint v4 = (e0+16u*il0)>>4;
            uchar4 A = pp[v4*8u], B = pp[v4*8u+4u];
            wv[0]=cb[A.x&0xFu]; wv[1]=cb[A.y&0xFu]; wv[2]=cb[A.z&0xFu]; wv[3]=cb[A.w&0xFu];
            wv[4]=cb[A.x>>4];   wv[5]=cb[A.y>>4];   wv[6]=cb[A.z>>4];   wv[7]=cb[A.w>>4];
            wv[8]=cb[B.x&0xFu]; wv[9]=cb[B.y&0xFu]; wv[10]=cb[B.z&0xFu]; wv[11]=cb[B.w&0xFu];
            wv[12]=cb[B.x>>4];  wv[13]=cb[B.y>>4];  wv[14]=cb[B.z>>4];   wv[15]=cb[B.w>>4];
        }
        for (uint i=0;i<16u;i++){
            uint sx=2u*il0+i/8u, sy=(tl/NL0)/8u, lx=(tl/NL0)%8u, ly=i%8u, ib=8u*sx+sy;
            float v=0.f;
            if (loop_k+16u*il0+i<K && nrow<N) v=nrm*wv[i];
            sa[64u*ib+8u*ly+lx]=(half)v;
        }
        for (uint i=0;i<8u;i++){
            uint sx=tl%NL1, sy=(tl/NL1)/8u, lx=i, ly=(tl/NL1)%8u, ib=4u*sx+sy;
            sb[64u*ib+8u*ly+lx]=(loop_k+iy+i<K && r1+lr1<M)?yrow[loop_k+iy+i]:(half)0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const half* lsma = sa + 4u*64u*(sg%2u);
        threadgroup const half* lsmb = sb + 2u*64u*(sg/2u);
        for (uint ik=0; ik<NK/8u; ik++){
            for (uint i=0;i<4u;i++) simdgroup_load(ma[i], lsma+64u*i, 8u);
            for (uint i=0;i<2u;i++) simdgroup_load(mb[i], lsmb+64u*i, 8u);
            for (uint i=0;i<8u;i++) simdgroup_multiply_accumulate(mc[i], mb[i/4u], ma[i%4u], mc[i]);
            lsma += 8u*64u; lsmb += 4u*64u;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint m_sg = 16u*(sg>>1u), n_sg = 32u*(sg&1u);
    for (uint i=0;i<8u;i++)
        simdgroup_store(mc[i], &Cs[(m_sg+8u*(i/4u))*64u + n_sg+8u*(i%4u)], 64u);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx=tl; idx<32u*64u; idx+=128u){
        uint m=idx/64u, n=idx%64u;
        if (r1+m<M && r0+n<N) y[(size_t)(r1+m)*N + r0+n] = (half)Cs[m*64u+n];
    }
}

kernel void cq4_gemm_dense_f16(
    device const half*  code [[buffer(0)]],
    device const half*  wf16 [[buffer(1)]],
    device       half*  y    [[buffer(2)]],
    constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]], constant uint& M [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]], uint tl [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float pool[2048];
    threadgroup half* sa = (threadgroup half*)pool;
    threadgroup half* sb = sa + 2048u;
    threadgroup float* Cs = pool;
    const uint NK=32u, NL0=2u, NL1=4u;
    uint r0 = tg.y*64u, r1 = tg.x*32u;
    uint lr0 = tl/NL0, lr1 = tl/NL1, il0 = tl%NL0;
    uint nrow = r0+lr0;
    uint iy = 8u*(tl%NL1);
    device const half* yrow = code + (size_t)(r1+lr1)*K;
    device const half* wrow = wf16 + (size_t)nrow*K;
    simdgroup_matrix<half,8,8> ma[4], mb[2];
    simdgroup_matrix<float,8,8> mc[8];
    for (uint i=0;i<8u;i++) mc[i]=make_filled_simdgroup_matrix<float,8,8>(0.f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint loop_k=0; loop_k<K; loop_k+=NK){
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i=0;i<16u;i++){
            uint sx=2u*il0+i/8u, sy=(tl/NL0)/8u, lx=(tl/NL0)%8u, ly=i%8u, ib=8u*sx+sy;
            uint k=loop_k+16u*il0+i;
            half v=(k<K && nrow<N)? wrow[k] : (half)0;
            sa[64u*ib+8u*ly+lx]=v;
        }
        for (uint i=0;i<8u;i++){
            uint sx=tl%NL1, sy=(tl/NL1)/8u, lx=i, ly=(tl/NL1)%8u, ib=4u*sx+sy;
            sb[64u*ib+8u*ly+lx]=(loop_k+iy+i<K && r1+lr1<M)?yrow[loop_k+iy+i]:(half)0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const half* lsma = sa + 4u*64u*(sg%2u);
        threadgroup const half* lsmb = sb + 2u*64u*(sg/2u);
        for (uint ik=0; ik<NK/8u; ik++){
            for (uint i=0;i<4u;i++) simdgroup_load(ma[i], lsma+64u*i, 8u);
            for (uint i=0;i<2u;i++) simdgroup_load(mb[i], lsmb+64u*i, 8u);
            for (uint i=0;i<8u;i++) simdgroup_multiply_accumulate(mc[i], mb[i/4u], ma[i%4u], mc[i]);
            lsma += 8u*64u; lsmb += 4u*64u;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint m_sg = 16u*(sg>>1u), n_sg = 32u*(sg&1u);
    for (uint i=0;i<8u;i++)
        simdgroup_store(mc[i], &Cs[(m_sg+8u*(i/4u))*64u + n_sg+8u*(i%4u)], 64u);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx=tl; idx<32u*64u; idx+=128u){
        uint m=idx/64u, n=idx%64u;
        if (r1+m<M && r0+n<N) y[(size_t)(r1+m)*N + r0+n] = (half)Cs[m*64u+n];
    }
}

kernel void lmhead_rotate_wide(device const half* act [[buffer(0)]], device const half* recip [[buffer(1)]],
                          device const half* rotation [[buffer(2)]], device half* code [[buffer(3)]],
                          constant uint& K [[buffer(4)]],
                          uint tg [[threadgroup_position_in_grid]],
                          uint t [[thread_position_in_threadgroup]],
                          threadgroup float* red [[threadgroup(0)]]) {
    uint c = t & 31u, r = t >> 5u;
    uint j = tg*32u + c;
    float acc = 0;
    if (j < K) for (uint k=r; k<K; k+=8u) acc += (float)act[k]*(float)recip[k]*(float)rotation[(size_t)k*K + j];
    red[t] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (r == 0u && j < K) {
        float a = 0;
        for (uint rr=0; rr<8u; ++rr) a += red[rr*32u + c];
        code[j] = (half)a;
    }
}

kernel void emb_ortho_wide(device const uchar* packed [[buffer(0)]],
                      device const half* codebook [[buffer(1)]],
                      device const half* norms [[buffer(2)]],
                      device const half* recip [[buffer(3)]],
                      device const half* rotation [[buffer(4)]],
                      device half* out [[buffer(5)]],
                      constant uint& K [[buffer(6)]], constant uint& row [[buffer(7)]],
                      constant float& oscale [[buffer(8)]],
                      constant uint& bits [[buffer(9)]],
                      constant uint& il [[buffer(10)]],
                      uint tg [[threadgroup_position_in_grid]],
                      uint t [[thread_position_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]],
                      threadgroup float* dq [[threadgroup(0)]]) {
    uint pgb = (K*bits + 7u)/8u;
    device const uchar* prow = packed + (size_t)row*pgb;
    if (il != 0u) { for (uint i=t; i<K; i+=256u) dq[i]=(float)codebook[cq_il_idx(packed, row, 1u, 0u, pgb, i, bits)]; }
    else { for (uint i=t; i<K; i+=256u) dq[i]=(float)codebook[cq_idx(prow, i, bits)]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float nrm = (float)norms[row];
    uint j = tg*8u + sg;
    if (j >= K) return;
    device const half* rrow = rotation + (size_t)j*K;
    float acc=0;
    for (uint i=lane*4u; i<K; i+=128u) {
        half4 r = *(device const half4*)(rrow + i);
        acc += dq[i]*(float)r.x + dq[i+1]*(float)r.y + dq[i+2]*(float)r.z + dq[i+3]*(float)r.w;
    }
    acc = simd_sum(acc);
    if (lane == 0u) out[j] = (half)((float)(half)(acc * nrm * (float)recip[j]) * oscale);
}

kernel void emb_hadamard(device const uchar* packed_row [[buffer(0)]],
                         device const half* codebook [[buffer(1)]],
                         device const half* norms_row [[buffer(2)]],
                         device const half* recip [[buffer(3)]],
                         device const char* lsign [[buffer(4)]],
                         device const char* rsign [[buffer(5)]],
                         device const uint* perm [[buffer(6)]],
                         device half* out [[buffer(7)]], constant uint& gs [[buffer(8)]],
                         constant uint& bits [[buffer(9)]],
                         uint g [[threadgroup_position_in_grid]],
                         uint t [[thread_position_in_threadgroup]], uint T [[threads_per_threadgroup]],
                         threadgroup float* z [[threadgroup(0)]]) {
    uint pgb = (gs*bits+7u)/8u;
    device const uchar* pg = packed_row + (size_t)g*pgb;
    for (uint k=t; k<gs; k+=T) {
        uint dst=perm[k]; z[dst]=(float)codebook[cq_idx(pg, k, bits)]*(float)rsign[dst];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint h=1; h<gs; h<<=1) {
        for (uint k=t; k<gs; k+=T) if ((k&h)==0){ float a=z[k],b=z[k+h]; z[k]=a+b; z[k+h]=a-b; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float nrm=(float)norms_row[g], inv=rsqrt((float)gs);
    for (uint k=t; k<gs; k+=T) { uint col=g*gs+k; out[col]=(half)(z[k]*inv*(float)lsign[k]*nrm*(float)recip[col]); }
}

constant uint EM_TM = 16, EM_TN = 16, EM_TK = 32;
kernel void emb_ortho_m(device const uchar* packed [[buffer(0)]], device const half* codebook [[buffer(1)]],
                        device const half* norms [[buffer(2)]], device const half* recip [[buffer(3)]],
                        device const half* rotation [[buffer(4)]], device const uint* rows [[buffer(5)]],
                        device half* out [[buffer(6)]], constant uint& K [[buffer(7)]], constant uint& M [[buffer(8)]],
                        constant uint& bits [[buffer(9)]],
                        constant uint& il [[buffer(10)]],
                        uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    threadgroup half As[EM_TM*EM_TK];
    threadgroup half Bs[EM_TN*EM_TK];
    uint nJ = (K+EM_TN-1)/EM_TN, jt = tg%nJ, mt = tg/nJ;
    uint m0 = mt*EM_TM, j0 = jt*EM_TN, tm = tid/EM_TN, tn = tid%EM_TN, pgb = (K*bits+7u)/8u;
    float acc = 0;
    for (uint k0=0; k0<K; k0+=EM_TK) {
        for (uint e=tid; e<EM_TM*EM_TK; e+=EM_TM*EM_TN) {
            uint r=e/EM_TK, c=e%EM_TK, mm=m0+r, ki=k0+c; half v=0;
            if (mm<M && ki<K) {
                uint row=rows[mm];
                uint idx = (il!=0u) ? cq_il_idx(packed, row, 1u, 0u, pgb, ki, bits)
                                    : cq_idx(packed + (size_t)row*pgb, ki, bits);
                v=codebook[idx];
            }
            As[e]=v;
        }
        for (uint e=tid; e<EM_TN*EM_TK; e+=EM_TM*EM_TN) {
            uint r=e/EM_TK, c=e%EM_TK, jj=j0+r, ki=k0+c; half v=0;
            if (jj<K && ki<K) v=rotation[(size_t)jj*K + ki];
            Bs[e]=v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint c=0;c<EM_TK;++c) acc += (float)As[tm*EM_TK+c]*(float)Bs[tn*EM_TK+c];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint mm=m0+tm, jj=j0+tn;
    if (mm<M && jj<K) out[(size_t)mm*K+jj] = (half)(acc*(float)norms[rows[mm]]*(float)recip[jj]);
}

kernel void emb_hadamard_m(device const uchar* packed [[buffer(0)]], device const half* codebook [[buffer(1)]],
                           device const half* norms [[buffer(2)]], device const half* recip [[buffer(3)]],
                           device const char* lsign [[buffer(4)]], device const char* rsign [[buffer(5)]],
                           device const uint* perm [[buffer(6)]], device half* out [[buffer(7)]],
                           constant uint& gs [[buffer(8)]], constant uint& ng [[buffer(9)]], constant uint& K [[buffer(10)]],
                           constant uint& bits [[buffer(11)]],
                           uint tg [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
                           uint T [[threads_per_threadgroup]], threadgroup float* z [[threadgroup(0)]]) {
    uint g=tg%ng, m=tg/ng, pgb=(gs*bits+7u)/8u;
    device const uchar* pg = packed + ((size_t)m*ng + g)*pgb;
    for (uint k=t;k<gs;k+=T){ uint dst=perm[k]; z[dst]=(float)codebook[cq_idx(pg, k, bits)]*(float)rsign[dst]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint h=1;h<gs;h<<=1){ for(uint k=t;k<gs;k+=T) if((k&h)==0){float a=z[k],b=z[k+h]; z[k]=a+b; z[k+h]=a-b;} threadgroup_barrier(mem_flags::mem_threadgroup); }
    float nrm=(float)norms[(size_t)m*ng+g], inv=rsqrt((float)gs);
    for (uint k=t;k<gs;k+=T){ uint col=g*gs+k; out[(size_t)m*K+col]=(half)(z[k]*inv*(float)lsign[k]*nrm*(float)recip[col]); }
}

kernel void gather_f16(device const half* table [[buffer(0)]], device const uint* rows [[buffer(1)]],
                       device half* out [[buffer(2)]], constant uint& D [[buffer(3)]],
                       constant uint& n [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i>=n) return;
    uint row=i/D, d=i%D;
    out[i] = table[(size_t)rows[row]*D + d];
}

kernel void copy_bytes(device const uchar* in [[buffer(0)]], device uchar* out [[buffer(1)]],
                       constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i<n) out[i]=in[i];
}

kernel void binary_f16(device const half* a [[buffer(0)]], device const half* b [[buffer(1)]],
                       device half* y [[buffer(2)]], constant uint& n [[buffer(3)]],
                       constant int& op [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    uint i0 = i * 4;
    if (i0 >= n) return;
    uint cnt = min(4u, n - i0);
    for (uint j = 0; j < cnt; ++j) {
        float av=(float)a[i0+j], bv=(float)b[i0+j], r;
        switch(op){ case 2: r=av-bv; break; case 3: r=av*bv; break; case 4: r=av/bv; break;
                    case 5: r=(av!=bv)?1.0f:0.0f; break; default: r=av+bv; }
        if (op==1) r=clamp(r,-65500.0f,65500.0f);
        y[i0+j]=(half)r;
    }
}

static inline float erf_approx(float x) {
    float sgn = x < 0.0f ? -1.0f : 1.0f;
    float ax = fabs(x);
    float t = 1.0f / (1.0f + 0.3275911f * ax);
    float poly = t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f
               + t * (-1.453152027f + t * 1.061405429f))));
    return sgn * (1.0f - poly * precise::exp(-ax * ax));
}

static inline float scalar_apply(float v, float p, int op) {
    switch (op) {
        case 0: return v+p;
        case 1: return v-p;
        case 3: return v/p;
        case 4: return precise::exp(v);
        case 5: return precise::sqrt(v);
        case 6: return precise::cos(v);
        case 7: return precise::sin(v);
        case 8: return precise::log(v);
        case 9: return fabs(v);
        case 10: return precise::pow(v, p);
        case 11: return (v != p) ? 1.0f : 0.0f;
        case 12: return (v > 0.0f) ? v : v*p;
        default: return v*p;
    }
}

kernel void scalar_f16(device const half* in [[buffer(0)]], device half* y [[buffer(1)]],
                       constant uint& n [[buffer(2)]], constant int& op [[buffer(3)]],
                       constant float& p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    uint i0 = i * 4;
    if (i0 >= n) return;
    uint cnt = min(4u, n - i0);
    for (uint j = 0; j < cnt; ++j) y[i0+j]=(half)scalar_apply((float)in[i0+j], p, op);
}

kernel void unary_f16(device const half* in [[buffer(0)]], device half* y [[buffer(1)]],
                      constant uint& n [[buffer(2)]], constant int& op [[buffer(3)]],
                      uint i [[thread_position_in_grid]]) {
    uint i0 = i * 4;
    if (i0 >= n) return;
    uint cnt = min(4u, n - i0);
    for (uint j = 0; j < cnt; ++j) {
        float x=(float)in[i0+j], r;
        if (op==0) r=gelu_tanh(x);
        else if (op==1) r=precise::tanh(x);
        else if (op==2) r=x/(1.0f+precise::exp(-x));
        else if (op==4) r=0.5f*x*(1.0f+erf_approx(x*0.70710678f));
        else if (op==5) r=1.0f/(1.0f+precise::exp(-x));
        else r=max(x,0.0f);
        y[i0+j]=(half)r;
    }
}

kernel void swiglu_f16(device const half* gate [[buffer(0)]], device const half* up [[buffer(1)]],
                       device half* y [[buffer(2)]], constant uint& n [[buffer(3)]],
                       constant float& scale [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i>=n) return;
    float x=(float)gate[i];
    half g1=(half)gelu_tanh(x);
    half g2=(half)((float)g1*scale);
    y[i]=(half)((float)g2*(float)up[i]);
}

kernel void rms_norm_add_simd_f16(device const half* x [[buffer(0)]], device const half* res [[buffer(1)]],
                                  device const half* w [[buffer(2)]], device half* ysum [[buffer(3)]],
                                  device half* ynorm [[buffer(4)]], constant uint& dim [[buffer(5)]],
                                  constant float& eps [[buffer(6)]], constant uint& rows [[buffer(7)]],
                                  constant uint& clipped [[buffer(8)]],
                                  uint tg [[threadgroup_position_in_grid]],
                                  uint sgid [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
    uint row = tg * 4 + sgid;
    if (row >= rows) return;
    device const half* xr = x + (size_t)row * dim;
    device const half* rr = res + (size_t)row * dim;
    device half* os = ysum + (size_t)row * dim;
    device half* on = ynorm + (size_t)row * dim;
    float partial = 0;
    for (uint i = lane; i < dim; i += 32) {
        float v = (float)xr[i] + (float)rr[i];
        if (clipped) v = clamp(v, -65500.0f, 65500.0f);
        half h = (half)v;
        os[i] = h;
        float f = (float)h;
        partial += f * f;
    }
    partial += simd_shuffle_xor(partial, 16);
    partial += simd_shuffle_xor(partial, 8);
    partial += simd_shuffle_xor(partial, 4);
    partial += simd_shuffle_xor(partial, 2);
    partial += simd_shuffle_xor(partial, 1);
    float inv = 1.0f / sqrt(partial / (float)dim + eps);
    for (uint i = lane; i < dim; i += 32) on[i] = (half)((float)os[i] * inv * (float)w[i]);
}

kernel void rms_norm_simd_f16(device const half* in [[buffer(0)]], device const half* w [[buffer(1)]],
                              device half* y [[buffer(2)]], constant uint& dim [[buffer(3)]],
                              constant float& eps [[buffer(4)]], constant uint& rows [[buffer(5)]],
                              uint tg [[threadgroup_position_in_grid]],
                              uint sgid [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
    uint row = tg * 4 + sgid;
    if (row >= rows) return;
    device const half* x = in + (size_t)row * dim;
    device half* o = y + (size_t)row * dim;
    float partial = 0;
    for (uint i = lane * 4; i + 3 < dim; i += 128) {
        half4 v4 = *(device const half4*)(x + i);
        float4 f = float4(v4);
        partial += f.x*f.x + f.y*f.y + f.z*f.z + f.w*f.w;
    }
    for (uint i = (dim & ~3u) + lane; i < dim; i += 32) { float v = (float)x[i]; partial += v*v; }
    partial += simd_shuffle_xor(partial, 16);
    partial += simd_shuffle_xor(partial, 8);
    partial += simd_shuffle_xor(partial, 4);
    partial += simd_shuffle_xor(partial, 2);
    partial += simd_shuffle_xor(partial, 1);
    float inv = 1.0f / sqrt(partial / (float)dim + eps);
    for (uint i = lane * 4; i + 3 < dim; i += 128) {
        half4 v4 = *(device const half4*)(x + i);
        half4 w4 = *(device const half4*)(w + i);
        float4 f = float4(v4) * inv * float4(w4);
        *(device half4*)(o + i) = half4(f);
    }
    for (uint i = (dim & ~3u) + lane; i < dim; i += 32) o[i] = (half)((float)x[i] * inv * (float)w[i]);
}

kernel void rms_norm_f16(device const half* in [[buffer(0)]], device const half* w [[buffer(1)]],
                         device half* y [[buffer(2)]], constant uint& dim [[buffer(3)]],
                         constant float& eps [[buffer(4)]],
                         uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
                         uint nt [[threads_per_threadgroup]], threadgroup float* red [[threadgroup(0)]]) {
    device const half* x = in + (size_t)row*dim;
    device half* o = y + (size_t)row*dim;
    float partial=0; for (uint i=t;i<dim;i+=nt){ float v=(float)x[i]; partial+=v*v; }
    red[t]=partial; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s=nt/2; s>0; s>>=1){ if (t<s) red[t]+=red[t+s]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float inv = 1.0f/sqrt(red[0]/(float)dim + eps);
    for (uint i=t;i<dim;i+=nt) o[i]=(half)((float)x[i]*inv*(float)w[i]);
}
kernel void rms_norm_add_f16(device const half* in [[buffer(0)]], device const half* w [[buffer(1)]],
                             device const half* res [[buffer(2)]], device half* y [[buffer(3)]],
                             constant uint& dim [[buffer(4)]], constant float& eps [[buffer(5)]],
                             uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
                             uint nt [[threads_per_threadgroup]], threadgroup float* red [[threadgroup(0)]]) {
    device const half* x = in + (size_t)row*dim;
    device const half* r = res + (size_t)row*dim;
    device half* o = y + (size_t)row*dim;
    float partial=0; for (uint i=t;i<dim;i+=nt){ float v=(float)x[i]; partial+=v*v; }
    red[t]=partial; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s=nt/2; s>0; s>>=1){ if (t<s) red[t]+=red[t+s]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float inv = 1.0f/sqrt(red[0]/(float)dim + eps);
    for (uint i=t;i<dim;i+=nt){ float rr=(float)r[i]+(float)(half)((float)x[i]*inv*(float)w[i]); o[i]=(half)clamp(rr,-65500.0f,65500.0f); }
}
kernel void rms_norm_add_scale_f16(device const half* in [[buffer(0)]], device const half* w [[buffer(1)]],
                             device const half* res [[buffer(2)]], device half* y [[buffer(3)]],
                             constant uint& dim [[buffer(4)]], constant float& eps [[buffer(5)]],
                             constant float& out_scale [[buffer(6)]],
                             uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
                             uint nt [[threads_per_threadgroup]], threadgroup float* red [[threadgroup(0)]]) {
    device const half* x = in + (size_t)row*dim;
    device const half* r = res + (size_t)row*dim;
    device half* o = y + (size_t)row*dim;
    float partial=0; for (uint i=t;i<dim;i+=nt){ float v=(float)x[i]; partial+=v*v; }
    red[t]=partial; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s=nt/2; s>0; s>>=1){ if (t<s) red[t]+=red[t+s]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float inv = 1.0f/sqrt(red[0]/(float)dim + eps);
    for (uint i=t;i<dim;i+=nt){ float rr=((float)r[i]+(float)(half)((float)x[i]*inv*(float)w[i]))*out_scale; o[i]=(half)clamp(rr,-65500.0f,65500.0f); }
}
kernel void rms_norm_add_rms_f16(device const half* in [[buffer(0)]], device const half* w1 [[buffer(1)]],
                                 device const half* res [[buffer(2)]], device half* h_out [[buffer(3)]],
                                 device const half* w2 [[buffer(4)]], device half* xn_out [[buffer(5)]],
                                 constant uint& dim [[buffer(6)]], constant float& eps [[buffer(7)]],
                                 constant float& out_scale [[buffer(8)]],
                                 uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
                                 uint nt [[threads_per_threadgroup]], threadgroup float* red [[threadgroup(0)]]) {
    device const half* x = in + (size_t)row*dim;
    device const half* r = res + (size_t)row*dim;
    device half* o = h_out + (size_t)row*dim;
    device half* o2 = xn_out + (size_t)row*dim;
    float partial=0; for (uint i=t;i<dim;i+=nt){ float v=(float)x[i]; partial+=v*v; }
    red[t]=partial; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s=nt/2; s>0; s>>=1){ if (t<s) red[t]+=red[t+s]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float inv = 1.0f/sqrt(red[0]/(float)dim + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float p2=0;
    for (uint i=t;i<dim;i+=nt){
        float rr=((float)r[i]+(float)(half)((float)x[i]*inv*(float)w1[i]))*out_scale;
        half hv=(half)clamp(rr,-65500.0f,65500.0f);
        o[i]=hv;
        float v=(float)hv; p2+=v*v;
    }
    red[t]=p2; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s=nt/2; s>0; s>>=1){ if (t<s) red[t]+=red[t+s]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float inv2 = 1.0f/sqrt(red[0]/(float)dim + eps);
    for (uint i=t;i<dim;i+=nt) o2[i]=(half)((float)o[i]*inv2*(float)w2[i]);
}

kernel void cast_f16_f32(device const half* in [[buffer(0)]], device float* out [[buffer(1)]], constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]){ if(i<n) out[i]=(float)in[i]; }
kernel void cast_f32_f16(device const float* in [[buffer(0)]], device half* out [[buffer(1)]], constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]){ if(i<n) out[i]=(half)in[i]; }
kernel void cast_i8_f16(device const char* in [[buffer(0)]], device half* out [[buffer(1)]], constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]){ if(i<n) out[i]=(half)(float)in[i]; }
kernel void cast_f16_i8(device const half* in [[buffer(0)]], device char* out [[buffer(1)]], constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]){ if(i<n){ float v=rint((float)in[i]); out[i]=(char)clamp(v,-128.0f,127.0f);} }

kernel void strided_copy_f16(device const half* in [[buffer(0)]], device half* out [[buffer(1)]],
    constant uint* oshape [[buffer(2)]], constant uint* sstride [[buffer(3)]],
    constant uint& ndim [[buffer(4)]], constant uint& total [[buffer(5)]], constant uint& base [[buffer(6)]],
    uint i [[thread_position_in_grid]]) {
    if (i>=total) return;
    uint rem=i, src=base;
    for (int d=int(ndim)-1; d>=0; --d){ uint c=rem%oshape[d]; rem/=oshape[d]; src+=c*sstride[d]; }
    out[i]=in[src];
}

kernel void transpose2d_f16(device const half* in [[buffer(0)]], device half* y [[buffer(1)]],
                             constant uint& R [[buffer(2)]], constant uint& C [[buffer(3)]],
                             uint3 tgid [[threadgroup_position_in_grid]],
                             uint3 tp3 [[thread_position_in_threadgroup]]) {
    uint2 tp = tp3.xy;
    threadgroup half tile[32][33];
    uint b = tgid.z;
    uint c0 = tgid.x * 32, r0 = tgid.y * 32;
    device const half* src = in + (size_t)b * R * C;
    device half* dst = y + (size_t)b * R * C;
    for (uint dr = tp.y; dr < 32; dr += 8) {
        uint r = r0 + dr, c = c0 + tp.x;
        tile[dr][tp.x] = (r < R && c < C) ? src[(size_t)r * C + c] : (half)0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint dr = tp.y; dr < 32; dr += 8) {
        uint c = c0 + dr, r = r0 + tp.x;
        if (c < C && r < R) dst[(size_t)c * R + r] = tile[tp.x][dr];
    }
}

kernel void strided_copy_rows_f16(device const half* in [[buffer(0)]], device half* out [[buffer(1)]],
    constant uint* oshape [[buffer(2)]], constant uint* sstride [[buffer(3)]],
    constant uint& ndim [[buffer(4)]], constant uint& rows [[buffer(5)]], constant uint& base [[buffer(6)]],
    constant uint& inner4 [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]) {
    uint v = gid.x, row = gid.y;
    if (v >= inner4 || row >= rows) return;
    uint rem = row, src = base;
    for (int d = int(ndim) - 1; d >= 0; --d) { uint c = rem % oshape[d]; rem /= oshape[d]; src += c * sstride[d]; }
    device const half4* s4 = (device const half4*)(in + src);
    device half4* d4 = (device half4*)(out + (size_t)row * inner4 * 4);
    d4[v] = s4[v];
}

kernel void kv_append_i8(device const half* src [[buffer(0)]], device char* int8base [[buffer(1)]],
    device float* scalebase [[buffer(2)]], constant uint& kv_heads [[buffer(3)]],
    constant uint& hdim [[buffer(4)]], constant uint& current_len [[buffer(5)]], constant uint& group_size [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    uint num_groups = (hdim + group_size - 1)/group_size;
    if (gid >= kv_heads*num_groups) return;
    uint h = gid / num_groups, g = gid % num_groups;
    uint gstart = g*group_size, gcount = min(group_size, hdim - gstart);
    device const half* hs = src + (size_t)h*hdim + gstart;
    float maxabs = 0;
    for (uint k=0;k<gcount;++k) maxabs = max(maxabs, fabs((float)hs[k]));
    float scale = maxabs/127.0f; if (scale < 1e-10f) scale = 1e-10f;
    float inv = 1.0f/scale;
    uint int8_stride = kv_heads*hdim, scale_stride = kv_heads*num_groups;
    device char* dst = int8base + (size_t)current_len*int8_stride + (size_t)h*hdim + gstart;
    for (uint k=0;k<gcount;++k){ float q = clamp(rint((float)hs[k]*inv), -128.0f, 127.0f); dst[k]=(char)q; }
    scalebase[(size_t)current_len*scale_stride + (size_t)h*num_groups + g] = scale;
}

kernel void kv_slide_save(device const char* int8base [[buffer(0)]], device const float* scalebase [[buffer(1)]],
    device char* scr_i8 [[buffer(2)]], device float* scr_sc [[buffer(3)]],
    constant uint& kv_heads [[buffer(4)]], constant uint& hdim [[buffer(5)]], constant uint& group_size [[buffer(6)]],
    constant uint& shift_src [[buffer(7)]], constant uint& remaining [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {
    uint num_groups=(hdim+group_size-1)/group_size, per=kv_heads*num_groups;
    if (gid >= remaining*per) return;
    uint s=gid/per, hg=gid%per, h=hg/num_groups, g=hg%num_groups;
    uint gstart=g*group_size, gcount=min(group_size, hdim-gstart);
    uint i8s=kv_heads*hdim, scs=kv_heads*num_groups, srcseq=shift_src+s;
    device const char* sp=int8base+(size_t)srcseq*i8s+(size_t)h*hdim+gstart;
    device char* dp=scr_i8+(size_t)s*i8s+(size_t)h*hdim+gstart;
    for (uint k=0;k<gcount;++k) dp[k]=sp[k];
    scr_sc[(size_t)s*scs+(size_t)h*num_groups+g]=scalebase[(size_t)srcseq*scs+(size_t)h*num_groups+g];
}

kernel void kv_slide_restore(device const half* src [[buffer(0)]], device char* int8base [[buffer(1)]],
    device float* scalebase [[buffer(2)]], device const char* scr_i8 [[buffer(3)]], device const float* scr_sc [[buffer(4)]],
    constant uint& kv_heads [[buffer(5)]], constant uint& hdim [[buffer(6)]], constant uint& group_size [[buffer(7)]],
    constant uint& keep_sink [[buffer(8)]], constant uint& remaining [[buffer(9)]],
    uint gid [[thread_position_in_grid]]) {
    uint num_groups=(hdim+group_size-1)/group_size, per=kv_heads*num_groups;
    if (gid >= (remaining+1)*per) return;
    uint s=gid/per, hg=gid%per, h=hg/num_groups, g=hg%num_groups;
    uint gstart=g*group_size, gcount=min(group_size, hdim-gstart);
    uint i8s=kv_heads*hdim, scs=kv_heads*num_groups, dstseq=keep_sink+s;
    device char* dp=int8base+(size_t)dstseq*i8s+(size_t)h*hdim+gstart;
    if (s < remaining) {
        device const char* sp=scr_i8+(size_t)s*i8s+(size_t)h*hdim+gstart;
        for (uint k=0;k<gcount;++k) dp[k]=sp[k];
        scalebase[(size_t)dstseq*scs+(size_t)h*num_groups+g]=scr_sc[(size_t)s*scs+(size_t)h*num_groups+g];
    } else {
        device const half* hs=src+(size_t)h*hdim+gstart;
        float maxabs=0; for (uint k=0;k<gcount;++k) maxabs=max(maxabs,fabs((float)hs[k]));
        float scale=maxabs/127.0f; if (scale<1e-10f) scale=1e-10f; float inv=1.0f/scale;
        for (uint k=0;k<gcount;++k){ float q=clamp(rint((float)hs[k]*inv),-128.0f,127.0f); dp[k]=(char)q; }
        scalebase[(size_t)dstseq*scs+(size_t)h*num_groups+g]=scale;
    }
}

kernel void kv_slide_restore_m(device const half* src [[buffer(0)]], device char* int8base [[buffer(1)]],
    device float* scalebase [[buffer(2)]], device const char* scr_i8 [[buffer(3)]], device const float* scr_sc [[buffer(4)]],
    constant uint& kv_heads [[buffer(5)]], constant uint& hdim [[buffer(6)]], constant uint& group_size [[buffer(7)]],
    constant uint& keep_sink [[buffer(8)]], constant uint& remaining [[buffer(9)]], constant uint& M [[buffer(10)]],
    uint gid [[thread_position_in_grid]]) {
    uint num_groups=(hdim+group_size-1)/group_size, per=kv_heads*num_groups;
    if (gid >= (remaining+M)*per) return;
    uint s=gid/per, hg=gid%per, h=hg/num_groups, g=hg%num_groups;
    uint gstart=g*group_size, gcount=min(group_size, hdim-gstart);
    uint i8s=kv_heads*hdim, scs=kv_heads*num_groups, dstseq=keep_sink+s;
    device char* dp=int8base+(size_t)dstseq*i8s+(size_t)h*hdim+gstart;
    if (s < remaining) {
        device const char* sp=scr_i8+(size_t)s*i8s+(size_t)h*hdim+gstart;
        for (uint k=0;k<gcount;++k) dp[k]=sp[k];
        scalebase[(size_t)dstseq*scs+(size_t)h*num_groups+g]=scr_sc[(size_t)s*scs+(size_t)h*num_groups+g];
    } else {
        uint t=s-remaining;
        device const half* hs=src+(size_t)t*kv_heads*hdim+(size_t)h*hdim+gstart;
        float maxabs=0; for (uint k=0;k<gcount;++k) maxabs=max(maxabs,fabs((float)hs[k]));
        float scale=maxabs/127.0f; if (scale<1e-10f) scale=1e-10f; float inv=1.0f/scale;
        for (uint k=0;k<gcount;++k){ float q=clamp(rint((float)hs[k]*inv),-128.0f,127.0f); dp[k]=(char)q; }
        scalebase[(size_t)dstseq*scs+(size_t)h*num_groups+g]=scale;
    }
}

kernel void strided_scatter_f16(device const half* in [[buffer(0)]], device half* out [[buffer(1)]],
    constant uint* ishape [[buffer(2)]], constant uint* ostride [[buffer(3)]],
    constant uint& ndim [[buffer(4)]], constant uint& total [[buffer(5)]], constant uint& base [[buffer(6)]],
    uint i [[thread_position_in_grid]]) {
    if (i>=total) return;
    uint rem=i, dst=base;
    for (int d=int(ndim)-1; d>=0; --d){ uint c=rem%ishape[d]; rem/=ishape[d]; dst+=c*ostride[d]; }
    out[dst]=in[i];
}

kernel void strided_scatter_rows_f16(device const half* in [[buffer(0)]], device half* out [[buffer(1)]],
    constant uint* ishape [[buffer(2)]], constant uint* ostride [[buffer(3)]],
    constant uint& ndim [[buffer(4)]], constant uint& rows [[buffer(5)]], constant uint& base [[buffer(6)]],
    constant uint& inner4 [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]) {
    uint v = gid.x, row = gid.y;
    if (v >= inner4 || row >= rows) return;
    uint rem = row, dst = base;
    for (int d = int(ndim) - 1; d >= 0; --d) { uint c = rem % ishape[d]; rem /= ishape[d]; dst += c * ostride[d]; }
    device const half4* s4 = (device const half4*)(in + (size_t)row * inner4 * 4);
    device half4* d4 = (device half4*)(out + dst);
    d4[v] = s4[v];
}

kernel void binary_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                       device float* y [[buffer(2)]], constant uint& n [[buffer(3)]],
                       constant int& op [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    uint i0 = i * 4;
    if (i0 >= n) return;
    uint cnt = min(4u, n - i0);
    for (uint j = 0; j < cnt; ++j) {
        float av=a[i0+j], bv=b[i0+j], r;
        switch(op){ case 2: r=av-bv; break; case 3: r=av*bv; break; case 4: r=av/bv; break;
                    case 5: r=(av!=bv)?1.0f:0.0f; break; default: r=av+bv; }
        if (op==1) r=clamp(r,-65500.0f,65500.0f);
        y[i0+j]=r;
    }
}
kernel void scalar_f32(device const float* in [[buffer(0)]], device float* y [[buffer(1)]],
                       constant uint& n [[buffer(2)]], constant int& op [[buffer(3)]],
                       constant float& p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    uint i0 = i * 4;
    if (i0 >= n) return;
    uint cnt = min(4u, n - i0);
    for (uint j = 0; j < cnt; ++j) y[i0+j]=scalar_apply(in[i0+j], p, op);
}
kernel void unary_f32(device const float* in [[buffer(0)]], device float* y [[buffer(1)]],
                      constant uint& n [[buffer(2)]], constant int& op [[buffer(3)]],
                      uint i [[thread_position_in_grid]]) {
    uint i0 = i * 4;
    if (i0 >= n) return;
    uint cnt = min(4u, n - i0);
    for (uint j = 0; j < cnt; ++j) {
        float x=in[i0+j], r;
        if (op==0) r=gelu_tanh(x);
        else if (op==1) r=precise::tanh(x);
        else if (op==2) r=x/(1.0f+precise::exp(-x));
        else if (op==4) r=0.5f*x*(1.0f+erf_approx(x*0.70710678f));
        else if (op==5) r=1.0f/(1.0f+precise::exp(-x));
        else r=max(x,0.0f);
        y[i0+j]=r;
    }
}

kernel void clamp_f16(device const half* in [[buffer(0)]], device half* y [[buffer(1)]],
                      constant uint& n [[buffer(2)]], constant float& lo [[buffer(3)]],
                      constant float& hi [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    uint i0 = i * 4;
    if (i0 >= n) return;
    uint cnt = min(4u, n - i0);
    for (uint j = 0; j < cnt; ++j) y[i0+j]=(half)clamp((float)in[i0+j], lo, hi);
}

kernel void clamp_f32(device const float* in [[buffer(0)]], device float* y [[buffer(1)]],
                      constant uint& n [[buffer(2)]], constant float& lo [[buffer(3)]],
                      constant float& hi [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    uint i0 = i * 4;
    if (i0 >= n) return;
    uint cnt = min(4u, n - i0);
    for (uint j = 0; j < cnt; ++j) y[i0+j]=clamp(in[i0+j], lo, hi);
}
kernel void glu_f16(device const half* x [[buffer(0)]], device half* y [[buffer(1)]],
                    constant uint& split [[buffer(2)]], constant uint& inner [[buffer(3)]],
                    constant uint& n [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i>=n) return;
    uint per_outer = split*inner;
    uint o = i/per_outer, r = i%per_outer;
    size_t base = (size_t)o*2u*per_outer;
    float a = (float)x[base + r];
    float g = (float)x[base + per_outer + r];
    y[i] = (half)(a / (1.0f + precise::exp(-g)));
}

kernel void layer_norm_f16(device const half* in [[buffer(0)]], device const half* w [[buffer(1)]],
                           device const half* b [[buffer(2)]], device half* y [[buffer(3)]],
                           constant uint& dim [[buffer(4)]], constant float& eps [[buffer(5)]],
                           constant uint& has_bias [[buffer(6)]],
                           uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
                           uint nt [[threads_per_threadgroup]], threadgroup float* red [[threadgroup(0)]]) {
    device const half* x = in + (size_t)row*dim;
    device half* o = y + (size_t)row*dim;
    float s=0, ss=0;
    for (uint i=t;i<dim;i+=nt){ float v=(float)x[i]; s+=v; ss+=v*v; }
    red[t]=s; red[nt+t]=ss; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint st=nt/2; st>0; st>>=1){ if (t<st){ red[t]+=red[t+st]; red[nt+t]+=red[nt+t+st]; } threadgroup_barrier(mem_flags::mem_threadgroup); }
    float mean = red[0]/(float)dim;
    float var = red[nt]/(float)dim - mean*mean;
    float inv = 1.0f/sqrt(var + eps);
    for (uint i=t;i<dim;i+=nt){
        float v = ((float)x[i]-mean)*inv*(float)w[i];
        if (has_bias != 0u) v += (float)b[i];
        o[i]=(half)v;
    }
}

kernel void softmax_rows_f16(device const half* in [[buffer(0)]], device half* y [[buffer(1)]],
                             constant uint& cols [[buffer(2)]],
                             uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
                             uint nt [[threads_per_threadgroup]], threadgroup float* red [[threadgroup(0)]]) {
    device const half* x = in + (size_t)row*cols;
    device half* o = y + (size_t)row*cols;
    float m=-INFINITY; for (uint i=t;i<cols;i+=nt) m=max(m,(float)x[i]);
    red[t]=m; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s=nt/2; s>0; s>>=1){ if (t<s) red[t]=max(red[t],red[t+s]); threadgroup_barrier(mem_flags::mem_threadgroup); }
    m=red[0]; threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum=0; for (uint i=t;i<cols;i+=nt) sum+=exp((float)x[i]-m);
    red[t]=sum; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s=nt/2; s>0; s>>=1){ if (t<s) red[t]+=red[t+s]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float inv = red[0]>0.0f ? 1.0f/red[0] : 0.0f;
    for (uint i=t;i<cols;i+=nt) o[i]=(half)(exp((float)x[i]-m)*inv);
}

kernel void conv1d_k3_f16(device const half* x [[buffer(0)]], device const half* w [[buffer(1)]],
                          device half* y [[buffer(2)]],
                          constant uint& Cin [[buffer(3)]], constant uint& L [[buffer(4)]],
                          constant uint& Lout [[buffer(5)]], constant uint& stride [[buffer(6)]],
                          uint2 gid [[thread_position_in_grid]]) {
    if (gid.x>=Lout) return;
    int center = (int)(gid.x*stride);
    float acc=0;
    device const half* wr = w + (size_t)gid.y*Cin*3u;
    for (uint c=0;c<Cin;++c){
        device const half* xc = x + (size_t)c*L;
        float x0 = (center>0) ? (float)xc[center-1] : 0.0f;
        float x1 = (float)xc[center];
        float x2 = (center+1<(int)L) ? (float)xc[center+1] : 0.0f;
        acc = fma(x0,(float)wr[c*3u],acc);
        acc = fma(x1,(float)wr[c*3u+1u],acc);
        acc = fma(x2,(float)wr[c*3u+2u],acc);
    }
    y[(size_t)gid.y*Lout + gid.x] = (half)acc;
}

struct AttnFU { uint T, S, HQ, HKV, D, DV; float scale; uint causal, pos_off, window; float logit_cap; uint mask_mode; };

static inline float attn_apply_mask(float score, device const half* mask, constant AttnFU& U,
                                    uint b, uint h, uint t, uint i) {
    if (U.mask_mode != 0u) {
        size_t mi = (U.mask_mode>=3u) ? (((size_t)b*U.HQ + h)*U.T + t)*U.S + i
                                      : ((size_t)b*U.T + t)*U.S + i;
        float mv = (float)mask[mi];
        if (U.mask_mode==1u || U.mask_mode==3u) { score = isfinite(mv) ? score+mv : -INFINITY; }
        else if (mv == 0.0f) score = -INFINITY;
    }
    if (U.logit_cap > 0.0f && score > -INFINITY) score = U.logit_cap * precise::tanh(score / U.logit_cap);
    return score;
}

kernel void attn_f16(device const half* q [[buffer(0)]], device const half* k [[buffer(1)]],
                     device const half* v [[buffer(2)]], device half* o [[buffer(3)]],
                     device const half* mask [[buffer(4)]], constant AttnFU& U [[buffer(5)]],
                     uint3 tg [[threadgroup_position_in_grid]], uint l [[thread_index_in_threadgroup]]) {
    const uint QR = 4u;
    uint t0 = tg.x * QR, h = tg.y, b = tg.z;
    uint kvh = h / (U.HQ / U.HKV);
    device const half* kb = k + (size_t)b*U.S*U.HKV*U.D + (size_t)kvh*U.D;
    device const half* vb = v + (size_t)b*U.S*U.HKV*U.DV + (size_t)kvh*U.DV;
    uint d4 = U.D/4u, dv4 = U.DV/4u;
    threadgroup half4 qs[4u*32u];
    uint nq = min(QR, U.T - t0);
    for (uint r=0;r<nq;++r) {
        device const half4* qr = (device const half4*)(q + ((size_t)b*U.T + t0 + r)*U.HQ*U.D + (size_t)h*U.D);
        for (uint d=l; d<d4; d+=32u) qs[r*32u + d] = qr[d];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    float4 acc[4][2];
    for (uint r=0;r<4u;++r){ acc[r][0]=float4(0.0f); acc[r][1]=float4(0.0f); }
    float m[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint aq1 = U.pos_off + t0 + nq - 1u;
    uint kend_all = U.causal ? min(U.S, aq1+1u) : U.S;
    uint kstart = 0u;
    if (U.window>0u) { uint aq0 = U.pos_off + t0; kstart = (aq0>U.window) ? aq0-U.window : 0u; }
    size_t kstride = (size_t)U.HKV*U.D, vstride = (size_t)U.HKV*U.DV;
    for (uint kv0=kstart; kv0<kend_all; kv0+=32u) {
        uint i = kv0 + l;
        float4 sc = float4(-INFINITY);
        if (i < kend_all) {
            device const half4* kr = (device const half4*)(kb + (size_t)i*kstride);
            float4 p = float4(0.0f);
            for (uint d=0; d<d4; ++d) {
                float4 kv4 = float4(kr[d]);
                float4 q0 = float4(qs[d]);
                p.x = fma(kv4.x,q0.x,p.x); p.x = fma(kv4.y,q0.y,p.x); p.x = fma(kv4.z,q0.z,p.x); p.x = fma(kv4.w,q0.w,p.x);
                if (nq>1u){ float4 q1 = float4(qs[32u+d]);
                    p.y = fma(kv4.x,q1.x,p.y); p.y = fma(kv4.y,q1.y,p.y); p.y = fma(kv4.z,q1.z,p.y); p.y = fma(kv4.w,q1.w,p.y); }
                if (nq>2u){ float4 q2 = float4(qs[64u+d]);
                    p.z = fma(kv4.x,q2.x,p.z); p.z = fma(kv4.y,q2.y,p.z); p.z = fma(kv4.z,q2.z,p.z); p.z = fma(kv4.w,q2.w,p.z); }
                if (nq>3u){ float4 q3 = float4(qs[96u+d]);
                    p.w = fma(kv4.x,q3.x,p.w); p.w = fma(kv4.y,q3.y,p.w); p.w = fma(kv4.z,q3.z,p.w); p.w = fma(kv4.w,q3.w,p.w); }
            }
            sc = p * U.scale;
            for (uint r=0;r<4u;++r) {
                if (r >= nq) { sc[r] = -INFINITY; continue; }
                uint t = t0 + r, aq = U.pos_off + t;
                if (U.causal && i > aq) { sc[r] = -INFINITY; continue; }
                if (U.window>0u && i < aq && (aq - i) > U.window) { sc[r] = -INFINITY; continue; }
                sc[r] = attn_apply_mask(sc[r], mask, U, b, h, t, i);
            }
        }
        float4 wgt = float4(0.0f);
        bool anyrow = false;
        for (uint r=0;r<4u;++r) {
            float bm = simd_max(sc[r]);
            if (bm > -INFINITY) {
                anyrow = true;
                if (bm > m[r]) {
                    float corr = exp(m[r] - bm);
                    sum[r] *= corr; acc[r][0] *= corr; acc[r][1] *= corr;
                    m[r] = bm;
                }
                wgt[r] = (sc[r] > -INFINITY) ? exp(sc[r] - m[r]) : 0.0f;
                sum[r] += simd_sum(wgt[r]);
            }
        }
        if (!anyrow) continue;
        uint nk = min(32u, kend_all - kv0);
        uint half_id = l >> 4u, dl = l & 15u;
        for (uint j2=0; j2<nk; j2+=2u) {
            uint key = j2 + half_id;
            ushort src = (ushort)min(key, 31u);
            float4 wk;
            for (uint r=0;r<4u;++r) wk[r] = simd_shuffle(wgt[r], src);
            if (key < nk && (wk.x != 0.0f || wk.y != 0.0f || wk.z != 0.0f || wk.w != 0.0f)) {
                device const half4* vr = (device const half4*)(vb + (size_t)(kv0+key)*vstride);
                uint ci = 0u;
                for (uint dd=dl; dd<dv4; dd+=16u, ++ci) {
                    float4 vv = float4(vr[dd]);
                    acc[0][ci] = fma(float4(wk.x), vv, acc[0][ci]);
                    acc[1][ci] = fma(float4(wk.y), vv, acc[1][ci]);
                    acc[2][ci] = fma(float4(wk.z), vv, acc[2][ci]);
                    acc[3][ci] = fma(float4(wk.w), vv, acc[3][ci]);
                }
            }
        }
    }
    uint dl = l & 15u;
    for (uint r=0;r<nq;++r) {
        float inv = sum[r]>0.0f ? 1.0f/sum[r] : 0.0f;
        device half4* orow = (device half4*)(o + ((size_t)b*U.T + t0 + r)*U.HQ*U.DV + (size_t)h*U.DV);
        uint ci = 0u;
        for (uint dd=dl; dd<dv4; dd+=16u, ++ci) {
            float4 vv;
            vv[0] = acc[r][ci][0] + simd_shuffle_xor(acc[r][ci][0], (ushort)16);
            vv[1] = acc[r][ci][1] + simd_shuffle_xor(acc[r][ci][1], (ushort)16);
            vv[2] = acc[r][ci][2] + simd_shuffle_xor(acc[r][ci][2], (ushort)16);
            vv[3] = acc[r][ci][3] + simd_shuffle_xor(acc[r][ci][3], (ushort)16);
            if (l < 16u) orow[dd] = half4(vv*inv);
        }
    }
}

kernel void attn_f16_d64(device const half* q [[buffer(0)]], device const half* k [[buffer(1)]],
                     device const half* v [[buffer(2)]], device half* o [[buffer(3)]],
                     device const half* mask [[buffer(4)]], constant AttnFU& U [[buffer(5)]],
                     uint3 tg [[threadgroup_position_in_grid]], uint l [[thread_index_in_threadgroup]]) {
    const uint QR = 8u;
    uint t0 = tg.x * QR, h = tg.y, b = tg.z;
    uint kvh = h / (U.HQ / U.HKV);
    device const half* kb = k + (size_t)b*U.S*U.HKV*U.D + (size_t)kvh*U.D;
    device const half* vb = v + (size_t)b*U.S*U.HKV*U.DV + (size_t)kvh*U.DV;
    uint d4 = U.D/4u, dv4 = U.DV/4u;
    threadgroup half4 qs[8u*16u];
    uint nq = min(QR, U.T - t0);
    for (uint r=0;r<nq;++r) {
        device const half4* qr = (device const half4*)(q + ((size_t)b*U.T + t0 + r)*U.HQ*U.D + (size_t)h*U.D);
        for (uint d=l; d<d4; d+=32u) qs[r*16u + d] = qr[d];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    float4 acc[8];
    for (uint r=0;r<8u;++r) acc[r]=float4(0.0f);
    float m[8], sum[8];
    for (uint r=0;r<8u;++r){ m[r]=-INFINITY; sum[r]=0.0f; }
    uint aq1 = U.pos_off + t0 + nq - 1u;
    uint kend_all = U.causal ? min(U.S, aq1+1u) : U.S;
    uint kstart = 0u;
    if (U.window>0u) { uint aq0 = U.pos_off + t0; kstart = (aq0>U.window) ? aq0-U.window : 0u; }
    size_t kstride = (size_t)U.HKV*U.D, vstride = (size_t)U.HKV*U.DV;
    for (uint kv0=kstart; kv0<kend_all; kv0+=32u) {
        uint i = kv0 + l;
        float sc[8];
        for (uint r=0;r<8u;++r) sc[r]=-INFINITY;
        if (i < kend_all) {
            device const half4* kr = (device const half4*)(kb + (size_t)i*kstride);
            float p[8] = {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
            for (uint d=0; d<d4; ++d) {
                float4 kv4 = float4(kr[d]);
                for (uint r=0;r<8u;++r) {
                    float4 qr4 = float4(qs[r*16u+d]);
                    p[r] = fma(kv4.x,qr4.x,p[r]); p[r] = fma(kv4.y,qr4.y,p[r]);
                    p[r] = fma(kv4.z,qr4.z,p[r]); p[r] = fma(kv4.w,qr4.w,p[r]);
                }
            }
            for (uint r=0;r<8u;++r) {
                if (r >= nq) { sc[r] = -INFINITY; continue; }
                sc[r] = p[r] * U.scale;
                uint t = t0 + r, aq = U.pos_off + t;
                if (U.causal && i > aq) { sc[r] = -INFINITY; continue; }
                if (U.window>0u && i < aq && (aq - i) > U.window) { sc[r] = -INFINITY; continue; }
                sc[r] = attn_apply_mask(sc[r], mask, U, b, h, t, i);
            }
        }
        float wgt[8];
        bool anyrow = false;
        for (uint r=0;r<8u;++r) {
            float bm = simd_max(sc[r]);
            if (bm > -INFINITY) {
                anyrow = true;
                if (bm > m[r]) {
                    float corr = exp(m[r] - bm);
                    sum[r] *= corr; acc[r] *= corr;
                    m[r] = bm;
                }
                wgt[r] = (sc[r] > -INFINITY) ? exp(sc[r] - m[r]) : 0.0f;
                sum[r] += simd_sum(wgt[r]);
            } else wgt[r] = 0.0f;
        }
        if (!anyrow) continue;
        uint nk = min(32u, kend_all - kv0);
        uint half_id = l >> 4u, dl = l & 15u;
        for (uint j2=0; j2<nk; j2+=2u) {
            uint key = j2 + half_id;
            ushort src = (ushort)min(key, 31u);
            float wk[8];
            float wsum = 0.0f;
            for (uint r=0;r<8u;++r){ wk[r] = simd_shuffle(wgt[r], src); wsum += wk[r]; }
            if (key < nk && wsum != 0.0f && dl < dv4) {
                float4 vv = float4(((device const half4*)(vb + (size_t)(kv0+key)*vstride))[dl]);
                for (uint r=0;r<8u;++r) acc[r] = fma(float4(wk[r]), vv, acc[r]);
            }
        }
    }
    uint dl = l & 15u;
    for (uint r=0;r<nq;++r) {
        float inv = sum[r]>0.0f ? 1.0f/sum[r] : 0.0f;
        device half4* orow = (device half4*)(o + ((size_t)b*U.T + t0 + r)*U.HQ*U.DV + (size_t)h*U.DV);
        float4 vv;
        vv[0] = acc[r][0] + simd_shuffle_xor(acc[r][0], (ushort)16);
        vv[1] = acc[r][1] + simd_shuffle_xor(acc[r][1], (ushort)16);
        vv[2] = acc[r][2] + simd_shuffle_xor(acc[r][2], (ushort)16);
        vv[3] = acc[r][3] + simd_shuffle_xor(acc[r][3], (ushort)16);
        if (l < 16u && dl < dv4) orow[dl] = half4(vv*inv);
    }
}

kernel void attn_flash_f16(device const half* q [[buffer(0)]], device const half* k [[buffer(1)]],
                           device const half* v [[buffer(2)]], device half* y [[buffer(3)]],
                           device const half* mask [[buffer(4)]],
                           constant uint& T [[buffer(5)]], constant uint& S [[buffer(6)]],
                           constant uint& H [[buffer(7)]], constant float& scale [[buffer(8)]],
                           constant uint& mask_mode [[buffer(9)]],
                           uint2 tgid [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint sgid [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint BQ = 64, BK = 32, D = 64, NT = 256;
    threadgroup half KVs[2 * BK * D];
    threadgroup half* Ks = KVs;
    threadgroup half* Vs = KVs + BK * D;
    threadgroup float Ss[8][8 * BK];
    threadgroup half Ps[8][8 * BK];
    threadgroup float Dg[8][64];
    threadgroup float Ms[8][8];
    threadgroup float Ls[8][8];
    threadgroup uint Rf[8];

    uint qb = tgid.x, h = tgid.y;
    uint q0 = qb * BQ;
    uint row0 = q0 + sgid * 8;
    bool interior = q0 + BQ <= T;

    for (uint x = tid; x < 8 * 64; x += NT) Dg[x / 64][x % 64] = 0;
    if (tid < 64) { Ms[tid / 8][tid % 8] = -INFINITY; Ls[tid / 8][tid % 8] = 0; }

    simdgroup_half8x8 qf[8];
    if (interior) {
        for (uint d = 0; d < 8; ++d)
            simdgroup_load(qf[d], q + ((size_t)row0 * H + h) * D + d * 8, (size_t)H * D);
    } else {
        threadgroup half* Qe = KVs;
        for (uint x = tid; x < BQ * D; x += NT) {
            uint r = x / D, d = x % D;
            uint t = q0 + r;
            Qe[x] = (t < T) ? q[((size_t)t * H + h) * D + d] : (half)0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint d = 0; d < 8; ++d)
            simdgroup_load(qf[d], Qe + (size_t)sgid * 8 * D + d * 8, D);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 of[8];
    for (uint d = 0; d < 8; ++d) of[d] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint kb = 0; kb < S; kb += BK) {
        for (uint x = tid; x < BK * D / 4; x += NT) {
            uint r = (x * 4) / D, d = (x * 4) % D;
            uint sidx = kb + r;
            half4 kv4 = half4(0.0h), vv4 = half4(0.0h);
            if (sidx < S) {
                kv4 = *(device const half4*)(k + ((size_t)sidx * H + h) * D + d);
                vv4 = *(device const half4*)(v + ((size_t)sidx * H + h) * D + d);
            }
            *(threadgroup half4*)(Ks + r * D + d) = kv4;
            *(threadgroup half4*)(Vs + r * D + d) = vv4;
        }
        if (lane == 0) Rf[sgid] = 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint j = 0; j < 4; ++j) {
            simdgroup_float8x8 sf = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (uint d = 0; d < 8; ++d) {
                simdgroup_half8x8 kt;
                simdgroup_load(kt, Ks + (size_t)j * 8 * D + d * 8, D, 0, true);
                simdgroup_multiply_accumulate(sf, qf[d], kt, sf);
            }
            simdgroup_store(sf, Ss[sgid] + j * 8, BK);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        uint r = lane / 4, c0 = (lane % 4) * 8;
        uint t = row0 + r;
        float sv[8];
        float mloc = -INFINITY;
        for (uint c = 0; c < 8; ++c) {
            uint sidx = kb + c0 + c;
            float val = Ss[sgid][r * BK + c0 + c] * scale;
            bool dead = sidx >= S || t >= T;
            if (!dead && mask_mode == 2u && mask[(size_t)t * S + sidx] == (half)0.0h) dead = true;
            sv[c] = dead ? -INFINITY : val;
            mloc = max(mloc, sv[c]);
        }
        mloc = max(mloc, simd_shuffle_xor(mloc, 1));
        mloc = max(mloc, simd_shuffle_xor(mloc, 2));
        float mold = Ms[sgid][r];
        float mnew = max(mold, mloc);
        float sloc = 0;
        for (uint c = 0; c < 8; ++c) {
            float pv = (mnew == -INFINITY || sv[c] == -INFINITY) ? 0.0f : fast::exp(sv[c] - mnew);
            Ps[sgid][r * BK + c0 + c] = (half)pv;
            sloc += pv;
        }
        sloc += simd_shuffle_xor(sloc, 1);
        sloc += simd_shuffle_xor(sloc, 2);
        float alpha = (mold == -INFINITY) ? 0.0f : fast::exp(mold - mnew);
        bool first = mold == -INFINITY;
        if ((lane % 4) == 0) {
            Ms[sgid][r] = mnew;
            Ls[sgid][r] = Ls[sgid][r] * alpha + sloc;
            if (mnew != mold && !first) Rf[sgid] = 1;
            Dg[sgid][r * 8 + r] = alpha;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        bool rescale = Rf[sgid] != 0 || kb == 0;
        simdgroup_half8x8 pf[4];
        for (uint j = 0; j < 4; ++j)
            simdgroup_load(pf[j], Ps[sgid] + j * 8, BK);
        if (rescale) {
            simdgroup_float8x8 dgf;
            simdgroup_load(dgf, Dg[sgid], 8);
            for (uint d = 0; d < 8; ++d) {
                simdgroup_float8x8 acc;
                simdgroup_multiply(acc, dgf, of[d]);
                for (uint j = 0; j < 4; ++j) {
                    simdgroup_half8x8 vf;
                    simdgroup_load(vf, Vs + (size_t)j * 8 * D + d * 8, D);
                    simdgroup_multiply_accumulate(acc, pf[j], vf, acc);
                }
                of[d] = acc;
            }
        } else {
            for (uint d = 0; d < 8; ++d) {
                for (uint j = 0; j < 4; ++j) {
                    simdgroup_half8x8 vf;
                    simdgroup_load(vf, Vs + (size_t)j * 8 * D + d * 8, D);
                    simdgroup_multiply_accumulate(of[d], pf[j], vf, of[d]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    {
        uint r = lane / 4;
        if ((lane % 4) == 0) {
            float l = Ls[sgid][r];
            Dg[sgid][r * 8 + r] = l > 0.0f ? 1.0f / l : 0.0f;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 dgf;
        simdgroup_load(dgf, Dg[sgid], 8);
        if (interior) {
            for (uint d = 0; d < 8; ++d) {
                simdgroup_multiply(of[d], dgf, of[d]);
                simdgroup_store(of[d], Ss[sgid], 8);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                for (uint e = lane; e < 64; e += 32) {
                    uint r2 = e / 8, cc = e % 8;
                    y[((size_t)(row0 + r2) * H + h) * D + d * 8 + cc] = (half)Ss[sgid][e];
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        } else {
            threadgroup half* Oe = KVs;
            for (uint d = 0; d < 8; ++d) {
                simdgroup_multiply(of[d], dgf, of[d]);
                simdgroup_store(of[d], Ss[sgid], 8);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                for (uint e = lane; e < 64; e += 32) {
                    uint r2 = e / 8, cc = e % 8;
                    Oe[(size_t)sgid * 8 * D + r2 * D + d * 8 + cc] = (half)Ss[sgid][e];
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint x = tid; x < BQ * D; x += NT) {
                uint r2 = x / D, d = x % D;
                uint t = q0 + r2;
                if (t < T) y[((size_t)t * H + h) * D + d] = Oe[x];
            }
        }
    }
}

kernel void attn_maskfill_f16(device const half* mask [[buffer(0)]], device half* out [[buffer(1)]],
                              constant uint& ts [[buffer(2)]],
                              uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x, h = gid.y;
    if (i >= ts) return;
    out[(size_t)h*ts + i] = (mask[i] == (half)0.0f) ? (half)(-30000.0f) : (half)0.0f;
}

kernel void gemm_batch_f32a(device const float* a [[buffer(0)]], device const half* b [[buffer(1)]],
                            device uchar* y [[buffer(2)]],
                            constant uint& M [[buffer(3)]], constant uint& K [[buffer(4)]],
                            constant uint& N [[buffer(5)]], constant uint& f32out [[buffer(6)]],
                            uint2 gid [[thread_position_in_grid]]) {
    uint mn = gid.x, bidx = gid.y;
    if (mn >= M*N) return;
    uint m = mn / N, nn = mn % N;
    device const float* ar = a + (size_t)bidx*M*K + (size_t)m*K;
    device const half* br = b + (size_t)bidx*K*N + nn;
    float acc = 0;
    for (uint k = 0; k < K; ++k) acc = fma(ar[k], (float)br[(size_t)k*N], acc);
    size_t oi = (size_t)bidx*M*N + mn;
    if (f32out != 0u) ((device float*)y)[oi] = acc;
    else ((device half*)y)[oi] = (half)acc;
}

kernel void gemm_batch_f16(device const half* a [[buffer(0)]], device const half* b [[buffer(1)]],
                           device uchar* y [[buffer(2)]],
                           constant uint& M [[buffer(3)]], constant uint& K [[buffer(4)]],
                           constant uint& N [[buffer(5)]], constant uint& f32out [[buffer(6)]],
                           uint2 gid [[thread_position_in_grid]]) {
    uint mn = gid.x, bidx = gid.y;
    if (mn >= M*N) return;
    uint m = mn / N, nn = mn % N;
    device const half* ar = a + (size_t)bidx*M*K + (size_t)m*K;
    device const half* br = b + (size_t)bidx*K*N + nn;
    float acc = 0;
    for (uint k = 0; k < K; ++k) acc = fma((float)ar[k], (float)br[(size_t)k*N], acc);
    size_t oi = (size_t)bidx*M*N + mn;
    if (f32out != 0u) ((device float*)y)[oi] = acc;
    else ((device half*)y)[oi] = (half)acc;
}

kernel void conv1d_dw_f16(device const half* x [[buffer(0)]], device const half* w [[buffer(1)]],
                          device half* y [[buffer(2)]],
                          constant uint& L [[buffer(3)]], constant uint& Lout [[buffer(4)]],
                          constant uint& Kk [[buffer(5)]], constant uint& stride [[buffer(6)]],
                          uint2 gid [[thread_position_in_grid]]) {
    uint l = gid.x, c = gid.y;
    if (l >= Lout) return;
    device const half* xc = x + (size_t)c*L + (size_t)l*stride;
    device const half* wc = w + (size_t)c*Kk;
    float acc = 0;
    for (uint k = 0; k < Kk; ++k) acc = fma((float)xc[k], (float)wc[k], acc);
    y[(size_t)c*Lout + l] = (half)acc;
}

kernel void reduce_axis_f16(device const half* in [[buffer(0)]], device half* y [[buffer(1)]],
                            constant uint& axis_size [[buffer(2)]], constant uint& inner [[buffer(3)]],
                            constant uint& n [[buffer(4)]], constant int& op [[buffer(5)]],
                            uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    uint outer = i / inner, in_i = i % inner;
    size_t base = (size_t)outer * axis_size * inner + in_i;
    float r;
    if (op <= 2) {
        float sum = 0;
        for (uint a = 0; a < axis_size; ++a) sum += (float)in[base + (size_t)a*inner];
        float mean = sum / (float)max(axis_size, 1u);
        if (op == 0) r = sum;
        else if (op == 1) r = mean;
        else {
            float var = 0;
            for (uint a = 0; a < axis_size; ++a) { float d = (float)in[base + (size_t)a*inner] - mean; var += d*d; }
            r = var / (float)max(axis_size, 1u);
        }
    } else {
        r = axis_size ? (float)in[base] : 0.0f;
        for (uint a = 1; a < axis_size; ++a) {
            float v = (float)in[base + (size_t)a*inner];
            r = (op == 3) ? min(r, v) : max(r, v);
        }
    }
    y[i] = (half)r;
}

kernel void reduce_axis_f32(device const float* in [[buffer(0)]], device float* y [[buffer(1)]],
                            constant uint& axis_size [[buffer(2)]], constant uint& inner [[buffer(3)]],
                            constant uint& n [[buffer(4)]], constant int& op [[buffer(5)]],
                            uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    uint outer = i / inner, in_i = i % inner;
    size_t base = (size_t)outer * axis_size * inner + in_i;
    float r;
    if (op <= 2) {
        float sum = 0;
        for (uint a = 0; a < axis_size; ++a) sum += in[base + (size_t)a*inner];
        float mean = sum / (float)max(axis_size, 1u);
        if (op == 0) r = sum;
        else if (op == 1) r = mean;
        else {
            float var = 0;
            for (uint a = 0; a < axis_size; ++a) { float d = in[base + (size_t)a*inner] - mean; var += d*d; }
            r = var / (float)max(axis_size, 1u);
        }
    } else {
        r = axis_size ? in[base] : 0.0f;
        for (uint a = 1; a < axis_size; ++a) {
            float v = in[base + (size_t)a*inner];
            r = (op == 3) ? min(r, v) : max(r, v);
        }
    }
    y[i] = r;
}

kernel void cumsum_f16(device const half* in [[buffer(0)]], device half* y [[buffer(1)]],
                       constant uint& axis_size [[buffer(2)]], constant uint& inner [[buffer(3)]],
                       constant uint& n [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    uint outer = i / inner, in_i = i % inner;
    size_t base = (size_t)outer * axis_size * inner + in_i;
    float run = 0;
    for (uint a = 0; a < axis_size; ++a) {
        run += (float)in[base + (size_t)a*inner];
        y[base + (size_t)a*inner] = (half)run;
    }
}

kernel void cumsum_f32(device const float* in [[buffer(0)]], device float* y [[buffer(1)]],
                       constant uint& axis_size [[buffer(2)]], constant uint& inner [[buffer(3)]],
                       constant uint& n [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    uint outer = i / inner, in_i = i % inner;
    size_t base = (size_t)outer * axis_size * inner + in_i;
    float run = 0;
    for (uint a = 0; a < axis_size; ++a) {
        run += in[base + (size_t)a*inner];
        y[base + (size_t)a*inner] = run;
    }
}

kernel void concat2_f16(device const half* a [[buffer(0)]], device const half* b [[buffer(1)]],
                        device half* y [[buffer(2)]], constant uint& a_axis [[buffer(3)]],
                        constant uint& b_axis [[buffer(4)]], constant uint& inner [[buffer(5)]],
                        constant uint& n [[buffer(6)]], constant uint& a_outer [[buffer(7)]],
                        constant uint& b_outer [[buffer(8)]], uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    uint in_i = i % inner, rest = i / inner;
    uint ab = a_axis + b_axis;
    uint ax = rest % ab, outer = rest / ab;
    y[i] = (ax < a_axis) ? a[((size_t)(outer % a_outer)*a_axis + ax)*inner + in_i]
                         : b[((size_t)(outer % b_outer)*b_axis + (ax - a_axis))*inner + in_i];
}

kernel void gather_f32idx_f16(device const half* table [[buffer(0)]], device const float* rows [[buffer(1)]],
                              device half* out [[buffer(2)]], constant uint& D [[buffer(3)]],
                              constant uint& n [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    uint row = i / D, d = i % D;
    out[i] = table[(size_t)(uint)rows[row]*D + d];
}

kernel void rope_full_f16(device const half* x [[buffer(0)]], device half* y [[buffer(1)]],
                          constant uint& S [[buffer(2)]], constant uint& H [[buffer(3)]],
                          constant uint& D [[buffer(4)]], constant uint& rot [[buffer(5)]],
                          constant uint& pos0 [[buffer(6)]], constant float& theta [[buffer(7)]],
                          constant uint& gptj [[buffer(8)]],
                          uint2 gid [[thread_position_in_grid]]) {
    uint idx = gid.x, tok = gid.y;
    uint hr = rot / 2;
    if (idx >= hr) {
        uint d = rot + (idx - hr);
        if (d >= D) return;
        size_t off = (size_t)tok * D + d;
        y[off] = x[off];
        return;
    }
    uint seq = (tok / H) % S;
    float freq = 1.0f / precise::pow(theta, (2.0f * idx) / (float)rot);
    float angle = (float)(pos0 + seq) * freq;
    float c = (float)(half)precise::cos(angle);
    float sn = (float)(half)precise::sin(angle);
    uint d0 = gptj ? 2u * idx : idx;
    uint d1 = gptj ? 2u * idx + 1u : idx + hr;
    size_t off0 = (size_t)tok * D + d0, off1 = (size_t)tok * D + d1;
    float v0 = (float)x[off0], v1 = (float)x[off1];
    y[off0] = (half)(v0*c - v1*sn);
    y[off1] = (half)(v1*c + v0*sn);
}

kernel void maxpool1d_f16(device const half* x [[buffer(0)]], device half* y [[buffer(1)]],
                          constant uint& L [[buffer(2)]], constant uint& Lout [[buffer(3)]],
                          constant uint& K [[buffer(4)]], constant uint& stride [[buffer(5)]],
                          uint2 gid [[thread_position_in_grid]]) {
    uint lo = gid.x, nc = gid.y;
    if (lo >= Lout) return;
    size_t base = (size_t)nc * L + (size_t)lo * stride;
    float r = -INFINITY;
    for (uint k = 0; k < K && lo * stride + k < L; ++k) r = max(r, (float)x[base + k]);
    y[(size_t)nc * Lout + lo] = (half)r;
}

kernel void bilinear_f16(device const half* in [[buffer(0)]], device half* y [[buffer(1)]],
                         constant uint& sh [[buffer(2)]], constant uint& sw [[buffer(3)]],
                         constant uint& dh [[buffer(4)]], constant uint& dw [[buffer(5)]],
                         constant uint& E [[buffer(6)]], constant uint& align [[buffer(7)]],
                         uint2 gid [[thread_position_in_grid]]) {
    uint e = gid.x, pix = gid.y;
    if (e >= E || pix >= dh * dw) return;
    uint dy = pix / dw, dx = pix % dw;
    float syf, sxf;
    if (align) {
        float sc_h = (sh > 1 && dh > 1) ? (float)(sh - 1) / (float)(dh - 1) : 0.0f;
        float sc_w = (sw > 1 && dw > 1) ? (float)(sw - 1) / (float)(dw - 1) : 0.0f;
        syf = dy * sc_h; sxf = dx * sc_w;
    } else {
        float sc_h = dh ? (float)sh / (float)dh : 0.0f;
        float sc_w = dw ? (float)sw / (float)dw : 0.0f;
        syf = clamp((dy + 0.5f) * sc_h - 0.5f, 0.0f, (float)sh - 1.0f);
        sxf = clamp((dx + 0.5f) * sc_w - 0.5f, 0.0f, (float)sw - 1.0f);
    }
    int y0 = (int)floor(syf), x0 = (int)floor(sxf);
    int y1 = min(y0 + 1, (int)sh - 1), x1 = min(x0 + 1, (int)sw - 1);
    float fy = syf - y0, fx = sxf - x0;
    float v00 = (float)in[((size_t)y0*sw + x0)*E + e], v01 = (float)in[((size_t)y0*sw + x1)*E + e];
    float v10 = (float)in[((size_t)y1*sw + x0)*E + e], v11 = (float)in[((size_t)y1*sw + x1)*E + e];
    y[(size_t)pix*E + e] = (half)(v00*(1-fx)*(1-fy) + v01*fx*(1-fy) + v10*(1-fx)*fy + v11*fx*fy);
}

kernel void conv1d_gen_f16(device const half* x [[buffer(0)]], device const half* w [[buffer(1)]],
                           device const half* bias [[buffer(2)]], device half* y [[buffer(3)]],
                           constant uint& Cin [[buffer(4)]], constant uint& L [[buffer(5)]],
                           constant uint& Cout [[buffer(6)]], constant uint& Lout [[buffer(7)]],
                           constant uint& K [[buffer(8)]], constant uint& stride [[buffer(9)]],
                           constant uint& has_bias [[buffer(10)]], constant uint& w_ck_co [[buffer(11)]],
                           uint3 gid [[thread_position_in_grid]]) {
    uint lo = gid.x, co = gid.y, n = gid.z;
    if (lo >= Lout || co >= Cout) return;
    float acc = has_bias ? (float)bias[co] : 0.0f;
    for (uint ci = 0; ci < Cin; ++ci) {
        device const half* xr = x + ((size_t)n*Cin + ci)*L + (size_t)lo*stride;
        for (uint k = 0; k < K; ++k) {
            float wv = w_ck_co ? (float)w[((size_t)ci*K + k)*Cout + co]
                               : (float)w[((size_t)co*Cin + ci)*K + k];
            acc = fma((float)xr[k], wv, acc);
        }
    }
    y[((size_t)n*Cout + co)*Lout + lo] = (half)acc;
}

kernel void conv1d_nlc_dw_f16(device const half* x [[buffer(0)]], device const half* w [[buffer(1)]],
                              device const half* bias [[buffer(2)]], device half* y [[buffer(3)]],
                              constant uint& L [[buffer(4)]], constant uint& C [[buffer(5)]],
                              constant uint& K [[buffer(6)]], constant uint& dil [[buffer(7)]],
                              constant uint& pad [[buffer(8)]], constant uint& has_bias [[buffer(9)]],
                              uint3 gid [[thread_position_in_grid]]) {
    uint c = gid.x, l = gid.y, n = gid.z;
    if (c >= C || l >= L) return;
    float acc = has_bias ? (float)bias[c] : 0.0f;
    for (uint k = 0; k < K; ++k) {
        long p = (long)l - (long)pad + (long)k * dil;
        if (p >= 0 && p < (long)L)
            acc = fma((float)x[((size_t)n*L + p)*C + c], (float)w[(size_t)c*K + k], acc);
    }
    y[((size_t)n*L + l)*C + c] = (half)acc;
}

kernel void conv2d_f16(device const half* x [[buffer(0)]], device const half* w [[buffer(1)]],
                       device const half* bias [[buffer(2)]], device half* y [[buffer(3)]],
                       constant uint& Cin [[buffer(4)]], constant uint& H [[buffer(5)]],
                       constant uint& W [[buffer(6)]], constant uint& Cout [[buffer(7)]],
                       constant uint& Ho [[buffer(8)]], constant uint& Wo [[buffer(9)]],
                       constant uint& K [[buffer(10)]], constant uint& stride [[buffer(11)]],
                       constant uint& pad [[buffer(12)]], constant uint& dw [[buffer(13)]],
                       constant uint& has_bias [[buffer(14)]],
                       uint3 gid [[thread_position_in_grid]]) {
    uint wo0 = gid.x * 4, ho = gid.y, nco = gid.z;
    if (wo0 >= Wo || ho >= Ho) return;
    uint n = nco / Cout, co = nco % Cout;
    float acc[4];
    float bv = has_bias ? (float)bias[co] : 0.0f;
    for (uint q = 0; q < 4; ++q) acc[q] = bv;
    uint ci0 = dw ? co : 0, ci1 = dw ? co + 1 : Cin;
    for (uint ci = ci0; ci < ci1; ++ci) {
        for (uint kh = 0; kh < K; ++kh) {
            long h = (long)ho * stride - (long)pad + kh;
            if (h < 0 || h >= (long)H) continue;
            device const half* xr = x + (((size_t)n*Cin + ci)*H + h)*W;
            for (uint kw = 0; kw < K; ++kw) {
                float wv = dw ? (float)w[((size_t)co*K + kh)*K + kw]
                              : (float)w[(((size_t)co*Cin + ci)*K + kh)*K + kw];
                for (uint q = 0; q < 4; ++q) {
                    long ww = (long)(wo0 + q) * stride - (long)pad + kw;
                    if (ww >= 0 && ww < (long)W && wo0 + q < Wo)
                        acc[q] = fma((float)xr[ww], wv, acc[q]);
                }
            }
        }
    }
    for (uint q = 0; q < 4 && wo0 + q < Wo; ++q)
        y[(((size_t)n*Cout + co)*Ho + ho)*Wo + wo0 + q] = (half)acc[q];
}

kernel void batchnorm_f16(device const half* x [[buffer(0)]], device const half* w [[buffer(1)]],
                          device const half* b [[buffer(2)]], device const half* rm [[buffer(3)]],
                          device const half* rv [[buffer(4)]], device half* y [[buffer(5)]],
                          constant uint& C [[buffer(6)]], constant uint& inner [[buffer(7)]],
                          constant float& eps [[buffer(8)]], constant uint& n [[buffer(9)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    uint c = (i / inner) % C;
    float inv = 1.0f / precise::sqrt((float)rv[c] + eps);
    y[i] = (half)(((float)x[i] - (float)rm[c]) * inv * (float)w[c] + (float)b[c]);
}

kernel void groupnorm_f16(device const half* x [[buffer(0)]], device const half* w [[buffer(1)]],
                          device const half* b [[buffer(2)]], device half* y [[buffer(3)]],
                          constant uint& cpg [[buffer(4)]], constant uint& S [[buffer(5)]],
                          constant uint& C [[buffer(6)]], constant float& eps [[buffer(7)]],
                          uint2 tgp [[threadgroup_position_in_grid]],
                          uint2 tp [[thread_position_in_threadgroup]], uint2 ntp [[threads_per_threadgroup]],
                          threadgroup float* red [[threadgroup(0)]]) {
    uint g = tgp.x, n = tgp.y, t = tp.x, nt = ntp.x;
    size_t base = ((size_t)n*C + (size_t)g*cpg) * S;
    uint count = cpg * S;
    float sum = 0, sq = 0;
    for (uint i = t; i < count; i += nt) { float v = (float)x[base + i]; sum += v; sq += v*v; }
    red[t] = sum; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s2 = nt/2; s2 > 0; s2 >>= 1) { if (t < s2) red[t] += red[t+s2]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    sum = red[0]; threadgroup_barrier(mem_flags::mem_threadgroup);
    red[t] = sq; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s2 = nt/2; s2 > 0; s2 >>= 1) { if (t < s2) red[t] += red[t+s2]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    sq = red[0];
    float mean = sum / count;
    float inv = 1.0f / precise::sqrt(sq / count - mean*mean + eps);
    for (uint i = t; i < count; i += nt) {
        uint c = g*cpg + i / S;
        y[base + i] = (half)(((float)x[base + i] - mean) * inv * (float)w[c] + (float)b[c]);
    }
}

kernel void bias_add_rows_f16(device half* y [[buffer(0)]], device const half* bias [[buffer(1)]],
                              constant uint& C [[buffer(2)]], constant uint& n [[buffer(3)]],
                              uint i [[thread_position_in_grid]]) {
    if (i < n) y[i] = (half)((float)y[i] + (float)bias[i % C]);
}

struct EwStep { int kind; int code; float p0; float p1; };

kernel void elemwise_chain_f16(device const uchar* in [[buffer(0)]], device uchar* y [[buffer(1)]],
                               constant EwStep* steps [[buffer(2)]], constant uint& nsteps [[buffer(3)]],
                               device const uchar* s0 [[buffer(4)]], device const uchar* s1 [[buffer(5)]],
                               device const uchar* s2 [[buffer(6)]], constant uint& n [[buffer(7)]],
                               constant uint& flags [[buffer(8)]],
                               constant uint& inner [[buffer(9)]],
                               uint gid [[thread_position_in_grid]]) {
    uint i0 = gid * 4;
    if (i0 >= n) return;
    uint cnt = min(4u, n - i0);
    float xv[4];
    bool in32 = (flags & 1u) != 0;
    for (uint q = 0; q < cnt; ++q)
        xv[q] = in32 ? ((device const float*)in)[i0+q] : (float)((device const half*)in)[i0+q];
    bool prec32 = in32;
    for (uint s = 0; s < nsteps; ++s) {
        EwStep st = steps[s];
        if (st.kind == 4) { prec32 = st.code != 0; continue; }
        if (st.kind == 2) {
            uint slot = (st.code >> 6) & 3u;
            int op = st.code & 15;
            bool rhs = (st.code & 16) != 0;
            bool sf32 = (st.code & 32) != 0;
            device const uchar* sp = slot == 0 ? s0 : slot == 1 ? s1 : s2;
            uint bmode = (st.code >> 8) & 3u;
            for (uint q = 0; q < cnt; ++q) {
                uint si = i0 + q;
                if (bmode == 1u) si = si / inner;
                else if (bmode == 2u) si = si % inner;
                float o = sf32 ? ((device const float*)sp)[si] : (float)((device const half*)sp)[si];
                float a = rhs ? o : xv[q], b = rhs ? xv[q] : o;
                float r;
                switch (op) { case 2: r=a-b; break; case 3: r=a*b; break; case 4: r=a/b; break;
                              case 5: r=(a!=b)?1.0f:0.0f; break; default: r=a+b; }
                if (op == 1) r = clamp(r, -65500.0f, 65500.0f);
                xv[q] = r;
            }
        } else if (st.kind == 0) {
            for (uint q = 0; q < cnt; ++q) {
                float x = xv[q];
                if (st.code == 0) x = gelu_tanh(x);
                else if (st.code == 1) x = precise::tanh(x);
                else if (st.code == 2) x = x/(1.0f+precise::exp(-x));
                else if (st.code == 4) x = 0.5f*x*(1.0f+erf_approx(x*0.70710678f));
                else if (st.code == 5) x = 1.0f/(1.0f+precise::exp(-x));
                else x = max(x, 0.0f);
                xv[q] = x;
            }
        } else if (st.kind == 1) {
            for (uint q = 0; q < cnt; ++q) xv[q] = scalar_apply(xv[q], st.p0, st.code);
        } else {
            for (uint q = 0; q < cnt; ++q) xv[q] = clamp(xv[q], st.p0, st.p1);
        }
        if (!prec32) for (uint q = 0; q < cnt; ++q) xv[q] = (float)(half)xv[q];
    }
    if ((flags & 2u) != 0) { for (uint q = 0; q < cnt; ++q) ((device float*)y)[i0+q] = xv[q]; }
    else { for (uint q = 0; q < cnt; ++q) ((device half*)y)[i0+q] = (half)xv[q]; }
}

kernel void bcast_binary_rows_f16(device const half* a [[buffer(0)]], device const half* b [[buffer(1)]],
    device half* out [[buffer(2)]], constant uint* oshape [[buffer(3)]],
    constant uint* astride [[buffer(4)]], constant uint* bstride [[buffer(5)]],
    constant uint& ndim [[buffer(6)]], constant uint& rows [[buffer(7)]], constant int& op [[buffer(8)]],
    constant uint& inner [[buffer(9)]], constant uint& ainner [[buffer(10)]], constant uint& binner [[buffer(11)]],
    uint2 gid [[thread_position_in_grid]]) {
    uint v = gid.x, row = gid.y;
    if (v >= inner || row >= rows) return;
    uint rem = row, ai = 0, bi = 0;
    for (int d = int(ndim) - 1; d >= 0; --d) { uint c = rem % oshape[d]; rem /= oshape[d]; ai += c * astride[d]; bi += c * bstride[d]; }
    float av = (float)a[ai + v * ainner], bv = (float)b[bi + v * binner], r;
    switch(op){ case 2:r=av-bv;break; case 3:r=av*bv;break; case 4:r=av/bv;break;
                case 5:r=(av!=bv)?1.0f:0.0f;break; default:r=av+bv; }
    if (op==1) r=clamp(r,-65500.0f,65500.0f);
    out[(size_t)row * inner + v] = (half)r;
}

kernel void bcast_binary_f16(device const half* a [[buffer(0)]], device const half* b [[buffer(1)]],
    device half* out [[buffer(2)]], constant uint* oshape [[buffer(3)]],
    constant uint* astride [[buffer(4)]], constant uint* bstride [[buffer(5)]],
    constant uint& ndim [[buffer(6)]], constant uint& total [[buffer(7)]], constant int& op [[buffer(8)]],
    uint i [[thread_position_in_grid]]) {
    if (i>=total) return;
    uint rem=i, ai=0, bi=0;
    for (int d=int(ndim)-1; d>=0; --d){ uint c=rem%oshape[d]; rem/=oshape[d]; ai+=c*astride[d]; bi+=c*bstride[d]; }
    float av=(float)a[ai], bv=(float)b[bi], r;
    switch(op){ case 2:r=av-bv;break; case 3:r=av*bv;break; case 4:r=av/bv;break;
                case 5:r=(av!=bv)?1.0f:0.0f;break; default:r=av+bv; }
    if (op==1) r=clamp(r,-65500.0f,65500.0f);
    out[i]=(half)r;
}

kernel void attn_decode_i8(
    device const half*  q     [[buffer(0)]],
    device const half*  knew  [[buffer(1)]],
    device const half*  vnew  [[buffer(2)]],
    device const char*  kc    [[buffer(3)]],
    device const char*  vc    [[buffer(4)]],
    device const float* ks    [[buffer(5)]],
    device const float* vs    [[buffer(6)]],
    device       half*  out   [[buffer(7)]],
    constant uint& num_q_heads [[buffer(8)]],  constant uint& num_kv_heads [[buffer(9)]],
    constant uint& head_dim    [[buffer(10)]], constant uint& v_hdim      [[buffer(11)]],
    constant uint& history_len [[buffer(12)]], constant float& scale      [[buffer(13)]],
    constant uint& kv_start    [[buffer(14)]], constant uint& kv_end      [[buffer(15)]],
    device       float* part_o [[buffer(16)]], device float* part_ml [[buffer(17)]],
    constant uint& nwg         [[buffer(18)]],
    uint tg [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
    uint T [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], threadgroup float* smem [[threadgroup(0)]])
{
    uint h = tg / nwg, w = tg % nwg;
    uint kvh = h / (num_q_heads / num_kv_heads);
    uint ngK = (head_dim + 31u)/32u, ngV = (v_hdim + 31u)/32u;
    uint nsg = T / 32u;
    device const half* qh = q + (size_t)h*head_dim;

    float qreg[16];
    { uint i=0; for (uint d=lane; d<head_dim; d+=32) qreg[i++] = (float)qh[d]; }

    float o_acc[16];
    for (uint i=0;i<16;++i) o_acc[i] = 0.0f;
    float m_i = -INFINITY, l_i = 0.0f;

    uint gsg = w*nsg + sg, stride = nwg*nsg;
    for (uint k = kv_start + gsg; k < kv_end; k += stride) {
        float partial = 0.0f;
        if (k < history_len) {
            device const char*  kk  = kc + ((size_t)k*num_kv_heads + kvh)*head_dim;
            device const float* kss = ks + ((size_t)k*num_kv_heads + kvh)*ngK;
            uint i=0; for (uint d=lane; d<head_dim; d+=32) { partial += qreg[i] * ((float)kk[d]*kss[d/32]); ++i; }
        } else {
            device const half*  kk = knew + ((size_t)(k-history_len)*num_kv_heads + kvh)*head_dim;
            uint i=0; for (uint d=lane; d<head_dim; d+=32) { partial += qreg[i] * (float)kk[d]; ++i; }
        }
        float s = simd_sum(partial) * scale;
        float m_new = max(m_i, s);
        float resc  = exp(m_i - m_new);
        float p     = exp(s - m_new);
        l_i = l_i * resc + p;
        if (k < history_len) {
            device const char*  vvv = vc + ((size_t)k*num_kv_heads+kvh)*v_hdim;
            device const float* vss = vs + ((size_t)k*num_kv_heads+kvh)*ngV;
            uint i=0; for (uint d=lane; d<v_hdim; d+=32) { o_acc[i] = o_acc[i]*resc + p*((float)vvv[d]*vss[d/32]); ++i; }
        } else {
            device const half*  vvv = vnew + ((size_t)(k-history_len)*num_kv_heads+kvh)*v_hdim;
            uint i=0; for (uint d=lane; d<v_hdim; d+=32) { o_acc[i] = o_acc[i]*resc + p*(float)vvv[d]; ++i; }
        }
        m_i = m_new;
    }

    threadgroup float* Otg = smem;
    threadgroup float* mtg = Otg + (size_t)nsg*v_hdim;
    threadgroup float* ltg = mtg + nsg;
    if (lane == 0) { mtg[sg] = m_i; ltg[sg] = l_i; }
    { uint i=0; for (uint d=lane; d<v_hdim; d+=32) { Otg[(size_t)sg*v_hdim + d] = o_acc[i]; ++i; } }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float gm = -INFINITY;
    for (uint i=0;i<nsg;++i) gm = max(gm, mtg[i]);
    float gl = 0.0f;
    for (uint i=0;i<nsg;++i) gl += ltg[i] * exp(mtg[i] - gm);

    if (nwg == 1u) {
        float inv = gl > 0.0f ? 1.0f/gl : 0.0f;
        for (uint d=t; d<v_hdim; d+=T) {
            float acc = 0.0f;
            for (uint i=0;i<nsg;++i) acc += Otg[(size_t)i*v_hdim + d] * exp(mtg[i] - gm);
            out[(size_t)h*v_hdim + d] = (half)(acc * inv);
        }
    } else {
        uint slot = h*nwg + w;
        if (t == 0) { part_ml[(size_t)slot*2] = gm; part_ml[(size_t)slot*2 + 1] = gl; }
        for (uint d=t; d<v_hdim; d+=T) {
            float acc = 0.0f;
            for (uint i=0;i<nsg;++i) acc += Otg[(size_t)i*v_hdim + d] * exp(mtg[i] - gm);
            part_o[(size_t)slot*v_hdim + d] = acc;
        }
    }
}

struct AFU { uint nqh, nkvh, hd, vhd, hist, kv_start, kv_end, nwg, slot, has_new; float eps, scale; };
kernel void attn_decode_fused_i8(
    device const half*  q     [[buffer(0)]],
    device const half*  kraw  [[buffer(1)]],
    device const half*  vraw  [[buffer(2)]],
    device       char*  kc    [[buffer(3)]],
    device       char*  vc    [[buffer(4)]],
    device       float* ks    [[buffer(5)]],
    device       float* vs    [[buffer(6)]],
    device       half*  out   [[buffer(7)]],
    device const half*  qw    [[buffer(8)]],
    device const half*  kw    [[buffer(9)]],
    device const half*  vw    [[buffer(10)]],
    device const half*  cs    [[buffer(11)]],
    device const half*  sn    [[buffer(12)]],
    device       float* part_o [[buffer(13)]],
    device       float* part_ml[[buffer(14)]],
    constant AFU& U            [[buffer(15)]],
    uint tg [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
    uint T [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    threadgroup float* smem [[threadgroup(0)]],
    threadgroup half*  hnew [[threadgroup(1)]])
{
    const uint hd = U.hd, vhd = U.vhd, nwg = U.nwg;
    const uint h = tg / nwg, w = tg % nwg;
    const uint nsg = T / 32u;
    const uint ngK = (hd + 31u)/32u, ngV = (vhd + 31u)/32u;
    threadgroup float* Otg = smem;
    threadgroup float* mtg = Otg + (size_t)nsg*vhd;
    threadgroup float* ltg = mtg + nsg;
    threadgroup float* red = ltg + nsg;
    threadgroup half*  qr_s = hnew;
    threadgroup half*  kr_s = hnew + hd;
    threadgroup half*  vn_s = hnew + 2u*hd;

    const uint hh = hd/2u;
    {
        device const half* xh = q + (size_t)h*hd;
        float partial=0;
        for (uint i=t;i<hd;i+=T){ float v=(float)xh[i]; partial+=v*v; }
        red[t]=partial; threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s2=T/2; s2>0; s2>>=1){ if (t<s2) red[t]+=red[t+s2]; threadgroup_barrier(mem_flags::mem_threadgroup); }
        float inv = 1.0f/sqrt(red[0]/(float)hd + U.eps);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i=t;i<hd;i+=T){
            uint j = (i<hh) ? i+hh : i-hh;
            float xi = (float)xh[i]*inv*(float)qw[i];
            float xj = (float)xh[j]*inv*(float)qw[j];
            float rot = (i<hh) ? -xj : xj;
            qr_s[i] = (half)(xi*(float)cs[i] + rot*(float)sn[i]);
        }
    }
    if (U.has_new != 0u) {
        {
            float partial=0;
            for (uint i=t;i<hd;i+=T){ float v=(float)kraw[i]; partial+=v*v; }
            red[t]=partial; threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s2=T/2; s2>0; s2>>=1){ if (t<s2) red[t]+=red[t+s2]; threadgroup_barrier(mem_flags::mem_threadgroup); }
            float inv = 1.0f/sqrt(red[0]/(float)hd + U.eps);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i=t;i<hd;i+=T){
                uint j = (i<hh) ? i+hh : i-hh;
                float xi = (float)kraw[i]*inv*(float)kw[i];
                float xj = (float)kraw[j]*inv*(float)kw[j];
                float rot = (i<hh) ? -xj : xj;
                kr_s[i] = (half)(xi*(float)cs[i] + rot*(float)sn[i]);
            }
        }
        {
            float partial=0;
            for (uint i=t;i<vhd;i+=T){ float v=(float)vraw[i]; partial+=v*v; }
            red[t]=partial; threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s2=T/2; s2>0; s2>>=1){ if (t<s2) red[t]+=red[t+s2]; threadgroup_barrier(mem_flags::mem_threadgroup); }
            float inv = 1.0f/sqrt(red[0]/(float)vhd + U.eps);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i=t;i<vhd;i+=T) vn_s[i] = (half)((float)vraw[i]*inv*(float)vw[i]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (U.has_new != 0u && tg == 0u) {
        for (uint g2=t; g2<ngK; g2+=T) {
            uint gstart=g2*32u, gcount=min(32u, hd-gstart);
            float maxabs=0;
            for (uint kk2=0;kk2<gcount;++kk2) maxabs=max(maxabs, fabs((float)kr_s[gstart+kk2]));
            float scale=maxabs/127.0f; if (scale<1e-10f) scale=1e-10f;
            float inv=1.0f/scale;
            device char* dst = kc + (size_t)U.slot*hd + gstart;
            for (uint kk2=0;kk2<gcount;++kk2){ float qv=clamp(rint((float)kr_s[gstart+kk2]*inv),-128.0f,127.0f); dst[kk2]=(char)qv; }
            ks[(size_t)U.slot*ngK + g2] = scale;
        }
        for (uint g2=t; g2<ngV; g2+=T) {
            uint gstart=g2*32u, gcount=min(32u, vhd-gstart);
            float maxabs=0;
            for (uint kk2=0;kk2<gcount;++kk2) maxabs=max(maxabs, fabs((float)vn_s[gstart+kk2]));
            float scale=maxabs/127.0f; if (scale<1e-10f) scale=1e-10f;
            float inv=1.0f/scale;
            device char* dst = vc + (size_t)U.slot*vhd + gstart;
            for (uint kk2=0;kk2<gcount;++kk2){ float qv=clamp(rint((float)vn_s[gstart+kk2]*inv),-128.0f,127.0f); dst[kk2]=(char)qv; }
            vs[(size_t)U.slot*ngV + g2] = scale;
        }
    }

    float qreg[16];
    { uint i=0; for (uint d=lane; d<hd; d+=32) qreg[i++] = (float)qr_s[d]; }
    float o_acc[16];
    for (uint i=0;i<16;++i) o_acc[i] = 0.0f;
    float m_i = -INFINITY, l_i = 0.0f;

    uint gsg = w*nsg + sg, stride = nwg*nsg;
    for (uint k2 = U.kv_start + gsg; k2 < U.kv_end; k2 += stride) {
        bool fresh_kv = (U.has_new != 0u) && (k2 == U.slot);
        float partial = 0.0f;
        if (fresh_kv) {
            uint i=0; for (uint d=lane; d<hd; d+=32) { partial += qreg[i] * (float)kr_s[d]; ++i; }
        } else {
            device const char*  kk  = kc + (size_t)k2*hd;
            device const float* kss = ks + (size_t)k2*ngK;
            uint i=0; for (uint d=lane; d<hd; d+=32) { partial += qreg[i] * ((float)kk[d]*kss[d/32]); ++i; }
        }
        float s2 = simd_sum(partial) * U.scale;
        float m_new = max(m_i, s2);
        float resc  = exp(m_i - m_new);
        float p     = exp(s2 - m_new);
        l_i = l_i * resc + p;
        if (fresh_kv) {
            uint i=0; for (uint d=lane; d<vhd; d+=32) { o_acc[i] = o_acc[i]*resc + p*(float)vn_s[d]; ++i; }
        } else {
            device const char*  vvv = vc + (size_t)k2*vhd;
            device const float* vss = vs + (size_t)k2*ngV;
            uint i=0; for (uint d=lane; d<vhd; d+=32) { o_acc[i] = o_acc[i]*resc + p*((float)vvv[d]*vss[d/32]); ++i; }
        }
        m_i = m_new;
    }

    if (lane == 0) { mtg[sg] = m_i; ltg[sg] = l_i; }
    { uint i=0; for (uint d=lane; d<vhd; d+=32) { Otg[(size_t)sg*vhd + d] = o_acc[i]; ++i; } }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float gm = -INFINITY;
    for (uint i=0;i<nsg;++i) gm = max(gm, mtg[i]);
    float gl = 0.0f;
    for (uint i=0;i<nsg;++i) gl += ltg[i] * exp(mtg[i] - gm);

    if (nwg == 1u) {
        float inv = gl > 0.0f ? 1.0f/gl : 0.0f;
        for (uint d=t; d<vhd; d+=T) {
            float acc = 0.0f;
            for (uint i=0;i<nsg;++i) acc += Otg[(size_t)i*vhd + d] * exp(mtg[i] - gm);
            out[(size_t)h*vhd + d] = (half)(acc * inv);
        }
    } else {
        uint slot2 = h*nwg + w;
        if (t == 0) { part_ml[(size_t)slot2*2] = gm; part_ml[(size_t)slot2*2 + 1] = gl; }
        for (uint d=t; d<vhd; d+=T) {
            float acc = 0.0f;
            for (uint i=0;i<nsg;++i) acc += Otg[(size_t)i*vhd + d] * exp(mtg[i] - gm);
            part_o[(size_t)slot2*vhd + d] = acc;
        }
    }
}

kernel void attn_decode_combine(
    device const float* part_o [[buffer(0)]],
    device const float* part_ml[[buffer(1)]],
    device       half*  out    [[buffer(2)]],
    constant uint& v_hdim [[buffer(3)]], constant uint& nwg [[buffer(4)]],
    uint h [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
    uint T [[threads_per_threadgroup]])
{
    float gm = -INFINITY;
    for (uint w=0;w<nwg;++w) gm = max(gm, part_ml[(size_t)(h*nwg+w)*2]);
    float gl = 0.0f;
    for (uint w=0;w<nwg;++w) gl += part_ml[(size_t)(h*nwg+w)*2 + 1] * exp(part_ml[(size_t)(h*nwg+w)*2] - gm);
    float inv = gl > 0.0f ? 1.0f/gl : 0.0f;
    for (uint d=t; d<v_hdim; d+=T) {
        float acc = 0.0f;
        for (uint w=0;w<nwg;++w) acc += part_o[(size_t)(h*nwg+w)*v_hdim + d] * exp(part_ml[(size_t)(h*nwg+w)*2] - gm);
        out[(size_t)h*v_hdim + d] = (half)(acc * inv);
    }
}

kernel void attn_prefill_i8(
    device const half*  q     [[buffer(0)]],
    device const half*  knew  [[buffer(1)]],
    device const half*  vnew  [[buffer(2)]],
    device const char*  kc    [[buffer(3)]],
    device const char*  vc    [[buffer(4)]],
    device const float* ks    [[buffer(5)]],
    device const float* vs    [[buffer(6)]],
    device       half*  out   [[buffer(7)]],
    constant uint& num_q_heads [[buffer(8)]],  constant uint& num_kv_heads [[buffer(9)]],
    constant uint& head_dim    [[buffer(10)]], constant uint& v_hdim      [[buffer(11)]],
    constant uint& history_len [[buffer(12)]], constant float& scale      [[buffer(13)]],
    constant uint& q_pos0      [[buffer(14)]], constant uint& new_len     [[buffer(15)]],
    constant uint& window      [[buffer(16)]], constant uint& is_causal   [[buffer(17)]],
    constant uint& maxsc       [[buffer(18)]], constant uint& sinkN       [[buffer(19)]],
    constant uint& ringR       [[buffer(20)]],
    uint flat [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
    uint T [[threads_per_threadgroup]], threadgroup float* sc [[threadgroup(0)]])
{
    uint h = flat % num_q_heads, m = flat / num_q_heads;
    uint kvh = h / (num_q_heads / num_kv_heads);
    uint ngK = (head_dim + 31u)/32u, ngV = (v_hdim + 31u)/32u;
    uint total_keys = history_len + new_len;
    uint pos_m = q_pos0 + m;
    uint kv_end = is_causal ? min(total_keys, pos_m + 1u) : total_keys;
    uint S = (ringR > 0u) ? sinkN : 0u;
    uint Wsr = S + ringR;
    uint rstart, nactive;
    if (ringR > 0u && kv_end > Wsr) { rstart = kv_end - ringR; nactive = Wsr; }
    else { rstart = S; nactive = kv_end; }
    device const half* qh = q + ((size_t)m*num_q_heads + h)*head_dim;
    threadgroup float* red = sc + maxsc;
    if (nactive == 0u) {
        for (uint d=t; d<v_hdim; d+=T) out[((size_t)m*num_q_heads + h)*v_hdim + d] = (half)0;
        return;
    }
    if (ringR > 0u) {
        float lmax = -INFINITY;
        for (uint j = t; j < nactive; j += T) {
            uint k = (j < S) ? j : rstart + (j - S);
            float dot = 0;
            if (k < history_len) {
                uint slot = (ringR > 0u && k >= Wsr) ? (S + ((k - S) % ringR)) : k;
                device const char* kk = kc + ((size_t)slot*num_kv_heads + kvh)*head_dim;
                device const float* kss = ks + ((size_t)slot*num_kv_heads + kvh)*ngK;
                for (uint d=0; d<head_dim; ++d) dot += (float)qh[d] * ((float)kk[d] * kss[d/32]);
            } else {
                device const half* kk = knew + ((size_t)(k-history_len)*num_kv_heads + kvh)*head_dim;
                for (uint d=0; d<head_dim; ++d) dot += (float)qh[d] * (float)kk[d];
            }
            float s = dot * scale; sc[j] = s; lmax = max(lmax, s);
        }
        red[t]=lmax; threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s=T/2;s>0;s>>=1){ if(t<s) red[t]=max(red[t],red[t+s]); threadgroup_barrier(mem_flags::mem_threadgroup); }
        float gmax = red[0]; threadgroup_barrier(mem_flags::mem_threadgroup);
        float lsum=0;
        for (uint j = t; j < nactive; j += T){ float e=exp(sc[j]-gmax); sc[j]=e; lsum+=e; }
        red[t]=lsum; threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s=T/2;s>0;s>>=1){ if(t<s) red[t]+=red[t+s]; threadgroup_barrier(mem_flags::mem_threadgroup); }
        float inv = red[0] > 0 ? 1.0f/red[0] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint d = t; d < v_hdim; d += T) {
            float acc=0;
            for (uint j=0; j<nactive; ++j) {
                uint k = (j < S) ? j : rstart + (j - S);
                float vv;
                if (k < history_len) {
                    uint slot = (k >= Wsr) ? (S + ((k - S) % ringR)) : k;
                    device const char* vvv = vc + ((size_t)slot*num_kv_heads+kvh)*v_hdim;
                    device const float* vss = vs + ((size_t)slot*num_kv_heads+kvh)*ngV;
                    vv = (float)vvv[d]*vss[d/32];
                } else {
                    device const half* vvv = vnew + ((size_t)(k-history_len)*num_kv_heads+kvh)*v_hdim;
                    vv = (float)vvv[d];
                }
                acc += sc[j]*vv;
            }
            out[((size_t)m*num_q_heads + h)*v_hdim + d] = (half)(acc*inv);
        }
    } else {
        uint TILE = maxsc;
        float rmax = -INFINITY, rsum = 0.0f;
        float acc[8];
        for (uint a=0;a<8;++a) acc[a]=0.0f;
        for (uint tile=0; tile<nactive; tile += TILE) {
            uint tcount = min(TILE, nactive - tile);
            for (uint jj=t; jj<tcount; jj+=T) {
                uint k = tile + jj;
                float dot = 0;
                if (k < history_len) {
                    device const char* kk = kc + ((size_t)k*num_kv_heads + kvh)*head_dim;
                    device const float* kss = ks + ((size_t)k*num_kv_heads + kvh)*ngK;
                    for (uint d=0; d<head_dim; ++d) dot += (float)qh[d] * ((float)kk[d] * kss[d/32]);
                } else {
                    device const half* kk = knew + ((size_t)(k-history_len)*num_kv_heads + kvh)*head_dim;
                    for (uint d=0; d<head_dim; ++d) dot += (float)qh[d] * (float)kk[d];
                }
                sc[jj] = dot * scale;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float lmax=-INFINITY;
            for (uint jj=t; jj<tcount; jj+=T) lmax=max(lmax, sc[jj]);
            red[t]=lmax; threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s=T/2;s>0;s>>=1){ if(t<s) red[t]=max(red[t],red[t+s]); threadgroup_barrier(mem_flags::mem_threadgroup); }
            float tmax=red[0]; threadgroup_barrier(mem_flags::mem_threadgroup);
            float rmax_new = max(rmax, tmax);
            float resc = exp(rmax - rmax_new);
            for (uint jj=t; jj<tcount; jj+=T) sc[jj]=exp(sc[jj]-rmax_new);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float lsum=0;
            for (uint jj=t; jj<tcount; jj+=T) lsum+=sc[jj];
            red[t]=lsum; threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s=T/2;s>0;s>>=1){ if(t<s) red[t]+=red[t+s]; threadgroup_barrier(mem_flags::mem_threadgroup); }
            float tsum=red[0]; threadgroup_barrier(mem_flags::mem_threadgroup);
            rsum = rsum*resc + tsum;
            uint ai=0;
            for (uint d=t; d<v_hdim; d+=T) {
                float a = acc[ai]*resc;
                for (uint jj=0; jj<tcount; ++jj) {
                    uint k = tile + jj;
                    float vv;
                    if (k < history_len) {
                        device const char* vvv = vc + ((size_t)k*num_kv_heads+kvh)*v_hdim;
                        device const float* vss = vs + ((size_t)k*num_kv_heads+kvh)*ngV;
                        vv = (float)vvv[d]*vss[d/32];
                    } else {
                        device const half* vvv = vnew + ((size_t)(k-history_len)*num_kv_heads+kvh)*v_hdim;
                        vv = (float)vvv[d];
                    }
                    a += sc[jj]*vv;
                }
                acc[ai]=a; ++ai;
            }
            rmax = rmax_new;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float inv = rsum > 0 ? 1.0f/rsum : 0.0f;
        uint ai=0;
        for (uint d=t; d<v_hdim; d+=T) { out[((size_t)m*num_q_heads + h)*v_hdim + d]=(half)(acc[ai]*inv); ++ai; }
    }
}

kernel void attn_prefill_mma2(
    device const half*  q     [[buffer(0)]],
    device const half*  knew  [[buffer(1)]],
    device const half*  vnew  [[buffer(2)]],
    device const char*  kc    [[buffer(3)]],
    device const char*  vc    [[buffer(4)]],
    device const float* ks    [[buffer(5)]],
    device const float* vs    [[buffer(6)]],
    device       half*  out   [[buffer(7)]],
    constant uint& num_q_heads [[buffer(8)]],  constant uint& num_kv_heads [[buffer(9)]],
    constant uint& head_dim    [[buffer(10)]], constant uint& v_hdim      [[buffer(11)]],
    constant uint& history_len [[buffer(12)]], constant float& scale      [[buffer(13)]],
    constant uint& q_pos0      [[buffer(14)]], constant uint& new_len     [[buffer(15)]],
    constant uint& Mtot        [[buffer(16)]],
    uint tgx [[threadgroup_position_in_grid]], uint tl [[thread_index_in_threadgroup]])
{
    const uint NSG=16u, LD=40u, BK=64u, QB=8u, HD=512u, VD=512u, NFH=32u;
    const uint LDK=72u, LDP=72u;
    threadgroup float poolA[4096];
    threadgroup half  poolB[4608];
    threadgroup float Rsc[NSG*QB*8u];
    threadgroup float mrun[8u*QB];
    threadgroup float lrun[8u*QB];
    threadgroup float resc8[8u*QB];
    threadgroup int   need_rescale[NSG];

    threadgroup half*  Ks = (threadgroup half*)poolA;
    threadgroup float* Ss = poolA;
    threadgroup half*  Vs = (threadgroup half*)poolA;
    threadgroup half*  Qs = poolB;
    threadgroup half*  Ps = poolB;

    uint sg = tl >> 5, lane = tl & 31u;
    uint h = sg >> 1, hf = sg & 1u, kvh = 0u;
    uint vd0 = hf*256u;
    uint m0 = tgx*QB;
    uint total_keys = history_len + new_len;

    uint pos_last = q_pos0 + m0 + (QB-1u);
    uint kv_end_tile = min(total_keys, pos_last + 1u);
    simdgroup_matrix<float,8,8> O[NFH];

    for (uint f=0; f<NFH; ++f) O[f] = make_filled_simdgroup_matrix<float,8,8>(0.f);
    if (hf==0u && lane < QB) { mrun[h*QB+lane] = -INFINITY; lrun[h*QB+lane]=0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k0=0; k0<kv_end_tile; k0+=BK) {
        uint kcount = min(BK, kv_end_tile - k0);
        simdgroup_matrix<float,8,8> C0=make_filled_simdgroup_matrix<float,8,8>(0.f);
        simdgroup_matrix<float,8,8> C1=C0,C2=C0,C3=C0,C4=C0,C5=C0,C6=C0,C7=C0;
        for (uint c=0; c<HD; c+=32u) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i=tl; i<32u*BK; i+=NSG*32u) {
                uint d=i/BK, k=i%BK;
                half val=(half)0;
                if (k<kcount) {
                    uint kk=k0+k;
                    if (kk<history_len) {
                        device const char*  kp=kc+((size_t)kk*num_kv_heads+kvh)*HD;
                        device const float* sp=ks+((size_t)kk*num_kv_heads+kvh)*(HD/32u);
                        val=(half)((float)kp[c+d]*sp[(c+d)/32u]);
                    } else {
                        device const half* kp=knew+((size_t)(kk-history_len)*num_kv_heads+kvh)*HD;
                        val=kp[c+d];
                    }
                }
                Ks[d*LDK + k]=val;
            }
            if (hf==0u) for (uint i=lane; i<QB*32u; i+=32u) {
                uint r=i>>5, d=i&31u;
                Qs[h*QB*LD + r*LD + d] = (m0+r<Mtot) ? q[((size_t)(m0+r)*num_q_heads + h)*HD + c + d] : (half)0;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (hf==0u) for (uint kk=0; kk<32u; kk+=8u) {
                simdgroup_matrix<half,8,8> A,B0,B1,B2,B3,B4,B5,B6,B7;
                simdgroup_load(A, &Qs[h*QB*LD + kk], LD);
                simdgroup_load(B0,&Ks[kk*LDK + 0u], LDK);
                simdgroup_load(B1,&Ks[kk*LDK + 8u], LDK);
                simdgroup_load(B2,&Ks[kk*LDK + 16u],LDK);
                simdgroup_load(B3,&Ks[kk*LDK + 24u],LDK);
                simdgroup_load(B4,&Ks[kk*LDK + 32u],LDK);
                simdgroup_load(B5,&Ks[kk*LDK + 40u],LDK);
                simdgroup_load(B6,&Ks[kk*LDK + 48u],LDK);
                simdgroup_load(B7,&Ks[kk*LDK + 56u],LDK);
                simdgroup_multiply_accumulate(C0,A,B0,C0);
                simdgroup_multiply_accumulate(C1,A,B1,C1);
                simdgroup_multiply_accumulate(C2,A,B2,C2);
                simdgroup_multiply_accumulate(C3,A,B3,C3);
                simdgroup_multiply_accumulate(C4,A,B4,C4);
                simdgroup_multiply_accumulate(C5,A,B5,C5);
                simdgroup_multiply_accumulate(C6,A,B6,C6);
                simdgroup_multiply_accumulate(C7,A,B7,C7);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (hf==0u) {
            simdgroup_store(C0,&Ss[h*QB*BK + 0u], BK);
            simdgroup_store(C1,&Ss[h*QB*BK + 8u], BK);
            simdgroup_store(C2,&Ss[h*QB*BK + 16u],BK);
            simdgroup_store(C3,&Ss[h*QB*BK + 24u],BK);
            simdgroup_store(C4,&Ss[h*QB*BK + 32u],BK);
            simdgroup_store(C5,&Ss[h*QB*BK + 40u],BK);
            simdgroup_store(C6,&Ss[h*QB*BK + 48u],BK);
            simdgroup_store(C7,&Ss[h*QB*BK + 56u],BK);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (hf==0u && lane < QB) {
            uint r=lane, qpos=q_pos0+m0+r;
            threadgroup float* srow=&Ss[h*QB*BK + r*BK];
            float tmax=-INFINITY;
            for (uint k=0;k<kcount;++k){
                uint kk=k0+k;
                float s=(kk<=qpos)? srow[k]*scale : -INFINITY;
                srow[k]=s; tmax=max(tmax,s);
            }
            float mo=mrun[h*QB+r], mnew=max(mo,tmax);
            float resc=exp(mo-mnew), tsum=0.0f;
            for (uint k=0;k<kcount;++k){
                float e=(srow[k]>-INFINITY)?exp(srow[k]-mnew):0.0f;
                srow[k]=e; tsum+=e;
            }
            lrun[h*QB+r]=lrun[h*QB+r]*resc+tsum;
            mrun[h*QB+r]=mnew; resc8[h*QB+r]=resc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane == 0u) {
            int nr = 0;
            for (uint r=0; r<QB; ++r) if (resc8[h*QB+r] != 1.0f) nr = 1;
            need_rescale[sg] = nr;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (need_rescale[sg]) {
            for (uint f=0; f<NFH; ++f) {
                simdgroup_store(O[f], &Rsc[sg*QB*8u], 8u);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                for (uint e=lane; e<QB*8u; e+=32u){ uint r=e>>3; Rsc[sg*QB*8u + e] *= resc8[h*QB+r]; }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                simdgroup_load(O[f], &Rsc[sg*QB*8u], 8u);
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        if (hf==0u) for (uint i=lane; i<QB*BK; i+=32u){ uint r=i/BK,k=i%BK; Ps[h*QB*LDP + r*LDP + k]=(k<kcount)?(half)Ss[h*QB*BK + r*BK + k]:(half)0; }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint hl = h*32u + lane;
        for (uint jc=0;jc<256u;jc+=32u){
            uint c = vd0 + jc;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i=hl;i<BK*32u;i+=8u*32u){
                uint k=i>>5, d=i&31u;
                half val=(half)0;
                if (k<kcount){
                    uint kk=k0+k;
                    if (kk<history_len){
                        device const char*  vp=vc+((size_t)kk*num_kv_heads+kvh)*VD;
                        device const float* sp=vs+((size_t)kk*num_kv_heads+kvh)*(VD/32u);
                        val=(half)((float)vp[c+d]*sp[(c+d)/32u]);
                    } else {
                        device const half* vp=vnew+((size_t)(kk-history_len)*num_kv_heads+kvh)*VD;
                        val=vp[c+d];
                    }
                }
                Vs[hf*BK*LD + k*LD + d]=val;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint fc=0; fc<32u; fc+=8u){
                uint f = jc/8u + fc/8u;
                for (uint kk=0; kk<BK; kk+=8u){
                    simdgroup_matrix<half,8,8> P,V0;
                    simdgroup_load(P,  &Ps[h*QB*LDP + kk], LDP);
                    simdgroup_load(V0, &Vs[hf*BK*LD + kk*LD + fc], LD);
                    simdgroup_multiply_accumulate(O[f], P, V0, O[f]);
                }
            }
        }
    }
    for (uint f=0; f<NFH; ++f){
        simdgroup_store(O[f], &Rsc[sg*QB*8u], 8u);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e=lane; e<QB*8u; e+=32u){
            uint r=e>>3, d=e&7u;
            if (m0+r<Mtot){
                float inv=lrun[h*QB+r]>0?1.0f/lrun[h*QB+r]:0.0f;
                out[((size_t)(m0+r)*num_q_heads + h)*VD + (vd0 + f*8u + d)] = (half)(Rsc[sg*QB*8u + e]*inv);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void attn_prefill_mma_hd256(
    device const half*  q     [[buffer(0)]],
    device const half*  knew  [[buffer(1)]],
    device const half*  vnew  [[buffer(2)]],
    device const char*  kc    [[buffer(3)]],
    device const char*  vc    [[buffer(4)]],
    device const float* ks    [[buffer(5)]],
    device const float* vs    [[buffer(6)]],
    device       half*  out   [[buffer(7)]],
    constant uint& num_q_heads [[buffer(8)]],  constant uint& num_kv_heads [[buffer(9)]],
    constant uint& head_dim    [[buffer(10)]], constant uint& v_hdim      [[buffer(11)]],
    constant uint& history_len [[buffer(12)]], constant float& scale      [[buffer(13)]],
    constant uint& q_pos0      [[buffer(14)]], constant uint& new_len     [[buffer(15)]],
    constant uint& Mtot        [[buffer(16)]],
    constant uint& sinkN       [[buffer(17)]], constant uint& ringR [[buffer(18)]],
    uint tgx [[threadgroup_position_in_grid]], uint tl [[thread_index_in_threadgroup]])
{
    const uint NSG=8u, LD=40u, BK=64u, QB=8u, HD=256u, VD=256u, NFH=32u;
    const uint LDK=72u, LDP=72u;
    threadgroup float poolA[4096];
    threadgroup half  poolB[4608];
    threadgroup float Rsc[NSG*QB*8u];
    threadgroup float mrun[8u*QB];
    threadgroup float lrun[8u*QB];
    threadgroup float resc8[8u*QB];
    threadgroup int   need_rescale[NSG];

    threadgroup half*  Ks = (threadgroup half*)poolA;
    threadgroup float* Ss = poolA;
    threadgroup half*  Vs = (threadgroup half*)poolA;
    threadgroup half*  Qs = poolB;
    threadgroup half*  Ps = poolB;

    uint sg = tl >> 5, lane = tl & 31u;
    uint h = sg, kvh = 0u;
    uint m0 = tgx*QB;
    uint total_keys = history_len + new_len;

    uint S = (ringR > 0u) ? sinkN : 0u;
    uint R = ringR, Wsr = S + R;
    uint pos_first = q_pos0 + m0;
    uint pos_last = q_pos0 + m0 + (QB-1u);
    uint kv_end_tile = min(total_keys, pos_last + 1u);
    bool windowed = (R > 0u && pos_first >= Wsr);
    uint rstart = windowed ? (pos_first + 1u - R) : 0u;
    uint nactive = windowed ? (S + (kv_end_tile - rstart)) : kv_end_tile;
    simdgroup_matrix<float,8,8> O[NFH];

    for (uint f=0; f<NFH; ++f) O[f] = make_filled_simdgroup_matrix<float,8,8>(0.f);
    if (lane < QB) { mrun[h*QB+lane] = -INFINITY; lrun[h*QB+lane]=0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k0=0; k0<nactive; k0+=BK) {
        uint kcount = min(BK, nactive - k0);
        simdgroup_matrix<float,8,8> C0=make_filled_simdgroup_matrix<float,8,8>(0.f);
        simdgroup_matrix<float,8,8> C1=C0,C2=C0,C3=C0,C4=C0,C5=C0,C6=C0,C7=C0;
        for (uint c=0; c<HD; c+=32u) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i=tl; i<32u*BK; i+=NSG*32u) {
                uint d=i/BK, k=i%BK;
                half val=(half)0;
                if (k<kcount) {
                    uint j=k0+k;
                    uint kk = windowed ? ((j<S)?j:(rstart+(j-S))) : j;
                    if (kk<history_len) {
                        uint slot = (R>0u && kk>=Wsr)?(S+(kk-S)%R):kk;
                        device const char*  kp=kc+((size_t)slot*num_kv_heads+kvh)*HD;
                        device const float* sp=ks+((size_t)slot*num_kv_heads+kvh)*(HD/32u);
                        val=(half)((float)kp[c+d]*sp[(c+d)/32u]);
                    } else {
                        device const half* kp=knew+((size_t)(kk-history_len)*num_kv_heads+kvh)*HD;
                        val=kp[c+d];
                    }
                }
                Ks[d*LDK + k]=val;
            }
            for (uint i=lane; i<QB*32u; i+=32u) {
                uint r=i>>5, d=i&31u;
                Qs[h*QB*LD + r*LD + d] = (m0+r<Mtot) ? q[((size_t)(m0+r)*num_q_heads + h)*HD + c + d] : (half)0;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint kk=0; kk<32u; kk+=8u) {
                simdgroup_matrix<half,8,8> A,B0,B1,B2,B3,B4,B5,B6,B7;
                simdgroup_load(A, &Qs[h*QB*LD + kk], LD);
                simdgroup_load(B0,&Ks[kk*LDK + 0u], LDK);
                simdgroup_load(B1,&Ks[kk*LDK + 8u], LDK);
                simdgroup_load(B2,&Ks[kk*LDK + 16u],LDK);
                simdgroup_load(B3,&Ks[kk*LDK + 24u],LDK);
                simdgroup_load(B4,&Ks[kk*LDK + 32u],LDK);
                simdgroup_load(B5,&Ks[kk*LDK + 40u],LDK);
                simdgroup_load(B6,&Ks[kk*LDK + 48u],LDK);
                simdgroup_load(B7,&Ks[kk*LDK + 56u],LDK);
                simdgroup_multiply_accumulate(C0,A,B0,C0);
                simdgroup_multiply_accumulate(C1,A,B1,C1);
                simdgroup_multiply_accumulate(C2,A,B2,C2);
                simdgroup_multiply_accumulate(C3,A,B3,C3);
                simdgroup_multiply_accumulate(C4,A,B4,C4);
                simdgroup_multiply_accumulate(C5,A,B5,C5);
                simdgroup_multiply_accumulate(C6,A,B6,C6);
                simdgroup_multiply_accumulate(C7,A,B7,C7);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_store(C0,&Ss[h*QB*BK + 0u], BK);
        simdgroup_store(C1,&Ss[h*QB*BK + 8u], BK);
        simdgroup_store(C2,&Ss[h*QB*BK + 16u],BK);
        simdgroup_store(C3,&Ss[h*QB*BK + 24u],BK);
        simdgroup_store(C4,&Ss[h*QB*BK + 32u],BK);
        simdgroup_store(C5,&Ss[h*QB*BK + 40u],BK);
        simdgroup_store(C6,&Ss[h*QB*BK + 48u],BK);
        simdgroup_store(C7,&Ss[h*QB*BK + 56u],BK);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane < QB) {
            uint r=lane, qpos=q_pos0+m0+r;
            threadgroup float* srow=&Ss[h*QB*BK + r*BK];
            float tmax=-INFINITY;
            for (uint k=0;k<kcount;++k){
                uint j=k0+k;
                uint kk = windowed ? ((j<S)?j:(rstart+(j-S))) : j;
                bool att = (kk<=qpos) && (R==0u || kk<S || kk+R>qpos);
                float s = att ? srow[k]*scale : -INFINITY;
                srow[k]=s; tmax=max(tmax,s);
            }
            float mo=mrun[h*QB+r], mnew=max(mo,tmax);
            float resc=exp(mo-mnew), tsum=0.0f;
            for (uint k=0;k<kcount;++k){
                float e=(srow[k]>-INFINITY)?exp(srow[k]-mnew):0.0f;
                srow[k]=e; tsum+=e;
            }
            lrun[h*QB+r]=lrun[h*QB+r]*resc+tsum;
            mrun[h*QB+r]=mnew; resc8[h*QB+r]=resc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane == 0u) {
            int nr = 0;
            for (uint r=0; r<QB; ++r) if (resc8[h*QB+r] != 1.0f) nr = 1;
            need_rescale[sg] = nr;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (need_rescale[sg]) {
            for (uint f=0; f<NFH; ++f) {
                simdgroup_store(O[f], &Rsc[sg*QB*8u], 8u);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                for (uint e=lane; e<QB*8u; e+=32u){ uint r=e>>3; Rsc[sg*QB*8u + e] *= resc8[h*QB+r]; }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                simdgroup_load(O[f], &Rsc[sg*QB*8u], 8u);
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        for (uint i=lane; i<QB*BK; i+=32u){ uint r=i/BK,k=i%BK; Ps[h*QB*LDP + r*LDP + k]=(k<kcount)?(half)Ss[h*QB*BK + r*BK + k]:(half)0; }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint hl = h*32u + lane;
        for (uint jc=0;jc<VD;jc+=32u){
            uint c = jc;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i=hl;i<BK*32u;i+=NSG*32u){
                uint k=i>>5, d=i&31u;
                half val=(half)0;
                if (k<kcount){
                    uint j=k0+k;
                    uint kk = windowed ? ((j<S)?j:(rstart+(j-S))) : j;
                    if (kk<history_len){
                        uint slot = (R>0u && kk>=Wsr)?(S+(kk-S)%R):kk;
                        device const char*  vp=vc+((size_t)slot*num_kv_heads+kvh)*VD;
                        device const float* sp=vs+((size_t)slot*num_kv_heads+kvh)*(VD/32u);
                        val=(half)((float)vp[c+d]*sp[(c+d)/32u]);
                    } else {
                        device const half* vp=vnew+((size_t)(kk-history_len)*num_kv_heads+kvh)*VD;
                        val=vp[c+d];
                    }
                }
                Vs[k*LD + d]=val;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint fc=0; fc<32u; fc+=8u){
                uint f = jc/8u + fc/8u;
                for (uint kk=0; kk<BK; kk+=8u){
                    simdgroup_matrix<half,8,8> P,V0;
                    simdgroup_load(P,  &Ps[h*QB*LDP + kk], LDP);
                    simdgroup_load(V0, &Vs[kk*LD + fc], LD);
                    simdgroup_multiply_accumulate(O[f], P, V0, O[f]);
                }
            }
        }
    }
    for (uint f=0; f<NFH; ++f){
        simdgroup_store(O[f], &Rsc[sg*QB*8u], 8u);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e=lane; e<QB*8u; e+=32u){
            uint r=e>>3, d=e&7u;
            if (m0+r<Mtot){
                float inv=lrun[h*QB+r]>0?1.0f/lrun[h*QB+r]:0.0f;
                out[((size_t)(m0+r)*num_q_heads + h)*VD + (f*8u + d)] = (half)(Rsc[sg*QB*8u + e]*inv);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void kv_append_i8_m(device const half* src [[buffer(0)]], device char* int8base [[buffer(1)]],
    device float* scalebase [[buffer(2)]], constant uint& kv_heads [[buffer(3)]],
    constant uint& hdim [[buffer(4)]], constant uint& current_len [[buffer(5)]],
    constant uint& group_size [[buffer(6)]], constant uint& M [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
    uint num_groups = (hdim + group_size - 1)/group_size, per = kv_heads*num_groups;
    if (gid >= M*per) return;
    uint i = gid / per, hg = gid % per, h = hg / num_groups, g = hg % num_groups;
    uint gstart = g*group_size, gcount = min(group_size, hdim - gstart);
    device const half* hs = src + (size_t)i*kv_heads*hdim + (size_t)h*hdim + gstart;
    float maxabs = 0;
    for (uint k=0;k<gcount;++k) maxabs = max(maxabs, fabs((float)hs[k]));
    float scale = maxabs/127.0f; if (scale < 1e-10f) scale = 1e-10f;
    float inv = 1.0f/scale;
    uint i8s = kv_heads*hdim, scs = kv_heads*num_groups;
    device char* dst = int8base + (size_t)(current_len+i)*i8s + (size_t)h*hdim + gstart;
    for (uint k=0;k<gcount;++k){ float qv = clamp(rint((float)hs[k]*inv), -128.0f, 127.0f); dst[k]=(char)qv; }
    scalebase[(size_t)(current_len+i)*scs + (size_t)h*num_groups + g] = scale;
}
kernel void gemv_bias_f16(device const half* x [[buffer(0)]], device const half* w [[buffer(1)]],
    device const half* bias [[buffer(2)]], device half* y [[buffer(3)]],
    constant uint& K [[buffer(4)]], constant uint& N [[buffer(5)]],
    constant uint& tr [[buffer(6)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint n = tg * 4u + sgid;
    if (n >= N) return;
    float acc = 0;
    if (tr != 0u) {
        device const half* wr = w + (size_t)n * K;
        for (uint k = lane; k < K; k += 32u) acc = fma((float)x[k], (float)wr[k], acc);
    } else {
        for (uint k = lane; k < K; k += 32u) acc = fma((float)x[k], (float)w[(size_t)k * N + n], acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) y[n] = (half)((float)(half)acc + (float)bias[n]);
}
kernel void rel_pos_bias_f16(device const half* q [[buffer(0)]], device const half* r [[buffer(1)]],
    device half* y [[buffer(2)]], constant uint& T [[buffer(3)]], constant uint& H [[buffer(4)]],
    constant uint& D [[buffer(5)]], constant uint& rbs [[buffer(6)]],
    constant float& scale [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]]) {
    uint tj = gid.x, h = gid.y, b = gid.z;
    if (tj >= T * T || h >= H) return;
    uint t = tj / T, j = tj % T;
    device const half* qv = q + ((size_t)b * T + t) * H * D + h * D;
    uint rel = (T - 1u) - t + j;
    device const half* rv = r + (size_t)b * rbs + (size_t)rel * H * D + h * D;
    float acc = 0;
    for (uint d = 0; d < D; ++d) acc = fma((float)qv[d], (float)rv[d], acc);
    y[((size_t)b * H + h) * T * T + (size_t)t * T + j] = (half)(acc * scale);
}
kernel void conv_cache_append_f16(device const uchar* src [[buffer(0)]],
    device half* ring [[buffer(1)]], device half* out [[buffer(2)]],
    constant uint& hd [[buffer(3)]], constant uint& ws [[buffer(4)]],
    constant uint& nnew [[buffer(5)]], constant uint& head0 [[buffer(6)]],
    constant uint& count_new [[buffer(7)]], constant uint& num_rows [[buffer(8)]],
    constant uint& src_f32 [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]) {
    uint x = gid.x, y = gid.y;
    if (x >= hd || y >= ws + nnew) return;
    uint start_row = num_rows - nnew;
    if (y >= ws) {
        uint i = y - ws;
        uint sidx = (start_row + i) * hd + x;
        half v = src_f32 ? (half)((device const float*)src)[sidx] : ((device const half*)src)[sidx];
        ring[((head0 + i) % ws) * hd + x] = v;
        return;
    }
    uint pad = ws - count_new;
    if (y < pad) { out[y * hd + x] = 0.0h; return; }
    uint a = count_new - 1u - (y - pad);
    half v;
    if (a < nnew) {
        uint sidx = (num_rows - 1u - a) * hd + x;
        v = src_f32 ? (half)((device const float*)src)[sidx] : ((device const half*)src)[sidx];
    } else {
        uint b = a - nnew;
        v = ring[((head0 + 2u * ws - 1u - b) % ws) * hd + x];
    }
    out[y * hd + x] = v;
}
kernel void kv_append_ring_i8_m(device const half* src [[buffer(0)]], device char* int8base [[buffer(1)]],
    device float* scalebase [[buffer(2)]], constant uint& kv_heads [[buffer(3)]],
    constant uint& hdim [[buffer(4)]], constant uint& current_len [[buffer(5)]],
    constant uint& group_size [[buffer(6)]], constant uint& M [[buffer(7)]],
    constant uint& sink [[buffer(8)]], constant uint& W [[buffer(9)]],
    uint gid [[thread_position_in_grid]]) {
    uint num_groups = (hdim + group_size - 1)/group_size, per = kv_heads*num_groups;
    if (gid >= M*per) return;
    uint i = gid / per, hg = gid % per, h = hg / num_groups, g = hg % num_groups;
    uint gstart = g*group_size, gcount = min(group_size, hdim - gstart);
    device const half* hs = src + (size_t)i*kv_heads*hdim + (size_t)h*hdim + gstart;
    float maxabs = 0;
    for (uint k=0;k<gcount;++k) maxabs = max(maxabs, fabs((float)hs[k]));
    float scale = maxabs/127.0f; if (scale < 1e-10f) scale = 1e-10f;
    float inv = 1.0f/scale;
    uint i8s = kv_heads*hdim, scs = kv_heads*num_groups;
    uint pos = current_len + i;
    uint R = (W > sink) ? (W - sink) : 1u;
    uint slot = (pos < W) ? pos : sink + ((pos - sink) % R);
    if (pos >= sink && pos + R < current_len + M) return;
    device char* dst = int8base + (size_t)slot*i8s + (size_t)h*hdim + gstart;
    for (uint k=0;k<gcount;++k){ float qv = clamp(rint((float)hs[k]*inv), -128.0f, 127.0f); dst[k]=(char)qv; }
    scalebase[(size_t)slot*scs + (size_t)h*num_groups + g] = scale;
}
struct AdjU { uint n_recent, suppress_flag, suppress_id, vocab; float penalty; };
kernel void adjust_logits(device half* logits [[buffer(0)]],
                          device const uint* recent [[buffer(1)]],
                          constant AdjU& U [[buffer(2)]],
                          uint t [[thread_position_in_threadgroup]],
                          uint T [[threads_per_threadgroup]]) {
    if (U.penalty != 1.0f) {
        for (uint i=t; i<U.n_recent; i+=T) {
            uint id = recent[i];
            if (id >= U.vocab) continue;
            float v = (float)logits[id];
            logits[id] = (half)(v > 0.0f ? v/U.penalty : v*U.penalty);
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (t == 0u && U.suppress_flag != 0u && U.suppress_id < U.vocab) logits[U.suppress_id] = half(-65504.0f);
}

kernel void softcap_f16(device const half* in [[buffer(0)]], device half* y [[buffer(1)]],
                        constant uint& n [[buffer(2)]], constant float& cap [[buffer(3)]],
                        uint i [[thread_position_in_grid]]) {
    if (i>=n) return;
    half v1 = (half)((float)in[i]/cap);
    half v2 = (half)precise::tanh((float)v1);
    y[i]=(half)((float)v2*cap);
}

kernel void argmax_part(device const half* logits [[buffer(0)]],
                        device float* part [[buffer(1)]],
                        constant uint& V [[buffer(2)]],
                        constant uint& chunk [[buffer(3)]],
                        device const float* bias [[buffer(4)]],
                        constant uint& has_bias [[buffer(5)]],
                        uint p [[threadgroup_position_in_grid]],
                        uint t [[thread_position_in_threadgroup]],
                        uint T [[threads_per_threadgroup]],
                        threadgroup float* sb [[threadgroup(0)]],
                        threadgroup uint* si [[threadgroup(1)]],
                        threadgroup float* ss [[threadgroup(2)]]) {
    uint lo = p*chunk, hi = min(V, lo+chunk);
    float b = -INFINITY, s = -INFINITY; uint bi = lo;
    for (uint i = lo+t; i < hi; i += T) {
        float v = (float)logits[i];
        if (has_bias != 0u) v += bias[i];
        if (v > b) { s = b; b = v; bi = i; }
        else if (v > s) { s = v; }
    }
    sb[t] = b; si[t] = bi; ss[t] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = T >> 1; stride > 0u; stride >>= 1) {
        if (t < stride) {
            float ab = sb[t], asx = ss[t]; uint ai = si[t];
            float bb = sb[t+stride], bsx = ss[t+stride]; uint bidx = si[t+stride];
            if (ab > bb || (ab == bb && ai <= bidx)) { sb[t] = ab; si[t] = ai; ss[t] = max(asx, bb); }
            else { sb[t] = bb; si[t] = bidx; ss[t] = max(bsx, ab); }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t == 0u) { part[p*3u] = sb[0]; part[p*3u+1u] = ss[0]; part[p*3u+2u] = (float)si[0]; }
}

kernel void argmax_final(device const float* part [[buffer(0)]],
                         device float* out3 [[buffer(1)]],
                         constant uint& NP [[buffer(2)]],
                         uint t [[thread_position_in_threadgroup]],
                         uint T [[threads_per_threadgroup]],
                         threadgroup float* sb [[threadgroup(0)]],
                         threadgroup uint* si [[threadgroup(1)]],
                         threadgroup float* ss [[threadgroup(2)]]) {
    float b = -INFINITY, s = -INFINITY; uint bi = 0u;
    for (uint i = t; i < NP; i += T) {
        float pb = part[i*3u], ps = part[i*3u+1u]; uint pi = (uint)part[i*3u+2u];
        if (pb > b || (pb == b && pi <= bi)) { s = max(s, b); b = pb; bi = pi; s = max(s, ps); }
        else { s = max(s, pb); }
    }
    sb[t] = b; si[t] = bi; ss[t] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = T >> 1; stride > 0u; stride >>= 1) {
        if (t < stride) {
            float ab = sb[t], asx = ss[t]; uint ai = si[t];
            float bb = sb[t+stride], bsx = ss[t+stride]; uint bidx = si[t+stride];
            if (ab > bb || (ab == bb && ai <= bidx)) { sb[t] = ab; si[t] = ai; ss[t] = max(asx, bb); }
            else { sb[t] = bb; si[t] = bidx; ss[t] = max(bsx, ab); }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t == 0u) { out3[0] = sb[0]; out3[1] = ss[0]; out3[2] = (float)si[0]; }
}

kernel void argmax_logits(device const half* logits [[buffer(0)]],
                          device float* out3 [[buffer(1)]],
                          constant uint& V [[buffer(2)]],
                          uint t [[thread_position_in_threadgroup]],
                          uint T [[threads_per_threadgroup]],
                          threadgroup float* sb [[threadgroup(0)]],
                          threadgroup uint* si [[threadgroup(1)]],
                          threadgroup float* ss [[threadgroup(2)]]) {
    float b = -INFINITY, s = -INFINITY; uint bi = 0u;
    for (uint i = t; i < V; i += T) {
        float v = (float)logits[i];
        if (v > b) { s = b; b = v; bi = i; }
        else if (v > s) { s = v; }
    }
    sb[t] = b; si[t] = bi; ss[t] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = T >> 1; stride > 0u; stride >>= 1) {
        if (t < stride) {
            float ab = sb[t], asx = ss[t]; uint ai = si[t];
            float bb = sb[t+stride], bsx = ss[t+stride]; uint bidx = si[t+stride];
            if (ab > bb || (ab == bb && ai <= bidx)) { sb[t] = ab; si[t] = ai; ss[t] = max(asx, bb); }
            else { sb[t] = bb; si[t] = bidx; ss[t] = max(bsx, ab); }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t == 0u) { out3[0] = sb[0]; out3[1] = ss[0]; out3[2] = (float)si[0]; }
}

kernel void topk_rows_f16(device const half* in [[buffer(0)]],
                          device float* out [[buffer(1)]],
                          constant uint& F [[buffer(2)]],
                          constant uint& k [[buffer(3)]],
                          constant uint& B [[buffer(4)]],
                          uint row [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]]) {
    device const half* x = in + (size_t)row*F;
    device float* idx_out = out + (size_t)row*k;
    device float* val_out = out + (size_t)B*k + (size_t)row*k;
    for (uint j = 0u; j < k; ++j) {
        float best = -INFINITY; uint bi = 0xFFFFFFFFu;
        for (uint i = lane; i < F; i += 32u) {
            bool taken = false;
            for (uint p = 0u; p < j; ++p) if ((uint)idx_out[p] == i) taken = true;
            float v = (float)x[i];
            if (!taken && (v > best || (v == best && i < bi))) { best = v; bi = i; }
        }
        float gb = simd_max(best);
        uint cand = (best == gb) ? bi : 0xFFFFFFFFu;
        uint gi = simd_min(cand);
        if (lane == 0u) { idx_out[j] = (float)gi; val_out[j] = gb; }
        simdgroup_barrier(mem_flags::mem_device);
    }
}

kernel void cq4_moe_transform(
    device const half*  x       [[buffer(0)]],
    device const float* topk    [[buffer(1)]],
    device const half*  recip_c [[buffer(2)]],
    device const char*  lsign_c [[buffer(3)]],
    device const char*  rsign_c [[buffer(4)]],
    device const uint*  perm_c  [[buffer(5)]],
    device       half*  code    [[buffer(6)]],
    constant uint& K            [[buffer(7)]],
    constant uint& k_valid      [[buffer(8)]],
    constant uint& xz_stride    [[buffer(9)]],
    constant uint& xy_stride    [[buffer(10)]],
    constant uint& topk_k       [[buffer(11)]],
    constant uint& estride      [[buffer(12)]],
    uint3 tgp [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    threadgroup float* zmem     [[threadgroup(0)]])
{
    const uint g = tgp.x, slot = tgp.y, tok = tgp.z;
    const uint e = (uint)(topk[(size_t)tok*topk_k + slot] + 0.5f);
    device const half* recip = (device const half*)((device const uchar*)recip_c + (size_t)e*estride);
    device const char* lsign = lsign_c + (size_t)e*estride;
    device const char* rsign = rsign_c + (size_t)e*estride;
    device const uint* perm  = (device const uint*)((device const uchar*)perm_c + (size_t)e*estride);
    device const half* xs = x + (size_t)tok*xz_stride + (size_t)slot*xy_stride;
    uint b = g*128u + lane*4u;
    uint k = lane*4u;
    float xv[4];
    #pragma clang loop unroll(full)
    for (uint j = 0u; j < 4u; ++j) {
        float v = (b+j < k_valid) ? (float)xs[b+j] : 0.0f;
        xv[j] = v*(float)recip[b+j]*(float)lsign[k+j];
    }
    float x0 = xv[0], x1 = xv[1], x2 = xv[2], x3 = xv[3];
    cq4_hada128(x0, x1, x2, x3, lane);
    zmem[k+0] = x0*(float)rsign[k+0]; zmem[k+1] = x1*(float)rsign[k+1];
    zmem[k+2] = x2*(float)rsign[k+2]; zmem[k+3] = x3*(float)rsign[k+3];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device half* co = code + ((size_t)tok*topk_k + slot)*K + g*128u;
    co[k+0] = (half)zmem[perm[k+0]]; co[k+1] = (half)zmem[perm[k+1]];
    co[k+2] = (half)zmem[perm[k+2]]; co[k+3] = (half)zmem[perm[k+3]];
}

kernel void cq4_moe_gemv_up(
    device const half*  code1 [[buffer(0)]],
    device const half*  code3 [[buffer(1)]],
    device const float* topk  [[buffer(2)]],
    device const uchar* pk1   [[buffer(3)]],
    device const half*  nm1   [[buffer(4)]],
    device const half*  cb1   [[buffer(5)]],
    device const uchar* pk3   [[buffer(6)]],
    device const half*  nm3   [[buffer(7)]],
    device const half*  cb3   [[buffer(8)]],
    device       half*  y     [[buffer(9)]],
    constant uint& K          [[buffer(10)]],
    constant uint& N          [[buffer(11)]],
    constant uint& act        [[buffer(12)]],
    constant uint& topk_k     [[buffer(13)]],
    constant uint& estride1   [[buffer(14)]],
    constant uint& estride3   [[buffer(15)]],
    uint3 tgp [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tl   [[thread_index_in_threadgroup]])
{
    const uint slot = tgp.y, tok = tgp.z;
    const uint slotg = tok*topk_k + slot;
    const uint e = (uint)(topk[(size_t)tok*topk_k + slot] + 0.5f);
    const uint ng = K/128u;
    threadgroup float c1[16], c3[16];
    if (tl < 16u) {
        c1[tl] = (float)((device const half*)((device const uchar*)cb1 + (size_t)e*estride1))[tl];
        c3[tl] = (float)((device const half*)((device const uchar*)cb3 + (size_t)e*estride3))[tl];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint n0 = (tgp.x*8u + sgid)*4u;
    if (n0 >= N) return;
    float gv[4] = {0,0,0,0}, uv[4] = {0,0,0,0};
    for (int m = 0; m < 2; ++m) {
        device const uchar* pb = (m==0) ? pk1 + (size_t)e*estride1 : pk3 + (size_t)e*estride3;
        device const half*  nb = (m==0) ? (device const half*)((device const uchar*)nm1 + (size_t)e*estride1)
                                        : (device const half*)((device const uchar*)nm3 + (size_t)e*estride3);
        device const half*  cd = (m==0) ? code1 + (size_t)slotg*K : code3 + (size_t)slotg*K;
        threadgroup float* cb = (m==0) ? c1 : c3;
        float a0=0, a1=0, a2=0, a3=0;
        for (uint base = lane*16u; base < K; base += 512u) {
            uint gg = base/128u, off = base - gg*128u;
            device const half4* cp = (device const half4*)(cd + gg*128u + off);
            half4 c0=cp[0], c1v=cp[1], c2v=cp[2], c3v=cp[3];
            device const uchar* pan = pb + ((size_t)(n0>>2)*ng+gg)*256u + (off>>4)*32u;
            uint2 A01 = ((device const uint2*)pan)[0];
            uint2 A23 = ((device const uint2*)(pan+8u))[0];
            uint2 B01 = ((device const uint2*)(pan+16u))[0];
            uint2 B23 = ((device const uint2*)(pan+24u))[0];
            size_t nb4 = ((size_t)(n0>>2)*ng+gg)<<2;
            a0 += (float)nb[nb4+0u]*cq4_dot64_ilw(c0, c1v, c2v, c3v, A01.x, B01.x, cb);
            a1 += (float)nb[nb4+1u]*cq4_dot64_ilw(c0, c1v, c2v, c3v, A01.y, B01.y, cb);
            a2 += (float)nb[nb4+2u]*cq4_dot64_ilw(c0, c1v, c2v, c3v, A23.x, B23.x, cb);
            a3 += (float)nb[nb4+3u]*cq4_dot64_ilw(c0, c1v, c2v, c3v, A23.y, B23.y, cb);
        }
        a0 = simd_sum(a0); a1 = simd_sum(a1); a2 = simd_sum(a2); a3 = simd_sum(a3);
        if (m == 0) { gv[0]=a0; gv[1]=a1; gv[2]=a2; gv[3]=a3; }
        else { uv[0]=a0; uv[1]=a1; uv[2]=a2; uv[3]=a3; }
    }
    if (lane == 0u) {
        device half* ys = y + (size_t)slotg*N;
        for (uint r = 0u; r < 4u && n0+r < N; ++r) {
            float gh = (float)(half)gv[r];
            float av = (act == 1u) ? gelu_tanh(gh) : gh/(1.0f + precise::exp(-gh));
            ys[n0+r] = (half)((float)(half)av*(float)(half)uv[r]);
        }
    }
}

kernel void cq4_moe_transform2(
    device const half*  x        [[buffer(0)]],
    device const float* topk     [[buffer(1)]],
    device const half*  recip_c1 [[buffer(2)]],
    device const char*  lsign_c1 [[buffer(3)]],
    device const char*  rsign_c1 [[buffer(4)]],
    device const uint*  perm_c1  [[buffer(5)]],
    device const half*  recip_c3 [[buffer(6)]],
    device const char*  lsign_c3 [[buffer(7)]],
    device const char*  rsign_c3 [[buffer(8)]],
    device const uint*  perm_c3  [[buffer(9)]],
    device       half*  code1    [[buffer(10)]],
    device       half*  code3    [[buffer(11)]],
    constant uint& K             [[buffer(12)]],
    constant uint& topk_k        [[buffer(13)]],
    constant uint& estride1      [[buffer(14)]],
    constant uint& estride3      [[buffer(15)]],
    uint3 tgp [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    threadgroup float* zmem      [[threadgroup(0)]])
{
    const uint g = tgp.x, tok = tgp.z;
    const bool w3 = tgp.y >= topk_k;
    const uint slot = w3 ? tgp.y - topk_k : tgp.y;
    const uint e = (uint)(topk[(size_t)tok*topk_k + slot] + 0.5f);
    const uint es = w3 ? estride3 : estride1;
    device const half* recip = (device const half*)((device const uchar*)(w3 ? recip_c3 : recip_c1) + (size_t)e*es);
    device const char* lsign = (w3 ? lsign_c3 : lsign_c1) + (size_t)e*es;
    device const char* rsign = (w3 ? rsign_c3 : rsign_c1) + (size_t)e*es;
    device const uint* perm  = (device const uint*)((device const uchar*)(w3 ? perm_c3 : perm_c1) + (size_t)e*es);
    device const half* xs = x + (size_t)tok*K;
    uint b = g*128u + lane*4u;
    uint k = lane*4u;
    float x0=(float)xs[b+0]*(float)recip[b+0]*(float)lsign[k+0];
    float x1=(float)xs[b+1]*(float)recip[b+1]*(float)lsign[k+1];
    float x2=(float)xs[b+2]*(float)recip[b+2]*(float)lsign[k+2];
    float x3=(float)xs[b+3]*(float)recip[b+3]*(float)lsign[k+3];
    cq4_hada128(x0, x1, x2, x3, lane);
    zmem[k+0]=x0*(float)rsign[k+0]; zmem[k+1]=x1*(float)rsign[k+1];
    zmem[k+2]=x2*(float)rsign[k+2]; zmem[k+3]=x3*(float)rsign[k+3];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device half* co = (w3 ? code3 : code1) + ((size_t)tok*topk_k + slot)*K + g*128u;
    co[k+0]=(half)zmem[perm[k+0]]; co[k+1]=(half)zmem[perm[k+1]];
    co[k+2]=(half)zmem[perm[k+2]]; co[k+3]=(half)zmem[perm[k+3]];
}

kernel void cq4_moe_gemv_down_acc(
    device const half*  code2 [[buffer(0)]],
    device const float* topk  [[buffer(1)]],
    device const half*  probs [[buffer(2)]],
    device const uchar* pk2   [[buffer(3)]],
    device const half*  nm2   [[buffer(4)]],
    device const half*  cb2   [[buffer(5)]],
    device       half*  out   [[buffer(6)]],
    constant uint& K          [[buffer(7)]],
    constant uint& N          [[buffer(8)]],
    constant uint& topk_k     [[buffer(9)]],
    constant uint& normalize  [[buffer(10)]],
    constant float& eps       [[buffer(11)]],
    constant float& scaling   [[buffer(12)]],
    constant uint& E          [[buffer(13)]],
    constant uint& estride    [[buffer(14)]],
    uint3 tgp [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tl   [[thread_index_in_threadgroup]],
    threadgroup float* cbs    [[threadgroup(0)]])
{
    const uint tok = tgp.z;
    device const float* tk = topk + (size_t)tok*topk_k;
    device const half* pr = probs + (size_t)tok*E;
    uint order[16];
    for (uint j = 0u; j < topk_k; ++j) order[j] = j;
    for (uint a = 1u; a < topk_k; ++a) {
        uint v = order[a];
        uint ev = (uint)(tk[v] + 0.5f);
        uint b = a;
        while (b > 0u && (uint)(tk[order[b-1u]] + 0.5f) > ev) { order[b] = order[b-1u]; --b; }
        order[b] = v;
    }
    if (tl < topk_k*16u) {
        uint s = tl >> 4, ci = tl & 15u;
        uint e = (uint)(tk[s] + 0.5f);
        cbs[tl] = (float)((device const half*)((device const uchar*)cb2 + (size_t)e*estride))[ci];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float denom = 1.0f;
    if (normalize != 0u) {
        float s = 0.0f;
        for (uint j = 0u; j < topk_k; ++j) s += (float)pr[(uint)(tk[j] + 0.5f)];
        denom = s + eps;
    }
    const uint ng = K/128u;
    uint n0 = (tgp.x*8u + sgid)*2u;
    if (n0 >= N) return;
    half res[2] = {0.0h, 0.0h};
    for (uint j = 0u; j < topk_k; ++j) {
        uint slot = order[j];
        uint e = (uint)(tk[slot] + 0.5f);
        float p = (float)pr[e];
        if (p <= 0.0f) continue;
        float w = p/denom*scaling;
        device const uchar* pbase = pk2 + (size_t)e*estride;
        device const half*  nbase = (device const half*)((device const uchar*)nm2 + (size_t)e*estride);
        device const half*  cd = code2 + ((size_t)tok*topk_k + slot)*K;
        threadgroup const float* cb = cbs + (size_t)slot*16u;
        float acc[2] = {0, 0};
        for (uint base = lane*16u; base < K; base += 512u) {
            uint gg = base/128u, off = base - gg*128u;
            device const half4* cp = (device const half4*)(cd + gg*128u + off);
            half4 c0=cp[0], c1=cp[1], c2=cp[2], c3=cp[3];
            device const uchar* pan = pbase + ((size_t)(n0>>2)*ng+gg)*256u + (n0&3u)*4u + (off>>4)*32u;
            uint2 Aw = ((device const uint2*)pan)[0];
            uint2 Bw = ((device const uint2*)(pan+16u))[0];
            #pragma clang loop unroll(full)
            for (uint r = 0u; r < 2u; ++r) {
                float pv = cq4_dot64_ilw(c0, c1, c2, c3, r?Aw.y:Aw.x, r?Bw.y:Bw.x, cb);
                acc[r] += (float)nbase[(((size_t)(n0>>2)*ng+gg)<<2)+(n0&3u)+r]*pv;
            }
        }
        #pragma clang loop unroll(full)
        for (uint r = 0u; r < 2u; ++r) {
            float a = simd_sum(acc[r]);
            res[r] = (half)((float)res[r] + (float)(half)a*w);
        }
    }
    if (lane == 0u) {
        device half* ys = out + (size_t)tok*N;
        for (uint r = 0u; r < 2u && n0+r < N; ++r) ys[n0+r] = res[r];
    }
}

kernel void rope_pair_f16(
    device const half* x [[buffer(0)]],
    device const half* c [[buffer(1)]],
    device const half* s [[buffer(2)]],
    device       half* y [[buffer(3)]],
    constant uint& H     [[buffer(4)]],
    constant uint& D     [[buffer(5)]],
    uint2 g [[thread_position_in_grid]])
{
    uint d = g.x, h = g.y;
    if (d >= D || h >= H) return;
    size_t idx = (size_t)h*D + d;
    half xv = x[idx];
    half rh = (d < D/2u) ? (half)(-x[idx + D/2u]) : x[idx - D/2u];
    y[idx] = xv*c[d] + rh*s[d];
}

kernel void rms_norm_scale_f16(
    device const half* in [[buffer(0)]],
    device const half* w  [[buffer(1)]],
    device       half* y  [[buffer(2)]],
    constant uint& D      [[buffer(3)]],
    constant float& eps   [[buffer(4)]],
    constant float& oscale [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint t   [[thread_position_in_threadgroup]],
    uint T   [[threads_per_threadgroup]],
    threadgroup float* red [[threadgroup(0)]])
{
    device const half* x = in + (size_t)row*D;
    device half* o = y + (size_t)row*D;
    float ss = 0;
    for (uint i = t; i < D; i += T) { float v = (float)x[i]; ss += v*v; }
    red[t] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = T/2; s > 0; s >>= 1) {
        if (t < s) red[t] += red[t+s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = rsqrt(red[0]/(float)D + eps);
    for (uint i = t; i < D; i += T) {
        half y0 = (half)((float)x[i]*inv*(float)w[i]);
        o[i] = (half)((float)y0*oscale);
    }
}

kernel void softmax_topk_f16(
    device const half* in [[buffer(0)]],
    device       half* probs [[buffer(1)]],
    device      float* tk  [[buffer(2)]],
    constant uint& E       [[buffer(3)]],
    constant uint& k       [[buffer(4)]],
    constant uint& B       [[buffer(5)]],
    constant float& scale  [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    device const half* x = in + (size_t)row*E;
    device half* pr = probs + (size_t)row*E;
    device float* idx_out = tk + (size_t)row*k;
    device float* val_out = tk + (size_t)B*k + (size_t)row*k;
    float mx = -INFINITY;
    for (uint i = lane; i < E; i += 32u) mx = max(mx, (float)(half)((float)x[i]*scale));
    mx = simd_max(mx);
    float sum = 0;
    for (uint i = lane; i < E; i += 32u) sum += exp((float)(half)((float)x[i]*scale) - mx);
    sum = simd_sum(sum);
    float inv = 1.0f/sum;
    for (uint i = lane; i < E; i += 32u) pr[i] = (half)(exp((float)(half)((float)x[i]*scale) - mx)*inv);
    for (uint j = 0u; j < k; ++j) {
        float best = -INFINITY; uint bi = 0xFFFFFFFFu;
        for (uint i = lane; i < E; i += 32u) {
            bool taken = false;
            for (uint p = 0u; p < j; ++p) if ((uint)idx_out[p] == i) taken = true;
            float v = (float)(half)((float)x[i]*scale);
            if (!taken && (v > best || (v == best && i < bi))) { best = v; bi = i; }
        }
        float gb = simd_max(best);
        uint cand = (best == gb) ? bi : 0xFFFFFFFFu;
        uint gi = simd_min(cand);
        if (lane == 0u) { idx_out[j] = (float)gi; val_out[j] = gb; }
        simdgroup_barrier(mem_flags::mem_device);
    }
}

kernel void rope_pair_rms_f16(
    device const half* x [[buffer(0)]],
    device const half* w [[buffer(1)]],
    device const half* c [[buffer(2)]],
    device const half* s [[buffer(3)]],
    device       half* y [[buffer(4)]],
    constant uint& D     [[buffer(5)]],
    constant float& eps  [[buffer(6)]],
    uint h [[threadgroup_position_in_grid]],
    uint t [[thread_position_in_threadgroup]],
    uint T [[threads_per_threadgroup]],
    threadgroup float* red [[threadgroup(0)]],
    threadgroup half* xn   [[threadgroup(1)]])
{
    device const half* xr = x + (size_t)h*D;
    float ss = 0;
    for (uint i = t; i < D; i += T) { float v = (float)xr[i]; ss += v*v; }
    red[t] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint st = T/2; st > 0; st >>= 1) {
        if (t < st) red[t] += red[t+st];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = rsqrt(red[0]/(float)D + eps);
    for (uint i = t; i < D; i += T) xn[i] = (half)((float)xr[i]*inv*(float)w[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device half* yr = y + (size_t)h*D;
    for (uint i = t; i < D; i += T) {
        half xv = xn[i];
        half rh = (i < D/2u) ? (half)(-xn[i + D/2u]) : xn[i - D/2u];
        yr[i] = xv*c[i] + rh*s[i];
    }
}

kernel void rms2_add_clip_f16(
    device const half* a  [[buffer(0)]],
    device const half* wa [[buffer(1)]],
    device const half* b  [[buffer(2)]],
    device const half* wb [[buffer(3)]],
    device       half* y  [[buffer(4)]],
    constant uint& D      [[buffer(5)]],
    constant float& eps_a [[buffer(6)]],
    constant float& eps_b [[buffer(7)]],
    uint t [[thread_position_in_threadgroup]],
    uint T [[threads_per_threadgroup]],
    threadgroup float* red [[threadgroup(0)]])
{
    float sa = 0, sb = 0;
    for (uint i = t; i < D; i += T) {
        float va = (float)a[i], vb = (float)b[i];
        sa += va*va; sb += vb*vb;
    }
    red[t] = sa; red[T+t] = sb;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = T/2; s > 0; s >>= 1) {
        if (t < s) { red[t] += red[t+s]; red[T+t] += red[T+t+s]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inva = rsqrt(red[0]/(float)D + eps_a);
    float invb = rsqrt(red[T]/(float)D + eps_b);
    for (uint i = t; i < D; i += T) {
        half ya = (half)((float)a[i]*inva*(float)wa[i]);
        half yb = (half)((float)b[i]*invb*(float)wb[i]);
        y[i] = (half)clamp((float)ya + (float)yb, -65500.0f, 65500.0f);
    }
}

kernel void gated_deltanet_decode_f16(
    device const half* q  [[buffer(0)]],
    device const half* k  [[buffer(1)]],
    device const half* v  [[buffer(2)]],
    device const half* g  [[buffer(3)]],
    device const half* b  [[buffer(4)]],
    device const half* s  [[buffer(5)]],
    device       half* y  [[buffer(6)]],
    constant uint& Hq     [[buffer(7)]],
    constant uint& Hv     [[buffer(8)]],
    constant uint& K      [[buffer(9)]],
    constant uint& V      [[buffer(10)]],
    constant float& scale [[buffer(11)]],
    uint3 tgp [[threadgroup_position_in_grid]],
    uint t    [[thread_index_in_threadgroup]],
    threadgroup half* kq  [[threadgroup(0)]])
{
    const uint h = tgp.x, batch = tgp.y;
    const uint qk_head = h / (Hv / Hq);
    const uint qk_base = (batch * Hq + qk_head) * K;
    threadgroup half* kk = kq;
    threadgroup half* qq = kq + K;
    for (uint i = t; i < K; i += V) {
        kk[i] = k[qk_base + i];
        qq[i] = q[qk_base + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t >= V) return;

    float gate_log = (float)g[batch * Hv + h];
    if (!isfinite(gate_log)) gate_log = -20.0f;
    float beta = (float)b[batch * Hv + h];
    beta = isfinite(beta) ? clamp(beta, 0.0f, 1.0f) : 0.0f;
    const float alpha = exp(clamp(gate_log, -20.0f, 6.0f));

    const size_t s_stride = (size_t)Hv * V;
    device const half* sp = s + ((size_t)batch * K) * s_stride + (size_t)h * V + t;
    float proj = 0.0f;
    for (uint kd = 0; kd < K; ++kd) proj += (float)sp[kd * s_stride] * (float)kk[kd];
    const float delta = ((float)v[((size_t)batch * Hv + h) * V + t] - alpha * proj) * beta;

    device half* yp = y + ((size_t)batch * (1 + K) + 1) * s_stride + (size_t)h * V + t;
    float acc = 0.0f;
    for (uint kd = 0; kd < K; ++kd) {
        float s_new = (float)sp[kd * s_stride] * alpha + (float)kk[kd] * delta;
        if (!isfinite(s_new)) s_new = 0.0f;
        yp[kd * s_stride] = (half)s_new;
        acc += s_new * (float)qq[kd];
    }
    if (!isfinite(acc)) acc = 0.0f;
    y[((size_t)batch * (1 + K)) * s_stride + (size_t)h * V + t] = (half)(acc * scale);
}

kernel void gated_deltanet_prefill_f16(
    device const half* q   [[buffer(0)]],
    device const half* k   [[buffer(1)]],
    device const half* v   [[buffer(2)]],
    device const half* g   [[buffer(3)]],
    device const half* b   [[buffer(4)]],
    device const half* s   [[buffer(5)]],
    device       half* y   [[buffer(6)]],
    device      float* st  [[buffer(7)]],
    constant uint& T       [[buffer(8)]],
    constant uint& Hq      [[buffer(9)]],
    constant uint& Hv      [[buffer(10)]],
    constant uint& K       [[buffer(11)]],
    constant uint& V       [[buffer(12)]],
    constant float& scale  [[buffer(13)]],
    uint3 tgp [[threadgroup_position_in_grid]],
    uint t    [[thread_index_in_threadgroup]],
    threadgroup half* kq   [[threadgroup(0)]])
{
    const uint h = tgp.x, batch = tgp.y;
    const uint qk_head = h / (Hv / Hq);
    threadgroup half* kk = kq;
    threadgroup half* qq = kq + K;
    const size_t hv_stride = (size_t)Hv * V;
    device float* col = st + (((size_t)batch * Hv + h) * K) * V + t;

    if (t < V) {
        device const half* sp = s + ((size_t)batch * K) * hv_stride + (size_t)h * V + t;
        for (uint kd = 0; kd < K; ++kd) col[(size_t)kd * V] = (float)sp[kd * hv_stride];
    }

    for (uint step = 0; step < T; ++step) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const size_t qk_base = ((size_t)(batch * T + step) * Hq + qk_head) * K;
        for (uint i = t; i < K; i += V) {
            kk[i] = k[qk_base + i];
            qq[i] = q[qk_base + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t >= V) continue;

        float gate_log = (float)g[(size_t)(batch * T + step) * Hv + h];
        if (!isfinite(gate_log)) gate_log = -20.0f;
        float beta = (float)b[(size_t)(batch * T + step) * Hv + h];
        beta = isfinite(beta) ? clamp(beta, 0.0f, 1.0f) : 0.0f;
        const float alpha = exp(clamp(gate_log, -20.0f, 6.0f));

        float proj = 0.0f;
        for (uint kd = 0; kd < K; ++kd) proj += col[(size_t)kd * V] * (float)kk[kd];
        const float delta = ((float)v[((size_t)(batch * T + step) * Hv + h) * V + t] - alpha * proj) * beta;

        float acc = 0.0f;
        for (uint kd = 0; kd < K; ++kd) {
            float s_new = col[(size_t)kd * V] * alpha + (float)kk[kd] * delta;
            if (!isfinite(s_new)) s_new = 0.0f;
            col[(size_t)kd * V] = s_new;
            acc += s_new * (float)qq[kd];
        }
        if (!isfinite(acc)) acc = 0.0f;
        y[((size_t)(batch * (T + K) + step)) * hv_stride + (size_t)h * V + t] = (half)(acc * scale);
    }

    if (t < V) {
        for (uint kd = 0; kd < K; ++kd)
            y[((size_t)(batch * (T + K) + T + kd)) * hv_stride + (size_t)h * V + t] = (half)col[(size_t)kd * V];
    }
}
