#include <metal_stdlib>
using namespace metal;
#define ROWS 8u

inline float gelu_tanh(float x) {
    float c = 0.7978845608028654f * (x + 0.044715f*x*x*x);
    return 0.5f * x * (1.0f + precise::tanh(c));
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
    float a0=x0+x1,a1=x0-x1,a2=x2+x3,a3=x2-x3;
    x0=a0+a2; x1=a1+a3; x2=a0-a2; x3=a1-a3;
    #pragma clang loop unroll(full)
    for (uint d=1u; d<=16u; d<<=1){
        bool hi=(lane&d)!=0u;
        float p0=simd_shuffle_xor(x0,d),p1=simd_shuffle_xor(x1,d),p2=simd_shuffle_xor(x2,d),p3=simd_shuffle_xor(x3,d);
        x0=hi?p0-x0:x0+p0; x1=hi?p1-x1:x1+p1; x2=hi?p2-x2:x2+p2; x3=hi?p3-x3:x3+p3;
    }
    float s=rsqrt(128.0f);
    zmem[k+0]=x0*s*(float)rsign[k+0]; zmem[k+1]=x1*s*(float)rsign[k+1];
    zmem[k+2]=x2*s*(float)rsign[k+2]; zmem[k+3]=x3*s*(float)rsign[k+3];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    code[b+0]=(half)zmem[perm[k+0]]; code[b+1]=(half)zmem[perm[k+1]];
    code[b+2]=(half)zmem[perm[k+2]]; code[b+3]=(half)zmem[perm[k+3]];
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
    uint tg   [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tl   [[thread_index_in_threadgroup]])
{

    threadgroup float cb[16];
    if (tl<16) cb[tl]=(float)codebook[tl];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 4 rows per simdgroup: independent FMA chains hide load latency
    // (requires N % 4 == 0, guaranteed by quant_fast_eligible)
    uint K = num_groups*gs;
    uint n0 = (tg*ROWS + sgid)*4u;
    if (n0>=N) return;
    float a0=0,a1=0,a2=0,a3=0;
    for (uint base=lane*CQ4_VPL; base<K; base+=32u*CQ4_VPL){
        uint g = base/gs, off = base - g*gs;
        device const half4* cbase=(device const half4*)(code + g*gs + off);
        device const ushort4* p0=(device const ushort4*)(packed + ((size_t)(n0+0)*num_groups+g)*pgb + off/2u);
        device const ushort4* p1=(device const ushort4*)(packed + ((size_t)(n0+1)*num_groups+g)*pgb + off/2u);
        device const ushort4* p2=(device const ushort4*)(packed + ((size_t)(n0+2)*num_groups+g)*pgb + off/2u);
        device const ushort4* p3=(device const ushort4*)(packed + ((size_t)(n0+3)*num_groups+g)*pgb + off/2u);
        ushort4 w0=p0[0], w1=p1[0], w2=p2[0], w3=p3[0];
        float q0=0,q1=0,q2=0,q3=0;
        #pragma clang loop unroll(full)
        for (uint q=0;q<4;++q){
            half4 c=cbase[q];
            ushort wa=w0[q],wb=w1[q],wc=w2[q],wd=w3[q];
            float cx=(float)c.x, cy=(float)c.y, cz=(float)c.z, cw=(float)c.w;
            q0 += cx*cb[wa&0xF] + cy*cb[(wa>>4)&0xF] + cz*cb[(wa>>8)&0xF] + cw*cb[(wa>>12)&0xF];
            q1 += cx*cb[wb&0xF] + cy*cb[(wb>>4)&0xF] + cz*cb[(wb>>8)&0xF] + cw*cb[(wb>>12)&0xF];
            q2 += cx*cb[wc&0xF] + cy*cb[(wc>>4)&0xF] + cz*cb[(wc>>8)&0xF] + cw*cb[(wc>>12)&0xF];
            q3 += cx*cb[wd&0xF] + cy*cb[(wd>>4)&0xF] + cz*cb[(wd>>8)&0xF] + cw*cb[(wd>>12)&0xF];
        }
        a0 += (float)norms[(size_t)(n0+0)*num_groups+g]*q0;
        a1 += (float)norms[(size_t)(n0+1)*num_groups+g]*q1;
        a2 += (float)norms[(size_t)(n0+2)*num_groups+g]*q2;
        a3 += (float)norms[(size_t)(n0+3)*num_groups+g]*q3;
    }
    a0=simd_sum(a0); a1=simd_sum(a1); a2=simd_sum(a2); a3=simd_sum(a3);
    if (lane==0){ y[n0]=(half)a0; y[n0+1]=(half)a1; y[n0+2]=(half)a2; y[n0+3]=(half)a3; }
}

// single row per simdgroup: better threadgroup count for small N
kernel void cq4_gemv_n1(
    device const half*  code     [[buffer(0)]],
    device const uchar* packed   [[buffer(1)]],
    device const half*  codebook [[buffer(2)]],
    device const half*  norms    [[buffer(3)]],
    device       half*  y        [[buffer(4)]],
    constant uint& gs            [[buffer(5)]],
    constant uint& num_groups    [[buffer(6)]],
    constant uint& pgb           [[buffer(7)]],
    constant uint& N             [[buffer(8)]],
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
    acc=simd_sum(acc);
    if (lane==0) y[n]=(half)acc;
}

// 2-bit CQ gemv, 4 rows per simdgroup. Same activation-transform scheme as
// cq4; packed bytes hold 4 consecutive 2-bit indices (little-endian),
// codebook has 4 entries. 16 values per lane per iteration = uchar4.
kernel void cq2_gemv(
    device const half*  code     [[buffer(0)]],
    device const uchar* packed   [[buffer(1)]],
    device const half*  codebook [[buffer(2)]],
    device const half*  norms    [[buffer(3)]],
    device       half*  y        [[buffer(4)]],
    constant uint& gs            [[buffer(5)]],
    constant uint& num_groups    [[buffer(6)]],
    constant uint& pgb           [[buffer(7)]],
    constant uint& N             [[buffer(8)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tl   [[thread_index_in_threadgroup]])
{
    threadgroup float cb[4];
    if (tl<4) cb[tl]=(float)codebook[tl];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint K = num_groups*gs;
    uint n0 = (tg*ROWS + sgid)*4u;
    if (n0>=N) return;
    float a0=0,a1=0,a2=0,a3=0;
    for (uint base=lane*CQ4_VPL; base<K; base+=32u*CQ4_VPL){
        uint g = base/gs, off = base - g*gs;
        device const half4* cbase=(device const half4*)(code + g*gs + off);
        device const uchar4* p0=(device const uchar4*)(packed + ((size_t)(n0+0)*num_groups+g)*pgb + off/4u);
        device const uchar4* p1=(device const uchar4*)(packed + ((size_t)(n0+1)*num_groups+g)*pgb + off/4u);
        device const uchar4* p2=(device const uchar4*)(packed + ((size_t)(n0+2)*num_groups+g)*pgb + off/4u);
        device const uchar4* p3=(device const uchar4*)(packed + ((size_t)(n0+3)*num_groups+g)*pgb + off/4u);
        uchar4 w0=p0[0], w1=p1[0], w2=p2[0], w3=p3[0];
        float q0=0,q1=0,q2=0,q3=0;
        #pragma clang loop unroll(full)
        for (uint q=0;q<4;++q){
            half4 c=cbase[q];
            uchar a=w0[q],b=w1[q],d=w2[q],e=w3[q];
            float cx=(float)c.x, cy=(float)c.y, cz=(float)c.z, cw=(float)c.w;
            q0 += cx*cb[a&3u] + cy*cb[(a>>2)&3u] + cz*cb[(a>>4)&3u] + cw*cb[(a>>6)&3u];
            q1 += cx*cb[b&3u] + cy*cb[(b>>2)&3u] + cz*cb[(b>>4)&3u] + cw*cb[(b>>6)&3u];
            q2 += cx*cb[d&3u] + cy*cb[(d>>2)&3u] + cz*cb[(d>>4)&3u] + cw*cb[(d>>6)&3u];
            q3 += cx*cb[e&3u] + cy*cb[(e>>2)&3u] + cz*cb[(e>>4)&3u] + cw*cb[(e>>6)&3u];
        }
        a0 += (float)norms[(size_t)(n0+0)*num_groups+g]*q0;
        a1 += (float)norms[(size_t)(n0+1)*num_groups+g]*q1;
        a2 += (float)norms[(size_t)(n0+2)*num_groups+g]*q2;
        a3 += (float)norms[(size_t)(n0+3)*num_groups+g]*q3;
    }
    a0=simd_sum(a0); a1=simd_sum(a1); a2=simd_sum(a2); a3=simd_sum(a3);
    if (lane==0){ y[n0]=(half)a0; y[n0+1]=(half)a1; y[n0+2]=(half)a2; y[n0+3]=(half)a3; }
}

// 2-bit CQ gemv, single row per simdgroup (small N)
kernel void cq2_gemv_n1(
    device const half*  code     [[buffer(0)]],
    device const uchar* packed   [[buffer(1)]],
    device const half*  codebook [[buffer(2)]],
    device const half*  norms    [[buffer(3)]],
    device       half*  y        [[buffer(4)]],
    constant uint& gs            [[buffer(5)]],
    constant uint& num_groups    [[buffer(6)]],
    constant uint& pgb           [[buffer(7)]],
    constant uint& N             [[buffer(8)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tl   [[thread_index_in_threadgroup]])
{
    threadgroup float cb[4];
    if (tl<4) cb[tl]=(float)codebook[tl];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint n = tg*ROWS + sgid;
    if (n>=N) return;
    uint K = num_groups*gs;
    float acc=0;
    for (uint base=lane*CQ4_VPL; base<K; base+=32u*CQ4_VPL){
        uint g = base/gs, off = base - g*gs;
        device const half4* cbase=(device const half4*)(code + g*gs + off);
        device const uchar4* pr=(device const uchar4*)(packed + ((size_t)n*num_groups+g)*pgb + off/4u);
        uchar4 w=pr[0];
        float p = 0;
        #pragma clang loop unroll(full)
        for (uint q=0;q<4;++q){
            half4 c=cbase[q];
            uchar b = w[q];
            p += (float)c.x*cb[b&3u] + (float)c.y*cb[(b>>2)&3u]
               + (float)c.z*cb[(b>>4)&3u] + (float)c.w*cb[(b>>6)&3u];
        }
        acc += (float)norms[(size_t)n*num_groups+g]*p;
    }
    acc=simd_sum(acc);
    if (lane==0) y[n]=(half)acc;
}

// plain fp16 gemv: y[n] = dot(x, w[n*K..]) for pretransposed [N,K] weights
kernel void gemv_f16(
    device const half* x [[buffer(0)]],
    device const half* w [[buffer(1)]],
    device       half* y [[buffer(2)]],
    constant uint& K     [[buffer(3)]],
    constant uint& N     [[buffer(4)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint n = tg*8u + sgid;
    if (n>=N) return;
    device const half* row = w + (size_t)n*K;
    float acc=0;
    for (uint k=lane*4u; k+3u<K; k+=128u){
        half4 xv = *(device const half4*)(x+k);
        half4 wv = *(device const half4*)(row+k);
        acc += (float)xv.x*(float)wv.x + (float)xv.y*(float)wv.y
             + (float)xv.z*(float)wv.z + (float)xv.w*(float)wv.w;
    }
    for (uint k=(K & ~3u)+lane; k<K; k+=32u) acc += (float)x[k]*(float)row[k];
    acc=simd_sum(acc);
    if (lane==0) y[n]=(half)acc;
}



// single-dispatch 2-bit matmul: Hadamard transform into threadgroup memory,
// then 4-rows-per-simdgroup gemv (gs==128, K<=4096)
kernel void cq2_gemv_fused(
    device const half*  x        [[buffer(0)]],
    device const uchar* packed   [[buffer(1)]],
    device const half*  codebook [[buffer(2)]],
    device const half*  norms    [[buffer(3)]],
    device       half*  y        [[buffer(4)]],
    constant uint& gs            [[buffer(5)]],
    constant uint& num_groups    [[buffer(6)]],
    constant uint& pgb           [[buffer(7)]],
    constant uint& N             [[buffer(8)]],
    device const half* recip     [[buffer(9)]],
    device const char* lsign     [[buffer(10)]],
    device const char* rsign     [[buffer(11)]],
    device const uint* perm      [[buffer(12)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tl   [[thread_index_in_threadgroup]],
    threadgroup half* codem      [[threadgroup(0)]])
{
    threadgroup float cb[4];
    if (tl<4) cb[tl]=(float)codebook[tl];
    const uint K = num_groups*gs;
    threadgroup float* zs = (threadgroup float*)(codem + K) + sgid*128u;
    for (uint g=sgid; g<num_groups; g+=ROWS) {
        uint b = g*128u + lane*4u;
        uint k = lane*4u;
        float x0=(float)x[b+0]*(float)recip[b+0]*(float)lsign[k+0];
        float x1=(float)x[b+1]*(float)recip[b+1]*(float)lsign[k+1];
        float x2=(float)x[b+2]*(float)recip[b+2]*(float)lsign[k+2];
        float x3=(float)x[b+3]*(float)recip[b+3]*(float)lsign[k+3];
        float a0=x0+x1,a1=x0-x1,a2=x2+x3,a3=x2-x3;
        x0=a0+a2; x1=a1+a3; x2=a0-a2; x3=a1-a3;
        #pragma clang loop unroll(full)
        for (uint d=1u; d<=16u; d<<=1){
            bool hi=(lane&d)!=0u;
            float p0=simd_shuffle_xor(x0,d),p1=simd_shuffle_xor(x1,d),p2=simd_shuffle_xor(x2,d),p3=simd_shuffle_xor(x3,d);
            x0=hi?p0-x0:x0+p0; x1=hi?p1-x1:x1+p1; x2=hi?p2-x2:x2+p2; x3=hi?p3-x3:x3+p3;
        }
        float s=rsqrt(128.0f);
        zs[k+0]=x0*s*(float)rsign[k+0]; zs[k+1]=x1*s*(float)rsign[k+1];
        zs[k+2]=x2*s*(float)rsign[k+2]; zs[k+3]=x3*s*(float)rsign[k+3];
        simdgroup_barrier(mem_flags::mem_threadgroup);
        codem[g*128u+k+0]=(half)zs[perm[k+0]]; codem[g*128u+k+1]=(half)zs[perm[k+1]];
        codem[g*128u+k+2]=(half)zs[perm[k+2]]; codem[g*128u+k+3]=(half)zs[perm[k+3]];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint n0 = (tg*ROWS + sgid)*4u;
    if (n0>=N) return;
    float a0=0,a1=0,a2=0,a3=0;
    for (uint base=lane*CQ4_VPL; base<K; base+=32u*CQ4_VPL){
        uint g = base/gs, off = base - g*gs;
        threadgroup const half4* cbase=(threadgroup const half4*)(codem + g*gs + off);
        device const uchar4* p0=(device const uchar4*)(packed + ((size_t)(n0+0)*num_groups+g)*pgb + off/4u);
        device const uchar4* p1=(device const uchar4*)(packed + ((size_t)(n0+1)*num_groups+g)*pgb + off/4u);
        device const uchar4* p2=(device const uchar4*)(packed + ((size_t)(n0+2)*num_groups+g)*pgb + off/4u);
        device const uchar4* p3=(device const uchar4*)(packed + ((size_t)(n0+3)*num_groups+g)*pgb + off/4u);
        uchar4 w0=p0[0], w1=p1[0], w2=p2[0], w3=p3[0];
        float q0=0,q1=0,q2=0,q3=0;
        #pragma clang loop unroll(full)
        for (uint q=0;q<4;++q){
            half4 c=cbase[q];
            uchar a=w0[q],b=w1[q],d=w2[q],e=w3[q];
            float cx=(float)c.x, cy=(float)c.y, cz=(float)c.z, cw=(float)c.w;
            q0 += cx*cb[a&3u] + cy*cb[(a>>2)&3u] + cz*cb[(a>>4)&3u] + cw*cb[(a>>6)&3u];
            q1 += cx*cb[b&3u] + cy*cb[(b>>2)&3u] + cz*cb[(b>>4)&3u] + cw*cb[(b>>6)&3u];
            q2 += cx*cb[d&3u] + cy*cb[(d>>2)&3u] + cz*cb[(d>>4)&3u] + cw*cb[(d>>6)&3u];
            q3 += cx*cb[e&3u] + cy*cb[(e>>2)&3u] + cz*cb[(e>>4)&3u] + cw*cb[(e>>6)&3u];
        }
        a0 += (float)norms[(size_t)(n0+0)*num_groups+g]*q0;
        a1 += (float)norms[(size_t)(n0+1)*num_groups+g]*q1;
        a2 += (float)norms[(size_t)(n0+2)*num_groups+g]*q2;
        a3 += (float)norms[(size_t)(n0+3)*num_groups+g]*q3;
    }
    a0=simd_sum(a0); a1=simd_sum(a1); a2=simd_sum(a2); a3=simd_sum(a3);
    if (lane==0){ y[n0]=(half)a0; y[n0+1]=(half)a1; y[n0+2]=(half)a2; y[n0+3]=(half)a3; }
}


// out = clip( rms(a)*wa + rms(b)*wb )  — merges two branch norms + add
kernel void rms2_add_clip_f16(device const half* a [[buffer(0)]], device const half* wa [[buffer(1)]],
                              device const half* b [[buffer(2)]], device const half* wb [[buffer(3)]],
                              device half* y [[buffer(4)]],
                              constant uint& dim [[buffer(5)]], constant float& eps [[buffer(6)]],
                              uint t [[thread_position_in_threadgroup]],
                              uint nt [[threads_per_threadgroup]], threadgroup float* red [[threadgroup(0)]]) {
    float pa=0, pb=0;
    for (uint i=t;i<dim;i+=nt){ float v=(float)a[i]; pa+=v*v; float u=(float)b[i]; pb+=u*u; }
    pa=simd_sum(pa); pb=simd_sum(pb);
    uint sg=t/32u, lane=t%32u, nsg=nt/32u;
    if (lane==0){ red[sg]=pa; red[nsg+sg]=pb; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sa=0, sb=0;
    for (uint i=0;i<nsg;++i){ sa+=red[i]; sb+=red[nsg+i]; }
    float ia = 1.0f/sqrt(sa/(float)dim + eps);
    float ib = 1.0f/sqrt(sb/(float)dim + eps);
    for (uint i=t;i<dim;i+=nt){
        float r = (float)(half)((float)a[i]*ia*(float)wa[i]) + (float)(half)((float)b[i]*ib*(float)wb[i]);
        y[i]=(half)clamp(r,-65500.0f,65500.0f);
    }
}

// ---- MoE (M=1 decode) + router support ----------------------------------

kernel void softmax_rows_f16(device const half* in [[buffer(0)]], device half* out [[buffer(1)]],
                             constant uint& dim [[buffer(2)]],
                             uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
                             uint nt [[threads_per_threadgroup]], threadgroup float* red [[threadgroup(0)]]) {
    device const half* x = in + (size_t)row*dim;
    device half* o = out + (size_t)row*dim;
    float mx=-INFINITY; for (uint i=t;i<dim;i+=nt) mx=max(mx,(float)x[i]);
    red[t]=mx; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s=nt/2;s>0;s>>=1){ if(t<s) red[t]=max(red[t],red[t+s]); threadgroup_barrier(mem_flags::mem_threadgroup); }
    float m=red[0]; threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum=0; for (uint i=t;i<dim;i+=nt) sum+=exp((float)x[i]-m);
    red[t]=sum; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s=nt/2;s>0;s>>=1){ if(t<s) red[t]+=red[t+s]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float inv=1.0f/red[0];
    for (uint i=t;i<dim;i+=nt) o[i]=(half)(exp((float)x[i]-m)*inv);
}

// top-k of a single row; output fp32 [k indices][k values], descending
kernel void topk_row_f16(device const half* in [[buffer(0)]], device float* out [[buffer(1)]],
                         constant uint& F [[buffer(2)]], constant uint& k [[buffer(3)]],
                         uint lane [[thread_index_in_simdgroup]]) {
    for (uint j=0;j<k;++j){
        float best=-INFINITY; uint bi=0xFFFFFFFFu;
        for (uint i=lane;i<F;i+=32u){
            bool taken=false;
            for (uint p=0;p<j;++p) if ((uint)out[p]==i) taken=true;
            float v=(float)in[i];
            if (!taken && (v>best || (v==best && i<bi))){ best=v; bi=i; }
        }
        float gb = simd_max(best);
        uint cand = (best==gb) ? bi : 0xFFFFFFFFu;
        uint gi = simd_min(cand);
        if (lane==0){ out[j]=(float)gi; out[k+j]=gb; }
    }
}

// Hadamard transform of the shared hidden vector using each selected expert's
// per-matrix params (concatenated across experts). One threadgroup per
// (group g, slot). x element for (slot,k) = k<k_valid ? x[slot*x_stride+k] : 0.
kernel void cq2_moe_transform(
    device const half*  x        [[buffer(0)]],
    device const float* topk     [[buffer(1)]],
    device const half*  recip_c  [[buffer(2)]],
    device const char*  lsign_c  [[buffer(3)]],
    device const char*  rsign_c  [[buffer(4)]],
    device const uint*  perm_c   [[buffer(5)]],
    device       half*  code     [[buffer(6)]],
    constant uint& K             [[buffer(7)]],
    constant uint& k_valid       [[buffer(8)]],
    constant uint& x_stride      [[buffer(9)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint2 lp  [[thread_position_in_threadgroup]],
    threadgroup float* zmem      [[threadgroup(0)]])
{
    const uint g = tgp.x, slot = tgp.y, lane = lp.x;
    const uint e = (uint)topk[slot];
    device const half* recip = recip_c + (size_t)e*K;
    device const char* lsign = lsign_c + (size_t)e*128u;
    device const char* rsign = rsign_c + (size_t)e*128u;
    device const uint* perm  = perm_c  + (size_t)e*128u;
    device const half* xs = x + (size_t)slot*x_stride;
    uint b = g*128u + lane*4u;
    uint k = lane*4u;
    float xv[4];
    #pragma clang loop unroll(full)
    for (uint j=0;j<4;++j){
        float v = (b+j < k_valid) ? (float)xs[b+j] : 0.0f;
        xv[j] = v*(float)recip[b+j]*(float)lsign[k+j];
    }
    float x0=xv[0],x1=xv[1],x2=xv[2],x3=xv[3];
    float a0=x0+x1,a1=x0-x1,a2=x2+x3,a3=x2-x3;
    x0=a0+a2; x1=a1+a3; x2=a0-a2; x3=a1-a3;
    #pragma clang loop unroll(full)
    for (uint d=1u; d<=16u; d<<=1){
        bool hi=(lane&d)!=0u;
        float p0=simd_shuffle_xor(x0,d),p1=simd_shuffle_xor(x1,d),p2=simd_shuffle_xor(x2,d),p3=simd_shuffle_xor(x3,d);
        x0=hi?p0-x0:x0+p0; x1=hi?p1-x1:x1+p1; x2=hi?p2-x2:x2+p2; x3=hi?p3-x3:x3+p3;
    }
    float s=rsqrt(128.0f);
    zmem[k+0]=x0*s*(float)rsign[k+0]; zmem[k+1]=x1*s*(float)rsign[k+1];
    zmem[k+2]=x2*s*(float)rsign[k+2]; zmem[k+3]=x3*s*(float)rsign[k+3];
    simdgroup_barrier(mem_flags::mem_threadgroup);
    device half* co = code + (size_t)slot*K + g*128u;
    co[k+0]=(half)zmem[perm[k+0]]; co[k+1]=(half)zmem[perm[k+1]];
    co[k+2]=(half)zmem[perm[k+2]]; co[k+3]=(half)zmem[perm[k+3]];
}

// gate/up projections + activation*multiply, 4 rows per simdgroup per slot
kernel void cq2_moe_gemv_up(
    device const half*  code1    [[buffer(0)]],
    device const half*  code3    [[buffer(1)]],
    device const float* topk     [[buffer(2)]],
    device const uchar* pk1      [[buffer(3)]],
    device const half*  nm1      [[buffer(4)]],
    device const half*  cb1      [[buffer(5)]],
    device const uchar* pk3      [[buffer(6)]],
    device const half*  nm3      [[buffer(7)]],
    device const half*  cb3      [[buffer(8)]],
    device       half*  y        [[buffer(9)]],
    constant uint& K             [[buffer(10)]],
    constant uint& N             [[buffer(11)]],
    constant uint& act           [[buffer(12)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tl   [[thread_index_in_threadgroup]])
{
    const uint slot = tgp.y;
    const uint e = (uint)topk[slot];
    const uint ng = K/128u, pgb = 32u;
    threadgroup float c1[4], c3[4];
    if (tl<4){ c1[tl]=(float)cb1[(size_t)e*8u+tl]; c3[tl]=(float)cb3[(size_t)e*8u+tl]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint n0 = (tgp.x*8u + sgid)*4u;
    if (n0>=N) return;
    float g[4]={0,0,0,0}, u[4]={0,0,0,0};
    for (int m=0;m<2;++m){
        device const uchar* pb = (m==0) ? pk1 + (size_t)e*N*ng*pgb : pk3 + (size_t)e*N*ng*pgb;
        device const half*  nb = (m==0) ? nm1 + (size_t)e*N*ng : nm3 + (size_t)e*N*ng;
        device const half*  cd = (m==0) ? code1 + (size_t)slot*K : code3 + (size_t)slot*K;
        threadgroup float* cb = (m==0) ? c1 : c3;
        float a0=0,a1=0,a2=0,a3=0;
        for (uint base=lane*16u; base<K; base+=512u){
            uint gg = base/128u, off = base - gg*128u;
            device const half4* cp=(device const half4*)(cd + gg*128u + off);
            device const uchar4* q0=(device const uchar4*)(pb + ((size_t)(n0+0)*ng+gg)*pgb + off/4u);
            device const uchar4* q1=(device const uchar4*)(pb + ((size_t)(n0+1)*ng+gg)*pgb + off/4u);
            device const uchar4* q2=(device const uchar4*)(pb + ((size_t)(n0+2)*ng+gg)*pgb + off/4u);
            device const uchar4* q3=(device const uchar4*)(pb + ((size_t)(n0+3)*ng+gg)*pgb + off/4u);
            uchar4 w0=q0[0], w1=q1[0], w2=q2[0], w3=q3[0];
            float s0=0,s1=0,s2=0,s3=0;
            #pragma clang loop unroll(full)
            for (uint qq=0;qq<4;++qq){
                half4 c=cp[qq];
                uchar a=w0[qq],b=w1[qq],d=w2[qq],f=w3[qq];
                float cx=(float)c.x, cy=(float)c.y, cz=(float)c.z, cw=(float)c.w;
                s0 += cx*cb[a&3u] + cy*cb[(a>>2)&3u] + cz*cb[(a>>4)&3u] + cw*cb[(a>>6)&3u];
                s1 += cx*cb[b&3u] + cy*cb[(b>>2)&3u] + cz*cb[(b>>4)&3u] + cw*cb[(b>>6)&3u];
                s2 += cx*cb[d&3u] + cy*cb[(d>>2)&3u] + cz*cb[(d>>4)&3u] + cw*cb[(d>>6)&3u];
                s3 += cx*cb[f&3u] + cy*cb[(f>>2)&3u] + cz*cb[(f>>4)&3u] + cw*cb[(f>>6)&3u];
            }
            a0 += (float)nb[(size_t)(n0+0)*ng+gg]*s0;
            a1 += (float)nb[(size_t)(n0+1)*ng+gg]*s1;
            a2 += (float)nb[(size_t)(n0+2)*ng+gg]*s2;
            a3 += (float)nb[(size_t)(n0+3)*ng+gg]*s3;
        }
        a0=simd_sum(a0); a1=simd_sum(a1); a2=simd_sum(a2); a3=simd_sum(a3);
        if (m==0){ g[0]=a0;g[1]=a1;g[2]=a2;g[3]=a3; } else { u[0]=a0;u[1]=a1;u[2]=a2;u[3]=a3; }
    }
    if (lane==0){
        device half* ys = y + (size_t)slot*N;
        for (uint r=0;r<4;++r){
            float gg2 = g[r];
            float aa = (act==1u) ? gelu_tanh(gg2) : gg2/(1.0f+precise::exp(-gg2));
            ys[n0+r]=(half)((float)(half)aa*(float)(half)u[r]);
        }
    }
}

// v2 down: 8 rows per simdgroup (K is short: 768)

// down projection per slot, 4 rows per simdgroup
kernel void cq2_moe_gemv_down(
    device const half*  code2    [[buffer(0)]],
    device const float* topk     [[buffer(1)]],
    device const uchar* pk2      [[buffer(2)]],
    device const half*  nm2      [[buffer(3)]],
    device const half*  cb2      [[buffer(4)]],
    device       half*  y        [[buffer(5)]],
    constant uint& K             [[buffer(6)]],
    constant uint& N             [[buffer(7)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tl   [[thread_index_in_threadgroup]])
{
    const uint slot = tgp.y;
    const uint e = (uint)topk[slot];
    const uint ng = K/128u, pgb = 32u;
    threadgroup float cb[4];
    if (tl<4) cb[tl]=(float)cb2[(size_t)e*8u+tl];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint n0 = (tgp.x*8u + sgid)*8u;
    if (n0>=N) return;
    device const uchar* pbase = pk2 + (size_t)e*N*ng*pgb;
    device const half*  nbase = nm2 + (size_t)e*N*ng;
    device const half*  cd = code2 + (size_t)slot*K;
    float acc[8]={0,0,0,0,0,0,0,0};
    for (uint base=lane*16u; base<K; base+=512u){
        uint gg = base/128u, off = base - gg*128u;
        device const half4* cp=(device const half4*)(cd + gg*128u + off);
        half4 c0=cp[0], c1=cp[1], c2=cp[2], c3=cp[3];
        #pragma clang loop unroll(full)
        for (uint r=0;r<8;++r){
            device const uchar4* qq=(device const uchar4*)(pbase + ((size_t)(n0+r)*ng+gg)*pgb + off/4u);
            uchar4 w=qq[0];
            float s=0;
            {
                uchar a=w.x; s += (float)c0.x*cb[a&3u] + (float)c0.y*cb[(a>>2)&3u] + (float)c0.z*cb[(a>>4)&3u] + (float)c0.w*cb[(a>>6)&3u];
                a=w.y; s += (float)c1.x*cb[a&3u] + (float)c1.y*cb[(a>>2)&3u] + (float)c1.z*cb[(a>>4)&3u] + (float)c1.w*cb[(a>>6)&3u];
                a=w.z; s += (float)c2.x*cb[a&3u] + (float)c2.y*cb[(a>>2)&3u] + (float)c2.z*cb[(a>>4)&3u] + (float)c2.w*cb[(a>>6)&3u];
                a=w.w; s += (float)c3.x*cb[a&3u] + (float)c3.y*cb[(a>>2)&3u] + (float)c3.z*cb[(a>>4)&3u] + (float)c3.w*cb[(a>>6)&3u];
            }
            acc[r] += (float)nbase[(size_t)(n0+r)*ng+gg]*s;
        }
    }
    #pragma clang loop unroll(full)
    for (uint r=0;r<8;++r) acc[r]=simd_sum(acc[r]);
    if (lane==0){
        device half* ys = y + (size_t)slot*N;
        for (uint r=0;r<8;++r) ys[n0+r]=(half)acc[r];
    }
}

// v3 up: sequential w1/w3, 8 rows per simdgroup

// out[d] = sum_slots route_weight(slot) * eout[slot*D+d]
kernel void moe_accum_f16(
    device const half*  eout   [[buffer(0)]],
    device const half*  probs  [[buffer(1)]],
    device const float* topk   [[buffer(2)]],
    device       half*  out    [[buffer(3)]],
    constant uint& D           [[buffer(4)]],
    constant uint& k           [[buffer(5)]],
    constant uint& normalize   [[buffer(6)]],
    constant float& eps        [[buffer(7)]],
    constant float& scaling    [[buffer(8)]],
    uint d [[thread_position_in_grid]])
{
    if (d>=D) return;
    float denom = 1.0f;
    if (normalize != 0u){
        float s=0; for (uint j=0;j<k;++j) s += (float)probs[(uint)topk[j]];
        denom = s + eps;
    }
    float acc=0;
    for (uint j=0;j<k;++j){
        float p = (float)probs[(uint)topk[j]];
        if (p <= 0.0f) continue;
        float w = p/denom*scaling;
        acc += w*(float)eout[(size_t)j*D + d];
    }
    out[d]=(half)acc;
}

kernel void cq4_gemv_fused(
    device const half*  x        [[buffer(0)]],
    device const uchar* packed   [[buffer(1)]],
    device const half*  codebook [[buffer(2)]],
    device const half*  norms    [[buffer(3)]],
    device       half*  y        [[buffer(4)]],
    constant uint& gs            [[buffer(5)]],
    constant uint& num_groups    [[buffer(6)]],
    constant uint& pgb           [[buffer(7)]],
    constant uint& N             [[buffer(8)]],
    device const half* recip     [[buffer(9)]],
    device const char* lsign     [[buffer(10)]],
    device const char* rsign     [[buffer(11)]],
    device const uint* perm      [[buffer(12)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tl   [[thread_index_in_threadgroup]],
    threadgroup half* codem      [[threadgroup(0)]])
{
    threadgroup float cb[16];
    if (tl<16) cb[tl]=(float)codebook[tl];
    const uint K = num_groups*gs;
    threadgroup float* zs = (threadgroup float*)(codem + K) + sgid*128u;
    for (uint g=sgid; g<num_groups; g+=ROWS) {
        uint b = g*128u + lane*4u;
        uint k = lane*4u;
        float x0=(float)x[b+0]*(float)recip[b+0]*(float)lsign[k+0];
        float x1=(float)x[b+1]*(float)recip[b+1]*(float)lsign[k+1];
        float x2=(float)x[b+2]*(float)recip[b+2]*(float)lsign[k+2];
        float x3=(float)x[b+3]*(float)recip[b+3]*(float)lsign[k+3];
        float a0=x0+x1,a1=x0-x1,a2=x2+x3,a3=x2-x3;
        x0=a0+a2; x1=a1+a3; x2=a0-a2; x3=a1-a3;
        #pragma clang loop unroll(full)
        for (uint d=1u; d<=16u; d<<=1){
            bool hi=(lane&d)!=0u;
            float p0=simd_shuffle_xor(x0,d),p1=simd_shuffle_xor(x1,d),p2=simd_shuffle_xor(x2,d),p3=simd_shuffle_xor(x3,d);
            x0=hi?p0-x0:x0+p0; x1=hi?p1-x1:x1+p1; x2=hi?p2-x2:x2+p2; x3=hi?p3-x3:x3+p3;
        }
        float s=rsqrt(128.0f);
        zs[k+0]=x0*s*(float)rsign[k+0]; zs[k+1]=x1*s*(float)rsign[k+1];
        zs[k+2]=x2*s*(float)rsign[k+2]; zs[k+3]=x3*s*(float)rsign[k+3];
        simdgroup_barrier(mem_flags::mem_threadgroup);
        codem[g*128u+k+0]=(half)zs[perm[k+0]]; codem[g*128u+k+1]=(half)zs[perm[k+1]];
        codem[g*128u+k+2]=(half)zs[perm[k+2]]; codem[g*128u+k+3]=(half)zs[perm[k+3]];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint n0 = (tg*ROWS + sgid)*4u;
    if (n0>=N) return;
    float a0=0,a1=0,a2=0,a3=0;
    for (uint base=lane*CQ4_VPL; base<K; base+=32u*CQ4_VPL){
        uint g = base/gs, off = base - g*gs;
        threadgroup const half4* cbase=(threadgroup const half4*)(codem + g*gs + off);
        device const ushort4* p0=(device const ushort4*)(packed + ((size_t)(n0+0)*num_groups+g)*pgb + off/2u);
        device const ushort4* p1=(device const ushort4*)(packed + ((size_t)(n0+1)*num_groups+g)*pgb + off/2u);
        device const ushort4* p2=(device const ushort4*)(packed + ((size_t)(n0+2)*num_groups+g)*pgb + off/2u);
        device const ushort4* p3=(device const ushort4*)(packed + ((size_t)(n0+3)*num_groups+g)*pgb + off/2u);
        ushort4 w0=p0[0], w1=p1[0], w2=p2[0], w3=p3[0];
        float q0=0,q1=0,q2=0,q3=0;
        #pragma clang loop unroll(full)
        for (uint q=0;q<4;++q){
            half4 c=cbase[q];
            ushort wa=w0[q],wb=w1[q],wc=w2[q],wd=w3[q];
            float cx=(float)c.x, cy=(float)c.y, cz=(float)c.z, cw=(float)c.w;
            q0 += cx*cb[wa&0xF] + cy*cb[(wa>>4)&0xF] + cz*cb[(wa>>8)&0xF] + cw*cb[(wa>>12)&0xF];
            q1 += cx*cb[wb&0xF] + cy*cb[(wb>>4)&0xF] + cz*cb[(wb>>8)&0xF] + cw*cb[(wb>>12)&0xF];
            q2 += cx*cb[wc&0xF] + cy*cb[(wc>>4)&0xF] + cz*cb[(wc>>8)&0xF] + cw*cb[(wc>>12)&0xF];
            q3 += cx*cb[wd&0xF] + cy*cb[(wd>>4)&0xF] + cz*cb[(wd>>8)&0xF] + cw*cb[(wd>>12)&0xF];
        }
        a0 += (float)norms[(size_t)(n0+0)*num_groups+g]*q0;
        a1 += (float)norms[(size_t)(n0+1)*num_groups+g]*q1;
        a2 += (float)norms[(size_t)(n0+2)*num_groups+g]*q2;
        a3 += (float)norms[(size_t)(n0+3)*num_groups+g]*q3;
    }
    a0=simd_sum(a0); a1=simd_sum(a1); a2=simd_sum(a2); a3=simd_sum(a3);
    if (lane==0){ y[n0]=(half)a0; y[n0+1]=(half)a1; y[n0+2]=(half)a2; y[n0+3]=(half)a3; }
}

// Transform for up-to-3 matrices sharing one input, with the producing
// residual+norm chain fused in as a prologue:
//   h = clamp(res + rms(in)*w1)*out_scale   (written to h_out by tg (0,0))
//   xn = rms(h)*w2                          (kept in threadgroup memory)
// Each threadgroup redundantly computes the prologue (one simdgroup, cheap),
// then Hadamard-transforms its (group, matrix) slice of xn.
kernel void cq4_transform_multi_resid(
    device const half*  in       [[buffer(0)]],
    device const half*  w1       [[buffer(1)]],
    device const half*  res      [[buffer(2)]],
    device const half*  w2       [[buffer(3)]],
    device const half*  recip0   [[buffer(4)]],
    device const half*  recip1   [[buffer(5)]],
    device const half*  recip2   [[buffer(6)]],
    device const char*  lsign0   [[buffer(7)]],
    device const char*  lsign1   [[buffer(8)]],
    device const char*  lsign2   [[buffer(9)]],
    device const char*  rsign0   [[buffer(10)]],
    device const char*  rsign1   [[buffer(11)]],
    device const char*  rsign2   [[buffer(12)]],
    device const uint*  perm0    [[buffer(13)]],
    device const uint*  perm1    [[buffer(14)]],
    device const uint*  perm2    [[buffer(15)]],
    device       half*  code0    [[buffer(16)]],
    device       half*  code1    [[buffer(17)]],
    device       half*  code2    [[buffer(18)]],
    device       half*  h_out    [[buffer(19)]],
    constant uint&  K            [[buffer(20)]],
    constant float& out_scale    [[buffer(21)]],
    constant float& eps          [[buffer(22)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint2 lp  [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    threadgroup half* xn         [[threadgroup(0)]])
{
    const uint m = tgp.y, tl = lp.x;
    threadgroup half* hm = xn + K;
    threadgroup float* zsb = (threadgroup float*)(hm + K);   // 8*128 floats
    threadgroup float* red = zsb + 8u*128u;                  // 8 floats
    // ---- prologue with all 256 threads ----
    {
        float p=0; for (uint i=tl;i<K;i+=256u){ float v=(float)in[i]; p+=v*v; }
        p = simd_sum(p);
        if (lane==0) red[sgid]=p;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float s2=0; for (uint i=0;i<8u;++i) s2+=red[i];
        float inv1 = 1.0f/sqrt(s2/(float)K + eps);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float p2=0;
        for (uint i=tl;i<K;i+=256u){
            float rr=(float)res[i]+(float)(half)((float)in[i]*inv1*(float)w1[i]);
            half hv=(half)((float)(half)clamp(rr,-65500.0f,65500.0f)*out_scale);
            hm[i]=hv;
            float v=(float)hv; p2+=v*v;
        }
        p2 = simd_sum(p2);
        if (lane==0) red[sgid]=p2;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float s3=0; for (uint i=0;i<8u;++i) s3+=red[i];
        float inv2 = 1.0f/sqrt(s3/(float)K + eps);
        for (uint i=tl;i<K;i+=256u) xn[i]=(half)((float)hm[i]*inv2*(float)w2[i]);
        if (tgp.x==0u && m==0u) for (uint i=tl;i<K;i+=256u) h_out[i]=hm[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device const half* recip = (m==0u)?recip0:(m==1u)?recip1:recip2;
    device const char* lsign = (m==0u)?lsign0:(m==1u)?lsign1:lsign2;
    device const char* rsign = (m==0u)?rsign0:(m==1u)?rsign1:rsign2;
    device const uint* perm  = (m==0u)?perm0:(m==1u)?perm1:perm2;
    device       half* code  = (m==0u)?code0:(m==1u)?code1:code2;

    // each simdgroup transforms one 128-group
    uint g = tgp.x*8u + sgid;
    if (g >= K/128u) return;
    threadgroup float* zs = zsb + sgid*128u;
    uint b = g*128u + lane*4u;
    uint k = lane*4u;
    float x0=(float)xn[b+0]*(float)recip[b+0]*(float)lsign[k+0];
    float x1=(float)xn[b+1]*(float)recip[b+1]*(float)lsign[k+1];
    float x2=(float)xn[b+2]*(float)recip[b+2]*(float)lsign[k+2];
    float x3=(float)xn[b+3]*(float)recip[b+3]*(float)lsign[k+3];
    float a0=x0+x1,a1=x0-x1,a2=x2+x3,a3=x2-x3;
    x0=a0+a2; x1=a1+a3; x2=a0-a2; x3=a1-a3;
    #pragma clang loop unroll(full)
    for (uint d=1u; d<=16u; d<<=1){
        bool hi=(lane&d)!=0u;
        float p0=simd_shuffle_xor(x0,d),p1=simd_shuffle_xor(x1,d),p2s=simd_shuffle_xor(x2,d),p3=simd_shuffle_xor(x3,d);
        x0=hi?p0-x0:x0+p0; x1=hi?p1-x1:x1+p1; x2=hi?p2s-x2:x2+p2s; x3=hi?p3-x3:x3+p3;
    }
    float s=rsqrt(128.0f);
    zs[k+0]=x0*s*(float)rsign[k+0]; zs[k+1]=x1*s*(float)rsign[k+1];
    zs[k+2]=x2*s*(float)rsign[k+2]; zs[k+3]=x3*s*(float)rsign[k+3];
    simdgroup_barrier(mem_flags::mem_threadgroup);
    code[b+0]=(half)zs[perm[k+0]]; code[b+1]=(half)zs[perm[k+1]];
    code[b+2]=(half)zs[perm[k+2]]; code[b+3]=(half)zs[perm[k+3]];
}

// pd projection (single-dispatch transform+gemv) with its producing
// residual fused in: x = h2 = clamp(res + rms(in)*w1); tg 0 writes h2.
kernel void cq4_gemv_fused_resid(
    device const half*  in       [[buffer(0)]],
    device const uchar* packed   [[buffer(1)]],
    device const half*  codebook [[buffer(2)]],
    device const half*  norms    [[buffer(3)]],
    device       half*  y        [[buffer(4)]],
    constant uint& gs            [[buffer(5)]],
    constant uint& num_groups    [[buffer(6)]],
    constant uint& pgb           [[buffer(7)]],
    constant uint& N             [[buffer(8)]],
    device const half* recip     [[buffer(9)]],
    device const char* lsign     [[buffer(10)]],
    device const char* rsign     [[buffer(11)]],
    device const uint* perm      [[buffer(12)]],
    device const half* w1        [[buffer(13)]],
    device const half* res       [[buffer(14)]],
    device       half* h_out     [[buffer(15)]],
    constant float& eps          [[buffer(16)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tl   [[thread_index_in_threadgroup]],
    threadgroup half* codem      [[threadgroup(0)]])
{
    threadgroup float cb[16];
    if (tl<16) cb[tl]=(float)codebook[tl];
    const uint K = num_groups*gs;
    threadgroup half* hm = codem + K;                          // K halves
    threadgroup float* zsb = (threadgroup float*)(hm + K);     // ROWS*128 floats
    threadgroup float* red = zsb + ROWS*128u;                  // ROWS floats

    // prologue: h2 into tg memory (all threads)
    {
        float p=0; for (uint i=tl;i<K;i+=256u){ float v=(float)in[i]; p+=v*v; }
        p = simd_sum(p);
        if (lane==0) red[sgid]=p;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float s2=0; for (uint i=0;i<ROWS;++i) s2+=red[i];
        float inv1 = 1.0f/sqrt(s2/(float)K + eps);
        for (uint i=tl;i<K;i+=256u){
            float rr=(float)res[i]+(float)(half)((float)in[i]*inv1*(float)w1[i]);
            hm[i]=(half)clamp(rr,-65500.0f,65500.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tg==0u) for (uint i=tl;i<K;i+=256u) h_out[i]=hm[i];
    }
    // transform from tg memory
    threadgroup float* zs = zsb + sgid*128u;
    for (uint g=sgid; g<num_groups; g+=ROWS) {
        uint b = g*128u + lane*4u;
        uint k = lane*4u;
        float x0=(float)hm[b+0]*(float)recip[b+0]*(float)lsign[k+0];
        float x1=(float)hm[b+1]*(float)recip[b+1]*(float)lsign[k+1];
        float x2=(float)hm[b+2]*(float)recip[b+2]*(float)lsign[k+2];
        float x3=(float)hm[b+3]*(float)recip[b+3]*(float)lsign[k+3];
        float a0=x0+x1,a1=x0-x1,a2=x2+x3,a3=x2-x3;
        x0=a0+a2; x1=a1+a3; x2=a0-a2; x3=a1-a3;
        #pragma clang loop unroll(full)
        for (uint d=1u; d<=16u; d<<=1){
            bool hi=(lane&d)!=0u;
            float p0=simd_shuffle_xor(x0,d),p1=simd_shuffle_xor(x1,d),p2=simd_shuffle_xor(x2,d),p3=simd_shuffle_xor(x3,d);
            x0=hi?p0-x0:x0+p0; x1=hi?p1-x1:x1+p1; x2=hi?p2-x2:x2+p2; x3=hi?p3-x3:x3+p3;
        }
        float s=rsqrt(128.0f);
        zs[k+0]=x0*s*(float)rsign[k+0]; zs[k+1]=x1*s*(float)rsign[k+1];
        zs[k+2]=x2*s*(float)rsign[k+2]; zs[k+3]=x3*s*(float)rsign[k+3];
        simdgroup_barrier(mem_flags::mem_threadgroup);
        codem[g*128u+k+0]=(half)zs[perm[k+0]]; codem[g*128u+k+1]=(half)zs[perm[k+1]];
        codem[g*128u+k+2]=(half)zs[perm[k+2]]; codem[g*128u+k+3]=(half)zs[perm[k+3]];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint n0 = (tg*ROWS + sgid)*4u;
    if (n0>=N) return;
    float a0=0,a1=0,a2=0,a3=0;
    for (uint base=lane*CQ4_VPL; base<K; base+=32u*CQ4_VPL){
        uint g = base/gs, off = base - g*gs;
        threadgroup const half4* cbase=(threadgroup const half4*)(codem + g*gs + off);
        device const ushort4* p0=(device const ushort4*)(packed + ((size_t)(n0+0)*num_groups+g)*pgb + off/2u);
        device const ushort4* p1=(device const ushort4*)(packed + ((size_t)(n0+1)*num_groups+g)*pgb + off/2u);
        device const ushort4* p2=(device const ushort4*)(packed + ((size_t)(n0+2)*num_groups+g)*pgb + off/2u);
        device const ushort4* p3=(device const ushort4*)(packed + ((size_t)(n0+3)*num_groups+g)*pgb + off/2u);
        ushort4 w0=p0[0], w1v=p1[0], w2v=p2[0], w3v=p3[0];
        float q0=0,q1=0,q2=0,q3=0;
        #pragma clang loop unroll(full)
        for (uint q=0;q<4;++q){
            half4 c=cbase[q];
            ushort wa=w0[q],wb=w1v[q],wc=w2v[q],wd=w3v[q];
            float cx=(float)c.x, cy=(float)c.y, cz=(float)c.z, cw=(float)c.w;
            q0 += cx*cb[wa&0xF] + cy*cb[(wa>>4)&0xF] + cz*cb[(wa>>8)&0xF] + cw*cb[(wa>>12)&0xF];
            q1 += cx*cb[wb&0xF] + cy*cb[(wb>>4)&0xF] + cz*cb[(wb>>8)&0xF] + cw*cb[(wb>>12)&0xF];
            q2 += cx*cb[wc&0xF] + cy*cb[(wc>>4)&0xF] + cz*cb[(wc>>8)&0xF] + cw*cb[(wc>>12)&0xF];
            q3 += cx*cb[wd&0xF] + cy*cb[(wd>>4)&0xF] + cz*cb[(wd>>8)&0xF] + cw*cb[(wd>>12)&0xF];
        }
        a0 += (float)norms[(size_t)(n0+0)*num_groups+g]*q0;
        a1 += (float)norms[(size_t)(n0+1)*num_groups+g]*q1;
        a2 += (float)norms[(size_t)(n0+2)*num_groups+g]*q2;
        a3 += (float)norms[(size_t)(n0+3)*num_groups+g]*q3;
    }
    a0=simd_sum(a0); a1=simd_sum(a1); a2=simd_sum(a2); a3=simd_sum(a3);
    if (lane==0){ y[n0]=(half)a0; y[n0+1]=(half)a1; y[n0+2]=(half)a2; y[n0+3]=(half)a3; }
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

#define CQ4_MT 16u
kernel void cq4_gemm(
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
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tl   [[thread_index_in_threadgroup]])
{
    threadgroup float cb[16];
    if (tl<16) cb[tl]=(float)codebook[tl];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint n = tg.x*ROWS + sgid;
    if (n>=N) return;
    uint m0 = tg.y*CQ4_MT;
    uint K = num_groups*gs;
    float acc[CQ4_MT];
    for (uint mi=0; mi<CQ4_MT; ++mi) acc[mi]=0;
    for (uint base=lane*CQ4_VPL; base<K; base+=32u*CQ4_VPL){
        uint g = base/gs, off = base - g*gs;
        device const ushort4* pr=(device const ushort4*)(packed + ((size_t)n*num_groups+g)*pgb + off/2u);
        ushort4 w=pr[0];
        float nrm = (float)norms[(size_t)n*num_groups+g];
        float wv[16];
        #pragma clang loop unroll(full)
        for (uint q=0;q<4;++q){ ushort ww=w[q]; wv[q*4+0]=cb[ww&0xF]; wv[q*4+1]=cb[(ww>>4)&0xF];
                                wv[q*4+2]=cb[(ww>>8)&0xF]; wv[q*4+3]=cb[(ww>>12)&0xF]; }
        for (uint mi=0; mi<CQ4_MT && m0+mi<M; ++mi){
            device const half4* cbase=(device const half4*)(code + (size_t)(m0+mi)*K + g*gs + off);
            float p=0;
            #pragma clang loop unroll(full)
            for (uint q=0;q<4;++q){ half4 c=cbase[q]; p += (float)c.x*wv[q*4+0]+(float)c.y*wv[q*4+1]
                                                         +(float)c.z*wv[q*4+2]+(float)c.w*wv[q*4+3]; }
            acc[mi] += nrm*p;
        }
    }
    for (uint mi=0; mi<CQ4_MT && m0+mi<M; ++mi){
        float a = simd_sum(acc[mi]);
        if (lane==0) y[(size_t)(m0+mi)*N + n] = (half)a;
    }
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
    const uint ldA = 40u, ldB = 40u;
    threadgroup half As0[64u*40u];
    threadgroup half Bs0[32u*40u];
    threadgroup float Cs[64u*32u];
    threadgroup float cb[16];
    threadgroup half* As1 = (threadgroup half*)Cs;
    threadgroup half* Bs1 = As1 + 64u*40u;
    if (tl<16u) cb[tl]=(float)codebook[tl];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint K = num_groups*gs;
    uint m0 = tg.y*64u, n0 = tg.x*32u;
    uint rb = sg*16u;
    simdgroup_matrix<float,8,8> C00=make_filled_simdgroup_matrix<float,8,8>(0.f);
    simdgroup_matrix<float,8,8> C01=make_filled_simdgroup_matrix<float,8,8>(0.f);
    simdgroup_matrix<float,8,8> C02=make_filled_simdgroup_matrix<float,8,8>(0.f);
    simdgroup_matrix<float,8,8> C03=make_filled_simdgroup_matrix<float,8,8>(0.f);
    simdgroup_matrix<float,8,8> C10=make_filled_simdgroup_matrix<float,8,8>(0.f);
    simdgroup_matrix<float,8,8> C11=make_filled_simdgroup_matrix<float,8,8>(0.f);
    simdgroup_matrix<float,8,8> C12=make_filled_simdgroup_matrix<float,8,8>(0.f);
    simdgroup_matrix<float,8,8> C13=make_filled_simdgroup_matrix<float,8,8>(0.f);
    threadgroup half* Ac=As0; threadgroup half* Bc=Bs0;
    threadgroup half* An=As1; threadgroup half* Bn=Bs1;
    {
        uint g = 0u, e0 = 0u;
        for (uint i=tl; i<64u*32u; i+=128u){
            uint r=i>>5, k=i&31u;
            Ac[r*ldA+k] = (m0+r<M) ? code[(size_t)(m0+r)*K + k] : (half)0;
        }
        for (uint i=tl; i<32u*32u; i+=128u){
            uint k=i>>5, col=i&31u, n=n0+col;
            float v=0.f;
            if (n<N){
                uint e=e0+k;
                uchar by=packed[((size_t)n*num_groups+g)*pgb + (e>>1)];
                uint nib=(e&1u)?(uint)(by>>4):(uint)(by&0xF);
                v=(float)norms[(size_t)n*num_groups+g]*cb[nib];
            }
            Bc[k*ldB+col]=(half)v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k0=0; k0<K; k0+=32u){
        uint k1 = k0+32u;
        if (k1<K){
            uint g = k1/gs, e0 = k1 - g*gs;
            for (uint i=tl; i<64u*32u; i+=128u){
                uint r=i>>5, k=i&31u;
                An[r*ldA+k] = (m0+r<M) ? code[(size_t)(m0+r)*K + k1+k] : (half)0;
            }
            for (uint i=tl; i<32u*32u; i+=128u){
                uint k=i>>5, col=i&31u, n=n0+col;
                float v=0.f;
                if (n<N){
                    uint e=e0+k;
                    uchar by=packed[((size_t)n*num_groups+g)*pgb + (e>>1)];
                    uint nib=(e&1u)?(uint)(by>>4):(uint)(by&0xF);
                    v=(float)norms[(size_t)n*num_groups+g]*cb[nib];
                }
                Bn[k*ldB+col]=(half)v;
            }
        }
        for (uint kk=0; kk<32u; kk+=8u){
            simdgroup_matrix<half,8,8> A0,A1,B0,B1,B2,B3;
            simdgroup_load(A0,&Ac[(rb)*ldA+kk],ldA);
            simdgroup_load(A1,&Ac[(rb+8u)*ldA+kk],ldA);
            simdgroup_load(B0,&Bc[kk*ldB+0u],ldB);
            simdgroup_load(B1,&Bc[kk*ldB+8u],ldB);
            simdgroup_load(B2,&Bc[kk*ldB+16u],ldB);
            simdgroup_load(B3,&Bc[kk*ldB+24u],ldB);
            simdgroup_multiply_accumulate(C00,A0,B0,C00);
            simdgroup_multiply_accumulate(C01,A0,B1,C01);
            simdgroup_multiply_accumulate(C02,A0,B2,C02);
            simdgroup_multiply_accumulate(C03,A0,B3,C03);
            simdgroup_multiply_accumulate(C10,A1,B0,C10);
            simdgroup_multiply_accumulate(C11,A1,B1,C11);
            simdgroup_multiply_accumulate(C12,A1,B2,C12);
            simdgroup_multiply_accumulate(C13,A1,B3,C13);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup half* t;
        t=Ac; Ac=An; An=t;
        t=Bc; Bc=Bn; Bn=t;
    }
    simdgroup_store(C00,&Cs[(rb)*32u + 0u],32u);
    simdgroup_store(C01,&Cs[(rb)*32u + 8u],32u);
    simdgroup_store(C02,&Cs[(rb)*32u + 16u],32u);
    simdgroup_store(C03,&Cs[(rb)*32u + 24u],32u);
    simdgroup_store(C10,&Cs[(rb+8u)*32u + 0u],32u);
    simdgroup_store(C11,&Cs[(rb+8u)*32u + 8u],32u);
    simdgroup_store(C12,&Cs[(rb+8u)*32u + 16u],32u);
    simdgroup_store(C13,&Cs[(rb+8u)*32u + 24u],32u);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i=tl; i<64u*32u; i+=128u){
        uint r=i>>5, c=i&31u;
        if (m0+r<M && n0+c<N) y[(size_t)(m0+r)*N + n0+c]=Cs[r*32u+c];
    }
}

kernel void lmhead_rotate(device const half* act [[buffer(0)]], device const half* recip [[buffer(1)]],
                          device const half* rotation [[buffer(2)]], device half* code [[buffer(3)]],
                          constant uint& K [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i>=K) return;
    float acc = 0;
    for (uint k=0; k<K; ++k) acc += (float)act[k]*(float)recip[k]*(float)rotation[(size_t)k*K + i];
    code[i] = (half)acc;
}

kernel void emb_ortho(device const uchar* packed [[buffer(0)]],
                      device const half* codebook [[buffer(1)]],
                      device const half* norms [[buffer(2)]],
                      device const half* recip [[buffer(3)]],
                      device const half* rotation [[buffer(4)]],
                      device half* out [[buffer(5)]],
                      constant uint& K [[buffer(6)]], constant uint& row [[buffer(7)]],
                      uint t [[thread_position_in_threadgroup]], uint T [[threads_per_threadgroup]],
                      threadgroup float* dq [[threadgroup(0)]]) {
    uint pgb = K/2u;
    device const uchar* prow = packed + (size_t)row*pgb;
    for (uint i=t; i<K; i+=T) { uchar b=prow[i>>1]; uint idx=(i&1u)?(uint)(b>>4):(uint)(b&0xF); dq[i]=(float)codebook[idx]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float nrm = (float)norms[row];
    for (uint j=t; j<K; j+=T) {
        float acc=0;
        for (uint i=0;i<K;++i) acc += dq[i]*(float)rotation[(size_t)j*K + i];
        out[j] = (half)(acc * nrm * (float)recip[j]);
    }
}

// Parallel replacement for emb_ortho: stage 1 dequantizes the embedding row,
// stage 2 computes out[j] = dot(dq, rotation_row_j) * norm * recip[j] * scale
// with one simdgroup per output element.
kernel void emb_ortho_dequant(device const uchar* packed [[buffer(0)]],
                              device const half* codebook [[buffer(1)]],
                              device half* dq [[buffer(2)]],
                              constant uint& K [[buffer(3)]], constant uint& row [[buffer(4)]],
                              uint i [[thread_position_in_grid]]) {
    if (i>=K) return;
    device const uchar* prow = packed + (size_t)row*(K/2u);
    uchar b=prow[i>>1]; uint idx=(i&1u)?(uint)(b>>4):(uint)(b&0xF);
    dq[i]=codebook[idx];
}

kernel void rot_rowdot(device const half* dq [[buffer(0)]],
                       device const half* rotation [[buffer(1)]],
                       device const half* recip [[buffer(2)]],
                       device half* out [[buffer(3)]],
                       constant uint& K [[buffer(4)]], constant float& scale [[buffer(5)]],
                       uint tg [[threadgroup_position_in_grid]],
                       uint sgid [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
    uint j = tg*8u + sgid;
    if (j>=K) return;
    device const half* rrow = rotation + (size_t)j*K;
    float acc=0;
    for (uint i=lane*4u; i<K; i+=128u) {
        half4 d = *(device const half4*)(dq+i);
        half4 r = *(device const half4*)(rrow+i);
        acc += (float)d.x*(float)r.x + (float)d.y*(float)r.y + (float)d.z*(float)r.z + (float)d.w*(float)r.w;
    }
    acc = simd_sum(acc);
    if (lane==0) out[j]=(half)(acc*(float)recip[j]*scale);
}

// Parallel replacement for lmhead_rotate: code[i] = sum_k act[k]*recip[k]*rot[k*K+i],
// split over k-chunks; combine kernel sums the partials.
kernel void rot_coldot(device const half* act [[buffer(0)]], device const half* recip [[buffer(1)]],
                       device const half* rotation [[buffer(2)]], device float* partial [[buffer(3)]],
                       constant uint& K [[buffer(4)]], constant uint& nchunk [[buffer(5)]],
                       uint2 tg [[threadgroup_position_in_grid]],
                       uint2 tp [[thread_position_in_threadgroup]]) {
    uint i = tg.x*256u + tp.x;
    if (i>=K) return;
    uint chunk = tg.y;
    uint k0 = chunk*(K/nchunk), k1 = (chunk+1u==nchunk)?K:(k0+K/nchunk);
    float acc=0;
    for (uint k=k0;k<k1;++k) acc += (float)act[k]*(float)recip[k]*(float)rotation[(size_t)k*K+i];
    partial[(size_t)chunk*K+i]=acc;
}
kernel void rot_coldot_combine(device const float* partial [[buffer(0)]], device half* code [[buffer(1)]],
                               constant uint& K [[buffer(2)]], constant uint& nchunk [[buffer(3)]],
                               uint i [[thread_position_in_grid]]) {
    if (i>=K) return;
    float acc=0;
    for (uint c=0;c<nchunk;++c) acc += partial[(size_t)c*K+i];
    code[i]=(half)acc;
}

kernel void softcap_f16(device const half* in [[buffer(0)]], device half* y [[buffer(1)]],
                        constant uint& n [[buffer(2)]], constant float& cap [[buffer(3)]],
                        uint i [[thread_position_in_grid]]) {
    if (i>=n) return;
    y[i]=(half)(cap*precise::tanh((float)in[i]/cap));
}

// rms-norm per head followed by rope, one threadgroup per head
kernel void rms_rope_f16(device const half* in [[buffer(0)]], device const half* w [[buffer(1)]],
                         device const half* cs [[buffer(2)]], device const half* sn [[buffer(3)]],
                         device half* y [[buffer(4)]],
                         constant uint& hd [[buffer(5)]], constant float& eps [[buffer(6)]],
                         uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
                         uint nt [[threads_per_threadgroup]], threadgroup float* red [[threadgroup(0)]]) {
    device const half* x = in + (size_t)row*hd;
    device half* o = y + (size_t)row*hd;
    float partial=0; for (uint i=t;i<hd;i+=nt){ float v=(float)x[i]; partial+=v*v; }
    red[t]=partial; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s=nt/2; s>0; s>>=1){ if (t<s) red[t]+=red[t+s]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float inv = 1.0f/sqrt(red[0]/(float)hd + eps);
    uint hh=hd/2u;
    for (uint d=t;d<hd;d+=nt){
        float xn = (float)(half)((float)x[d]*inv*(float)w[d]);
        uint dr = (d<hh)?(d+hh):(d-hh);
        float xr = (float)(half)((float)x[dr]*inv*(float)w[dr]);
        float rot = (d<hh)? -xr : xr;
        o[d]=(half)(xn*(float)cs[d] + rot*(float)sn[d]);
    }
}

kernel void argmax_part(device const half* logits [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        constant uint& V [[buffer(2)]], constant uint& nchunk [[buffer(3)]],
                        uint tg [[threadgroup_position_in_grid]],
                        uint t [[thread_position_in_threadgroup]],
                        uint nt [[threads_per_threadgroup]],
                        threadgroup float* bv [[threadgroup(0)]],
                        threadgroup uint* bi [[threadgroup(1)]],
                        threadgroup float* sv [[threadgroup(2)]]) {
    uint per = (V + nchunk - 1u)/nchunk;
    uint lo = tg*per, hi = min(lo+per, V);
    float best=-INFINITY, second=-INFINITY; uint idx=0;
    for (uint i=lo+t;i<hi;i+=nt){
        float v=(float)logits[i];
        if (v>best){ second=best; best=v; idx=i; }
        else if (v>second) second=v;
    }
    bv[t]=best; bi[t]=idx; sv[t]=second;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s=nt/2; s>0; s>>=1){
        if (t<s){
            float ob=bv[t+s], os=sv[t+s];
            if (ob>bv[t] || (ob==bv[t] && bi[t+s]<bi[t])){ sv[t]=max(bv[t],os); bv[t]=ob; bi[t]=bi[t+s]; }
            else sv[t]=max(sv[t],ob);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t==0){ out[(size_t)tg*3]=bv[0]; out[(size_t)tg*3+1]=sv[0]; out[(size_t)tg*3+2]=(float)bi[0]; }
}

kernel void argmax_combine(device const float* parts [[buffer(0)]],
                           device float* out3 [[buffer(1)]],
                           constant uint& nchunk [[buffer(2)]],
                           uint t [[thread_position_in_threadgroup]]) {
    if (t!=0) return;
    float best=-INFINITY, second=-INFINITY; uint idx=0;
    for (uint c=0;c<nchunk;++c){
        float b=parts[(size_t)c*3], s=parts[(size_t)c*3+1];
        if (b>best){ second=max(second,best); second=max(second,s); best=b; idx=(uint)parts[(size_t)c*3+2]; }
        else second=max(second,b);
    }
    out3[0]=best; out3[1]=second; out3[2]=(float)idx;
}

kernel void emb_hadamard(device const uchar* packed_row [[buffer(0)]],
                         device const half* codebook [[buffer(1)]],
                         device const half* norms_row [[buffer(2)]],
                         device const half* recip [[buffer(3)]],
                         device const char* lsign [[buffer(4)]],
                         device const char* rsign [[buffer(5)]],
                         device const uint* perm [[buffer(6)]],
                         device half* out [[buffer(7)]], constant uint& gs [[buffer(8)]],
                         uint g [[threadgroup_position_in_grid]],
                         uint t [[thread_position_in_threadgroup]], uint T [[threads_per_threadgroup]],
                         threadgroup float* z [[threadgroup(0)]]) {
    uint pgb = (gs*4u+7u)/8u;
    device const uchar* pg = packed_row + (size_t)g*pgb;
    for (uint k=t; k<gs; k+=T) {
        uchar b=pg[k>>1]; uint idx=(k&1u)?(uint)(b>>4):(uint)(b&0xF);
        uint dst=perm[k]; z[dst]=(float)codebook[idx]*(float)rsign[dst];
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
                        uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    threadgroup half As[EM_TM*EM_TK];
    threadgroup half Bs[EM_TN*EM_TK];
    uint nJ = (K+EM_TN-1)/EM_TN, jt = tg%nJ, mt = tg/nJ;
    uint m0 = mt*EM_TM, j0 = jt*EM_TN, tm = tid/EM_TN, tn = tid%EM_TN, pgb = K/2u;
    float acc = 0;
    for (uint k0=0; k0<K; k0+=EM_TK) {
        for (uint e=tid; e<EM_TM*EM_TK; e+=EM_TM*EM_TN) {
            uint r=e/EM_TK, c=e%EM_TK, mm=m0+r; half v=0;
            if (mm<M) { uint row=rows[mm], ki=k0+c; uchar b=packed[(size_t)row*pgb+(ki>>1)]; uint idx=(ki&1u)?(uint)(b>>4):(uint)(b&0xF); v=codebook[idx]; }
            As[e]=v;
        }
        for (uint e=tid; e<EM_TN*EM_TK; e+=EM_TM*EM_TN) {
            uint r=e/EM_TK, c=e%EM_TK, jj=j0+r; half v=0;
            if (jj<K) v=rotation[(size_t)jj*K + k0 + c];
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
                           uint tg [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
                           uint T [[threads_per_threadgroup]], threadgroup float* z [[threadgroup(0)]]) {
    uint g=tg%ng, m=tg/ng, pgb=(gs*4u+7u)/8u;
    device const uchar* pg = packed + ((size_t)m*ng + g)*pgb;
    for (uint k=t;k<gs;k+=T){ uchar b=pg[k>>1]; uint idx=(k&1u)?(uint)(b>>4):(uint)(b&0xF); uint dst=perm[k]; z[dst]=(float)codebook[idx]*(float)rsign[dst]; }
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
    if (i>=n) return; float av=(float)a[i], bv=(float)b[i], r;
    switch(op){ case 2: r=av-bv; break; case 3: r=av*bv; break; case 4: r=av/bv; break; default: r=av+bv; }
    if (op==1) r=clamp(r,-65500.0f,65500.0f);
    y[i]=(half)r;
}

kernel void rope_f16(device const half* x [[buffer(0)]], device half* out [[buffer(1)]],
                     device const half* cs [[buffer(2)]], device const half* sn [[buffer(3)]],
                     constant uint& heads [[buffer(4)]], constant uint& hd [[buffer(5)]],
                     uint gid [[thread_position_in_grid]]) {
    uint total=heads*hd; if (gid>=total) return;
    uint d=gid%hd, h=gid/hd, hh=hd/2;
    float rot = (d<hh) ? -(float)x[h*hd+d+hh] : (float)x[h*hd+d-hh];
    out[gid]=(half)((float)x[gid]*(float)cs[d] + rot*(float)sn[d]);
}

kernel void scalar_f16(device const half* in [[buffer(0)]], device half* y [[buffer(1)]],
                       constant uint& n [[buffer(2)]], constant int& op [[buffer(3)]],
                       constant float& p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i>=n) return; float v=(float)in[i], r;
    switch(op){ case 0: r=v+p; break; case 1: r=v-p; break; case 3: r=v/p; break; default: r=v*p; }
    y[i]=(half)r;
}

kernel void unary_f16(device const half* in [[buffer(0)]], device half* y [[buffer(1)]],
                      constant uint& n [[buffer(2)]], constant int& op [[buffer(3)]],
                      uint i [[thread_position_in_grid]]) {
    if (i>=n) return; float x=(float)in[i], r;
    if (op==0) r=gelu_tanh(x);
    else if (op==1) r=precise::tanh(x);
    else if (op==2) r=x/(1.0f+precise::exp(-x));
    else r=max(x,0.0f);
    y[i]=(half)r;
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
    for (uint i=t;i<dim;i+=nt){ float rr=(float)r[i]+(float)(half)((float)x[i]*inv*(float)w[i]); o[i]=(half)((float)(half)clamp(rr,-65500.0f,65500.0f)*out_scale); }
}
// h_out = clamp(res + rms(in)*w1)*out_scale; xn_out = rms(h_out)*w2
// Fuses the residual-add norm with the following pre-norm (single row).
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
        float rr=(float)r[i]+(float)(half)((float)x[i]*inv*(float)w1[i]);
        half hv=(half)((float)(half)clamp(rr,-65500.0f,65500.0f)*out_scale);
        o[i]=hv;
        float v=(float)hv; p2+=v*v;
    }
    red[t]=p2; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s=nt/2; s>0; s>>=1){ if (t<s) red[t]+=red[t+s]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float inv2 = 1.0f/sqrt(red[0]/(float)dim + eps);
    for (uint i=t;i<dim;i+=nt) o2[i]=(half)((float)o[i]*inv2*(float)w2[i]);
}

// transform variant whose input is swiglu(gate, up) computed on the fly (gs==128)
kernel void cq4_transform_swiglu(
    device const half*  gate     [[buffer(0)]],
    device const half*  up       [[buffer(1)]],
    device const half*  recip    [[buffer(2)]],
    device const char*  lsign    [[buffer(3)]],
    device const char*  rsign    [[buffer(4)]],
    device const uint*  perm     [[buffer(5)]],
    device       half*  code     [[buffer(6)]],
    constant float& swi_scale    [[buffer(7)]],
    uint g    [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    threadgroup float* zmem      [[threadgroup(0)]])
{
    uint b = g*128u + lane*4u;
    uint k = lane*4u;
    float xv[4];
    #pragma clang loop unroll(full)
    for (uint j=0;j<4;++j){
        float xg=(float)gate[b+j];
        half g1=(half)gelu_tanh(xg);
        half g2=(half)((float)g1*swi_scale);
        half sw=(half)((float)g2*(float)up[b+j]);
        xv[j]=(float)sw*(float)recip[b+j]*(float)lsign[k+j];
    }
    float x0=xv[0],x1=xv[1],x2=xv[2],x3=xv[3];
    float a0=x0+x1,a1=x0-x1,a2=x2+x3,a3=x2-x3;
    x0=a0+a2; x1=a1+a3; x2=a0-a2; x3=a1-a3;
    #pragma clang loop unroll(full)
    for (uint d=1u; d<=16u; d<<=1){
        bool hi=(lane&d)!=0u;
        float p0=simd_shuffle_xor(x0,d),p1=simd_shuffle_xor(x1,d),p2=simd_shuffle_xor(x2,d),p3=simd_shuffle_xor(x3,d);
        x0=hi?p0-x0:x0+p0; x1=hi?p1-x1:x1+p1; x2=hi?p2-x2:x2+p2; x3=hi?p3-x3:x3+p3;
    }
    float s=rsqrt(128.0f);
    zmem[k+0]=x0*s*(float)rsign[k+0]; zmem[k+1]=x1*s*(float)rsign[k+1];
    zmem[k+2]=x2*s*(float)rsign[k+2]; zmem[k+3]=x3*s*(float)rsign[k+3];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    code[b+0]=(half)zmem[perm[k+0]]; code[b+1]=(half)zmem[perm[k+1]];
    code[b+2]=(half)zmem[perm[k+2]]; code[b+3]=(half)zmem[perm[k+3]];
}

// single-dispatch swiglu + transform + gemv (gs==128, K<=4096)
kernel void cq4_gemv_fused_swiglu(
    device const half*  gate     [[buffer(0)]],
    device const uchar* packed   [[buffer(1)]],
    device const half*  codebook [[buffer(2)]],
    device const half*  norms    [[buffer(3)]],
    device       half*  y        [[buffer(4)]],
    constant uint& gs            [[buffer(5)]],
    constant uint& num_groups    [[buffer(6)]],
    constant uint& pgb           [[buffer(7)]],
    constant uint& N             [[buffer(8)]],
    device const half* recip     [[buffer(9)]],
    device const char* lsign     [[buffer(10)]],
    device const char* rsign     [[buffer(11)]],
    device const uint* perm      [[buffer(12)]],
    device const half* up        [[buffer(13)]],
    constant float& swi_scale    [[buffer(14)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tl   [[thread_index_in_threadgroup]],
    threadgroup half* codem      [[threadgroup(0)]])
{
    threadgroup float cb[16];
    if (tl<16) cb[tl]=(float)codebook[tl];
    const uint K = num_groups*gs;
    threadgroup float* zs = (threadgroup float*)(codem + K) + sgid*128u;
    for (uint g=sgid; g<num_groups; g+=ROWS) {
        uint b = g*128u + lane*4u;
        uint k = lane*4u;
        float xv[4];
        #pragma clang loop unroll(full)
        for (uint j=0;j<4;++j){
            float xg=(float)gate[b+j];
            half g1=(half)gelu_tanh(xg);
            half g2=(half)((float)g1*swi_scale);
            half sw=(half)((float)g2*(float)up[b+j]);
            xv[j]=(float)sw*(float)recip[b+j]*(float)lsign[k+j];
        }
        float x0=xv[0],x1=xv[1],x2=xv[2],x3=xv[3];
        float a0=x0+x1,a1=x0-x1,a2=x2+x3,a3=x2-x3;
        x0=a0+a2; x1=a1+a3; x2=a0-a2; x3=a1-a3;
        #pragma clang loop unroll(full)
        for (uint d=1u; d<=16u; d<<=1){
            bool hi=(lane&d)!=0u;
            float p0=simd_shuffle_xor(x0,d),p1=simd_shuffle_xor(x1,d),p2=simd_shuffle_xor(x2,d),p3=simd_shuffle_xor(x3,d);
            x0=hi?p0-x0:x0+p0; x1=hi?p1-x1:x1+p1; x2=hi?p2-x2:x2+p2; x3=hi?p3-x3:x3+p3;
        }
        float s=rsqrt(128.0f);
        zs[k+0]=x0*s*(float)rsign[k+0]; zs[k+1]=x1*s*(float)rsign[k+1];
        zs[k+2]=x2*s*(float)rsign[k+2]; zs[k+3]=x3*s*(float)rsign[k+3];
        simdgroup_barrier(mem_flags::mem_threadgroup);
        codem[g*128u+k+0]=(half)zs[perm[k+0]]; codem[g*128u+k+1]=(half)zs[perm[k+1]];
        codem[g*128u+k+2]=(half)zs[perm[k+2]]; codem[g*128u+k+3]=(half)zs[perm[k+3]];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint n0 = (tg*ROWS + sgid)*4u;
    if (n0>=N) return;
    float a0v=0,a1v=0,a2v=0,a3v=0;
    for (uint base=lane*CQ4_VPL; base<K; base+=32u*CQ4_VPL){
        uint g = base/gs, off = base - g*gs;
        threadgroup const half4* cbase=(threadgroup const half4*)(codem + g*gs + off);
        device const ushort4* q0p=(device const ushort4*)(packed + ((size_t)(n0+0)*num_groups+g)*pgb + off/2u);
        device const ushort4* q1p=(device const ushort4*)(packed + ((size_t)(n0+1)*num_groups+g)*pgb + off/2u);
        device const ushort4* q2p=(device const ushort4*)(packed + ((size_t)(n0+2)*num_groups+g)*pgb + off/2u);
        device const ushort4* q3p=(device const ushort4*)(packed + ((size_t)(n0+3)*num_groups+g)*pgb + off/2u);
        ushort4 w0=q0p[0], w1=q1p[0], w2=q2p[0], w3=q3p[0];
        float q0=0,q1=0,q2=0,q3=0;
        #pragma clang loop unroll(full)
        for (uint q=0;q<4;++q){
            half4 c=cbase[q];
            ushort wa=w0[q],wb=w1[q],wc=w2[q],wd=w3[q];
            float cx=(float)c.x, cy=(float)c.y, cz=(float)c.z, cw=(float)c.w;
            q0 += cx*cb[wa&0xF] + cy*cb[(wa>>4)&0xF] + cz*cb[(wa>>8)&0xF] + cw*cb[(wa>>12)&0xF];
            q1 += cx*cb[wb&0xF] + cy*cb[(wb>>4)&0xF] + cz*cb[(wb>>8)&0xF] + cw*cb[(wb>>12)&0xF];
            q2 += cx*cb[wc&0xF] + cy*cb[(wc>>4)&0xF] + cz*cb[(wc>>8)&0xF] + cw*cb[(wc>>12)&0xF];
            q3 += cx*cb[wd&0xF] + cy*cb[(wd>>4)&0xF] + cz*cb[(wd>>8)&0xF] + cw*cb[(wd>>12)&0xF];
        }
        a0v += (float)norms[(size_t)(n0+0)*num_groups+g]*q0;
        a1v += (float)norms[(size_t)(n0+1)*num_groups+g]*q1;
        a2v += (float)norms[(size_t)(n0+2)*num_groups+g]*q2;
        a3v += (float)norms[(size_t)(n0+3)*num_groups+g]*q3;
    }
    a0v=simd_sum(a0v); a1v=simd_sum(a1v); a2v=simd_sum(a2v); a3v=simd_sum(a3v);
    if (lane==0){ y[n0]=(half)a0v; y[n0+1]=(half)a1v; y[n0+2]=(half)a2v; y[n0+3]=(half)a3v; }
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

kernel void bcast_binary_f16(device const half* a [[buffer(0)]], device const half* b [[buffer(1)]],
    device half* out [[buffer(2)]], constant uint* oshape [[buffer(3)]],
    constant uint* astride [[buffer(4)]], constant uint* bstride [[buffer(5)]],
    constant uint& ndim [[buffer(6)]], constant uint& total [[buffer(7)]], constant int& op [[buffer(8)]],
    uint i [[thread_position_in_grid]]) {
    if (i>=total) return;
    uint rem=i, ai=0, bi=0;
    for (int d=int(ndim)-1; d>=0; --d){ uint c=rem%oshape[d]; rem/=oshape[d]; ai+=c*astride[d]; bi+=c*bstride[d]; }
    float av=(float)a[ai], bv=(float)b[bi], r;
    switch(op){ case 2:r=av-bv;break; case 3:r=av*bv;break; case 4:r=av/bv;break; default:r=av+bv; }
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

// Fused decode-attention prologue + flash attention:
// per-head rms-norm + rope for q, rms+rope for the new k, rms for the new v,
// int8 kv-cache append (by threadgroup 0), and attention — one dispatch.
// use_local: the key/value at position local_slot comes from the freshly
// computed kr/vn instead of the cache (covers both append-at-end and
// ring-slot-replacement). All threadgroups recompute kr/vn redundantly.
kernel void attn_decode_fused_i8(
    device const half*  qraw  [[buffer(0)]],
    device const half*  kraw  [[buffer(1)]],
    device const half*  vraw  [[buffer(2)]],
    device const half*  qw    [[buffer(3)]],
    device const half*  kw    [[buffer(4)]],
    device const half*  vw    [[buffer(5)]],
    device const half*  cs    [[buffer(6)]],
    device const half*  sn    [[buffer(7)]],
    device       char*  kc    [[buffer(8)]],
    device       char*  vc    [[buffer(9)]],
    device       float* ks    [[buffer(10)]],
    device       float* vs    [[buffer(11)]],
    device       half*  out   [[buffer(12)]],
    constant uint& num_q_heads [[buffer(13)]],
    constant uint& head_dim    [[buffer(14)]],
    constant float& scale      [[buffer(15)]],
    constant uint& kv_start    [[buffer(16)]],
    constant uint& kv_end      [[buffer(17)]],
    constant uint& use_local   [[buffer(18)]],
    constant uint& local_slot  [[buffer(19)]],
    constant float& eps        [[buffer(20)]],
    device       float* part_o [[buffer(21)]],
    device       float* part_ml[[buffer(22)]],
    constant uint& nwg         [[buffer(23)]],
    uint tg [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
    uint T [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], threadgroup float* smem [[threadgroup(0)]])
{
    const uint hd = head_dim;
    uint h = tg / nwg, w = tg % nwg;
    uint nsg = T / 32u;
    uint hh = hd/2u;

    // smem layout (floats): Otg[nsg*hd] | mtg[nsg] | ltg[nsg] | kr[hd] | vn[hd] | red[256]
    threadgroup float* Otg = smem;
    threadgroup float* mtg = Otg + (size_t)nsg*hd;
    threadgroup float* ltg = mtg + nsg;
    threadgroup float* krm = ltg + nsg;
    threadgroup float* vnm = krm + hd;
    threadgroup float* red = vnm + hd;

    // fast rms sum: per-simdgroup simd_sum, then combine across simdgroups
    auto rms_inv = [&](device const half* src) {
        float p=0; for (uint i=t;i<hd;i+=T){ float v=(float)src[i]; p+=v*v; }
        p = simd_sum(p);
        if (lane==0) red[sg]=p;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float s=0; for (uint i=0;i<nsg;++i) s+=red[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        return 1.0f/sqrt(s/(float)hd + eps);
    };

    // ---- q: rms + rope for this head ----
    device const half* qh = qraw + (size_t)h*hd;
    float inv_q = rms_inv(qh);

    float qreg[16];
    { uint i=0; for (uint d=lane; d<hd; d+=32){
        float xn = (float)(half)((float)qh[d]*inv_q*(float)qw[d]);
        uint dr = (d<hh)?(d+hh):(d-hh);
        float xr = (float)(half)((float)qh[dr]*inv_q*(float)qw[dr]);
        float rot = (d<hh)? -xr : xr;
        qreg[i++] = (float)(half)(xn*(float)cs[d] + rot*(float)sn[d]);
    } }

    // ---- new k/v: rms(+rope) into threadgroup memory ----
    // only the workgroup whose key stripe covers local_slot (and the appender,
    // threadgroup 0) actually needs kr/vn
    bool need_local = false;
    if (use_local != 0u) {
        uint stride0 = nwg*nsg;
        uint gsg0 = (local_slot - kv_start) % stride0;
        uint w0 = gsg0 / nsg;
        need_local = (w == w0) || (tg == 0u);
    }
    if (need_local) {
        float inv_k = rms_inv(kraw);
        float inv_v = rms_inv(vraw);
        for (uint d=t; d<hd; d+=T){
            float xn = (float)(half)((float)kraw[d]*inv_k*(float)kw[d]);
            uint dr = (d<hh)?(d+hh):(d-hh);
            float xr = (float)(half)((float)kraw[dr]*inv_k*(float)kw[dr]);
            float rot = (d<hh)? -xr : xr;
            krm[d] = (float)(half)(xn*(float)cs[d] + rot*(float)sn[d]);
            vnm[d] = (float)(half)((float)vraw[d]*inv_v*(float)vw[d]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // append to cache: threadgroup 0 only, matching kv_append_i8 exactly
        if (tg == 0u) {
            uint num_groups = hd/32u;
            for (uint g=t; g<num_groups; g+=T){
                uint gstart=g*32u;
                float maxabs=0;
                for (uint k2=0;k2<32u;++k2) maxabs = max(maxabs, fabs(krm[gstart+k2]));
                float sc = maxabs/127.0f; if (sc < 1e-10f) sc = 1e-10f;
                float inv = 1.0f/sc;
                device char* dst = kc + (size_t)local_slot*hd + gstart;
                for (uint k2=0;k2<32u;++k2) dst[k2]=(char)clamp(rint(krm[gstart+k2]*inv), -128.0f, 127.0f);
                ks[(size_t)local_slot*num_groups + g] = sc;
                maxabs=0;
                for (uint k2=0;k2<32u;++k2) maxabs = max(maxabs, fabs(vnm[gstart+k2]));
                sc = maxabs/127.0f; if (sc < 1e-10f) sc = 1e-10f;
                inv = 1.0f/sc;
                device char* dstv = vc + (size_t)local_slot*hd + gstart;
                for (uint k2=0;k2<32u;++k2) dstv[k2]=(char)clamp(rint(vnm[gstart+k2]*inv), -128.0f, 127.0f);
                vs[(size_t)local_slot*num_groups + g] = sc;
            }
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- flash attention over [kv_start, kv_end) ----
    uint ngK = hd/32u;
    float o_acc[16];
    for (uint i=0;i<16;++i) o_acc[i] = 0.0f;
    float m_i = -INFINITY, l_i = 0.0f;

    uint gsg = w*nsg + sg, stride = nwg*nsg;
    for (uint k = kv_start + gsg; k < kv_end; k += stride) {
        float partial = 0.0f;
        bool local = (use_local != 0u) && (k == local_slot);
        if (local) {
            uint i=0; for (uint d=lane; d<hd; d+=32){ partial += qreg[i]*krm[d]; ++i; }
        } else {
            device const char*  kk  = kc + (size_t)k*hd;
            device const float* kss = ks + (size_t)k*ngK;
            uint i=0; for (uint d=lane; d<hd; d+=32){ partial += qreg[i] * ((float)kk[d]*kss[d/32]); ++i; }
        }
        float s = simd_sum(partial) * scale;
        float m_new = max(m_i, s);
        float resc  = exp(m_i - m_new);
        float p     = exp(s - m_new);
        l_i = l_i * resc + p;
        if (local) {
            uint i=0; for (uint d=lane; d<hd; d+=32){ o_acc[i] = o_acc[i]*resc + p*vnm[d]; ++i; }
        } else {
            device const char*  vvv = vc + (size_t)k*hd;
            device const float* vss = vs + (size_t)k*ngK;
            uint i=0; for (uint d=lane; d<hd; d+=32){ o_acc[i] = o_acc[i]*resc + p*((float)vvv[d]*vss[d/32]); ++i; }
        }
        m_i = m_new;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) { mtg[sg] = m_i; ltg[sg] = l_i; }
    { uint i=0; for (uint d=lane; d<hd; d+=32) { Otg[(size_t)sg*hd + d] = o_acc[i]; ++i; } }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float gm = -INFINITY;
    for (uint i=0;i<nsg;++i) gm = max(gm, mtg[i]);
    float gl = 0.0f;
    for (uint i=0;i<nsg;++i) gl += ltg[i] * exp(mtg[i] - gm);

    if (nwg == 1u) {
        float inv = gl > 0.0f ? 1.0f/gl : 0.0f;
        for (uint d=t; d<hd; d+=T) {
            float acc = 0.0f;
            for (uint i=0;i<nsg;++i) acc += Otg[(size_t)i*hd + d] * exp(mtg[i] - gm);
            out[(size_t)h*hd + d] = (half)(acc * inv);
        }
    } else {
        uint slot2 = h*nwg + w;
        if (t == 0) { part_ml[(size_t)slot2*2] = gm; part_ml[(size_t)slot2*2 + 1] = gl; }
        for (uint d=t; d<hd; d+=T) {
            float acc = 0.0f;
            for (uint i=0;i<nsg;++i) acc += Otg[(size_t)i*hd + d] * exp(mtg[i] - gm);
            part_o[(size_t)slot2*hd + d] = acc;
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
    device char* dst = int8base + (size_t)slot*i8s + (size_t)h*hdim + gstart;
    for (uint k=0;k<gcount;++k){ float qv = clamp(rint((float)hs[k]*inv), -128.0f, 127.0f); dst[k]=(char)qv; }
    scalebase[(size_t)slot*scs + (size_t)h*num_groups + g] = scale;
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
