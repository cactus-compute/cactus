#ifndef HEXAGON_PROTOS_H
#define HEXAGON_PROTOS_H

#include "hexagon_types.h"
#include <vector>
#include <iostream>

// Simple Half-Float simulation helper
// We treat __fp16 as float for logic check
static float half_to_float(uint16_t h) {
    // Very dummy conversion for "1.0" or integers
    // We are testing LOGIC flow, not bitwise float accuracy here.
    // Actually, for the kernel test, passing integers (which are valid floats) is easier.
    return (float)h; 
}

// Mock Intrinsics Implementation

static inline HVX_Vector Q6_V_vzero() {
    HVX_Vector v;
    std::memset(v.bytes, 0, 128);
    return v;
}

static inline HVX_Vector Q6_V_vldu_A(HVX_Vector* addr) {
    HVX_Vector v;
    std::memcpy(v.bytes, addr->bytes, 128);
    return v;
}

// Add: We perform byte-wise addition of 16-bit elements
// Note: This is a LOGIC MOCK. Real hardware does vector float add.
// We will simulate it by treating bytes as uint16_t and adding.
static inline HVX_Vector Q6_Vhf_vadd_VhfVhf(HVX_Vector a, HVX_Vector b) {
    HVX_Vector res;
    __fp16* pa = (__fp16*)a.bytes;
    __fp16* pb = (__fp16*)b.bytes;
    __fp16* pr = (__fp16*)res.bytes;
    
    // 64 elements
    for(int i=0; i<64; ++i) {
        // Use native half-float addition!
        pr[i] = pa[i] + pb[i];
    }
    return res;
}

static inline HVX_Vector Q6_V_vror_Vr(HVX_Vector v, int rot_bytes) {
    HVX_Vector res;
    // Rotate right by rot_bytes
    // If rot_bytes = 64, we swap halves.
    rot_bytes %= 128;
    if (rot_bytes < 0) rot_bytes += 128; // Handle negative just in case
    
    // Copy second part to start
    std::memcpy(res.bytes, v.bytes + (128 - rot_bytes), rot_bytes);
    // Copy first part to end
    std::memcpy(res.bytes + rot_bytes, v.bytes, 128 - rot_bytes);
    
    return res;
}

static inline void Q6_V_vstu_A(HVX_Vector* addr, HVX_Vector val) {
    std::memcpy(addr->bytes, val.bytes, 128);
}

#endif
