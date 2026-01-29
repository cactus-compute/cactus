#ifndef HEXAGON_TYPES_H
#define HEXAGON_TYPES_H

#include <cstdint>
#include <cstring>
#include <algorithm>

// Simulate HVX Vector as a struct of 128 bytes
typedef struct {
    uint8_t bytes[128];
} HVX_Vector;

// Mock __fp16 on x86 host for verification logic
#if defined(__x86_64__) || defined(_M_X64)
    // Force 16-bit size using macro to override potential native 32-bit __fp16
    #define __fp16 uint16_t
#endif

#endif
