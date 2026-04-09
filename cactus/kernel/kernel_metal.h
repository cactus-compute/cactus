#ifndef KERNEL_METAL_H
#define KERNEL_METAL_H

#ifdef __APPLE__
#ifdef CACTUS_USE_MPS

#include <cstddef>
#include <cstdint>

// Minimum FLOP count to justify MPS GPU launch overhead.
// Below this threshold, CPU paths (Accelerate/NEON) are faster.
// 256*256*256 = 16,777,216
constexpr size_t MPS_MIN_FLOPS = 256UL * 256UL * 256UL;

// Check if a Metal GPU device is available on this system.
bool cactus_metal_available();

// FP16 matrix multiplication via Metal Performance Shaders (MPS).
// A is [M x K] row-major, b_transposed is [N x K] (transposed/row-major), C is [M x N] row-major.
// Uses zero-copy unified memory buffers on Apple Silicon.
// NOTE: the __fp16 pointers are cast to uint16_t* internally to avoid
// NEON type conflicts in the ObjC++ compilation unit.
void cactus_matmul_f16_mps(const __fp16* a, const __fp16* b_transposed, __fp16* c,
                           size_t M, size_t K, size_t N);

#endif // CACTUS_USE_MPS
#endif // __APPLE__

#endif // KERNEL_METAL_H
