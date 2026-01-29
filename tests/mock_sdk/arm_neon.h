#ifndef ARM_NEON_H
#define ARM_NEON_H

// Mock types for x86 compilation of files that include <arm_neon.h> unconditionally
// We only need the types to exist so the compiler doesn't choke.

#include <cstdint>

typedef struct {
    float val[4];
} float16x8_t; // Dummy definition, size doesn't match but it's okay if not used.
// Actually float16x8_t is 8x16bit = 128 bit.
// Let's make it generic.
typedef struct {
    uint8_t bytes[16];
} uint8x16_t;

typedef float16x8_t float32x4_t; // Dummy

// Mock Intrinsics for compilation parsing
static inline float16x8_t vdupq_n_f16(float v) { return float16x8_t(); }
static inline float16x8_t vld1q_f16(const __fp16* p) { return float16x8_t(); }
static inline float16x8_t vaddq_f16(float16x8_t a, float16x8_t b) { return float16x8_t(); }
static inline void vst1q_f16(__fp16* p, float16x8_t v) {}

// Float32 mocks
static inline float32x4_t vdupq_n_f32(float v) { return float32x4_t(); }
static inline float32x4_t vld1q_f32(const float* p) { return float32x4_t(); }
static inline float32x4_t vaddq_f32(float32x4_t a, float32x4_t b) { return float32x4_t(); }
static inline float32x4_t vmulq_f32(float32x4_t a, float32x4_t b) { return float32x4_t(); }
static inline void vst1q_f32(float* p, float32x4_t v) {}
static inline float32x4_t vcvtq_f32_f16(float16x8_t v) { return float32x4_t(); } // Guessing usage
static inline float16x8_t vcvt_f16_f32(float32x4_t v) { return float16x8_t(); } // Guessing usage

typedef struct { float val[2]; } float16x4_t;
static inline float16x4_t vget_low_f16(float16x8_t v) { return float16x4_t(); }
static inline float16x4_t vget_high_f16(float16x8_t v) { return float16x4_t(); }
static inline float32x4_t vcvt_f32_f16(float16x4_t v) { return float32x4_t(); }

static inline float32x4_t vfmaq_f32(float32x4_t a, float32x4_t b, float32x4_t c) { return float32x4_t(); }
static inline float32x4_t vsubq_f32(float32x4_t a, float32x4_t b) { return float32x4_t(); }
static inline float32x4_t vdivq_f32(float32x4_t a, float32x4_t b) { return float32x4_t(); }
static inline float32x4_t vmlaq_f32(float32x4_t a, float32x4_t b, float32x4_t c) { return float32x4_t(); }
static inline float vaddvq_f32(float32x4_t v) { return 0.0f; }

static inline float32x4_t vmaxq_f32(float32x4_t a, float32x4_t b) { return float32x4_t(); }
static inline float32x4_t vminq_f32(float32x4_t a, float32x4_t b) { return float32x4_t(); }

static inline float16x8_t vminq_f16(float16x8_t a, float16x8_t b) { return float16x8_t(); }
static inline float16x8_t vmaxq_f16(float16x8_t a, float16x8_t b) { return float16x8_t(); }
static inline __fp16 vminv_f16(float16x8_t v) { return 0; }
static inline __fp16 vmaxv_f16(float16x8_t v) { return 0; }

typedef struct { float val[4]; } uint32x4_t; // Logic mask
static inline uint32x4_t vcgtq_f32(float32x4_t a, float32x4_t b) { return uint32x4_t(); }
static inline uint32x4_t vcltq_f32(float32x4_t a, float32x4_t b) { return uint32x4_t(); }
static inline float32x4_t vbslq_f32(uint32x4_t m, float32x4_t a, float32x4_t b) { return float32x4_t(); }

#endif
