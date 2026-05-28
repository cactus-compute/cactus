/* Cactus GPU kernel library — Apple Metal backend.
 *
 * This header is the public C++ surface for the GPU kernel pack. All Metal
 * objects (MTLDevice, MTLCommandQueue, MTLComputePipelineState, MTLBuffer)
 * are hidden behind opaque handles so callers don't need to import Metal
 * headers (which require Objective-C++ / `.mm` files to compile).
 *
 * Lifecycle:
 *   1. `cactus_gpu_context_create()` — one per process / per inference stream.
 *   2. `cactus_gpu_buffer_create_*()` — upload weights, allocate KV cache.
 *   3. `cactus_gpu_pipeline_create_*()` — build kernel pipelines (function
 *      constants baked at creation; immutable thereafter).
 *   4. Per-token: `cactus_gpu_command_buffer_begin()` →
 *      `cactus_gpu_dispatch_*()` calls → `cactus_gpu_command_buffer_commit()`.
 *
 * All headers in this library are plain C++ — implementation lives in `.mm`
 * Objective-C++ files but the public surface is C-callable for FFI ease.
 */
#ifndef CACTUS_GPU_H
#define CACTUS_GPU_H

#include <cstddef>
#include <cstdint>

#if defined(__APPLE__)
  #define CACTUS_HAS_METAL 1
#else
  #define CACTUS_HAS_METAL 0
#endif

namespace cactus {
namespace gpu {

// ============================================================================
// Opaque handles (defined in .mm files; never dereferenced by callers)
// ============================================================================
struct Context;
struct Buffer;
struct Pipeline;
struct CommandBuffer;

// ============================================================================
// Data types — wire-level, must match the .metal kernel side
// ============================================================================
enum class DType : uint8_t {
    FP16  = 0,
    FP32  = 1,
    INT8  = 2,
    INT4  = 3,  // Cactus CQ4 (group 128, hadamard or orthogonal rotation)
    INT32 = 4,
};

// Storage mode hint. On Apple Silicon, SHARED gives CPU + GPU same address
// (zero-copy). PRIVATE is GPU-only (faster for hot buffers like KV cache).
enum class StorageMode : uint8_t {
    SHARED  = 0,
    PRIVATE = 1,
};

// ============================================================================
// Context — owns MTLDevice + MTLCommandQueue + the kernel library
// ============================================================================
// Creates a context against the default Metal device. Loads the cactus
// kernel library from the supplied .metallib path. Returns nullptr on
// failure (no Metal, file not found, etc).
Context* context_create(const char* metallib_path);
void     context_destroy(Context* ctx);
bool     context_has_metal(const Context* ctx);  // always false on non-Apple

// Device-side memory total, in bytes. (Used for budget decisions.)
uint64_t context_recommended_max_working_set(const Context* ctx);

// ============================================================================
// Buffers — MTLBuffer wrappers
// ============================================================================
// Wrap an existing host pointer (e.g., mmapped weight file) as a shared MTLBuffer.
// Zero-copy on Apple Silicon. The host memory must stay alive for the lifetime
// of the buffer.
Buffer* buffer_wrap_host_memory(Context* ctx, void* host_ptr, size_t size);

// Allocate a new device buffer. Use PRIVATE for KV cache, SHARED for any
// buffer the CPU might need to read or write.
Buffer* buffer_create(Context* ctx, size_t size, StorageMode mode);

// SHARED only: returns the CPU-visible pointer. nullptr for PRIVATE buffers.
void*   buffer_contents(Buffer* buf);
size_t  buffer_size(const Buffer* buf);

void    buffer_destroy(Buffer* buf);

// ============================================================================
// Pipelines — compiled kernel + function constants
// ============================================================================
// Function constants are name → value pairs that the Metal compiler bakes
// into the kernel at pipeline creation time. We pass them as raw bytes plus
// a type tag (matches MTLDataType internally).
enum class FCType : uint8_t { BOOL = 0, INT32 = 1, FP32 = 2, UINT32 = 3 };

struct FunctionConstant {
    const char* name;       // Metal name (must match [[function_constant(...)]] index name)
    FCType      type;
    uint32_t    index;      // [[function_constant(N)]]
    union { bool b; int32_t i32; float f32; uint32_t u32; } value;
};

// Build a compute pipeline for `kernel_name`, applying the given function
// constants. Returns nullptr on compile failure. Pipelines are immutable
// once built; cache them per (kernel_name, hash(function_constants)).
Pipeline* pipeline_create(
    Context* ctx,
    const char* kernel_name,
    const FunctionConstant* constants,
    size_t                  num_constants);

void pipeline_destroy(Pipeline* p);

// ============================================================================
// Command buffer + dispatch
// ============================================================================
// A CommandBuffer is one unit of GPU work. Open one, encode N kernel
// dispatches into it, commit. The runtime keeps a pool of these (~4) and
// pipelines them.
CommandBuffer* command_buffer_begin(Context* ctx);

// Encode a 1D, 2D, or 3D dispatch. `threadgroup_*` is the threadgroup
// shape; `grid_*` is the total grid (Apple's modern non-uniform threadgroup
// dispatch — kernel handles the remainder lanes itself).
// `buffers` are bound in slot order [0..num_buffers).
struct BufferBinding {
    Buffer* buffer;
    size_t  offset;
};

void command_buffer_dispatch(
    CommandBuffer*       cb,
    Pipeline*            pipeline,
    const BufferBinding* buffers,
    size_t               num_buffers,
    uint32_t grid_x,           uint32_t grid_y,           uint32_t grid_z,
    uint32_t threadgroup_x,    uint32_t threadgroup_y,    uint32_t threadgroup_z);

// Memory barrier — ensures all preceding kernel writes are visible to the
// next kernel's reads. Use between any two dispatches that have a
// read-after-write dependency. Apple is permissive about implicit barriers
// across MTLBuffer reads, but we insert explicit ones for correctness.
void command_buffer_barrier(CommandBuffer* cb);

// Commit (non-blocking). Use `command_buffer_wait()` only when you need
// the result on CPU (e.g., to read the sampled token id).
void command_buffer_commit(CommandBuffer* cb);
void command_buffer_wait(CommandBuffer* cb);
bool command_buffer_is_complete(const CommandBuffer* cb);

// ============================================================================
// High-level convenience: build cactus kernel pipelines by enum name
// ============================================================================
// These functions wrap `pipeline_create()` with the right function-constant
// set per kernel. Callers don't need to know the constant indices.

// INT4 matrix-vector (decode hot path): out = W_int4 * x_fp16
// K must be a multiple of 128 (the CQ4 group size). N must be a multiple of 4.
Pipeline* pipeline_mul_mv_int4_fp16(Context* ctx, uint32_t K, uint32_t N);

// INT4 matrix-matrix (prefill): out = W_int4 * X_fp16, batched.
Pipeline* pipeline_mul_mm_int4_fp16(Context* ctx, uint32_t K, uint32_t N, uint32_t M_tile);

// FP16 × FP16 matrix-matrix (e.g., LM head).
Pipeline* pipeline_mul_mm_fp16(Context* ctx, uint32_t K, uint32_t N);

// RMSNorm with fused weight scale. `axis_size` = the last dim (model dim).
Pipeline* pipeline_rms_norm_fp16(Context* ctx, uint32_t axis_size);

// Flash attention (single Q-tile path for decode + multi-Q for prefill).
// `head_dim_q` and `head_dim_v` selected at kernel-name level (specialized).
// `num_query_groups` is num_query_heads / num_kv_heads (GQA factor).
Pipeline* pipeline_flash_attn(
    Context* ctx,
    uint32_t head_dim_q,
    uint32_t head_dim_v,
    uint32_t num_query_groups,
    bool     causal,
    bool     has_softcap);

// RoPE applied in place to Q and K tensors.
Pipeline* pipeline_rope_apply(
    Context* ctx,
    uint32_t head_dim,
    bool     is_neox,
    float    theta_base);

// SwiGLU activation: out = silu(gate) * up.
Pipeline* pipeline_swiglu(Context* ctx, uint32_t hidden_dim);

// Append a single token's K,V into the per-layer cache at offset.
Pipeline* pipeline_kv_cache_append(Context* ctx, uint32_t num_kv_heads, uint32_t head_dim);

// Sampling: argmax (greedy) and top-k/top-p stochastic.
Pipeline* pipeline_sample_argmax(Context* ctx, uint32_t vocab_size);
Pipeline* pipeline_sample_top_k_top_p(Context* ctx, uint32_t vocab_size);

// Embedding lookup: gathers row `token_id` from the embedding matrix.
Pipeline* pipeline_embed_lookup(Context* ctx, uint32_t hidden_dim);

} // namespace gpu
} // namespace cactus

#endif  // CACTUS_GPU_H
