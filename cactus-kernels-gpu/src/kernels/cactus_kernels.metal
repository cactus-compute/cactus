/* Master Metal source. Concatenates all kernel sources for a single
 * .metallib compile pass. The Metal compiler treats each `#include` as
 * inline source, so we just include everything here. */
#include "matmul_int4.metal"
#include "rms_norm.metal"
#include "rope.metal"
#include "swiglu.metal"
#include "embed_and_kv.metal"
#include "sample.metal"
#include "flash_attn.metal"
#include "residual_add.metal"
#include "matmul_fp16.metal"
