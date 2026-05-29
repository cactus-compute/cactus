/* residual_add: y = y + x. In-place add for the residual stream.
 *
 * Both buffers are fp16 length AXIS_SIZE_RES. One thread per element. */
#include "common.metal"

constant uint AXIS_SIZE_RES [[function_constant(80)]];

kernel void residual_add(
    device       half * y    [[buffer(0)]],   // accumulator
    device const half * x    [[buffer(1)]],   // addend
    uint3 gid_v [[thread_position_in_grid]])
{
    const uint i = gid_v.x;
    if (i >= AXIS_SIZE_RES) return;
    y[i] = half(float(y[i]) + float(x[i]));
}
