uint lane = thread_index_in_simdgroup;
uint group = thread_position_in_grid.x / 32;
uint local_group = thread_position_in_threadgroup.x / 32;
uint start = group * 128 + lane * 4;
float4 v = 0;
if (group < GROUPS) {
    for (uint j = 0; j < 4; ++j) {
        float z = float(x[start + j]);
        v[j] = z * cq_fp8(a[(start + j) % K]);
    }
}
v = cq_fwht(v, lane) / 128.0f;
threadgroup float rotated[512];
for (uint j = 0; j < 4; ++j) rotated[local_group * 128 + lane * 4 + j] = v[j];
threadgroup_barrier(mem_flags::mem_threadgroup);
if (group < GROUPS) {
    uint unit = lane / 4;
    for (uint slot = 0; slot < SLOTS; ++slot) {
        for (uint j = 0; j < 4; ++j) {
            uint coord = (lane % 4) * 4 + j;
            out[((group * 8 + unit) * SLOTS + slot) * 16 + coord] =
                rotated[local_group * 128 + unit * 16 + permutation[slot * 16 + coord]];
        }
    }
}
