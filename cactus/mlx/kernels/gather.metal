uint lane = thread_index_in_simdgroup;
uint output_group = thread_position_in_grid.x / 32;
if (output_group >= R * (K / 128)) return;
uint g = output_group % (K / 128), output_row = output_group / (K / 128);
uint row = rows[output_row];
ulong group = ulong(row) * (K / 128) + g;
float norm = cq_fp8(norms[group]) * float(scales[row / SCALE_ROWS]);
float4 v;
for (uint j = 0; j < 4; ++j)
    v[j] = cq_decode(codes, group, lane * 4 + j, table, levels) * (norm / 11.313708498984761f);
v = cq_fwht(v, lane) * 0.08838834764831845f;
for (uint j = 0; j < 4; ++j) {
    uint col = g * 128 + lane * 4 + j;
    out[ulong(output_row) * K + col] = static_cast<OUT>(v[j] * cq_fp8(a[col]) * cq_fp8(c[row]));
}
