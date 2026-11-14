uint x = thread_position_in_grid.x, y = thread_position_in_grid.y;
uint width = column_weights_shape[0], column_taps = column_weights_shape[1], row_taps = row_weights_shape[1];
float3 total = 0.0f;
for (uint r = 0; r < row_taps; ++r) {
    ulong row = ulong(rows[y * row_taps + r]) * pixels_shape[1];
    float3 line = 0.0f;
    for (uint c = 0; c < column_taps; ++c) {
        ulong p = (row + columns[x * column_taps + c]) * 3;
        line += column_weights[x * column_taps + c] * float3(pixels[p], pixels[p + 1], pixels[p + 2]);
    }
    total += row_weights[y * row_taps + r] * line;
}
uint plane = row_weights_shape[0] * width;
for (uint channel = 0; channel < 3; ++channel)
    out[channel * plane + y * width + x] = (total[channel] / 255.0f - mean[channel]) / std[channel];
