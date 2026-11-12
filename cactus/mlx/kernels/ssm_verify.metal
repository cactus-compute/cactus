// Short recurrent scan; retain either each prefix or only the final state.
uint h = thread_position_in_grid.z;
uint d = thread_position_in_grid.y;
uint lane = thread_index_in_simdgroup;
constexpr uint V = DS / 32;
float recurrent[V];
for (uint j = 0; j < V; ++j) {
    uint s = lane * V + j;
    recurrent[j] = state[(h * DH + d) * DS + s];
}
float decay_rate = -fast::exp(float(a_log[h]));
for (uint t = 0; t < NT; ++t) {
    float delta = float(dt[t * H + h]);
    float decay = fast::exp(decay_rate * delta);
    float value = float(x[(t * H + h) * DH + d]);
    float acc = 0;
    for (uint j = 0; j < V; ++j) {
        uint s = lane * V + j;
        uint bc = (t * NG + h / (H / NG)) * DS + s;
        float injection = value * delta * float(b[bc]);
        recurrent[j] = decay * recurrent[j] + injection;
        if constexpr (SNAPSHOTS) history[((t * H + h) * DH + d) * DS + s] = recurrent[j];
        acc += recurrent[j] * float(c[bc]);
    }
    acc = simd_sum(acc);
    if (lane == 0) out[(t * H + h) * DH + d] = T(acc + value * float(skip[h]));
}
if constexpr (!SNAPSHOTS) {
    for (uint j = 0; j < V; ++j) history[(h * DH + d) * DS + lane * V + j] = recurrent[j];
}
