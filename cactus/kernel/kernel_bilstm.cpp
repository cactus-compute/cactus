/**
 * kernel_bilstm.cpp — Bidirectional LSTM sequence kernel
 *
 * Processes a full sequence through a bidirectional LSTM layer using the
 * existing NEON-optimized cactus_lstm_cell_f16 kernel.
 *
 * Output: concat(h_fwd, h_bwd) at each timestep → (batch, T, 2*hidden)
 */

#include "kernel.h"
#include <vector>

void cactus_bilstm_sequence_f16(
    const __fp16* input,
    const __fp16* weight_ih_fwd,
    const __fp16* weight_hh_fwd,
    const __fp16* bias_ih_fwd,
    const __fp16* bias_hh_fwd,
    const __fp16* weight_ih_bwd,
    const __fp16* weight_hh_bwd,
    const __fp16* bias_ih_bwd,
    const __fp16* bias_hh_bwd,
    __fp16* output,
    size_t batch_size,
    size_t seq_len,
    size_t input_size,
    size_t hidden_size
) {
    const size_t output_size = 2 * hidden_size;

    for (size_t b = 0; b < batch_size; ++b) {
        const __fp16* batch_input = input + b * seq_len * input_size;
        __fp16* batch_output = output + b * seq_len * output_size;

        // ── Forward pass ──
        std::vector<__fp16> h_fwd(hidden_size, static_cast<__fp16>(0.0f));
        std::vector<__fp16> c_fwd(hidden_size, static_cast<__fp16>(0.0f));
        std::vector<__fp16> h_new(hidden_size);
        std::vector<__fp16> c_new(hidden_size);

        for (size_t t = 0; t < seq_len; ++t) {
            const __fp16* x_t = batch_input + t * input_size;

            cactus_lstm_cell_f16(
                x_t, h_fwd.data(), c_fwd.data(),
                weight_ih_fwd, weight_hh_fwd, bias_ih_fwd, bias_hh_fwd,
                h_new.data(), c_new.data(),
                1, input_size, hidden_size
            );

            // Store h_fwd in output (first half)
            __fp16* out_t = batch_output + t * output_size;
            std::copy(h_new.begin(), h_new.end(), out_t);

            std::copy(h_new.begin(), h_new.end(), h_fwd.begin());
            std::copy(c_new.begin(), c_new.end(), c_fwd.begin());
        }

        // ── Backward pass ──
        std::vector<__fp16> h_bwd(hidden_size, static_cast<__fp16>(0.0f));
        std::vector<__fp16> c_bwd(hidden_size, static_cast<__fp16>(0.0f));

        for (size_t t_rev = 0; t_rev < seq_len; ++t_rev) {
            size_t t = seq_len - 1 - t_rev;
            const __fp16* x_t = batch_input + t * input_size;

            cactus_lstm_cell_f16(
                x_t, h_bwd.data(), c_bwd.data(),
                weight_ih_bwd, weight_hh_bwd, bias_ih_bwd, bias_hh_bwd,
                h_new.data(), c_new.data(),
                1, input_size, hidden_size
            );

            // Store h_bwd in output (second half)
            __fp16* out_t = batch_output + t * output_size;
            std::copy(h_new.begin(), h_new.end(), out_t + hidden_size);

            std::copy(h_new.begin(), h_new.end(), h_bwd.begin());
            std::copy(c_new.begin(), c_new.end(), c_bwd.begin());
        }
    }
}
