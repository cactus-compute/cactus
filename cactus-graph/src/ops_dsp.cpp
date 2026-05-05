#include "../cactus_graph.h"
#include "cactus_kernels.h"
#include <cstring>
#include <cmath>
#include <stdexcept>

void compute_rfft_node(
    GraphNode& node,
    const nodes_vector& nodes,
    const node_index_map_t& node_index_map) {

    const auto& input = get_input(node, 0, nodes, node_index_map);
    size_t n = input.total_size;

    if (input.precision == Precision::FP32) {
        cactus_rfft_f32_1d(
            input.data_as<float>(),
            node.output_buffer.data_as<float>(),
            n, "backward");
    } else if (input.precision == Precision::FP16) {
        std::vector<float> in_f32(n);
        Quantization::fp16_to_fp32(input.data_as<__fp16>(), in_f32.data(), n);
        std::vector<float> out_f32((n / 2 + 1) * 2);
        cactus_rfft_f32_1d(in_f32.data(), out_f32.data(), n, "backward");
        Quantization::fp32_to_fp16(out_f32.data(), node.output_buffer.data_as<__fp16>(), out_f32.size());
    } else {
        throw std::runtime_error("RFFT requires FP32 or FP16 input");
    }
}

void compute_irfft_node(
    GraphNode& node,
    const nodes_vector& nodes,
    const node_index_map_t& node_index_map) {

    const auto& input = get_input(node, 0, nodes, node_index_map);
    size_t n = node.output_buffer.total_size;

    if (input.precision == Precision::FP32) {
        cactus_irfft_f32_1d(
            input.data_as<float>(),
            node.output_buffer.data_as<float>(),
            n, "backward");
    } else if (input.precision == Precision::FP16) {
        size_t in_len = input.total_size;
        std::vector<float> in_f32(in_len);
        Quantization::fp16_to_fp32(input.data_as<__fp16>(), in_f32.data(), in_len);
        std::vector<float> out_f32(n);
        cactus_irfft_f32_1d(in_f32.data(), out_f32.data(), n, "backward");
        Quantization::fp32_to_fp16(out_f32.data(), node.output_buffer.data_as<__fp16>(), n);
    } else {
        throw std::runtime_error("IRFFT requires FP32 or FP16 input");
    }
}

void compute_mel_filter_bank_node(
    GraphNode& node,
    const nodes_vector&,
    const node_index_map_t&) {

    size_t num_freq_bins = node.output_buffer.shape[0];
    size_t num_mel = node.output_buffer.shape[1];

    const char* norm_strs[] = {nullptr, "slaney"};
    const char* scale_strs[] = {"htk", "kaldi", "slaney"};

    int norm_idx = node.params.mel_norm_type;
    int scale_idx = node.params.mel_scale_type;
    const char* norm = (norm_idx >= 0 && norm_idx <= 1) ? norm_strs[norm_idx] : "slaney";
    const char* scale = (scale_idx >= 0 && scale_idx <= 2) ? scale_strs[scale_idx] : "slaney";

    std::vector<float> filters(num_mel * num_freq_bins);
    cactus_generate_mel_filter_bank(
        filters.data(),
        static_cast<int>(num_freq_bins),
        static_cast<int>(num_mel),
        node.params.min_frequency,
        node.params.max_frequency,
        static_cast<int>(node.params.sampling_rate),
        norm, scale, false);

    if (node.output_buffer.precision == Precision::FP32) {
        std::memcpy(node.output_buffer.get_data(), filters.data(), filters.size() * sizeof(float));
    } else if (node.output_buffer.precision == Precision::FP16) {
        Quantization::fp32_to_fp16(filters.data(), node.output_buffer.data_as<__fp16>(), filters.size());
    }
}

void compute_spectrogram_node(
    GraphNode& node,
    const nodes_vector& nodes,
    const node_index_map_t& node_index_map) {

    const auto& waveform_buf = get_input(node, 0, nodes, node_index_map);
    const auto& mel_buf = get_input(node, 1, nodes, node_index_map);

    size_t waveform_length = waveform_buf.total_size;
    size_t frame_length = node.params.num_fft_bins;
    size_t hop_length = node.params.hop_length;
    size_t fft_length = node.params.stride;
    float power = node.params.power;
    bool center = node.params.center;
    float mel_floor = node.params.mel_floor;
    float dither_val = node.params.dither;
    float preemph = node.params.preemphasis_coef;
    bool rm_dc = node.params.remove_dc_offset;

    const char* pad_strs[] = {"reflect", "constant"};
    const char* pad_mode = (node.params.pad_mode_type >= 0 && node.params.pad_mode_type <= 1)
        ? pad_strs[node.params.pad_mode_type] : "reflect";

    const char* log_strs[] = {nullptr, "log", "log10", "dB"};
    const char* log_mel = (node.params.log_mel_mode >= 0 && node.params.log_mel_mode <= 3)
        ? log_strs[node.params.log_mel_mode] : nullptr;

    std::vector<float> waveform_f32(waveform_length);
    if (waveform_buf.precision == Precision::FP32) {
        std::memcpy(waveform_f32.data(), waveform_buf.get_data(), waveform_length * sizeof(float));
    } else if (waveform_buf.precision == Precision::FP16) {
        Quantization::fp16_to_fp32(waveform_buf.data_as<__fp16>(), waveform_f32.data(), waveform_length);
    }

    std::vector<float> mel_f32(mel_buf.total_size);
    if (mel_buf.precision == Precision::FP32) {
        std::memcpy(mel_f32.data(), mel_buf.get_data(), mel_buf.total_size * sizeof(float));
    } else if (mel_buf.precision == Precision::FP16) {
        Quantization::fp16_to_fp32(mel_buf.data_as<__fp16>(), mel_f32.data(), mel_buf.total_size);
    }

    size_t num_frequency_bins = fft_length / 2 + 1;
    size_t num_mel_bins = mel_buf.total_size / num_frequency_bins;
    size_t pad_len = center ? frame_length / 2 : 0;
    size_t padded_len = waveform_length + 2 * pad_len;
    size_t num_frames = 1 + (padded_len - frame_length) / hop_length;

    std::vector<float> output_f32(num_mel_bins * num_frames);

    cactus_compute_spectrogram_f32(
        waveform_f32.data(), waveform_length,
        nullptr, 0,
        frame_length, hop_length, &fft_length,
        output_f32.data(), power,
        center, pad_mode, true,
        dither_val,
        preemph != 0.0f ? &preemph : nullptr,
        mel_f32.data(), mel_f32.size(),
        mel_floor, log_mel,
        1.0f, 1e-10f, nullptr, rm_dc);

    if (node.output_buffer.precision == Precision::FP32) {
        std::memcpy(node.output_buffer.get_data(), output_f32.data(), output_f32.size() * sizeof(float));
    } else if (node.output_buffer.precision == Precision::FP16) {
        Quantization::fp32_to_fp16(output_f32.data(), node.output_buffer.data_as<__fp16>(), output_f32.size());
    }
}
