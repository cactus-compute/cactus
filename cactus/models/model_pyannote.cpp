#include "model.h"
#include "../graph/graph.h"
#include "../kernel/kernel.h"
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace cactus {
namespace engine {

static constexpr size_t WINDOW_SAMPLES    = 160000;
static constexpr size_t FRAMES_PER_WINDOW = 589;
static constexpr int    NUM_CLASSES       = 7;
static constexpr int    NUM_SPEAKERS      = 3;

PyAnnoteModel::PyAnnoteModel() : Model() {}
PyAnnoteModel::PyAnnoteModel(const Config& config) : Model(config) {}

void PyAnnoteModel::load_weights_to_graph(CactusGraph* gb) {
    const std::string& p = model_folder_path_;

    weight_nodes_.sinc_filters = gb->mmap_weights(p + "/sincnet_sinc_filters.weights");
    weight_nodes_.wav_norm_weight = gb->mmap_weights(p + "/sincnet_wav_norm_weight.weights");
    weight_nodes_.wav_norm_bias = gb->mmap_weights(p + "/sincnet_wav_norm_bias.weights");
    weight_nodes_.norm0_weight = gb->mmap_weights(p + "/sincnet_norm0_weight.weights");
    weight_nodes_.norm0_bias = gb->mmap_weights(p + "/sincnet_norm0_bias.weights");
    weight_nodes_.conv1_weight = gb->mmap_weights(p + "/sincnet_conv1_weight.weights");
    weight_nodes_.conv1_bias = gb->mmap_weights(p + "/sincnet_conv1_bias.weights");
    weight_nodes_.norm1_weight = gb->mmap_weights(p + "/sincnet_norm1_weight.weights");
    weight_nodes_.norm1_bias = gb->mmap_weights(p + "/sincnet_norm1_bias.weights");
    weight_nodes_.conv2_weight = gb->mmap_weights(p + "/sincnet_conv2_weight.weights");
    weight_nodes_.conv2_bias = gb->mmap_weights(p + "/sincnet_conv2_bias.weights");
    weight_nodes_.norm2_weight = gb->mmap_weights(p + "/sincnet_norm2_weight.weights");
    weight_nodes_.norm2_bias = gb->mmap_weights(p + "/sincnet_norm2_bias.weights");

    for (int layer = 0; layer < 4; ++layer) {
        auto& lw = weight_nodes_.lstm_layers[layer];
        std::string fwd = p + "/lstm_fwd_" + std::to_string(layer);
        std::string bwd = p + "/lstm_bwd_" + std::to_string(layer);
        lw.w_ih_fwd = gb->mmap_weights(fwd + "_weight_ih.weights");
        lw.w_hh_fwd = gb->mmap_weights(fwd + "_weight_hh.weights");
        lw.b_ih_fwd = gb->mmap_weights(fwd + "_bias_ih.weights");
        lw.b_hh_fwd = gb->mmap_weights(fwd + "_bias_hh.weights");
        lw.w_ih_bwd = gb->mmap_weights(bwd + "_weight_ih.weights");
        lw.w_hh_bwd = gb->mmap_weights(bwd + "_weight_hh.weights");
        lw.b_ih_bwd = gb->mmap_weights(bwd + "_bias_ih.weights");
        lw.b_hh_bwd = gb->mmap_weights(bwd + "_bias_hh.weights");
    }

    weight_nodes_.linear0_weight = gb->mmap_weights(p + "/linear_0_weight.weights");
    weight_nodes_.linear0_bias = gb->mmap_weights(p + "/linear_0_bias.weights");
    weight_nodes_.linear1_weight = gb->mmap_weights(p + "/linear_1_weight.weights");
    weight_nodes_.linear1_bias = gb->mmap_weights(p + "/linear_1_bias.weights");
    weight_nodes_.classifier_weight = gb->mmap_weights(p + "/classifier_weight.weights");
    weight_nodes_.classifier_bias = gb->mmap_weights(p + "/classifier_bias.weights");
}

void PyAnnoteModel::build_graph() {
    audio_input_ = graph_.input({1, 1, 160000}, Precision::FP16);

    size_t x = graph_.groupnorm(audio_input_, weight_nodes_.wav_norm_weight, weight_nodes_.wav_norm_bias, 1, 1e-5f);
    x = graph_.conv1d(x, weight_nodes_.sinc_filters, 10);
    x = graph_.abs(x);
    x = graph_.maxpool1d(x, 3, 3);
    x = graph_.groupnorm(x, weight_nodes_.norm0_weight, weight_nodes_.norm0_bias, 80, 1e-5f);
    x = graph_.leaky_relu(x, 0.01f);

    x = graph_.conv1d(x, weight_nodes_.conv1_weight, weight_nodes_.conv1_bias, 1);
    x = graph_.maxpool1d(x, 3, 3);
    x = graph_.groupnorm(x, weight_nodes_.norm1_weight, weight_nodes_.norm1_bias, 60, 1e-5f);
    x = graph_.leaky_relu(x, 0.01f);

    x = graph_.conv1d(x, weight_nodes_.conv2_weight, weight_nodes_.conv2_bias, 1);
    x = graph_.maxpool1d(x, 3, 3);
    x = graph_.groupnorm(x, weight_nodes_.norm2_weight, weight_nodes_.norm2_bias, 60, 1e-5f);
    x = graph_.leaky_relu(x, 0.01f);

    x = graph_.transposeN(x, {0, 2, 1});

    for (int layer = 0; layer < 4; ++layer) {
        auto& lw = weight_nodes_.lstm_layers[layer];
        x = graph_.bilstm_sequence(x,
                                   lw.w_ih_fwd, lw.w_hh_fwd, lw.b_ih_fwd, lw.b_hh_fwd,
                                   lw.w_ih_bwd, lw.w_hh_bwd, lw.b_ih_bwd, lw.b_hh_bwd);
    }

    const auto& bilstm_shape = graph_.get_output_buffer(x).shape;
    size_t T = bilstm_shape[1];
    x = graph_.reshape(x, {T, bilstm_shape[2]});

    x = graph_.add(graph_.matmul(x, weight_nodes_.linear0_weight, true), weight_nodes_.linear0_bias);
    x = graph_.leaky_relu(x, 0.01f);
    x = graph_.add(graph_.matmul(x, weight_nodes_.linear1_weight, true), weight_nodes_.linear1_bias);
    x = graph_.leaky_relu(x, 0.01f);

    x = graph_.add(graph_.matmul(x, weight_nodes_.classifier_weight, true), weight_nodes_.classifier_bias);
    x = graph_.softmax(x, -1);
    x = graph_.reshape(x, {1, T, 7});

    output_node_ = x;
}

bool PyAnnoteModel::init(const std::string& model_folder, size_t context_size,
                         const std::string& system_prompt, bool do_warmup) {
    (void)context_size;
    (void)system_prompt;
    (void)do_warmup;

    if (initialized_) {
        return true;
    }

    model_folder_path_ = model_folder;

    try {
        load_weights_to_graph(&graph_);
        build_graph();

        chunk_buf_.resize(WINDOW_SAMPLES);

        hamming_.resize(FRAMES_PER_WINDOW);
        for (size_t i = 0; i < FRAMES_PER_WINDOW; ++i)
            hamming_[i] = 0.54f - 0.46f * std::cos(2.0f * M_PI * i / (FRAMES_PER_WINDOW - 1));

        initialized_ = true;
        return true;
    } catch (const std::exception& e) {
        initialized_ = false;
        return false;
    }
}

std::vector<float> PyAnnoteModel::diarize(const float* pcm_f32, size_t num_samples, size_t step_samples, bool raw_powerset) {
    if (!initialized_) throw std::runtime_error("PyAnnote model not initialized");

    const size_t total_frames = std::max(
        FRAMES_PER_WINDOW,
        static_cast<size_t>(std::round((double)num_samples * FRAMES_PER_WINDOW / WINDOW_SAMPLES))
    );

    std::vector<float> aggregated(total_frames * NUM_CLASSES, 0.0f);
    std::vector<float> weight_sum(total_frames, 0.0f);

    auto process_chunk = [&](size_t chunk_start) {
        const size_t copy_len = std::min(WINDOW_SAMPLES, num_samples - chunk_start);
        for (size_t i = 0; i < copy_len; ++i)
            chunk_buf_[i] = static_cast<__fp16>(pcm_f32[chunk_start + i]);
        std::fill(chunk_buf_.begin() + copy_len, chunk_buf_.end(), static_cast<__fp16>(0.0f));

        graph_.set_input(audio_input_, chunk_buf_.data(), Precision::FP16);
        graph_.execute();

        const __fp16* chunk_scores = graph_.get_output_buffer(output_node_).data_as<__fp16>();
        const size_t frame_offset = static_cast<size_t>(std::round((double)chunk_start * FRAMES_PER_WINDOW / WINDOW_SAMPLES));

        for (size_t f = 0; f < FRAMES_PER_WINDOW; ++f) {
            const size_t out_f = frame_offset + f;
            if (out_f >= total_frames) break;
            const float w = hamming_[f];
            weight_sum[out_f] += w;
            for (int c = 0; c < NUM_CLASSES; ++c)
                aggregated[out_f * NUM_CLASSES + c] += w * static_cast<float>(chunk_scores[f * NUM_CLASSES + c]);
        }
    };

    const size_t last_start = num_samples > WINDOW_SAMPLES ? num_samples - WINDOW_SAMPLES : 0;
    bool last_processed = false;
    for (size_t s = 0; s + WINDOW_SAMPLES <= num_samples; s += step_samples) {
        if (s >= last_start) last_processed = true;
        process_chunk(s);
    }
    if (!last_processed)
        process_chunk(last_start);

    for (size_t f = 0; f < total_frames; ++f) {
        if (weight_sum[f] > 0.0f) {
            const float inv_w = 1.0f / weight_sum[f];
            float* agg = aggregated.data() + f * NUM_CLASSES;
            for (int c = 0; c < NUM_CLASSES; ++c)
                agg[c] *= inv_w;
        }
    }

    if (raw_powerset) {
        return aggregated;
    }

    std::vector<float> speaker_probs(total_frames * NUM_SPEAKERS, 0.0f);
    for (size_t f = 0; f < total_frames; ++f) {
        const float* agg = aggregated.data() + f * NUM_CLASSES;
        float* dst = speaker_probs.data() + f * NUM_SPEAKERS;
        dst[0] = agg[1] + agg[4] + agg[5];
        dst[1] = agg[2] + agg[4] + agg[6];
        dst[2] = agg[3] + agg[5] + agg[6];
    }

    return speaker_probs;
}

}
}
