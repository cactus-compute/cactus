#include "model.h"
#include "../graph/graph.h"
#include "../kernel/kernel.h"
#include <stdexcept>

namespace cactus {
namespace engine {

SileroVADModel::SileroVADModel() : Model() {
    weight_nodes_.encoder_blocks.resize(4);
}

SileroVADModel::SileroVADModel(const Config& config) : Model(config) {
    weight_nodes_.encoder_blocks.resize(4);
}

SileroVADModel::~SileroVADModel() = default;

void SileroVADModel::load_weights_to_graph(CactusGraph* gb) {
    weight_nodes_.stft_basis = gb->mmap_weights(model_folder_path_ + "/stft_basis.weights");

    for (uint32_t i = 0; i < 4; i++) {
        std::string block_prefix = model_folder_path_ + "/encoder_block_" + std::to_string(i) + "_";
        weight_nodes_.encoder_blocks[i].conv_weight = gb->mmap_weights(block_prefix + "conv_weight.weights");
        weight_nodes_.encoder_blocks[i].conv_bias = gb->mmap_weights(block_prefix + "conv_bias.weights");
    }

    weight_nodes_.lstm_weight_ih = gb->mmap_weights(model_folder_path_ + "/lstm_weight_ih.weights");
    weight_nodes_.lstm_weight_hh = gb->mmap_weights(model_folder_path_ + "/lstm_weight_hh.weights");
    weight_nodes_.lstm_bias_ih = gb->mmap_weights(model_folder_path_ + "/lstm_bias_ih.weights");
    weight_nodes_.lstm_bias_hh = gb->mmap_weights(model_folder_path_ + "/lstm_bias_hh.weights");

    weight_nodes_.output_conv_weight = gb->mmap_weights(model_folder_path_ + "/output_conv_weight.weights");
    weight_nodes_.output_conv_bias = gb->mmap_weights(model_folder_path_ + "/output_conv_bias.weights");
}

void SileroVADModel::build_graph() {
    const size_t input_size = CONTEXT_SIZE + CHUNK_SIZE + REFLECT_PAD_SIZE;

    graph_nodes_.input = graph_.input({1, 1, input_size}, Precision::FP16);
    graph_nodes_.h_prev = graph_.input({1, HIDDEN_SIZE}, Precision::FP16);
    graph_nodes_.c_prev = graph_.input({1, HIDDEN_SIZE}, Precision::FP16);

    auto stft = graph_.stft_magnitude(graph_nodes_.input, weight_nodes_.stft_basis, 128, 129);

    const size_t strides[4] = {1, 2, 2, 1};
    auto x = stft;
    for (uint32_t i = 0; i < 4; i++) {
        auto conv = graph_.conv1d_k3(x, weight_nodes_.encoder_blocks[i].conv_weight, strides[i]);
        auto bias_reshaped = graph_.reshape(weight_nodes_.encoder_blocks[i].conv_bias,
            {1, graph_.get_output_buffer(weight_nodes_.encoder_blocks[i].conv_bias).shape[0], 1});
        auto with_bias = graph_.add(conv, bias_reshaped);
        x = graph_.relu(with_bias);
    }

    auto x_squeezed = graph_.reshape(x, {1, HIDDEN_SIZE});

    graph_nodes_.lstm_output = graph_.lstm_cell(
        x_squeezed,
        graph_nodes_.h_prev,
        graph_nodes_.c_prev,
        weight_nodes_.lstm_weight_ih,
        weight_nodes_.lstm_weight_hh,
        weight_nodes_.lstm_bias_ih,
        weight_nodes_.lstm_bias_hh
    );

    auto h_new = graph_.slice(graph_nodes_.lstm_output, 2, 0, 1);
    auto c_new = graph_.slice(graph_nodes_.lstm_output, 2, 1, 1);
    graph_nodes_.h_new = graph_.reshape(h_new, {1, HIDDEN_SIZE});
    graph_nodes_.c_new = graph_.reshape(c_new, {1, HIDDEN_SIZE});

    auto h_relu = graph_.relu(graph_nodes_.h_new);
    auto h_unsqueezed = graph_.reshape(h_relu, {1, HIDDEN_SIZE, 1});
    auto logits = graph_.conv1d(h_unsqueezed, weight_nodes_.output_conv_weight, weight_nodes_.output_conv_bias, 1);
    graph_nodes_.output = graph_.sigmoid(logits);
}

bool SileroVADModel::init(const std::string& model_folder, size_t context_size,
                          const std::string& system_prompt, bool do_warmup) {
    (void)context_size;
    (void)system_prompt;
    (void)do_warmup;

    if (initialized_) {
        return true;
    }

    state_.h.resize(HIDDEN_SIZE, (__fp16)0.0f);
    state_.c.resize(HIDDEN_SIZE, (__fp16)0.0f);
    state_.context.resize(CONTEXT_SIZE, 0.0f);
    state_.input_buf.resize(CONTEXT_SIZE + CHUNK_SIZE + REFLECT_PAD_SIZE, 0.0f);
    state_.input_fp16.resize(CONTEXT_SIZE + CHUNK_SIZE + REFLECT_PAD_SIZE);

    model_folder_path_ = model_folder;

    try {
        load_weights_to_graph(&graph_);
        build_graph();
        initialized_ = true;
        return true;
    } catch (const std::exception& e) {
        initialized_ = false;
        return false;
    }
}

float SileroVADModel::process_chunk(const std::vector<float>& audio_chunk) {
    const float* audio = audio_chunk.data();
    size_t samples = audio_chunk.size();

    if (samples != CHUNK_SIZE) {
        throw std::runtime_error("SileroVAD: Expected 512 samples");
    }
    if (!initialized_) {
        throw std::runtime_error("SileroVAD: Not initialized");
    }

    const size_t input_size = CONTEXT_SIZE + CHUNK_SIZE + REFLECT_PAD_SIZE;

    float* input_buf = state_.input_buf.data();
    __fp16* input_fp16 = state_.input_fp16.data();

    std::memcpy(input_buf, state_.context.data(), CONTEXT_SIZE * sizeof(float));
    std::memcpy(input_buf + CONTEXT_SIZE, audio, CHUNK_SIZE * sizeof(float));

    for (size_t i = 0; i < REFLECT_PAD_SIZE; i++) {
        input_buf[CONTEXT_SIZE + CHUNK_SIZE + i] =
            input_buf[CONTEXT_SIZE + CHUNK_SIZE - 2 - i];
    }

    cactus_fp32_to_fp16(input_buf, input_fp16, input_size);

    graph_.set_input(graph_nodes_.input, input_fp16, Precision::FP16);
    graph_.set_input(graph_nodes_.h_prev, state_.h.data(), Precision::FP16);
    graph_.set_input(graph_nodes_.c_prev, state_.c.data(), Precision::FP16);
    graph_.execute();

    void* h_out = graph_.get_output(graph_nodes_.h_new);
    void* c_out = graph_.get_output(graph_nodes_.c_new);
    void* vad_out = graph_.get_output(graph_nodes_.output);

    const __fp16* h_new = static_cast<const __fp16*>(h_out);
    const __fp16* c_new = static_cast<const __fp16*>(c_out);
    const __fp16* vad_score = static_cast<const __fp16*>(vad_out);

    std::memcpy(state_.h.data(), h_new, HIDDEN_SIZE * sizeof(__fp16));
    std::memcpy(state_.c.data(), c_new, HIDDEN_SIZE * sizeof(__fp16));

    float vad_score_f32 = static_cast<float>(vad_score[0]);

    std::memcpy(state_.context.data(), input_buf + CHUNK_SIZE, CONTEXT_SIZE * sizeof(float));

    return vad_score_f32;
}

void SileroVADModel::reset_states() {
    state_.h.assign(HIDDEN_SIZE, (__fp16)0.0f);
    state_.c.assign(HIDDEN_SIZE, (__fp16)0.0f);
    state_.context.assign(CONTEXT_SIZE, 0.0f);
}

std::vector<SileroVADModel::SpeechTimestamp> SileroVADModel::get_speech_timestamps(const std::vector<float>& audio, const SpeechTimestampsOptions& options) {
    const size_t audio_length_samples = audio.size();
    const size_t window_size_samples = options.window_size_samples;
    const float min_speech_samples = options.sampling_rate * options.min_speech_duration_ms / 1000.0f;
    const float speech_pad_samples = options.sampling_rate * options.speech_pad_ms / 1000.0f;
    const float max_speech_samples = options.sampling_rate * options.max_speech_duration_s - window_size_samples - 2 * speech_pad_samples;
    const float min_silence_samples = options.sampling_rate * options.min_silence_duration_ms / 1000.0f;
    const float neg_threshold = (options.neg_threshold == 0.0f)
        ? std::max(options.threshold - 0.15f, 0.01f)
        : options.neg_threshold;
    reset_states();

    std::vector<float> speech_probs;
    speech_probs.reserve(audio_length_samples / window_size_samples + 1);

    std::vector<float> chunk(window_size_samples, 0.0f);

    for (size_t current_start = 0; current_start < audio_length_samples; current_start += window_size_samples) {
        size_t remaining = audio_length_samples - current_start;
        size_t copy_len = std::min(remaining, window_size_samples);

        std::memcpy(chunk.data(), audio.data() + current_start, copy_len * sizeof(float));

        if (copy_len < window_size_samples) {
            std::memset(chunk.data() + copy_len, 0, (window_size_samples - copy_len) * sizeof(float));
        }

        float speech_prob = process_chunk(chunk);
        speech_probs.push_back(speech_prob);
    }

    bool triggered = false;
    std::vector<SpeechTimestamp> speeches;
    SpeechTimestamp current_speech = {0, 0};
    size_t temp_end = 0;
    size_t prev_end = 0;
    size_t next_start = 0;
    std::vector<std::pair<size_t, size_t>> possible_ends;

    const float min_silence_samples_at_max_speech = options.sampling_rate * options.min_silence_at_max_speech / 1000.0f;

    for (size_t i = 0; i < speech_probs.size(); ++i) {
        float speech_prob = speech_probs[i];
        size_t cur_sample = window_size_samples * i;

        if (speech_prob >= options.threshold && temp_end) {
            size_t sil_dur = cur_sample - temp_end;
            if (sil_dur > min_silence_samples_at_max_speech) {
                possible_ends.push_back({temp_end, sil_dur});
            }
            temp_end = 0;
            if (next_start < prev_end) {
                next_start = cur_sample;
            }
        }

        if (speech_prob >= options.threshold && !triggered) {
            triggered = true;
            current_speech.start = cur_sample;
            continue;
        }

        if (triggered && (cur_sample - current_speech.start > max_speech_samples)) {
            if (options.use_max_poss_sil_at_max_speech && !possible_ends.empty()) {
                auto max_silence = std::max_element(possible_ends.begin(), possible_ends.end(),
                    [](const std::pair<size_t, size_t>& a, const std::pair<size_t, size_t>& b) {
                        return a.second < b.second;
                    });
                prev_end = max_silence->first;
                size_t dur = max_silence->second;
                current_speech.end = prev_end;
                speeches.push_back(current_speech);
                current_speech = {0, 0};
                next_start = prev_end + dur;

                if (next_start < prev_end + cur_sample) {
                    current_speech.start = next_start;
                } else {
                    triggered = false;
                }
                prev_end = next_start = temp_end = 0;
                possible_ends.clear();
            } else {
                if (prev_end) {
                    current_speech.end = prev_end;
                    speeches.push_back(current_speech);
                    current_speech = {0, 0};
                    if (next_start < prev_end) {
                        triggered = false;
                    } else {
                        current_speech.start = next_start;
                    }
                    prev_end = next_start = temp_end = 0;
                    possible_ends.clear();
                } else {
                    current_speech.end = cur_sample;
                    speeches.push_back(current_speech);
                    current_speech = {0, 0};
                    prev_end = next_start = temp_end = 0;
                    triggered = false;
                    possible_ends.clear();
                    continue;
                }
            }
        }

        if (speech_prob < neg_threshold && triggered) {
            if (!temp_end) {
                temp_end = cur_sample;
            }
            size_t sil_dur_now = cur_sample - temp_end;

            if (!options.use_max_poss_sil_at_max_speech && sil_dur_now > min_silence_samples_at_max_speech) {
                prev_end = temp_end;
            }

            if (sil_dur_now < min_silence_samples) {
                continue;
            } else {
                current_speech.end = temp_end;
                if ((current_speech.end - current_speech.start) > min_speech_samples) {
                    speeches.push_back(current_speech);
                }
                current_speech = {0, 0};
                prev_end = next_start = temp_end = 0;
                triggered = false;
                possible_ends.clear();
                continue;
            }
        }
    }

    if (triggered && (audio_length_samples - current_speech.start) > min_speech_samples) {
        current_speech.end = audio_length_samples;
        speeches.push_back(current_speech);
    }

    for (size_t i = 0; i < speeches.size(); ++i) {
        if (i == 0) {
            speeches[i].start = speeches[i].start > static_cast<size_t>(speech_pad_samples)
                ? speeches[i].start - static_cast<size_t>(speech_pad_samples)
                : 0;
        }
        if (i != speeches.size() - 1) {
            size_t silence_duration = speeches[i + 1].start - speeches[i].end;
            if (silence_duration < 2 * static_cast<size_t>(speech_pad_samples)) {
                speeches[i].end += silence_duration / 2;
                speeches[i + 1].start = speeches[i + 1].start > silence_duration / 2
                    ? speeches[i + 1].start - silence_duration / 2
                    : 0;
            } else {
                speeches[i].end = std::min(audio_length_samples, speeches[i].end + static_cast<size_t>(speech_pad_samples));
                speeches[i + 1].start = speeches[i + 1].start > static_cast<size_t>(speech_pad_samples)
                    ? speeches[i + 1].start - static_cast<size_t>(speech_pad_samples)
                    : 0;
            }
        } else {
            speeches[i].end = std::min(audio_length_samples, speeches[i].end + static_cast<size_t>(speech_pad_samples));
        }
    }

    return speeches;
}

size_t SileroVADModel::forward(const std::vector<float>&, const std::vector<uint32_t>&, bool) {
    return 0;
}

}
}
