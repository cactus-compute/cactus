#include "model.h"

namespace cactus { namespace engine {

PyAnnoteModel::PyAnnoteModel() : Model() {}
PyAnnoteModel::PyAnnoteModel(const Config& config) : Model(config) {}

bool PyAnnoteModel::init(const std::string& model_folder, size_t,
                          const std::string&, bool) {
    model_folder_path_ = model_folder;
    try {
        auto* gb = new CactusGraph();
        graph_handle_ = gb;
        load_weights_to_graph(gb);
        initialized_ = true;
        return true;
    } catch (const std::exception& e) {
        CACTUS_LOG_ERROR("pyannote", "Init failed: " << e.what());
        return false;
    }
}

void PyAnnoteModel::load_weights_to_graph(CactusGraph* gb) {
    const std::string& p = model_folder_path_;

    w_.sinc_filters = gb->mmap_weights(p + "/sincnet_sinc_filters.weights");
    w_.wav_norm_weight = gb->mmap_weights(p + "/sincnet_wav_norm_weight.weights");
    w_.wav_norm_bias = gb->mmap_weights(p + "/sincnet_wav_norm_bias.weights");
    w_.norm0_weight = gb->mmap_weights(p + "/sincnet_norm0_weight.weights");
    w_.norm0_bias = gb->mmap_weights(p + "/sincnet_norm0_bias.weights");
    w_.conv1_weight = gb->mmap_weights(p + "/sincnet_conv1_weight.weights");
    w_.conv1_bias = gb->mmap_weights(p + "/sincnet_conv1_bias.weights");
    w_.norm1_weight = gb->mmap_weights(p + "/sincnet_norm1_weight.weights");
    w_.norm1_bias = gb->mmap_weights(p + "/sincnet_norm1_bias.weights");
    w_.conv2_weight = gb->mmap_weights(p + "/sincnet_conv2_weight.weights");
    w_.conv2_bias = gb->mmap_weights(p + "/sincnet_conv2_bias.weights");
    w_.norm2_weight = gb->mmap_weights(p + "/sincnet_norm2_weight.weights");
    w_.norm2_bias = gb->mmap_weights(p + "/sincnet_norm2_bias.weights");

    for (int layer = 0; layer < 4; ++layer) {
        auto& lw = w_.lstm_layers[layer];
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

    w_.linear0_weight = gb->mmap_weights(p + "/linear_0_weight.weights");
    w_.linear0_bias = gb->mmap_weights(p + "/linear_0_bias.weights");
    w_.linear1_weight = gb->mmap_weights(p + "/linear_1_weight.weights");
    w_.linear1_bias = gb->mmap_weights(p + "/linear_1_bias.weights");
    w_.classifier_weight = gb->mmap_weights(p + "/classifier_weight.weights");
    w_.classifier_bias = gb->mmap_weights(p + "/classifier_bias.weights");

    audio_input_ = gb->input({1, 1, 160000}, Precision::FP16);

    size_t x = gb->groupnorm(audio_input_, w_.wav_norm_weight, w_.wav_norm_bias, 1, 1e-5f);
    x = gb->conv1d(x, w_.sinc_filters, 10);
    x = gb->abs(x);
    x = gb->maxpool1d(x, 3, 3);
    x = gb->groupnorm(x, w_.norm0_weight, w_.norm0_bias, 80, 1e-5f);
    x = gb->leaky_relu(x, 0.01f);

    x = gb->conv1d(x, w_.conv1_weight, w_.conv1_bias, 1);
    x = gb->maxpool1d(x, 3, 3);
    x = gb->groupnorm(x, w_.norm1_weight, w_.norm1_bias, 60, 1e-5f);
    x = gb->leaky_relu(x, 0.01f);

    x = gb->conv1d(x, w_.conv2_weight, w_.conv2_bias, 1);
    x = gb->maxpool1d(x, 3, 3);
    x = gb->groupnorm(x, w_.norm2_weight, w_.norm2_bias, 60, 1e-5f);
    x = gb->leaky_relu(x, 0.01f);

    x = gb->transposeN(x, {0, 2, 1});

    for (int layer = 0; layer < 4; ++layer) {
        auto& lw = w_.lstm_layers[layer];
        x = gb->bilstm_sequence(x,
            lw.w_ih_fwd, lw.w_hh_fwd, lw.b_ih_fwd, lw.b_hh_fwd,
            lw.w_ih_bwd, lw.w_hh_bwd, lw.b_ih_bwd, lw.b_hh_bwd);
    }

    const auto& bilstm_shape = gb->get_output_buffer(x).shape;
    size_t T = bilstm_shape[1];
    x = gb->reshape(x, {T, bilstm_shape[2]});

    x = gb->add(gb->matmul(x, w_.linear0_weight, true), w_.linear0_bias);
    x = gb->leaky_relu(x, 0.01f);
    x = gb->add(gb->matmul(x, w_.linear1_weight, true), w_.linear1_bias);
    x = gb->leaky_relu(x, 0.01f);

    x = gb->add(gb->matmul(x, w_.classifier_weight, true), w_.classifier_bias);
    x = gb->softmax(x, -1);
    x = gb->reshape(x, {1, T, 7});

    output_node_ = x;
}

std::vector<float> PyAnnoteModel::diarize(const float* pcm_f32, size_t num_samples) {
    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    if (!gb) throw std::runtime_error("PyAnnote model not initialized");

    std::vector<__fp16> audio(160000, static_cast<__fp16>(0.0f));
    size_t copy_len = std::min(num_samples, static_cast<size_t>(160000));
    for (size_t i = 0; i < copy_len; ++i)
        audio[i] = static_cast<__fp16>(pcm_f32[i]);

    gb->set_input(audio_input_, audio.data(), Precision::FP16);
    gb->execute();

    const auto& out_buf = gb->get_output_buffer(output_node_);
    const __fp16* out_data = out_buf.data_as<__fp16>();
    size_t total = out_buf.total_size;

    std::vector<float> result(total);
    for (size_t i = 0; i < total; ++i)
        result[i] = static_cast<float>(out_data[i]);
    return result;
}

}} // namespace cactus::engine
