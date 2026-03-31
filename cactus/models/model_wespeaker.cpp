#include "model.h"
#include "../graph/graph.h"
#include "../kernel/kernel.h"
#include <stdexcept>

namespace cactus {
namespace engine {

WeSpeakerModel::WeSpeakerModel() : Model() {}
WeSpeakerModel::WeSpeakerModel(const Config& config) : Model(config) {}

WeSpeakerModel::ResBlockWeights WeSpeakerModel::load_resblock(CactusGraph* gb, const std::string& prefix, bool has_shortcut) {
    ResBlockWeights rb;
    rb.has_shortcut = has_shortcut;
    rb.conv1_w = gb->mmap_weights(prefix + "_conv1_weight.weights");
    rb.conv2_w = gb->mmap_weights(prefix + "_conv2_weight.weights");
    rb.bn1_w = gb->mmap_weights(prefix + "_bn1_weight.weights");
    rb.bn1_b = gb->mmap_weights(prefix + "_bn1_bias.weights");
    rb.bn1_mean = gb->mmap_weights(prefix + "_bn1_running_mean.weights");
    rb.bn1_var = gb->mmap_weights(prefix + "_bn1_running_var.weights");
    rb.bn2_w = gb->mmap_weights(prefix + "_bn2_weight.weights");
    rb.bn2_b = gb->mmap_weights(prefix + "_bn2_bias.weights");
    rb.bn2_mean = gb->mmap_weights(prefix + "_bn2_running_mean.weights");
    rb.bn2_var = gb->mmap_weights(prefix + "_bn2_running_var.weights");
    if (has_shortcut) {
        rb.shortcut_conv_w = gb->mmap_weights(prefix + "_shortcut_0_weight.weights");
        rb.shortcut_bn_w = gb->mmap_weights(prefix + "_shortcut_1_weight.weights");
        rb.shortcut_bn_b = gb->mmap_weights(prefix + "_shortcut_1_bias.weights");
        rb.shortcut_bn_mean = gb->mmap_weights(prefix + "_shortcut_1_running_mean.weights");
        rb.shortcut_bn_var = gb->mmap_weights(prefix + "_shortcut_1_running_var.weights");
    }
    return rb;
}

size_t WeSpeakerModel::build_resblock(CactusGraph* gb, size_t x, const ResBlockWeights& rb, bool stride2) {
    size_t identity = x;
    size_t out = stride2 ? gb->conv2d_k3s2p1(x, rb.conv1_w) : gb->conv2d_k3s1p1(x, rb.conv1_w);
    out = gb->batchnorm(out, rb.bn1_w, rb.bn1_b, rb.bn1_mean, rb.bn1_var);
    out = gb->relu(out);

    out = gb->conv2d_k3s1p1(out, rb.conv2_w);
    out = gb->batchnorm(out, rb.bn2_w, rb.bn2_b, rb.bn2_mean, rb.bn2_var);

    if (rb.has_shortcut) {
        identity = gb->conv2d_k3s2p1(x, rb.shortcut_conv_w);
        identity = gb->batchnorm(identity, rb.shortcut_bn_w, rb.shortcut_bn_b,
                                  rb.shortcut_bn_mean, rb.shortcut_bn_var);
    }

    out = gb->add(out, identity);
    out = gb->relu(out);
    return out;
}

void WeSpeakerModel::load_weights_to_graph(CactusGraph* gb) {
    const std::string& p = model_folder_path_;

    weight_nodes_.conv1_w = gb->mmap_weights(p + "/resnet_conv1_weight.weights");
    weight_nodes_.bn1_w = gb->mmap_weights(p + "/resnet_bn1_weight.weights");
    weight_nodes_.bn1_b = gb->mmap_weights(p + "/resnet_bn1_bias.weights");
    weight_nodes_.bn1_mean = gb->mmap_weights(p + "/resnet_bn1_running_mean.weights");
    weight_nodes_.bn1_var = gb->mmap_weights(p + "/resnet_bn1_running_var.weights");

    auto load_layer = [&](const std::string& layer_name, int num_blocks, bool first_has_shortcut) {
        std::vector<ResBlockWeights> blocks;
        for (int i = 0; i < num_blocks; ++i) {
            std::string prefix = p + "/resnet_" + layer_name + "_" + std::to_string(i);
            blocks.push_back(load_resblock(gb, prefix, i == 0 && first_has_shortcut));
        }
        return blocks;
    };

    weight_nodes_.layer1 = load_layer("layer1", 3, false);
    weight_nodes_.layer2 = load_layer("layer2", 4, true);
    weight_nodes_.layer3 = load_layer("layer3", 6, true);
    weight_nodes_.layer4 = load_layer("layer4", 3, true);

    weight_nodes_.seg1_w = gb->mmap_weights(p + "/resnet_seg_1_weight.weights");
    weight_nodes_.seg1_b = gb->mmap_weights(p + "/resnet_seg_1_bias.weights");
}

void WeSpeakerModel::build_graph(size_t num_frames) {
    audio_input_ = graph_.input({1, 1, 80, num_frames}, Precision::FP16);

    size_t x = graph_.conv2d_k3s1p1(audio_input_, weight_nodes_.conv1_w);
    x = graph_.batchnorm(x, weight_nodes_.bn1_w, weight_nodes_.bn1_b, weight_nodes_.bn1_mean, weight_nodes_.bn1_var);
    x = graph_.relu(x);

    for (auto& rb : weight_nodes_.layer1) x = build_resblock(&graph_, x, rb, false);
    for (size_t i = 0; i < weight_nodes_.layer2.size(); ++i) x = build_resblock(&graph_, x, weight_nodes_.layer2[i], i == 0);
    for (size_t i = 0; i < weight_nodes_.layer3.size(); ++i) x = build_resblock(&graph_, x, weight_nodes_.layer3[i], i == 0);
    for (size_t i = 0; i < weight_nodes_.layer4.size(); ++i) x = build_resblock(&graph_, x, weight_nodes_.layer4[i], i == 0);

    x = graph_.stats_pool(x);
    x = graph_.add(graph_.matmul(x, weight_nodes_.seg1_w, true), weight_nodes_.seg1_b);

    output_node_ = x;
}

bool WeSpeakerModel::init(const std::string& model_folder, size_t context_size,
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
        initialized_ = true;
        return true;
    } catch (const std::exception& e) {
        initialized_ = false;
        return false;
    }
}

std::vector<float> WeSpeakerModel::embed(const float* fbank_features, size_t num_features) {
    if (!initialized_) throw std::runtime_error("WeSpeaker model not initialized");

    static constexpr size_t NUM_MEL = 80;
    const size_t num_frames = num_features / NUM_MEL;
    if (num_frames == 0) throw std::runtime_error("WeSpeaker: empty fbank input");

    if (num_frames != current_num_frames_) {
        graph_.hard_reset();
        load_weights_to_graph(&graph_);
        build_graph(num_frames);
        current_num_frames_ = num_frames;
    }

    input_buf_.resize(num_features);
    for (size_t i = 0; i < num_features; ++i)
        input_buf_[i] = static_cast<__fp16>(fbank_features[i]);

    graph_.set_input(audio_input_, input_buf_.data(), Precision::FP16);
    graph_.execute();

    const auto& out_buf = graph_.get_output_buffer(output_node_);
    const __fp16* out_data = out_buf.data_as<__fp16>();
    size_t total = out_buf.total_size;

    std::vector<float> result(total);
    for (size_t i = 0; i < total; ++i)
        result[i] = static_cast<float>(out_data[i]);
    return result;
}

}
}
