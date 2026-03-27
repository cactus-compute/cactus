#include "model.h"

namespace cactus { namespace engine {

WeSpeakerModel::WeSpeakerModel() : Model() {}
WeSpeakerModel::WeSpeakerModel(const Config& config) : Model(config) {}

bool WeSpeakerModel::init(const std::string& model_folder, size_t,
                           const std::string&, bool) {
    model_folder_path_ = model_folder;
    try {
        auto* gb = new CactusGraph();
        graph_handle_ = gb;
        load_weights_to_graph(gb);
        initialized_ = true;
        return true;
    } catch (const std::exception& e) {
        CACTUS_LOG_ERROR("wespeaker", "Init failed: " << e.what());
        return false;
    }
}

static WeSpeakerModel::ResBlockWeights load_resblock(CactusGraph* gb, const std::string& prefix, bool has_shortcut) {
    WeSpeakerModel::ResBlockWeights rb;
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

static size_t build_resblock(CactusGraph* gb, size_t x, const WeSpeakerModel::ResBlockWeights& rb, bool stride2) {
    size_t identity = x;

    size_t out;
    if (stride2) {
        out = gb->conv2d_k3s2p1(x, rb.conv1_w);
    } else {
        out = gb->conv2d_k3s1p1(x, rb.conv1_w);
    }
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

    w_.conv1_w = gb->mmap_weights(p + "/resnet_conv1_weight.weights");
    w_.bn1_w = gb->mmap_weights(p + "/resnet_bn1_weight.weights");
    w_.bn1_b = gb->mmap_weights(p + "/resnet_bn1_bias.weights");
    w_.bn1_mean = gb->mmap_weights(p + "/resnet_bn1_running_mean.weights");
    w_.bn1_var = gb->mmap_weights(p + "/resnet_bn1_running_var.weights");

    auto load_layer = [&](const std::string& layer_name, int num_blocks, bool first_has_shortcut) {
        std::vector<ResBlockWeights> blocks;
        for (int i = 0; i < num_blocks; ++i) {
            std::string prefix = p + "/resnet_" + layer_name + "_" + std::to_string(i);
            blocks.push_back(load_resblock(gb, prefix, i == 0 && first_has_shortcut));
        }
        return blocks;
    };

    w_.layer1 = load_layer("layer1", 3, false);
    w_.layer2 = load_layer("layer2", 4, true);
    w_.layer3 = load_layer("layer3", 6, true);
    w_.layer4 = load_layer("layer4", 3, true);

    w_.seg1_w = gb->mmap_weights(p + "/resnet_seg_1_weight.weights");
    w_.seg1_b = gb->mmap_weights(p + "/resnet_seg_1_bias.weights");

    audio_input_ = gb->input({1, 1, 80, 298}, Precision::FP16);

    size_t x = gb->conv2d_k3s1p1(audio_input_, w_.conv1_w);
    x = gb->batchnorm(x, w_.bn1_w, w_.bn1_b, w_.bn1_mean, w_.bn1_var);
    x = gb->relu(x);

    for (auto& rb : w_.layer1) x = build_resblock(gb, x, rb, false);
    for (size_t i = 0; i < w_.layer2.size(); ++i) x = build_resblock(gb, x, w_.layer2[i], i == 0);
    for (size_t i = 0; i < w_.layer3.size(); ++i) x = build_resblock(gb, x, w_.layer3[i], i == 0);
    for (size_t i = 0; i < w_.layer4.size(); ++i) x = build_resblock(gb, x, w_.layer4[i], i == 0);

    x = gb->stats_pool(x);

    x = gb->add(gb->matmul(x, w_.seg1_w, true), w_.seg1_b);

    output_node_ = x;
}

std::vector<float> WeSpeakerModel::embed(const float* pcm_f32, size_t num_samples) {
    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    if (!gb) throw std::runtime_error("WeSpeaker model not initialized");

    std::vector<__fp16> input(1 * 1 * 80 * 298, static_cast<__fp16>(0.0f));

    size_t copy_len = std::min(num_samples, static_cast<size_t>(1 * 1 * 80 * 298));
    for (size_t i = 0; i < copy_len; ++i)
        input[i] = static_cast<__fp16>(pcm_f32[i]);

    gb->set_input(audio_input_, input.data(), Precision::FP16);
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
