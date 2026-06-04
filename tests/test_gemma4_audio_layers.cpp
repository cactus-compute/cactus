#include "test_utils.h"
#include <iostream>
#include <string>
#include <vector>
#include <cmath>

using namespace cactus::engine;
using TestUtils::load_bin;
using TestUtils::cosine_sim;

static std::vector<float> extract(CactusGraph* gb, size_t node) {
    gb->execute();
    const auto& buf = gb->get_output_buffer(node);
    size_t n = buf.total_size;
    std::vector<float> out(n);
    const __fp16* src = buf.data_as<__fp16>();
    for (size_t i = 0; i < n; i++) out[i] = static_cast<float>(src[i]);
    return out;
}

int main() {
    const char* model_path = std::getenv("CACTUS_TEST_GEMMA4_MODEL");
    std::string assets = std::getenv("CACTUS_TEST_ASSETS") ? std::getenv("CACTUS_TEST_ASSETS") : "../assets";
    if (!model_path) { std::cerr << "Set CACTUS_TEST_GEMMA4_MODEL\n"; return 1; }

    auto mel = load_bin(assets + "/audio_test_mel_input.bin");
    if (mel.empty()) { std::cerr << "No mel input\n"; return 1; }

    auto model = create_model(model_path);
    if (!model || !model->init(model_path, 2048, "", false)) { std::cerr << "Init failed\n"; return 1; }

    auto* mm = dynamic_cast<Gemma4MmModel*>(model.get());
    auto* gb = static_cast<CactusGraph*>(model->graph_handle_);
    size_t num_frames = mel.size() / 128;

    gb->soft_reset();
    auto backend = ComputeBackend::CPU;
    auto& enc = mm->audio_encoder();
    size_t hidden = enc.build_sscp(gb, mel, num_frames, backend);
    auto ctx = enc.build_conformer_context(gb, hidden);

    for (uint32_t i = 0; i < 12; i++) {
        size_t ffw_s = enc.build_conformer_ffw(gb, hidden, i, false, backend);
        size_t attn = enc.build_conformer_attention(gb, ffw_s, i, ctx, backend);
        size_t lconv = enc.build_conformer_lconv1d(gb, attn, i, backend);
        size_t ffw_e = enc.build_conformer_ffw(gb, lconv, i, true, backend);
        hidden = gb->rms_norm(ffw_e, enc.audio_weights_.layers[i].block_norm,
                              enc.get_config().audio_rms_norm_eps);

        auto cpp = extract(gb, hidden);
        char buf[64];
        snprintf(buf, sizeof(buf), "%s/audio_ref_L%02d_block.bin", assets.c_str(), i);
        auto ref = load_bin(buf);
        float cos = ref.empty() ? -1 : cosine_sim(ref, cpp);
        std::cout << "L" << i << "_block: cos=" << cos << "\n";
    }
    return 0;
}
