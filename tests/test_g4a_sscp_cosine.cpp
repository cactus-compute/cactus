#include "test_utils.h"
#include "../cactus/npu/npu.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

using TestUtils::load_bin;
using TestUtils::cosine_sim;

int main() {
    const char* model_path = std::getenv("CACTUS_TEST_GEMMA4_MODEL");
    std::string assets = std::getenv("CACTUS_TEST_ASSETS") ? std::getenv("CACTUS_TEST_ASSETS") : "assets";
    if (!model_path) { std::cerr << "Set CACTUS_TEST_GEMMA4_MODEL\n"; return 1; }

    auto mel_raw = load_bin(assets + "/audio_test_mel_input.bin");
    if (mel_raw.empty()) { std::cerr << "No mel input\n"; return 1; }

    // Take first 48 frames, convert to FP16
    size_t chunk_frames = 48, mel_bins = 128;
    std::vector<__fp16> mel48(chunk_frames * mel_bins);
    for (size_t i = 0; i < chunk_frames * mel_bins; i++)
        mel48[i] = (__fp16)mel_raw[i];

    auto sscp_ref = load_bin(assets + "/cactus_ref_sscp.bin");  // [12, 1024] float32
    std::cout << "sscp_ref: " << sscp_ref.size() << " floats ("
              << sscp_ref.size()/1024 << " tokens x 1024)\n";

    // SSCP-only encoder
    auto enc = cactus::npu::create_encoder();
    if (!enc || !enc->load(model_path)) { std::cerr << "Encoder load failed\n"; return 1; }

    auto in_shape  = enc->get_input_shape();
    auto out_shape = enc->get_output_shape();
    std::cout << "Encoder in=" << in_shape[0] << "x" << in_shape[1]
              << " out=" << out_shape[0] << "x" << out_shape[1] << "\n";

    size_t out_n = (size_t)out_shape[0] * out_shape[1];
    std::vector<__fp16> out_buf(out_n);
    enc->encode(mel48.data(), out_buf.data(), in_shape);

    std::vector<float> cpp_out(out_n);
    for (size_t i = 0; i < out_n; i++) cpp_out[i] = (float)out_buf[i];

    int out_dim = out_shape[1];

    // Compare first token
    std::cout << "NPU tok0 first 8:";
    for (int i = 0; i < 8 && i < out_dim; i++) std::cout << " " << cpp_out[i];
    std::cout << "\nRef tok0 first 8:";
    for (int i = 0; i < 8 && i < (int)sscp_ref.size(); i++) std::cout << " " << sscp_ref[i];
    std::cout << "\n";

    // Cosine similarity on all tokens
    size_t n = std::min(cpp_out.size(), sscp_ref.size());
    float cos = cosine_sim(std::vector<float>(sscp_ref.begin(), sscp_ref.begin()+n),
                           std::vector<float>(cpp_out.begin(), cpp_out.begin()+n));
    std::cout << "Cosine similarity (SSCP NPU vs Python): " << cos << "\n";

    // Per-token cosine
    for (int t = 0; t < out_shape[0] && t < (int)(sscp_ref.size()/out_dim); t++) {
        float c = cosine_sim(std::vector<float>(sscp_ref.begin()+t*out_dim, sscp_ref.begin()+(t+1)*out_dim),
                             std::vector<float>(cpp_out.begin()+t*out_dim, cpp_out.begin()+(t+1)*out_dim));
        double n2 = 0;
        for (int d = 0; d < out_dim; d++) n2 += (double)cpp_out[t*out_dim+d]*cpp_out[t*out_dim+d];
        std::cout << "  tok" << t << " cos=" << c << " npu_norm=" << std::sqrt(n2) << "\n";
    }

    return (cos > 0.95f) ? 0 : 1;
}
