#include "test_utils.h"
#include "../cactus/npu/npu.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <numeric>

using TestUtils::load_bin;
using TestUtils::cosine_sim;

static std::vector<float> fp16_to_float(const std::vector<float>& v) { return v; }

int main() {
    const char* model_path = std::getenv("CACTUS_TEST_GEMMA4_MODEL");
    std::string assets = std::getenv("CACTUS_TEST_ASSETS") ? std::getenv("CACTUS_TEST_ASSETS") : "assets";
    if (!model_path) { std::cerr << "Set CACTUS_TEST_GEMMA4_MODEL\n"; return 1; }

    // Load 48-frame mel input
    auto mel_raw = load_bin(assets + "/audio_test_mel_input.bin");
    if (mel_raw.empty()) { std::cerr << "No mel input\n"; return 1; }

    // mel_raw is float32, 100 frames x 128 bins. Take first 48 frames.
    size_t chunk_frames = 48, mel_bins = 128;
    std::vector<__fp16> mel48(chunk_frames * mel_bins);
    for (size_t i = 0; i < chunk_frames * mel_bins; i++)
        mel48[i] = (__fp16)mel_raw[i];

    // Load Python/cactus-weight reference (12 tokens x 1536 dims, float32)
    auto ref48 = load_bin(assets + "/cactus_ref_encoder.bin");  // [12, 1536] float32
    if (ref48.empty()) { std::cerr << "No reference enc48\n"; return 1; }

    std::cout << "mel: " << chunk_frames << "x" << mel_bins
              << " ref: " << ref48.size() << " floats\n";

    // Create NPU encoder
    auto enc = cactus::npu::create_encoder();
    if (!enc || !enc->load(model_path)) { std::cerr << "Encoder load failed\n"; return 1; }

    auto in_shape  = enc->get_input_shape();
    auto out_shape = enc->get_output_shape();
    std::cout << "Encoder in=" << in_shape[0] << "x" << in_shape[1]
              << " out=" << out_shape[0] << "x" << out_shape[1] << "\n";

    // Verify shapes match
    if (in_shape[0] != (int)chunk_frames || in_shape[1] != (int)mel_bins) {
        std::cerr << "Shape mismatch: expected [" << chunk_frames << "," << mel_bins
                  << "] got [" << in_shape[0] << "," << in_shape[1] << "]\n";
        return 1;
    }

    std::vector<__fp16> out_buf(out_shape[0] * out_shape[1]);
    enc->encode(mel48.data(), out_buf.data(), in_shape);

    // Convert output to float32
    std::vector<float> cpp_out(out_buf.size());
    for (size_t i = 0; i < out_buf.size(); i++)
        cpp_out[i] = (float)out_buf[i];

    std::cout << "NPU output: " << cpp_out.size() << " floats\n";

    // Compare
    size_t n = std::min(cpp_out.size(), ref48.size());
    float cos = cosine_sim(std::vector<float>(ref48.begin(), ref48.begin()+n),
                           std::vector<float>(cpp_out.begin(), cpp_out.begin()+n));
    std::cout << "Cosine similarity (NPU vs HF): " << cos << "\n";

    // Print first token
    int out_dim = out_shape[1];
    std::cout << "NPU tok0 first 8:";
    for (int i = 0; i < 8 && i < out_dim; i++)
        std::cout << " " << cpp_out[i];
    std::cout << "\n";
    std::cout << "HF  tok0 first 8:";
    for (int i = 0; i < 8 && i < (int)ref48.size(); i++)
        std::cout << " " << ref48[i];
    std::cout << "\n";

    double npu_norm = 0, hf_norm = 0;
    for (int i = 0; i < out_dim && i < (int)cpp_out.size(); i++) npu_norm += (double)cpp_out[i]*cpp_out[i];
    for (int i = 0; i < out_dim && i < (int)ref48.size(); i++) hf_norm += (double)ref48[i]*ref48[i];
    std::cout << "NPU tok0 norm=" << std::sqrt(npu_norm) << " HF tok0 norm=" << std::sqrt(hf_norm) << "\n";

    if (cos > 0.9f) {
        std::cout << "PASS: cosine similarity " << cos << " > 0.9\n";
        return 0;
    } else {
        std::cerr << "FAIL: cosine similarity " << cos << " <= 0.9\n";
        return 1;
    }
}
