#include "test_utils.h"
#include <iostream>
#include <vector>
#include <cmath>

using namespace cactus::engine;
using TestUtils::load_bin;
using TestUtils::cosine_sim;

int main() {
    const char* model_path = std::getenv("CACTUS_TEST_GEMMA4_MODEL");
    std::string assets = std::getenv("CACTUS_TEST_ASSETS") ? std::getenv("CACTUS_TEST_ASSETS") : "../assets";

    auto mel = load_bin(assets + "/test_wav_mel.bin");
    auto ref = load_bin(assets + "/test_wav_hf_projected.bin");
    if (mel.empty() || ref.empty()) { std::cerr << "Missing files\n"; return 1; }

    auto model = create_model(model_path);
    if (!model || !model->init(model_path, 2048, "", false)) { std::cerr << "Init failed\n"; return 1; }

    auto cpp = model->get_audio_embeddings(mel);
    float cos = cosine_sim(ref, cpp);
    std::cout << "Real audio projected cosine (C++ vs HF FP32): " << cos << "\n";
    std::cout << "C++ size=" << cpp.size() << " ref size=" << ref.size() << "\n";

    double cpp_sum = 0, ref_sum = 0;
    for (auto v : cpp) cpp_sum += std::abs(v);
    for (auto v : ref) ref_sum += std::abs(v);
    std::cout << "C++ mean_abs=" << cpp_sum/cpp.size() << " ref mean_abs=" << ref_sum/ref.size() << "\n";

    return 0;
}
