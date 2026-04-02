#include "test_utils.h"
#include <iostream>
#include <string>
#include <vector>
#include <cmath>

using namespace cactus::engine;
using TestUtils::load_bin;

static std::string get_assets_dir() {
    const char* dir = std::getenv("CACTUS_TEST_ASSETS");
    if (dir) return dir;
    return "../assets";
}

bool test_real_audio_transcription() {
    const char* model_path = std::getenv("CACTUS_TEST_GEMMA4_MODEL");
    std::string assets = get_assets_dir();
    if (!model_path) {
        std::cerr << "  SKIP: CACTUS_TEST_GEMMA4_MODEL not set\n";
        return true;
    }

    auto mel = load_bin(assets + "/test_wav_mel.bin");
    if (mel.empty()) {
        std::cerr << "  SKIP: run python tests/generate_audio_reference.py and ensure test_wav_mel.bin exists\n";
        return true;
    }

    size_t num_frames = mel.size() / 128;
    std::cout << "  Mel: " << num_frames << " frames x 128 bins\n";

    auto model = create_model(model_path);
    if (!model || !model->init(model_path, 2048, "", true)) {
        std::cerr << "  FAIL: model init\n";
        return false;
    }

    auto* tokenizer = model->get_tokenizer();
    if (!tokenizer) {
        std::cerr << "  FAIL: no tokenizer\n";
        return false;
    }

    uint32_t audio_token_id = model->get_config().audio_token_id;
    if (audio_token_id == 0) audio_token_id = 258881;

    size_t after_stage1 = (num_frames + 1) / 2;
    size_t num_soft_tokens = (after_stage1 + 1) / 2;
    std::cout << "  Encoder will produce " << num_soft_tokens << " soft tokens (computed from " << num_frames << " frames)\n";

    std::string prompt_text = "Transcribe the audio.";
    auto text_before = tokenizer->encode("<bos><|turn>user\n" + prompt_text + "<|audio>");
    auto text_after = tokenizer->encode("<audio|><turn|>\n<|turn>model\n");

    std::vector<uint32_t> tokens;
    tokens.insert(tokens.end(), text_before.begin(), text_before.end());
    for (size_t i = 0; i < num_soft_tokens; i++)
        tokens.push_back(audio_token_id);
    tokens.insert(tokens.end(), text_after.begin(), text_after.end());

    std::cout << "  Tokens: " << tokens.size() << " (" << text_before.size() << " + " << num_soft_tokens << " audio + " << text_after.size() << ")\n";

    std::cout << "  Generating...\n";

    std::string output;
    for (int i = 0; i < 100; i++) {
        uint32_t token = model->decode_with_audio(tokens, mel, 0.0f, 1.0f, 1, "");
        std::string piece = tokenizer->decode({token});
        output += piece;
        tokens.push_back(token);

        if (output.find("<turn|>") != std::string::npos || output.find("<eos>") != std::string::npos)
            break;
    }

    std::cout << "  Output: \"" << output << "\"\n";

    if (output.empty()) {
        std::cerr << "  FAIL: empty output\n";
        return false;
    }

    bool has_content = false;
    for (char c : output) {
        if (std::isalpha(c)) { has_content = true; break; }
    }
    if (!has_content) {
        std::cerr << "  FAIL: output has no alphabetic content\n";
        return false;
    }

    return true;
}

int main() {
    std::cout << "[1] real_audio_transcription:\n";
    bool ok = test_real_audio_transcription();
    std::cout << "  -> " << (ok ? "PASS" : "FAIL") << "\n";
    return ok ? 0 : 1;
}
