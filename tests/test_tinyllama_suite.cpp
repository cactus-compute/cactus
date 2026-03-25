#include "test_utils.h"
#include "../cactus/models/tinyllama/model_tinyllama.h"
#include "../cactus/ffi/cactus_utils.h"
#include "../libs/audio/wav.h"
#include <iostream>
#include <string>
#include <vector>
#include <sys/stat.h>

using namespace cactus::engine;
using namespace cactus::audio;
using namespace EngineTestUtils;

static const char* get_model_path() {
    const char* path = std::getenv("CACTUS_TEST_TINYLLAMA_MODEL");
    if (path) return path;
    return std::getenv("CACTUS_TEST_MODEL");
}

static const char* get_image_path() {
    return std::getenv("CACTUS_TEST_IMAGE");
}

static std::string get_assets_dir() {
    const char* dir = std::getenv("CACTUS_TEST_ASSETS");
    if (dir) return dir;
    return "../assets";
}

static bool has_npu_package(const char* model_path, const std::string& name) {
    struct stat st;
    return stat((std::string(model_path) + "/" + name).c_str(), &st) == 0;
}

static const char* g_options = R"({
    "max_tokens": 256,
    "stop_sequences": ["<turn|>", "<eos>"],
    "telemetry_enabled": false
})";


bool test_text_generation() {
    const char* model_path = get_model_path();
    if (!model_path) {
        std::cerr << "  SKIP: model path not set\n";
        return true;
    }

    const char* messages = R"([
        {"role": "system", "content": "/no_think You are a helpful assistant. Be concise."},
        {"role": "user", "content": "What is the capital of France?"}
    ])";

    return EngineTestUtils::run_test("TEXT GENERATION", model_path, messages, g_options,
        [](int result, const StreamingData& data, const std::string& response, const Metrics& m) {
            std::string text;
            for (const auto& t : data.tokens) text += t;
            std::string lower_text;
            for (char c : text) lower_text += std::tolower(c);

            bool has_paris = lower_text.find("paris") != std::string::npos;
            std::cout << "├─ Output: " << text.substr(0, 200) << "\n"
                      << "├─ Contains 'paris': " << (has_paris ? "YES" : "NO") << "\n";
            m.print_json();
            return result > 0 && data.token_count > 0 && has_paris;
        }, nullptr, -1, "What is the capital of France?");
}


bool test_tool_call() {
    const char* model_path = get_model_path();
    if (!model_path) {
        std::cerr << "  SKIP: model path not set\n";
        return true;
    }

    const char* messages = R"([
        {"role": "system", "content": "/no_think You are a helpful assistant that can use tools."},
        {"role": "user", "content": "What's the weather in Tokyo?"}
    ])";

    const char* tools = R"([{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City, Country"}
                },
                "required": ["location"]
            }
        }
    }])";

    const char* options = R"({
        "max_tokens": 256,
        "stop_sequences": ["<turn|>", "<eos>"],
        "force_tools": true,
        "telemetry_enabled": false
    })";

    return EngineTestUtils::run_test("TOOL CALL", model_path, messages, options,
        [](int result, const StreamingData&, const std::string& response, const Metrics& m) {
            bool has_function = response.find("\"function_calls\":[") != std::string::npos;
            bool has_tool = has_function && response.find("get_weather") != std::string::npos;
            std::cout << "├─ Function call: " << (has_function ? "YES" : "NO") << "\n"
                      << "├─ Correct tool: " << (has_tool ? "YES" : "NO") << "\n";
            m.print_json();
            return result > 0 && has_function && has_tool;
        }, tools, -1, "What's the weather in Tokyo?");
}


bool test_1k_context() {
    const char* model_path = get_model_path();
    if (!model_path) {
        std::cerr << "  SKIP: model path not set\n";
        return true;
    }

    std::string msg = "[{\"role\": \"system\", \"content\": \"/no_think You are helpful. ";
    for (int i = 0; i < 50; i++)
        msg += "Context " + std::to_string(i) + ": The quick brown fox jumps over the lazy dog. ";
    msg += "\"}, {\"role\": \"user\", \"content\": \"";
    for (int i = 0; i < 50; i++)
        msg += "Data point " + std::to_string(i) + " = " + std::to_string(i * 2.71828) + ". ";
    msg += "Summarize the data briefly.\"}]";

    return EngineTestUtils::run_test("1K CONTEXT", model_path, msg.c_str(), g_options,
        [](int result, const StreamingData& data, const std::string&, const Metrics& m) {
            std::cout << "├─ Tokens generated: " << data.token_count << "\n";
            m.print_json();
            return result > 0 && data.token_count > 0;
        }, nullptr, 100, "Summarize the data briefly.");
}


bool test_tinyllama_vision(bool expect_npu) {
    const char* model_path = get_model_path();
    const char* image_path = get_image_path();
    if (!model_path || !image_path) {
        std::cerr << "  SKIP: CACTUS_TEST_TINYLLAMA_MODEL or CACTUS_TEST_IMAGE not set\n";
        return true;
    }

    if (expect_npu && !has_npu_package(model_path, "vision_encoder.mlpackage")) {
        std::cerr << "  SKIP: vision_encoder.mlpackage not found in model folder\n";
        return true;
    }

    auto model = create_model(model_path);
    if (!model) {
        std::cerr << "  FAIL: create_model returned null\n";
        return false;
    }

    auto* mm = dynamic_cast<TinyLlamaMmModel*>(model.get());
    if (mm && !expect_npu) {
        mm->vision_encoder().disable_npu_ = true;
    }

    if (!model->init(model_path, 2048, "", false)) {
        std::cerr << "  FAIL: model init\n";
        return false;
    }

    auto* tokenizer = model->get_tokenizer();
    if (!tokenizer) {
        std::cerr << "  FAIL: no tokenizer\n";
        return false;
    }

    std::vector<ChatMessage> messages;
    messages.push_back({"user", "Describe this image briefly.", "", {image_path}});
    std::string prompt = tokenizer->format_chat_prompt(messages, true, "", false);
    auto tokens = tokenizer->encode(prompt);

    size_t vision_count = 0;
    for (auto t : tokens)
        if (t == 262145) vision_count++;
    std::cout << "  tokens: " << tokens.size() << ", vision soft tokens: " << vision_count << "\n";

    std::vector<std::string> images = {image_path};
    std::string output;

    for (int i = 0; i < 150; i++) {
        uint32_t token = model->decode_with_images(tokens, images, 0.0f, 1.0f, 1, "");
        std::string piece = tokenizer->decode({token});
        output += piece;
        tokens.push_back(token);
        if (piece.find("<turn|>") != std::string::npos || piece.find("<eos>") != std::string::npos)
            break;
    }

    std::cout << "  Output: " << output.substr(0, 300) << "\n";
    std::cout << "  NPU" << (expect_npu ? " (expected)" : " (not required)") << ": "
              << (has_npu_package(model_path, "vision_encoder.mlpackage") ? "available" : "not available") << "\n";

    if (output.empty()) {
        std::cerr << "  FAIL: empty output\n";
        return false;
    }

    bool has_content = false;
    for (char c : output)
        if (std::isalpha(c)) { has_content = true; break; }

    if (!has_content) {
        std::cerr << "  FAIL: output has no alphabetic content\n";
        return false;
    }

    return true;
}


bool test_tinyllama_audio(bool expect_npu) {
    const char* model_path = get_model_path();
    std::string assets = get_assets_dir();
    if (!model_path) {
        std::cerr << "  SKIP: model path not set\n";
        return true;
    }

    if (expect_npu && !has_npu_package(model_path, "audio_encoder.mlpackage")) {
        std::cerr << "  SKIP: audio_encoder.mlpackage not found in model folder\n";
        return true;
    }

    std::string audio_path = assets + "/test.wav";
    struct stat st;
    if (stat(audio_path.c_str(), &st) != 0) {
        std::cerr << "  SKIP: test.wav not found in assets\n";
        return true;
    }

    auto model = create_model(model_path);
    if (!model) {
        std::cerr << "  FAIL: create_model returned null\n";
        return false;
    }

    auto* mm = dynamic_cast<TinyLlamaMmModel*>(model.get());
    if (mm && !expect_npu)
        mm->audio_encoder().disable_npu_ = true;

    if (!model->init(model_path, 2048, "", true)) {
        std::cerr << "  FAIL: model init\n";
        return false;
    }

    AudioFP32 wav = load_wav(audio_path.c_str());
    auto audio_samples = resample_to_16k_fp32(wav.samples, wav.sample_rate);

    size_t pad_amt = 320 - (audio_samples.size() % 320);
    if (pad_amt < 320)
        audio_samples.resize(audio_samples.size() + pad_amt, 0.0f);

    const auto& cfg = model->get_config();
    size_t mel_bins = cfg.audio_input_feat_size;
    uint32_t audio_token_id = cfg.audio_token_id;
    if (audio_token_id == 0) audio_token_id = 258881;

    auto spec_cfg = get_htk_spectrogram_config();
    AudioProcessor ap;
    size_t fft_for_mel = spec_cfg.fft_override > 0 ? spec_cfg.fft_override : spec_cfg.n_fft;
    ap.init_mel_filters(fft_for_mel / 2 + 1, mel_bins, 0.0f, 8000.0f, 16000, nullptr, "htk");
    auto mel = ap.compute_spectrogram(audio_samples, spec_cfg);

    size_t num_frames = mel.size() / mel_bins;

#ifdef __APPLE__
    {
        static constexpr float LN2 = 0.693147180559945f;
        for (auto& v : mel) v -= LN2;
    }
#endif

    auto audio_features = transpose_mel_to_frame_major(mel, mel_bins, num_frames);

    size_t after_stage1 = (num_frames + 1) / 2;
    size_t num_soft_tokens = (after_stage1 + 1) / 2;

    auto* tokenizer = model->get_tokenizer();
    auto prefix = tokenizer->encode("<bos><|turn>user\nTranscribe the audio.<|audio>");
    auto suffix = tokenizer->encode("<audio|><turn|>\n<|turn>model\n");

    std::vector<uint32_t> tokens;
    tokens.insert(tokens.end(), prefix.begin(), prefix.end());
    for (size_t i = 0; i < num_soft_tokens; i++)
        tokens.push_back(audio_token_id);
    tokens.insert(tokens.end(), suffix.begin(), suffix.end());

    std::string output;
    for (int i = 0; i < 200; i++) {
        uint32_t token = model->decode_with_audio(tokens, audio_features, 0.0f, 1.0f, 1, "");
        std::string piece = tokenizer->decode({token});
        output += piece;
        tokens.push_back(token);
        if (output.find("<turn|>") != std::string::npos || output.find("<eos>") != std::string::npos)
            break;
    }

    std::cout << "  Transcript: " << output << "\n";

    if (output.empty()) {
        std::cerr << "  FAIL: empty output\n";
        return false;
    }

    bool has_content = false;
    for (char c : output)
        if (std::isalpha(c)) { has_content = true; break; }

    if (!has_content) {
        std::cerr << "  FAIL: output has no alphabetic content\n";
        return false;
    }

    return true;
}


int main() {
    TestUtils::TestRunner runner("TinyLlama Suite");

    runner.run_test("text_generation", test_text_generation());
    runner.run_test("tool_call", test_tool_call());
    runner.run_test("1k_context", test_1k_context());
    runner.run_test("vision", test_tinyllama_vision(false));
    runner.run_test("vision_npu", test_tinyllama_vision(true));
    runner.run_test("audio", test_tinyllama_audio(false));
    runner.run_test("audio_npu", test_tinyllama_audio(true));

    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
