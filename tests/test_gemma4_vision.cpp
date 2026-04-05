#include "test_utils.h"
#include <iostream>
#include <string>
#include <vector>
#include <cstring>

using namespace cactus::engine;

static const char* get_gemma4_model_path() {
    const char* path = std::getenv("CACTUS_TEST_GEMMA4_MODEL");
    if (path) return path;
    return std::getenv("CACTUS_TEST_MODEL");
}

static const char* get_test_image_path() {
    return std::getenv("CACTUS_TEST_IMAGE");
}

bool test_vlm_model_creation() {
    const char* model_path = get_gemma4_model_path();
    if (!model_path) {
        std::cerr << "  SKIP: CACTUS_TEST_GEMMA4_MODEL not set\n";
        return true;
    }

    auto model = create_model(model_path);
    if (!model) {
        std::cerr << "  FAIL: create_model returned null\n";
        return false;
    }

    return true;
}

bool test_vlm_init() {
    const char* model_path = get_gemma4_model_path();
    if (!model_path) {
        std::cerr << "  SKIP: CACTUS_TEST_GEMMA4_MODEL not set\n";
        return true;
    }

    auto model = create_model(model_path);
    if (!model) {
        std::cerr << "  FAIL: create_model returned null\n";
        return false;
    }

    bool ok = model->init(model_path, 2048, "", false);
    if (!ok) {
        std::cerr << "  FAIL: model init failed\n";
        return false;
    }

    return true;
}

bool test_vlm_decode_with_image() {
    const char* model_path = get_gemma4_model_path();
    const char* image_path = get_test_image_path();
    if (!model_path || !image_path) {
        std::cerr << "  SKIP: CACTUS_TEST_GEMMA4_MODEL or CACTUS_TEST_IMAGE not set\n";
        return true;
    }

    auto model = create_model(model_path);
    if (!model) {
        std::cerr << "  FAIL: create_model returned null\n";
        return false;
    }

    if (!model->init(model_path, 2048, "", false)) {
        std::cerr << "  FAIL: model init failed\n";
        return false;
    }

    auto* tokenizer = model->get_tokenizer();
    if (!tokenizer) {
        std::cerr << "  FAIL: no tokenizer\n";
        return false;
    }

    std::vector<ChatMessage> messages;
    messages.push_back({"user", "Describe this image briefly.", "", {image_path}, {}});
    std::string prompt = tokenizer->format_chat_prompt(messages, true, "", false);
    auto tokens = tokenizer->encode(prompt);

    size_t vision_count = 0;
    for (auto t : tokens)
        if (t == 262145) vision_count++;
    std::cout << "  tokens: " << tokens.size() << ", vision soft tokens (262145): " << vision_count << "\n";

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

    std::cout << "  VLM output: " << output.substr(0, 500) << "\n";

    if (output.empty()) {
        std::cerr << "  FAIL: empty output\n";
        return false;
    }

    return true;
}

int main() {
    int pass = 0, fail = 0, total = 0;

    auto run = [&](const char* name, bool (*fn)()) {
        total++;
        std::cout << "[" << total << "] " << name << ": ";
        if (fn()) {
            std::cout << "PASS\n";
            pass++;
        } else {
            std::cout << "FAIL\n";
            fail++;
        }
    };

    auto test_image_embeddings = []() -> bool {
        const char* mp = std::getenv("CACTUS_TEST_GEMMA4_MODEL");
        const char* ip = std::getenv("CACTUS_TEST_IMAGE");
        if (!mp || !ip) return true;
        auto m = create_model(mp);
        if (!m || !m->init(mp, 2048, "", false)) return false;
        auto embeddings = m->get_image_embeddings(ip);
        return !embeddings.empty();
    };

    run("image_embeddings", +test_image_embeddings);
    run("vlm_model_creation", test_vlm_model_creation);
    run("vlm_init", test_vlm_init);
    run("vlm_decode_with_image", test_vlm_decode_with_image);

    std::cout << "\n" << pass << "/" << total << " passed";
    if (fail > 0) std::cout << " (" << fail << " FAILED)";
    std::cout << "\n";
    return fail > 0 ? 1 : 0;
}
