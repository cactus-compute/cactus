#include "test_utils.h"
#include <fstream>
#include <cstdlib>
#include <iostream>

using namespace EngineTestUtils;

static const char* g_model_path = std::getenv("CACTUS_TEST_MODEL");
static const char* g_assets_path = std::getenv("CACTUS_TEST_ASSETS");

static const char* g_options = R"({
        "max_tokens": 256,
    "stop_sequences": ["<|im_end|>", "<end_of_turn>"],
    "telemetry_enabled": false
    })";

bool test_vlm_multiturn() {
    std::string model_path_str(g_model_path ? g_model_path : "");

    std::string vision_file = model_path_str + "/vision_patch_embedding.weights";
    std::ifstream vf(vision_file);
    if (!vf.good()) {
        std::cout << "Skipping VLM multi-turn test: vision weights not found." << std::endl;
        return true;
    }
    vf.close();

    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║       VLM MULTI-TURN TEST                ║\n"
              << "╚══════════════════════════════════════════╝\n";

    cactus_model_t model = cactus_init(g_model_path, nullptr, false);
    if (!model) {
        std::cerr << "Failed to initialize model for VLM multi-turn test" << std::endl;
        return false;
    }

    std::string img_path = std::string(g_assets_path) + "/test_monkey.png";

    std::string messages1 = "[{\"role\": \"user\", "
        "\"content\": \"Describe what is happening in this image in two sentences.\", "
        "\"images\": [\"" + img_path + "\"]}]";

    StreamingData stream_data1;
    stream_data1.model = model;
    char response1[4096];

    std::cout << "\n[Turn 1]\n";
    std::cout << "User: Describe what is happening in this image in two sentences.\n";
    std::cout << "Assistant: ";

    int result1 = cactus_complete(model, messages1.c_str(), response1, sizeof(response1),
                                  g_options, nullptr, stream_callback, &stream_data1);

    std::cout << "\n\n[Results - Turn 1]\n";
    Metrics metrics1;
    metrics1.parse(response1);
    metrics1.print_json();

    bool success1 = result1 > 0 && stream_data1.token_count > 0;

    if (!success1) {
        std::cout << "└─ Status: FAILED ✗\n";
        cactus_destroy(model);
        return false;
    }

    std::string assistant_response;
    for (const auto& token : stream_data1.tokens) {
        assistant_response += token;
    }

    std::string messages2 = "[{\"role\": \"user\", "
        "\"content\": \"Describe what is happening in this image in two sentences.\", "
        "\"images\": [\"" + img_path + "\"]}, "
        "{\"role\": \"assistant\", \"content\": \"" + escape_json(assistant_response) + "\"}, "
        "{\"role\": \"user\", \"content\": \"Describe the image once again.\"}]";

    StreamingData stream_data2;
    stream_data2.model = model;
    char response2[4096];

    std::cout << "\n[Turn 2]\n";
    std::cout << "User: Describe the image once again.\n";
    std::cout << "Assistant: ";

    int result2 = cactus_complete(model, messages2.c_str(), response2, sizeof(response2),
                                  g_options, nullptr, stream_callback, &stream_data2);

    std::cout << "\n\n[Results - Turn 2]\n";
    Metrics metrics2;
    metrics2.parse(response2);
    metrics2.print_json();

    bool success2 = result2 > 0 && stream_data2.token_count > 0;

    if (!success2) {
        std::cout << "└─ Status: FAILED ✗ (Follow-up message failed)\n";
    }

    cactus_destroy(model);
    return success1 && success2;
}

int main() {
    TestUtils::TestRunner runner("VLM Tests");
    runner.run_test("vlm_multiturn", test_vlm_multiturn());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
