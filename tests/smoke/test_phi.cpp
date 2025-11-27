#include "../cactus/cactus.h"
#include <iostream>
#include <cstring>

int main() {
    const char* model_path = "/mnt/disks/data/cactus/output/phi3_test";

    std::cout << "=== Phi-3 Mini 4K Instruct Test ===\n" << std::endl;
    std::cout << "Initializing model from: " << model_path << std::endl;

    cactus_model_t model = cactus_init(model_path, 2048, nullptr);
    if (!model) {
        std::cerr << "Failed to initialize Phi-3 model!" << std::endl;
        return 1;
    }

    std::cout << "✓ Model initialized successfully!\n" << std::endl;

    const char* messages = R"([
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ])";

    const char* options = R"({
        "max_tokens": 50,
        "stop_sequences": ["<|end|>", "<|endoftext|>"]
    })";

    char response[4096];
    std::cout << "Generating response..." << std::endl;

    int result = cactus_complete(model, messages, response, sizeof(response),
                                 options, nullptr, nullptr, nullptr);

    if (result < 0) {
        std::cerr << "Generation failed with error code: " << result << std::endl;
        std::cerr << "Response: " << response << std::endl;
        cactus_reset(model);
        return 1;
    }

    std::cout << "✓ Generation succeeded! (response length: " << result << " bytes)\n" << std::endl;
    std::cout << "=== Response JSON ===\n" << response << "\n" << std::endl;

    cactus_reset(model);
    std::cout << "\n✓ Test completed successfully!" << std::endl;

    return 0;
}