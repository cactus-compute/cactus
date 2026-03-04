#include "../cactus/cactus.h"
#include <iostream>
#include <cstdlib>

using namespace cactus::engine;

int main(int argc, char* argv[]) {
    std::string model_path = "weights/gemma-3n-e2b-it-int8";
    if (argc > 1) model_path = argv[1];

    auto model = create_model(model_path);
    if (!model || !model->init(model_path, 2048, "", false)) {
        std::cerr << "Failed to init model" << std::endl;
        return 1;
    }

    auto* tokenizer = model->get_tokenizer();
    auto tokens = tokenizer->encode("Hello");

    model->reset_cache();
    uint32_t next = model->decode(tokens, 1.0f, 0.0f, 1);
    std::cout << "Hello -> " << next << " = \"" << tokenizer->decode({next}) << "\"" << std::endl;

    return 0;
}
