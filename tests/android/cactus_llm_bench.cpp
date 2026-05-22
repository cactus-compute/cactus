#include "cactus_engine.h"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<uint32_t> load_tokens(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open token input: " + path);
    }
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::vector<uint32_t> tokens;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) continue;
        tokens.push_back(static_cast<uint32_t>(std::stoul(item)));
    }
    if (tokens.empty()) {
        throw std::runtime_error("token input is empty: " + path);
    }
    return tokens;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <model_dir> <token_ids.csv> <decode_tokens>\n";
        return 2;
    }

    try {
        std::string model_dir = argv[1];
        auto tokens = load_tokens(argv[2]);
        size_t decode_tokens = static_cast<size_t>(std::stoul(argv[3]));

        cactus_log_set_level(3);
        cactus_model_t model = cactus_init(model_dir.c_str(), nullptr, false);
        if (!model) {
            const char* error = cactus_get_last_error();
            std::cerr << (error ? error : "cactus_init failed") << "\n";
            return 1;
        }

        std::vector<char> buffer(65536);
        int rc = cactus_benchmark_tokens(
            model,
            tokens.data(),
            tokens.size(),
            decode_tokens,
            buffer.data(),
            buffer.size()
        );
        cactus_destroy(model);
        if (rc < 0) {
            const char* error = cactus_get_last_error();
            std::cerr << (error ? error : buffer.data()) << "\n";
            return 1;
        }
        std::cout << buffer.data() << "\n";
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "Error: " << exc.what() << "\n";
        return 1;
    }
}
