#include "cactus-engine/cactus_engine.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: native_transcribe_json <model_path> <audio_path> [options_json]\n";
        return 2;
    }

    const char* model_path = argv[1];
    const char* audio_path = argv[2];
    const char* options_json = argc >= 4 ? argv[3] : "{\"max_tokens\":500,\"telemetry_enabled\":true}";

    cactus_model_t model = cactus_init(model_path, nullptr, false);
    if (!model) {
        const char* error = cactus_get_last_error();
        std::cerr << (error && error[0] ? error : "cactus_init failed") << "\n";
        return 1;
    }

    std::vector<char> response(1024 * 1024, 0);
    int rc = cactus_transcribe(
        model,
        audio_path,
        "",
        response.data(),
        response.size(),
        options_json,
        nullptr,
        nullptr,
        nullptr,
        0
    );
    cactus_destroy(model);

    if (rc < 0) {
        std::cerr << response.data() << "\n";
        return 1;
    }

    std::cout << response.data() << "\n";
    return 0;
}
