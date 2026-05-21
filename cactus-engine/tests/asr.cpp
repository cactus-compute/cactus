#include "../cactus_engine.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " <model_path> [audio_file.wav] [--language <code>]\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 2;
    }

    std::string model_path = argv[1];
    std::string audio_path;
    std::string language = "en";

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--language" && i + 1 < argc) {
            language = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (audio_path.empty() && arg.rfind("-", 0) != 0) {
            audio_path = arg;
        }
    }

    if (audio_path.empty()) {
        std::cerr << "Error: live microphone capture not supported by asr binary; "
                     "pass an audio file path.\n";
        print_usage(argv[0]);
        return 2;
    }

    std::cout << "Loading model from " << model_path << "...\n";
    cactus_model_t model = cactus_init(model_path.c_str(), nullptr, false);
    if (!model) {
        std::cerr << "Failed to initialize model from " << model_path << "\n";
        const char* err = cactus_get_last_error();
        if (err && *err) std::cerr << "  " << err << "\n";
        return 1;
    }
    std::cout << "Model loaded.\n";

    std::ostringstream opts;
    opts << "{\"language\":\"" << language << "\",\"telemetry_enabled\":false,\"auto_handoff\":false,\"max_tokens\":512}";

    constexpr size_t kBufSize = 1 << 16;
    std::string response(kBufSize, '\0');

    int rc = cactus_transcribe(
        model,
        audio_path.c_str(),
        nullptr,
        response.data(),
        response.size(),
        opts.str().c_str(),
        nullptr,
        nullptr,
        nullptr,
        0);

    if (rc <= 0) {
        std::cerr << "Transcription failed: " << response.c_str() << "\n";
        cactus_destroy(model);
        return 1;
    }

    response.resize(std::strlen(response.c_str()));
    std::cout << response << std::endl;

    cactus_destroy(model);
    return 0;
}
