#include "../cactus/ffi/cactus_ffi.h"
#include "../libs/audio/wav.h"
#include <iostream>
#include <string>
#include <cstring>
#include <vector>
#include <fstream>

static const char* MODEL_PATH = "/Users/noahcylich/Documents/Desert/cactus2/weights/gemma-4-e2b-it";
static const char* WAV_PATH = "/Users/noahcylich/Documents/Desert/cactus2/tests/assets/record.wav";

void token_cb(const char* token, uint32_t, void*) {
    std::cout << token << std::flush;
}

int main() {
    AudioFP32 wav = load_wav(WAV_PATH);
    if (wav.samples.empty()) {
        std::cerr << "FAIL: Could not load WAV file\n";
        return 1;
    }

    std::vector<int16_t> pcm16(wav.samples.size());
    for (size_t i = 0; i < wav.samples.size(); i++) {
        float s = wav.samples[i];
        if (s > 1.0f) s = 1.0f;
        if (s < -1.0f) s = -1.0f;
        pcm16[i] = static_cast<int16_t>(s * 32767.0f);
    }

    const uint8_t* pcm_buffer = reinterpret_cast<const uint8_t*>(pcm16.data());
    size_t pcm_buffer_size = pcm16.size() * sizeof(int16_t);

    cactus_model_t model = cactus_init(MODEL_PATH, nullptr, false);
    if (!model) {
        std::cerr << "FAIL: Could not load model\n";
        return 1;
    }

    const char* messages = R"([{"role":"user","content":""}])";
    const char* options = R"({"max_tokens":128,"temperature":0.7,"enable_thinking_if_supported":false,"telemetry_enabled":false})";

    char response[4096] = {0};
    std::cout << "Response: ";
    int result = cactus_complete(
        model, messages, response, sizeof(response),
        options, nullptr, token_cb, nullptr,
        pcm_buffer, pcm_buffer_size
    );
    std::cout << "\n";

    cactus_destroy(model);

    if (result < 0) {
        std::cerr << "FAIL: cactus_complete returned error\n";
        return 1;
    }

    std::string resp(response);
    if (resp.find("\"success\":true") == std::string::npos) {
        std::cerr << "FAIL: response not successful\n";
        return 1;
    }

    if (resp.find("\"response\":\"\"") != std::string::npos) {
        std::cerr << "FAIL: empty response\n";
        return 1;
    }

    if (resp.find("sorry") != std::string::npos || resp.find("Sorry") != std::string::npos) {
        std::cerr << "FAIL: model said sorry (likely didn't receive audio)\n";
        return 1;
    }

    std::cout << "PASS\n";
    return 0;
}
