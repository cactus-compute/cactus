#include "e2e_driver.h"

#ifdef WITH_LITERT

// LiteRT-LM backend
// Uses MediaPipe genai C API: LlmInferenceEngine
// Gate: WITH_LITERT CMake option
//
// TODO: When implementing:
// 1. #include mediapipe genai headers
// 2. load() creates an LlmInferenceEngine session from .task file
// 3. generate() calls PredictSync, timing prefill/decode
// 4. unload() destroys the session
//
// LiteRT-LM may not expose separate prefill/decode timing natively.
// Use wall-clock timing around the predict call and estimate TTFT
// from first token callback if available.

#include <iostream>

namespace {

bool available() {
    // TODO: check if LiteRT library is loadable
    return false;
}

void* load(const char* /*model_path*/, int /*threads*/) {
    std::cerr << "[litert] Not yet implemented\n";
    return nullptr;
}

e2e::E2EResult generate(void* /*handle*/, const char* /*prompt*/, int /*max_tokens*/) {
    return {};
}

void unload(void* /*handle*/) {}

static int reg = [] {
    e2e::register_e2e_backend({
        "litert", "litert",
        available, load, generate, unload
    });
    return 0;
}();

} // namespace

#endif // WITH_LITERT
