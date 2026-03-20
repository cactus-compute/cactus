#include "e2e_driver.h"

#ifdef WITH_EXECUTORCH

// TODO: ExecuTorch backend
// Requires ExecuTorch SDK with TextLLMRunner.
// API: TextLLMRunner::create(model_path) -> generate(prompt, config)
// Gate: WITH_EXECUTORCH CMake option
//
// When implementing:
// 1. #include the ExecuTorch text runner headers
// 2. load() creates a TextLLMRunner from the .pte file
// 3. generate() uses the runner's generate method, timing prefill/decode
// 4. unload() destroys the runner
//
// The TextLLMRunner may not expose separate prefill/decode timing.
// In that case, measure total generation time and estimate from
// the runner's reported metrics if available.

#include <iostream>

namespace {

bool available() {
    // TODO: check if ExecuTorch library is loadable
    return false;
}

void* load(const char* /*model_path*/, int /*threads*/) {
    std::cerr << "[executorch] Not yet implemented\n";
    return nullptr;
}

e2e::E2EResult generate(void* /*handle*/, const char* /*prompt*/, int /*max_tokens*/) {
    return {};
}

void unload(void* /*handle*/) {}

static int reg = [] {
    e2e::register_e2e_backend({
        "executorch", "executorch",
        available, load, generate, unload
    });
    return 0;
}();

} // namespace

#endif // WITH_EXECUTORCH
