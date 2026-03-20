#include "e2e_driver.h"

#ifdef WITH_ONNXRT_GENAI

#include "ort_genai_c.h"
#include <chrono>
#include <iostream>
#include <cstring>

namespace {

#define OGA_CHECK(expr) do { \
    OgaResult* _r = (expr); \
    if (_r) { \
        std::cerr << "[onnxrt] " << OgaResultGetError(_r) << "\n"; \
        OgaDestroyResult(_r); \
        return {}; \
    } \
} while(0)

#define OGA_CHECK_NULL(expr) do { \
    OgaResult* _r = (expr); \
    if (_r) { \
        std::cerr << "[onnxrt] " << OgaResultGetError(_r) << "\n"; \
        OgaDestroyResult(_r); \
        return nullptr; \
    } \
} while(0)

struct OnnxRTHandle {
    OgaModel* model;
    OgaTokenizer* tokenizer;
};

bool available() {
    return true;
}

void* load(const char* model_path, int /*threads*/) {
    OgaModel* model = nullptr;
    OgaResult* result = OgaCreateModel(model_path, &model);
    if (result) {
        std::cerr << "[onnxrt] Failed to load: " << OgaResultGetError(result) << "\n";
        OgaDestroyResult(result);
        return nullptr;
    }

    OgaTokenizer* tokenizer = nullptr;
    result = OgaCreateTokenizer(model, &tokenizer);
    if (result) {
        std::cerr << "[onnxrt] Failed to create tokenizer: " << OgaResultGetError(result) << "\n";
        OgaDestroyResult(result);
        OgaDestroyModel(model);
        return nullptr;
    }

    return new OnnxRTHandle{model, tokenizer};
}

e2e::E2EResult generate(void* handle, const char* prompt, int max_tokens) {
    auto* h = static_cast<OnnxRTHandle*>(handle);
    e2e::E2EResult res = {};

    // Tokenize
    OgaSequences* input_sequences = nullptr;
    OGA_CHECK(OgaCreateSequences(&input_sequences));

    OgaResult* enc_result = OgaTokenizerEncode(h->tokenizer, prompt, input_sequences);
    if (enc_result) {
        std::cerr << "[onnxrt] encode failed: " << OgaResultGetError(enc_result) << "\n";
        OgaDestroyResult(enc_result);
        OgaDestroySequences(input_sequences);
        return res;
    }

    size_t input_len = OgaSequencesGetSequenceCount(input_sequences, 0);
    res.prefill_tokens = static_cast<int>(input_len);

    // Create generator params
    OgaGeneratorParams* params = nullptr;
    OGA_CHECK(OgaCreateGeneratorParams(h->model, &params));
    OgaGeneratorParamsSetSearchNumber(params, "max_length", static_cast<double>(input_len + max_tokens));

    // Create generator
    OgaGenerator* generator = nullptr;
    OGA_CHECK(OgaCreateGenerator(h->model, params, &generator));

    // Append input tokens
    const int32_t* input_data = OgaSequencesGetSequenceData(input_sequences, 0);
    OgaGenerator_AppendTokens(generator, input_data, input_len);

    auto wall_start = std::chrono::steady_clock::now();

    // First generate_next_token = prefill + first token
    auto prefill_start = std::chrono::steady_clock::now();
    OgaGenerator_GenerateNextToken(generator);
    auto prefill_end = std::chrono::steady_clock::now();

    double prefill_ms = std::chrono::duration<double, std::milli>(prefill_end - prefill_start).count();
    res.ttft_ms = prefill_ms;

    // Decode loop
    auto decode_start = std::chrono::steady_clock::now();
    int generated = 1;
    for (int i = 1; i < max_tokens && !OgaGenerator_IsDone(generator); i++) {
        OgaGenerator_GenerateNextToken(generator);
        generated++;
    }
    auto decode_end = std::chrono::steady_clock::now();

    double decode_ms = std::chrono::duration<double, std::milli>(decode_end - decode_start).count();
    auto wall_end = std::chrono::steady_clock::now();

    res.decode_tokens = generated;
    res.prefill_tps = (prefill_ms > 0.0) ? (res.prefill_tokens * 1000.0 / prefill_ms) : 0.0;
    res.decode_tps = (decode_ms > 0.0) ? (generated * 1000.0 / decode_ms) : 0.0;
    res.total_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

    OgaDestroyGenerator(generator);
    OgaDestroyGeneratorParams(params);
    OgaDestroySequences(input_sequences);

    return res;
}

void unload(void* handle) {
    auto* h = static_cast<OnnxRTHandle*>(handle);
    OgaDestroyTokenizer(h->tokenizer);
    OgaDestroyModel(h->model);
    delete h;
}

static int reg = [] {
    e2e::register_e2e_backend({
        "onnxrt", "onnxrt",
        available, load, generate, unload
    });
    return 0;
}();

} // namespace

#endif // WITH_ONNXRT_GENAI
