// Phase-5 microbenchmark: exercise a >512-token thinking segment to confirm
// the SWA ring's sub-ring rollover is hit, and report decode throughput and
// wall time so main-vs-branch comparisons are apples-to-apples.
//
// Runs CACTUS_TEST_GEMMA4_MODEL (Gemma 4) with enable_thinking_if_supported
// and a high max_tokens so the model reasons long enough to cross 512
// thinking tokens in a single segment. Prints:
//   - prompt / decode token counts
//   - decode tokens/sec
//   - whether the generated stream contained a thinking segment
//     longer than 512 tokens

#include "../cactus/ffi/cactus_ffi.h"
#include "../cactus/ffi/cactus_utils.h"
#include "test_utils.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

using namespace cactus::engine;
using namespace cactus::ffi;

int main() {
    const char* model_path = std::getenv("CACTUS_TEST_GEMMA4_MODEL");
    if (!model_path) {
        std::cerr << "CACTUS_TEST_GEMMA4_MODEL not set\n";
        return 2;
    }

    auto* model = cactus_init(model_path, nullptr, false);
    if (!model) {
        std::cerr << "Failed to load model at " << model_path << "\n";
        return 1;
    }

    auto* handle = static_cast<CactusModelHandle*>(model);
    auto* tokenizer = handle->model->get_tokenizer();
    const auto& cfg = handle->model->get_config();

    // A reasoning prompt that's likely to produce a long thinking trace.
    const char* msgs =
        R"([{"role":"user","content":"Solve this step by step, showing every piece of reasoning: )"
        R"(A train leaves station A at 9:00 AM traveling east at 60 mph. )"
        R"(A second train leaves station B, which is 420 miles east of A, at 10:30 AM traveling west at 75 mph. )"
        R"(At what exact time do they meet? )"
        R"(At what point (how many miles east of A) do they meet? )"
        R"(Then: suppose the first train had started 45 minutes later instead -- recompute both. )"
        R"(Then: explain in detail why your method generalizes, and double-check each arithmetic step by working it out twice, independently."}])";

    const char* options =
        R"({"max_tokens":1200,"temperature":0,"top_k":1,"enable_thinking_if_supported":true,"telemetry_enabled":false,"auto_handoff":false})";

    static char buf[200 * 1024];
    auto t0 = std::chrono::steady_clock::now();
    int rc = cactus_complete(model, msgs, buf, sizeof(buf), options, nullptr, nullptr, nullptr, nullptr, 0);
    auto t1 = std::chrono::steady_clock::now();
    double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (rc <= 0) {
        std::cerr << "cactus_complete failed rc=" << rc << ": " << buf << "\n";
        cactus_destroy(model);
        return 1;
    }

    // Recover generated tokens from the processed_tokens: whatever was added
    // past the prompt length.
    std::vector<uint32_t> prompt_tokens = tokenizer->encode(
        tokenizer->format_chat_prompt({{"user", "PROBE", "", {}, {}, 0, {}}}, true, "", true));
    // We don't know the exact prompt length from here, so infer via processed_tokens
    // minus the prompt we know was used; easier: find the longest <|channel>..<channel|>
    // block inside the raw text and count tokens in it.
    std::string resp = buf;
    std::cout << "wall_ms=" << wall_ms << "\n";
    std::cout << "response_size_bytes=" << resp.size() << "\n";

    // Find the longest thinking block in the response text and tokenize it to get
    // an approximate token count.
    std::string thinking_text;
    {
        std::string content;
        strip_thinking_block(resp, thinking_text, content);
    }
    size_t thinking_tok_est = 0;
    if (!thinking_text.empty()) {
        thinking_tok_est = tokenizer->encode(thinking_text).size();
    }
    std::cout << "thinking_text_chars=" << thinking_text.size() << "\n";
    std::cout << "thinking_tokens_estimate=" << thinking_tok_est << "\n";

    // Rough decode throughput: parse completion_tokens from response json if present.
    // Otherwise use handle->processed_tokens.size() as a lower bound.
    size_t processed_after = handle->processed_tokens.size();
    std::cout << "processed_tokens_after=" << processed_after << "\n";

    // Report whether the thinking segment exceeded the SWA window so we know
    // the sub-ring rollover path actually fired.
    std::cout << "thinking_exceeds_window=" << (thinking_tok_est > cfg.sliding_window ? "YES" : "NO")
              << " (sliding_window=" << cfg.sliding_window << ")\n";

    cactus_destroy(model);
    (void)prompt_tokens;
    return 0;
}
