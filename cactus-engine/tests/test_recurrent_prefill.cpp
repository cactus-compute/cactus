#include "test_utils.h"
#include <cstdlib>
#include <iostream>
#include <vector>

static const char* g_model_path = std::getenv("CACTUS_TEST_MODEL");

static std::string benchmark_tokens_json(cactus_model_t model, const std::vector<uint32_t>& ids, size_t max_new) {
    std::vector<char> response(1 << 16, 0);
    int rc = cactus_benchmark_tokens(model, ids.data(), ids.size(), max_new, response.data(), response.size());
    return rc < 0 ? std::string() : std::string(response.data());
}

bool test_repeated_chunked_prefill() {
    if (!g_model_path) { std::cout << "  [WARN] CACTUS_TEST_MODEL not set; skipping\n"; return true; }

    cactus_model_t model = cactus_init(g_model_path, nullptr, false);
    if (!model) { std::cerr << "  [✗] model init failed\n"; return false; }

    std::vector<uint32_t> ids(600);
    for (size_t i = 0; i < ids.size(); i++) ids[i] = static_cast<uint32_t>(100 + (i % 200));

    std::string first = benchmark_tokens_json(model, ids, 4);
    std::string second = benchmark_tokens_json(model, ids, 4);
    cactus_destroy(model);

    if (first.empty() || first.find("\"success\":true") == std::string::npos) {
        std::cerr << "  [✗] first chunked prefill failed\n";
        return false;
    }
    if (second.empty() || second.find("\"success\":true") == std::string::npos) {
        std::cerr << "  [✗] second chunked prefill failed\n";
        return false;
    }
    return true;
}

int main() {
    TestUtils::TestRunner runner("Recurrent Prefill Tests");
    runner.run_test("repeated_chunked_prefill", test_repeated_chunked_prefill());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
