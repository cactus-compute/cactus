#include "test_utils.h"
#include <vector>

using namespace TestUtils;

bool test_sample_fp32_bitmask_greedy() {
    std::vector<float> logits = {1.0f, 10.0f, 5.0f, 3.0f};
    uint32_t output = 0;
    const uint32_t bitmask[] = {0b1101u};

    cactus_sample_f32(
        logits.data(), &output, logits.size(),
        0.0f, 1.0f, 0, 0,
        bitmask);

    return output == 2;
}

bool test_sample_fp16_bitmask_greedy() {
    std::vector<__fp16> logits = {1.0f, 10.0f, 5.0f, 3.0f};
    uint32_t output = 0;
    const uint32_t bitmask[] = {0b1101u};

    cactus_sample_f16(
        logits.data(), &output, logits.size(),
        0.0f, 1.0f, 0, 0,
        bitmask);

    return output == 2;
}

int main() {
    TestRunner runner("Sampling Tests");
    runner.run_test("sample_fp32_bitmask", test_sample_fp32_bitmask_greedy());
    runner.run_test("sample_fp16_bitmask", test_sample_fp16_bitmask_greedy());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
