#include "test_utils.h"
#include "../src/engine.h"

#include <limits>
#include <vector>

using namespace TestUtils;
using cactus::engine::select_bucket;

namespace {

const std::vector<size_t> kFibonacci = {100, 200, 300, 500, 800, 1300, 2003};

bool picks_exact_fit() {
    return select_bucket(kFibonacci, 300) == 2;
}

bool picks_smallest_that_fits() {
    return select_bucket(kFibonacci, 201) == 2
        && select_bucket(kFibonacci, 301) == 3
        && select_bucket(kFibonacci, 801) == 5;
}

bool picks_first_bucket_for_tiny_input() {
    return select_bucket(kFibonacci, 1) == 0
        && select_bucket(kFibonacci, 100) == 0;
}

bool reports_none_when_above_every_bucket() {
    return select_bucket(kFibonacci, 2004) == kFibonacci.size();
}

bool reports_none_for_empty_ladder() {
    return select_bucket({}, 100) == 0;
}

bool ignores_zero_capacities() {
    const std::vector<size_t> with_zero = {0, 500, 0, 200};
    return select_bucket(with_zero, 150) == 3
        && select_bucket(with_zero, 300) == 1;
}

bool is_independent_of_declaration_order() {
    const std::vector<size_t> shuffled = {2003, 300, 100, 1300, 200, 800, 500};
    return shuffled[select_bucket(shuffled, 250)] == 300
        && shuffled[select_bucket(shuffled, 900)] == 1300;
}

bool handles_zero_required_frames() {
    return select_bucket(kFibonacci, 0) == 0;
}

bool unknown_required_frames_falls_back() {
    return select_bucket(kFibonacci, std::numeric_limits<size_t>::max()) == kFibonacci.size();
}

bool handles_duplicate_capacities() {
    const std::vector<size_t> duplicated = {300, 300, 800};
    return duplicated[select_bucket(duplicated, 250)] == 300;
}

}

int main() {
    TestRunner runner("Audio Encoder Bucket Selection");
    runner.run_test("Exact fit", picks_exact_fit());
    runner.run_test("Smallest that fits", picks_smallest_that_fits());
    runner.run_test("Tiny input takes first bucket", picks_first_bucket_for_tiny_input());
    runner.run_test("None when above every bucket", reports_none_when_above_every_bucket());
    runner.run_test("None for empty ladder", reports_none_for_empty_ladder());
    runner.run_test("Zero capacities ignored", ignores_zero_capacities());
    runner.run_test("Order independent", is_independent_of_declaration_order());
    runner.run_test("Zero required frames", handles_zero_required_frames());
    runner.run_test("Unknown required frames falls back", unknown_required_frames_falls_back());
    runner.run_test("Duplicate capacities", handles_duplicate_capacities());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
