// End-to-end test for openai/privacy-filter (token-classification encoder).
// Drives the C++ engine directly since `classify()` is not yet FFI-exposed.
//
// Run:
//   CACTUS_OPF_MODEL=/path/to/converted/weights ./test_opf
//
// The weights directory is produced by `cactus convert openai/privacy-filter
// <out> --precision FP16`.

#include "test_utils.h"

#include "../cactus/engine/engine.h"
#include "../cactus/models/model.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace EngineTestUtils;

static const char* g_opf_model_path = std::getenv("CACTUS_OPF_MODEL");

namespace {

struct TestCase {
    std::string text;
    std::vector<std::string> expected_labels;  // multiset; order-insensitive
};

bool spans_match(const std::vector<cactus::engine::OPFModel::Span>& got,
                 const std::vector<std::string>& expected) {
    if (got.size() != expected.size()) return false;
    std::vector<std::string> got_labels;
    got_labels.reserve(got.size());
    for (const auto& s : got) got_labels.push_back(s.label);
    auto sorted_got = got_labels;
    auto sorted_expected = expected;
    std::sort(sorted_got.begin(), sorted_got.end());
    std::sort(sorted_expected.begin(), sorted_expected.end());
    return sorted_got == sorted_expected;
}

}  // namespace

static bool test_opf_classify() {
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║       OPF TOKEN-CLASSIFICATION TEST      ║\n"
              << "╚══════════════════════════════════════════╝\n";

    if (!g_opf_model_path) {
        std::cout << "⊘ SKIP │ CACTUS_OPF_MODEL not set\n";
        return true;
    }

    auto model = cactus::engine::create_model(g_opf_model_path);
    if (!model) {
        std::cout << "✗ FAIL │ create_model returned null\n";
        return false;
    }
    if (!model->init(g_opf_model_path, /*context=*/4096, "", /*warmup=*/false)) {
        std::cout << "✗ FAIL │ init() returned false\n";
        return false;
    }
    auto* opf = dynamic_cast<cactus::engine::OPFModel*>(model.get());
    if (!opf) {
        std::cout << "✗ FAIL │ model is not an OPFModel instance\n";
        return false;
    }

    const std::vector<TestCase> cases = {
        {"My name is Alice Smith",
         {"private_person"}},
        {"Contact me at bob.jones@example.com or 555-123-4567",
         {"private_email", "private_phone"}},
        {"Send $500 to account 1234567890 by March 15, 2025",
         {"account_number", "private_date"}},
        {"Visit https://secret-url.example/xyz for the password hunter2",
         {"private_url", "secret"}},
        {"The weather is nice today and the cat is on the mat",
         {}},  // no PII
        // Long mixed-PII paragraph. Boundary calls on opaque identifiers
        // (insurance / group / employee / plate / VIN / passport / DL) are
        // where the model is most likely to disagree with this enumeration —
        // see the README note on using `spans_contains_all` for looser checks.
        {"Jordan Ellis lives at 1427 Willow Bend Avenue, Apt. 4C, Brookhaven, NY 11719. "
         "Their phone number is 555-0147 and their email is jordan.ellis@example-test.com. "
         "According to the intake sheet, their date of birth is March 14, 1992, "
         "and their customer reference ID is CUST-48291-XY. "
         "Emergency contact information lists Morgan Ellis at 555-0199. "
         "Employment records show Jordan works at Northgate Analytics, "
         "employee ID NA-20488, with a mailing address of 88 Harbor Plaza, "
         "Suite 210, Brookhaven, NY 11719. "
         "Banking details on file include account ending in 4421 "
         "and routing placeholder 021000000. "
         "Their insurance member ID is ZX-118-44-902, group number 70021, "
         "and primary care provider is listed as Dr. Lena Hart at 555-0113. "
         "Previous residence was recorded as 77 Maple Crest Lane, Albany, NY 12203. "
         "Vehicle registration references plate KXT-2041 and "
         "VIN placeholder 1HGBH41JXMN109186. "
         "A prior support request included passport placeholder X0000000 "
         "and driver’s license placeholder D1234567",
         {
             // Observed behavior (not purely definitional — "Dr. Lena Hart" is
             // missed, the insurance string ZX-118-44-902 reads as a person,
             // and opaque identifiers (NA-20488, plate, VIN, driver's license)
             // all land in `account_number` rather than `secret`. This is a
             // regression anchor for current output, not a labeling oracle.
             "private_person", "private_person", "private_person", "private_person",
             "private_address", "private_address", "private_address", "private_address",
             "private_phone", "private_phone", "private_phone",
             "private_email",
             "private_date",
             "account_number", "account_number", "account_number",
             "account_number", "account_number", "account_number",
         }},
    };

    bool all_ok = true;
    for (const auto& c : cases) {
        Timer t;
        auto spans = opf->classify(c.text);
        double elapsed = t.elapsed_ms();

        bool ok = spans_match(spans, c.expected_labels);
        all_ok = all_ok && ok;

        std::cout << (ok ? "✓" : "✗") << " " << c.text << "\n"
                  << "  (" << spans.size() << " span" << (spans.size() == 1 ? "" : "s")
                  << ", " << std::fixed << std::setprecision(1) << elapsed << " ms)";
        for (const auto& s : spans) {
            std::cout << "\n    [" << s.token_start << ".." << s.token_end << ") " << s.label;
        }
        std::cout << "\n";
    }
    return all_ok;
}

int main() {
    bool ok = test_opf_classify();
    std::cout << (ok ? "\nOPF: PASS\n" : "\nOPF: FAIL\n");
    return ok ? 0 : 1;
}
