#include "test_utils.h"

#include "../cactus/engine/engine.h"
#include "../cactus/models/model.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <map>
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

    const std::string long_paragraph =
        "Jordan Ellis lives at 1427 Willow Bend Avenue, Apt. 4C, Brookhaven, NY 11719. "
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
        "and driver's license placeholder D1234567";

    const std::vector<std::string> long_paragraph_labels = {
        "private_person", "private_person", "private_person", "private_person",
        "private_address", "private_address", "private_address", "private_address",
        "private_phone", "private_phone", "private_phone",
        "private_email",
        "private_date",
        "account_number", "account_number", "account_number",
        "account_number", "account_number", "account_number",
    };

    std::string long_paragraph_x6;
    long_paragraph_x6.reserve(long_paragraph.size() * 6 + 12);
    for (int i = 0; i < 6; ++i) {
        if (i > 0) long_paragraph_x6 += "\n\n";
        long_paragraph_x6 += long_paragraph;
    }
    std::vector<std::string> long_paragraph_x6_labels;
    auto push_n = [&](const char* label, int n) {
        for (int i = 0; i < n; ++i) long_paragraph_x6_labels.emplace_back(label);
    };
    push_n("account_number",  42);
    push_n("private_address", 25);
    push_n("private_person",  24);
    push_n("private_phone",   18);
    push_n("private_date",     6);
    push_n("private_email",    6);

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
         {}},
        {long_paragraph, long_paragraph_labels},
        {long_paragraph_x6, long_paragraph_x6_labels},
    };

    bool all_ok = true;
    for (const auto& c : cases) {
        Timer t;
        auto spans = opf->classify(c.text);
        double elapsed = t.elapsed_ms();

        bool ok = spans_match(spans, c.expected_labels);
        all_ok = all_ok && ok;

        std::string preview = c.text.substr(0, 80);
        if (c.text.size() > 80) preview += "...";
        std::cout << (ok ? "✓" : "✗") << " " << preview << "\n"
                  << "  (" << spans.size() << " span" << (spans.size() == 1 ? "" : "s")
                  << ", " << std::fixed << std::setprecision(1) << elapsed << " ms";
        if (!ok) {
            std::cout << ", expected " << c.expected_labels.size();
        }
        std::cout << ")";
        if (c.text.size() <= 200) {
            for (const auto& s : spans) {
                std::cout << "\n    [" << s.token_start << ".." << s.token_end << ") " << s.label;
            }
        } else {
            std::map<std::string, int> hist;
            for (const auto& s : spans) hist[s.label]++;
            for (const auto& kv : hist) {
                std::cout << "\n    " << kv.second << " × " << kv.first;
            }
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
