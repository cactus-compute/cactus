// Single-file BFCL simple_python function-calling benchmark.
//
// Builds as `test_bfcl_simple`; expects a Gemma-family or LFM2 model dir
// in CACTUS_TEST_MODEL. Runs 20 embedded BFCL cases through cactus_complete
// and reports a pass rate, matching BFCL's "simple" scoring (exactly one
// call, name matches one alternative, every ground-truth arg's value
// appears in its acceptable-values list; "" in the list means the arg may
// be omitted).

#include "../cactus/ffi/cactus_ffi.h"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

namespace {

struct Case {
    const char* id;
    const char* messages_json;
    const char* tools_json;
    const char* ground_truth_json;
};

#include "test_bfcl_simple_cases.inc"

// ---------- minimal JSON cursor helpers ----------

void skip_ws(const std::string& s, size_t& p) {
    while (p < s.size() && std::isspace(static_cast<unsigned char>(s[p]))) p++;
}

size_t end_of_string(const std::string& s, size_t p) {
    size_t q = p + 1;
    while (q < s.size()) {
        if (s[q] == '\\' && q + 1 < s.size()) { q += 2; continue; }
        if (s[q] == '"') return q + 1;
        q++;
    }
    return s.size();
}

size_t end_of_bracketed(const std::string& s, size_t p, char open, char close) {
    int depth = 0;
    bool in_str = false;
    while (p < s.size()) {
        char c = s[p];
        if (in_str) {
            if (c == '\\' && p + 1 < s.size()) { p += 2; continue; }
            if (c == '"') in_str = false;
        } else {
            if (c == '"') in_str = true;
            else if (c == open) depth++;
            else if (c == close) { depth--; if (depth == 0) return p + 1; }
        }
        p++;
    }
    return s.size();
}

size_t end_of_value(const std::string& s, size_t p) {
    skip_ws(s, p);
    if (p >= s.size()) return p;
    char c = s[p];
    if (c == '"') return end_of_string(s, p);
    if (c == '{') return end_of_bracketed(s, p, '{', '}');
    if (c == '[') return end_of_bracketed(s, p, '[', ']');
    size_t q = p;
    while (q < s.size() && s[q] != ',' && s[q] != ']' && s[q] != '}' &&
           !std::isspace(static_cast<unsigned char>(s[q]))) q++;
    return q;
}

std::string trim(const std::string& s) {
    size_t a = 0;
    while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a]))) a++;
    size_t b = s.size();
    while (b > a && std::isspace(static_cast<unsigned char>(s[b-1]))) b--;
    return s.substr(a, b - a);
}

std::string extract_field(const std::string& obj, const std::string& key) {
    size_t p = 0;
    skip_ws(obj, p);
    if (p >= obj.size() || obj[p] != '{') return {};
    p++;
    while (p < obj.size()) {
        skip_ws(obj, p);
        if (p >= obj.size() || obj[p] == '}') break;
        if (obj[p] == ',') { p++; continue; }
        if (obj[p] != '"') break;
        size_t key_end = end_of_string(obj, p);
        std::string k = obj.substr(p + 1, key_end - p - 2);
        p = key_end;
        skip_ws(obj, p);
        if (p < obj.size() && obj[p] == ':') p++;
        skip_ws(obj, p);
        size_t val_end = end_of_value(obj, p);
        if (k == key) return obj.substr(p, val_end - p);
        p = val_end;
    }
    return {};
}

std::vector<std::string> array_items(const std::string& arr) {
    std::vector<std::string> out;
    size_t p = 0;
    skip_ws(arr, p);
    if (p >= arr.size() || arr[p] != '[') return out;
    p++;
    while (p < arr.size()) {
        skip_ws(arr, p);
        if (p >= arr.size() || arr[p] == ']') break;
        if (arr[p] == ',') { p++; continue; }
        size_t end = end_of_value(arr, p);
        out.push_back(arr.substr(p, end - p));
        p = end;
    }
    return out;
}

std::map<std::string, std::string> object_fields(const std::string& obj) {
    std::map<std::string, std::string> out;
    size_t p = 0;
    skip_ws(obj, p);
    if (p >= obj.size() || obj[p] != '{') return out;
    p++;
    while (p < obj.size()) {
        skip_ws(obj, p);
        if (p >= obj.size() || obj[p] == '}') break;
        if (obj[p] == ',') { p++; continue; }
        if (obj[p] != '"') break;
        size_t key_end = end_of_string(obj, p);
        std::string k = obj.substr(p + 1, key_end - p - 2);
        p = key_end;
        skip_ws(obj, p);
        if (p < obj.size() && obj[p] == ':') p++;
        skip_ws(obj, p);
        size_t val_end = end_of_value(obj, p);
        out[k] = obj.substr(p, val_end - p);
        p = val_end;
    }
    return out;
}

std::string decode_string_literal(const std::string& quoted) {
    if (quoted.size() < 2 || quoted.front() != '"' || quoted.back() != '"') return quoted;
    std::string out;
    for (size_t i = 1; i + 1 < quoted.size(); i++) {
        char c = quoted[i];
        if (c == '\\' && i + 2 < quoted.size()) {
            char n = quoted[i + 1];
            if (n == 'n') out += '\n';
            else if (n == 't') out += '\t';
            else if (n == '"') out += '"';
            else if (n == '\\') out += '\\';
            else out += n;
            i++;
        } else {
            out += c;
        }
    }
    return out;
}

// ---------- value matcher ----------

// Gemma 4 sometimes emits bare identifiers instead of quoted strings:
// {"word": TANGERINE} vs {"word": "TANGERINE"}. Treat them the same.
std::string maybe_quote_bare_identifier(const std::string& s) {
    if (s.empty() || s.front() == '"' || s.front() == '{' || s.front() == '[') return s;
    for (char c : s) {
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-' || c == '.')) return s;
    }
    if (std::isdigit(static_cast<unsigned char>(s.front()))) return s;
    return "\"" + s + "\"";
}

// Accept x^n and x**n as equivalent, plus whitespace-insensitive.
std::string normalize_math(const std::string& s) {
    std::string out;
    for (size_t i = 0; i < s.size(); i++) {
        char c = s[i];
        if (std::isspace(static_cast<unsigned char>(c))) continue;
        if (c == '^') { out += "**"; continue; }
        out += c;
    }
    return out;
}

bool values_equal(const std::string& a_raw, const std::string& b_raw) {
    std::string a = maybe_quote_bare_identifier(trim(a_raw));
    std::string b = maybe_quote_bare_identifier(trim(b_raw));
    if (a == b) return true;
    try {
        size_t ea = 0, eb = 0;
        double da = std::stod(a, &ea);
        double db = std::stod(b, &eb);
        if (ea == a.size() && eb == b.size() && std::fabs(da - db) < 1e-9) return true;
    } catch (...) {}
    if (!a.empty() && !b.empty() && a.front() == '"' && b.front() == '"') {
        std::string da = decode_string_literal(a);
        std::string db = decode_string_literal(b);
        return da == db || normalize_math(da) == normalize_math(db);
    }
    if (!a.empty() && !b.empty() && a.front() == '[' && b.front() == '[') {
        auto ai = array_items(a);
        auto bi = array_items(b);
        if (ai.size() != bi.size()) return false;
        for (size_t i = 0; i < ai.size(); i++) if (!values_equal(ai[i], bi[i])) return false;
        return true;
    }
    return false;
}

bool args_match(const std::string& call_args_json, const std::string& gt_args_json) {
    auto call_args = object_fields(call_args_json);
    auto gt_fields = object_fields(gt_args_json);
    for (const auto& kv : gt_fields) {
        const std::string& key = kv.first;
        auto acceptable = array_items(kv.second);
        bool omission_ok = false;
        for (const auto& a : acceptable) if (trim(a) == "\"\"") { omission_ok = true; break; }

        auto it = call_args.find(key);
        if (it == call_args.end()) {
            if (omission_ok) continue;
            return false;
        }
        bool any = false;
        for (const auto& a : acceptable) {
            std::string av = trim(a);
            if (av == "\"\"") continue;
            if (values_equal(it->second, a)) { any = true; break; }
        }
        if (!any) return false;
    }
    return true;
}

// ---------- runner ----------

bool run_case(cactus_model_t model, const Case& c, std::string& note, std::string& got) {
    const char* options = R"({
        "max_tokens": 512,
        "temperature": 0,
        "top_k": 1,
        "force_tools": true,
        "enable_thinking_if_supported": false,
        "auto_handoff": false,
        "telemetry_enabled": false
    })";

    cactus_reset(model);

    std::vector<char> resp(32 * 1024);
    int rc = cactus_complete(model, c.messages_json, resp.data(), resp.size(),
                             options, c.tools_json, nullptr, nullptr, nullptr, 0);
    if (rc < 0) {
        note = "cactus_complete failed (" + std::to_string(rc) + ")";
        return false;
    }
    std::string response(resp.data());

    auto fcs = array_items(extract_field(response, "function_calls"));
    if (fcs.empty()) {
        note = "no function_calls produced";
        return false;
    }
    if (fcs.size() > 1) {
        note = "expected 1 call, got " + std::to_string(fcs.size());
        return false;
    }
    got = fcs[0];

    std::string name_raw = extract_field(fcs[0], "name");
    std::string args_raw = extract_field(fcs[0], "arguments");
    if (name_raw.empty() || args_raw.empty()) { note = "malformed call"; return false; }
    std::string model_name = decode_string_literal(name_raw);

    for (const auto& alt : array_items(trim(c.ground_truth_json))) {
        auto entries = object_fields(alt);
        if (entries.size() != 1) continue;
        const auto& gt_name = entries.begin()->first;
        const auto& gt_args = entries.begin()->second;
        if (model_name != gt_name) { note = "name mismatch: got '" + model_name + "' expected '" + gt_name + "'"; continue; }
        if (args_match(args_raw, gt_args)) return true;
        note = "args don't match " + gt_name;
    }
    if (note.empty()) note = "no ground-truth alternative matched";
    return false;
}

} // namespace

int main() {
    const char* model_path = std::getenv("CACTUS_TEST_MODEL");
    if (!model_path) {
        std::fprintf(stderr, "SKIP: set CACTUS_TEST_MODEL to a tool-capable model directory\n");
        return 0;
    }

    std::printf("BFCL simple_python (%zu cases)\nModel: %s\n%s\n",
                sizeof(kCases) / sizeof(kCases[0]), model_path, std::string(60, '-').c_str());

    cactus_model_t model = cactus_init(model_path, nullptr, false);
    if (!model) {
        const char* err = cactus_get_last_error();
        std::fprintf(stderr, "Failed to init model (%s)\n", err ? err : "");
        return 1;
    }

    size_t passed = 0;
    const size_t case_count = sizeof(kCases) / sizeof(kCases[0]);
    for (size_t i = 0; i < case_count; i++) {
        const auto& c = kCases[i];
        std::string note, got;
        bool ok = run_case(model, c, note, got);
        if (ok) passed++;
        std::printf("%s %-24s", ok ? "[PASS]" : "[FAIL]", c.id);
        if (!ok) {
            std::printf("  -- %s", note.c_str());
            if (!got.empty()) std::printf("\n         got: %s", got.c_str());
        }
        std::printf("\n");
    }

    cactus_destroy(model);

    std::printf("%s\nPass rate: %zu/%zu (%.1f%%)\n",
                std::string(60, '-').c_str(), passed, case_count,
                100.0 * static_cast<double>(passed) / static_cast<double>(case_count));
    return passed == case_count ? 0 : 1;
}
