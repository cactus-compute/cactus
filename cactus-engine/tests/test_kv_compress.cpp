#include "test_utils.h"
#include "../src/kv_compress.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace TestUtils;
using namespace cactus::kvcompress;

// --------------------------------------------------------------------------- //
// Minimal JSON reader for the fixture file produced by gen_kv_compress_fixtures //
// --------------------------------------------------------------------------- //
namespace {

struct Fixture {
    std::string name;
    int n = 0, head_dim = 0, n_kv_heads = 0, budget = 0, sink = 0;
    float recent_frac = 0.f;
    std::vector<float> keys;                       // [n_kv_heads * n * head_dim]
    std::vector<std::vector<float>> scores;        // [n_kv_heads][n]
    std::vector<std::vector<int>> kept;            // [n_kv_heads][B]
};

struct RopeFixture {
    int head_dim = 0, abs_pos = 0, rank = 0;
    float rope_theta = 0.f;
    std::vector<float> pre_rope, post_rope_at_abs, post_rope_at_rank;
};

// Tiny tokenizing scanner. The fixture file is machine-generated with a known shape, so a
// streaming number/string/structure scanner is enough (no general-purpose JSON needed).
struct Scanner {
    const std::string& s;
    size_t i = 0;
    explicit Scanner(const std::string& str) : s(str) {}
    void skip_ws() { while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\t' || s[i] == '\r')) ++i; }
    char peek() { skip_ws(); return i < s.size() ? s[i] : '\0'; }
    void expect(char c) { skip_ws(); if (i >= s.size() || s[i] != c) throw std::runtime_error(std::string("expected ") + c); ++i; }
    bool accept(char c) { skip_ws(); if (i < s.size() && s[i] == c) { ++i; return true; } return false; }
    std::string parse_string() {
        skip_ws(); expect('"');
        std::string out;
        while (i < s.size() && s[i] != '"') out.push_back(s[i++]);
        expect('"');
        return out;
    }
    double parse_number() {
        skip_ws();
        size_t start = i;
        while (i < s.size() && (std::isdigit((unsigned char)s[i]) || s[i] == '-' || s[i] == '+' ||
                                s[i] == '.' || s[i] == 'e' || s[i] == 'E')) ++i;
        return std::stod(s.substr(start, i - start));
    }
    void skip_value() {  // skip an arbitrary JSON value (only used to skip unknown keys)
        char c = peek();
        if (c == '"') { parse_string(); return; }
        if (c == '{') { parse_object_skip(); return; }
        if (c == '[') { expect('['); if (peek() != ']') { do { skip_value(); } while (accept(',')); } expect(']'); return; }
        parse_number();
    }
    void parse_object_skip() {
        expect('{');
        if (peek() != '}') { do { parse_string(); expect(':'); skip_value(); } while (accept(',')); }
        expect('}');
    }
    template <typename F>
    void parse_object(F&& on_key) {  // on_key(key) -> bool (true if consumed value)
        expect('{');
        if (peek() != '}') {
            do {
                std::string key = parse_string();
                expect(':');
                if (!on_key(key)) skip_value();
            } while (accept(','));
        }
        expect('}');
    }
    std::vector<float> parse_float_array() {
        std::vector<float> v;
        expect('[');
        if (peek() != ']') do { v.push_back(static_cast<float>(parse_number())); } while (accept(','));
        expect(']');
        return v;
    }
    std::vector<int> parse_int_array() {
        std::vector<int> v;
        expect('[');
        if (peek() != ']') do { v.push_back(static_cast<int>(std::llround(parse_number()))); } while (accept(','));
        expect(']');
        return v;
    }
};

Fixture parse_fixture(Scanner& sc) {
    Fixture f;
    sc.parse_object([&](const std::string& k) -> bool {
        if (k == "name") { f.name = sc.parse_string(); return true; }
        if (k == "n") { f.n = (int)sc.parse_number(); return true; }
        if (k == "head_dim") { f.head_dim = (int)sc.parse_number(); return true; }
        if (k == "n_kv_heads") { f.n_kv_heads = (int)sc.parse_number(); return true; }
        if (k == "budget") { f.budget = (int)sc.parse_number(); return true; }
        if (k == "recent_frac") { f.recent_frac = (float)sc.parse_number(); return true; }
        if (k == "sink") { f.sink = (int)sc.parse_number(); return true; }
        if (k == "keys") { f.keys = sc.parse_float_array(); return true; }
        if (k == "scores") {
            sc.expect('[');
            if (sc.peek() != ']') do { f.scores.push_back(sc.parse_float_array()); } while (sc.accept(','));
            sc.expect(']');
            return true;
        }
        if (k == "kept") {
            sc.expect('[');
            if (sc.peek() != ']') do { f.kept.push_back(sc.parse_int_array()); } while (sc.accept(','));
            sc.expect(']');
            return true;
        }
        return false;
    });
    return f;
}

struct FixtureFile {
    std::vector<Fixture> fixtures;
    RopeFixture rope;
};

FixtureFile load_fixtures(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open fixture file: " + path);
    std::stringstream ss; ss << in.rdbuf();
    std::string data = ss.str();
    Scanner sc(data);
    FixtureFile out;
    sc.parse_object([&](const std::string& k) -> bool {
        if (k == "fixtures") {
            sc.expect('[');
            if (sc.peek() != ']') do { out.fixtures.push_back(parse_fixture(sc)); } while (sc.accept(','));
            sc.expect(']');
            return true;
        }
        if (k == "rope") {
            RopeFixture r;
            sc.parse_object([&](const std::string& rk) -> bool {
                if (rk == "head_dim") { r.head_dim = (int)sc.parse_number(); return true; }
                if (rk == "rope_theta") { r.rope_theta = (float)sc.parse_number(); return true; }
                if (rk == "abs_pos") { r.abs_pos = (int)sc.parse_number(); return true; }
                if (rk == "rank") { r.rank = (int)sc.parse_number(); return true; }
                if (rk == "pre_rope") { r.pre_rope = sc.parse_float_array(); return true; }
                if (rk == "post_rope_at_abs") { r.post_rope_at_abs = sc.parse_float_array(); return true; }
                if (rk == "post_rope_at_rank") { r.post_rope_at_rank = sc.parse_float_array(); return true; }
                return false;
            });
            out.rope = r;
            return true;
        }
        return false;
    });
    return out;
}

std::string fixture_path() {
    if (const char* env = std::getenv("CACTUS_KV_FIXTURES")) return env;
    return "experiments/niah/fixtures/kv_compress_fixtures.json";
}

Params params_for(const Fixture& f) {
    Params p;
    p.recent_frac = f.recent_frac;
    p.sink = (size_t)f.sink;
    p.abs_budget = f.budget;  // fixture's budget = round(budget_frac * n)
    return p;
}

using EngineTestUtils::rope_reference;

FixtureFile g_ff;
const float* head_ptr(const Fixture& f, int h) {
    return f.keys.data() + (size_t)h * f.n * f.head_dim;
}

}  // namespace

// --------------------------------------------------------------------------- //
// Tests                                                                        //
// --------------------------------------------------------------------------- //

bool test_keydiff_score_matches_python() {
    bool ok = true;
    for (const auto& f : g_ff.fixtures) {
        for (int h = 0; h < f.n_kv_heads; ++h) {
            std::vector<float> out(f.n);
            keydiff_score(head_ptr(f, h), f.n, f.head_dim, out.data());
            for (int i = 0; i < f.n; ++i) {
                if (std::abs(out[i] - f.scores[h][i]) > 1e-4f) {
                    std::cerr << "keydiff mismatch " << f.name << " h" << h << " i" << i
                              << " cpp=" << out[i] << " py=" << f.scores[h][i] << "\n";
                    ok = false;
                }
            }
        }
    }
    return ok;
}

bool test_keepset_matches_python_exactly() {
    bool ok = true;
    for (const auto& f : g_ff.fixtures) {
        Params p = params_for(f);
        for (int h = 0; h < f.n_kv_heads; ++h) {
            std::vector<float> sc(f.n);
            keydiff_score(head_ptr(f, h), f.n, f.head_dim, sc.data());
            std::vector<int> kept = keepset_for_head(sc.data(), f.n, p);
            if (kept != f.kept[h]) {
                std::cerr << "keepset mismatch " << f.name << " h" << h
                          << " cpp_size=" << kept.size() << " py_size=" << f.kept[h].size() << "\n";
                ok = false;
            }
        }
    }
    return ok;
}

bool test_exactly_B_per_cell_rectangular() {
    bool ok = true;
    for (const auto& f : g_ff.fixtures) {
        Params p = params_for(f);
        long B = std::min<long>(std::max<long>(1, f.budget), f.n);
        for (int h = 0; h < f.n_kv_heads; ++h) {
            std::vector<float> sc(f.n);
            keydiff_score(head_ptr(f, h), f.n, f.head_dim, sc.data());
            std::vector<int> kept = keepset_for_head(sc.data(), f.n, p);
            if ((long)kept.size() != B) { ok = false; continue; }
            for (size_t k = 1; k < kept.size(); ++k)
                if (kept[k] <= kept[k - 1]) ok = false;          // strictly sorted, unique
            for (int idx : kept) if (idx < 0 || idx >= f.n) ok = false;
        }
    }
    return ok;
}

bool test_sink_always_included() {
    bool ok = true;
    for (const auto& f : g_ff.fixtures) {
        long B = std::min<long>(std::max<long>(1, f.budget), f.n);
        long sink = std::min<long>(std::max<long>(0, f.sink), f.n);
        if (sink > B) continue;  // budget too small to hold the full sink
        for (const auto& kept : f.kept) {
            for (int s = 0; s < sink; ++s) {
                if (std::find(kept.begin(), kept.end(), s) == kept.end()) ok = false;
            }
        }
    }
    return ok;
}

bool test_recent_window_included() {
    bool ok = true;
    for (const auto& f : g_ff.fixtures) {
        long B = std::min<long>(std::max<long>(1, f.budget), f.n);
        long sink = std::min<long>(std::max<long>(0, f.sink), f.n);
        long n_recent = std::min<long>(std::lround((double)f.recent_frac * B), f.n);
        // The full recent window is guaranteed kept only when reserved fits in the budget.
        if (sink + n_recent > B) continue;
        for (const auto& kept : f.kept) {
            for (long r = f.n - n_recent; r < f.n; ++r) {
                if (r < 0) continue;
                if (std::find(kept.begin(), kept.end(), (int)r) == kept.end()) ok = false;
            }
        }
    }
    return ok;
}

bool test_outlier_kept_filler_dropped() {
    // Fixtures plant a centroid-opposite outlier at mid+h; the adjacent filler stays on the
    // centroid. KeyDiff must keep the outlier (a mid-context geometry pick).
    bool ok = true;
    for (const auto& f : g_ff.fixtures) {
        if (f.name != std::string("base_keydiff")) continue;
        int mid = f.n / 2;
        for (int h = 0; h < f.n_kv_heads; ++h) {
            int outlier = mid + h;
            const auto& kept = f.kept[h];
            if (std::find(kept.begin(), kept.end(), outlier) == kept.end()) {
                std::cerr << "outlier not kept " << f.name << " h" << h << " idx" << outlier << "\n";
                ok = false;
            }
        }
    }
    return ok;
}

bool test_padding_when_budget_exceeds_reserves() {
    // reserved_overflow: large recent_frac + sink on a tight budget exercises the |reserved|>B
    // fallback. Still exactly B unique sorted indices.
    bool ok = false;
    for (const auto& f : g_ff.fixtures) {
        if (f.name != std::string("reserved_overflow")) continue;
        ok = true;
        Params p = params_for(f);
        long B = std::min<long>(std::max<long>(1, f.budget), f.n);
        for (int h = 0; h < f.n_kv_heads; ++h) {
            std::vector<float> sc(f.n);
            keydiff_score(head_ptr(f, h), f.n, f.head_dim, sc.data());
            std::vector<int> kept = keepset_for_head(sc.data(), f.n, p);
            if ((long)kept.size() != B) ok = false;
            if (kept != f.kept[h]) ok = false;
        }
    }
    return ok;
}

bool test_renumber_mapping() {
    // RoPE-at-rank for a survivor equals the reference rope of that vector at its rank position.
    // Also check Route-B delta rotation: rotating the post-RoPE-at-abs vector by (rank - abs)
    // recovers post-RoPE-at-rank.
    const RopeFixture& r = g_ff.rope;
    if (r.head_dim == 0) return false;

    // RoPE-at-rank matches reference.
    std::vector<float> at_rank = rope_reference(r.pre_rope, r.rank, r.rope_theta);
    for (int i = 0; i < r.head_dim; ++i)
        if (std::abs(at_rank[i] - r.post_rope_at_rank[i]) > 1e-3f) return false;

    // Route B via the PRODUCTION rope_rotate_row (shipped code, not a local reimplementation):
    // rotating post_rope_at_abs by (rank - abs) in place must yield post_rope_at_rank.
    std::vector<float> delta = r.post_rope_at_abs;
    rope_rotate_row(delta.data(), (size_t)r.head_dim, r.rope_theta,
                    (double)(r.rank - r.abs_pos));
    for (int i = 0; i < r.head_dim; ++i)
        if (std::abs(delta[i] - r.post_rope_at_rank[i]) > 1e-3f) return false;

    return true;
}

int main() {
    TestUtils::TestRunner runner("KV Compress Math Tests");
    try {
        g_ff = load_fixtures(fixture_path());
    } catch (const std::exception& e) {
        std::cerr << "Failed to load fixtures: " << e.what() << "\n"
                  << "Run: python experiments/niah/gen_kv_compress_fixtures.py\n"
                  << "Or set CACTUS_KV_FIXTURES to the fixture json path.\n";
        return 1;
    }
    std::cout << "Loaded " << g_ff.fixtures.size() << " fixtures\n";

    runner.run_test("keydiff_score_matches_python", test_keydiff_score_matches_python());
    runner.run_test("keepset_matches_python_exactly", test_keepset_matches_python_exactly());
    runner.run_test("exactly_B_per_cell_rectangular", test_exactly_B_per_cell_rectangular());
    runner.run_test("sink_always_included", test_sink_always_included());
    runner.run_test("recent_window_included", test_recent_window_included());
    runner.run_test("outlier_kept_filler_dropped", test_outlier_kept_filler_dropped());
    runner.run_test("padding_when_budget_exceeds_reserves", test_padding_when_budget_exceeds_reserves());
    runner.run_test("renumber_mapping", test_renumber_mapping());

    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
