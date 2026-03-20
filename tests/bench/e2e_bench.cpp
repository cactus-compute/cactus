#include "e2e_driver.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// Minimal JSON parser for model config
struct ModelVariant {
    std::string backend;
    std::string path;
};

struct ModelConfig {
    std::string name;
    std::vector<ModelVariant> variants;
};

static std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    size_t end = s.find_last_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    return s.substr(start, end - start + 1);
}

// Simple JSON parser for the model config format
static std::vector<ModelConfig> parse_model_config(const std::string& path) {
    std::vector<ModelConfig> configs;
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "Error: cannot open model config: " << path << "\n";
        return configs;
    }

    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());

    // Find each model block
    size_t pos = 0;
    while ((pos = content.find("\"name\"", pos)) != std::string::npos) {
        // Extract name
        size_t colon = content.find(':', pos);
        size_t quote1 = content.find('"', colon + 1);
        size_t quote2 = content.find('"', quote1 + 1);
        if (colon == std::string::npos || quote1 == std::string::npos || quote2 == std::string::npos) break;

        ModelConfig mc;
        mc.name = content.substr(quote1 + 1, quote2 - quote1 - 1);

        // Find variants block
        size_t var_start = content.find("\"variants\"", quote2);
        if (var_start == std::string::npos) break;
        size_t brace_start = content.find('{', var_start);
        if (brace_start == std::string::npos) break;

        // Find matching closing brace
        int depth = 1;
        size_t brace_end = brace_start + 1;
        while (brace_end < content.size() && depth > 0) {
            if (content[brace_end] == '{') depth++;
            else if (content[brace_end] == '}') depth--;
            brace_end++;
        }

        std::string variants_block = content.substr(brace_start + 1, brace_end - brace_start - 2);

        // Parse key-value pairs in variants
        size_t vpos = 0;
        while (vpos < variants_block.size()) {
            size_t kq1 = variants_block.find('"', vpos);
            if (kq1 == std::string::npos) break;
            size_t kq2 = variants_block.find('"', kq1 + 1);
            if (kq2 == std::string::npos) break;
            std::string key = variants_block.substr(kq1 + 1, kq2 - kq1 - 1);

            size_t vq1 = variants_block.find('"', kq2 + 1);
            if (vq1 == std::string::npos) break;
            // Skip the colon
            size_t vq2 = variants_block.find('"', vq1 + 1);
            if (vq2 == std::string::npos) break;
            std::string val = variants_block.substr(vq1 + 1, vq2 - vq1 - 1);

            mc.variants.push_back({key, val});
            vpos = vq2 + 1;
        }

        configs.push_back(mc);
        pos = brace_end;
    }

    return configs;
}

static void print_usage() {
    std::cerr << "Usage: e2e_bench [options]\n"
              << "  --model-config <path>   Model config JSON file (required unless --model-path)\n"
              << "  --model-path <path>     Direct path to a single Cactus model\n"
              << "  --model <name>          Run only this model from config\n"
              << "  --backends <list>       Comma-separated backend filter (e.g. cactus,llama_cpp)\n"
              << "  --rounds <n>            Number of measured rounds (default: 10)\n"
              << "  --prompt <text>         Override prompt text\n"
              << "  --max-tokens <n>        Max tokens to generate (default: 128)\n"
              << "  --threads <n>           Thread count for backends\n"
              << "  --output <path>         Write CSV results to file\n"
              << "  --dry-run               Print schedule without running\n"
              << "  --help                  Show this help\n";
}

static bool backend_matches(const char* name, const std::string& filter) {
    if (filter.empty()) return true;
    std::istringstream ss(filter);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (trim(item) == name) return true;
    }
    return false;
}

int main(int argc, char** argv) {
    std::string config_path;
    std::string model_path;
    std::string model_filter;
    std::string backends_filter;
    std::string output_path;
    std::string prompt = "Explain the theory of relativity in simple terms.";
    int rounds = 10;
    int max_tokens = 128;
    int threads = 0;
    bool dry_run = false;

    for (int i = 1; i < argc; i++) {
        std::string arg(argv[i]);
        if (arg == "--help") { print_usage(); return 0; }
        else if (arg == "--model-config" && i + 1 < argc) config_path = argv[++i];
        else if (arg == "--model-path" && i + 1 < argc) model_path = argv[++i];
        else if (arg == "--model" && i + 1 < argc) model_filter = argv[++i];
        else if (arg == "--backends" && i + 1 < argc) backends_filter = argv[++i];
        else if (arg == "--rounds" && i + 1 < argc) rounds = std::stoi(argv[++i]);
        else if (arg == "--prompt" && i + 1 < argc) prompt = argv[++i];
        else if (arg == "--max-tokens" && i + 1 < argc) max_tokens = std::stoi(argv[++i]);
        else if (arg == "--threads" && i + 1 < argc) threads = std::stoi(argv[++i]);
        else if (arg == "--output" && i + 1 < argc) output_path = argv[++i];
        else if (arg == "--dry-run") dry_run = true;
        else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage();
            return 1;
        }
    }

    // Build list of models to benchmark
    struct BenchModel {
        std::string name;
        std::vector<ModelVariant> variants;
    };
    std::vector<BenchModel> models;

    if (!model_path.empty()) {
        // Direct model path mode — cactus only
        BenchModel bm;
        bm.name = "direct";
        bm.variants.push_back({"cactus", model_path});
        models.push_back(bm);
    } else if (!config_path.empty()) {
        auto configs = parse_model_config(config_path);
        for (auto& mc : configs) {
            if (!model_filter.empty() && mc.name != model_filter) continue;
            BenchModel bm;
            bm.name = mc.name;
            bm.variants = mc.variants;
            models.push_back(bm);
        }
    } else {
        std::cerr << "Error: must specify --model-config or --model-path\n";
        print_usage();
        return 1;
    }

    if (models.empty()) {
        std::cerr << "Error: no models to benchmark\n";
        return 1;
    }

    // Get available backends
    const auto& all_backends = e2e::get_e2e_backends();
    std::vector<int> active_indices;
    for (int i = 0; i < static_cast<int>(all_backends.size()); i++) {
        if (!backend_matches(all_backends[i].name, backends_filter)) continue;
        if (!all_backends[i].available()) {
            std::cerr << "[skip] " << all_backends[i].name << " not available\n";
            continue;
        }
        active_indices.push_back(i);
    }

    if (active_indices.empty()) {
        std::cerr << "Error: no backends available\n";
        return 1;
    }

    std::cerr << "Active backends: ";
    for (size_t i = 0; i < active_indices.size(); i++) {
        if (i > 0) std::cerr << ", ";
        std::cerr << all_backends[active_indices[i]].name;
    }
    std::cerr << "\n";

    int n_active = static_cast<int>(active_indices.size());
    auto schedule = e2e::build_schedule(n_active, rounds);

    // Also build a warmup round (round -1, base order)
    std::vector<e2e::ScheduledRun> warmup_schedule;
    for (int slot = 0; slot < n_active; slot++) {
        warmup_schedule.push_back({slot, -1});
    }

    if (dry_run) {
        std::cerr << "\n=== DRY RUN: Schedule ===\n";
        std::cerr << "Warmup round: ";
        for (const auto& sr : warmup_schedule) {
            std::cerr << all_backends[active_indices[sr.backend_idx]].name << " ";
        }
        std::cerr << "\n";
        for (int r = 0; r < rounds; r++) {
            std::cerr << "Round " << (r + 1) << ": ";
            for (const auto& sr : schedule) {
                if (sr.round == r) {
                    std::cerr << all_backends[active_indices[sr.backend_idx]].name << " ";
                }
            }
            std::cerr << "\n";
        }
        return 0;
    }

    std::vector<e2e::E2ERunRecord> all_records;

    for (auto& bm : models) {
        std::cerr << "\n=== Model: " << bm.name << " ===\n";

        // Load phase: load each active backend's variant
        std::vector<void*> handles(n_active, nullptr);
        for (int ai = 0; ai < n_active; ai++) {
            const auto& backend = all_backends[active_indices[ai]];
            std::string path;
            for (const auto& v : bm.variants) {
                if (v.backend == backend.name) {
                    path = v.path;
                    break;
                }
            }
            if (path.empty()) {
                std::cerr << "[skip] No model variant for " << backend.name << "\n";
                continue;
            }
            std::cerr << "[load] " << backend.name << " <- " << path << "\n";
            handles[ai] = backend.load(path.c_str(), threads);
            if (!handles[ai]) {
                std::cerr << "[fail] " << backend.name << " failed to load\n";
            }
        }

        // Warmup round (discarded, 16 tokens)
        std::cerr << "[warmup] running warmup round...\n";
        for (const auto& sr : warmup_schedule) {
            if (!handles[sr.backend_idx]) continue;
            const auto& backend = all_backends[active_indices[sr.backend_idx]];
            backend.generate(handles[sr.backend_idx], prompt.c_str(), 16);
        }

        // Measured rounds
        std::vector<int> rep_counter_per_backend(n_active, 0);
        for (const auto& sr : schedule) {
            if (!handles[sr.backend_idx]) continue;
            const auto& backend = all_backends[active_indices[sr.backend_idx]];

            std::cerr << "[round " << (sr.round + 1) << "] " << backend.name << "...";
            auto result = backend.generate(handles[sr.backend_idx], prompt.c_str(), max_tokens);
            std::cerr << " decode=" << std::fixed << std::setprecision(1) << result.decode_tps
                      << " tps, prefill=" << result.prefill_tps << " tps\n";

            e2e::E2ERunRecord rec;
            rec.backend = backend.name;
            rec.model = bm.name;
            rec.rep = rep_counter_per_backend[sr.backend_idx]++;
            rec.result = result;
            all_records.push_back(rec);
        }

        // Print summary for this model
        e2e::print_summary(bm.name, all_records, rounds);

        // Unload phase
        for (int ai = 0; ai < n_active; ai++) {
            if (handles[ai]) {
                all_backends[active_indices[ai]].unload(handles[ai]);
            }
        }
    }

    // Write CSV if requested
    if (!output_path.empty()) {
        e2e::write_csv(output_path, all_records);
        std::cerr << "Results written to: " << output_path << "\n";
    }

    return 0;
}
