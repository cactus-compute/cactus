#include "engine.h"
#include "src/kimi_k2_model.h"

#include <chrono>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
}

std::vector<uint32_t> parse_ids(const std::string& text) {
    std::vector<uint32_t> ids;
    std::stringstream ss(text);
    std::string part;
    while (std::getline(ss, part, ',')) {
        if (part.empty()) continue;
        ids.push_back(static_cast<uint32_t>(std::stoul(part)));
    }
    return ids;
}

void usage(const char* argv0) {
    std::cerr << "Usage: " << argv0
              << " --model <dir> --ids <comma_ids> [--max-new-tokens n] [--context n]"
              << " [--temperature f] [--top-p f] [--top-k n] [--stop-ids ids]"
              << " [--profile-prefix path] [--warmup-moe-experts]\n";
}

std::string profile_path(const std::string& prefix, const std::string& suffix) {
    if (prefix.empty()) return "";
    return prefix + "_" + suffix + ".txt";
}

} // namespace

int main(int argc, char** argv) {
    std::string model_dir;
    std::string ids_csv;
    std::string stop_ids_csv;
    std::string profile_prefix;
    bool warmup_moe_experts = false;
    size_t max_new_tokens = 128;
    size_t context_size = 2048;
    float temperature = 0.0f;
    float top_p = 1.0f;
    size_t top_k = 1;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need_value = [&](const char* name) -> char* {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };
        if (arg == "--model") {
            model_dir = need_value("--model");
        } else if (arg == "--ids") {
            ids_csv = need_value("--ids");
        } else if (arg == "--max-new-tokens") {
            max_new_tokens = static_cast<size_t>(std::stoul(need_value("--max-new-tokens")));
        } else if (arg == "--context") {
            context_size = static_cast<size_t>(std::stoul(need_value("--context")));
        } else if (arg == "--temperature") {
            temperature = std::stof(need_value("--temperature"));
        } else if (arg == "--top-p") {
            top_p = std::stof(need_value("--top-p"));
        } else if (arg == "--top-k") {
            top_k = static_cast<size_t>(std::stoul(need_value("--top-k")));
        } else if (arg == "--stop-ids") {
            stop_ids_csv = need_value("--stop-ids");
        } else if (arg == "--profile-prefix") {
            profile_prefix = need_value("--profile-prefix");
        } else if (arg == "--warmup-moe-experts") {
            warmup_moe_experts = true;
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            return 0;
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (model_dir.empty() || ids_csv.empty()) {
        usage(argv[0]);
        return 2;
    }

    try {
        std::vector<uint32_t> prompt_ids = parse_ids(ids_csv);
        if (prompt_ids.empty()) {
            throw std::runtime_error("--ids must contain at least one token id");
        }

        std::unordered_set<uint32_t> stop_ids;
        for (uint32_t id : parse_ids(stop_ids_csv)) stop_ids.insert(id);

        cactus::engine::KimiK2Model model;
        auto init_start = Clock::now();
        if (!model.init(model_dir, context_size, "", false)) {
            std::cerr << "init failed\n";
            return 1;
        }
        auto init_end = Clock::now();
        std::cout << "READY init_ms=" << elapsed_ms(init_start, init_end) << "\n" << std::flush;

        if (warmup_moe_experts) {
            auto warmup_start = Clock::now();
            model.prefetch_moe_expert_pages();
            auto warmup_end = Clock::now();
            std::cout << "WARMUP moe_expert_prefetch_ms=" << elapsed_ms(warmup_start, warmup_end)
                      << "\n" << std::flush;
        }

        uint32_t current = 0;
        size_t generated = 0;
        auto generation_start = Clock::now();
        auto first_start = generation_start;
        auto first_end = first_start;

        if (max_new_tokens > 0) {
            auto step_start = Clock::now();
            current = model.decode(prompt_ids, 0.0f, 1.0f, 1, profile_path(profile_prefix, "prefill"));
            bool first = true;
            auto step_end = Clock::now();
            first_end = step_end;
            if (first) {
                ++generated;
                std::cout << "TOKEN id=" << current
                          << " step_ms=" << elapsed_ms(step_start, step_end)
                          << " elapsed_ms=" << elapsed_ms(generation_start, step_end)
                          << "\n" << std::flush;
            }
        } else {
            model.prefill(prompt_ids, 128, "", false);
            first_end = Clock::now();
        }

        bool stopped = generated > 0 && stop_ids.find(current) != stop_ids.end();
        while (!stopped && generated < max_new_tokens) {
            auto step_start = Clock::now();
            current = model.decode({current}, temperature, top_p, top_k,
                                   profile_path(profile_prefix, "decode_" + std::to_string(generated)));
            auto step_end = Clock::now();
            ++generated;
            std::cout << "TOKEN id=" << current
                      << " step_ms=" << elapsed_ms(step_start, step_end)
                      << " elapsed_ms=" << elapsed_ms(generation_start, step_end)
                      << "\n" << std::flush;
            stopped = stop_ids.find(current) != stop_ids.end();
        }

        auto end = Clock::now();
        double total_ms = elapsed_ms(generation_start, end);
        double ttft_ms = elapsed_ms(generation_start, first_end);
        double decode_ms = std::max(0.0, total_ms - ttft_ms);
        double decode_tps = (generated > 1 && decode_ms > 0.0)
            ? (static_cast<double>(generated - 1) * 1000.0) / decode_ms
            : 0.0;
        std::cout << "DONE tokens=" << generated
                  << " ttft_ms=" << ttft_ms
                  << " total_ms=" << total_ms
                  << " decode_tps=" << decode_tps
                  << " stopped=" << (stopped ? 1 : 0)
                  << "\n" << std::flush;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
