#include "../cactus/ffi/cactus_ffi.h"
#include "../cactus/kernel/kernel.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <cstring>

constexpr int MAX_TOKENS = 600;
constexpr size_t MAX_BYTES_PER_TOKEN = 64;
constexpr size_t RESPONSE_BUFFER_SIZE = MAX_TOKENS * MAX_BYTES_PER_TOKEN;
constexpr int NUM_REPS = 10;

static const char* PROMPT = "Explain the theory of general relativity in detail, covering spacetime curvature, "
    "the equivalence principle, gravitational time dilation, and how it differs from Newtonian gravity.";

struct BenchState {
    std::vector<double> token_times_ms;
    std::chrono::high_resolution_clock::time_point prev_time;
    int token_index;
};

void bench_callback(const char* /*token*/, uint32_t /*token_id*/, void* user_data) {
    auto* state = static_cast<BenchState*>(user_data);
    auto now = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(now - state->prev_time).count();
    state->token_times_ms.push_back(ms);
    state->prev_time = now;
    state->token_index++;
}

struct Variant { int id; const char* name; };

void run_pass(cactus_model_t model, const Variant& v, int num_reps,
              const std::string& messages, const std::string& options,
              std::ofstream* csv, int rep_offset = 0) {
    for (int rep = 0; rep < num_reps; rep++) {
        cactus_set_decode_attention_variant(v.id);
        cactus_reset(model);
        cactus_reset_attn_counters();

        BenchState state;
        state.token_index = 0;
        state.prev_time = std::chrono::high_resolution_clock::now();

        std::vector<char> response_buffer(RESPONSE_BUFFER_SIZE, 0);

        std::cerr << "  rep " << (rep + 1) << "/" << num_reps << "... " << std::flush;

        int result = cactus_complete(
            model,
            messages.c_str(),
            response_buffer.data(),
            response_buffer.size(),
            options.c_str(),
            nullptr,
            bench_callback,
            &state
        );

        if (result < 0) {
            std::cerr << "ERROR\n";
            return;
        }

        uint64_t attn_ns = 0, attn_calls = 0;
        cactus_get_attn_counters(&attn_ns, &attn_calls);
        double total_ms = 0;
        for (auto& t : state.token_times_ms) total_ms += t;
        double ms_per_tok = (state.token_index > 1)
            ? (total_ms - state.token_times_ms[0]) / (state.token_index - 1)
            : 0;

        std::cerr << state.token_index << " tok, " << ms_per_tok << " ms/tok";
        if (attn_calls > 0) {
            std::cerr << ", attn=" << (attn_ns / 1e6) << "ms"
                      << " (" << (attn_ns / attn_calls / 1e3) << "us/call)";
        }
        std::cerr << "\n";

        if (csv) {
            for (int i = 0; i < (int)state.token_times_ms.size(); i++) {
                *csv << v.name << "," << (rep + rep_offset) << "," << i << ","
                     << state.token_times_ms[i] << "\n";
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path> [output.csv] [--variant deferscale|interleaved]\n";
        return 1;
    }

    const char* model_path = argv[1];
    const char* csv_path = "attn_e2e_results.csv";
    int base_variant_id = 0;
    const char* base_variant_name = "baseline";
    int test_variant_id = 2;
    const char* test_variant_name = "deferscale";

    auto parse_variant = [](const std::string& vname, int& id, const char*& name) {
        if (vname == "interleaved") { id = 1; name = "interleaved"; }
        else if (vname == "deferscale") { id = 2; name = "deferscale"; }
        else if (vname == "baseline") { id = 0; name = "baseline"; }
    };

    for (int i = 2; i < argc; i++) {
        if (std::string(argv[i]) == "--variant" && i + 1 < argc) {
            parse_variant(argv[++i], test_variant_id, test_variant_name);
        } else if (std::string(argv[i]) == "--baseline" && i + 1 < argc) {
            parse_variant(argv[++i], base_variant_id, base_variant_name);
        } else {
            csv_path = argv[i];
        }
    }

    std::cerr << "Loading model from " << model_path << "...\n";
    cactus_model_t model = cactus_init(model_path, nullptr, false);
    if (!model) {
        std::cerr << "Failed to initialize model\n";
        return 1;
    }
    std::cerr << "Model loaded.\n";
    std::cerr << "Testing: " << base_variant_name << " vs " << test_variant_name << "\n";

    std::string messages = R"([{"role":"user","content":")" + std::string(PROMPT) + R"("}])";
    std::string options = R"({"temperature":0.0,"max_tokens":)" + std::to_string(MAX_TOKENS) +
                          R"(,"confidence_threshold":0.0,"stop_sequences":["<|im_end|>","<end_of_turn>"]})";

    std::ofstream csv(csv_path);
    if (!csv.is_open()) {
        std::cerr << "Failed to open " << csv_path << " for writing\n";
        cactus_destroy(model);
        return 1;
    }
    csv << "variant,rep,token_index,time_ms\n";

    Variant base_v = {base_variant_id, base_variant_name};
    Variant test_v = {test_variant_id, test_variant_name};

    // Warmup
    std::cerr << "\n=== WARMUP (discarded) ===\n";
    run_pass(model, base_v, 1, messages, options, nullptr);

    // Interleaved: B, TT, BB, TT, BB, ...
    int base_rep = 0, test_rep = 0;

    std::cerr << "\n--- " << base_variant_name << " rep " << (base_rep + 1) << " ---\n";
    run_pass(model, base_v, 1, messages, options, &csv, base_rep);
    base_rep++;

    while (base_rep < NUM_REPS || test_rep < NUM_REPS) {
        for (int i = 0; i < 2 && test_rep < NUM_REPS; i++, test_rep++) {
            std::cerr << "\n--- " << test_variant_name << " rep " << (test_rep + 1) << " ---\n";
            run_pass(model, test_v, 1, messages, options, &csv, test_rep);
        }
        for (int i = 0; i < 2 && base_rep < NUM_REPS; i++, base_rep++) {
            std::cerr << "\n--- " << base_variant_name << " rep " << (base_rep + 1) << " ---\n";
            run_pass(model, base_v, 1, messages, options, &csv, base_rep);
        }
    }

    csv.close();
    cactus_destroy(model);
    std::cerr << "\nResults written to " << csv_path << "\n";
    return 0;
}
