#if __has_include("cactus/cactus.h")
#include "cactus/cactus.h"
#elif __has_include("cactus_engine.h")
#include "cactus_engine.h"
#else
#error "No Cactus public header found"
#endif

#include <chrono>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Args {
    std::string model_path;
    std::string model_name = "model";
    std::string prompt_name = "custom";
    std::string prompt = "Write a detailed explanation of how transformers work in deep learning.";
    int max_tokens = 256;
    int reps = 5;
    int warmup = 2;
    float temperature = 0.0f;
    float top_p = 0.95f;
    int top_k = 1;
    float min_p = 0.15f;
    int seed = 0;
    int mtp_max_draft_tokens = 2;
    bool mtp_fixed_draft = false;
};

struct Result {
    int prefill_tokens = 0;
    int decode_tokens = 0;
    double prefill_tps = 0.0;
    double decode_tps = 0.0;
    double ttft_ms = 0.0;
    double total_ms = 0.0;
    bool mtp_requested = false;
    bool mtp_enabled = false;
    int mtp_drafted_tokens = 0;
    int mtp_accepted_tokens = 0;
    int mtp_rejected_tokens = 0;
    int mtp_rounds = 0;
    double mtp_assistant_draft_ms = 0.0;
    double mtp_target_verify_ms = 0.0;
    double mtp_sampling_or_argmax_ms = 0.0;
    double mtp_kv_transaction_ms = 0.0;
    double mtp_callback_stream_ms = 0.0;
    std::string mtp_fallback_reason;
};

void usage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " --model-path <path> [options]\n"
              << "  --model-name <name>   Name written to CSV (default: model)\n"
              << "  --prompt-name <name>  Prompt label written to CSV (default: custom)\n"
              << "  --prompt <text>       Prompt to generate from\n"
              << "  --max-tokens <n>      Decode tokens per rep (default: 256)\n"
              << "  --reps <n>            Measured reps (default: 5)\n"
              << "  --warmup <n>          Warmup reps, not written to CSV (default: 2)\n"
              << "  --temperature <f>     Sampling temperature (default: 0)\n"
              << "  --top-p <f>           Top-p sampling threshold (default: 0.95)\n"
              << "  --top-k <n>           Top-k sampling threshold (default: 1)\n"
              << "  --min-p <f>           Min-p sampling threshold (default: 0.15)\n"
              << "  --seed <n>            Sampling seed; 0 uses random seed (default: 0)\n"
              << "  --mtp-max-draft <n>   Max draft tokens per MTP round; 0 disables MTP (default: 2)\n"
              << "  --mtp-fixed-draft     Keep MTP draft length fixed instead of adaptive ramp-up\n";
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("Missing value for ") + name);
            }
            return argv[++i];
        };

        if (arg == "--model-path") {
            args.model_path = need_value("--model-path");
        } else if (arg == "--model-name") {
            args.model_name = need_value("--model-name");
        } else if (arg == "--prompt-name") {
            args.prompt_name = need_value("--prompt-name");
        } else if (arg == "--prompt") {
            args.prompt = need_value("--prompt");
        } else if (arg == "--max-tokens") {
            args.max_tokens = std::stoi(need_value("--max-tokens"));
        } else if (arg == "--reps") {
            args.reps = std::stoi(need_value("--reps"));
        } else if (arg == "--warmup") {
            args.warmup = std::stoi(need_value("--warmup"));
        } else if (arg == "--temperature") {
            args.temperature = std::stof(need_value("--temperature"));
        } else if (arg == "--top-p") {
            args.top_p = std::stof(need_value("--top-p"));
        } else if (arg == "--top-k") {
            args.top_k = std::stoi(need_value("--top-k"));
        } else if (arg == "--min-p") {
            args.min_p = std::stof(need_value("--min-p"));
        } else if (arg == "--seed") {
            args.seed = std::stoi(need_value("--seed"));
        } else if (arg == "--mtp-max-draft") {
            args.mtp_max_draft_tokens = std::stoi(need_value("--mtp-max-draft"));
        } else if (arg == "--mtp-fixed-draft") {
            args.mtp_fixed_draft = true;
        } else if (arg == "--help") {
            usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (args.model_path.empty()) {
        throw std::runtime_error("Missing --model-path");
    }
    if (args.max_tokens <= 0 || args.reps <= 0 || args.warmup < 0) {
        throw std::runtime_error("--max-tokens and --reps must be positive, and --warmup must be non-negative");
    }
    if (args.mtp_max_draft_tokens < 0) {
        throw std::runtime_error("--mtp-max-draft must be non-negative");
    }
    return args;
}

std::string escape_json(const std::string& value) {
    std::ostringstream out;
    for (unsigned char c : value) {
        switch (c) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c);
                } else {
                    out << c;
                }
        }
    }
    return out.str();
}

std::string csv_escape(const std::string& value) {
    bool quote = value.find_first_of(",\"\n\r") != std::string::npos;
    if (!quote) return value;

    std::string out = "\"";
    for (char c : value) {
        if (c == '"') out += "\"\"";
        else out += c;
    }
    out += "\"";
    return out;
}

double json_number(const char* json, const char* key) {
    std::string needle = std::string("\"") + key + "\"";
    const char* pos = std::strstr(json, needle.c_str());
    if (!pos) return 0.0;
    pos += needle.size();
    while (*pos == ' ' || *pos == '\t' || *pos == ':' || *pos == '\n' || *pos == '\r') {
        ++pos;
    }
    return std::strtod(pos, nullptr);
}

bool json_bool(const char* json, const char* key) {
    std::string needle = std::string("\"") + key + "\"";
    const char* pos = std::strstr(json, needle.c_str());
    if (!pos) return false;
    pos += needle.size();
    while (*pos == ' ' || *pos == '\t' || *pos == ':' || *pos == '\n' || *pos == '\r') {
        ++pos;
    }
    return std::strncmp(pos, "true", 4) == 0;
}

std::string json_string(const char* json, const char* key) {
    std::string needle = std::string("\"") + key + "\"";
    const char* pos = std::strstr(json, needle.c_str());
    if (!pos) return {};
    pos += needle.size();
    while (*pos == ' ' || *pos == '\t' || *pos == ':' || *pos == '\n' || *pos == '\r') {
        ++pos;
    }
    if (*pos != '"') return {};
    ++pos;

    std::string result;
    while (*pos && *pos != '"') {
        if (*pos == '\\' && pos[1]) {
            ++pos;
            switch (*pos) {
                case '"': result.push_back('"'); break;
                case '\\': result.push_back('\\'); break;
                case 'n': result.push_back('\n'); break;
                case 'r': result.push_back('\r'); break;
                case 't': result.push_back('\t'); break;
                default: result.push_back(*pos); break;
            }
        } else {
            result.push_back(*pos);
        }
        ++pos;
    }
    return result;
}

std::string cactus_error(const char* fallback) {
    const char* err = cactus_get_last_error();
    if (err && err[0]) return err;
    return fallback;
}

std::string diagnostic_shape(const Args& args) {
    if (args.mtp_max_draft_tokens == 0) return "baseline";
    const char* tree_env = std::getenv("CACTUS_GEMMA4_MTP_TREE_MAIN_TOKENS");
    if (args.mtp_fixed_draft && tree_env && tree_env[0]) {
        char* end = nullptr;
        long main_tokens = std::strtol(tree_env, &end, 10);
        if (end != tree_env && *end == '\0' && main_tokens > 0
            && args.mtp_max_draft_tokens == main_tokens + 1) {
            return "alt_main" + std::to_string(main_tokens);
        }
    }
    return "base_main" + std::to_string(args.mtp_max_draft_tokens);
}

void configure_diagnostic_trace_env(const Args& args, int rep, bool active) {
    const char* trace_dir = std::getenv("CACTUS_GEMMA4_MTP_TRACE_CSV_DIR");
    if (active && trace_dir && trace_dir[0]) {
        std::string shape = diagnostic_shape(args);
        std::string rep_s = std::to_string(rep);
        std::string run_id = args.model_name + "_" + args.prompt_name + "_" + shape + "_rep" + rep_s;
        setenv("CACTUS_MTP_DIAG_RUN_ID", run_id.c_str(), 1);
        setenv("CACTUS_MTP_DIAG_MODEL", args.model_name.c_str(), 1);
        setenv("CACTUS_MTP_DIAG_PROMPT", args.prompt_name.c_str(), 1);
        setenv("CACTUS_MTP_DIAG_SHAPE", shape.c_str(), 1);
        setenv("CACTUS_MTP_DIAG_REP", rep_s.c_str(), 1);
    } else {
        unsetenv("CACTUS_MTP_DIAG_RUN_ID");
        unsetenv("CACTUS_MTP_DIAG_MODEL");
        unsetenv("CACTUS_MTP_DIAG_PROMPT");
        unsetenv("CACTUS_MTP_DIAG_SHAPE");
        unsetenv("CACTUS_MTP_DIAG_REP");
    }
}

Result run_once(cactus_model_t model, const Args& args, int rep, bool trace_active) {
    configure_diagnostic_trace_env(args, rep, trace_active);
    cactus_reset(model);

    std::string messages = "[{\"role\":\"user\",\"content\":\"" + escape_json(args.prompt) + "\"}]";
    std::string options = "{\"temperature\":" + std::to_string(args.temperature)
        + ",\"top_p\":" + std::to_string(args.top_p)
        + ",\"top_k\":" + std::to_string(args.top_k)
        + ",\"min_p\":" + std::to_string(args.min_p)
        + ",\"max_tokens\":"
        + std::to_string(args.max_tokens)
        + ",\"auto_handoff\":false,\"confidence_threshold\":0.0";
    if (args.seed > 0) {
        options += ",\"seed\":" + std::to_string(args.seed);
    }
    if (args.mtp_max_draft_tokens > 0) {
        options += ",\"mtp\":true";
        options += ",\"mtp_max_draft_tokens\":" + std::to_string(args.mtp_max_draft_tokens);
        if (args.mtp_fixed_draft) {
            options += ",\"mtp_fixed_draft\":true";
        }
    }
    options += ",\"stop_sequences\":[\"<|im_end|>\",\"<end_of_turn>\"]}";

    size_t response_size = static_cast<size_t>(args.max_tokens) * 512 + 65536;
    std::vector<char> response(response_size, 0);

    auto start = std::chrono::steady_clock::now();
    int rc = cactus_complete(model, messages.c_str(), response.data(), response.size(),
                             options.c_str(), nullptr, nullptr, nullptr, nullptr, 0);
    auto end = std::chrono::steady_clock::now();

    if (rc < 0) {
        throw std::runtime_error(response.data()[0] ? response.data() : cactus_error("cactus_complete failed"));
    }

    Result result;
    result.prefill_tokens = static_cast<int>(json_number(response.data(), "prefill_tokens"));
    result.decode_tokens = static_cast<int>(json_number(response.data(), "decode_tokens"));
    result.prefill_tps = json_number(response.data(), "prefill_tps");
    result.decode_tps = json_number(response.data(), "decode_tps");
    result.ttft_ms = json_number(response.data(), "time_to_first_token_ms");
    result.total_ms = json_number(response.data(), "total_time_ms");
    result.mtp_requested = json_bool(response.data(), "requested");
    result.mtp_enabled = json_bool(response.data(), "enabled");
    result.mtp_drafted_tokens = static_cast<int>(json_number(response.data(), "drafted_tokens"));
    result.mtp_accepted_tokens = static_cast<int>(json_number(response.data(), "accepted_tokens"));
    result.mtp_rejected_tokens = static_cast<int>(json_number(response.data(), "rejected_tokens"));
    result.mtp_rounds = static_cast<int>(json_number(response.data(), "rounds"));
    result.mtp_assistant_draft_ms = json_number(response.data(), "assistant_draft_ms");
    result.mtp_target_verify_ms = json_number(response.data(), "target_verify_ms");
    result.mtp_sampling_or_argmax_ms = json_number(response.data(), "sampling_or_argmax_ms");
    result.mtp_kv_transaction_ms = json_number(response.data(), "kv_transaction_ms");
    result.mtp_callback_stream_ms = json_number(response.data(), "callback_stream_ms");
    result.mtp_fallback_reason = json_string(response.data(), "fallback_reason");
    if (result.total_ms <= 0.0) {
        result.total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    }
    return result;
}

void write_csv_row(const Args& args, int rep, const Result& result) {
    std::cout << "cactus,"
              << csv_escape(args.model_name) << ","
              << csv_escape(args.prompt_name) << ","
              << diagnostic_shape(args) << ","
              << args.mtp_max_draft_tokens << ","
              << rep << ","
              << result.prefill_tokens << ","
              << result.decode_tokens << ","
              << std::fixed << std::setprecision(2)
              << result.prefill_tps << ","
              << result.decode_tps << ","
              << result.ttft_ms << ","
              << result.total_ms << ","
              << (result.mtp_requested ? "true" : "false") << ","
              << (result.mtp_enabled ? "true" : "false") << ","
              << result.mtp_drafted_tokens << ","
              << result.mtp_accepted_tokens << ","
              << result.mtp_rejected_tokens << ","
              << result.mtp_rounds << ","
              << csv_escape(result.mtp_fallback_reason) << ","
              << result.mtp_assistant_draft_ms << ","
              << result.mtp_target_verify_ms << ","
              << result.mtp_sampling_or_argmax_ms << ","
              << result.mtp_kv_transaction_ms << ","
              << result.mtp_callback_stream_ms << "\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);

        std::cerr << "Loading " << args.model_name << " from " << args.model_path << "\n";
        cactus_model_t model = cactus_init(args.model_path.c_str(), nullptr, false);
        if (!model) {
            throw std::runtime_error(cactus_error("cactus_init failed"));
        }

        for (int i = 0; i < args.warmup; ++i) {
            Result warmup = run_once(model, args, i, false);
            std::cerr << "  warmup " << i << ": decode=" << std::fixed << std::setprecision(1)
                      << warmup.decode_tps << " tok/s\n";
        }

        for (int rep = 0; rep < args.reps; ++rep) {
            Result result = run_once(model, args, rep, true);
            std::cerr << "  rep " << rep << ": decode=" << std::fixed << std::setprecision(1)
                      << result.decode_tps << " tok/s, prefill=" << result.prefill_tps
                      << " tok/s, total=" << result.total_ms << " ms\n";
            write_csv_row(args, rep, result);
        }

        cactus_destroy(model);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
