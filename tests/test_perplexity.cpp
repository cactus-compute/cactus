#include "test_utils.h"
#include "../cactus/ffi/cactus_utils.h"
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <getopt.h>

using namespace cactus::engine;

struct PerplexityConfig {
    std::string model_path;
    std::string corpus_path;
    size_t context_size = 512;
    size_t chunk_size = 256;
    size_t max_tokens = 0;
    std::string kv_precision = "INT8";
    size_t kv_group_size = 32;
    std::string kv_split = "SAME";
    bool cached_mode = true;
    size_t position_buckets = 8;
};

static void print_usage(const char* program) {
    std::cout << "Usage: " << program << " [options]\n"
              << "\n"
              << "Perplexity evaluation for INT4 KV cache accuracy testing.\n"
              << "\n"
              << "Options:\n"
              << "  --model PATH        Path to model directory (required)\n"
              << "  --corpus PATH       Path to text corpus file (required)\n"
              << "  --context SIZE      Context window size (default: 512)\n"
              << "  --chunk_size SIZE   Tokens per forward pass in cached mode (default: 256)\n"
              << "  --max_tokens N      Limit corpus to first N tokens (default: all)\n"
              << "  --kv_precision P    KV cache precision: FP16, INT8, INT4, K8V4 (default: INT8)\n"
              << "  --kv_group_size G   Quantization group size: 16, 32, 64 (default: 32)\n"
              << "  --kv_split S        K/V split: SAME, K8V4 (default: SAME)\n"
              << "  --uncached          Use non-cached scoring (faster, no KV cache effect)\n"
              << "  --buckets N         Number of position buckets for per-position stats (default: 8)\n"
              << "  --help              Show this help\n"
              << "\n"
              << "Environment variables:\n"
              << "  CACTUS_KV_PRECISION      Override KV cache precision (FP16, INT8)\n"
              << "  CACTUS_KV_WINDOW_SIZE    Override KV cache window size\n"
              << "\n"
              << "Examples:\n"
              << "  " << program << " --model weights/Qwen3-0.6B --corpus wikitext-2-raw/wiki.test.raw\n"
              << "  " << program << " --model weights/Qwen3-0.6B --corpus corpus.txt --kv_precision FP16\n"
              << "  " << program << " --model weights/Qwen3-0.6B --corpus corpus.txt --max_tokens 10000\n";
}

static PerplexityConfig parse_args(int argc, char** argv) {
    PerplexityConfig config;

    static struct option long_options[] = {
        {"model",         required_argument, nullptr, 'm'},
        {"corpus",        required_argument, nullptr, 'c'},
        {"context",       required_argument, nullptr, 'x'},
        {"chunk_size",    required_argument, nullptr, 's'},
        {"max_tokens",    required_argument, nullptr, 'n'},
        {"kv_precision",  required_argument, nullptr, 'p'},
        {"kv_group_size", required_argument, nullptr, 'g'},
        {"kv_split",      required_argument, nullptr, 'k'},
        {"uncached",      no_argument,       nullptr, 'u'},
        {"buckets",       required_argument, nullptr, 'b'},
        {"help",          no_argument,       nullptr, 'h'},
        {nullptr, 0, nullptr, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "m:c:x:s:n:p:g:k:ub:h", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'm': config.model_path = optarg; break;
            case 'c': config.corpus_path = optarg; break;
            case 'x': config.context_size = std::stoul(optarg); break;
            case 's': config.chunk_size = std::stoul(optarg); break;
            case 'n': config.max_tokens = std::stoul(optarg); break;
            case 'p': config.kv_precision = optarg; break;
            case 'g': config.kv_group_size = std::stoul(optarg); break;
            case 'k': config.kv_split = optarg; break;
            case 'u': config.cached_mode = false; break;
            case 'b': config.position_buckets = std::stoul(optarg); break;
            case 'h': print_usage(argv[0]); exit(0);
            default:  print_usage(argv[0]); exit(1);
        }
    }

    if (config.model_path.empty()) {
        const char* env = std::getenv("CACTUS_TEST_MODEL");
        if (env) config.model_path = env;
    }

    if (config.model_path.empty() || config.corpus_path.empty()) {
        std::cerr << "Error: --model and --corpus are required\n\n";
        print_usage(argv[0]);
        exit(1);
    }

    return config;
}

static std::vector<uint32_t> tokenize_corpus(cactus_model_t model, const std::string& path, size_t max_tokens) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Error: cannot open corpus file: " << path << "\n";
        exit(1);
    }

    std::vector<uint32_t> all_tokens;
    static constexpr size_t CHUNK_CHARS = 8192;
    std::string chunk;
    chunk.reserve(CHUNK_CHARS + 256);

    while (file && (max_tokens == 0 || all_tokens.size() < max_tokens)) {
        chunk.clear();
        std::string line;
        while (chunk.size() < CHUNK_CHARS && std::getline(file, line)) {
            if (!chunk.empty()) chunk += '\n';
            chunk += line;
        }
        if (chunk.empty()) break;

        std::vector<uint32_t> buffer(chunk.size() + 256);
        size_t out_len = 0;
        int result = cactus_tokenize(model, chunk.c_str(), buffer.data(), buffer.size(), &out_len);

        if (result < 0 || out_len == 0) continue;

        all_tokens.insert(all_tokens.end(), buffer.begin(), buffer.begin() + out_len);

        if (max_tokens > 0 && all_tokens.size() % 10000 < CHUNK_CHARS) {
            std::cout << "\r  Tokenized: " << all_tokens.size() << " tokens" << std::flush;
        }
    }
    std::cout << "\r  Tokenized: " << all_tokens.size() << " tokens\n";

    if (max_tokens > 0 && all_tokens.size() > max_tokens)
        all_tokens.resize(max_tokens);

    return all_tokens;
}

struct PerplexityResult {
    double mean_perplexity;
    double bits_per_byte;
    double total_nll;
    size_t tokens_scored;
    std::vector<double> bucket_perplexities;
    std::vector<size_t> bucket_counts;
    double elapsed_seconds;
};

static PerplexityResult eval_cached(Model& model, const std::vector<uint32_t>& tokens,
                                     const PerplexityConfig& config) {
    PerplexityResult result = {};
    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<double> per_position_logprobs;
    size_t scored = 0;

    double total_logprob = model.score_tokens_cached_logprob(
        tokens, config.chunk_size, &scored, &per_position_logprobs);

    auto end_time = std::chrono::high_resolution_clock::now();
    result.elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();
    result.tokens_scored = scored;
    result.total_nll = -total_logprob;

    if (scored > 0) {
        result.mean_perplexity = std::exp(-total_logprob / scored);
        result.bits_per_byte = (-total_logprob / scored) / std::log(2.0);
    }

    size_t bucket_size = (scored + config.position_buckets - 1) / config.position_buckets;
    if (bucket_size == 0) bucket_size = 1;

    result.bucket_perplexities.resize(config.position_buckets, 0.0);
    result.bucket_counts.resize(config.position_buckets, 0);

    size_t window_pos = 0;
    size_t window_size = config.context_size;

    for (size_t i = 0; i < per_position_logprobs.size(); i++) {
        size_t bucket = std::min(window_pos * config.position_buckets / window_size,
                                  config.position_buckets - 1);
        result.bucket_perplexities[bucket] += per_position_logprobs[i];
        result.bucket_counts[bucket]++;
        window_pos++;
        if (window_pos >= window_size)
            window_pos = 0;
    }

    for (size_t b = 0; b < config.position_buckets; b++) {
        if (result.bucket_counts[b] > 0) {
            double avg_logprob = result.bucket_perplexities[b] / result.bucket_counts[b];
            result.bucket_perplexities[b] = std::exp(-avg_logprob);
        }
    }

    return result;
}

static PerplexityResult eval_uncached(cactus_model_t model_handle,
                                       const std::vector<uint32_t>& tokens,
                                       const PerplexityConfig& config) {
    PerplexityResult result = {};
    auto start_time = std::chrono::high_resolution_clock::now();

    double total_logprob = 0.0;
    size_t total_scored = 0;

    size_t stride = config.context_size;

    for (size_t offset = 0; offset < tokens.size(); offset += stride) {
        size_t window_end = std::min(offset + stride, tokens.size());

        if (window_end - offset < 2)
            break;

        size_t context = std::min(offset, config.context_size);

        char response[512];
        int ret = cactus_score_window(
            model_handle,
            tokens.data(),
            tokens.size(),
            offset == 0 ? 1 : offset,
            window_end,
            context,
            response,
            sizeof(response)
        );

        if (ret < 0) {
            std::cerr << "Warning: score_window failed at offset " << offset << "\n";
            continue;
        }

        std::string resp(response);
        double logprob = EngineTestUtils::json_number(resp, "logprob");
        size_t scored = static_cast<size_t>(EngineTestUtils::json_number(resp, "tokens"));

        total_logprob += logprob;
        total_scored += scored;

        double progress = 100.0 * window_end / tokens.size();
        std::cout << "\r  Processing: " << std::fixed << std::setprecision(1) << progress << "%"
                  << " (" << total_scored << " tokens scored)" << std::flush;
    }
    std::cout << "\n";

    auto end_time = std::chrono::high_resolution_clock::now();
    result.elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();
    result.tokens_scored = total_scored;
    result.total_nll = -total_logprob;

    if (total_scored > 0) {
        result.mean_perplexity = std::exp(-total_logprob / total_scored);
        result.bits_per_byte = (-total_logprob / total_scored) / std::log(2.0);
    }

    return result;
}

static void print_results(const PerplexityResult& result, const PerplexityConfig& config) {
    std::cout << "\n=== Results ===\n"
              << "  Tokens scored:     " << result.tokens_scored << "\n"
              << "  Total NLL:         " << std::fixed << std::setprecision(4) << result.total_nll << "\n"
              << "  Mean perplexity:   " << std::fixed << std::setprecision(4) << result.mean_perplexity << "\n"
              << "  Bits per token:    " << std::fixed << std::setprecision(4) << result.bits_per_byte << "\n"
              << "  Elapsed:           " << std::fixed << std::setprecision(1) << result.elapsed_seconds << "s\n"
              << "  Throughput:        " << std::fixed << std::setprecision(1)
              << (result.tokens_scored / result.elapsed_seconds) << " tokens/s\n";

    if (!result.bucket_perplexities.empty() && config.cached_mode) {
        size_t window_size = config.context_size;
        size_t bucket_width = window_size / config.position_buckets;
        std::cout << "\n  Per-position perplexity (within " << window_size << "-token windows):\n";
        for (size_t b = 0; b < result.bucket_perplexities.size(); b++) {
            size_t lo = b * bucket_width;
            size_t hi = (b + 1) * bucket_width - 1;
            if (b == result.bucket_perplexities.size() - 1)
                hi = window_size - 1;

            std::cout << "    Positions " << std::setw(4) << lo << "-" << std::setw(4) << hi
                      << ":  ";
            if (result.bucket_counts[b] > 0) {
                std::cout << std::fixed << std::setprecision(4) << result.bucket_perplexities[b]
                          << "  (n=" << result.bucket_counts[b] << ")";
            } else {
                std::cout << "N/A";
            }
            std::cout << "\n";
        }
    }
}

int main(int argc, char** argv) {
    PerplexityConfig config = parse_args(argc, argv);

    if (config.kv_split != "SAME" && config.kv_split != "K8V4") {
        std::cerr << "Error: KV split '" << config.kv_split
                  << "' is not yet implemented. Supported: SAME, K8V4\n";
        return 1;
    }

    if (config.kv_precision == "INT4" || config.kv_precision == "INT4_NF4_ASYM") {
        setenv("CACTUS_KV_KEY_PRECISION", "INT4_NF4", 1);
        setenv("CACTUS_KV_VALUE_PRECISION", "INT4_ASYMMETRIC", 1);
        setenv("CACTUS_KV_PRECISION", "INT8", 1);
    } else if (config.kv_precision == "K8V4" || config.kv_split == "K8V4") {
        setenv("CACTUS_KV_KEY_PRECISION", "INT8", 1);
        setenv("CACTUS_KV_VALUE_PRECISION", "INT4_ASYMMETRIC", 1);
        setenv("CACTUS_KV_PRECISION", "INT8", 1);
    } else {
        setenv("CACTUS_KV_PRECISION", config.kv_precision.c_str(), 1);
    }
    setenv("CACTUS_KV_GROUP_SIZE", std::to_string(config.kv_group_size).c_str(), 1);
    setenv("CACTUS_KV_WINDOW_SIZE", std::to_string(config.context_size).c_str(), 1);

    std::cout << "=== Perplexity Evaluation ===\n"
              << "  Model:         " << config.model_path << "\n"
              << "  Corpus:        " << config.corpus_path << "\n"
              << "  Context:       " << config.context_size << "\n"
              << "  Chunk size:    " << config.chunk_size << "\n"
              << "  KV precision:  " << config.kv_precision << "\n"
              << "  KV group size: " << config.kv_group_size << "\n"
              << "  KV split:      " << config.kv_split << "\n"
              << "  Mode:          " << (config.cached_mode ? "cached" : "uncached") << "\n";

    std::cout << "\nLoading model...\n";
    cactus_model_t model_handle = cactus_init(config.model_path.c_str(), nullptr, false);
    if (!model_handle) {
        std::cerr << "Error: failed to load model from " << config.model_path << "\n";
        const char* err = cactus_get_last_error();
        if (err) std::cerr << "  " << err << "\n";
        return 1;
    }

    std::cout << "Tokenizing corpus...\n";
    std::vector<uint32_t> tokens = tokenize_corpus(model_handle, config.corpus_path, config.max_tokens);

    if (tokens.size() < 2) {
        std::cerr << "Error: corpus too short (need at least 2 tokens)\n";
        cactus_destroy(model_handle);
        return 1;
    }

    std::cout << "\nEvaluating";
    if (config.cached_mode) {
        std::cout << " (cached, chunk_size=" << config.chunk_size << ")...\n";
    } else {
        std::cout << " (uncached, stride=" << config.context_size << ")...\n";
    }

    PerplexityResult result;

    if (config.cached_mode) {
        auto* handle = static_cast<CactusModelHandle*>(model_handle);
        result = eval_cached(*handle->model, tokens, config);
    } else {
        result = eval_uncached(model_handle, tokens, config);
    }

    print_results(result, config);

    std::cout << "\n=== CSV Output ===\n"
              << "model,corpus,kv_precision,kv_group_size,kv_split,mode,context,chunk_size,"
              << "tokens_scored,mean_ppl,bits_per_token,nll,elapsed_s\n"
              << config.model_path << ","
              << config.corpus_path << ","
              << config.kv_precision << ","
              << config.kv_group_size << ","
              << config.kv_split << ","
              << (config.cached_mode ? "cached" : "uncached") << ","
              << config.context_size << ","
              << config.chunk_size << ","
              << result.tokens_scored << ","
              << std::fixed << std::setprecision(4) << result.mean_perplexity << ","
              << std::fixed << std::setprecision(4) << result.bits_per_byte << ","
              << std::fixed << std::setprecision(4) << result.total_nll << ","
              << std::fixed << std::setprecision(1) << result.elapsed_seconds << "\n";

    cactus_destroy(model_handle);
    return 0;
}
