#include "test_utils.h"
#include "cactus_graph.h"
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;

static std::string make_temp_dir(const std::string& suffix) {
    std::string dir = fs::temp_directory_path().string() + "/cactus_test_" + suffix;
    fs::create_directories(dir);
    return dir;
}

static void write_file(const std::string& path, const std::string& content) {
    std::ofstream(path, std::ios::binary) << content;
}

static bool expect_init_fails(const std::string& path) {
    cactus_model_t model = cactus_init(path.c_str(), nullptr, false);
    if (model) { cactus_destroy(model); return false; }
    return true;
}

static const char* MINIMAL_CONFIG = R"({"model_type":"qwen","model_variant":"default","precision":"INT8","num_layers":2,"hidden_dim":64,"ffn_intermediate_dim":128,"attention_heads":2,"attention_kv_heads":2,"attention_head_dim":32,"vocab_size":100,"context_length":512})";

static bool test_missing_directory() {
    return expect_init_fails("/nonexistent/path/to/model");
}

static bool test_missing_config() {
    std::string dir = make_temp_dir("missing_config");
    write_file(dir + "/dummy.bin", "placeholder");
    bool ok = expect_init_fails(dir);
    fs::remove_all(dir);
    return ok;
}

static bool test_corrupt_weights() {
    std::string dir = make_temp_dir("corrupt_weights");
    write_file(dir + "/config.txt", MINIMAL_CONFIG);
    write_file(dir + "/vocab.txt", "hello\nworld\n");
    write_file(dir + "/weights.bin", std::string("\xDE\xAD\xBE\xEF", 4) + std::string(124, '\xDE'));
    bool ok = expect_init_fails(dir);
    fs::remove_all(dir);
    return ok;
}

static bool test_empty_weight_file() {
    std::string dir = make_temp_dir("empty_weights");
    write_file(dir + "/config.txt", MINIMAL_CONFIG);
    write_file(dir + "/vocab.txt", "hello\nworld\n");
    write_file(dir + "/weights.bin", "");
    bool ok = expect_init_fails(dir);
    fs::remove_all(dir);
    return ok;
}

static bool test_missing_vocab() {
    std::string dir = make_temp_dir("missing_vocab");
    write_file(dir + "/config.txt", MINIMAL_CONFIG);
    bool ok = expect_init_fails(dir);
    fs::remove_all(dir);
    return ok;
}

static bool test_incomplete_transpiled_bundle_fails_explicitly() {
    std::string dir = make_temp_dir("incomplete_transpiled");
    fs::create_directories(dir + "/components");
    write_file(dir + "/config.txt",
        "model_type=qwen\n"
        "model_variant=default\n"
        "precision=FP32\n"
        "num_layers=1\n"
        "hidden_dim=8\n"
        "ffn_intermediate_dim=16\n"
        "attention_heads=1\n"
        "attention_kv_heads=1\n"
        "attention_head_dim=8\n"
        "vocab_size=3\n"
        "bos_token_id=0\n"
        "eos_token_id=2\n");
    write_file(dir + "/vocab.txt", "0\t<pad>\n1\thello\n2\tworld\n");
    write_file(dir + "/tokenizer_config.txt", R"({"tokenizer_type":"bpe","vocab_format":"id_tab_token"})");
    write_file(dir + "/components/manifest.json", R"({
        "task":"causal_lm_logits",
        "family":"test",
        "component_order":["decoder"],
        "components":[{"component":"decoder","logical_inputs":["input_ids"],"logical_outputs":["logits"]}]
    })");

    cactus_model_t model = cactus_init(dir.c_str(), nullptr, false);
    if (model) {
        cactus_destroy(model);
        fs::remove_all(dir);
        return false;
    }
    std::string error = cactus_get_last_error();
    fs::remove_all(dir);
    return error.find("transpiled causal LM bundle") != std::string::npos &&
           error.find("decoder graph") != std::string::npos;
}

static bool test_transpiled_causal_lm_completion_works() {
    std::string dir = make_temp_dir("transpiled_completion");
    fs::create_directories(dir + "/components/decoder");
    write_file(dir + "/config.txt",
        "model_type=qwen\n"
        "model_variant=default\n"
        "precision=FP32\n"
        "num_layers=1\n"
        "hidden_dim=8\n"
        "ffn_intermediate_dim=16\n"
        "attention_heads=1\n"
        "attention_kv_heads=1\n"
        "attention_head_dim=8\n"
        "vocab_size=1\n"
        "bos_token_id=0\n"
        "eos_token_id=999\n");
    write_file(dir + "/vocab.txt", "0\tA\n");
    write_file(dir + "/merges.txt", "");
    write_file(dir + "/tokenizer_config.txt", R"({"tokenizer_type":"bpe","vocab_format":"id_tab_token"})");

    CactusGraph graph;
    size_t input = graph.input({1, 512}, Precision::FP32);
    size_t logits = graph.reshape(input, {1, 512, 1});
    graph.save(dir + "/components/decoder/graph.cactus");

    write_file(dir + "/components/manifest.json",
        "{"
        "\"task\":\"causal_lm_logits\","
        "\"family\":\"test\","
        "\"component_order\":[\"decoder\"],"
        "\"components\":[{"
        "\"component\":\"decoder\","
        "\"graph\":\"components/decoder/graph.cactus\","
        "\"logical_inputs\":[\"input_ids\"],"
        "\"logical_outputs\":[\"logits\"],"
        "\"runtime_input_node_ids\":[" + std::to_string(input) + "],"
        "\"output_node_ids\":[" + std::to_string(logits) + "]"
        "}]"
        "}");

    cactus_model_t model = cactus_init(dir.c_str(), nullptr, false);
    if (!model) {
        fs::remove_all(dir);
        return false;
    }

    char response[1024];
    const char* messages = R"([{"role":"user","content":"A"}])";
    const char* options = R"({"max_tokens":1,"temperature":0.0,"top_k":1,"auto_handoff":false,"confidence_threshold":-1.0})";
    int result = cactus_complete(model, messages, response, sizeof(response), options, nullptr, nullptr, nullptr, nullptr, 0);
    cactus_destroy(model);
    fs::remove_all(dir);
    if (result <= 0) {
        return false;
    }
    EngineTestUtils::Metrics metrics;
    metrics.parse(response);
    return metrics.success && metrics.completion_tokens == 1.0;
}

static bool test_transpiled_mtp_unavailable_fails_explicitly() {
    std::string dir = make_temp_dir("transpiled_mtp_unavailable");
    fs::create_directories(dir + "/components/decoder");
    write_file(dir + "/config.txt",
        "model_type=qwen\nmodel_variant=default\nprecision=FP32\nnum_layers=1\nhidden_dim=8\n"
        "ffn_intermediate_dim=16\nattention_heads=1\nattention_kv_heads=1\nattention_head_dim=8\n"
        "vocab_size=1\nbos_token_id=0\neos_token_id=999\n");
    write_file(dir + "/vocab.txt", "0\tA\n");
    write_file(dir + "/merges.txt", "");
    write_file(dir + "/tokenizer_config.txt", R"({"tokenizer_type":"bpe","vocab_format":"id_tab_token"})");

    CactusGraph graph;
    size_t input = graph.input({1, 512}, Precision::FP32);
    size_t logits = graph.reshape(input, {1, 512, 1});
    graph.save(dir + "/components/decoder/graph.cactus");
    write_file(dir + "/components/manifest.json",
        "{"
        "\"task\":\"causal_lm_logits\","
        "\"family\":\"test\","
        "\"component_order\":[\"decoder\"],"
        "\"components\":[{\"component\":\"decoder\",\"graph\":\"components/decoder/graph.cactus\","
        "\"logical_inputs\":[\"input_ids\"],\"logical_outputs\":[\"logits\"],"
        "\"runtime_input_node_ids\":[" + std::to_string(input) + "],"
        "\"output_node_ids\":[" + std::to_string(logits) + "]}]"
        "}");

    cactus_model_t model = cactus_init(dir.c_str(), nullptr, false);
    if (!model) {
        fs::remove_all(dir);
        return false;
    }
    char response[1024];
    const char* messages = R"([{"role":"user","content":"A"}])";
    const char* options = R"({"max_tokens":1,"mtp_enabled":true,"auto_handoff":false,"confidence_threshold":-1.0})";
    int result = cactus_complete(model, messages, response, sizeof(response), options, nullptr, nullptr, nullptr, nullptr, 0);
    cactus_destroy(model);
    fs::remove_all(dir);
    return result < 0 && std::string(response).find("MTP requested but unavailable: unsupported_target") != std::string::npos;
}

static bool test_transpiled_mtp_completion_works() {
    std::string dir = make_temp_dir("transpiled_mtp_completion");
    fs::create_directories(dir + "/components/decoder");
    fs::create_directories(dir + "/components/assistant");
    write_file(dir + "/config.txt",
        "model_type=qwen\nmodel_variant=default\nprecision=FP32\nnum_layers=1\nhidden_dim=8\n"
        "ffn_intermediate_dim=16\nattention_heads=1\nattention_kv_heads=1\nattention_head_dim=8\n"
        "vocab_size=1\nbos_token_id=0\neos_token_id=999\n");
    write_file(dir + "/vocab.txt", "0\tA\n");
    write_file(dir + "/merges.txt", "");
    write_file(dir + "/tokenizer_config.txt", R"({"tokenizer_type":"bpe","vocab_format":"id_tab_token"})");

    CactusGraph target_graph;
    size_t target_input = target_graph.input({1, 512}, Precision::FP32);
    size_t target_logits = target_graph.reshape(target_input, {1, 512, 1});
    target_graph.save(dir + "/components/decoder/graph.cactus");

    CactusGraph assistant_graph;
    size_t assistant_input = assistant_graph.input({1, 512}, Precision::FP32);
    size_t assistant_logits = assistant_graph.reshape(assistant_input, {1, 512, 1});
    assistant_graph.save(dir + "/components/assistant/graph.cactus");

    write_file(dir + "/components/manifest.json",
        "{"
        "\"task\":\"causal_lm_logits\","
        "\"family\":\"test\","
        "\"component_order\":[\"decoder\",\"assistant\"],"
        "\"spec_decode\":{"
        "\"version\":1,"
        "\"method\":\"assistant_chain\","
        "\"target\":{\"verifier_logits\":\"decoder:logits\",\"target_hidden_state\":\"decoder:hidden\","
        "\"assistant_shared_state_tensors\":[\"decoder:shared\"]},"
        "\"assistant\":{\"current_token\":\"assistant:current_token\","
        "\"previous_target_hidden\":\"assistant:previous_target_hidden\","
        "\"target_shared_state_inputs\":[\"assistant:shared\"],"
        "\"position\":\"assistant:position\","
        "\"logits_output\":\"assistant:logits\","
        "\"next_hidden_output\":\"assistant:next_hidden\"}},"
        "\"components\":["
        "{\"component\":\"decoder\",\"graph\":\"components/decoder/graph.cactus\","
        "\"logical_inputs\":[\"input_ids\"],\"logical_outputs\":[\"logits\"],"
        "\"runtime_input_node_ids\":[" + std::to_string(target_input) + "],"
        "\"output_node_ids\":[" + std::to_string(target_logits) + "]},"
        "{\"component\":\"assistant\",\"graph\":\"components/assistant/graph.cactus\","
        "\"logical_inputs\":[\"input_ids\"],\"logical_outputs\":[\"logits\"],"
        "\"runtime_input_node_ids\":[" + std::to_string(assistant_input) + "],"
        "\"output_node_ids\":[" + std::to_string(assistant_logits) + "]}"
        "]"
        "}");

    cactus_model_t model = cactus_init(dir.c_str(), nullptr, false);
    if (!model) {
        fs::remove_all(dir);
        return false;
    }
    char response[1024];
    const char* messages = R"([{"role":"user","content":"A"}])";
    const char* options = R"({"max_tokens":2,"mtp_enabled":true,"mtp_draft_tokens":1,"temperature":0.0,"auto_handoff":false,"confidence_threshold":-1.0})";
    int result = cactus_complete(model, messages, response, sizeof(response), options, nullptr, nullptr, nullptr, nullptr, 0);
    cactus_destroy(model);
    fs::remove_all(dir);
    if (result <= 0) {
        return false;
    }
    EngineTestUtils::Metrics metrics;
    metrics.parse(response);
    return metrics.success && metrics.completion_tokens == 2.0;
}

int main() {
    TestUtils::TestRunner runner("Model Loading Failure Tests");
    runner.run_test("missing_directory", test_missing_directory());
    runner.run_test("missing_config", test_missing_config());
    runner.run_test("corrupt_weights", test_corrupt_weights());
    runner.run_test("empty_weight_file", test_empty_weight_file());
    runner.run_test("missing_vocab", test_missing_vocab());
    runner.run_test("incomplete_transpiled_bundle_fails_explicitly", test_incomplete_transpiled_bundle_fails_explicitly());
    runner.run_test("transpiled_causal_lm_completion_works", test_transpiled_causal_lm_completion_works());
    runner.run_test("transpiled_mtp_unavailable_fails_explicitly", test_transpiled_mtp_unavailable_fails_explicitly());
    runner.run_test("transpiled_mtp_completion_works", test_transpiled_mtp_completion_works());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
