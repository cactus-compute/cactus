#include "test_utils.h"
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

int main() {
    TestUtils::TestRunner runner("Model Loading Failure Tests");
    runner.run_test("missing_directory", test_missing_directory());
    runner.run_test("missing_config", test_missing_config());
    runner.run_test("corrupt_weights", test_corrupt_weights());
    runner.run_test("empty_weight_file", test_empty_weight_file());
    runner.run_test("missing_vocab", test_missing_vocab());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
