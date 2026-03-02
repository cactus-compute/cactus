#include "test_utils.h"

#include "../cactus/engine/engine.h"
#include "../cactus/models/model.h"

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

namespace fs = std::filesystem;

using cactus::engine::Model;
using cactus::engine::Qwen3p5Model;
using cactus::engine::QwenModel;
using cactus::engine::create_model;

static std::string make_temp_model_dir(const std::string& suffix, const std::string& config_content) {
    const fs::path dir = fs::temp_directory_path() / ("cactus_model_factory_" + suffix);
    fs::remove_all(dir);
    fs::create_directories(dir);
    std::ofstream(dir / "config.txt", std::ios::binary) << config_content;
    return dir.string();
}

static bool test_qwen_default_routes_to_qwen_model() {
    const std::string config = R"(model_type=qwen
num_layers=2
hidden_dim=64
ffn_intermediate_dim=128
attention_heads=2
attention_kv_heads=2
attention_head_dim=32
vocab_size=100
context_length=128
precision=FP16
)";
    const std::string dir = make_temp_model_dir("qwen_default", config);
    std::unique_ptr<Model> model = create_model(dir);
    const bool ok = model && dynamic_cast<QwenModel*>(model.get()) != nullptr &&
                    dynamic_cast<Qwen3p5Model*>(model.get()) == nullptr;
    fs::remove_all(dir);
    return ok;
}

static bool test_qwen_layer_types_routes_to_qwen3p5() {
    const std::string config = R"(model_type=qwen
num_layers=2
hidden_dim=64
ffn_intermediate_dim=128
attention_heads=2
attention_kv_heads=2
attention_head_dim=32
vocab_size=100
context_length=128
precision=FP16
layer_types=attention,deltanet
)";
    const std::string dir = make_temp_model_dir("qwen_deltanet", config);
    std::unique_ptr<Model> model = create_model(dir);
    const bool ok = model && dynamic_cast<Qwen3p5Model*>(model.get()) != nullptr;
    fs::remove_all(dir);
    return ok;
}

static bool test_qwen_layer_types_bracketed_routes_to_qwen3p5() {
    const std::string config = R"(model_type=qwen
num_layers=2
hidden_dim=64
ffn_intermediate_dim=128
attention_heads=2
attention_kv_heads=2
attention_head_dim=32
vocab_size=100
context_length=128
precision=FP16
layer_types=[attention, GATED_DELTANET]
)";
    const std::string dir = make_temp_model_dir("qwen_deltanet_bracketed", config);
    std::unique_ptr<Model> model = create_model(dir);
    const bool ok = model && dynamic_cast<Qwen3p5Model*>(model.get()) != nullptr;
    fs::remove_all(dir);
    return ok;
}

int main() {
    TestUtils::TestRunner runner("Model Factory Routing Tests");
    runner.run_test("qwen_default_to_qwen", test_qwen_default_routes_to_qwen_model());
    runner.run_test("qwen_deltanet_to_qwen3p5", test_qwen_layer_types_routes_to_qwen3p5());
    runner.run_test("qwen_deltanet_bracketed_to_qwen3p5", test_qwen_layer_types_bracketed_routes_to_qwen3p5());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
