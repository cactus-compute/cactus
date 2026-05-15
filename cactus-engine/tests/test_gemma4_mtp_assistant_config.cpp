#include "test_utils.h"
#include "models/gemma4/gemma4_mtp_assistant.h"

#include <string>

using namespace TestUtils;
using namespace cactus::engine;

bool test_assistant_config_defaults_match_existing_e2b_layout() {
    Gemma4MtpAssistantConfig config = parse_gemma4_mtp_assistant_config("{}");

    return config.target_hidden_dim == 1536
        && config.centroid_count == 2048
        && config.top_centroid_count == 32
        && config.tokens_per_centroid == 128
        && config.layers.size() == 4
        && config.layers[0].head_dim == 256
        && config.layers[3].head_dim == 512;
}

bool test_assistant_config_reads_manifest_overrides() {
    const std::string manifest = R"({
        "target_hidden_dim": 2048,
        "centroid_count": 4096,
        "top_centroid_count": 16,
        "tokens_per_centroid": 64,
        "layers": [
            {"head_dim": 128, "num_heads": 8, "rot_dim": 128, "rope_freq": 10000.0, "window": 1024, "full_attention": false},
            {"head_dim": 256, "num_heads": 8, "rot_dim": 64, "rope_freq": 1000000.0, "window": 0, "full_attention": true}
        ]
    })";

    Gemma4MtpAssistantConfig config = parse_gemma4_mtp_assistant_config(manifest);

    return config.target_hidden_dim == 2048
        && config.centroid_count == 4096
        && config.top_centroid_count == 16
        && config.tokens_per_centroid == 64
        && config.layers.size() == 2
        && config.layers[0].window == 1024
        && !config.layers[0].full_attention
        && config.layers[1].head_dim == 256
        && config.layers[1].rot_dim == 64
        && config.layers[1].full_attention;
}

int main() {
    TestRunner runner("Gemma 4 MTP Assistant Config Tests");

    runner.run_test("defaults", test_assistant_config_defaults_match_existing_e2b_layout());
    runner.run_test("manifest_overrides", test_assistant_config_reads_manifest_overrides());

    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
