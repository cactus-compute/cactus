#include "test_utils.h"
#include "src/spec_decode.h"

#include <stdexcept>

static cactus::engine::SpecDecodeManifest valid_manifest() {
    cactus::engine::SpecDecodeManifest manifest;
    manifest.version = 1;
    manifest.method = "single_position_mtp";
    manifest.target.verifier_logits = "decoder:logits";
    manifest.target.target_hidden_state = "decoder:hidden";
    manifest.target.target_token_embedding = "embedding:output";
    manifest.target.shared_kv_full_key = "decoder:full_key";
    manifest.target.shared_kv_full_value = "decoder:full_value";
    manifest.target.shared_kv_sliding_key = "decoder:sliding_key";
    manifest.target.shared_kv_sliding_value = "decoder:sliding_value";
    manifest.assistant.current_token_embedding = "assistant:current_token_embedding";
    manifest.assistant.previous_target_hidden = "assistant:previous_target_hidden";
    manifest.assistant.position = "assistant:position";
    manifest.assistant.shared_kv_full_key = "assistant:full_key";
    manifest.assistant.shared_kv_full_value = "assistant:full_value";
    manifest.assistant.shared_kv_sliding_key = "assistant:sliding_key";
    manifest.assistant.shared_kv_sliding_value = "assistant:sliding_value";
    manifest.assistant.logits_output = "assistant:logits";
    manifest.assistant.next_hidden_output = "assistant:next_hidden";
    return manifest;
}

static bool test_valid_spec_decode_manifest_loads() {
    auto manifest = valid_manifest();
    try {
        cactus::engine::validate_spec_decode_manifest(manifest);
    } catch (...) {
        return false;
    }
    return true;
}

static bool test_missing_target_role_fails_clearly() {
    auto manifest = valid_manifest();
    manifest.target.target_token_embedding.clear();
    try {
        cactus::engine::validate_spec_decode_manifest(manifest);
    } catch (const std::invalid_argument& e) {
        return std::string(e.what()).find("target.target_token_embedding") != std::string::npos;
    }
    return false;
}

static bool test_missing_assistant_role_fails_clearly() {
    auto manifest = valid_manifest();
    manifest.assistant.logits_output.clear();
    try {
        cactus::engine::validate_spec_decode_manifest(manifest);
    } catch (const std::invalid_argument& e) {
        return std::string(e.what()).find("assistant.logits_output") != std::string::npos;
    }
    return false;
}

static bool test_unsupported_method_fails_clearly() {
    auto manifest = valid_manifest();
    manifest.method = "tree";
    try {
        cactus::engine::validate_spec_decode_manifest(manifest);
    } catch (const std::invalid_argument& e) {
        return std::string(e.what()).find("unsupported spec_decode method") != std::string::npos;
    }
    return false;
}

static bool test_unsupported_version_fails_clearly() {
    auto manifest = valid_manifest();
    manifest.version = 2;
    try {
        cactus::engine::validate_spec_decode_manifest(manifest);
    } catch (const std::invalid_argument& e) {
        return std::string(e.what()).find("unsupported spec_decode version") != std::string::npos;
    }
    return false;
}

static bool test_token_only_assistant_manifest_is_rejected_for_mtp() {
    auto manifest = valid_manifest();
    manifest.assistant.current_token_embedding.clear();
    try {
        cactus::engine::validate_spec_decode_manifest(manifest);
    } catch (const std::invalid_argument& e) {
        return std::string(e.what()).find("assistant.current_token_embedding") != std::string::npos;
    }
    return false;
}

static bool test_shared_kv_roles_must_be_named() {
    auto manifest = valid_manifest();
    manifest.target.shared_kv_full_key.clear();
    try {
        cactus::engine::validate_spec_decode_manifest(manifest);
    } catch (const std::invalid_argument& e) {
        return std::string(e.what()).find("target.shared_kv.full_attention.key") != std::string::npos;
    }
    return false;
}

int main() {
    TestUtils::TestRunner runner("Spec Decode Manifest Tests");
    runner.run_test("valid_spec_decode_manifest_loads", test_valid_spec_decode_manifest_loads());
    runner.run_test("missing_target_role_fails_clearly", test_missing_target_role_fails_clearly());
    runner.run_test("missing_assistant_role_fails_clearly", test_missing_assistant_role_fails_clearly());
    runner.run_test("unsupported_method_fails_clearly", test_unsupported_method_fails_clearly());
    runner.run_test("unsupported_version_fails_clearly", test_unsupported_version_fails_clearly());
    runner.run_test("token_only_assistant_manifest_is_rejected_for_mtp", test_token_only_assistant_manifest_is_rejected_for_mtp());
    runner.run_test("shared_kv_roles_must_be_named", test_shared_kv_roles_must_be_named());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
