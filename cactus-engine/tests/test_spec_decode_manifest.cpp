#include "test_utils.h"
#include "src/spec_decode.h"

#include <stdexcept>

static cactus::engine::SpecDecodeManifest valid_manifest() {
    cactus::engine::SpecDecodeManifest manifest;
    manifest.version = 1;
    manifest.method = "assistant_chain";
    manifest.target.verifier_logits = "decoder:logits";
    manifest.target.target_hidden_state = "decoder:hidden";
    manifest.target.assistant_shared_state_tensors = {"decoder:shared_0"};
    manifest.assistant.current_token = "assistant:current_token";
    manifest.assistant.previous_target_hidden = "assistant:previous_target_hidden";
    manifest.assistant.target_shared_state_inputs = {"assistant:shared_0"};
    manifest.assistant.position = "assistant:position";
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

static bool test_missing_required_roles_fails_clearly() {
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

int main() {
    TestUtils::TestRunner runner("Spec Decode Manifest Tests");
    runner.run_test("valid_spec_decode_manifest_loads", test_valid_spec_decode_manifest_loads());
    runner.run_test("missing_required_roles_fails_clearly", test_missing_required_roles_fails_clearly());
    runner.run_test("unsupported_method_fails_clearly", test_unsupported_method_fails_clearly());
    runner.run_test("unsupported_version_fails_clearly", test_unsupported_version_fails_clearly());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
