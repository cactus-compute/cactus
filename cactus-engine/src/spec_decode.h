#pragma once

#include <stdexcept>
#include <string>
#include <vector>

namespace cactus {
namespace engine {

struct SpecDecodeTargetRoles {
    std::string verifier_logits;
    std::string target_hidden_state;
    std::vector<std::string> assistant_shared_state_tensors;
};

struct SpecDecodeAssistantRoles {
    std::string current_token;
    std::string previous_target_hidden;
    std::vector<std::string> target_shared_state_inputs;
    std::string position;
    std::string logits_output;
    std::string next_hidden_output;
};

struct SpecDecodeManifest {
    int version = 0;
    std::string method;
    SpecDecodeTargetRoles target;
    SpecDecodeAssistantRoles assistant;
};

inline void require_spec_role(bool present, const std::string& role) {
    if (!present) {
        throw std::invalid_argument("spec_decode manifest missing required role: " + role);
    }
}

inline void validate_spec_decode_manifest(const SpecDecodeManifest& manifest) {
    if (manifest.version != 1) {
        throw std::invalid_argument("unsupported spec_decode version: " + std::to_string(manifest.version));
    }
    if (manifest.method != "assistant_chain") {
        throw std::invalid_argument("unsupported spec_decode method: " + manifest.method);
    }

    require_spec_role(!manifest.target.verifier_logits.empty(), "target.verifier_logits");
    require_spec_role(!manifest.target.target_hidden_state.empty(), "target.target_hidden_state");
    require_spec_role(!manifest.target.assistant_shared_state_tensors.empty(), "target.assistant_shared_state_tensors");
    require_spec_role(!manifest.assistant.current_token.empty(), "assistant.current_token");
    require_spec_role(!manifest.assistant.previous_target_hidden.empty(), "assistant.previous_target_hidden");
    require_spec_role(!manifest.assistant.target_shared_state_inputs.empty(), "assistant.target_shared_state_inputs");
    require_spec_role(!manifest.assistant.position.empty(), "assistant.position");
    require_spec_role(!manifest.assistant.logits_output.empty(), "assistant.logits_output");
    require_spec_role(!manifest.assistant.next_hidden_output.empty(), "assistant.next_hidden_output");
}

struct SpecDraftInput {
    uint32_t current_token = 0;
    size_t position = 0;
    std::vector<float> previous_target_hidden;
    std::vector<std::vector<float>> target_shared_state;
};

struct SpecDraftOutput {
    std::vector<float> logits;
    std::vector<float> next_hidden;
};

struct SpecVerifierInput {
    std::vector<uint32_t> context_tokens;
    std::vector<uint32_t> candidate_tokens;
};

struct SpecVerifierOutput {
    std::vector<std::vector<float>> verifier_logits;
    std::vector<float> target_hidden_state;
    std::vector<std::vector<float>> assistant_shared_state;
};

class DraftAssistantRuntime {
public:
    virtual ~DraftAssistantRuntime() = default;
    virtual SpecDraftOutput draft(const SpecDraftInput& input) = 0;
};

class TargetSpeculativeCapability {
public:
    virtual ~TargetSpeculativeCapability() = default;
    virtual SpecVerifierOutput verify_full_context(const SpecVerifierInput& input) = 0;
};

} // namespace engine
} // namespace cactus
