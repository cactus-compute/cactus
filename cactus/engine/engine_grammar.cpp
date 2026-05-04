#include "engine.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <dlpack/dlpack.h>

namespace cactus {
namespace engine {

namespace {

static xgrammar::VocabType to_xgrammar_vocab_type(const TokenizerRuntimeConfig& runtime_config) {
    const auto has_none = runtime_config.decoder == TokenizerRuntimeConfig::Decoder::NONE
        && runtime_config.normalizer == TokenizerRuntimeConfig::Normalizer::NONE;
    const auto is_byte_level = runtime_config.decoder == TokenizerRuntimeConfig::Decoder::BYTE_LEVEL
        || runtime_config.normalizer == TokenizerRuntimeConfig::Normalizer::BYTE_LEVEL;
    if (runtime_config.byte_fallback) {
        return xgrammar::VocabType::BYTE_FALLBACK;
    }
    if (is_byte_level || has_none) {
        return xgrammar::VocabType::BYTE_LEVEL;
    }
    return xgrammar::VocabType::RAW;
}

} // anonymous namespace

GrammarVocabulary::GrammarVocabulary(xgrammar::TokenizerInfo tokenizer_info)
    : tokenizer_info(std::move(tokenizer_info)) {
    const auto& stop_token_ids = this->tokenizer_info.GetStopTokenIds();
    stop_token_ids_.assign(stop_token_ids.begin(), stop_token_ids.end());
}

size_t GrammarVocabulary::vocab_size() const {
    return static_cast<size_t>(tokenizer_info.GetVocabSize());
}

const std::vector<uint32_t>& GrammarVocabulary::stop_token_ids() const {
    return stop_token_ids_;
}

const xgrammar::TokenizerInfo& GrammarVocabulary::raw_value() const {
    return tokenizer_info;
}

GrammarVocabulary Tokenizer::get_grammar_vocabulary() const {
    std::vector<uint32_t> stop_token_ids = {get_eos_token()};
    std::string default_stop = get_default_stop_sequence();
    if (!default_stop.empty()) {
        std::vector<uint32_t> encoded = encode(default_stop);
        if (encoded.size() == 1 && std::find(stop_token_ids.begin(), stop_token_ids.end(), encoded[0]) == stop_token_ids.end()) {
            stop_token_ids.push_back(encoded[0]);
        }
    }

    const auto& encoded_vocab = get_encoded_vocab();
    std::vector<int32_t> stop_token_ids_int32(stop_token_ids.begin(), stop_token_ids.end());
    return GrammarVocabulary(xgrammar::TokenizerInfo(
        encoded_vocab,
        to_xgrammar_vocab_type(runtime_config_),
        static_cast<int>(encoded_vocab.size()),
        stop_token_ids_int32,
        false
    ));
}

Grammar::Grammar() : grammar(xgrammar::NullObj{}), is_universal_(false) {}

Grammar::Grammar(xgrammar::Grammar raw_grammar)
    : grammar(raw_grammar), is_universal_(false) {}

Grammar Grammar::ebnf(const std::string& ebnf, const std::string& start_symbol) {
    return Grammar(xgrammar::Grammar::FromEBNF(ebnf, start_symbol));
}

Grammar Grammar::json() {
    return Grammar(xgrammar::Grammar::BuiltinJSONGrammar());
}

Grammar Grammar::universal() {
    static auto grammar = Grammar::structural_tag(R"({
        "type": "structural_tag",
        "format": {
            "type": "any_text"
        }
    })");
    grammar.is_universal_ = true;
    return grammar;
}

Grammar Grammar::json_schema(
    const std::string& json_schema,
    bool any_whitespace,
    int indent,
    std::pair<std::string, std::string> separators,
    bool strict_mode,
    int max_whitespace_count
) {
    return Grammar(xgrammar::Grammar::FromJSONSchema(
        json_schema,
        any_whitespace,
        indent,
        separators,
        strict_mode,
        max_whitespace_count >= 0 ? std::optional<int>(max_whitespace_count) : std::nullopt
    ));
}

Grammar Grammar::regex(const std::string& regex) {
    return Grammar(xgrammar::Grammar::FromRegex(regex));
}

Grammar Grammar::structural_tag(const std::string& structural_tag_json, const GrammarVocabulary* vocab) {
    auto result = xgrammar::Grammar::FromStructuralTag(
        structural_tag_json,
        vocab ? std::optional<xgrammar::TokenizerInfo>(vocab->raw_value()) : std::nullopt
    );
    if (std::holds_alternative<xgrammar::Grammar>(result)) {
        return Grammar(std::get<xgrammar::Grammar>(std::move(result)));
    }
    throw std::get<xgrammar::StructuralTagError>(result);
}

Grammar Grammar::unite(const std::vector<Grammar>& grammars) {
    std::vector<xgrammar::Grammar> handles;
    handles.reserve(grammars.size());
    for (const auto& grammar : grammars) {
        if (grammar.is_empty()) continue;
        if (grammar.is_universal()) return grammar;
        handles.push_back(grammar.raw_value());
    }
    return handles.empty() ? Grammar() : Grammar(xgrammar::Grammar::Union(handles));
}

Grammar Grammar::concatenate(const std::vector<Grammar>& grammars) {
    std::vector<xgrammar::Grammar> handles;
    handles.reserve(grammars.size());
    for (const auto& grammar : grammars) {
        if (grammar.is_empty()) continue;
        handles.push_back(grammar.raw_value());
    }
    return handles.empty() ? Grammar() : Grammar(xgrammar::Grammar::Concat(handles));
}

bool Grammar::is_empty() const {
    return grammar.IsNull();
}

bool Grammar::is_universal() const {
    return is_universal_;
}

std::string Grammar::ebnf() const {
    if (is_empty()) {
        throw std::runtime_error("Cannot get EBNF for empty grammar");
    }
    return grammar.ToString();
}

const xgrammar::Grammar& Grammar::raw_value() const {
    return grammar;
}

GrammarEngine::GrammarEngine(const GrammarVocabulary& vocab)
    : compiler(nullptr), tokenizer_info(vocab.raw_value()) {
    compiler = xgrammar::GrammarCompiler(
        tokenizer_info,
        static_cast<int>(std::max(1u, std::thread::hardware_concurrency()))
    );
}

GrammarMatcher GrammarEngine::compile_matcher(const Grammar& grammar) {
    if (grammar.is_empty()) {
        throw std::runtime_error("Cannot create GrammarMatcher with empty grammar");
    }

    auto compiled_grammar = compiler.CompileGrammar(grammar.raw_value());
    return GrammarMatcher(xgrammar::GrammarMatcher(compiled_grammar), tokenizer_info);
}

GrammarMatcher::GrammarMatcher(xgrammar::GrammarMatcher matcher, xgrammar::TokenizerInfo tokenizer_info)
    : matcher(std::move(matcher)), tokenizer_info(std::move(tokenizer_info)) {}

void GrammarMatcher::rollback(int tokens) {
    matcher.Rollback(tokens);
}

void GrammarMatcher::reset() {
    matcher.Reset();
}

GrammarMatcher GrammarMatcher::fork() const {
    return GrammarMatcher(matcher.Fork(), tokenizer_info);
}

bool GrammarMatcher::is_completed() const {
    return matcher.IsCompleted();
}

bool GrammarMatcher::is_terminated() const {
    return matcher.IsTerminated();
}

bool GrammarMatcher::accept(uint32_t token_id, bool log_rejection) {
    const bool accepted = matcher.AcceptToken(token_id);
    if (!accepted && log_rejection) {
        CACTUS_LOG_WARN("grammar matcher", "Token id: " << token_id << " was not accepted by grammar matcher.");
    }
    return accepted;
}

bool GrammarMatcher::next_bitmask(std::vector<int32_t>& bitmask, size_t logits_buffer_size) {
    const size_t vocab_bitmask_size = xgrammar::GetBitmaskSize(tokenizer_info.GetVocabSize());
    const size_t logits_bitmask_size = xgrammar::GetBitmaskSize(logits_buffer_size);
    bitmask.assign(std::max(vocab_bitmask_size, logits_bitmask_size), 0);

    int64_t bitmask_shape[1];
    int64_t bitmask_strides[1];
    bitmask_shape[0] = vocab_bitmask_size;
    bitmask_strides[0] = 1;

    DLTensor bitmask_tensor;
    bitmask_tensor.data = bitmask.data();
    bitmask_tensor.device = DLDevice{kDLCPU, 0};
    bitmask_tensor.ndim = 1;
    bitmask_tensor.dtype = xgrammar::GetBitmaskDLType();
    bitmask_tensor.shape = bitmask_shape;
    bitmask_tensor.strides = bitmask_strides;
    bitmask_tensor.byte_offset = 0;

    return matcher.FillNextTokenBitmask(&bitmask_tensor, 0);
}

} // namespace engine
} // namespace cactus
