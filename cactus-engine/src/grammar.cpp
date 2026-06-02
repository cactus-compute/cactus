#include "engine.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>
#include <thread>
#include <dlpack/dlpack.h>
#include <picojson/picojson.h>

namespace cactus {
namespace engine {

namespace {

std::string xgrammar_tokenizer_metadata(const Tokenizer& tokenizer) {
    const auto& tokenizer_json_metadata = tokenizer.tokenizer_json_metadata();
    picojson::object minimal_tokenizer;
    minimal_tokenizer["decoder"] = tokenizer_json_metadata.decoder;
    minimal_tokenizer["normalizer"] = tokenizer_json_metadata.normalizer;
    minimal_tokenizer["pre_tokenizer"] = tokenizer_json_metadata.pre_tokenizer;

    const std::string detected_metadata =
        xgrammar::TokenizerInfo::DetectMetadataFromHF(picojson::value(minimal_tokenizer).serialize(false));

    picojson::value metadata_value;
    picojson::parse(metadata_value, detected_metadata);

    picojson::object metadata = metadata_value.get<picojson::object>();
    metadata["vocab_size"] = picojson::value(static_cast<int64_t>(tokenizer.get_encoded_vocab().size()));

    picojson::array stop_token_ids { picojson::value(static_cast<int64_t>(tokenizer.get_eos_token())) };
    const auto default_stop_sequence = tokenizer.get_default_stop_sequence();
    if (!default_stop_sequence.empty()) {
        const auto token = tokenizer.encode(default_stop_sequence)[0];
        stop_token_ids.emplace_back(static_cast<int64_t>(token));
    }
    metadata["stop_token_ids"] = picojson::value(std::move(stop_token_ids));

    return picojson::value(metadata).serialize(false);
}

}  // namespace

GrammarVocabulary GrammarVocabulary::from_model_dir(const std::string& model_dir) {
    auto tokenizer = Tokenizer::from_model_dir(model_dir);
    if (!tokenizer) throw std::runtime_error("Unable to load tokenizer");
    return GrammarVocabulary::from_tokenizer(*tokenizer);
}

GrammarVocabulary GrammarVocabulary::from_tokenizer(const Tokenizer& tokenizer) {
    auto metadata = xgrammar_tokenizer_metadata(tokenizer);
    return GrammarVocabulary(
        xgrammar::TokenizerInfo::FromVocabAndMetadata(tokenizer.get_encoded_vocab(), metadata)
    );
}

GrammarVocabulary::GrammarVocabulary(xgrammar::TokenizerInfo tokenizer_info)
    : tokenizer_info(std::move(tokenizer_info)) {
    const auto& stop_token_ids = this->tokenizer_info.GetStopTokenIds();
    stop_token_ids_.assign(stop_token_ids.begin(), stop_token_ids.end());
}

bool GrammarVocabulary::add_prefix_space() const {
    return tokenizer_info.GetAddPrefixSpace();
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

Grammar::Grammar() : grammar(xgrammar::NullObj{}) {}

Grammar::Grammar(xgrammar::Grammar raw_grammar)
    : grammar(std::move(raw_grammar)) {}

Grammar Grammar::ebnf(const std::string& ebnf, const std::string& start_symbol) {
    return Grammar(xgrammar::Grammar::FromEBNF(ebnf, start_symbol));
}

Grammar Grammar::epsilon() {
    static auto grammar = Grammar::ebnf("root ::= \"\"");
    return grammar;
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
    if (handles.empty()) return Grammar();
    return Grammar(xgrammar::Grammar::Concat(handles));
}

Grammar Grammar::optional(const Grammar& grammar) {
    if (grammar.is_empty()) return grammar;
    return Grammar::unite({Grammar::epsilon(), grammar});
}

Grammar Grammar::star(const Grammar& grammar) {
    if (grammar.is_empty()) return grammar;

    picojson::object structural_tag;
    structural_tag["type"] = picojson::value("structural_tag");

    picojson::object star;
    star["type"] = picojson::value("star");

    picojson::object grammar_obj;
    grammar_obj["type"] = picojson::value("grammar");
    grammar_obj["grammar"] = picojson::value(grammar.ebnf());

    star["content"] = picojson::value(grammar_obj);
    structural_tag["format"] = picojson::value(star);
    return Grammar::structural_tag(picojson::value(structural_tag).serialize());
}

Grammar Grammar::repeat(const Grammar& grammar, int count) {
    if (grammar.is_empty()) return grammar;
    if (count == 0) return Grammar::epsilon();
    std::vector<Grammar> grammars(count, grammar);
    return Grammar::concatenate(grammars);
}

Grammar Grammar::repeat_range(const Grammar& grammar, int min_count, int max_count) {
    if (grammar.is_empty()) return Grammar();
    if (max_count != -1 && max_count < min_count) {
        throw std::runtime_error("repeat_range max_count must be >= min_count or size_t(-1)");
    }
    if (min_count == 0 && max_count == 0) return Grammar::epsilon();
    if (min_count == max_count) return Grammar::repeat(grammar, min_count);

    picojson::value format_json;

    picojson::object structural_tag;
    structural_tag["type"] = picojson::value("structural_tag");

    picojson::object repeat;
    repeat["type"] = picojson::value("repeat");
    repeat["min"] = picojson::value(static_cast<double>(min_count));
    repeat["max"] = picojson::value(static_cast<double>(max_count));

    picojson::object grammar_obj;
    grammar_obj["type"] = picojson::value("grammar");
    grammar_obj["grammar"] = picojson::value(grammar.ebnf());

    repeat["content"] = picojson::value(grammar_obj);

    structural_tag["format"] = picojson::value(repeat);
    return Grammar::structural_tag(picojson::value(structural_tag).serialize());
}

bool Grammar::is_empty() const {
    return grammar.IsNull();
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
    : compiler(nullptr), tokenizer_info(std::move(vocab.raw_value())) {
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
    return GrammarMatcher(
        xgrammar::GrammarMatcher(compiled_grammar),
        std::move(compiled_grammar),
        tokenizer_info
    );
}

GrammarMatcher::GrammarMatcher(
    xgrammar::GrammarMatcher matcher,
    xgrammar::CompiledGrammar compiled_grammar,
    xgrammar::TokenizerInfo tokenizer_info
)
    : matcher(std::move(matcher)),
      compiled_grammar(std::move(compiled_grammar)),
      tokenizer_info(std::move(tokenizer_info)) {}

Grammar GrammarMatcher::grammar() const {
    return Grammar(compiled_grammar.GetGrammar());
}

void GrammarMatcher::rollback(int tokens) {
    matcher.Rollback(tokens);
}

void GrammarMatcher::reset() {
    matcher.Reset();
}

GrammarMatcher GrammarMatcher::fork() const {
    return GrammarMatcher(matcher.Fork(), compiled_grammar, tokenizer_info);
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
