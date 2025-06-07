#include "rn-llama.h"
#include "rn-tts.h"

// Include multimodal support
#include "tools/mtmd/mtmd.h"
#include "tools/mtmd/clip.h"

namespace rnllama {

// Computes FNV-1a hash of the data
static std::string fnv_hash(const uint8_t * data, size_t len) {
    const uint64_t fnv_prime = 0x100000001b3ULL;
    uint64_t hash = 0xcbf29ce484222325ULL;

    for (size_t i = 0; i < len; ++i) {
        hash ^= data[i];
        hash *= fnv_prime;
    }
    return std::to_string(hash);
}

static const std::string base64_chars =
             "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
             "abcdefghijklmnopqrstuvwxyz"
             "0123456789+/";

// Base64 decoding function
static std::vector<uint8_t> base64_decode(const std::string &encoded_string) {
    std::vector<uint8_t> decoded;
    int in_len = encoded_string.size();
    int i = 0;
    int j = 0;
    int in_ = 0;
    unsigned char char_array_4[4], char_array_3[3];

    while (in_len-- && (encoded_string[in_] != '=')) {
        if (isspace(encoded_string[in_])) {
            in_++;
            continue;
        }

        if (encoded_string[in_] == '=' || base64_chars.find(encoded_string[in_]) == std::string::npos) {
            break;
        }

        char_array_4[i++] = encoded_string[in_]; in_++;
        if (i == 4) {
            for (i = 0; i < 4; i++) {
                char_array_4[i] = base64_chars.find(char_array_4[i]);
            }

            char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

            for (i = 0; i < 3; i++) {
                decoded.push_back(char_array_3[i]);
            }
            i = 0;
        }
    }

    if (i) {
        for (j = i; j < 4; j++) {
            char_array_4[j] = 0;
        }

        for (j = 0; j < 4; j++) {
            char_array_4[j] = base64_chars.find(char_array_4[j]);
        }

        char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
        char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

        for (j = 0; j < i - 1; j++) {
            decoded.push_back(char_array_3[j]);
        }
    }

    return decoded;
}

static const std::vector<lm_ggml_type> kv_cache_types = {
    LM_GGML_TYPE_F32,
    LM_GGML_TYPE_F16,
    LM_GGML_TYPE_BF16,
    LM_GGML_TYPE_Q8_0,
    LM_GGML_TYPE_Q4_0,
    LM_GGML_TYPE_Q4_1,
    LM_GGML_TYPE_IQ4_NL,
    LM_GGML_TYPE_Q5_0,
    LM_GGML_TYPE_Q5_1,
};

lm_ggml_type kv_cache_type_from_str(const std::string & s) {
    for (const auto & type : kv_cache_types) {
        if (lm_ggml_type_name(type) == s) {
            return type;
        }
    }
    throw std::runtime_error("Unsupported cache type: " + s);
}

static void llama_batch_clear(llama_batch *batch) {
    batch->n_tokens = 0;
}

static void llama_batch_add(llama_batch *batch, llama_token id, llama_pos pos, std::vector<llama_seq_id> seq_ids, bool logits) {
    batch->token   [batch->n_tokens] = id;
    batch->pos     [batch->n_tokens] = pos;
    batch->n_seq_id[batch->n_tokens] = seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); i++) {
        batch->seq_id[batch->n_tokens][i] = seq_ids[i];
    }
    batch->logits  [batch->n_tokens] = logits ? 1 : 0;
    batch->n_tokens += 1;
}

// NOTE: Edit from https://github.com/ggerganov/llama.cpp/blob/master/examples/server/server.cpp

static void log(const char *level, const char *function, int line,
                       const char *format, ...)
{
    va_list args;
    #if defined(__ANDROID__)
        char prefix[256];
        snprintf(prefix, sizeof(prefix), "%s:%d %s", function, line, format);

        va_start(args, format);
        android_LogPriority priority;
        if (strcmp(level, "ERROR") == 0) {
            priority = ANDROID_LOG_ERROR;
        } else if (strcmp(level, "WARNING") == 0) {
            priority = ANDROID_LOG_WARN;
        } else if (strcmp(level, "INFO") == 0) {
            priority = ANDROID_LOG_INFO;
        } else {
            priority = ANDROID_LOG_DEBUG;
        }
        __android_log_vprint(priority, "RNLlama", prefix, args);
        va_end(args);
    #else
        printf("[%s] %s:%d ", level, function, line);
        va_start(args, format);
        vprintf(format, args);
        va_end(args);
        printf("\n");
    #endif
}

#if RNLLAMA_VERBOSE != 1
#define LOG_VERBOSE(MSG, ...)
#else
#define LOG_VERBOSE(MSG, ...)                                       \
    do                                                              \
    {                                                               \
        if (rnllama_verbose)                                        \
        {                                                           \
            log("VERBOSE", __func__, __LINE__, MSG, ##__VA_ARGS__); \
        }                                                           \
    } while (0)
#endif

#define LOG_ERROR(MSG, ...) log("ERROR", __func__, __LINE__, MSG, ##__VA_ARGS__)
#define LOG_WARNING(MSG, ...) log("WARNING", __func__, __LINE__, MSG, ##__VA_ARGS__)
#define LOG_INFO(MSG, ...) log("INFO", __func__, __LINE__, MSG, ##__VA_ARGS__)

static size_t common_part(const std::vector<llama_token> &a, const std::vector<llama_token> &b)
{
    size_t i;
    for (i = 0; i < a.size() && i < b.size() && a[i] == b[i]; i++)
    {
    }
    return i;
}

static bool ends_with(const std::string &str, const std::string &suffix)
{
    return str.size() >= suffix.size() &&
           0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix);
}

static size_t find_partial_stop_string(const std::string &stop,
                                       const std::string &text)
{
    if (!text.empty() && !stop.empty())
    {
        const char text_last_char = text.back();
        for (int64_t char_index = stop.size() - 1; char_index >= 0; char_index--)
        {
            if (stop[char_index] == text_last_char)
            {
                const std::string current_partial = stop.substr(0, char_index + 1);
                if (ends_with(text, current_partial))
                {
                    return text.size() - char_index - 1;
                }
            }
        }
    }
    return std::string::npos;
}

// format incomplete utf-8 multibyte character for output
std::string tokens_to_output_formatted_string(const llama_context *ctx, const llama_token token)
{
    std::string out = token == -1 ? "" : common_token_to_piece(ctx, token);
    // if the size is 1 and first bit is 1, meaning it's a partial character
    //   (size > 1 meaning it's already a known token)
    if (out.size() == 1 && (out[0] & 0x80) == 0x80)
    {
        std::stringstream ss;
        ss << std::hex << (out[0] & 0xff);
        std::string res(ss.str());
        out = "byte: \\x" + res;
    }
    return out;
}

std::string tokens_to_str(llama_context *ctx, const std::vector<llama_token>::const_iterator begin, const std::vector<llama_token>::const_iterator end)
{
    std::string ret;
    for (auto it = begin; it != end; ++it)
    {
        ret += common_token_to_piece(ctx, *it);
    }
    return ret;
}

struct llama_rn_context_mtmd {
  mtmd_context *mtmd_ctx = nullptr;
};

llama_rn_context::~llama_rn_context() {
    if (ctx_sampling != nullptr) {
        common_sampler_free(ctx_sampling);
    }

    releaseMultimodal();
}

void llama_rn_context::rewind() {
    is_interrupted = false;
    params.antiprompt.clear();
    params.sampling.grammar.clear();
    num_prompt_tokens = 0;
    num_tokens_predicted = 0;
    generated_text = "";
    generated_text.reserve(params.n_ctx);
    generated_token_probs.clear();
    audio_tokens.clear();
    truncated = false;
    context_full = false;
    stopped_eos = false;
    stopped_word = false;
    stopped_limit = false;
    stopping_word = "";
    incomplete = false;
    n_remain = 0;
    n_past = 0;
    params.sampling.n_prev = n_ctx;
    next_token_uses_guide_token = true;
    guide_tokens.clear();
}

bool llama_rn_context::initSampling() {
    if (ctx_sampling != nullptr) {
        common_sampler_free(ctx_sampling);
    }
    ctx_sampling = common_sampler_init(model, params.sampling);
    return ctx_sampling != nullptr;
}

bool llama_rn_context::loadModel(common_params &params_)
{
    params = params_;
    llama_init = common_init_from_params(params);
    model = llama_init.model.get();
    ctx = llama_init.context.get();
    if (model == nullptr)
    {
        LOG_ERROR("unable to load model: %s", params_.model.path.c_str());
        return false;
    }
    templates = common_chat_templates_init(model, params.chat_template);
    n_ctx = llama_n_ctx(ctx);

    // Initialize context shift flag
    LOG_INFO("ctx_shift: %s", params.ctx_shift ? "enabled" : "disabled");

    // We can uncomment for debugging or after this fix: https://github.com/ggerganov/llama.cpp/pull/11101
    // LOG_INFO("%s\n", common_params_get_system_info(params).c_str());

    return true;
}

bool llama_rn_context::validateModelChatTemplate(bool use_jinja, const char *name) const {
    const char * tmpl = llama_model_chat_template(model, name);
    if (tmpl == nullptr) {
      return false;
    }
    return common_chat_verify_template(tmpl, use_jinja);
}

common_chat_params llama_rn_context::getFormattedChatWithJinja(
  const std::string &messages,
  const std::string &chat_template,
  const std::string &json_schema,
  const std::string &tools,
  const bool &parallel_tool_calls,
  const std::string &tool_choice
) const {
    common_chat_templates_inputs inputs;
    inputs.use_jinja = true;
    inputs.messages = common_chat_msgs_parse_oaicompat(json::parse(messages));
    auto useTools = !tools.empty();
    if (useTools) {
        inputs.tools = common_chat_tools_parse_oaicompat(json::parse(tools));
    }
    inputs.parallel_tool_calls = parallel_tool_calls;
    if (!tool_choice.empty()) {
        inputs.tool_choice = common_chat_tool_choice_parse_oaicompat(tool_choice);
    }
    if (!json_schema.empty()) {
        inputs.json_schema = json::parse(json_schema);
    }
    inputs.extract_reasoning = params.reasoning_format != COMMON_REASONING_FORMAT_NONE;

    // If chat_template is provided, create new one and use it (probably slow)
    if (!chat_template.empty()) {
        auto tmps = common_chat_templates_init(model, chat_template);
        return common_chat_templates_apply(tmps.get(), inputs);
    } else {
        return common_chat_templates_apply(templates.get(), inputs);
    }
}

std::string llama_rn_context::getFormattedChat(
  const std::string &messages,
  const std::string &chat_template
) const {
    common_chat_templates_inputs inputs;
    inputs.messages = common_chat_msgs_parse_oaicompat(json::parse(messages));
    inputs.use_jinja = false;

    // If chat_template is provided, create new one and use it (probably slow)
    if (!chat_template.empty()) {
        auto tmps = common_chat_templates_init(model, chat_template);
        return common_chat_templates_apply(tmps.get(), inputs).prompt;
    } else {
        return common_chat_templates_apply(templates.get(), inputs).prompt;
    }
}

void llama_rn_context::truncatePrompt(std::vector<llama_token> &prompt_tokens) {
    const int n_left = n_ctx - params.n_keep;
    const int n_block_size = n_left / 2;
    const int erased_blocks = (prompt_tokens.size() - params.n_keep - n_block_size) / n_block_size;

    // Keep n_keep tokens at start of prompt (at most n_ctx - 4)
    std::vector<llama_token> new_tokens(prompt_tokens.begin(), prompt_tokens.begin() + params.n_keep);

    new_tokens.insert(new_tokens.end(), prompt_tokens.begin() + params.n_keep + erased_blocks * n_block_size, prompt_tokens.end());

    LOG_INFO("input truncated, n_ctx: %d, n_keep: %d, n_left: %d, old_size: %d, new_size: %d",
        n_ctx,
        params.n_keep,
        n_left,
        prompt_tokens.size(),
        new_tokens.size()
    );

    truncated = true;
    prompt_tokens = new_tokens;
}

void llama_rn_context::loadPrompt(const std::vector<std::string> &media_paths) {
    bool has_media = !media_paths.empty();

    if (!has_media) {
        std::vector<llama_token> text_tokens;
        // Text-only path
        text_tokens = ::common_tokenize(ctx, params.prompt, true, true);
        num_prompt_tokens = text_tokens.size();

        // LOG tokens
        std::stringstream ss;
        ss << "\n" << __func__ << ": prompt_tokens = ";
        for (auto& token : text_tokens) {
            ss << token << " ";
        }
        LOG_INFO("%s\n", ss.str().c_str());

        if (params.n_keep < 0) {
            params.n_keep = (int)num_prompt_tokens;
        }
        params.n_keep = std::min(n_ctx - 4, params.n_keep);

        // Handle truncation if needed
        if (num_prompt_tokens >= (size_t)n_ctx) {
            if (!params.ctx_shift) {
                context_full = true;
                return;
            }
            truncatePrompt(text_tokens);
            num_prompt_tokens = text_tokens.size();
            LM_GGML_ASSERT(num_prompt_tokens < (size_t)n_ctx);
        }

        // Update sampling context
        for (auto & token : text_tokens) {
            common_sampler_accept(ctx_sampling, token, false);
        }

        // compare the evaluated prompt with the new prompt
        n_past = common_part(embd, text_tokens);

        embd = text_tokens;
        if (n_past == num_prompt_tokens) {
            // we have to evaluate at least 1 token to generate logits.
            n_past--;
        }

        // Manage KV cache
        llama_kv_self_seq_rm(ctx, 0, n_past, -1);

        LOG_VERBOSE("prompt ingested, n_past: %d, cached: %s, to_eval: %s",
            n_past,
            tokens_to_str(ctx, embd.cbegin(), embd.cbegin() + n_past).c_str(),
            tokens_to_str(ctx, embd.cbegin() + n_past, embd.cend()).c_str()
        );
    } else {
        // Multimodal path - process all media paths
        processMedia(params.prompt, media_paths);
        num_prompt_tokens = embd.size();
    }

    has_next_token = true;

    LOG_INFO("[DEBUG] Input processed: n_past=%d, embd.size=%zu, num_prompt_tokens=%zu, has_media=%d",
             n_past, embd.size(), num_prompt_tokens, has_media ? 1 : 0);
}

void llama_rn_context::setGuideTokens(const std::vector<llama_token> &tokens) {
    guide_tokens = tokens;
}

void llama_rn_context::beginCompletion() {
    // number of tokens to keep when resetting context
    n_remain = params.n_predict;
    llama_perf_context_reset(ctx);
    is_predicting = true;
}

void llama_rn_context::endCompletion() {
    is_predicting = false;
}

completion_token_output llama_rn_context::nextToken()
{
    completion_token_output result;
    result.tok = -1;

    if (embd.size() >= (size_t)params.n_ctx)
    {
        if (!params.ctx_shift) {
            // If context shifting is disabled, stop generation
            LOG_WARNING("context full, n_ctx: %d, tokens: %d", params.n_ctx, embd.size());
            has_next_token = false;
            context_full = true;
            return result;
        }

        // Shift context

        const int n_left    = n_past - params.n_keep - 1;
        const int n_discard = n_left/2;

        llama_kv_self_seq_rm (ctx, 0, params.n_keep + 1            , params.n_keep + n_discard + 1);
        llama_kv_self_seq_add(ctx, 0, params.n_keep + 1 + n_discard, n_past, -n_discard);

        for (size_t i = params.n_keep + 1 + n_discard; i < embd.size(); i++)
        {
            embd[i - n_discard] = embd[i];
        }
        embd.resize(embd.size() - n_discard);

        n_past -= n_discard;
        truncated = true;

        LOG_VERBOSE("context shifted, new n_past: %d, new size: %d", n_past, embd.size());
    }

    bool tg = true;
    while (n_past < embd.size())
    {
        int n_eval = (int)embd.size() - n_past;
        tg = n_eval == 1;
        if (n_eval > params.n_batch)
        {
            n_eval = params.n_batch;
        }
        if (llama_decode(ctx, llama_batch_get_one(&embd[n_past], n_eval)))
        {
            LOG_ERROR("failed to eval, n_eval: %d, n_past: %d, n_threads: %d, embd: %s",
                n_eval,
                n_past,
                params.cpuparams.n_threads,
                tokens_to_str(ctx, embd.cbegin() + n_past, embd.cend()).c_str()
            );
            has_next_token = false;
            return result;
        }
        n_past += n_eval;

        if(is_interrupted) {
            LOG_INFO("Decoding Interrupted");
            embd.resize(n_past);
            has_next_token = false;
            return result;
        }
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);

    if (params.n_predict == 0)
    {
        has_next_token = false;
        result.tok = llama_vocab_eos(vocab);
        return result;
    }

    {
        // out of user input, sample next token
        std::vector<llama_token_data> candidates;
        candidates.reserve(llama_vocab_n_tokens(vocab));

        llama_token new_token_id = common_sampler_sample(ctx_sampling, ctx, -1);

        if (next_token_uses_guide_token && !guide_tokens.empty() && !llama_vocab_is_control(vocab, new_token_id) && !llama_vocab_is_eog(vocab, new_token_id)) {
            new_token_id = guide_tokens[0];
            guide_tokens.erase(guide_tokens.begin());
        }
        next_token_uses_guide_token = (new_token_id == 198);
        result.tok = new_token_id;

        llama_token_data_array cur_p = *common_sampler_get_candidates(ctx_sampling);

        const int32_t n_probs = params.sampling.n_probs;

        // deprecated
        /*if (params.sampling.temp <= 0 && n_probs > 0)
        {
            // For llama_sample_token_greedy we need to sort candidates
            llama_sampler_init_softmax();

        }*/


        for (size_t i = 0; i < std::min(cur_p.size, (size_t)n_probs); ++i)
        {
            result.probs.push_back({cur_p.data[i].id, cur_p.data[i].p});
        }

        common_sampler_accept(ctx_sampling, result.tok, true);
        if (tg) {
            num_tokens_predicted++;
        }
    }

    // add it to the context
    embd.push_back(result.tok);
    // decrement remaining sampling budget
    --n_remain;

    if (!embd.empty() && embd.back() == llama_vocab_eos(vocab))
    {
        // stopping_word = llama_token_to_piece(ctx, embd.back());
        has_next_token = false;
        stopped_eos = true;
        LOG_VERBOSE("eos token found", "");
        return result;
    }

    has_next_token = params.n_predict == -1 || n_remain != 0;
    return result;
}

size_t llama_rn_context::findStoppingStrings(const std::string &text, const size_t last_token_size,
                            const stop_type type)
{
    size_t stop_pos = std::string::npos;
    for (const std::string &word : params.antiprompt)
    {
        size_t pos;
        if (type == STOP_FULL)
        {
            const size_t tmp = word.size() + last_token_size;
            const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;
            pos = text.find(word, from_pos);
        }
        else
        {
            pos = find_partial_stop_string(word, text);
        }
        if (pos != std::string::npos &&
            (stop_pos == std::string::npos || pos < stop_pos))
        {
            if (type == STOP_FULL)
            {
                stopping_word = word;
                stopped_word = true;
                has_next_token = false;
            }
            stop_pos = pos;
        }
    }
    return stop_pos;
}

completion_token_output llama_rn_context::doCompletion()
{
    const completion_token_output token_with_probs = nextToken();

    const std::string token_text = token_with_probs.tok == -1 ? "" : common_token_to_piece(ctx, token_with_probs.tok);
    generated_text += token_text;

    if (isVocoderEnabled()) {
        tts_type type = getTTSType();
        if ((type == OUTETTS_V0_2 || type == OUTETTS_V0_3) && (token_with_probs.tok >= 151672 && token_with_probs.tok <= 155772)) {
            audio_tokens.push_back(token_with_probs.tok);
        }
    }

    if (params.sampling.n_probs > 0)
    {
        generated_token_probs.push_back(token_with_probs);
    }

    // check if there is incomplete UTF-8 character at the end
    for (unsigned i = 1; i < 5 && i <= generated_text.size(); ++i) {
        unsigned char c = generated_text[generated_text.size() - i];
        if ((c & 0xC0) == 0x80) {
            // continuation byte: 10xxxxxx
            continue;
        }
        if ((c & 0xE0) == 0xC0) {
            // 2-byte character: 110xxxxx ...
            incomplete = i < 2;
        } else if ((c & 0xF0) == 0xE0) {
            // 3-byte character: 1110xxxx ...
            incomplete = i < 3;
        } else if ((c & 0xF8) == 0xF0) {
            // 4-byte character: 11110xxx ...
            incomplete = i < 4;
        }
        // else 1-byte character or invalid byte
        break;
    }

    if (incomplete && !has_next_token)
    {
        has_next_token = true;
        n_remain++;
    }

    if (!has_next_token && n_remain == 0)
    {
        stopped_limit = true;
    }

    LOG_VERBOSE("next token, token: %s, token_text: %s, has_next_token: %d, n_remain: %d, num_tokens_predicted: %d, stopped_eos: %d, stopped_word: %d, stopped_limit: %d, stopping_word: %s",
        common_token_to_piece(ctx, token_with_probs.tok),
        tokens_to_output_formatted_string(ctx, token_with_probs.tok).c_str(),
        has_next_token,
        n_remain,
        num_tokens_predicted,
        stopped_eos,
        stopped_word,
        stopped_limit,
        stopping_word.c_str()
    );
    return token_with_probs;
}

std::vector<float> llama_rn_context::getEmbedding(common_params &embd_params)
{
    static const int n_embd = llama_model_n_embd(llama_get_model(ctx));
    if (!embd_params.embedding)
    {
        LOG_WARNING("embedding disabled, embedding: %s", embd_params.embedding);
        return std::vector<float>(n_embd, 0.0f);
    }
    float *data;

    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);
    printf("pooling_type: %d\n", pooling_type);
    if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
        data = llama_get_embeddings(ctx);
    } else {
        data = llama_get_embeddings_seq(ctx, 0);
    }

    if (!data) {
        return std::vector<float>(n_embd, 0.0f);
    }
    std::vector<float> embedding(data, data + n_embd), out(data, data + n_embd);
    common_embd_normalize(embedding.data(), out.data(), n_embd, embd_params.embd_normalize);
    return out;
}

std::string llama_rn_context::bench(int pp, int tg, int pl, int nr)
{
    if (is_predicting) {
        LOG_ERROR("cannot benchmark while predicting", "");
        return std::string("[]");
    }

    is_predicting = true;

    double pp_avg = 0;
    double tg_avg = 0;

    double pp_std = 0;
    double tg_std = 0;

    // TODO: move batch into llama_rn_context (related https://github.com/mybigday/llama.rn/issues/30)
    llama_batch batch = llama_batch_init(
        std::min(pp, params.n_ubatch), // max n_tokens is limited by n_ubatch
        0,                         // No embeddings
        1                          // Single sequence
    );

    for (int i = 0; i < nr; i++)
    {
        llama_batch_clear(&batch);

        const int n_tokens = pp;

        for (int i = 0; i < n_tokens; i++)
        {
            llama_batch_add(&batch, 0, i, {0}, false);
        }
        batch.logits[batch.n_tokens - 1] = 1; // true

        llama_kv_self_clear(ctx);

        const int64_t t_pp_start = llama_time_us();
        if (llama_decode(ctx, batch) != 0)
        {
            LOG_ERROR("llama_decode() failed during prompt", "");
        }
        const int64_t t_pp_end = llama_time_us();
        llama_kv_self_clear(ctx);

        if (is_interrupted) break;

        const int64_t t_tg_start = llama_time_us();

        for (int i = 0; i < tg; i++)
        {
            llama_batch_clear(&batch);

            for (int j = 0; j < pl; j++)
            {
                llama_batch_add(&batch, 0, i, {j}, true);
            }

            if (llama_decode(ctx, batch) != 0)
            {
                LOG_ERROR("llama_decode() failed during text generation", "");
            }
            if (is_interrupted) break;
        }

        const int64_t t_tg_end = llama_time_us();

        llama_kv_self_clear(ctx);

        const double t_pp = (t_pp_end - t_pp_start) / 1000000.0;
        const double t_tg = (t_tg_end - t_tg_start) / 1000000.0;

        const double speed_pp = pp / t_pp;
        const double speed_tg = (pl * tg) / t_tg;

        pp_avg += speed_pp;
        tg_avg += speed_tg;

        pp_std += speed_pp * speed_pp;
        tg_std += speed_tg * speed_tg;
    }

    pp_avg /= nr;
    tg_avg /= nr;

    if (nr > 1) {
        pp_std = sqrt(pp_std / (nr - 1) - pp_avg * pp_avg * nr / (nr - 1));
        tg_std = sqrt(tg_std / (nr - 1) - tg_avg * tg_avg * nr / (nr - 1));
    } else {
        pp_std = 0;
        tg_std = 0;
    }

    if (is_interrupted) llama_kv_self_clear(ctx);
    endCompletion();

    char model_desc[128];
    llama_model_desc(model, model_desc, sizeof(model_desc));
    return std::string("[\"") + model_desc + std::string("\",") +
        std::to_string(llama_model_size(model)) + std::string(",") +
        std::to_string(llama_model_n_params(model)) + std::string(",") +
        std::to_string(pp_avg) + std::string(",") +
        std::to_string(pp_std) + std::string(",") +
        std::to_string(tg_avg) + std::string(",") +
        std::to_string(tg_std) +
        std::string("]");
}

int llama_rn_context::applyLoraAdapters(std::vector<common_adapter_lora_info> lora) {
    for (auto &la : lora) {
        la.ptr = llama_adapter_lora_init(model, la.path.c_str());
        if (la.ptr == nullptr) {
            LOG_ERROR("failed to apply lora adapter '%s'\n", la.path.c_str());
            return -1;
        }
    }
    this->lora = lora;
    common_set_adapter_lora(ctx, lora);
    return 0;
}

void llama_rn_context::removeLoraAdapters() {
    this->lora.clear();
    common_set_adapter_lora(ctx, this->lora); // apply empty list
}

std::vector<common_adapter_lora_info> llama_rn_context::getLoadedLoraAdapters() {
    return this->lora;
}

bool llama_rn_context::initMultimodal(const std::string &mmproj_path, bool use_gpu) {
    LOG_INFO("[DEBUG] Initializing multimodal with mmproj path: %s", mmproj_path.c_str());

    if (model == nullptr) {
        LOG_ERROR("[DEBUG] Model not loaded, cannot initialize multimodal", "");
        return false;
    }

    LOG_INFO("[DEBUG] Model info: n_ctx=%d, n_embd=%d",
             llama_n_ctx(ctx),
             llama_model_n_embd(model));

    // Initialize mtmd context
    mtmd_context_params mtmd_params = mtmd_context_params_default();
    mtmd_params.use_gpu = use_gpu;
    mtmd_params.print_timings = false;
    mtmd_params.n_threads = params.cpuparams.n_threads;
    mtmd_params.verbosity = (lm_ggml_log_level)LM_GGML_LOG_LEVEL_INFO;

    LOG_INFO("[DEBUG] Initializing mtmd context with threads=%d", mtmd_params.n_threads);

    auto mtmd_ctx = mtmd_init_from_file(mmproj_path.c_str(), model, mtmd_params);
    if (mtmd_ctx == nullptr) {
        LOG_ERROR("[DEBUG] Failed to initialize multimodal context with mmproj: %s", mmproj_path.c_str());
        return false;
    }
    mtmd_wrapper = new llama_rn_context_mtmd();
    mtmd_wrapper->mtmd_ctx = mtmd_ctx;

    has_multimodal = true;

    // Check if the model uses M-RoPE or non-causal attention
    bool uses_mrope = mtmd_decode_use_mrope(mtmd_ctx);
    bool uses_non_causal = mtmd_decode_use_non_causal(mtmd_ctx);
    LOG_INFO("[DEBUG] Model multimodal properties: uses_mrope=%d, uses_non_causal=%d",
             uses_mrope ? 1 : 0,
             uses_non_causal ? 1 : 0);

    // Disable context shifting when multimodal is enabled
    // This is because an media chunk may contain multiple tokens
    // and context shifting could break the media representation
    params.ctx_shift = false;

    // params.n_cache_reuse = 0;

    LOG_INFO("Multimodal context initialized successfully with mmproj: %s", mmproj_path.c_str());
    LOG_INFO("Context shifting disabled for multimodal support");
    return true;
}

struct mtmd_tokenize_result {
    std::vector<std::string> bitmap_hashes;
    std::vector<llama_token> tokens;
    std::vector<size_t> chunk_pos; // both text and media
    std::vector<size_t> chunk_pos_media; // media only
    mtmd_input_chunks* chunks = nullptr;
};

mtmd_tokenize_result tokenizeWithMedia(llama_rn_context_mtmd *mtmd_wrapper, const std::string &prompt, const std::vector<std::string> &media_paths) {
    mtmd_tokenize_result result;
    mtmd::bitmaps bitmaps;

    // Load all media paths
    for (const auto& media_path : media_paths) {
        LOG_INFO("[DEBUG] Loading media: %s",
                 media_path.substr(0, 50).c_str()); // Only log part of path for base64

        // Check if it's a base64 media
        if (media_path.compare(0, 11, "data:image/") == 0 || media_path.compare(0, 11, "data:audio/") == 0) {
            LOG_INFO("[DEBUG] Detected base64 encoded media");

            // Parse base64 data
            std::vector<std::string> parts;
            size_t comma_pos = media_path.find(',');
            if (comma_pos == std::string::npos) {
                throw std::runtime_error("Invalid base64 media format, missing comma separator");
            }

            std::string header = media_path.substr(0, comma_pos);
            std::string base64_data = media_path.substr(comma_pos + 1);

            if (header.find("base64") == std::string::npos) {
                bitmaps.entries.clear();
                throw std::runtime_error("Image must be base64 encoded");
            }

            // Decode base64
            std::vector<uint8_t> media_data = base64_decode(base64_data);
            LOG_INFO("[DEBUG] Base64 decoded, size: %zu bytes", media_data.size());

            // Load bitmap from memory buffer using direct initialization
            mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_buf(media_data.data(), media_data.size()));
            if (!bmp.ptr) {
                bitmaps.entries.clear();
                throw std::runtime_error("Failed to load base64 media");
            }

            // Calculate bitmap hash (for KV caching)
            std::string hash = fnv_hash(bmp.data(), bmp.n_bytes());
            bmp.set_id(hash.c_str());
            LOG_INFO("[DEBUG] Bitmap hash: %s", hash.c_str());
            bitmaps.entries.push_back(std::move(bmp));
            result.bitmap_hashes.push_back(hash.c_str());
        } else if (media_path.compare(0, 7, "http://") == 0 || media_path.compare(0, 8, "https://") == 0) {
            // HTTP URLs are not supported yet
            LOG_ERROR("[DEBUG] HTTP/HTTPS URLs are not supported yet: %s", media_path.c_str());
            throw std::runtime_error("HTTP/HTTPS URLs are not supported yet");
        } else {
            // Regular file path
            LOG_INFO("[DEBUG] Loading media from file");

            // Check if file exists
            FILE* file = fopen(media_path.c_str(), "rb");
            if (file == nullptr) {
                bitmaps.entries.clear();
                throw std::runtime_error("File does not exist or cannot be opened");
            }

            // Get file size
            fseek(file, 0, SEEK_END);
            long file_size = ftell(file);
            fseek(file, 0, SEEK_SET);
            LOG_INFO("[DEBUG] File exists and size is %ld bytes", file_size);
            fclose(file);

            // Create bitmap directly
            mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(media_path.c_str()));
            if (!bmp.ptr) {
                bitmaps.entries.clear();
                throw std::runtime_error("Failed to load media");
            }

            // Calculate bitmap hash (for KV caching)
            std::string hash = fnv_hash(bmp.data(), bmp.nx()*bmp.ny()*3);
            bmp.set_id(hash.c_str());
            LOG_INFO("[DEBUG] Bitmap hash: %s", hash.c_str());
            bitmaps.entries.push_back(std::move(bmp));
            result.bitmap_hashes.push_back(hash.c_str());
        }
    }

    // Create input chunks
    LOG_INFO("[DEBUG] Initializing input chunks");
    result.chunks = mtmd_input_chunks_init();
    if (result.chunks == nullptr) {
        bitmaps.entries.clear();
        throw std::runtime_error("Failed to initialize input chunks");
    }

    mtmd_input_text input_text;
    input_text.text = prompt.c_str(); // Use the full prompt with image marker
    input_text.add_special = true;  // Add BOS token if this is the first message
    input_text.parse_special = true;       // Parse special tokens like <__media__>

    /**
     * Tokenize the text and media together.
     *
     * Example of tokenization for "foo bar <__media__> baz <__media__>":
     *
     * 1. Input text with media markers:
     *
     *    "foo bar <__media__> baz <__media__>"
     *
     * 2. Model-specific markers are added.
     *
     * 3. Text is split and tokenized into chunks:
     *
     *    ┌─────────────┐  ┌─────────────────────────┐  ┌─────────┐  ┌─────────────────────────┐
     *    │ TEXT CHUNK  │  │ IMAGE CHUNK             │  │ TEXT    │  │ IMAGE CHUNK             │
     *    │ "foo bar "  │  │                         │  │ " baz " │  │                         │
     *    └─────────────┘  └─────────────────────────┘  └─────────┘  └─────────────────────────┘
     *          │                     │                      │                    │
     *          ▼                     ▼                      ▼                    ▼
     *    ┌─────────────┐  ┌─────────────────────────┐  ┌─────────┐  ┌─────────────────────────┐
     *    │ [1234,5678] │  │ Image Data Structure    │  │ [9012]  │  │ Image Data Structure    │
     *    └─────────────┘  └─────────────────────────┘  └─────────┘  └─────────────────────────┘
     *
     * 4. Image token structure differences:
     *
     *    For Qwen2VL (uses M-RoPE with 2D positions):
     *    ┌─────────────────────────────────────────┐
     *    │ MEDIA_CHUNK                             │
     *    │ ┌───────────────────────────────────┐   │
     *    │ │ mtmd_image_tokens:                │   │
     *    │ │  nx = 16, ny = 16                 │   │ ← 2D grid (16×16 = 256 tokens)
     *    │ │  use_mrope_pos = true             │   │ ← Uses M-RoPE positioning
     *    │ │  batch_f32 = [image_embeddings]   │   │
     *    │ └───────────────────────────────────┘   │
     *    └─────────────────────────────────────────┘
     *
     *    For other models (uses 1D positions):
     *    ┌─────────────────────────────────────────┐
     *    │ MEDIA_CHUNK                             │
     *    │ ┌───────────────────────────────────┐   │
     *    │ │ mtmd_image_tokens:                │   │
     *    │ │  nx = 256, ny = 1                 │   │ ← 1D sequence (256 tokens)
     *    │ │  use_mrope_pos = false            │   │ ← Uses standard positioning
     *    │ │  batch_f32 = [image_embeddings]   │   │
     *    │ └───────────────────────────────────┘   │
     *    └─────────────────────────────────────────┘
     *
     * 5. Final chunks array:
     *    chunks[0] = TEXT_CHUNK([1234, 5678])
     *    chunks[1] = MEDIA_CHUNK(first_image)
     *    chunks[2] = TEXT_CHUNK([9012])
     *    chunks[3] = MEDIA_CHUNK(second_image)
     */
    LOG_INFO("[DEBUG] Tokenizing text and %zu media", bitmaps.entries.size());
    auto bitmaps_c_ptr = bitmaps.c_ptr();
    int32_t res = mtmd_tokenize(mtmd_wrapper->mtmd_ctx, result.chunks, &input_text, bitmaps_c_ptr.data(), bitmaps_c_ptr.size());
    if (res != 0) {
        mtmd_input_chunks_free(result.chunks);
        bitmaps.entries.clear();
        throw std::runtime_error("Failed to tokenize text and media");
    }

    // Log chunk information
    size_t num_chunks = mtmd_input_chunks_size(result.chunks);
    LOG_INFO("[DEBUG] Tokenization successful: num_chunks=%zu", num_chunks);

    // Track the total number of tokens (both text and image)
    size_t total_token_count = 0;

    /**
     * Evaluate the chunks.
     *
     * For our example "foo bar <__media__> baz <__media__>":
     *
     * Token organization in memory:
     *
     *    all_tokens: [t0][t1][NULL][NULL]...[NULL][t2][NULL][NULL]...[NULL]
     *    positions:   0   1    2    3   ...  257   258  259  260 ...  514
     *    chunk_pos:   0        2                   258  259
     *
     *    Where:
     *    - [t0][t1] are text tokens for "foo bar " (positions 0-1)
     *    - [NULL]x256 are placeholder tokens for the first image (positions 2-257)
     *    - [t2] is the text token for " baz " (position 258)
     *    - [NULL]x256 are placeholder tokens for the second image (positions 259-514)
     */
    for (size_t i = 0; i < num_chunks; i++) {
        result.chunk_pos.push_back(total_token_count);

        const mtmd_input_chunk* chunk = mtmd_input_chunks_get(result.chunks, i);
        mtmd_input_chunk_type chunk_type = mtmd_input_chunk_get_type(chunk);

        if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t n_tokens;
            const llama_token* tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);
            LOG_INFO("[DEBUG] Chunk %zu: type=TEXT, n_tokens=%zu", i, n_tokens);

            // Add text tokens
            result.tokens.insert(result.tokens.end(), tokens, tokens + n_tokens);
            total_token_count += n_tokens;
        } else if (chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE || chunk_type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
            result.chunk_pos_media.push_back(total_token_count);

            size_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk);
            size_t n_pos = mtmd_input_chunk_get_n_pos(chunk);
            LOG_INFO("[DEBUG] Chunk %zu: type=%s, n_tokens=%zu, n_pos=%zu",
                     i, chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE ? "IMAGE" : "AUDIO", n_tokens, n_pos);

            for (size_t j = 0; j < n_pos; j++) {
                result.tokens.push_back(LLAMA_TOKEN_NULL); // Placeholder token
            }
            total_token_count += n_pos;
        }
    }

    bitmaps.entries.clear();

    return result;
}
void llama_rn_context::processMedia(
    const std::string &prompt,
    const std::vector<std::string> &media_paths
) {
    if (!isMultimodalEnabled()) {
        throw std::runtime_error("Multimodal is not enabled but image paths are provided");
    }

    // Multimodal path
    std::string full_prompt = prompt;
    auto default_media_marker = mtmd_default_marker();
    // Add media marker if it doesn't already exist
    if (full_prompt.find(default_media_marker) == std::string::npos) {
        full_prompt += " ";
        full_prompt += default_media_marker;
    }

    LOG_INFO("[DEBUG] Processing message with role=user, content=%s", full_prompt.c_str());
    LOG_INFO("[DEBUG] Processing %zu media with prompt: %s", media_paths.size(), prompt.c_str());
    LOG_INFO("[DEBUG] Current context state: n_past=%d, n_ctx=%d", n_past, n_ctx);

    auto result = tokenizeWithMedia(mtmd_wrapper, full_prompt, media_paths);

    auto all_tokens = result.tokens;
    auto chunks = result.chunks;
    auto chunk_pos = result.chunk_pos;
    auto chunk_pos_media = result.chunk_pos_media;
    auto bitmap_hashes = result.bitmap_hashes;

    // Check if we have enough context space for all tokens
    if (all_tokens.size() >= (size_t)n_ctx) {
        mtmd_input_chunks_free(chunks);
        context_full = true;
        throw std::runtime_error("Not enough context space");
    }

    n_past = common_part(embd, all_tokens);

    llama_pos new_n_past = n_past;

    // Adjust n_past to position of the text chunk
    // TODO: Edit the text chunk to remove the tokens before n_past to speed up
    // need to update the mtmd api
    auto adjusted_n_past = -1;
    for (size_t i = 0; i < chunk_pos.size(); i++) {
        if (n_past < chunk_pos[i]) {
            break;
        }
        bool is_end = i + 1 == chunk_pos.size();
        if (
            chunk_pos[i] < n_past &&
            (!is_end && chunk_pos[i + 1] > n_past)
            // is_end & n_past < total_token_count:
            // don't need to adjust and it will skip eval_chunk_single, let nextToken() to finish the job
        ) {
            adjusted_n_past = chunk_pos[i];
        }
    }
    if (adjusted_n_past != -1) {
        n_past = adjusted_n_past;
        new_n_past = n_past;
        LOG_INFO("[DEBUG] Adjusted n_past to %d", n_past);
    }

    // Compare bitmap hashes, if they are not the same, backtrack n_past to the position of the first mismatch
    if (mtmd_bitmap_past_hashes.size() > 0) {
        for (size_t i = 0; i < bitmap_hashes.size(); i++) {
            auto pos = chunk_pos_media[i];
            if (n_past < pos) {
                break;
            }
            if (i >= mtmd_bitmap_past_hashes.size()) {
                break;
            }
            if (bitmap_hashes[i] != mtmd_bitmap_past_hashes[i]) {
                LOG_INFO(
                    "[DEBUG] Bitmap hash mismatch at position %zu, %s != %s",
                    i, bitmap_hashes[i].c_str(), mtmd_bitmap_past_hashes[i].c_str()
                );
                n_past = chunk_pos_media[i];
                new_n_past = n_past;
                break;
            }
        }
    }

    // Clear all KV cache entries after position n_past
    llama_kv_self_seq_rm(ctx, 0, n_past, -1);

    LOG_INFO("[DEBUG] Evaluating chunks: n_past=%d, n_batch=%d", n_past, params.n_batch);

    size_t num_chunks = mtmd_input_chunks_size(chunks);

    for (size_t i = 0; i < chunk_pos.size(); i++) {

        LOG_INFO("[DEBUG] Evaluating chunk %zu: n_past=%d, chunk_pos=%zu", i, n_past, chunk_pos[i]);

        // Process chunk only if it's after the current n_past
        if (chunk_pos[i] >= n_past) {
            bool chunk_logits_last = (i == num_chunks - 1);
            auto chunk = mtmd_input_chunks_get(chunks, i);

            int32_t res = mtmd_helper_eval_chunk_single(
                mtmd_wrapper->mtmd_ctx,
                ctx,
                chunk,
                n_past,
                0,
                params.n_batch,
                chunk_logits_last,
                &new_n_past
            );
            if (res != 0) {
                mtmd_input_chunks_free(chunks);
                throw std::runtime_error("Failed to evaluate chunks");
            }
            n_past = new_n_past;
        }
    }

    if (n_past == all_tokens.size() && n_past > 0 && all_tokens[n_past - 1] != LLAMA_TOKEN_NULL) {
        // we have to evaluate at least 1 token to generate logits.
        n_past--;
    }

    // Update embd with all tokens (both text and media)
    embd = all_tokens;

    mtmd_bitmap_past_hashes = bitmap_hashes;

    // Update sampling context with text tokens only
    for (auto & token : all_tokens) {
        if (token == LLAMA_TOKEN_NULL) {
            continue;
        }
        common_sampler_accept(ctx_sampling, token, false);
    }

    // Clean up media resources
    LOG_INFO("[DEBUG] Cleaning up resources");
    mtmd_input_chunks_free(chunks);
}

llama_rn_tokenize_result llama_rn_context::tokenize(const std::string &text, const std::vector<std::string> &media_paths) {
    if (media_paths.size() > 0) {
        if (!isMultimodalEnabled()) {
            throw std::runtime_error("Multimodal is not enabled but media paths are provided");
        }
        auto result = tokenizeWithMedia(mtmd_wrapper, text, media_paths);
        mtmd_input_chunks_free(result.chunks);
        llama_rn_tokenize_result tokenize_result = {
            .tokens = result.tokens,
            .has_media = true,
            .bitmap_hashes = result.bitmap_hashes,
            .chunk_pos = result.chunk_pos,
            .chunk_pos_media = result.chunk_pos_media,
        };
        return tokenize_result;
    }
    std::vector<llama_token> text_tokens;
    text_tokens = common_tokenize(ctx, text, false);
    llama_rn_tokenize_result tokenize_result = {
        .tokens = text_tokens,
        .has_media = false,
        .bitmap_hashes = {},
        .chunk_pos = {},
        .chunk_pos_media = {},
    };
    return tokenize_result;
}

bool llama_rn_context::isMultimodalEnabled() const {
    return has_multimodal && mtmd_wrapper != nullptr;
}

bool llama_rn_context::isMultimodalSupportVision() const {
    return isMultimodalEnabled() && mtmd_support_vision(mtmd_wrapper->mtmd_ctx);
}

bool llama_rn_context::isMultimodalSupportAudio() const {
    return isMultimodalEnabled() && mtmd_support_audio(mtmd_wrapper->mtmd_ctx);
}

void llama_rn_context::releaseMultimodal() {
    if (mtmd_wrapper && mtmd_wrapper->mtmd_ctx != nullptr) {
        mtmd_free(mtmd_wrapper->mtmd_ctx);
        mtmd_wrapper->mtmd_ctx = nullptr;
        delete mtmd_wrapper;
        mtmd_wrapper = nullptr;
        has_multimodal = false;
    }
}

struct llama_rn_context_vocoder {
    common_init_result init_result;
    llama_model *model = nullptr;
    llama_context *ctx = nullptr;
    tts_type type = UNKNOWN;
};

bool llama_rn_context::initVocoder(const std::string &vocoder_model_path) {
    if (vocoder_wrapper != nullptr) {
        return true;
    }
    params.model.path = vocoder_model_path;
    params.embedding = true;
    params.ctx_shift = false;
    params.n_ubatch = params.n_batch;

    llama_rn_context_vocoder *wrapper = new llama_rn_context_vocoder{
        .init_result = common_init_from_params(params),
    };

    wrapper->model = wrapper->init_result.model.get();
    wrapper->ctx = wrapper->init_result.context.get();

    if (wrapper->model == nullptr || wrapper->ctx == nullptr) {
        LOG_ERROR("Failed to load vocoder model: %s", vocoder_model_path.c_str());
        delete wrapper;
        return false;
    }

    wrapper->type = getTTSType();
    vocoder_wrapper = wrapper;
    has_vocoder = true;
    return true;
}

bool llama_rn_context::isVocoderEnabled() const {
    return has_vocoder && vocoder_wrapper != nullptr;
}

void llama_rn_context::releaseVocoder() {
    if (vocoder_wrapper != nullptr) {
        delete vocoder_wrapper;
        vocoder_wrapper = nullptr;
    }
    has_vocoder = false;
}

tts_type llama_rn_context::getTTSType(json speaker) {
    if (vocoder_wrapper == nullptr) {
        return UNKNOWN;
    }
    if (speaker.is_object() && speaker.contains("version")) {
        std::string version = speaker["version"].get<std::string>();
        if (version == "0.2") {
            return OUTETTS_V0_2;
        } else if (version == "0.3") {
            return OUTETTS_V0_3;
        } else {
            LOG_ERROR("Unsupported speaker version '%s'\n", version.c_str());
        }
    }
    if (vocoder_wrapper->type != UNKNOWN) {
        return vocoder_wrapper->type;
    }
    const char *chat_template = llama_model_chat_template(model, nullptr);
    if (chat_template && std::string(chat_template) == "outetts-0.3") {
        return OUTETTS_V0_3;
    }
    return OUTETTS_V0_2;
}

static std::string audio_text_from_speaker(json speaker, const tts_type type = OUTETTS_V0_2) {
    std::string audio_text = "<|text_start|>";

    if (type == OUTETTS_V0_2 || type == OUTETTS_V0_3) {
        std::string separator = (type == OUTETTS_V0_3) ? "<|space|>" : "<|text_sep|>";
        for (const auto &word : speaker["words"]) {
            audio_text += word["word"].get<std::string>() + separator;
        }
    }

    return audio_text;
}

static std::string audio_data_from_speaker(json speaker, const tts_type type = OUTETTS_V0_2) {
    std::string audio_data = "<|audio_start|>\n";

    if (type == OUTETTS_V0_2 || type == OUTETTS_V0_3) {
        std::string code_start = (type == OUTETTS_V0_3) ? "" : "<|code_start|>";
        std::string code_end = (type == OUTETTS_V0_3) ? "<|space|>" : "<|code_end|>";
        for (const auto &word : speaker["words"]) {
            std::string word_text = word["word"].get<std::string>();
            double duration = word["duration"].get<double>();
            std::vector<int> codes = word["codes"].get<std::vector<int>>();

            // Create the audio output entry
            std::ostringstream word_entry;
            word_entry << word_text << "<|t_" << std::fixed << std::setprecision(2)
                       << duration << "|>" + code_start;
            for (const auto &Code : codes) {
                word_entry << "<|" << Code << "|>";
            }
            word_entry << code_end << "\n";
            audio_data += word_entry.str();
        }
    }

    return audio_data;
}

static const std::map<int, std::string> ones = {
    {0, "zero"}, {1, "one"}, {2, "two"}, {3, "three"}, {4, "four"},
    {5, "five"}, {6, "six"}, {7, "seven"}, {8, "eight"}, {9, "nine"},
    {10, "ten"}, {11, "eleven"}, {12, "twelve"}, {13, "thirteen"}, {14, "fourteen"},
    {15, "fifteen"}, {16, "sixteen"}, {17, "seventeen"}, {18, "eighteen"}, {19, "nineteen"}
};

static const std::map<int, std::string> tens = {
    {2, "twenty"}, {3, "thirty"}, {4, "forty"}, {5, "fifty"},
    {6, "sixty"}, {7, "seventy"}, {8, "eighty"}, {9, "ninety"}
};

// Convert a number less than 1000 to words
static std::string convert_less_than_thousand(int num) {
    std::string result;

    if (num >= 100) {
        result += ones.at(num / 100) + " hundred ";
        num %= 100;
    }

    if (num >= 20) {
        result += tens.at(num / 10);
        if (num % 10 > 0) {
            result += "-" + ones.at(num % 10);
        }
    } else if (num > 0) {
        result += ones.at(num);
    }

    return result;
}

static std::string number_to_words(const std::string & number_str) {
    try {
        size_t decimal_pos = number_str.find('.');
        std::string integer_part = number_str.substr(0, decimal_pos);

        int int_number = std::stoi(integer_part);
        std::string result;

        if (int_number == 0) {
            result = "zero";
        } else {
            if (int_number >= 1000000000) {
                int billions = int_number / 1000000000;
                result += convert_less_than_thousand(billions) + " billion ";
                int_number %= 1000000000;
            }

            if (int_number >= 1000000) {
                int millions = int_number / 1000000;
                result += convert_less_than_thousand(millions) + " million ";
                int_number %= 1000000;
            }

            if (int_number >= 1000) {
                int thousands = int_number / 1000;
                result += convert_less_than_thousand(thousands) + " thousand ";
                int_number %= 1000;
            }

            if (int_number > 0) {
                result += convert_less_than_thousand(int_number);
            }
        }

        // Handle decimal part
        if (decimal_pos != std::string::npos) {
            result += " point";
            std::string decimal_part = number_str.substr(decimal_pos + 1);
            for (char digit : decimal_part) {
                result += " " + ones.at(digit - '0');
            }
        }

        return result;
    } catch (const std::exception& e) {
        // Skip if fails
        return " ";
    }
}

static std::string replace_numbers_with_words(const std::string & input_text) {
    std::regex number_pattern(R"(\d+(\.\d+)?)");
    std::string result;
    auto it = std::sregex_iterator(input_text.begin(), input_text.end(), number_pattern);
    auto end = std::sregex_iterator();

    size_t last_pos = 0;
    for (std::sregex_iterator i = it; i != end; ++i) {
        const std::smatch& match = *i;
        result.append(input_text, last_pos, match.position() - last_pos);
        result.append(number_to_words(match.str()));
        last_pos = match.position() + match.length();
    }
    result.append(input_text, last_pos);

    return result;
}

// Based on: https://github.com/edwko/OuteTTS/blob/a613e79c489d8256dd657ea9168d78de75895d82/outetts/version/v1/prompt_processor.py#L39
static std::string process_text(const std::string & text, const tts_type tts_type = OUTETTS_V0_2) {

    // For now I skipped text romanization as I am unsure how to handle
    // uroman and MeCab implementations in C++
    // maybe something like https://github.com/anyascii/anyascii/ could work.
    // currently only English would be supported in this function

    std::string processed_text = replace_numbers_with_words(text);

    std::transform(processed_text.begin(), processed_text.end(),
                  processed_text.begin(), ::tolower);

    std::regex special_chars(R"([-_/,\.\\])");
    processed_text = std::regex_replace(processed_text, special_chars, " ");

    std::regex non_alpha(R"([^a-z\s])");
    processed_text = std::regex_replace(processed_text, non_alpha, "");

    std::regex multiple_spaces(R"(\s+)");
    processed_text = std::regex_replace(processed_text, multiple_spaces, " ");

    processed_text = std::regex_replace(processed_text, std::regex(R"(^\s+|\s+$)"), "");

    /*
        Replace spaces with the separator token same as in line 365

        for (auto & c : prompt_user) {
        if (c == ' ') {
            prompt_clean += "<|text_sep|>";
    */
    std::string separator = (tts_type == OUTETTS_V0_3) ? "<|space|>" : "<|text_sep|>";
    processed_text = std::regex_replace(processed_text, std::regex(R"(\s)"), separator);

    return processed_text;
}

std::string llama_rn_context::getFormattedAudioCompletion(const std::string &speaker_json_str, const std::string &text_to_speak) {
    if (!isVocoderEnabled()) {
        throw std::runtime_error("Vocoder is not enabled but audio completion is requested");
    }
    std::string audio_text = default_audio_text;
    std::string audio_data = default_audio_data;

    json speaker = speaker_json_str.empty() ? json::object() : json::parse(speaker_json_str);
    const tts_type type = getTTSType(speaker);
    if (type == UNKNOWN) {
        LOG_ERROR("Unknown TTS version");
        return "";
    }

    if (type == OUTETTS_V0_3) {
        audio_text = std::regex_replace(audio_text, std::regex(R"(<\|text_sep\|>)"), "<|space|>");
        audio_data = std::regex_replace(audio_data, std::regex(R"(<\|code_start\|>)"), "");
        audio_data = std::regex_replace(audio_data, std::regex(R"(<\|code_end\|>)"), "<|space|>");
    }

    if (!speaker_json_str.empty()) {
        audio_text = audio_text_from_speaker(speaker, type);
        audio_data = audio_data_from_speaker(speaker, type);
    }

    return "<|im_start|>\n" + audio_text + process_text(text_to_speak, type) + "<|text_end|>\n" + audio_data + "\n";
}

std::vector<llama_token> llama_rn_context::getAudioCompletionGuideTokens(const std::string &text_to_speak) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const tts_type type = getTTSType();
    std::string clean_text = process_text(text_to_speak, type);

    const std::string& delimiter = (type == OUTETTS_V0_3 ? "<|space|>" : "<|text_sep|>");

    std::vector<llama_token> result;
    size_t start = 0;
    size_t end = clean_text.find(delimiter);

    //first token is always a newline, as it was not previously added
    result.push_back(common_tokenize(vocab, "\n", false, true)[0]);

    while (end != std::string::npos) {
        std::string current_word = clean_text.substr(start, end - start);
        auto tmp = common_tokenize(vocab, current_word, false, true);
        result.push_back(tmp[0]);
        start = end + delimiter.length();
        end = clean_text.find(delimiter, start);
    }

    // Add the last part
    std::string current_word = clean_text.substr(start);
    auto tmp = common_tokenize(vocab, current_word, false, true);
    if (tmp.size() > 0) {
        result.push_back(tmp[0]);
    }
    return result;
}

static void fill_hann_window(int length, bool periodic, float * output) {
    int offset = -1;
    if (periodic) {
        offset = 0;
    }
    for (int i = 0; i < length; i++) {
        output[i] = 0.5 * (1.0 - cosf((2.0 * M_PI * i) / (length + offset)));
    }
}

static void twiddle(float * real, float * imag, int k, int N) {
    float angle = 2 * M_PI * k / N;
    *real = cos(angle);
    *imag = sin(angle);
}

static void irfft(int n, const float * inp_cplx, float * out_real) {
    int N = n / 2 + 1;

    std::vector<float> real_input(N);
    std::vector<float> imag_input(N);
    for (int i = 0; i < N; ++i) {
        real_input[i] = inp_cplx[2 * i];
        imag_input[i] = inp_cplx[2 * i + 1];
    }

    std::vector<float> real_output(n);
    std::vector<float> imag_output(n);

    for (int k = 0; k < n; ++k) {
        real_output[k] = 0.0f;
        imag_output[k] = 0.0f;
        for (int m = 0; m < N; ++m) {
            float twiddle_real;
            float twiddle_imag;

            twiddle(&twiddle_real, &twiddle_imag, k * m, n);

            real_output[k] += real_input[m] * twiddle_real - imag_input[m] * twiddle_imag;
            imag_output[k] += real_input[m] * twiddle_imag + imag_input[m] * twiddle_real;
        }
    }

    for (int i = 0; i < n; ++i) {
        out_real[i] = real_output[i] / N;
    }
}

static void fold(const std::vector<float> & data, int64_t n_out, int64_t n_win, int64_t n_hop, int64_t n_pad, std::vector<float> & output) {
    int64_t output_height = n_out;
    int64_t kernel_w = n_win;
    int64_t stride_w = n_hop;
    int64_t width    = n_out;

    output.resize(width, 0.0f);

    int64_t col_idx = 0;
    for (int64_t w_col = 0; w_col < width; ++w_col) {
        int64_t start = w_col * stride_w - n_pad;
        int64_t end   = start + kernel_w;

        for (int64_t w_im = start; w_im < end; ++w_im) {
            if (w_im >= 0 && w_im < output_height && col_idx < (int64_t) data.size()) {
                output[w_im] += data[col_idx];
            }
            col_idx++;
        }
    }

    output.resize(n_out - 2 * n_pad);
}

static std::vector<float> embd_to_audio(
        const float * embd,
        const int n_codes,
        const int n_embd,
        const int n_thread) {
    const int n_fft = 1280;
    const int n_hop = 320;
    const int n_win = 1280;
    const int n_pad = (n_win - n_hop)/2;
    const int n_out = (n_codes - 1)*n_hop + n_win;

    std::vector<float> hann(n_fft);

    fill_hann_window(hann.size(), true, hann.data());

    int n_spec = n_embd*n_codes;

    std::vector<float> E (n_spec);
    std::vector<float> S (n_spec);
    std::vector<float> ST(n_spec);

    for (int l = 0; l < n_codes; ++l) {
        for (int k = 0; k < n_embd; ++k) {
            E[k*n_codes + l] = embd[l*n_embd + k];
        }
    }

    for (int k = 0; k < n_embd/2; ++k) {
        for (int l = 0; l < n_codes; ++l) {
            float mag = E[(k           )*n_codes + l];
            float phi = E[(k + n_embd/2)*n_codes + l];

            mag = exp(mag);

            if (mag > 1e2) {
                mag = 1e2;
            }
            S[2*(k*n_codes + l) + 0] = mag*cosf(phi);
            S[2*(k*n_codes + l) + 1] = mag*sinf(phi);
        }
    }

    for (int l = 0; l < n_codes; ++l) {
        for (int k = 0; k < n_embd/2; ++k) {
            ST[l*n_embd + 2*k + 0] = S[2*(k*n_codes + l) + 0];
            ST[l*n_embd + 2*k + 1] = S[2*(k*n_codes + l) + 1];
        }
    }

    std::vector<float> res  (n_codes*n_fft);
    std::vector<float> hann2(n_codes*n_fft);

    std::vector<std::thread> workers(n_thread);
    for (int i = 0; i < n_thread; ++i) {
        workers[i] = std::thread([&, i]() {
            for (int l = i; l < n_codes; l += n_thread) {
                irfft(n_fft, ST.data() + l*n_embd, res.data() + l*n_fft);
                for (int j = 0; j < n_fft; ++j) {
                    res  [l*n_fft + j] *= hann[j];
                    hann2[l*n_fft + j]  = hann[j] * hann[j];
                }
            }
        });
    }
    for (int i = 0; i < n_thread; ++i) {
        workers[i].join();
    }

    std::vector<float> audio;
    std::vector<float> env;

    fold(res,   n_out, n_win, n_hop, n_pad, audio);
    fold(hann2, n_out, n_win, n_hop, n_pad, env); // TODO: can be done once

    for (size_t i = 0; i < audio.size(); ++i) {
        audio[i] /= env[i];
    }

    return audio;
}

std::vector<float> llama_rn_context::decodeAudioTokens(const std::vector<llama_token> &tokens) {
    if (!isVocoderEnabled()) {
        throw std::runtime_error("Vocoder is not enabled but audio completion is requested");
    }
    std::vector<llama_token> tokens_audio = tokens;
    tts_type type = getTTSType();
    if (type == OUTETTS_V0_3 || type == OUTETTS_V0_2) {
        tokens_audio.erase(std::remove_if(tokens_audio.begin(), tokens_audio.end(), [](llama_token t) { return t < 151672 || t > 155772; }), tokens_audio.end());
        for (auto & token : tokens_audio) {
            token -= 151672;
        }
    } else {
        LOG_ERROR("Unsupported audio tokens");
        return std::vector<float>();
    }
    const int n_codes = tokens_audio.size();
    llama_batch batch = llama_batch_init(n_codes, 0, 1);
    for (size_t i = 0; i < tokens_audio.size(); ++i) {
        llama_batch_add(&batch, tokens_audio[i], i, { 0 }, true);
    }
    if (batch.n_tokens != n_codes) {
        LOG_ERROR("batch.n_tokens != n_codes: %d != %d", batch.n_tokens, n_codes);
        return std::vector<float>();
    }
    if (llama_encode(vocoder_wrapper->ctx, batch) != 0) {
        LOG_ERROR("llama_encode() failed");
        return std::vector<float>();
    }
    llama_synchronize(vocoder_wrapper->ctx);
    const int n_embd = llama_model_n_embd(vocoder_wrapper->model);
    const float * embd = llama_get_embeddings(vocoder_wrapper->ctx);
    return embd_to_audio(embd, n_codes, n_embd, params.cpuparams.n_threads);
}

}
