#include "cactus.h"
#include "common.h"
#include <algorithm>
#include <vector>
#include <string>
#include <sstream>
#include "llama.h"

namespace cactus {

void cactus_context::truncatePrompt(std::vector<llama_token> &prompt_tokens) {
    const int n_left = n_ctx - params.n_keep;
    const int n_block_size = (n_left > 0) ? n_left / 2 : 0;
    const int erased_blocks = (n_block_size > 0) ? (prompt_tokens.size() - params.n_keep - n_block_size) / n_block_size : 0;

    int keep_count = std::max(0, params.n_keep);
    keep_count = std::min(keep_count, (int)prompt_tokens.size());

    std::vector<llama_token> new_tokens(prompt_tokens.begin(), prompt_tokens.begin() + keep_count);

    size_t end_part_start_index = (size_t)keep_count + (size_t)std::max(0, erased_blocks * n_block_size);
    if (end_part_start_index < prompt_tokens.size()) {
        new_tokens.insert(new_tokens.end(), prompt_tokens.begin() + end_part_start_index, prompt_tokens.end());
    }

    LOG_VERBOSE("input truncated, n_ctx: %d, n_keep: %d, n_left: %d, new_tokens_size: %zu",
        n_ctx,
        params.n_keep,
        n_left,
        new_tokens.size()
    );

    truncated = true;
    prompt_tokens = new_tokens;
}

void cactus_context::loadPrompt() {
    std::vector<llama_token> prompt_tokens = ::common_tokenize(ctx, params.prompt, true, true);
    num_prompt_tokens = prompt_tokens.size();

    std::stringstream ss;
    ss << "\n" << __func__ << ": prompt_tokens = ";
    for (auto& token : prompt_tokens) {
        ss << token << " ";
    }
    LOG_INFO("%s\n", ss.str().c_str());

    if (params.n_keep < 0)
    {
        params.n_keep = (int)num_prompt_tokens;
    }
    params.n_keep = std::min(n_ctx > 4 ? n_ctx - 4 : 0, params.n_keep);
    params.n_keep = std::max(0, params.n_keep);

    if (num_prompt_tokens >= (size_t) n_ctx)
    {
        truncatePrompt(prompt_tokens);
        num_prompt_tokens = prompt_tokens.size();

        LM_GGML_ASSERT(num_prompt_tokens < (size_t) n_ctx || n_ctx == 0);
    }
    for (auto & token : prompt_tokens)
    {
        common_sampler_accept(ctx_sampling, token, false);
    }

    n_past = common_part(embd, prompt_tokens);

    embd = prompt_tokens;
    n_past = std::min(n_past, embd.size());

    if (n_past == num_prompt_tokens && n_past > 0)
    {
        n_past--;
    }

    if (n_past > 0) {
         llama_kv_self_seq_rm(ctx, 0, n_past, -1);
    }

    LOG_VERBOSE("prompt ingested, n_past: %d, cached_size: %zu, to_eval_size: %zu",
        n_past,
        (size_t)n_past,
        embd.size() - n_past
    );

    has_next_token = true;
}

void cactus_context::loadPrompt(const std::vector<std::string> &media_paths) {
    bool has_media = !media_paths.empty();

    if (!has_media) {
        loadPrompt();
        return;
    }

    if (!isMultimodalEnabled()) {
        throw std::runtime_error("Multimodal is not enabled but media paths are provided");
    }

    processMedia(params.prompt, media_paths);
    num_prompt_tokens = embd.size();

    if (params.n_keep < 0) {
        params.n_keep = (int)num_prompt_tokens;
    }
    params.n_keep = std::min(n_ctx > 4 ? n_ctx - 4 : 0, params.n_keep);
    params.n_keep = std::max(0, params.n_keep);

    if (n_past == num_prompt_tokens && n_past > 0) {
        n_past--;
    }

    has_next_token = true;
    LOG_VERBOSE("Input processed: n_past=%d, embd.size=%zu, num_prompt_tokens=%zu, has_media=%d",
             n_past, embd.size(), num_prompt_tokens, has_media ? 1 : 0);
}

void cactus_context::beginCompletion() {
    n_remain = params.n_predict;
    llama_perf_context_reset(ctx);
    is_predicting = true;
}

completion_token_output cactus_context::nextToken()
{
    completion_token_output result;
    result.tok = -1;

    if (embd.size() >= (size_t)params.n_ctx)
    {
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

        LOG_VERBOSE("context shifted, new n_past: %d, new size: %zu", n_past, embd.size());
    }

    bool tg = true;
    while ((size_t)n_past < embd.size())
    {
        int n_eval = (int)embd.size() - n_past;
        tg = n_eval == 1;
        if (n_eval > params.n_batch)
        {
            n_eval = params.n_batch;
        }

        if (n_eval <= 0) {
            LOG_WARNING("No tokens to evaluate (n_eval=%d)", n_eval);
            break;
        }

        if (llama_decode(ctx, llama_batch_get_one(&embd[n_past], n_eval)) != 0)
        {
            LOG_ERROR("failed to eval, n_eval: %d, n_past: %d, n_threads: %d, embd_size: %zu",
                n_eval,
                n_past,
                params.cpuparams.n_threads,
                embd.size()
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

    if (!model) {
        LOG_ERROR("Model is null in nextToken");
        has_next_token = false;
        return result;
    }
    const llama_vocab* vocab = llama_model_get_vocab(model);
    if (!vocab) {
         LOG_ERROR("Vocab is null in nextToken");
         has_next_token = false;
         return result;
    }

    if (params.n_predict == 0)
    {
        has_next_token = false;
        result.tok = llama_vocab_eos(vocab);
        return result;
    }

    {
        std::vector<llama_token_data> candidates;
        candidates.reserve(llama_vocab_n_tokens(vocab));

        llama_token new_token_id = common_sampler_sample(ctx_sampling, ctx, -1);

        if (next_token_uses_guide_token && !guide_tokens.empty() && 
            !llama_vocab_is_control(vocab, new_token_id) && 
            !llama_vocab_is_eog(vocab, new_token_id)) {
            new_token_id = guide_tokens[0];
            guide_tokens.erase(guide_tokens.begin());
        }
        next_token_uses_guide_token = (new_token_id == 198);
        result.tok = new_token_id;

        llama_token_data_array cur_p = *common_sampler_get_candidates(ctx_sampling);

        const int32_t n_probs = params.sampling.n_probs;
        const size_t vocab_size = llama_vocab_n_tokens(vocab);

        for (size_t i = 0; i < std::min((size_t)cur_p.size, (size_t)n_probs); ++i)
        {
            if (cur_p.data[i].id < vocab_size) {
                 result.probs.push_back({cur_p.data[i].id, cur_p.data[i].p});
            }
        }

        common_sampler_accept(ctx_sampling, result.tok, true);
        if (tg) {
            num_tokens_predicted++;
        }
    }

    embd.push_back(result.tok);
    if (n_remain > 0) {
        --n_remain;
    }

    if (!embd.empty() && embd.back() == llama_vocab_eos(vocab))
    {
        has_next_token = false;
        stopped_eos = true;
        LOG_VERBOSE("eos token found", "");
        return result;
    }

    has_next_token = params.n_predict == -1 || n_remain > 0;
    return result;
}

size_t cactus_context::findStoppingStrings(const std::string &text, const size_t last_token_size,
                            const stop_type type)
{
    size_t stop_pos = std::string::npos;

    for (const std::string &word : params.antiprompt)
    {
        if (word.empty()) continue;

        size_t pos;
        if (type == STOP_FULL)
        {
            size_t from_pos = 0;
            size_t tmp_len = word.size() + last_token_size;
            if (text.size() > tmp_len) {
                from_pos = text.size() - tmp_len;
            }
            pos = text.find(word, from_pos);
        }
        else
        {
             pos = cactus::find_partial_stop_string(word, text);
        }

        if (pos != std::string::npos)
        {
            if (stop_pos == std::string::npos || pos < stop_pos)
            {
                stop_pos = pos;
                if (type == STOP_FULL)
                {
                    stopping_word = word;
                    stopped_word = true;
                    has_next_token = false;
                }
            }
        }
    }
        return stop_pos;
}

completion_token_output cactus_context::doCompletion()
{
    const completion_token_output token_with_probs = nextToken();

    if (token_with_probs.tok == -1 && !has_next_token) {
        return token_with_probs;
    }
    
    std::string token_text;
    if (ctx && token_with_probs.tok != -1) {
         token_text = common_token_to_piece(ctx, token_with_probs.tok);
    }
    generated_text += token_text;

    if (isVocoderEnabled()) {
        tts_type type = getTTSType();
        if ((type == TTS_OUTETTS_V0_2 || type == TTS_OUTETTS_V0_3) && 
            (token_with_probs.tok >= 151672 && token_with_probs.tok <= 155772)) {
            audio_tokens.push_back(token_with_probs.tok);
        }
    }

    if (params.sampling.n_probs > 0)
    {
        generated_token_probs.push_back(token_with_probs);
    }

    incomplete = false;
    if (!generated_text.empty()) {
         unsigned char c = generated_text.back();
         int expected_continuation_bytes = 0;
         if ((c & 0xC0) == 0x80) {
             int lookback = 1;
             while (lookback < 4 && lookback < generated_text.size()) {
                 unsigned char prev_c = generated_text[generated_text.size() - 1 - lookback];
                 if ((prev_c & 0xC0) == 0xC0) {
                     if      ((prev_c & 0xE0) == 0xC0) expected_continuation_bytes = 1;
                     else if ((prev_c & 0xF0) == 0xE0) expected_continuation_bytes = 2;
                     else if ((prev_c & 0xF8) == 0xF0) expected_continuation_bytes = 3;
                     incomplete = lookback < expected_continuation_bytes;
                     break;
                 } else if ((prev_c & 0x80) == 0x00) {
                     break;
                 }
                 lookback++;
             }
         } else if ((c & 0xE0) == 0xC0) {
            incomplete = true;
         } else if ((c & 0xF0) == 0xE0) {
             incomplete = true;
         } else if ((c & 0xF8) == 0xF0) {
             incomplete = true;
         }
    }

    if (incomplete && !has_next_token)
    {
        has_next_token = true;
        if (params.n_predict != -1) {
            n_remain++;
        }
    }

    if (!has_next_token && n_remain == 0 && params.n_predict != -1)
    {
        stopped_limit = true;
    }

    LOG_VERBOSE("next token, token_id: %d, token_text: %s, has_next_token: %d, n_remain: %d, incomplete: %d, num_tokens_predicted: %d, stopped_eos: %d, stopped_word: %d, stopped_limit: %d, stopping_word: %s",
        token_with_probs.tok,
        tokens_to_output_formatted_string(ctx, token_with_probs.tok).c_str(),
        has_next_token,
        n_remain,
        incomplete,
        num_tokens_predicted,
        stopped_eos,
        stopped_word,
        stopped_limit,
        stopping_word.c_str()
    );
    return token_with_probs;
}

} // namespace cactus