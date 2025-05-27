#include "ggml.h"   
#include "cactus.h"
#include "common.h" 
#include "mtmd.h"  
#include <algorithm>
#include <vector>
#include <string>
#include <sstream> 
#include "llama.h" 

namespace cactus {

/**
 * @brief Truncates a prompt if it's too long for the context
 * 
 * @param prompt_tokens Tokens to truncate
 */
void cactus_context::truncatePrompt(std::vector<llama_token> &prompt_tokens) {
    const int n_left = n_ctx - params.n_keep;

    // Handle case where n_left might be zero or negative if n_keep >= n_ctx
    const int n_block_size = (n_left > 0) ? n_left / 2 : 0;

    // Avoid division by zero if n_block_size is 0
    const int erased_blocks = (n_block_size > 0) ? (prompt_tokens.size() - params.n_keep - n_block_size) / n_block_size : 0;

    // Ensure n_keep is not negative and within bounds
    int keep_count = std::max(0, params.n_keep);
    keep_count = std::min(keep_count, (int)prompt_tokens.size());

    // Keep n_keep tokens at start of prompt (at most n_ctx - 4)
    std::vector<llama_token> new_tokens(prompt_tokens.begin(), prompt_tokens.begin() + keep_count);

    // Calculate start index for the end part, ensuring it's valid
    size_t end_part_start_index = (size_t)keep_count + (size_t)std::max(0, erased_blocks * n_block_size);
    if (end_part_start_index < prompt_tokens.size()) {
        new_tokens.insert(new_tokens.end(), prompt_tokens.begin() + end_part_start_index, prompt_tokens.end());
    }

    // Restore log call
    LOG_VERBOSE("input truncated, n_ctx: %d, n_keep: %d, n_left: %d, new_tokens_size: %zu",
        n_ctx,
        params.n_keep,
        n_left,
        new_tokens.size()
    );

    truncated = true;
    prompt_tokens = new_tokens;
}

/**
 * @brief Loads a prompt into the context
 * 
 * Tokenizes and prepares a prompt for inference. If an image is provided and
 * mtmd_context is available, it uses libmtmd to process multimodal input.
 * Otherwise, it processes as a text-only prompt.
 */
void cactus_context::loadPrompt() {
    // Automatically prepend <__image__> tag if an image is provided and the tag is missing.
    if (!this->params.image.empty() && !this->params.image[0].empty()) {
        if (this->params.prompt.find("<__image__>") == std::string::npos) {
            this->params.prompt = "<__image__>\n" + this->params.prompt;
            LOG_INFO("Automatically prepended <__image__> tag to prompt in CactusContext::loadPrompt().");
        }
    }

    // Ensure embd is clear for this new prompt load, n_past should be 0 (typically after rewind)
    // rewind() clears embd and sets n_past to 0.
    // If loadPrompt is called without rewind, existing n_past and embd state might interfere.
    // For now, assume loadPrompt is for a fresh start or after rewind.
    embd.clear(); 
    n_past = 0; // Explicitly reset n_past here for the current prompt loading logic.
                // If this function could be part of a longer conversation turn, 
                // n_past management would need to be more sophisticated.

    // Check if multimodal context is available and prompt is not empty
    if (ctx_mtmd != nullptr && !params.image.empty() && !params.prompt.empty()) {
        LOG_INFO("Multimodal prompt detected. Using libmtmd.");

        mtmd_input_text input_text;
        input_text.text = params.prompt.c_str(); 
        input_text.add_special = true; 
        input_text.parse_special = true;

        mtmd_bitmap *bitmap = mtmd_helper_bitmap_init_from_file(params.image[0].c_str());
        if (!bitmap) {
            LOG_ERROR("Failed to load image %s for mtmd.", params.image[0].c_str());
            // Fallback to text-only or error out? For now, log error and proceed to text only if possible,
            // but mtmd_tokenize will likely fail if marker is present but no bitmap.
            // Better to ensure prompt is text-only if image fails.
            // This path implies an issue, probably should not proceed with multimodal tokenize.
            goto text_only_prompt; // Jump to text-only processing if image load fails
        }

        // Prepare the bitmap for mtmd_tokenize
        const mtmd_bitmap *bitmaps_array[] = {bitmap};
        size_t n_bitmaps = 1; 

        // Initialize the chunks structure
        mtmd_input_chunks *chunks = mtmd_input_chunks_init();
        if (!chunks) {
            LOG_ERROR("Failed to initialize mtmd_input_chunks.");
            mtmd_bitmap_free(bitmap);
            goto text_only_prompt;
        }

        int tokenize_res = mtmd_tokenize(ctx_mtmd, chunks, &input_text, bitmaps_array, n_bitmaps);
        mtmd_bitmap_free(bitmap);

        if (tokenize_res != 0) {
            LOG_ERROR("mtmd_tokenize failed with code %d. Check prompt markers and image count.", tokenize_res);
            mtmd_input_chunks_free(chunks);
            goto text_only_prompt;
        }

        // Feed text tokens from chunks to the sampler
        if (ctx_sampling) {
            for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
                const mtmd_input_chunk *chunk = mtmd_input_chunks_get(chunks, i);
                if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                    size_t n_text_tokens = 0;
                    const llama_token *text_tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_text_tokens);
                    for (size_t j = 0; j < n_text_tokens; ++j) {
                        common_sampler_accept(ctx_sampling, text_tokens[j], false);
                    }
                }
            }
        } else {
            LOG_WARNING("ctx_sampling is null, cannot accept prompt tokens into sampler for multimodal input.");
        }

        // Get number of tokens/positions from chunks *before* freeing them.
        this->num_prompt_tokens = static_cast<size_t>(mtmd_helper_get_n_pos(chunks));

        llama_pos new_n_past = 0;
        int eval_res = mtmd_helper_eval_chunks(ctx_mtmd, ctx, chunks, (llama_pos)this->n_past, 0, params.n_batch, true, &new_n_past);
        mtmd_input_chunks_free(chunks); 

        if (eval_res == 0) {
            this->n_past = static_cast<size_t>(new_n_past);
            LOG_INFO("mtmd_helper_eval_chunks successful. n_past updated to: %zu, num_prompt_tokens: %zu", this->n_past, this->num_prompt_tokens);
        } else {
            LOG_ERROR("mtmd_helper_eval_chunks failed with code %d.", eval_res);
            this->n_past = 0; 
            this->num_prompt_tokens = 0;
        }
        // TODO: Revisit how to correctly update sampler state after mtmd_helper_eval_chunks.

    } else {

text_only_prompt:
        LOG_INFO("No image or mtmd_context not available/prompt not suitable. Processing as text-only prompt.");
        std::vector<llama_token> prompt_tokens_text = ::common_tokenize(ctx, params.prompt, true, true);
        this->num_prompt_tokens = prompt_tokens_text.size();

        std::stringstream ss;
        ss << "\n" << __func__ << ": text_only_prompt_tokens = ";
        for (auto& token : prompt_tokens_text) {
            ss << token << " ";
        }
        LOG_INFO("%s\n", ss.str().c_str());

        if (params.n_keep < 0) {
            params.n_keep = (int)this->num_prompt_tokens;
        }
        params.n_keep = std::min(n_ctx > 4 ? n_ctx - 4 : 0, params.n_keep);
        params.n_keep = std::max(0, params.n_keep);

        if (this->num_prompt_tokens >= (size_t) n_ctx) {
            truncatePrompt(prompt_tokens_text); // truncatePrompt works on its argument by reference
            this->num_prompt_tokens = prompt_tokens_text.size();
            LM_GGML_ASSERT(this->num_prompt_tokens < (size_t) n_ctx || n_ctx == 0);
        }
        
        for (auto & token : prompt_tokens_text) {
            common_sampler_accept(ctx_sampling, token, false);
        }

        // n_past here refers to overlap with previous `embd` content, which is now cleared.
        // So, common_part(this->embd, prompt_tokens_text) will be 0.
        this->n_past = common_part(this->embd, prompt_tokens_text); // embd is empty, so n_past = 0
        this->embd = prompt_tokens_text;

        // n_past = std::min(this->n_past, this->embd.size()); // n_past is 0
        // if (this->n_past == this->num_prompt_tokens && this->n_past > 0) { this->n_past--; }
        // if (this->n_past > 0) { llama_kv_self_seq_rm(ctx, 0, this->n_past, -1); }
        if (this->num_prompt_tokens > 0 && this->n_past == this->num_prompt_tokens) {
            // This case means embd was identical to prompt_tokens_text and fully matched.
            // To ensure at least one token is evaluated to get logits for sampling.
            this->n_past--; 

        } else if (this->n_past > 0) {
             // This means some prefix of prompt_tokens_text matched a previous `embd` (not possible here as embd was cleared).
             // Since common_part will be 0, this block won't run.
        }
         // If n_past is 0, all tokens in `this->embd` are new and need evaluation.
    }

    LOG_VERBOSE("prompt loaded, n_past: %zu, embd_size (text part for nextToken): %zu",
        this->n_past,
        this->embd.size()
    );
    has_next_token = true;
}


/**
 * @brief Begins the completion/generation process
 * 
 * Sets up internal state for token generation
 */
void cactus_context::beginCompletion() {
    // number of tokens to keep when resetting context
    n_remain = params.n_predict;
    llama_perf_context_reset(ctx);
    is_predicting = true;
}


/**
 * @brief Generates the next token
 * 
 * @return The generated token and its probabilities
 */
completion_token_output cactus_context::nextToken()
{
    completion_token_output result;
    result.tok = -1;

    // If embd is not empty, it means we have a text-only prompt (or text part after image, if not using mtmd_helper_eval_chunks)
    // that needs to be processed first to fill the KV cache.
    // If mtmd_helper_eval_chunks was used in loadPrompt, embd will be empty, and n_past is already set.
    if (!embd.empty()) {
        // This loop processes the initial prompt tokens stored in `embd`.
        // `n_past` is 0 if it's a fresh text prompt.
        while ((size_t)n_past < embd.size()) {
            int n_eval = (int)embd.size() - n_past;
            if (n_eval > params.n_batch) {
                n_eval = params.n_batch;
            }

            if (n_eval <= 0) { 
                LOG_WARNING("nextToken: No prompt tokens to evaluate in embd (n_eval=%d)", n_eval);
                break; 
            }

            if (llama_decode(ctx, llama_batch_get_one(&embd[n_past], n_eval)) != 0) {
                LOG_ERROR("nextToken: failed to eval prompt, n_eval: %d, n_past: %zu", n_eval, n_past);
                has_next_token = false;
                return result;
            }
            n_past += n_eval;

            if(is_interrupted) {
                LOG_INFO("nextToken: Decoding Interrupted during prompt processing");
                has_next_token = false;
                return result;
            }
        }
        // After this loop, the initial prompt in `embd` (if any) is processed.
        // `embd` itself is not cleared here, it holds the prompt tokens.
        // For generation, we will sample a new token and then decode *that* token.
    }

    // --- Token Generation Phase --- 

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

    if (params.n_predict == 0 && n_remain == 0) { 
        has_next_token = false;
        result.tok = llama_vocab_eos(vocab);
        LOG_VERBOSE("nextToken: n_predict is 0, EOS token returned.", "");
        return result;
    }
    
    // Check if prediction limit has been reached
    if (n_remain == 0 && params.n_predict != -1) { 
        has_next_token = false;
        result.tok = llama_vocab_eos(vocab); 
        LOG_VERBOSE("nextToken: n_remain is 0, EOS token returned.", "");
        stopped_limit = true; 
        return result;
    }

    // Sample the next token
    result.tok = common_sampler_sample(ctx_sampling, ctx, -1); 
    llama_token_data_array cur_p = *common_sampler_get_candidates(ctx_sampling);
    const int32_t n_probs = params.sampling.n_probs;
    for (size_t i = 0; i < std::min((size_t)cur_p.size, (size_t)n_probs); ++i) {
        if (cur_p.data[i].id < (llama_token)llama_vocab_n_tokens(vocab)) { 
             result.probs.push_back({cur_p.data[i].id, cur_p.data[i].p});
        }
    }

    common_sampler_accept(ctx_sampling, result.tok, true);
    num_tokens_predicted++;

    // Prepare batch for the new token and decode it
    if (llama_decode(ctx, llama_batch_get_one(&result.tok, 1)) != 0) {
        LOG_ERROR("nextToken: failed to eval generated token %d at n_past %zu", result.tok, n_past);
        has_next_token = false;
        return result;
    }

    // Increment n_past for the newly decoded token
    n_past += 1; 

    // Add the newly generated token to embd for context management (e.g. sliding window)
    // This `embd` will be used by the context shifting logic if n_ctx is exceeded.
    embd.push_back(result.tok);

    if (n_remain > 0 && params.n_predict != -1) {
        --n_remain;
    }

    if (result.tok == llama_vocab_eos(vocab)) {
        has_next_token = false;
        stopped_eos = true;
        LOG_VERBOSE("nextToken: EOS token %d generated.", result.tok);
        return result;
    }

    // Context shifting logic (from original cactus_completion.cpp, adapted)
    // This should happen *after* a token is generated and added to embd, 
    // and *before* the next nextToken call, to make space if needed.
    if (embd.size() >= (size_t)params.n_ctx) {
        if (params.n_ctx <= params.n_keep + 1) {
             LOG_ERROR("Context size (%d) too small for keep (%d)", params.n_ctx, params.n_keep);
             has_next_token = false; // Cannot proceed
             return result;
        }

        // Note: n_past here reflects the state *after* the current token was decoded.
        // The embd vector contains all tokens processed *including* the one just generated.
        // The goal is to shift KV cache and `embd` to make space for future tokens.

        // The number of tokens currently in the KV cache before this shift is `n_past`.
        // The number of tokens in `embd` is `embd.size()`.
        // These should be consistent if `embd` only ever grows by one token that is then decoded.
        LM_GGML_ASSERT(n_past == embd.size());

        const int n_total_in_kv = n_past; // Total tokens that have affected KV cache so far.
        const int n_to_shift_count = n_total_in_kv - params.n_keep -1; // Number of tokens to consider shifting out beyond the keep region.
        const int n_discard = (n_to_shift_count > 0) ? n_to_shift_count / 2 : 0;

        if (n_discard > 0) {
            // LOG_VERBOSE("KV Cache shifting temporarily disabled for debugging.");
            /* 
            llama_kv_self_seq_rm(ctx, 0, params.n_keep + 1, params.n_keep + 1 + n_discard);
            llama_kv_self_seq_add(ctx, 0, params.n_keep + 1 + n_discard, n_total_in_kv, -n_discard);
            
            embd.erase(embd.begin() + params.n_keep + 1, embd.begin() + params.n_keep + 1 + n_discard);
            
            n_past -= n_discard;

            LOG_VERBOSE("Context shifted: n_discard: %d, new n_past: %zu, new embd.size: %zu", 
                        n_discard, n_past, embd.size());
            */
        }
    }

    if(is_interrupted) { 
        LOG_INFO("nextToken: Decoding Interrupted after token generation");
        has_next_token = false;
        return result;
    }

    has_next_token = params.n_predict == -1 || n_remain > 0;
    return result;
}


/**
 * @brief Searches for stopping strings in generated text
 * 
 * @param text The text to search in
 * @param last_token_size Size of the last token
 * @param type Type of stopping to check for
 * @return Position of the stop string if found, npos otherwise
 */
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
            // Ensure from_pos is not negative
            size_t from_pos = 0;
            size_t tmp_len = word.size() + last_token_size;
            if (text.size() > tmp_len) {
                from_pos = text.size() - tmp_len;
            }
            pos = text.find(word, from_pos);
        }
        else // STOP_PARTIAL
        {
            // Relying on find_partial_stop_string from utils
             pos = cactus::find_partial_stop_string(word, text);
        }

        if (pos != std::string::npos)
        {
            // If we found a stop string, update stop_pos if it's the earliest one found
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


/**
 * @brief Performs a single completion step
 * 
 * Generates the next token and updates generated text
 * @return The generated token and its probabilities
 */
completion_token_output cactus_context::doCompletion()
{
    const completion_token_output token_with_probs = nextToken();

    // Handle potential error from nextToken where tok is -1
    if (token_with_probs.tok == -1 && !has_next_token) {
        return token_with_probs;
    }

    // Ensure context is valid before converting token to piece
    std::string token_text;
    if (ctx && token_with_probs.tok != -1) {
         token_text = common_token_to_piece(ctx, token_with_probs.tok);
    }

    generated_text += token_text;

    if (params.sampling.n_probs > 0)
    {
        generated_token_probs.push_back(token_with_probs);
    }

    // check if there is incomplete UTF-8 character at the end
    incomplete = false; 
    if (!generated_text.empty()) {
         unsigned char c = generated_text.back();
         int expected_continuation_bytes = 0;

         // Check if the last character is a continuation byte
         if ((c & 0xC0) == 0x80) { 
             int lookback = 1;

             // Check previous bytes to see if it's incomplete
             while (lookback < 4 && lookback < generated_text.size()) {
                 unsigned char prev_c = generated_text[generated_text.size() - 1 - lookback];

                 // Check if the previous byte is a start byte
                 if ((prev_c & 0xC0) == 0xC0) { // Found start byte 11xxxxxx
                     if      ((prev_c & 0xE0) == 0xC0) expected_continuation_bytes = 1;
                     else if ((prev_c & 0xF0) == 0xE0) expected_continuation_bytes = 2;
                     else if ((prev_c & 0xF8) == 0xF0) expected_continuation_bytes = 3;
                     incomplete = lookback < expected_continuation_bytes;
                     break;

                 // If the previous byte is an ASCII byte, the sequence is complete
                 } else if ((prev_c & 0x80) == 0x00) { 
                     break; 

                 } // else: found another continuation byte, keep looking back
                 lookback++;
             }
         } else if ((c & 0xE0) == 0xC0) { 
            incomplete = true; 
         } else if ((c & 0xF0) == 0xE0) { 
             incomplete = true; 
         } else if ((c & 0xF8) == 0xF0) { 
             incomplete = true; 
         }
         // else: starts with ASCII 0xxxxxxx or invalid byte, considered complete
    }

    if (incomplete && !has_next_token)
    {
        // Force generation of more tokens if the text ends mid-character
        has_next_token = true;
        if (params.n_predict != -1) { // Only increment n_remain if not infinite prediction
            n_remain++;
        }
    }

    // Check if prediction limit has been reached
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
