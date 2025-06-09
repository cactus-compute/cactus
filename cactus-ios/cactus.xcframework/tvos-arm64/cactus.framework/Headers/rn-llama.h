#ifndef RNLLAMA_H
#define RNLLAMA_H

#include <sstream>
#include <iostream>
#include <thread>
#include "chat.h"
#include "common.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"
#include "llama-impl.h"
#include "sampling.h"
#include "json.hpp"
#if defined(__ANDROID__)
#include <android/log.h>
#endif

using json = nlohmann::ordered_json;

namespace rnllama {

std::string tokens_to_output_formatted_string(const llama_context *ctx, const llama_token token);

std::string tokens_to_str(llama_context *ctx, const std::vector<llama_token>::const_iterator begin, const std::vector<llama_token>::const_iterator end);

lm_ggml_type kv_cache_type_from_str(const std::string & s);

enum stop_type
{
    STOP_FULL,
    STOP_PARTIAL,
};

// completion token output with probabilities
struct completion_token_output
{
    struct token_prob
    {
        llama_token tok;
        float prob;
    };

    std::vector<token_prob> probs;
    llama_token tok;
};

struct llama_rn_context_mtmd;

struct llama_rn_context_vocoder;

struct llama_rn_tokenize_result {
    std::vector<llama_token> tokens;
    bool has_media = false;
    std::vector<std::string> bitmap_hashes;
    std::vector<size_t> chunk_pos; // both text and media
    std::vector<size_t> chunk_pos_media; // media only
};

enum tts_type {
    UNKNOWN = -1,
    OUTETTS_V0_2 = 1,
    OUTETTS_V0_3 = 2,
};

// Main context class
struct llama_rn_context {
    bool is_predicting = false;
    bool is_interrupted = false;
    bool has_next_token = false;
    std::string generated_text;
    std::vector<completion_token_output> generated_token_probs;
    std::vector<llama_token> audio_tokens;

    size_t num_prompt_tokens = 0;
    size_t num_tokens_predicted = 0;
    llama_pos n_past = 0;
    size_t n_remain = 0;
    std::vector<std::string> mtmd_bitmap_past_hashes;

    std::vector<llama_token> embd;
    common_params params;
    common_init_result llama_init;

    bool next_token_uses_guide_token = true;
    std::vector<llama_token> guide_tokens;

    llama_model *model = nullptr;
    float loading_progress = 0;
    bool is_load_interrupted = false;

    llama_context *ctx = nullptr;
    common_sampler *ctx_sampling = nullptr;
    common_chat_templates_ptr templates;

    int n_ctx;

    bool context_full = false;
    bool truncated = false;
    bool stopped_eos = false;
    bool stopped_word = false;
    bool stopped_limit = false;
    std::string stopping_word;
    bool incomplete = false;

    std::vector<common_adapter_lora_info> lora;

    llama_rn_context_mtmd *mtmd_wrapper = nullptr;
    bool has_multimodal = false;

    llama_rn_context_vocoder *vocoder_wrapper = nullptr;
    bool has_vocoder = false;

    ~llama_rn_context();

    void rewind();
    bool initSampling();
    bool loadModel(common_params &params_);
    bool validateModelChatTemplate(bool use_jinja, const char *name) const;
    common_chat_params getFormattedChatWithJinja(
      const std::string &messages,
      const std::string &chat_template,
      const std::string &json_schema,
      const std::string &tools,
      const bool &parallel_tool_calls,
      const std::string &tool_choice
    ) const;
    std::string getFormattedChat(
      const std::string &messages,
      const std::string &chat_template
    ) const;
    void truncatePrompt(std::vector<llama_token> &prompt_tokens);
    void loadPrompt(const std::vector<std::string> &media_paths);
    void setGuideTokens(const std::vector<llama_token> &tokens);
    void beginCompletion();
    void endCompletion();
    completion_token_output nextToken();
    size_t findStoppingStrings(const std::string &text, const size_t last_token_size, const stop_type type);
    completion_token_output doCompletion();
    std::vector<float> getEmbedding(common_params &embd_params);
    std::string bench(int pp, int tg, int pl, int nr);
    int applyLoraAdapters(std::vector<common_adapter_lora_info> lora);
    void removeLoraAdapters();
    std::vector<common_adapter_lora_info> getLoadedLoraAdapters();

    // Multimodal methods
    bool initMultimodal(const std::string &mmproj_path, bool use_gpu);
    bool isMultimodalEnabled() const;
    bool isMultimodalSupportVision() const;
    bool isMultimodalSupportAudio() const;
    void releaseMultimodal();

    // Process multiple media and add them to the context
    void processMedia(
        const std::string &prompt,
        const std::vector<std::string> &media_paths
    );

    llama_rn_tokenize_result tokenize(const std::string &text, const std::vector<std::string> &media_paths);

    // Vocoder methods
    bool initVocoder(const std::string &vocoder_model_path);
    tts_type getTTSType(json speaker = nullptr);
    std::string getFormattedAudioCompletion(const std::string &speaker_json_str, const std::string &text_to_speak);
    std::vector<llama_token> getAudioCompletionGuideTokens(const std::string &text_to_speak);
    std::vector<float> decodeAudioTokens(const std::vector<llama_token> &tokens);
    bool isVocoderEnabled() const;
    void releaseVocoder();
};

// Logging macros
extern bool rnllama_verbose;

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

} // namespace rnllama

#endif /* RNLLAMA_H */
