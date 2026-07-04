#pragma once

#include <vector>
#include <random>
#include <string>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <cstdint>
#include <atomic>
#include <limits>

#include "cactus_graph.h"
#include "kv_compress.h"

class CactusGraph;

namespace cactus {
namespace npu {

struct NPUNamedInput {
    std::string name;
    enum class DataType { FP16, INT32 };
    const void* data;
    DataType data_type = DataType::FP16;
    std::vector<int> shape;
};

class NPUEncoder {
public:
    virtual ~NPUEncoder() = default;
    virtual bool load(const std::string& model_path, const std::string& compute_units = "") = 0;
    virtual bool preallocate(
        const std::vector<int>& input_shape,
        const std::string& input_name = "x",
        const std::string& output_name = "") = 0;
    virtual size_t encode(
        const __fp16* input, __fp16* output,
        const std::vector<int>& shape,
        const std::string& input_name = "x",
        const std::string& output_name = "") = 0;
    virtual bool is_available() const = 0;
    virtual std::vector<int> get_input_shape() const = 0;
    virtual std::vector<int> get_output_shape() const = 0;
    virtual bool has_input(const std::string&) const { return false; }
    virtual std::vector<int> get_input_shape_for(const std::string&) const { return {}; }
    virtual __fp16* get_output_buffer() = 0;
    virtual size_t get_output_buffer_size() const = 0;
    virtual size_t encode_multimodal_input(
        const std::vector<NPUNamedInput>& inputs,
        __fp16* output,
        const std::string& output_name = "") = 0;
};

std::unique_ptr<NPUEncoder> create_encoder();
bool is_npu_available();

} // namespace npu
namespace engine {

struct Config {
    uint32_t vocab_size = 151936;
    uint32_t bos_token_id = 151643;
    uint32_t eos_token_id = 151645;
    uint32_t decoder_start_token_id = 151643;
    std::vector<uint32_t> decoder_prompt_token_ids;
    uint32_t num_layers = 28;
    uint32_t hidden_dim = 1024;
    uint32_t ffn_intermediate_dim = 3072;
    uint32_t attention_heads = 16;
    uint32_t attention_kv_heads = 8;
    uint32_t attention_head_dim = 128;
    float layer_norm_eps = 1e-6f;
    float rope_theta = 1000000.0f;
    uint32_t num_experts = 0;
    uint32_t num_shared_experts = 0;
    uint32_t num_top_experts = 0;
    uint32_t moe_every_n_layers = 0;
    uint32_t moe_intermediate_dim = 0;
    uint32_t num_dense_layers = 0;
    uint32_t num_experts_per_tok = 0;
    bool norm_topk_prob = false;
    bool use_expert_bias = false;
    float routed_scaling_factor = 1.0f;
    bool tie_word_embeddings = true;

    uint32_t vision_hidden_dim = 0;
    uint32_t vision_num_layers = 0;
    uint32_t vision_attention_heads = 0;
    uint32_t vision_image_size = 0;
    uint32_t vision_patch_size = 0;
    uint32_t vision_num_channels = 3;
    uint32_t vision_embed_dim = 0;
    uint32_t visual_tokens_per_img = 0;
    bool use_pixel_shuffle = false;
    uint32_t pixel_shuffle_factor = 1;
    bool use_image_tokens = false;
    uint32_t image_token_id = 0;
    bool use_layout_tags = false;
    uint32_t image_seq_len = 64;

    uint32_t global_image_size = 2048;
    uint32_t max_tile_size = 512;
    float rescale_factor = 0.00392156862745098f;
    float image_mean = 0.5f;
    float image_std = 0.5f;
    
    uint32_t downsample_factor = 2;
    uint32_t min_tiles = 2;
    uint32_t max_tiles = 10;
    bool use_thumbnail = true;
    uint32_t min_image_tokens = 64;
    uint32_t max_image_tokens = 256;
    uint32_t max_num_patches = 1024;
    uint32_t tile_size = 512;
    float max_pixels_tolerance = 2.0f;
    bool do_image_splitting = true;
    bool encoder_act_gelu = false;
    bool decoder_act_gelu = false;
    uint32_t num_encoder_layers = 0;
    uint32_t num_decoder_layers = 0;
    float partial_rotary_factor = 0.0f;
    uint32_t pad_token_id = 0;
    uint32_t conv_kernel_size = 0;
    uint32_t subsampling_conv_kernel_size = 0;
    uint32_t subsampling_conv_stride = 0;
    uint32_t subsampling_conv_channels = 0;
    uint32_t subsampling_factor = 0;
    uint32_t num_mel_bins = 80;
    std::string encoder_hidden_act = "silu";
    uint32_t linear_num_key_heads = 0;
    uint32_t linear_key_head_dim = 0;
    uint32_t linear_num_value_heads = 0;
    uint32_t linear_value_head_dim = 0;
    uint32_t linear_q_proj_dim = 0;
    uint32_t linear_k_proj_dim = 0;
    uint32_t linear_v_proj_dim = 0;

    uint32_t kv_lora_rank = 0;
    uint32_t q_lora_rank = 0;
    uint32_t qk_head_dim = 0;
    uint32_t qk_nope_head_dim = 0;
    uint32_t qk_rope_head_dim = 0;
    uint32_t v_head_dim = 0;
    uint32_t rope_interleave = 0;
    bool attention_bias = false;
    float rope_scaling_factor = 1.0f;
    float rope_mscale_all_dim = 0.0f;

    enum class ModelType {QWEN = 0, GEMMA = 1, NOMIC = 3, LFM2 = 5, SIGLIP2 = 6, WHISPER = 7, MOONSHINE = 8, PARAKEET = 10, QWEN3P5 = 11, PARAKEET_TDT = 12, GEMMA3N = 13, YOUTU = 14, GEMMA4 = 15, NEEDLE = 18};
    uint32_t predictor_hidden_dim = 0;
    uint32_t predictor_num_layers = 0;
    uint32_t tdt_joint_dim = 0;
    uint32_t tdt_num_durations = 0;
    uint32_t tdt_blank_id = 0;
    std::vector<uint32_t> tdt_durations;

    ModelType model_type = ModelType::GEMMA4;

    enum class ModelVariant {DEFAULT = 0, VLM = 1, EXTRACT = 2, RAG = 3};
    ModelVariant model_variant = ModelVariant::DEFAULT;

    enum class Activation {GELU = 0, SILU = 1};
    Activation activation = Activation::SILU;

    enum class Backend {CPU = 0, NPU = 1};
    Backend default_backend = Backend::CPU;

    enum class Precision {INT8 = 0, FP16 = 1, FP32 = 2};
    Precision precision = Precision::FP32;

    float default_temperature = 0.6f;
    float default_top_p = 0.95f;
    size_t default_top_k = 20;
    float default_max_tps = -1.0f;
    float default_cloud_handoff_threshold = 0.0f;

    std::vector<std::string> layer_types;
    size_t conv_L_cache = 0;

    // Rolling bounded KV compaction (default ON, 4096 -> 2048). Override at runtime with
    // CACTUS_KV_COMPRESS_AT (trigger) / CACTUS_KV_COMPRESS_TO (target); CACTUS_KV_COMPRESS_AT=0 disables.
    bool kv_compress = true;
    float kv_compress_recent_frac = 0.30f;
    uint32_t kv_compress_sink = 4;
    int32_t kv_compress_trigger_len = 4096;
    int32_t kv_compress_target_len = 2048;
    bool kv_compress_preserve_special = true;

    uint32_t altup_num_inputs = 4;
    uint32_t laurel_rank = 64;
    static constexpr uint32_t UNSET_U32 = UINT32_MAX;
    static constexpr float UNSET_F32 = -1e30f;
    uint32_t hidden_size_per_layer_input = UNSET_U32;
    uint32_t num_kv_shared_layers = UNSET_U32;
    uint32_t sliding_window = UNSET_U32;
    float rope_local_base_freq = UNSET_F32;
    float final_logit_softcapping = UNSET_F32;
    float global_partial_rotary_factor = UNSET_F32;
    uint32_t expert_intermediate_size = 0;
    uint32_t global_head_dim = UNSET_U32;
    uint32_t num_global_kv_heads = 0;
    bool attention_k_eq_v = false;
    bool enable_moe_block = false;
    std::vector<float> activation_sparsity_ppf;

    uint32_t vision_head_dim = 64;
    uint32_t vision_kv_heads = 12;
    uint32_t vision_intermediate_size = 3072;
    uint32_t vision_position_embedding_size = 10240;
    uint32_t vision_pooling_kernel_size = 3;
    uint32_t vision_default_output_length = 280;
    float vision_rope_theta = 100.0f;

    uint32_t audio_hidden_dim = 0;
    uint32_t audio_num_layers = 0;
    uint32_t audio_num_heads = 0;
    uint32_t audio_head_dim = 0;
    uint32_t audio_input_feat_size = 128;
    uint32_t audio_conf_conv_kernel_size = 5;
    uint32_t audio_chunk_size = 12;
    uint32_t audio_context_left = 13;
    uint32_t audio_context_right = 0;
    float audio_logit_cap = 50.0f;
    float audio_residual_weight = 0.5f;
    uint32_t audio_output_proj_dims = 0;
    uint32_t audio_vocab_size = 128;
    uint32_t audio_vocab_offset = 0;
    uint32_t audio_soft_tokens = 188;
    uint32_t audio_sscp_conv0_channels = 128;
    uint32_t audio_sscp_conv1_channels = 32;
    float audio_sscp_conv_eps = 1e-3f;
    float audio_rms_norm_eps = 1e-6f;
    uint32_t audio_fft_length = 1024;
    uint32_t audio_token_id = 0;
    bool audio_fft_overdrive = false;

    static bool is_gemma_family(ModelType t) {
        return t == ModelType::GEMMA || t == ModelType::GEMMA3N || t == ModelType::GEMMA4;
    }

    static bool is_gemma3_family(ModelType t) {
        return t == ModelType::GEMMA || t == ModelType::GEMMA3N;
    }

    bool from_json(const std::string& json_path);
    std::string to_json() const;
    // Disable rolling unless 0 < target < trigger (when trigger > 0).
    void validate_kv_compress();
    bool parse_kv_compress_override(const char* trigger_env, const char* target_env);
};



struct MergeRule {
    std::string first;
    std::string second;
    std::string merged;
    uint32_t priority;
    
    MergeRule(const std::string& f, const std::string& s, const std::string& m, uint32_t p)
        : first(f), second(s), merged(m), priority(p) {}
};


struct ToolCallInfo {
    std::string name;
    std::string arguments;
};

struct ChatMessage {
    std::string role;
    std::string content;
    std::string name;
    std::vector<std::string> images;
    std::vector<std::string> audio;
    size_t audio_soft_token_count = 0;
    std::vector<ToolCallInfo> tool_calls;
};

struct ToolConstraintSpec {
    std::string name;
    std::vector<std::string> parameter_names;
    std::vector<std::vector<std::string>> parameter_enums;
    std::vector<std::string> required_parameter_names;
};

struct TokenizerRuntimeConfig {
    enum class TokenizerType { UNKNOWN, BPE, SENTENCEPIECE };
    enum class VocabFormat { UNKNOWN, ID_TAB_TOKEN, LINE_TOKEN };
    enum class Normalizer { NONE, METASPACE, BYTE_LEVEL };
    enum class Decoder { NONE, REPLACE_METASPACE, BYTE_LEVEL };

    TokenizerType tokenizer_type = TokenizerType::UNKNOWN;
    VocabFormat vocab_format = VocabFormat::UNKNOWN;
    Normalizer normalizer = Normalizer::NONE;
    Decoder decoder = Decoder::NONE;
    bool byte_fallback = false;
    bool has_chat_template = false;
};

TokenizerRuntimeConfig load_tokenizer_runtime_config(const std::string& config_file);
void load_special_tokens_map(const std::string& config_file, std::unordered_map<std::string, uint32_t>& special_tokens);
std::vector<std::string> split_with_special_tokens(const std::string& text, const std::unordered_map<std::string, uint32_t>& special_tokens);

inline std::string extract_json_string(const std::string& json, size_t& pos) {
    std::string value;
    while (pos < json.size() && json[pos] != '"') {
        if (json[pos] == '\\' && pos + 1 < json.size()) {
            pos++;
            if (json[pos] == 'n') value += '\n';
            else if (json[pos] == 't') value += '\t';
            else if (json[pos] == 'r') value += '\r';
            else if (json[pos] == '"') value += '"';
            else if (json[pos] == '\\') value += '\\';
            else value += json[pos];
        } else {
            value += json[pos];
        }
        pos++;
    }
    if (pos < json.size()) pos++;
    return value;
}

class Tokenizer {
public:
    virtual ~Tokenizer() = default;

    virtual std::vector<uint32_t> encode(const std::string& text) const = 0;
    virtual std::string decode(const std::vector<uint32_t>& tokens) const = 0;

    virtual std::vector<uint32_t> apply_chat_template(const std::vector<ChatMessage>& messages, bool add_generation_prompt = true) const;
    virtual std::string format_chat_prompt(const std::vector<ChatMessage>& messages, bool add_generation_prompt = true, const std::string& tools_json = "", bool enable_thinking_if_supported = false) const;

    virtual uint32_t get_vocab_size() const = 0;
    virtual uint32_t get_unk_token() const = 0;
    virtual uint32_t get_bos_token() const = 0;
    virtual uint32_t get_eos_token() const = 0;
    virtual std::unordered_set<uint32_t> special_token_ids() const { return {}; }
    virtual bool has_chat_template() const { return has_chat_template_; }
    bool is_qwen_family() const { return model_type_ == ModelType::QWEN; }
    bool is_lfm2_family() const { return model_type_ == ModelType::LFM2; }
    bool has_function_call_tokens() const { return encode("<start_function_call>").size() == 1; }
    std::string get_default_stop_sequence() const;

    virtual bool load_vocabulary_with_config(const std::string& vocab_file, const std::string& merges_file, const std::string& config_file) = 0;
    
    uint32_t get_image_token_id() const { return image_token_id_; }
    uint32_t get_fake_token_id() const { return fake_token_id_; }
    uint32_t get_global_img_token_id() const { return global_img_token_id_; }

    void set_image_soft_token_count(size_t n) { image_soft_token_count_ = n; }
    size_t get_image_soft_token_count() const { return image_soft_token_count_; }

    void set_lfm2_vision_config(const Config& cfg) { lfm2_vision_config_ = cfg; has_lfm2_vision_config_ = true; }

protected:
    enum class ModelType { UNKNOWN, GEMMA4, GEMMA, QWEN, LFM2, NEEDLE };
    ModelType model_type_ = ModelType::UNKNOWN;
    enum class ModelVariant { DEFAULT, VLM, EXTRACT, RAG};
    ModelVariant model_variant_ = ModelVariant::DEFAULT;
    bool has_chat_template_ = false;
    std::string chat_template_;
    
    uint32_t image_token_id_ = 396;
    uint32_t fake_token_id_ = 49189;
    uint32_t global_img_token_id_ = 49152;


    uint32_t vision_patch_size_ = 16;
    uint32_t vision_pooling_kernel_size_ = 3;
    uint32_t vision_default_output_length_ = 280;
    uint32_t vision_image_size_ = 768;
    size_t image_soft_token_count_ = 0;
    Config lfm2_vision_config_{};
    bool has_lfm2_vision_config_ = false;
    TokenizerRuntimeConfig runtime_config_;

    void detect_model_type(const std::string& config_path);
    void load_chat_template(const std::string& template_file);
    std::string format_gemma4_style(const std::vector<ChatMessage>& messages, bool add_generation_prompt, const std::string& tools_json, bool enable_thinking_if_supported = false) const;
    std::string format_qwen_style(const std::vector<ChatMessage>& messages, bool add_generation_prompt, const std::string& tools_json, bool enable_thinking_if_supported = false) const;
    std::string format_lfm2_style(const std::vector<ChatMessage>& messages, bool add_generation_prompt, const std::string& tools_json, bool enable_thinking_if_supported = false) const;
    std::string format_needle_style(const std::vector<ChatMessage>& messages, bool add_generation_prompt, const std::string& tools_json) const;
    std::string format_gemma_style(const std::vector<ChatMessage>& messages, bool add_generation_prompt, const std::string& tools_json) const;
};

class BPETokenizer : public Tokenizer {
public:
    BPETokenizer();
    ~BPETokenizer();

    bool load_vocabulary_mmap(const std::string& vocab_file, const std::string& merges_file);
    bool load_vocabulary_with_config(const std::string& vocab_file, const std::string& merges_file, const std::string& config_file) override;

    std::vector<uint32_t> encode(const std::string& text) const override;
    std::string decode(const std::vector<uint32_t>& tokens) const override;

    uint32_t get_vocab_size() const override { return vocab_size_; }
    uint32_t get_unk_token() const override { return unk_token_id_; }
    uint32_t get_bos_token() const override { return bos_token_id_; }
    uint32_t get_eos_token() const override { return eos_token_id_; }
    std::unordered_set<uint32_t> special_token_ids() const override {
        std::unordered_set<uint32_t> ids;
        for (const auto& kv : special_tokens_) ids.insert(kv.second);
        return ids;
    }

private:
    std::unordered_map<std::string, uint32_t> token_to_id_;
    std::vector<std::string> id_to_token_;
    std::vector<MergeRule> merge_rules_;
    std::unordered_map<std::string, uint32_t> merge_map_;  

    uint32_t vocab_size_;
    uint32_t unk_token_id_;
    uint32_t bos_token_id_;
    uint32_t eos_token_id_;

    void* vocab_mmap_ptr_;
    size_t vocab_mmap_size_;

    void* merges_mmap_ptr_;
    size_t merges_mmap_size_;

    std::vector<std::string> apply_bpe(const std::vector<std::string>& tokens) const;
    std::pair<int, uint32_t> find_best_merge_fast(const std::vector<std::string>& tokens) const;
    
    std::string bytes_to_unicode(const std::string& text) const;
    std::string unicode_to_bytes(const std::string& text) const;
    std::vector<std::string> byte_level_split(const std::string& text) const;
    std::vector<std::string> utf8_split(const std::string& text) const;

    void cleanup_mmap();
    
private:
    mutable std::unordered_map<uint8_t, std::string> byte_to_unicode_;
    mutable std::unordered_map<std::string, uint8_t> unicode_to_byte_;
    void init_byte_mappings() const;

    std::unordered_map<std::string, uint32_t> special_tokens_;
    std::vector<std::string> split_with_special_tokens(const std::string& text) const;
    void load_special_tokens(const std::string& config_file);
};

class SPTokenizer : public Tokenizer {
public:
    SPTokenizer();
    ~SPTokenizer();

    bool load_vocabulary_with_config(const std::string& vocab_file, const std::string& merges_file, const std::string& config_file) override;

    std::vector<uint32_t> encode(const std::string& text) const override;
    std::string decode(const std::vector<uint32_t>& tokens) const override;

    uint32_t get_vocab_size() const override { return vocab_size_; }
    uint32_t get_unk_token() const override { return unk_token_id_; }
    uint32_t get_bos_token() const override { return bos_token_id_; }
    uint32_t get_eos_token() const override { return eos_token_id_; }
    std::unordered_set<uint32_t> special_token_ids() const override {
        std::unordered_set<uint32_t> ids;
        for (const auto& kv : special_tokens_) ids.insert(kv.second);
        return ids;
    }

private:
    struct TrieNode {
        std::unordered_map<char32_t, std::unique_ptr<TrieNode>> children;
        int32_t token_id = -1;
        float score = 0.0f;
    };

    std::unique_ptr<TrieNode> trie_root_;
    std::unordered_map<std::string, uint32_t> token_to_id_;
    std::vector<std::string> id_to_token_;
    std::vector<float> token_scores_;

    uint32_t vocab_size_;
    uint32_t unk_token_id_;
    uint32_t bos_token_id_;
    uint32_t eos_token_id_;
    uint32_t pad_token_id_;

    bool sp_bpe_mode_ = false;
    bool sp_add_dummy_prefix_ = false;
    bool sp_byte_fallback_ = false;

    void* vocab_mmap_ptr_;
    size_t vocab_mmap_size_;

    void build_trie();
    std::vector<std::pair<std::string, uint32_t>> tokenize_with_trie(const std::string& text) const;
    std::vector<uint32_t> tokenize_with_bpe(const std::string& text) const;
    std::string preprocess_text(const std::string& text) const;
    std::string postprocess_text(const std::string& text) const;
    std::vector<std::string> split_by_unicode_spaces(const std::string& text) const;

    void cleanup_mmap();

    std::unordered_map<std::string, uint32_t> special_tokens_;
    std::vector<std::string> split_with_special_tokens(const std::string& text) const;
    void load_special_tokens(const std::string& config_file);
};



class ToolCallConstrainer {
public:
    enum class State {
        DONE,
        GEMMA_START,
        GEMMA_EXPECT_CALL,
        GEMMA_NAME,
        GEMMA_ARGS,
        GEMMA_EXPECT_END,
        NEEDLE_START
    };

    void init(Config::ModelType model_type,
              const std::vector<ToolConstraintSpec>& tools,
              Tokenizer* tokenizer);

    const std::unordered_map<uint32_t, float>& get_bias() const { return current_bias_; }
    const std::vector<float>* get_dense_bias() const { return dense_ready_ ? &dense_bias_ : nullptr; }

    void update(uint32_t token_id, const std::string& decoded_text);

    void reset();

    bool is_active() const { return active_; }

private:
    enum class Region { FREE, IN_NAME, IN_ARG_KEY };
    struct TrieNode {
        std::unordered_map<char, std::unique_ptr<TrieNode>> children;
        bool is_terminal = false;
    };

    bool active_ = false;
    State state_ = State::GEMMA_START;
    Config::ModelType model_type_ = Config::ModelType::GEMMA4;
    Tokenizer* tokenizer_ = nullptr;

    std::vector<ToolConstraintSpec> tool_specs_;
    std::string generated_text_;

    std::string call_start_tag_;
    std::string call_end_tag_;
    std::string response_start_tag_;
    std::vector<uint32_t> call_start_sequence_;
    std::vector<uint32_t> call_end_sequence_;
    size_t forced_tag_progress_ = 0;

    std::unordered_set<uint32_t> gemma_call_start_tokens_;
    std::unordered_set<uint32_t> gemma_call_end_tokens_;
    std::unordered_set<uint32_t> gemma_response_start_tokens_;
    std::unordered_set<uint32_t> gemma_call_prefix_tokens_;

    std::unordered_set<uint32_t> backtick_tokens_;
    std::unordered_set<uint32_t> open_brace_tokens_;
    std::unordered_set<uint32_t> close_brace_tokens_;

    std::unordered_map<uint32_t, float> current_bias_;
    std::vector<float> dense_bias_;
    bool dense_ready_ = false;

    void compute_bias();
    void tokenize_grammar_elements();
    void add_tokens_for_string(const std::string& str, std::unordered_set<uint32_t>& token_set);
    void advance_forced_tag(const std::vector<uint32_t>& sequence, uint32_t token_id);
    void init_common_tokens();

    bool is_needle() const { return model_type_ == Config::ModelType::NEEDLE; }

    Region region_ = Region::FREE;
    std::string constraint_buffer_;
    std::string constrained_buf_;
    std::string current_function_;
    bool in_arguments_ = false;
    int arguments_depth_ = 0;
    int nesting_depth_ = 0;
    bool in_string_value_ = false;
    bool prev_char_escape_ = false;
    bool at_key_start_ = false;
    bool await_enum_open_ = false;
    const TrieNode* active_enum_trie_ = nullptr;
    std::unordered_set<std::string> emitted_keys_;
    std::unique_ptr<TrieNode> name_trie_;
    std::unordered_map<std::string, std::unique_ptr<TrieNode>> param_tries_;
    std::unordered_map<std::string, std::unordered_map<std::string, std::unique_ptr<TrieNode>>> enum_tries_;
    std::unique_ptr<TrieNode> remaining_key_trie_;
    std::vector<std::string> token_strings_;
    std::unordered_map<char, std::vector<uint32_t>> token_index_;
    uint32_t escape_tag_token_ = 0;
    bool escape_tag_valid_ = false;

    void build_constraint_tables();
    void reset_constraint_state();
    void trie_insert(TrieNode* root, const std::string& word);
    const TrieNode* trie_seek(const TrieNode* root, const std::string& prefix) const;
    bool trie_token_valid(const std::string& token_text, const TrieNode* node, char terminator) const;
    void mark_trie_bias(const TrieNode* node, char terminator, const std::vector<char>& extra_allowed,
                        int32_t terminal_token = -1, bool exact_terminator = false);
    const TrieNode* current_param_trie() const;
    void rebuild_remaining_keys();
    bool required_satisfied() const;

    void feed_needle_text(const std::string& text);
    void feed_needle_char(char ch);
    bool needle_at_arg_key_start() const;
    bool needle_is_value_string_start() const;

    void feed_gemma_text(const std::string& text);
    void feed_gemma_char(char ch);
};

class Model {
public:
    struct DebugNode {
        uint32_t layer_idx;
        std::string name;
        size_t node_id;
    };

    Model();
    explicit Model(const Config& config);
    ~Model();

    const Config& get_config() const { return config_; }
    Tokenizer* get_tokenizer() const { return tokenizer_.get(); }
    const std::vector<DebugNode>& get_debug_nodes() const;

    bool init(const std::string& bundle_dir, size_t context_size,
              const std::string& system_prompt = "", bool do_warmup = true);

    uint32_t decode(const std::vector<uint32_t>& tokens, float temperature = -1.0f, float top_p = -1.0f,
                    size_t top_k = 0, const std::string& profile_file = "", float* out_entropy = nullptr,
                    float min_p = 0.15f, float repetition_penalty = 1.1f);
    void set_sample_seed(uint64_t seed) { sample_rng_.seed(static_cast<std::mt19937::result_type>(seed)); }
    bool prefill_and_sample_first_token(const std::vector<uint32_t>& tokens, uint32_t& out_token,
                                        float* out_uncertainty = nullptr);

    std::vector<std::vector<uint32_t>> decode_batch(const std::vector<uint32_t>& seed_tokens,
                                                    size_t max_new_tokens);
    std::vector<std::vector<uint32_t>> generate_batch(const std::vector<std::vector<uint32_t>>& prompts,
                                                      size_t max_new_tokens, bool stop_on_eos = false);

    bool supports_dynamic_batch();
    void set_decode_slots(size_t num_slots);
    std::vector<uint32_t> batch_stop_token_ids() const;

    void prefill(const std::vector<uint32_t>& tokens, size_t chunk_size = 128, const std::string& profile_file = "",
                 bool prepare_decode = true);

    void prefill_with_images(const std::vector<uint32_t>& tokens, const std::vector<std::string>& image_paths,
                             const std::string& profile_file = "");

    void prefill_with_audio(const std::vector<uint32_t>& tokens,
                            const std::vector<std::vector<float>>& audio_features_per_message,
                            const std::string& profile_file = "");

    void prefill_with_media(const std::vector<uint32_t>& tokens,
                            const std::vector<std::string>& image_paths,
                            const std::vector<std::vector<float>>& audio_features_per_message,
                            const std::string& profile_file = "");

    uint32_t decode_with_images(const std::vector<uint32_t>& tokens, const std::vector<std::string>& image_paths,
                                float temperature = -1.0f, float top_p = -1.0f,
                                size_t top_k = 0, const std::string& profile_file = "", float* out_entropy = nullptr,
                                float min_p = 0.15f, float repetition_penalty = 1.1f);

    uint32_t decode_with_audio(const std::vector<uint32_t>& tokens,
                               const std::vector<std::vector<float>>& audio_features_per_message,
                               float temperature = 0.0f, float top_p = 0.0f,
                               size_t top_k = 0, const std::string& profile_file = "", float* out_entropy = nullptr,
                               float min_p = 0.15f, float repetition_penalty = 1.1f,
                               float* out_token_time_start = nullptr, float* out_token_time_end = nullptr);

    struct ParakeetTdtStreamState {
        bool initialized = false;
        uint32_t last_token = 0;
        size_t time_index = 0;
        std::vector<std::vector<uint8_t>> dec_state;
        std::vector<uint32_t> pending;
        float confirmed_sec = 0.0f;
        size_t decoded_tokens = 0;
        double raw_decode_ms = 0.0;
    };

    std::vector<uint32_t> transcribe_parakeet_tdt(const std::vector<float>& audio_features,
                                                  ParakeetTdtStreamState* stream = nullptr,
                                                  bool is_final = true,
                                                  size_t end_frame = 0,
                                                  const std::atomic<bool>* should_stop = nullptr);
    std::vector<uint32_t> transcribe_whisper_seq2seq(const std::vector<float>& audio_features,
                                                     const std::vector<uint32_t>& decoder_prompt_tokens,
                                                     size_t max_tokens,
                                                     const std::vector<std::vector<uint32_t>>& stop_token_sequences,
                                                     const std::atomic<bool>* should_stop = nullptr,
                                                     int64_t suppress_token_id = -1);

    std::vector<float> get_embeddings(const std::vector<uint32_t>& tokens, bool pooled = true,
                                       bool normalize = false, const std::string& profile_file = "");

    std::vector<float> get_text_embeddings(const std::vector<uint32_t>& tokens, bool normalize = false);
    std::vector<float> get_lm_embeddings(const std::vector<uint32_t>& tokens, bool normalize = false);
    bool has_lm_embedding() const { return decoder_embed_ != nullptr; }
    bool has_text_embedding() const { return components_.count("text_embedding") > 0; }
    bool supports_warm_media_injection() const {
        if (lm_encoder_media_step_ != nullptr) return true;
        return encoder_ != nullptr && decoder_ != nullptr
            && input_index(*decoder_, "inputs_embeds") >= 0;
    }

    std::vector<float> get_image_embeddings(const std::string& image_path);

    std::vector<float> get_audio_embeddings(const std::vector<float>& audio_features);

    void reset_cache();
    void record_sampled_token(uint32_t token) {
        if (token_history_.size() >= MAX_TOKEN_HISTORY) {
            token_history_.erase(token_history_.begin(), token_history_.begin() + (MAX_TOKEN_HISTORY / 2));
        }
        token_history_.push_back(token);
    }

    double score_tokens_window_logprob(const std::vector<uint32_t>& tokens, size_t start, size_t end,
                                        size_t context, size_t* tokens_scored);

    void set_cache_window(size_t window_size, size_t sink_size = 4);
    size_t get_cache_size() const { return cache_total_seq_len_; }

    size_t get_prefill_chunk_size() const { return 128; }
    bool can_generate() const { return decoder_ != nullptr; }  // false for embedding/encoder-only models
    double last_prefill_cache_copy_ms() const { return last_prefill_cache_copy_ms_; }
    size_t last_prefill_padding_tokens() const { return last_prefill_padding_tokens_; }
    size_t last_prefill_scalar_tail_tokens() const { return last_prefill_scalar_tail_tokens_; }
    size_t last_prefill_tail_chunk_tokens() const { return last_prefill_tail_chunk_tokens_; }
    size_t last_prefill_tail_padding_tokens() const { return last_prefill_tail_padding_tokens_; }

    bool load_npu_audio_encoder(const std::string& model_path, const std::string& compute_units = "");
    bool has_npu_audio_encoder() const { return npu_audio_encoder_ != nullptr; }

    bool load_npu_vision_encoder(const std::string& model_path);
    bool has_npu_vision_encoder() const { return npu_vision_encoder_ != nullptr; }

    bool load_npu_source_encoder(const std::string& model_path);

    void remove_thinking_tokens(const std::vector<std::pair<size_t, size_t>>& ranges);
    void compact_kv_cache() {}

    void compress_kv_cache_keydiff(const cactus::kvcompress::Params& params);
    void maybe_roll_compact();
    std::vector<size_t> compressible_layers() const;
    void apply_kv_compress_env_override();

    void set_tool_constraints(const std::vector<ToolConstraintSpec>& tools);
    void clear_tool_constraints();
    void update_tool_constraints(uint32_t token_id);

    void set_vocab_bias(const std::unordered_map<uint32_t, float>& bias) { vocab_bias_ = bias; }
    void clear_vocab_bias() { vocab_bias_.clear(); }
    bool has_vocab_bias() const { return !vocab_bias_.empty(); }
    const std::unordered_map<uint32_t, float>& get_vocab_bias() const { return vocab_bias_; }

    bool has_handoff_probe() const { return handoff_probe_loaded_; }
    void reset_handoff_probe_rollout() { handoff_probe_hidden_.clear(); }
    bool has_handoff_probe_rollout() const;
    float handoff_probe_wrong_probability() const;

private:
    struct Binding {
        int node_id = -1;
        std::string path;
    };

    struct CacheStateBinding {
        std::string layer_key;
        int key_node_id = -1;
        int value_node_id = -1;
    };

    struct Component {
        std::string name;
        std::string graph_path;
        std::vector<int> runtime_input_node_ids;
        std::vector<std::string> logical_inputs;
        std::vector<int> output_node_ids;
        std::vector<std::string> logical_outputs;
        std::vector<Binding> bindings;
        std::vector<CacheStateBinding> cache_states;
        std::map<std::string, std::string> metadata;
        std::unique_ptr<CactusGraph> graph;
        std::vector<std::vector<uint8_t>> input_buffers;
    };

    void copy_cache_state(const Component& src, Component& dst);

    struct ChunkedPrefillResult {
        size_t logical_tokens = 0;
        size_t executed_tokens = 0;
        size_t padding_tokens = 0;
        size_t scalar_tail_tokens = 0;
        size_t last_logit_row = 0;
    };

    bool load_manifest();
    bool setup_tokenizer();
    bool load_components(const std::unordered_set<std::string>& required_components);
    bool load_component_graph(Component& comp);
    void unload_component_graph(Component& comp);
    bool bind_runtime_buffers(Component& comp);
    void run_step(uint32_t token_id, size_t position, bool read_logits, bool use_fused = true);
    void run_step_batch(const std::vector<uint32_t>& token_ids, const std::vector<size_t>& positions);
    void set_component_batch(Component& comp, size_t batch);
    size_t decoder_cache_num_slots();
    void run_encoder_step(uint32_t token_id, size_t position);
    void run_media_step(size_t position, const uint8_t* feature_row, size_t feature_row_bytes,
                        Precision feature_precision);
    void write_media_embeds_row(Component& comp, int embeds_idx, const uint8_t* feature_row,
                                size_t feature_row_bytes, Precision feature_precision);
    void reset_encoder_cross_kv_route_state();
    bool finish_encoder_cross_kv_prepare();
    bool finish_encoder_cross_kv_prepare_after_source();
    bool prepare_encoder_cross_kv_from_text(const std::vector<uint32_t>& tokens);
    bool prepare_encoder_cross_kv_from_audio(const std::vector<float>& audio_features);
    bool run_encoder_cross_kv_decoder_step(uint32_t token_id, size_t position);
    std::vector<uint32_t> run_encoder_cross_kv_decode_loop(
        const std::vector<uint32_t>& decoder_prompt_tokens,
        size_t max_tokens,
        const std::vector<std::vector<uint32_t>>& stop_token_sequences,
        const std::atomic<bool>* should_stop);
    void copy_component_outputs_to_inputs(const Component& source, Component& target);
    bool copy_cross_kv_outputs_to_decoder_cache_inputs(const Component& source, Component& target, size_t source_len);
    void copy_encoder_outputs_to_decoder(const Component& enc);
    void copy_component_outputs_to_chunk_inputs(const Component& source, Component& target, size_t token_index);
    void copy_component_outputs_to_chunk_inputs_range(const Component& source, Component& target, size_t token_offset);
    bool cache_states_compatible(const Component& source, const Component& target) const;
    void move_cache_states(Component& source, Component& target, size_t logical_current = std::numeric_limits<size_t>::max());
    void set_cache_current_len(Component& comp, size_t len);
    void reset_component_cache_states(Component& comp);
    void reset_prefill_stats();
    size_t component_chunk_tokens(const Component& comp, const std::string& input_name) const;
    size_t component_output_tokens(const Component& comp, const std::string& output_name) const;
    ChunkedPrefillResult run_chunked_prefill(const std::vector<uint32_t>& tokens, size_t start_position,
                                             size_t chunk_size, bool prepare_decode);
    void execute_prefill_chunk(Component& chunk_comp, Component* enc_comp, size_t encoder_chunk,
                               size_t chunk_tokens, const std::vector<uint32_t>& tokens,
                               size_t processed, size_t start_position);
    void run_full_context_text();
    void prepare_sampling_context(float repetition_penalty);
    uint32_t sample_component_logits(Component& comp, float temperature, float top_p, size_t top_k,
                                     float min_p, bool greedy, float* out_uncertainty);
    uint32_t argmax_component_logits(Component& comp, size_t logit_row = std::numeric_limits<size_t>::max(),
                                     float* out_uncertainty = nullptr);
    uint32_t argmax_logits_at(const BufferDesc& desc, void* ptr, size_t row_off, float* out_uncertainty);
    std::vector<uint32_t> argmax_component_logits_batch(Component& comp, size_t batch);
    void write_int_input(Component& comp, const std::string& name, int64_t value);
    void write_int_input_at(Component& comp, const std::string& name, size_t index, int64_t value);
    void write_bytes_input(Component& comp, const std::string& name, const void* data, size_t byte_size);
    int input_index(const Component& comp, const std::string& name) const;
    int output_index(const Component& comp, const std::string& name) const;
    uint32_t argmax_last_logits(float* out_uncertainty = nullptr);
    bool load_handoff_probe();
    void maybe_capture_handoff_probe_hidden(const Component& comp, const std::string& output_name = "probe_hidden");
    void run_vision_encoder(const std::string& image_path);
    void run_vision_encoder_lfm2_vl(const std::string& image_path);
    void encode_lfm2_vl_image_into_features(const std::string& image_path);
    bool load_lfm2_vl_position_grid();
    bool lfm2_vl_use_npu_vision() const;
    bool lfm2_vl_encode_tile_npu(const float* pixel_values, const int64_t* mask, const float* pos_embeds,
                                 size_t max_patches, int dim, size_t patch_dim, std::vector<float>& enc_out);
    void run_audio_encoder(const std::vector<float>& audio_features);
    void run_audio_encoder_messages(const std::vector<std::vector<float>>& audio_features_per_message);
    bool run_chunk_prefill_path(const std::vector<uint32_t>& tokens,
                                const std::vector<std::string>& image_paths,
                                const std::vector<std::vector<float>>& audio_features_per_message);
    bool build_lm_encoder_outputs_dynamic_gemma4(
        const std::vector<uint32_t>& tokens,
        std::map<std::string, std::vector<uint8_t>>& store_bytes,
        std::map<std::string, Precision>& store_prec,
        std::map<std::string, std::vector<size_t>>& store_shape);

    std::string bundle_dir_;
    std::map<std::string, Component> components_;
    Component* encoder_ = nullptr;
    Component* decoder_ = nullptr;
    Component* decoder_prefill_ = nullptr;
    Component* decoder_embed_ = nullptr;
    Component* prefill_encoder_ = nullptr;
    enum class DecodeRoute { CACHED_STEP, DIRECT_DECODER_STEP, FULL_CONTEXT_TEXT, ENCODER_CROSS_KV_STEP };
    DecodeRoute decode_route_ = DecodeRoute::CACHED_STEP;
    Component* source_encoder_ = nullptr;
    Component* decoder_cross_kv_ = nullptr;
    Component* vision_encoder_ = nullptr;
    Component* vision_projector_ = nullptr;
    Component* audio_encoder_ = nullptr;
    Component* lm_encoder_media_step_ = nullptr;
    Component* decoder_prefill_chunk_ = nullptr;
    Component* lm_encoder_ = nullptr;
    Component* lm_encoder_text_chunk_ = nullptr;
    Component* lm_encoder_media_chunk_ = nullptr;
    std::string encoder_cross_kv_source_kind_;
    bool encoder_cross_kv_ready_ = false;
    size_t encoder_cross_kv_source_len_ = 0;
    bool prefill_tail_pad_disabled_ = false;

    std::string family_;
    std::string npu_audio_encoder_mlpackage_;
    std::string npu_audio_compute_units_;
    std::string npu_vision_encoder_mlpackage_;
    std::string npu_source_encoder_mlpackage_;

    std::unique_ptr<npu::NPUEncoder> npu_audio_encoder_;
    std::unique_ptr<npu::NPUEncoder> npu_vision_encoder_;
    std::unique_ptr<npu::NPUEncoder> npu_source_encoder_;

    bool audio_encode_via_npu(const std::vector<float>& audio_features);
    bool vision_encode_via_npu(const std::vector<float>& pixel_values,
                               const std::vector<int64_t>* pixel_position_ids = nullptr);
    bool source_encode_via_npu(const std::vector<uint32_t>& tokens);

    std::map<std::string, std::vector<uint8_t>> media_features_;
    std::map<std::string, std::vector<size_t>> media_feature_shapes_;
    std::map<std::string, Precision> media_feature_precisions_;

    std::vector<float> lfm2_pos_grid_;
    int lfm2_pos_grid_h_ = 0;
    int lfm2_pos_grid_w_ = 0;
    int lfm2_pos_grid_dim_ = 0;
    bool lfm2_pos_grid_loaded_ = false;

    Config config_;
    std::unique_ptr<Tokenizer> tokenizer_;
    bool initialized_ = false;
    size_t cache_total_seq_len_ = 0;
    std::vector<uint32_t> cache_token_ids_;        // token id per cache row (canonical head-0 view)
    std::unordered_set<uint32_t> special_ids_;     // special-token ids force-kept during compaction
    cactus::kvcompress::SpecialRowTracker special_rows_;  // per-(layer,head) special rows for compaction protect
    size_t cache_max_seq_len_ = 4096;
    size_t last_logit_position_ = 0;
    double last_prefill_cache_copy_ms_ = 0.0;
    size_t last_prefill_padding_tokens_ = 0;
    size_t last_prefill_scalar_tail_tokens_ = 0;
    size_t last_prefill_tail_chunk_tokens_ = 0;
    size_t last_prefill_tail_padding_tokens_ = 0;
    std::vector<uint32_t> context_tokens_;

    static constexpr size_t MAX_TOKEN_HISTORY = 128;
    std::vector<uint32_t> token_history_;
    std::vector<uint32_t> samp_recent_;
    std::vector<float> samp_bias_dense_;
    bool samp_has_bias_ = false;
    float samp_penalty_ = 1.0f;
    bool samp_ctx_active_ = false;
    std::mt19937 sample_rng_{std::random_device{}()};
    FusedEmbedCtx fused_embed_ctx_;
    int ple_probe_state_ = 0;

    ToolCallConstrainer tool_constrainer_;
    std::unordered_map<uint32_t, float> vocab_bias_;
    int64_t suppressed_token_id_ = -1;

    bool handoff_probe_loaded_ = false;
    uint32_t handoff_probe_feat_dim_ = 0;
    uint32_t handoff_probe_t_h_ = 0;
    uint32_t handoff_probe_h1_ = 0;
    uint32_t handoff_probe_h2_ = 0;
    std::vector<float> handoff_probe_norm_weight_;
    std::vector<float> handoff_probe_norm_bias_;
    std::vector<float> handoff_probe_proj_weight_;
    std::vector<float> handoff_probe_proj_bias_;
    std::vector<float> handoff_probe_attn_query_;
    std::vector<float> handoff_probe_head0_weight_;
    std::vector<float> handoff_probe_head0_bias_;
    std::vector<float> handoff_probe_head2_weight_;
    std::vector<float> handoff_probe_head2_bias_;
    std::vector<float> handoff_probe_head4_weight_;
    std::vector<float> handoff_probe_head4_bias_;
    std::vector<float> handoff_probe_hidden_;

    mutable std::vector<DebugNode> debug_nodes_;
};

class ConvCache {
public:
    struct CircularView {
        const void* ptr1;
        size_t len1;
        const void* ptr2;
        size_t len2;
        size_t total_len;
    };

    void init(size_t layers, size_t hidden_dim, size_t window_len, Precision model_precision);
    CircularView get_window(size_t layer) const;
    void update(CactusGraph* gb, size_t layer, const size_t latest_token);
    void reset();

    bool is_empty() const { return num_layers == 0; }

    size_t num_layers = 0;
    size_t hidden_size = 0;
    size_t window_size = 0;
    Precision precision = Precision::FP32;
    size_t element_size = 4;

private:
    struct LayerState {
        std::vector<uint8_t> data;
        size_t head = 0;
        size_t count = 0;
    };

    std::vector<LayerState> layer_states;
};

class Siglip2Preprocessor {
public:
    struct Config {
        int patch_size = 16;
        int downsample_factor = 2;
        int min_tiles = 2;
        int max_tiles = 10;
        bool use_thumbnail = true;
        int min_image_tokens = 64;
        int max_image_tokens = 256;
        int max_num_patches = 1024;
        int tile_size = 512;
        float max_pixels_tolerance = 2.0f;
        bool do_resize = true;
        bool do_rescale = true;
        bool do_normalize = true;
        bool do_convert_rgb = true;
        bool do_image_splitting = true;
        float rescale_factor = 1.0f / 255.0f;
        float image_mean[3] = {0.5f, 0.5f, 0.5f};
        float image_std[3] = {0.5f, 0.5f, 0.5f};
    };

    struct PreprocessedImage {
        std::vector<float> pixel_values;
        std::vector<int> pixel_attention_mask;
        std::vector<std::pair<int, int>> spatial_shapes;
        std::vector<size_t> pixel_values_shape;
        std::vector<size_t> pixel_attention_mask_shape;
        std::vector<size_t> spatial_shapes_shape;
        int num_patches_height;
        int num_patches_width;
        int actual_num_patches;
        int num_tiles;
        int patch_dim;
        int max_patches_per_tile;
        int image_rows;
        int image_cols;
        int image_height;
        int image_width;
        int tokens_per_tile;
        int thumbnail_tokens;

        ~PreprocessedImage();
    };

    struct SpatialShapeResult {
        std::vector<std::pair<int, int>> shapes;
        int grid_rows;
        int grid_cols;
    };

    explicit Siglip2Preprocessor(const Config& config);
    Siglip2Preprocessor();
    ~Siglip2Preprocessor();

    PreprocessedImage preprocess_from_file(const std::string& image_path);
    PreprocessedImage preprocess_from_memory(const unsigned char* img_data, int width, int height, int channels);
    SpatialShapeResult compute_spatial_shapes(int height, int width);

private:
    Config config_;

    std::pair<int64_t, int64_t> compute_pixel_limits() const;
    std::vector<unsigned char> convert_to_rgb(const unsigned char* img_data, int width, int height, int channels);
    std::pair<int, int> smart_resize(int height, int width);
    bool is_image_too_large(int height, int width);
    std::pair<int, int> get_grid_layout(int height, int width);
    std::pair<int, int> find_closest_aspect_ratio(float aspect_ratio, int width, int height);
    std::vector<float> resize_image(const unsigned char* img_data, int src_width, int src_height,
                                    int dst_width, int dst_height, int channels);
    std::vector<float> normalize_image(const float* img_data, int width, int height, int channels);
    std::vector<std::vector<float>> convert_image_to_patches(
        const std::vector<float>& image, int width, int height, int channels, int patch_size);
    PreprocessedImage pad_patches(const std::vector<std::vector<float>>& tile_patches,
                                  const std::vector<std::pair<int, int>>& spatial_shapes,
                                  int patch_dim,
                                  int max_patches_per_tile);
    int round_by_factor(int number, int factor);
};

std::unique_ptr<Model> create_model(const std::string& model_folder);

struct Gemma4ImagePreprocessed {
    std::vector<float> pixel_values;
    std::vector<int64_t> pixel_position_ids;
    size_t num_patches = 0;
    size_t max_patches = 0;
    size_t patch_dim = 0;
};

Gemma4ImagePreprocessed preprocess_gemma4_image(const std::string& image_path, const Config& config);

struct Lfm2VlImagePreprocessed {
    std::vector<float> pixel_values;
    std::vector<int64_t> pixel_attention_mask;
    std::vector<std::pair<int, int>> spatial_shapes;
    size_t max_num_patches = 0;
    size_t patch_dim = 0;
};

Lfm2VlImagePreprocessed preprocess_lfm2_vl_image(const std::string& image_path, const Config& config);

void interpolate_position_embeddings(const float* grid, int grid_h, int grid_w, int dim,
                                             int out_h, int out_w, float* out);

void pixel_unshuffle(const float* feature, int h, int w, int dim, int factor, float* out);

struct Lfm2VlTokenLayout {
    int grid_rows = 1;
    int grid_cols = 1;
    int tokens_per_tile = 0;
    int thumbnail_tokens = 0;
    bool has_thumbnail = false;
};

Lfm2VlTokenLayout lfm2_vl_token_layout(int image_height, int image_width, const Config& config);

struct Qwen3VlImagePreprocessed {
    std::vector<float> pixel_values;
    size_t grid_t = 1;
    size_t grid_h = 0;
    size_t grid_w = 0;
    size_t patch_dim = 0;
};

Qwen3VlImagePreprocessed preprocess_qwen3_vl_image(const std::string& image_path, const Config& config);


struct SpectrogramConfig {
    size_t n_fft = 400;
    size_t hop_length = 160;
    size_t frame_length = 400;
    float power = 2.0f;
    bool center = true;
    const char* pad_mode = "reflect";
    bool onesided = true;
    float dither = 0.0f;
    float mel_floor = 1e-10f;
    const char* log_mel = nullptr;
    float reference = 1.0f;
    float min_value = 1e-10f;
    bool remove_dc_offset = false;
    float preemphasis = 0.0f;
    bool hann_periodic = true;
    float window_a0 = 0.5f;
    size_t fft_override = 0;
    bool mel_floor_additive = false;
};

namespace index {
    constexpr uint32_t MAGIC = 0x43414354;
    constexpr uint32_t VERSION = 1;

    struct Document {
        int id;
        std::vector<float> embedding;
        std::string content;
        std::string metadata;
    };

    struct QueryResult {
        int doc_id;
        float score;

        QueryResult(int doc_id, float score) : doc_id(doc_id), score(score) {}
    };

    struct QueryOptions {
        size_t top_k = 10;
        float score_threshold = -1.0f;
    };

    class Index {
        public:
            Index(const std::string& index_path, const std::string& data_path, size_t embedding_dim);
            ~Index();

            Index(const Index&) = delete;
            Index& operator=(const Index&) = delete;
            Index(Index&&) = delete;
            Index& operator=(Index&&) = delete;

            void add_documents(const std::vector<Document>& documents);
            void delete_documents(const std::vector<int>& doc_ids);
            std::vector<Document> get_documents(const std::vector<int>& doc_ids);
            std::vector<std::vector<QueryResult>> query(const std::vector<std::vector<float>>& embeddings, const QueryOptions& options);
            void compact();

        private:
            struct IndexHeader {
                uint32_t magic;
                uint32_t version;
                uint32_t embedding_dim;
                uint32_t num_documents;
            };

            struct IndexEntry {
                int32_t doc_id;
                uint64_t data_offset;
                uint8_t flags; // bit 0: tombstone

                const __fp16* embedding() const {
                    return reinterpret_cast<const __fp16*>(this + 1);
                }

                static size_t size(size_t embedding_dim) {
                    return sizeof(IndexEntry) + embedding_dim * sizeof(__fp16);
                }
            };

            struct DataHeader {
                uint32_t magic;
                uint32_t version;
            };

            struct DataEntry {
                uint16_t content_len;
                uint16_t metadata_len;

                const char* content() const {
                    return reinterpret_cast<const char*>(this + 1);
                }

                const char* metadata() const {
                    return content() + content_len;
                }
            };

            void parse_index_header();
            void parse_data_header();
            void build_doc_id_map();
            void validate_documents(const std::vector<Document>& documents);
            void validate_doc_ids(const std::vector<int>& doc_ids);
            ssize_t write_full(int fd, const void* buf, size_t count);

            std::unordered_map<int, uint32_t> doc_id_map_;

            std::string index_path_, data_path_;
            size_t embedding_dim_;
            size_t index_entry_size_;
            uint32_t num_documents_;

            int index_fd_, data_fd_;
            void *mapped_index_, *mapped_data_;
            size_t index_file_size_, data_file_size_;
    };
} // namespace index

}
}
