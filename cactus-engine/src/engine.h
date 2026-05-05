#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <cstdint>

#include "cactus_graph.h"

class CactusGraph;

namespace cactus {
namespace npu {

struct NPUNamedInput {
    std::string name;
    const __fp16* data;
    std::vector<int> shape;
};

class NPUEncoder {
public:
    virtual ~NPUEncoder() = default;
    virtual bool load(const std::string& model_path) = 0;
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
    virtual __fp16* get_output_buffer() = 0;
    virtual size_t get_output_buffer_size() const = 0;
    virtual size_t encode_multimodal_input(
        const std::vector<NPUNamedInput>& inputs,
        __fp16* output,
        const std::string& output_name = "") = 0;
};

std::unique_ptr<NPUEncoder> create_encoder();
bool is_npu_available();

struct NPUBufferRef {
    const __fp16* data;
    size_t count;
};

struct NPUPrefillDirectResult {
    NPUBufferRef hidden;
    std::vector<NPUBufferRef> k_caches;
    std::vector<NPUBufferRef> v_caches;
    bool valid;
};

class NPUPrefill {
public:
    virtual ~NPUPrefill() = default;
    virtual bool load(const std::string& model_path) = 0;
    virtual bool is_available() const = 0;
    virtual int get_chunk_size() const = 0;
    virtual int get_hidden_dim() const = 0;
    virtual int get_num_layers() const = 0;
    virtual int get_num_kv_heads() const = 0;
    virtual int get_head_dim() const = 0;
    virtual NPUPrefillDirectResult prefill_chunk_direct(
        const std::vector<__fp16>& embeddings,
        int position_offset = 0,
        const std::string& input_name = "x") = 0;
};

std::unique_ptr<NPUPrefill> create_prefill();

} // namespace npu
namespace engine {


struct Config {
    uint32_t vocab_size = 151936;
    uint32_t bos_token_id = 151643;
    uint32_t eos_token_id = 151645;
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

    enum class ModelType {GEMMA4 = 15};
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
    size_t default_rolling_entropy_window = 10;

    std::vector<std::string> layer_types;
    size_t conv_L_cache = 0;

    uint32_t altup_num_inputs = 4;
    uint32_t laurel_rank = 64;
    uint32_t hidden_size_per_layer_input = 256;
    uint32_t num_kv_shared_layers = 0;
    uint32_t sliding_window = 512;
    float rope_local_base_freq = 10000.0f;
    float final_logit_softcapping = 0.0f;
    float global_partial_rotary_factor = 1.0f;
    uint32_t expert_intermediate_size = 0;
    uint32_t global_head_dim = 0;
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
    uint32_t channel_open_token_id = 100;
    uint32_t channel_close_token_id = 101;

    static bool is_gemma_family(ModelType t) {
        return t == ModelType::GEMMA4;
    }

    bool from_json(const std::string& json_path);
    std::string to_json() const;
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
    virtual bool has_chat_template() const { return has_chat_template_; }
    std::string get_default_stop_sequence() const;

    virtual bool load_vocabulary_with_config(const std::string& vocab_file, const std::string& merges_file, const std::string& config_file) = 0;
    
    uint32_t get_image_token_id() const { return image_token_id_; }
    uint32_t get_fake_token_id() const { return fake_token_id_; }
    uint32_t get_global_img_token_id() const { return global_img_token_id_; }

protected:
    enum class ModelType { UNKNOWN, GEMMA4 };
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
    TokenizerRuntimeConfig runtime_config_;

    void detect_model_type(const std::string& config_path);
    void load_chat_template(const std::string& template_file);
    std::string format_gemma4_style(const std::vector<ChatMessage>& messages, bool add_generation_prompt, const std::string& tools_json, bool enable_thinking_if_supported = false) const;
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
        GEMMA_IN_FUNC_NAME,
        GEMMA_EXPECT_BRACE,
        GEMMA_IN_ARGUMENTS,
        GEMMA_EXPECT_END
    };

    void init(Config::ModelType model_type,
              const std::vector<ToolConstraintSpec>& tools,
              Tokenizer* tokenizer);

    const std::unordered_map<uint32_t, float>& get_bias() const { return current_bias_; }

    void update(uint32_t token_id, const std::string& decoded_text);

    void reset();

    bool is_active() const { return active_; }

private:
    bool active_ = false;
    State state_ = State::GEMMA_START;
    Config::ModelType model_type_ = Config::ModelType::GEMMA4;
    Tokenizer* tokenizer_ = nullptr;

    std::vector<ToolConstraintSpec> tool_specs_;
    std::vector<std::string> function_names_;
    std::string generated_text_;
    int brace_depth_ = 0;

    std::string call_start_tag_;
    std::string call_end_tag_;

    std::unordered_set<uint32_t> all_func_name_tokens_;
    std::unordered_map<std::string, std::vector<uint32_t>> func_name_sequences_;

    std::unordered_set<uint32_t> gemma_call_start_tokens_;
    std::unordered_set<uint32_t> gemma_call_end_tokens_;
    std::unordered_set<uint32_t> gemma_response_start_tokens_;
    std::unordered_set<uint32_t> gemma_call_prefix_tokens_;
    std::unordered_set<uint32_t> escape_tokens_;

    std::unordered_set<uint32_t> backtick_tokens_;
    std::unordered_set<uint32_t> open_brace_tokens_;
    std::unordered_set<uint32_t> close_brace_tokens_;
    std::unordered_set<uint32_t> colon_tokens_;
    std::unordered_set<uint32_t> comma_tokens_;

    std::unordered_map<uint32_t, float> current_bias_;

    void compute_bias();
    void tokenize_grammar_elements();
    void add_tokens_for_string(const std::string& str, std::unordered_set<uint32_t>& token_set);
    void add_tokens_for_prefix_string(const std::string& prefix, std::unordered_set<uint32_t>& token_set);
    void tokenize_function_names(bool quote_names);
    void init_common_tokens();
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
    virtual ~Model();

    const Config& get_config() const { return config_; }
    Tokenizer* get_tokenizer() const { return tokenizer_.get(); }
    const std::vector<DebugNode>& get_debug_nodes() const;

    virtual bool init(const std::string& model_folder, size_t context_size, const std::string& system_prompt = "", bool do_warmup = true);
    
    virtual bool init(CactusGraph* external_graph, const std::string& model_folder, size_t context_size,
              const std::string& system_prompt = "", bool do_warmup = true);

    virtual uint32_t decode(const std::vector<uint32_t>& tokens, float temperature = -1.0f, float top_p = -1.0f,
                      size_t top_k = 0, const std::string& profile_file = "", float* out_entropy = nullptr,
                      float min_p = 0.15f, float repetition_penalty = 1.1f);

    virtual void prefill(const std::vector<uint32_t>& tokens, size_t chunk_size = 256, const std::string& profile_file = "");

    virtual void prefill_with_images(const std::vector<uint32_t>& tokens, const std::vector<std::string>& image_paths,
                                     const std::string& profile_file = "");

    virtual uint32_t decode_with_images(const std::vector<uint32_t>& tokens, const std::vector<std::string>& image_paths,
                                          float temperature = -1.0f, float top_p = -1.0f,
                                          size_t top_k = 0, const std::string& profile_file = "", float* out_entropy = nullptr,
                                          float min_p = 0.15f, float repetition_penalty = 1.1f);

    virtual uint32_t decode_with_audio(const std::vector<uint32_t>& tokens, const std::vector<float>& audio_features, float temperature = 0.0f, float top_p = 0.0f,
                      size_t top_k = 0, const std::string& profile_file = "", float* out_entropy = nullptr,
                      float min_p = 0.15f, float repetition_penalty = 1.1f,
                      float* out_token_time_start = nullptr, float* out_token_time_end = nullptr);

    std::vector<float> get_embeddings(const std::vector<uint32_t>& tokens, bool pooled = true, bool normalize = false, const std::string& profile_file = "");
    
    virtual std::vector<float> get_image_embeddings(const std::string& image_path);
    
    virtual std::vector<float> get_audio_embeddings(const std::vector<float>& audio_features);

    virtual void reset_cache();
    void record_sampled_token(uint32_t token) {
        if (token_history_.size() >= MAX_TOKEN_HISTORY) {
            token_history_.erase(token_history_.begin(), token_history_.begin() + (MAX_TOKEN_HISTORY / 2));
        }
        token_history_.push_back(token);
    }

    double score_tokens_window_logprob(const std::vector<uint32_t>& tokens, size_t start, size_t end, size_t context, size_t* tokens_scored);



    void set_cache_window(size_t window_size, size_t sink_size = 4);
    size_t get_cache_size() const;

    bool load_npu_prefill(const std::string& model_path);
    bool has_npu_prefill() const;
    size_t get_prefill_chunk_size() const;

    virtual void remove_thinking_tokens(const std::vector<std::pair<size_t, size_t>>& ranges);
    virtual void compact_kv_cache() {}

    virtual void set_tool_constraints(const std::vector<ToolConstraintSpec>& tools);
    virtual void clear_tool_constraints();
    virtual void update_tool_constraints(uint32_t token_id);

    void* graph_handle_;

    void set_vocab_bias(const std::unordered_map<uint32_t, float>& bias) {
        vocab_bias_ = bias;
    }

    void clear_vocab_bias() {
        vocab_bias_.clear();
    }

    bool has_vocab_bias() const {
        return !vocab_bias_.empty();
    }

    const std::unordered_map<uint32_t, float>& get_vocab_bias() const {
        return vocab_bias_;
    }

protected:
    size_t sample_token(CactusGraph* gb, size_t logits_node_id, float temperature, float top_p, size_t top_k,
                        float min_p, float repetition_penalty,
                        const std::unordered_map<uint32_t, float>* extra_bias = nullptr) const;

    static void compute_entropy(CactusGraph* gb, size_t logits_node_id, float* out_entropy);

    virtual size_t forward(const std::vector<uint32_t>& tokens, bool use_cache = false) = 0;
    
    virtual size_t forward(const std::vector<float>& audio_features, const std::vector<uint32_t>& tokens, bool use_cache = false);
    
    virtual void load_weights_to_graph(CactusGraph* gb) = 0;
    
    virtual size_t build_attention(CactusGraph* gb, size_t normalized_input, uint32_t layer_idx,
                          ComputeBackend backend, bool use_cache = false, size_t position_offset = 0) = 0;
    
                          virtual size_t build_mlp(CactusGraph* gb, size_t normalized_h, uint32_t layer_idx,
                    ComputeBackend backend) const = 0;
    virtual size_t build_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                  ComputeBackend backend, bool use_cache = false, size_t position_offset = 0) = 0;
    virtual std::vector<size_t> get_kv_layer_dims() const {
        return std::vector<size_t>(config_.num_layers, config_.attention_head_dim);
    }
    virtual std::vector<size_t> get_kv_layer_heads() const {
        return std::vector<size_t>(config_.num_layers, config_.attention_kv_heads);
    }
    virtual std::vector<size_t> get_kv_layer_windows() const {
        return std::vector<size_t>(config_.num_layers, 0);
    }
    virtual void post_init() {}
    virtual void post_execute_updates(CactusGraph*, size_t) {}
    Config config_;
    std::unique_ptr<Tokenizer> tokenizer_;

    bool initialized_;
    float attention_scale_;

protected:
    std::vector<size_t> graph_cache_k_nodes_;
    std::vector<size_t> graph_cache_v_nodes_;
    size_t cache_total_seq_len_ = 0;
    size_t cache_window_size_ = 0;
    size_t cache_sink_size_ = 4;
    size_t cache_max_seq_len_ = 2048;
    void init_graph_cache(CactusGraph* gb);
    void invalidate_graph_cache(CactusGraph* gb);

    std::string embedding_file_path_;
    size_t embedding_node_id_;
    std::string model_folder_path_;
    size_t output_weight_node_id_;

    mutable std::vector<DebugNode> debug_nodes_;

    void capture_debug_node(uint32_t layer_idx, const std::string& name, size_t node_id) const;
    void clear_debug_nodes();

    bool init_internal(CactusGraph* gb, const std::string& model_folder, size_t context_size,
                       const std::string& system_prompt, bool do_warmup);
    bool owns_graph_;

    std::unique_ptr<npu::NPUPrefill> npu_prefill_;
    void prefill_npu(const std::vector<uint32_t>& tokens);
    virtual std::vector<__fp16> get_token_embeddings(const std::vector<uint32_t>& tokens);

    static constexpr size_t MAX_TOKEN_HISTORY = 128;
    ToolCallConstrainer tool_constrainer_;
    std::vector<uint32_t> token_history_;

private:
    std::unordered_map<uint32_t, float> vocab_bias_;
};

std::unique_ptr<Model> create_model(const std::string& model_folder);


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
