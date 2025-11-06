#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <cstdint>

#include "../graph/graph.h"

class CactusGraph;

namespace cactus {
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
    bool tie_word_embeddings = true;

    enum class ModelType {QWEN = 0, GEMMA = 1, SMOL = 2, NOMIC = 3, LFM2 = 4};
    ModelType model_type = ModelType::QWEN;

    enum class Activation {GELU = 0, SILU = 1};
    Activation activation = Activation::SILU;

    enum class Backend {CPU = 0, NPU = 1};
    Backend default_backend = Backend::CPU;

    enum class Precision {INT8 = 0, FP16 = 1, FP32 = 2};
    Precision precision = Precision::FP32;

    float default_temperature = 0.6f;
    float default_top_p = 0.95f;
    size_t default_top_k = 20;

    std::vector<std::string> layer_types;
    size_t conv_L_cache = 0;

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


struct ChatMessage {
    std::string role;
    std::string content;
};

class Tokenizer {
public:
    virtual ~Tokenizer() = default;

    virtual std::vector<uint32_t> encode(const std::string& text) const = 0;
    virtual std::string decode(const std::vector<uint32_t>& tokens) const = 0;

    virtual std::vector<uint32_t> apply_chat_template(const std::vector<ChatMessage>& messages, bool add_generation_prompt = true) const;
    virtual std::string format_chat_prompt(const std::vector<ChatMessage>& messages, bool add_generation_prompt = true, const std::string& tools_json = "") const;

    virtual uint32_t get_vocab_size() const = 0;
    virtual uint32_t get_unk_token() const = 0;
    virtual uint32_t get_bos_token() const = 0;
    virtual uint32_t get_eos_token() const = 0;
    virtual bool has_chat_template() const { return has_chat_template_; }

    virtual bool load_vocabulary_with_config(const std::string& vocab_file, const std::string& merges_file, const std::string& config_file) = 0;

protected:

    enum class ModelType { UNKNOWN, QWEN, GEMMA, LFM2 , SMOL, BERT };
    ModelType model_type_ = ModelType::UNKNOWN;
    bool has_chat_template_ = false;
    std::string chat_template_;

    void detect_model_type(const std::string& config_path);
    std::string format_qwen_style(const std::vector<ChatMessage>& messages, bool add_generation_prompt, const std::string& tools_json) const;
    std::string format_gemma_style(const std::vector<ChatMessage>& messages, bool add_generation_prompt, const std::string& tools_json) const;
    std::string format_lfm2_style(const std::vector<ChatMessage>& messages, bool add_generation_prompt, const std::string& tools_json) const;
    std::string format_smol_style(const std::vector<ChatMessage>& messages, bool add_generation_prompt, const std::string& tools_json) const;
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

    void cleanup_mmap();
    
private:
    mutable std::unordered_map<uint8_t, std::string> byte_to_unicode_;
    mutable std::unordered_map<std::string, uint8_t> unicode_to_byte_;
    void init_byte_mappings() const;

    std::unordered_map<std::string, uint32_t> special_tokens_;
    std::vector<std::string> split_with_special_tokens(const std::string& text) const;
    void load_special_tokens(const std::string& config_file);

    void load_chat_template(const std::string& template_file);

    std::unordered_map<std::string, uint32_t> tool_tokens_;
    bool has_tool_support_;
    void load_tokenizer_config(const std::string& config_file);
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
    
    void* vocab_mmap_ptr_;
    size_t vocab_mmap_size_;
    
    void build_trie();
    std::vector<std::pair<std::string, uint32_t>> tokenize_with_trie(const std::string& text) const;
    std::string preprocess_text(const std::string& text) const;
    std::string postprocess_text(const std::string& text) const;
    std::vector<std::string> split_by_unicode_spaces(const std::string& text) const;
    
    void cleanup_mmap();

    std::unordered_map<std::string, uint32_t> special_tokens_;
    std::vector<std::string> split_with_special_tokens(const std::string& text) const;
    void load_special_tokens(const std::string& config_file);

    void load_chat_template(const std::string& template_file);
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

struct KVCache {
    static constexpr size_t DEFAULT_WINDOW_SIZE = 512;
    static constexpr size_t DEFAULT_SINK_SIZE = 4;

    struct LayerCache {
        std::vector<uint8_t> keys;
        std::vector<uint8_t> values;
    };

    std::vector<LayerCache> layer_caches;

    size_t window_size = DEFAULT_WINDOW_SIZE;  
    size_t sink_size = DEFAULT_SINK_SIZE;    
    size_t current_seq_len = 0;
    size_t total_seq_len = 0;
    size_t max_seq_len = 2048;
    size_t num_kv_heads = 0;
    size_t head_dim = 0;
    size_t num_layers = 0;
    Precision precision;
    size_t element_size = 4;

    void set_window_size(size_t window, size_t sink = DEFAULT_SINK_SIZE);
    size_t get_effective_seq_len() const { return current_seq_len; }
    size_t get_total_seq_len() const { return total_seq_len; }

    void init(size_t num_layers, size_t max_seq, size_t num_kv_heads, size_t head_dim, Precision model_precision);
    void reset();
    void update_from_graph(CactusGraph* gb, const std::vector<size_t>& k_nodes,
                          const std::vector<size_t>& v_nodes, size_t seq_len,
                          size_t num_layers, size_t kv_heads, size_t head_dim);
    bool is_empty() const { return current_seq_len == 0; }
    void* get_key_ptr(size_t layer);
    void* get_value_ptr(size_t layer);

    struct CircularView {
        const void* ptr1;
        const void* ptr2;  
        size_t len1;
        size_t len2; 
        size_t total_len;
    };

    CircularView get_key_view(size_t layer);
    CircularView get_value_view(size_t layer);
};

class Model {
public:
    Model();
    explicit Model(const Config& config);
    virtual ~Model();

    const Config& get_config() const { return config_; }
    Tokenizer* get_tokenizer() const { return tokenizer_.get(); }

    bool init(const std::string& model_folder, size_t context_size, const std::string& system_prompt = "");
    uint32_t generate(const std::vector<uint32_t>& tokens, float temperature = -1.0f, float top_p = -1.0f,
                      size_t top_k = 0, const std::string& profile_file = "");

    std::vector<float> get_embeddings(const std::vector<uint32_t>& tokens, bool pooled = true, const std::string& profile_file = "");

    virtual void reset_cache() { kv_cache_.reset(); }
    void set_cache_window(size_t window_size, size_t sink_size = 4) { kv_cache_.set_window_size(window_size, sink_size); }

protected:
    virtual size_t forward(const std::vector<uint32_t>& tokens, bool use_cache = false) = 0;
    virtual void load_weights_to_graph(CactusGraph* gb) = 0;
    virtual size_t build_attention(CactusGraph* gb, size_t normalized_input, uint32_t layer_idx,
                          ComputeBackend backend, bool use_cache = false, size_t position_offset = 0) = 0;
    virtual size_t build_mlp(CactusGraph* gb, size_t normalized_h, uint32_t layer_idx,
                    ComputeBackend backend) const = 0;
    virtual size_t build_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                  ComputeBackend backend, bool use_cache = false, size_t position_offset = 0) = 0;
    void update_kv_cache(CactusGraph* gb, size_t seq_len);
    virtual void post_init() {}
    virtual void post_execute_updates(CactusGraph*, size_t) {}
    Config config_;
    std::unique_ptr<Tokenizer> tokenizer_;

    void* graph_handle_;
    bool initialized_;
    float attention_scale_;

protected:
    KVCache kv_cache_;
    std::vector<size_t> cache_k_output_nodes_;
    std::vector<size_t> cache_v_output_nodes_;

    std::string embedding_file_path_;
    size_t embedding_node_id_;
    std::string model_folder_path_;
    size_t output_weight_node_id_;
};

std::unique_ptr<Model> create_model(const std::string& model_folder);

}
}