#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <cstdint>

class CactusGraph;
enum class ComputeBackend;
enum class Precision;

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
    bool tie_word_embeddings = true;
    
    enum class Activation {GELU = 0, SILU = 1};
    Activation activation = Activation::SILU;
    
    enum class Backend {CPU = 0, NPU = 1};
    Backend default_backend = Backend::CPU;
    
    enum class Precision {INT8 = 0, FP16 = 1, FP32 = 2};
    Precision precision = Precision::FP32;
    
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

class BPETokenizer {
public:
    BPETokenizer();
    ~BPETokenizer();

    bool load_vocabulary_mmap(const std::string& vocab_file, const std::string& merges_file);
    bool load_vocabulary_with_config(const std::string& vocab_file, const std::string& merges_file, const std::string& config_file);

    std::vector<uint32_t> encode(const std::string& text) const;
    std::string decode(const std::vector<uint32_t>& tokens) const;
    
    std::vector<uint32_t> apply_chat_template(const std::vector<ChatMessage>& messages, bool add_generation_prompt = true) const;
    std::string format_chat_prompt(const std::vector<ChatMessage>& messages, bool add_generation_prompt = true) const;

    uint32_t get_vocab_size() const { return vocab_size_; }
    uint32_t get_unk_token() const { return unk_token_id_; }
    uint32_t get_bos_token() const { return bos_token_id_; }
    uint32_t get_eos_token() const { return eos_token_id_; }
    bool has_chat_template() const { return has_chat_template_; }

private:
    std::unordered_map<std::string, uint32_t> token_to_id_;
    std::vector<std::string> id_to_token_;
    std::vector<MergeRule> merge_rules_;

    uint32_t vocab_size_;
    uint32_t unk_token_id_;
    uint32_t bos_token_id_;
    uint32_t eos_token_id_;

    void* vocab_mmap_ptr_;
    size_t vocab_mmap_size_;

    void* merges_mmap_ptr_;
    size_t merges_mmap_size_;

    std::vector<std::string> apply_bpe(const std::vector<std::string>& tokens) const;
    std::pair<int, int> find_best_merge(const std::vector<std::string>& tokens) const;
    
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
    
    bool has_chat_template_;
    std::string chat_template_;
    void load_chat_template(const std::string& template_file);
    std::string apply_template_substitutions(const std::string& template_str, const std::vector<ChatMessage>& messages, bool add_generation_prompt) const;
};



struct KVCache {
    static constexpr size_t CACHE_PAGE_SIZE = 16;
    
    struct Page {
        std::vector<uint8_t> data;
        int ref_count = 0;
        bool in_use = false;
    };
    
    struct LayerPageTable {
        std::vector<size_t> key_pages;
        std::vector<size_t> value_pages;
    };
    
    std::vector<Page> page_pool;
    std::vector<LayerPageTable> layer_tables;
    std::vector<size_t> free_pages;
    
    std::vector<std::vector<uint8_t>> continuous_keys;
    std::vector<std::vector<uint8_t>> continuous_values;
    
    size_t current_seq_len = 0;
    size_t max_seq_len = 2048;
    size_t num_kv_heads = 0;
    size_t head_dim = 0;
    size_t num_layers = 0;
    Precision precision;
    size_t element_size = 4;
    size_t tokens_per_page = CACHE_PAGE_SIZE;
    size_t bytes_per_page = 0;
    
    void init(size_t num_layers, size_t max_seq, size_t num_kv_heads, size_t head_dim, Precision model_precision);
    void reset();
    void update_from_graph(CactusGraph* gb, const std::vector<size_t>& k_nodes, 
                          const std::vector<size_t>& v_nodes, size_t seq_len, 
                          size_t num_layers, size_t kv_heads, size_t head_dim);
    bool is_empty() const { return current_seq_len == 0; }
    void* get_key_ptr(size_t layer);
    void* get_value_ptr(size_t layer);
    
private:
    size_t allocate_page();
    void free_page(size_t page_idx);
    void materialize_continuous_buffer(size_t layer);
    size_t get_num_pages_needed(size_t seq_len) const;
};

class Model {
public:
    Model();
    explicit Model(const Config& config);
    ~Model();

    const Config& get_config() const { return config_; }
    BPETokenizer* get_tokenizer() const { return tokenizer_.get(); }

    bool init(const std::string& model_folder, size_t context_size, const std::string& system_prompt = "");
    uint32_t generate(const std::vector<uint32_t>& tokens, float temperature = 0.6f, float top_p = 0.95f, 
                      size_t top_k = 20, const std::string& profile_file = "");
    
    std::vector<float> get_embeddings(const std::vector<uint32_t>& tokens, bool pooled = true);
    
    void reset_cache() { kv_cache_.reset(); }
    
private:
    size_t forward(const std::vector<uint32_t>& tokens, bool use_cache = false);
    void load_weights_to_graph(CactusGraph* gb);
    size_t build_attention(CactusGraph* gb, size_t normalized_input, uint32_t layer_idx, 
                          ComputeBackend backend, bool use_cache = false, size_t position_offset = 0);
    size_t build_mlp(CactusGraph* gb, size_t normalized_h, uint32_t layer_idx, 
                    ComputeBackend backend) const;
    size_t build_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx, 
                                  ComputeBackend backend, bool use_cache = false, size_t position_offset = 0);
    void update_kv_cache(CactusGraph* gb, size_t seq_len);
    Config config_;
    std::unique_ptr<BPETokenizer> tokenizer_;

    void* graph_handle_;
    bool initialized_;
    float attention_scale_;
    
private:
    KVCache kv_cache_;
    std::vector<size_t> cache_k_output_nodes_;
    std::vector<size_t> cache_v_output_nodes_;
    
    
    std::string embedding_file_path_;
    size_t embedding_node_id_;
    std::string model_folder_path_;
    
    
    struct WeightNodeIDs {
        size_t output_weight;
        size_t output_norm_weight;
        
        struct LayerWeights {
            size_t attn_q_weight;
            size_t attn_k_weight;
            size_t attn_v_weight;
            size_t attn_output_weight;
            size_t input_layernorm_weight;
            size_t attn_q_norm_weight;
            size_t attn_k_norm_weight;
            size_t ffn_gate_weight;
            size_t ffn_up_weight;
            size_t ffn_down_weight;
            size_t post_attention_layernorm_weight;
        };
        
        std::vector<LayerWeights> layers;
    } mutable weight_nodes_;
};

}
}