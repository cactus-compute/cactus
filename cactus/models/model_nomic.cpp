#include "model.h"
#include "../graph/graph.h"
#include <cstddef>
#include <set>

namespace cactus {
namespace engine {

NomicModel::NomicModel() : Model() {}

NomicModel::NomicModel(const Config& config) : Model(config) {
    weight_nodes_.layers.resize(config.num_layers);
}

void NomicModel::load_weights_to_graph(CactusGraph* gb) {
    embedding_node_id_ = gb->mmap_embeddings(embedding_file_path_);
    weight_nodes_.embedding_layernorm_weight = gb->mmap_weights(model_folder_path_ + "/embedding_layernorm.weight");
    weight_nodes_.embedding_layernorm_bias = gb->mmap_weights(model_folder_path_ + "/embedding_layernorm.bias");

    for (uint32_t i = 0; i < config_.num_layers; i++) {
        auto& layer = weight_nodes_.layers[i];
        std::string layer_prefix = model_folder_path_ + "/layer_" + std::to_string(i) + "_";
        layer.attn_q_weight = gb->mmap_weights(layer_prefix + "attn_q.weights");
        layer.attn_k_weight = gb->mmap_weights(layer_prefix + "attn_k.weights");
        layer.attn_v_weight = gb->mmap_weights(layer_prefix + "attn_v.weights");
        layer.attn_q_bias = gb->mmap_weights(layer_prefix + "attn_q.bias");
        layer.attn_k_bias = gb->mmap_weights(layer_prefix + "attn_k.bias");
        layer.attn_v_bias = gb->mmap_weights(layer_prefix + "attn_v.bias");

        layer.attn_output_weight = gb->mmap_weights(layer_prefix + "attn_output.weights");
        layer.attn_output_bias = gb->mmap_weights(layer_prefix + "attn_output.bias");
        layer.ffn_norm_1_weight = gb->mmap_weights(layer_prefix + "norm1.weights");
        layer.ffn_norm_1_bias = gb->mmap_weights(layer_prefix + "norm1.bias");
        layer.ffn_norm_2_weight = gb->mmap_weights(layer_prefix + "norm2.weights");
        layer.ffn_norm_2_bias = gb->mmap_weights(layer_prefix + "norm2.bias");

        if ((i + 1) % config_.moe_every_n_layers != 0) {
            layer.ffn_up_weight = gb->mmap_weights(layer_prefix + "mlp_fc1.weights");
            layer.ffn_up_bias = gb->mmap_weights(layer_prefix + "mlp_fc1.bias");
            layer.ffn_down_weight = gb->mmap_weights(layer_prefix + "mlp_fc2.weights");
            layer.ffn_down_bias = gb->mmap_weights(layer_prefix + "mlp_fc2.bias");

        } else {
            layer.mlp_router_layer_weight = gb->mmap_weights(layer_prefix + "mlp_router.layer.weights");
            layer.mlp_experts_bias = gb->mmap_weights(layer_prefix + "mlp_experts.bias");

            for (uint32_t j = 0; j < config_.num_experts; j++) {
                layer.mlp_experts_mlp1_weight.push_back(gb->mmap_weights(layer_prefix + "mlp_expert_" + std::to_string(j) + ".mlp1.weights"));
                layer.mlp_experts_mlp2_weight.push_back(gb->mmap_weights(layer_prefix + "mlp_expert_" + std::to_string(j) + ".mlp2.weights"));
            }
        }
    }
}

size_t NomicModel::build_attention(CactusGraph* gb, size_t normalized_input, uint32_t layer_idx,
                                  ComputeBackend backend, bool use_cache, size_t position_offset) {
    if (use_cache) {
        throw std::runtime_error("NomicModel does not support generation, it's an encoder model");
    }
    (void)position_offset;

    const auto& layer = weight_nodes_.layers[layer_idx];

    auto q_proj = gb->matmul(normalized_input, layer.attn_q_weight, true, backend);
    q_proj = gb->add(q_proj, layer.attn_q_bias);

    auto k_proj = gb->matmul(normalized_input, layer.attn_k_weight, true, backend);
    k_proj = gb->add(k_proj, layer.attn_k_bias);

    auto v_proj = gb->matmul(normalized_input, layer.attn_v_weight, true, backend);
    v_proj = gb->add(v_proj, layer.attn_v_bias);

    const auto& q_shape = gb->get_output_buffer(q_proj).shape;
    const size_t seq_len = q_shape[0];
    const size_t num_heads = config_.attention_heads;
    const size_t head_dim = config_.attention_head_dim;

    if (num_heads == 0 || head_dim == 0) {
        throw std::runtime_error("Invalid attention head configuration for Nomic model");
    }

    auto reshape_to_heads = [&](size_t tensor) {
        return gb->reshape(tensor, {1, seq_len, num_heads, head_dim});
    };

    auto q_proj_4d = reshape_to_heads(q_proj);
    auto k_proj_4d = reshape_to_heads(k_proj);
    auto v_proj_4d = reshape_to_heads(v_proj);

    if (config_.rope_theta > 0) {
        q_proj_4d = gb->rope(q_proj_4d, config_.rope_theta, 0);
        k_proj_4d = gb->rope(k_proj_4d, config_.rope_theta, 0);
    }

    auto attn_output_4d = gb->attention(q_proj_4d, k_proj_4d, v_proj_4d, attention_scale_, false); 
    auto attn_output = gb->reshape(attn_output_4d, {seq_len, num_heads * head_dim});

    auto output = gb->matmul(attn_output, layer.attn_output_weight, true, backend);
    output = gb->add(output, layer.attn_output_bias);
    return output;
}

size_t NomicModel::build_mlp(CactusGraph* gb, size_t normalized_h, uint32_t layer_idx,
                           ComputeBackend backend) const {
    if ((layer_idx + 1) % config_.moe_every_n_layers != 0) {
        return build_standard_mlp(gb, normalized_h, layer_idx, backend);
    } else {
        return build_moe_mlp(gb, normalized_h, layer_idx, backend);
    }
}

size_t NomicModel::build_standard_mlp(CactusGraph* gb, size_t normalized_h, uint32_t layer_idx,
                                     ComputeBackend backend) const {
    const auto& layer = weight_nodes_.layers[layer_idx];
    auto hidden = gb->matmul(normalized_h, layer.ffn_up_weight, true, backend);
    hidden = gb->add(hidden, layer.ffn_up_bias);
    hidden = gb->gelu(hidden);
    hidden = gb->matmul(hidden, layer.ffn_down_weight, true, backend);
    hidden = gb->add(hidden, layer.ffn_down_bias);
    return hidden;
}

size_t NomicModel::build_moe_mlp(CactusGraph* gb, size_t normalized_h, uint32_t layer_idx,
                                ComputeBackend backend) const {
    const auto& layer = weight_nodes_.layers[layer_idx];
    const auto& router_shape = gb->get_output_buffer(layer.mlp_router_layer_weight).shape;
    if (router_shape.size() != 2) {
        throw std::runtime_error("MoE router weight must be 2D");
    }

    const size_t num_experts = config_.num_experts != 0 ? config_.num_experts : router_shape[0];
    const size_t seq_len = gb->get_output_buffer(normalized_h).shape[0];

    auto gate_weights = gb->matmul(normalized_h, layer.mlp_router_layer_weight, true, backend);
    auto gate_probs = gb->softmax(gate_weights);
    
    auto topk_indices_and_values = gb->topk(gate_probs, config_.num_top_experts);
    auto topk_idx = gb->index(topk_indices_and_values, 0, 0);
    auto topk_w = gb->index(topk_indices_and_values, 1, 0);
    auto expert_token_weights_matrix = gb->scatter_topk(topk_idx, topk_w, num_experts); 
    
    auto expert_outputs = 0; 
    for (size_t e = 0; e < num_experts; e++) {
        auto new_expert_output = gb->matmul(normalized_h, layer.mlp_experts_mlp1_weight[e], true, backend);  
        new_expert_output = gb->gelu(new_expert_output);
        new_expert_output = gb->matmul(new_expert_output, layer.mlp_experts_mlp2_weight[e], true, backend); 

        auto expert_token_weights = gb->index(expert_token_weights_matrix, e, 0); 
        const auto& expert_weights_prec = gb->get_output_buffer(new_expert_output).precision;
        if (gb->get_output_buffer(expert_token_weights).precision != expert_weights_prec) {
            expert_token_weights = gb->precision_cast(expert_token_weights, expert_weights_prec);
        }
        expert_token_weights = gb->reshape(expert_token_weights, {seq_len, 1});
        
        new_expert_output = gb->multiply(new_expert_output, expert_token_weights); 
        if (e == 0) {
            expert_outputs = new_expert_output;
        } else {
            expert_outputs = gb->add(expert_outputs, new_expert_output);
        }
    }

    return gb->add(expert_outputs, layer.mlp_experts_bias);
}

size_t NomicModel::build_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                          ComputeBackend backend, bool use_cache, size_t position_offset) {
    if (use_cache) {
        throw std::runtime_error("NomicModel does not support generation, it's an encoder model");
    }
    (void)position_offset;
    const auto& layer = weight_nodes_.layers[layer_idx];
    auto attn_output = build_attention(gb, hidden, layer_idx, backend);
    auto residual = gb->add(hidden, attn_output);
    auto normalized_residual = gb->layernorm(residual, layer.ffn_norm_1_weight, layer.ffn_norm_1_bias, config_.layer_norm_eps);
    auto mlp_output = build_mlp(gb, normalized_residual, layer_idx, backend);
    auto final_residual = gb->add(normalized_residual, mlp_output);
    auto normalized_final_residual = gb->layernorm(final_residual, layer.ffn_norm_2_weight, layer.ffn_norm_2_bias, config_.layer_norm_eps);
    return normalized_final_residual;
}

size_t NomicModel::forward(const std::vector<uint32_t>& tokens, bool use_cache) {
    if (use_cache) {
        throw std::runtime_error("NomicModel does not support generation, it's an encoder model");
    }
    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    gb->soft_reset();

    size_t seq_len = static_cast<size_t>(tokens.size());
    auto backend = config_.default_backend == Config::Backend::CPU
        ? ComputeBackend::CPU
        : ComputeBackend::NPU;

    size_t input_node_id = gb->input({seq_len}, Precision::FP32);
    std::vector<float> input_data(seq_len);
    for (size_t i = 0; i < seq_len; i++) {
        input_data[i] = static_cast<float>(tokens[i]);
    }
    gb->set_input(input_node_id, input_data.data(), Precision::FP32);
    
    size_t hidden = gb->embedding(embedding_node_id_, input_node_id);
    
    hidden = gb->layernorm(hidden, weight_nodes_.embedding_layernorm_weight, weight_nodes_.embedding_layernorm_bias, config_.layer_norm_eps);

    static std::set<uint32_t> skip_layers = {};
    for (uint32_t layer_idx = 0; layer_idx < config_.num_layers; layer_idx++) {
        if (skip_layers.count(layer_idx)) {
            continue;
        }
        hidden = build_transformer_block(gb, hidden, layer_idx, backend);
    }

    return hidden;
}

}
}
