#include "model_gemma4.h"
#include "../../graph/graph.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace cactus {
namespace engine {

Gemma4MmModel::Gemma4MmModel() : Model() {
    config_.model_type = Config::ModelType::GEMMA4;
}

Gemma4MmModel::Gemma4MmModel(const Config& config)
    : Model(config), vision_encoder_(config), audio_encoder_(config), language_model_(config) {}

bool Gemma4MmModel::init(const std::string& model_folder, size_t context_size,
                             const std::string& system_prompt, bool do_warmup) {
    if (!Model::init(model_folder, context_size, system_prompt, false))
        return false;

    auto* shared_graph = static_cast<CactusGraph*>(graph_handle_);
    if (!shared_graph)
        throw std::runtime_error("Shared graph was not initialized for Gemma4MmModel");

    bool has_vision = config_.vision_num_layers > 0 || config_.vision_embed_dim > 0;
    bool has_audio = config_.audio_num_layers > 0 || config_.audio_hidden_dim > 0;

    if (has_vision) {
        if (!vision_encoder_.init(shared_graph, model_folder, context_size, "", false))
            throw std::runtime_error("Failed to initialize vision encoder");
    }

    if (has_audio) {
        if (!audio_encoder_.init(shared_graph, model_folder, context_size, "", false))
            throw std::runtime_error("Failed to initialize audio encoder");
    }

    if (!language_model_.init(shared_graph, model_folder, context_size, system_prompt, false))
        throw std::runtime_error("Failed to initialize language model");

    output_weight_node_id_ = language_model_.output_weight_node_id_;

    if (do_warmup) {
        std::vector<uint32_t> warmup_tokens = {2};
        language_model_.forward(warmup_tokens);
        auto* gb2 = static_cast<CactusGraph*>(language_model_.graph_handle_);
        gb2->execute();
        language_model_.reset_cache();
    }

    return true;
}

void Gemma4MmModel::reset_cache() {
    Model::reset_cache();
    language_model_.reset_cache();
    prefill_completed_ = false;
    last_token_count_ = 0;
}

void Gemma4MmModel::compact_kv_cache() {
    language_model_.compact_kv_cache();
}

void Gemma4MmModel::remove_thinking_tokens(const std::vector<std::pair<size_t, size_t>>& ranges) {
    language_model_.remove_thinking_tokens(ranges);
}

void Gemma4MmModel::load_weights_to_graph(CactusGraph*) {
    output_weight_node_id_ = 0;
}

Gemma4MmModel::ForwardResult Gemma4MmModel::forward_multimodal(
    CactusGraph* gb, const std::vector<uint32_t>& tokens,
    const std::vector<std::string>& image_paths,
    const std::vector<float>* audio_features,
    size_t audio_num_frames,
    ComputeBackend backend, bool use_cache) {

    auto inputs = build_multimodal_inputs(
        gb, tokens, image_paths, audio_features, audio_num_frames, backend);

    size_t final_hidden = language_model_.forward_from_embeddings(
        gb,
        inputs.hidden_node,
        inputs.pli_hidden_source_node,
        inputs.pli_tokens,
        inputs.seq_len,
        backend,
        use_cache);

    return ForwardResult{final_hidden, inputs.seq_len};
}

Gemma4MmModel::MultimodalInputs Gemma4MmModel::build_multimodal_inputs(
    CactusGraph* gb, const std::vector<uint32_t>& tokens,
    const std::vector<std::string>& image_paths,
    const std::vector<float>* audio_features,
    size_t audio_num_frames,
    ComputeBackend backend) {

    size_t vision_soft_node = 0;
    size_t num_vision_soft_tokens = 0;
    size_t audio_soft_node = 0;
    size_t num_audio_soft_tokens = 0;

    if (!image_paths.empty()) {
        auto preprocessed = vision_encoder_.preprocess_image(image_paths[0]);
        size_t vision_output = vision_encoder_.forward_vision(gb, preprocessed, backend);
        vision_soft_node = vision_encoder_.build_vision_projector(gb, vision_output, backend);
        uint32_t k = config_.vision_pooling_kernel_size;
        num_vision_soft_tokens = (preprocessed.patch_width / k) * (preprocessed.patch_height / k);
    }

    if (audio_features && !audio_features->empty()) {
        size_t audio_output = audio_encoder_.forward_audio(gb, *audio_features, audio_num_frames, backend);
        audio_soft_node = audio_encoder_.build_audio_projector(gb, audio_output, backend);
        const auto& audio_buf = gb->get_output_buffer(audio_soft_node);
        num_audio_soft_tokens = audio_buf.shape[0];
    }

    uint32_t image_token_id = config_.image_token_id;
    uint32_t audio_token_id = config_.audio_token_id;
    uint32_t pad_token_id = config_.pad_token_id;

    std::vector<size_t> sequence_nodes;
    std::vector<uint32_t> current_text;
    std::vector<uint32_t> pli_tokens;
    size_t total_seq_len = 0;
    size_t vision_offset = 0;
    size_t audio_offset = 0;

    auto flush_text = [&]() {
        if (current_text.empty()) return;
        size_t seg_len = current_text.size();
        size_t input_node = gb->input({seg_len}, Precision::FP32);

        auto hidden = gb->scalar_multiply(
            gb->embedding(language_model_.embedding_node_id_, input_node),
            std::sqrt(static_cast<float>(config_.hidden_dim)));

        std::vector<float> input_data(seg_len);
        for (size_t i = 0; i < seg_len; i++)
            input_data[i] = static_cast<float>(current_text[i]);
        gb->set_input(input_node, input_data.data(), Precision::FP32);

        sequence_nodes.push_back(hidden);
        for (auto t : current_text)
            pli_tokens.push_back(t);
        total_seq_len += seg_len;
        current_text.clear();
    };

    auto append_soft_region = [&](size_t soft_node, size_t& soft_offset, size_t total_soft_tokens,
                                  size_t placeholder_count) {
        size_t to_insert = std::min(placeholder_count, total_soft_tokens - soft_offset);
        if (to_insert > 0) {
            sequence_nodes.push_back(gb->slice(soft_node, 0, soft_offset, to_insert));
            for (size_t j = 0; j < to_insert; j++)
                pli_tokens.push_back(pad_token_id);
            total_seq_len += to_insert;
            soft_offset += to_insert;
        }
    };

    auto flush_vision_region = [&](size_t placeholder_count) {
        append_soft_region(vision_soft_node, vision_offset, num_vision_soft_tokens, placeholder_count);
    };

    auto flush_audio_region = [&](size_t placeholder_count) {
        append_soft_region(audio_soft_node, audio_offset, num_audio_soft_tokens, placeholder_count);
    };

    bool in_image_region = false;
    bool in_audio_region = false;
    size_t region_count = 0;

    for (size_t i = 0; i < tokens.size(); i++) {
        uint32_t tok = tokens[i];
        bool is_vision_token = (tok == image_token_id && image_token_id != 0);
        bool is_audio_token = (tok == audio_token_id && audio_token_id != 0);

        if (is_vision_token) {
            if (in_audio_region) {
                flush_audio_region(region_count);
                in_audio_region = false;
            }
            if (!in_image_region) {
                flush_text();
                in_image_region = true;
                region_count = 0;
            }
            region_count++;
        } else if (is_audio_token) {
            if (in_image_region) {
                flush_vision_region(region_count);
                in_image_region = false;
            }
            if (!in_audio_region) {
                flush_text();
                in_audio_region = true;
                region_count = 0;
            }
            region_count++;
        } else {
            if (in_image_region) {
                flush_vision_region(region_count);
                in_image_region = false;
            }
            if (in_audio_region) {
                flush_audio_region(region_count);
                in_audio_region = false;
            }
            current_text.push_back(tok);
        }
    }

    if (in_image_region)
        flush_vision_region(region_count);
    if (in_audio_region)
        flush_audio_region(region_count);
    flush_text();

    if (sequence_nodes.empty())
        throw std::runtime_error("No embedding nodes built");

    size_t merged = sequence_nodes[0];
    for (size_t i = 1; i < sequence_nodes.size(); i++)
        merged = gb->concat(merged, sequence_nodes[i], 0);

    return MultimodalInputs{
        .hidden_node = merged,
        .pli_hidden_source_node = merged,
        .pli_tokens = std::move(pli_tokens),
        .seq_len = total_seq_len,
    };
}

uint32_t Gemma4MmModel::decode_multimodal(
    const std::vector<uint32_t>& tokens,
    const std::vector<std::string>& image_paths,
    const std::vector<float>* audio_features,
    size_t audio_num_frames,
    float temperature, float top_p, size_t top_k,
    const std::string& profile_file, float* out_entropy) {

    if (!initialized_ || !graph_handle_)
        throw std::runtime_error("Model not initialized - call init() first");

    bool has_media = !image_paths.empty() || (audio_features && !audio_features->empty());

    if (!has_media) {
        prefill_completed_ = false;
        last_token_count_ = tokens.size();
        return language_model_.decode(tokens, temperature, top_p, top_k, profile_file, out_entropy);
    }

    if (temperature < 0) temperature = config_.default_temperature;
    if (top_p < 0) top_p = config_.default_top_p;
    if (top_k == 0) top_k = config_.default_top_k;

    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    gb->soft_reset();
    auto backend = config_.default_backend == Config::Backend::CPU ? ComputeBackend::CPU : ComputeBackend::NPU;
    bool cache_empty = language_model_.kv_cache_.is_empty();
    bool need_prefill = cache_empty || !prefill_completed_;

    if (!need_prefill && tokens.size() <= last_token_count_) {
        reset_cache();
        need_prefill = true;
    }

    size_t seq_len_for_updates = 0;
    size_t final_hidden_node = 0;

    if (need_prefill) {
        auto result = forward_multimodal(gb, tokens, image_paths, audio_features,
                                          audio_num_frames, backend, true);
        final_hidden_node = result.final_hidden_node;
        seq_len_for_updates = result.seq_len;
        prefill_completed_ = true;
        last_token_count_ = tokens.size();
    } else {
        size_t delta = tokens.size() - last_token_count_;
        if (delta == 0) delta = 1;
        std::vector<uint32_t> incremental_tokens(tokens.end() - delta, tokens.end());
        final_hidden_node = language_model_.forward(incremental_tokens, true);
        seq_len_for_updates = incremental_tokens.size();
        last_token_count_ = tokens.size();
    }

    auto last_hidden = gb->index(final_hidden_node, seq_len_for_updates - 1, 0);
    const auto& last_buf = gb->get_output_buffer(last_hidden);
    last_hidden = gb->reshape(last_hidden, {1, last_buf.shape[0]});

    auto logits_node = gb->matmul(last_hidden, language_model_.output_weight_node_id_, true, backend);

    if (config_.final_logit_softcapping > 0.0f) {
        float inv_cap = 1.0f / config_.final_logit_softcapping;
        logits_node = gb->scalar_multiply(logits_node, inv_cap);
        logits_node = gb->tanh(logits_node);
        logits_node = gb->scalar_multiply(logits_node, config_.final_logit_softcapping);
    }

    auto sampled_token = gb->sample(logits_node, temperature, top_p, top_k);

    if (!profile_file.empty())
        gb->execute(profile_file);
    else
        gb->execute();

    compute_entropy(gb, logits_node, out_entropy);

    language_model_.post_execute_updates(gb, seq_len_for_updates);
    language_model_.update_kv_cache(gb, seq_len_for_updates);

    auto* output_ptr = gb->get_output(sampled_token);
    return *static_cast<uint32_t*>(output_ptr);
}

size_t Gemma4MmModel::forward(const std::vector<uint32_t>& tokens, bool use_cache) {
    return language_model_.forward(tokens, use_cache);
}

uint32_t Gemma4MmModel::decode(const std::vector<uint32_t>& tokens,
                                   float temperature, float top_p, size_t top_k,
                                   const std::string& profile_file, float* out_entropy) {
    if (!initialized_ || !graph_handle_)
        throw std::runtime_error("Model not initialized - call init() first");
    prefill_completed_ = false;
    last_token_count_ = tokens.size();
    return language_model_.decode(tokens, temperature, top_p, top_k, profile_file, out_entropy);
}

void Gemma4MmModel::prefill(const std::vector<uint32_t>& tokens, size_t chunk_size,
                                const std::string& profile_file) {
    if (!initialized_ || !graph_handle_)
        throw std::runtime_error("Model not initialized - call init() first");
    prefill_completed_ = false;
    last_token_count_ = tokens.size();
    language_model_.prefill(tokens, chunk_size, profile_file);
}

void Gemma4MmModel::prefill_with_images(const std::vector<uint32_t>& tokens,
                                            const std::vector<std::string>& image_paths,
                                            const std::string& profile_file) {
    if (!initialized_ || !graph_handle_)
        throw std::runtime_error("Model not initialized - call init() first");

    if (image_paths.empty()) {
        prefill(tokens, get_prefill_chunk_size(), profile_file);
        return;
    }

    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    gb->soft_reset();
    auto backend = config_.default_backend == Config::Backend::CPU ? ComputeBackend::CPU : ComputeBackend::NPU;

    auto result = forward_multimodal(gb, tokens, image_paths, nullptr, 0, backend, true);

    if (!profile_file.empty())
        gb->execute(profile_file);
    else
        gb->execute();

    language_model_.post_execute_updates(gb, result.seq_len);
    language_model_.update_kv_cache(gb, result.seq_len);

    prefill_completed_ = true;
    last_token_count_ = tokens.size();
}

uint32_t Gemma4MmModel::decode_with_images(
    const std::vector<uint32_t>& tokens, const std::vector<std::string>& image_paths,
    float temperature, float top_p, size_t top_k,
    const std::string& profile_file, float* out_entropy) {
    return decode_multimodal(tokens, image_paths, nullptr, 0,
                              temperature, top_p, top_k, profile_file, out_entropy);
}

uint32_t Gemma4MmModel::decode_with_audio(
    const std::vector<uint32_t>& tokens, const std::vector<float>& audio_features,
    float temperature, float top_p, size_t top_k,
    const std::string& profile_file, float* out_entropy,
    float* /*out_token_time_start*/, float* /*out_token_time_end*/) {
    size_t num_frames = audio_features.size() / config_.audio_input_feat_size;
    std::vector<std::string> empty_images;
    return decode_multimodal(tokens, empty_images, &audio_features, num_frames,
                              temperature, top_p, top_k, profile_file, out_entropy);
}

std::vector<float> Gemma4MmModel::get_image_embeddings(const std::string& image_path) {
    if (!initialized_ || !graph_handle_)
        throw std::runtime_error("Model not initialized - call init() first");
    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    gb->soft_reset();
    auto backend = config_.default_backend == Config::Backend::CPU ? ComputeBackend::CPU : ComputeBackend::NPU;

    auto preprocessed = vision_encoder_.preprocess_image(image_path);
    size_t vision_output = vision_encoder_.forward_vision(gb, preprocessed, backend);
    size_t projected = vision_encoder_.build_vision_projector(gb, vision_output, backend);

    gb->execute();

    const auto& buf = gb->get_output_buffer(projected);
    size_t total = buf.total_size;
    std::vector<float> embedding(total);
    const __fp16* fp16_data = buf.data_as<__fp16>();
    for (size_t i = 0; i < total; i++)
        embedding[i] = static_cast<float>(fp16_data[i]);
    return embedding;
}

std::vector<float> Gemma4MmModel::get_audio_embeddings(const std::vector<float>& audio_features) {
    if (!initialized_ || !graph_handle_)
        throw std::runtime_error("Model not initialized - call init() first");
    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    gb->soft_reset();
    auto backend = config_.default_backend == Config::Backend::CPU ? ComputeBackend::CPU : ComputeBackend::NPU;

    size_t num_frames = audio_features.size() / config_.audio_input_feat_size;
    size_t audio_output = audio_encoder_.forward_audio(gb, audio_features, num_frames, backend);
    size_t projected = audio_encoder_.build_audio_projector(gb, audio_output, backend);

    gb->execute();

    const auto& buf = gb->get_output_buffer(projected);
    size_t total = buf.total_size;
    std::vector<float> embedding(total);
    const __fp16* fp16_data = buf.data_as<__fp16>();
    for (size_t i = 0; i < total; i++)
        embedding[i] = static_cast<float>(fp16_data[i]);
    return embedding;
}

size_t Gemma4MmModel::build_attention(CactusGraph*, size_t, uint32_t, ComputeBackend, bool, size_t) {
    throw std::runtime_error("build_attention should not be called directly on Gemma4MmModel");
}

size_t Gemma4MmModel::build_mlp(CactusGraph*, size_t, uint32_t, ComputeBackend) const {
    throw std::runtime_error("build_mlp should not be called directly on Gemma4MmModel");
}

size_t Gemma4MmModel::build_transformer_block(CactusGraph*, size_t, uint32_t, ComputeBackend, bool, size_t) {
    throw std::runtime_error("build_transformer_block should not be called directly on Gemma4MmModel");
}

}
}
