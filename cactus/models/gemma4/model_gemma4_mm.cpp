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

    return ForwardResult{final_hidden, inputs.seq_len, inputs.audio_soft_node, inputs.num_audio_soft_tokens, inputs.audio_tower_node};
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
    size_t audio_tower_node = 0;

    if (!image_paths.empty()) {
        auto preprocessed = vision_encoder_.preprocess_image(image_paths[0]);
        size_t vision_output = vision_encoder_.forward_vision(gb, preprocessed, backend);
        vision_soft_node = vision_encoder_.build_vision_projector(gb, vision_output, backend);
        uint32_t k = config_.vision_pooling_kernel_size;
        num_vision_soft_tokens = (preprocessed.patch_width / k) * (preprocessed.patch_height / k);
    }

    if (audio_features && !audio_features->empty()) {
        size_t audio_output = audio_encoder_.forward_audio(gb, *audio_features, audio_num_frames, backend);
        audio_tower_node = audio_output;
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
        .audio_soft_node = audio_soft_node,
        .num_audio_soft_tokens = num_audio_soft_tokens,
        .audio_tower_node = audio_tower_node,
    };
}

uint32_t Gemma4MmModel::decode_multimodal(
    const std::vector<uint32_t>& tokens,
    const std::vector<std::string>& image_paths,
    const std::vector<float>* audio_features,
    size_t audio_num_frames,
    float temperature, float top_p, size_t top_k,
    const std::string& profile_file, float* out_entropy,
    float min_p, float repetition_penalty) {

    if (!initialized_ || !graph_handle_)
        throw std::runtime_error("Model not initialized - call init() first");

    bool has_media = !image_paths.empty() || (audio_features && !audio_features->empty());

    if (!has_media) {
        prefill_completed_ = false;
        last_token_count_ = tokens.size();
        return language_model_.decode(tokens, temperature, top_p, top_k, profile_file, out_entropy, min_p, repetition_penalty);
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
    size_t dbg_audio_soft_node = 0;
    size_t dbg_num_audio_toks = 0;
    size_t dbg_audio_tower_node = 0;

    if (need_prefill) {
        auto result = forward_multimodal(gb, tokens, image_paths, audio_features,
                                          audio_num_frames, backend, true);
        final_hidden_node = result.final_hidden_node;
        seq_len_for_updates = result.seq_len;
        dbg_audio_soft_node = result.audio_soft_node;
        dbg_num_audio_toks = result.num_audio_soft_tokens;
        dbg_audio_tower_node = result.audio_tower_node;
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

    size_t sampled_token =
        language_model_.sample_token(gb, logits_node, temperature, top_p, top_k, min_p, repetition_penalty, nullptr);

    if (!profile_file.empty())
        gb->execute(profile_file);
    else
        gb->execute();

    compute_entropy(gb, logits_node, out_entropy);

    language_model_.post_execute_updates(gb, seq_len_for_updates);
    language_model_.update_kv_cache(gb, seq_len_for_updates);

    auto* output_ptr = gb->get_output(sampled_token);
    uint32_t result_token = *static_cast<uint32_t*>(output_ptr);

    if (need_prefill && dbg_audio_tower_node != 0) {
        void* _atptr = gb->get_output(dbg_audio_tower_node);
        const auto& _atbuf = gb->get_output_buffer(dbg_audio_tower_node);
        if (_atptr && _atbuf.total_size > 0 && _atbuf.precision == Precision::FP16) {
            auto* _atf = static_cast<__fp16*>(_atptr);
            size_t _atdim = _atbuf.shape.size() > 0 ? _atbuf.shape.back() : 0;
            fprintf(stderr, "[mm-dbg] audio_tower_out: dim=%zu tok0 first8:", _atdim);
            for (size_t _i = 0; _i < 8 && _i < _atbuf.total_size; _i++)
                fprintf(stderr, " %.4f", (float)_atf[_i]);
            double _n2 = 0;
            for (size_t _i = 0; _i < _atdim && _i < _atbuf.total_size; _i++) _n2 += (double)_atf[_i]*(double)_atf[_i];
            fprintf(stderr, " norm=%.2f\n", std::sqrt(_n2));
        }
    }

    if (need_prefill && dbg_audio_soft_node != 0) {
        void* _aptr = gb->get_output(dbg_audio_soft_node);
        const auto& _abuf = gb->get_output_buffer(dbg_audio_soft_node);
        if (_aptr && _abuf.total_size > 0) {
            auto* _af = static_cast<__fp16*>(_aptr);
            size_t _adim = _abuf.shape.size() >= 2 ? _abuf.shape[1] : 1536;
            fprintf(stderr, "[mm-dbg] audio_proj: %zu toks, %zu dim\n", dbg_num_audio_toks, _adim);
            {
                double _norm0 = 0.0;
                for (size_t _di = 0; _di < _adim; _di++) { float _v = (float)_af[_di]; _norm0 += _v*_v; }
                fprintf(stderr, "[mm-dbg] audio_proj tok0 first8:");
                for (size_t _di = 0; _di < 8 && _di < _adim; _di++) fprintf(stderr, " %.4f", (float)_af[_di]);
                fprintf(stderr, " norm=%.2f\n", std::sqrt(_norm0));
            }
            for (size_t _ti = 0; _ti < dbg_num_audio_toks; _ti++) {
                float _v0 = (float)_af[_ti * _adim];
                float _v1 = (float)_af[_ti * _adim + 1];
                float _vmax = 0.0f;
                bool _has_nan = false, _has_inf = false;
                for (size_t _di = 0; _di < _adim; _di++) {
                    float _v = (float)_af[_ti * _adim + _di];
                    if (std::isnan(_v)) _has_nan = true;
                    if (std::isinf(_v)) _has_inf = true;
                    if (std::abs(_v) > _vmax) _vmax = std::abs(_v);
                }
                if (_has_nan || _has_inf || _vmax > 100.0f) {
                    fprintf(stderr, "[mm-dbg]   audio[%zu]: %.3f %.3f max=%.1f %s%s\n",
                            _ti, _v0, _v1, _vmax,
                            _has_nan ? "<<NaN!>>" : "", _has_inf ? "<<Inf!>>" : "");
                }
            }
        }
    }

    if (need_prefill) {
        const auto& _fhbuf = gb->get_output_buffer(final_hidden_node);
        void* _fhptr = gb->get_output(final_hidden_node);
        fprintf(stderr, "[mm-dbg] final_hidden: total=%zu shape[", _fhbuf.total_size);
        for (size_t _si = 0; _si < _fhbuf.shape.size(); _si++) fprintf(stderr, "%s%d", _si?",":"", _fhbuf.shape[_si]);
        fprintf(stderr, "] prec=%d ptr=%p\n", (int)_fhbuf.precision, _fhptr);
        if (_fhptr && _fhbuf.total_size >= 10) {
            auto* _h = static_cast<__fp16*>(_fhptr);
            size_t _hdim = _fhbuf.shape.size() >= 2 ? _fhbuf.shape[1] : 1536;
            size_t _nseq = _fhbuf.total_size / _hdim;
            fprintf(stderr, "[mm-dbg] final_hidden: %zu seq, %zu hdim\n", _nseq, _hdim);
            for (size_t _ti : {(size_t)0, (size_t)3, (size_t)4, (size_t)5, (size_t)6, (size_t)8, (size_t)10, (size_t)15, (size_t)16, (size_t)17, (size_t)18, (size_t)19, (size_t)20, (size_t)50, (size_t)102, (size_t)103, _nseq-1}) {
                if (_ti >= _nseq) continue;
                float _v0 = (float)_h[_ti * _hdim];
                float _v1 = (float)_h[_ti * _hdim + 1];
                bool _is_nan = std::isnan(_v0) || std::isnan(_v1);
                fprintf(stderr, "[mm-dbg]   tok[%zu]: %.3f %.3f %s\n", _ti, _v0, _v1, _is_nan ? "<<NaN!>>" : "");
            }
        }
    }

    static int _mm_dbg_step = 0;
    if (_mm_dbg_step < 3) {
        const auto& _lbuf = gb->get_output_buffer(logits_node);
        void* _lptr = gb->get_output(logits_node);
        size_t _lvocab = _lbuf.total_size;
        fprintf(stderr, "[mm-dbg] logits buf: total=%zu prec=%d shape[", _lvocab, (int)_lbuf.precision);
        for (size_t _si = 0; _si < _lbuf.shape.size(); _si++) fprintf(stderr, "%s%d", _si?",":"", _lbuf.shape[_si]);
        fprintf(stderr, "] ptr=%p seq_len=%zu\n", _lptr, seq_len_for_updates);
        const auto& _ihbuf = gb->get_output_buffer(last_hidden);
        fprintf(stderr, "[mm-dbg] last_hidden buf: total=%zu shape[", _ihbuf.total_size);
        for (size_t _si = 0; _si < _ihbuf.shape.size(); _si++) fprintf(stderr, "%s%d", _si?",":"", _ihbuf.shape[_si]);
        fprintf(stderr, "]\n");
        if (_lptr && _lvocab > 0) {
            std::vector<float> _lf(_lvocab);
            if (_lbuf.precision == Precision::FP32) {
                std::copy(static_cast<float*>(_lptr), static_cast<float*>(_lptr) + _lvocab, _lf.begin());
            } else if (_lbuf.precision == Precision::FP16) {
                auto* _h = static_cast<__fp16*>(_lptr);
                for (size_t _i = 0; _i < _lvocab; _i++) _lf[_i] = (float)_h[_i];
            } else {
                auto* _b = static_cast<uint16_t*>(_lptr);
                for (size_t _i = 0; _i < _lvocab; _i++) { uint32_t _u = (uint32_t)_b[_i] << 16; memcpy(&_lf[_i], &_u, 4); }
            }
            size_t top5[5] = {}; float top5v[5] = {-1e30f,-1e30f,-1e30f,-1e30f,-1e30f};
            for (size_t _i = 0; _i < _lvocab; _i++) {
                float _v = _lf[_i];
                if (_v > top5v[4]) {
                    top5v[4] = _v; top5[4] = _i;
                    for (int _j = 3; _j >= 0; _j--) if (top5v[_j+1] > top5v[_j]) { std::swap(top5v[_j],top5v[_j+1]); std::swap(top5[_j],top5[_j+1]); }
                }
            }
            fprintf(stderr, "[mm-dbg] step=%d prefill=%d prec=%d top5: [%zu]=%.2f [%zu]=%.2f [%zu]=%.2f [%zu]=%.2f [%zu]=%.2f → %u\n",
                _mm_dbg_step, (int)need_prefill, (int)_lbuf.precision,
                top5[0],top5v[0], top5[1],top5v[1], top5[2],top5v[2], top5[3],top5v[3], top5[4],top5v[4],
                result_token);
        }
        _mm_dbg_step++;
    }

    language_model_.record_sampled_token(result_token);
    return result_token;
}

size_t Gemma4MmModel::forward(const std::vector<uint32_t>& tokens, bool use_cache) {
    return language_model_.forward(tokens, use_cache);
}

uint32_t Gemma4MmModel::decode(const std::vector<uint32_t>& tokens,
                                   float temperature, float top_p, size_t top_k,
                                   const std::string& profile_file, float* out_entropy,
                                   float min_p, float repetition_penalty) {
    if (!initialized_ || !graph_handle_)
        throw std::runtime_error("Model not initialized - call init() first");
    prefill_completed_ = false;
    last_token_count_ = tokens.size();
    return language_model_.decode(tokens, temperature, top_p, top_k, profile_file, out_entropy, min_p, repetition_penalty);
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
    const std::string& profile_file, float* out_entropy,
    float min_p, float repetition_penalty) {
    return decode_multimodal(tokens, image_paths, nullptr, 0,
                              temperature, top_p, top_k, profile_file, out_entropy,
                              min_p, repetition_penalty);
}

uint32_t Gemma4MmModel::decode_with_audio(
    const std::vector<uint32_t>& tokens, const std::vector<float>& audio_features,
    float temperature, float top_p, size_t top_k,
    const std::string& profile_file, float* out_entropy,
    float min_p, float repetition_penalty,
    float* /*out_token_time_start*/, float* /*out_token_time_end*/) {
    size_t num_frames = audio_features.size() / config_.audio_input_feat_size;
    std::vector<std::string> empty_images;
    return decode_multimodal(tokens, empty_images, &audio_features, num_frames,
                              temperature, top_p, top_k, profile_file, out_entropy,
                              min_p, repetition_penalty);
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

    {
        const auto& ao_buf = gb->get_output_buffer(audio_output);
        if (ao_buf.total_size > 0 && ao_buf.precision == Precision::FP16) {
            const __fp16* ao_data = ao_buf.data_as<__fp16>();
            size_t ao_dim = ao_buf.shape.size() > 0 ? ao_buf.shape.back() : 0;
            fprintf(stderr, "[mm-dbg] audio_tower_out: dim=%zu tok0 first8:", ao_dim);
            for (size_t _i = 0; _i < 8 && _i < ao_buf.total_size; _i++)
                fprintf(stderr, " %.4f", (float)ao_data[_i]);
            double _n2 = 0;
            for (size_t _i = 0; _i < ao_dim && _i < ao_buf.total_size; _i++) _n2 += (double)ao_data[_i]*(double)ao_data[_i];
            fprintf(stderr, " norm=%.2f\n", std::sqrt(_n2));
        }
    }

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
