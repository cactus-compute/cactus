#include "../cactus_engine.h"
#include "engine.h"
#include "utils.h"
#include <cmath>
#include <cstring>
#include <mutex>
#include <random>

namespace cactus {
namespace engine {

std::vector<float> diffusion_alphas_cumprod(const DiffusionParams& params) {
    std::vector<float> alphas_cumprod(params.num_train_timesteps);
    const double sqrt_start = std::sqrt(static_cast<double>(params.beta_start));
    const double sqrt_end = std::sqrt(static_cast<double>(params.beta_end));
    double cumprod = 1.0;
    for (uint32_t i = 0; i < params.num_train_timesteps; ++i) {
        const double frac = params.num_train_timesteps > 1
            ? static_cast<double>(i) / (params.num_train_timesteps - 1) : 0.0;
        const double sqrt_beta = sqrt_start + (sqrt_end - sqrt_start) * frac;
        cumprod *= 1.0 - sqrt_beta * sqrt_beta;
        alphas_cumprod[i] = static_cast<float>(cumprod);
    }
    return alphas_cumprod;
}

std::vector<uint32_t> diffusion_lcm_timesteps(const DiffusionParams& params, uint32_t steps) {
    if (steps == 0 || params.original_inference_steps == 0) return {};
    const uint32_t k = params.num_train_timesteps / params.original_inference_steps;
    std::vector<uint32_t> timesteps(steps);
    for (uint32_t s = 0; s < steps; ++s) {
        const uint32_t index = static_cast<uint32_t>(
            std::floor(static_cast<double>(s) * params.original_inference_steps / steps));
        timesteps[s] = (params.original_inference_steps - index) * k - 1;
    }
    return timesteps;
}

std::vector<float> diffusion_guidance_embedding(float guidance_embedding_scale, size_t dim) {
    const size_t half = dim / 2;
    std::vector<float> embedding(dim, 0.0f);
    const double scaled = static_cast<double>(guidance_embedding_scale) * 1000.0;
    const double span = half > 1 ? static_cast<double>(half - 1) : 1.0;
    for (size_t i = 0; i < half; ++i) {
        const double freq = std::exp(-std::log(10000.0) * static_cast<double>(i) / span);
        embedding[i] = static_cast<float>(std::sin(scaled * freq));
        embedding[half + i] = static_cast<float>(std::cos(scaled * freq));
    }
    return embedding;
}

int Model::generate_image(const std::string& prompt, uint8_t* out_rgb, size_t out_capacity,
                          uint32_t* out_width, uint32_t* out_height,
                          int steps, float guidance_scale, uint64_t seed) {
    if (decode_route_ != DecodeRoute::ITERATIVE_DENOISE) {
        CACTUS_LOG_ERROR("model", "generate_image requires a text-to-image bundle");
        return -1;
    }
    if (steps < 1 || steps > static_cast<int>(diffusion_.original_inference_steps)) {
        CACTUS_LOG_ERROR("model", "generate_image steps must be in [1, " << diffusion_.original_inference_steps << "]");
        return -1;
    }
    Component& text_encoder = components_.at("text_encoder");
    Component& unet = components_.at("unet");
    Component& vae_decoder = components_.at("vae_decoder");
    if (!text_encoder.graph || !unet.graph || !vae_decoder.graph || !tokenizer_) return -1;

    const int ids_idx = input_index(text_encoder, "input_ids");
    const int sample_idx = input_index(unet, "sample");
    const int timestep_idx = input_index(unet, "timestep");
    const int hidden_idx = input_index(unet, "encoder_hidden_states");
    const int cond_idx = input_index(unet, "timestep_cond");
    const int latent_idx = input_index(vae_decoder, "x");
    if (ids_idx < 0 || sample_idx < 0 || timestep_idx < 0 || hidden_idx < 0 || cond_idx < 0 || latent_idx < 0) {
        CACTUS_LOG_ERROR("model", "text-to-image bundle is missing expected component inputs");
        return -1;
    }

    const auto& ids_desc = text_encoder.graph->get_output_buffer(
        static_cast<size_t>(text_encoder.runtime_input_node_ids[ids_idx]));
    const size_t max_tokens = ids_desc.total_size;
    const uint32_t bos = tokenizer_->get_bos_token();
    const uint32_t eos = tokenizer_->get_eos_token();
    auto tokens = tokenizer_->encode(prompt);
    if (tokens.size() > max_tokens - 2) tokens.resize(max_tokens - 2);
    write_int_input_at(text_encoder, "input_ids", 0, static_cast<int64_t>(bos));
    for (size_t i = 0; i < tokens.size(); ++i) {
        write_int_input_at(text_encoder, "input_ids", i + 1, static_cast<int64_t>(tokens[i]));
    }
    for (size_t i = tokens.size() + 1; i < max_tokens; ++i) {
        write_int_input_at(text_encoder, "input_ids", i, static_cast<int64_t>(eos));
    }
    text_encoder.graph->execute();

    const size_t hidden_node = static_cast<size_t>(text_encoder.output_node_ids[0]);
    const auto& hidden_desc = text_encoder.graph->get_output_buffer(hidden_node);
    write_typed_buffer(
        unet.input_buffers[hidden_idx],
        unet.graph->get_output_buffer(static_cast<size_t>(unet.runtime_input_node_ids[hidden_idx])).precision,
        text_encoder.graph->get_output(hidden_node),
        hidden_desc.byte_size,
        hidden_desc.precision);

    const auto& cond_desc = unet.graph->get_output_buffer(
        static_cast<size_t>(unet.runtime_input_node_ids[cond_idx]));
    const std::vector<float> guidance = diffusion_guidance_embedding(guidance_scale - 1.0f, cond_desc.total_size);
    write_typed_buffer(unet.input_buffers[cond_idx], cond_desc.precision,
                       guidance.data(), guidance.size() * sizeof(float), Precision::FP32);

    const auto& sample_desc = unet.graph->get_output_buffer(
        static_cast<size_t>(unet.runtime_input_node_ids[sample_idx]));
    const size_t latent_count = sample_desc.total_size;
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> latents(latent_count);
    for (float& value : latents) value = normal(rng);

    const std::vector<float> alphas_cumprod = diffusion_alphas_cumprod(diffusion_);
    const std::vector<uint32_t> timesteps = diffusion_lcm_timesteps(diffusion_, static_cast<uint32_t>(steps));

    const auto& timestep_desc = unet.graph->get_output_buffer(
        static_cast<size_t>(unet.runtime_input_node_ids[timestep_idx]));
    std::vector<float> eps(latent_count);
    for (int s = 0; s < steps; ++s) {
        const uint32_t t = timesteps[s];
        write_typed_buffer(unet.input_buffers[sample_idx], sample_desc.precision,
                           latents.data(), latents.size() * sizeof(float), Precision::FP32);
        const float timestep_value = static_cast<float>(t);
        write_typed_buffer(unet.input_buffers[timestep_idx], timestep_desc.precision,
                           &timestep_value, sizeof(float), Precision::FP32);
        unet.graph->execute();

        const size_t eps_node = static_cast<size_t>(unet.output_node_ids[0]);
        const auto& eps_desc = unet.graph->get_output_buffer(eps_node);
        if (eps_desc.precision == Precision::FP16) {
            const __fp16* data = static_cast<const __fp16*>(unet.graph->get_output(eps_node));
            for (size_t i = 0; i < latent_count; ++i) eps[i] = static_cast<float>(data[i]);
        } else {
            std::memcpy(eps.data(), unet.graph->get_output(eps_node), latent_count * sizeof(float));
        }

        const float acp_t = alphas_cumprod[t];
        const float sqrt_acp = std::sqrt(acp_t);
        const float sqrt_one_minus = std::sqrt(1.0f - acp_t);
        const float scaled_t = static_cast<float>(t) * diffusion_.timestep_scaling;
        const float c_skip = 0.25f / (scaled_t * scaled_t + 0.25f);
        const float c_out = scaled_t / std::sqrt(scaled_t * scaled_t + 0.25f);
        const bool final_step = s == steps - 1;
        const float acp_prev = final_step ? 1.0f : alphas_cumprod[timesteps[s + 1]];
        const float sqrt_acp_prev = std::sqrt(acp_prev);
        const float sqrt_one_minus_prev = std::sqrt(1.0f - acp_prev);
        for (size_t i = 0; i < latent_count; ++i) {
            const float x0 = (latents[i] - sqrt_one_minus * eps[i]) / sqrt_acp;
            const float denoised = c_out * x0 + c_skip * latents[i];
            latents[i] = final_step ? denoised
                                    : sqrt_acp_prev * denoised + sqrt_one_minus_prev * normal(rng);
        }
    }

    const auto& latent_desc = vae_decoder.graph->get_output_buffer(
        static_cast<size_t>(vae_decoder.runtime_input_node_ids[latent_idx]));
    write_typed_buffer(vae_decoder.input_buffers[latent_idx], latent_desc.precision,
                       latents.data(), latents.size() * sizeof(float), Precision::FP32);
    vae_decoder.graph->execute();

    const size_t image_node = static_cast<size_t>(vae_decoder.output_node_ids[0]);
    const auto& image_desc = vae_decoder.graph->get_output_buffer(image_node);
    if (image_desc.shape.size() != 4 || image_desc.shape[1] != 3) {
        CACTUS_LOG_ERROR("model", "vae_decoder output is not an NCHW RGB image");
        return -1;
    }
    const size_t height = image_desc.shape[2];
    const size_t width = image_desc.shape[3];
    const size_t out_bytes = height * width * 3;
    if (out_capacity < out_bytes) {
        CACTUS_LOG_ERROR("model", "generate_image buffer too small: need " << out_bytes << " bytes");
        return -2;
    }

    const size_t plane = height * width;
    auto write_pixels = [&](auto value_at) {
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                const size_t pixel = y * width + x;
                for (size_t c = 0; c < 3; ++c) {
                    const float value = std::min(1.0f, std::max(0.0f, value_at(c * plane + pixel)));
                    out_rgb[pixel * 3 + c] = static_cast<uint8_t>(std::lround(value * 255.0f));
                }
            }
        }
    };
    if (image_desc.precision == Precision::FP16) {
        const __fp16* data = static_cast<const __fp16*>(vae_decoder.graph->get_output(image_node));
        write_pixels([data](size_t i) { return static_cast<float>(data[i]); });
    } else {
        const float* data = static_cast<const float*>(vae_decoder.graph->get_output(image_node));
        write_pixels([data](size_t i) { return data[i]; });
    }
    if (out_width) *out_width = static_cast<uint32_t>(width);
    if (out_height) *out_height = static_cast<uint32_t>(height);
    return static_cast<int>(out_bytes);
}

}  // namespace engine
}  // namespace cactus

using namespace cactus::engine;
using namespace cactus::ffi;

extern "C" {

int cactus_generate_image(
    cactus_model_t model,
    const char* prompt,
    uint8_t* rgb_buffer,
    size_t buffer_size,
    unsigned int* image_width,
    unsigned int* image_height,
    int steps,
    float guidance_scale,
    unsigned long long seed
) {
    if (!model || !prompt || !rgb_buffer || buffer_size == 0) {
        CACTUS_LOG_ERROR("generate_image", "Invalid parameters for image generation");
        return -1;
    }

    try {
        auto* handle = static_cast<CactusModelHandle*>(model);
        std::lock_guard<std::mutex> lock(handle->model_mutex);
        uint32_t width = 0;
        uint32_t height = 0;
        const int result = handle->model->generate_image(
            prompt, rgb_buffer, buffer_size, &width, &height,
            steps, guidance_scale, static_cast<uint64_t>(seed));
        if (result > 0) {
            if (image_width) *image_width = width;
            if (image_height) *image_height = height;
        }
        return result;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        CACTUS_LOG_ERROR("generate_image", "Exception: " << e.what());
        return -1;
    } catch (...) {
        last_error_message = "Unknown error during image generation";
        CACTUS_LOG_ERROR("generate_image", last_error_message);
        return -1;
    }
}

}
