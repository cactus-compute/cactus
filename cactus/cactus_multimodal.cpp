#include "cactus.h"
#include "common.h"
#include "tools/mtmd/mtmd.h"
#include "tools/mtmd/clip.h"
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>

namespace cactus {

static std::string fnv_hash(const uint8_t * data, size_t len) {
    const uint64_t fnv_prime = 0x100000001b3ULL;
    uint64_t hash = 0xcbf29ce484222325ULL;

    for (size_t i = 0; i < len; ++i) {
        hash ^= data[i];
        hash *= fnv_prime;
    }
    return std::to_string(hash);
}

static const std::string base64_chars =
             "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
             "abcdefghijklmnopqrstuvwxyz"
             "0123456789+/";

static std::vector<uint8_t> base64_decode(const std::string &encoded_string) {
    std::vector<uint8_t> decoded;
    int in_len = encoded_string.size();
    int i = 0;
    int j = 0;
    int in_ = 0;
    unsigned char char_array_4[4], char_array_3[3];

    while (in_len-- && (encoded_string[in_] != '=')) {
        if (isspace(encoded_string[in_])) {
            in_++;
            continue;
        }

        if (encoded_string[in_] == '=' || base64_chars.find(encoded_string[in_]) == std::string::npos) {
            break;
        }

        char_array_4[i++] = encoded_string[in_]; in_++;
        if (i == 4) {
            for (i = 0; i < 4; i++) {
                char_array_4[i] = base64_chars.find(char_array_4[i]);
            }

            char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

            for (i = 0; i < 3; i++) {
                decoded.push_back(char_array_3[i]);
            }
            i = 0;
        }
    }

    if (i) {
        for (j = i; j < 4; j++) {
            char_array_4[j] = 0;
        }

        for (j = 0; j < 4; j++) {
            char_array_4[j] = base64_chars.find(char_array_4[j]);
        }

        char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
        char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

        for (j = 0; j < i - 1; j++) {
            decoded.push_back(char_array_3[j]);
        }
    }

    return decoded;
}

struct mtmd_tokenize_result {
    std::vector<std::string> bitmap_hashes;
    std::vector<llama_token> tokens;
    std::vector<size_t> chunk_pos;
    std::vector<size_t> chunk_pos_media;
    mtmd_input_chunks* chunks = nullptr;
};

static mtmd_tokenize_result tokenizeWithMedia(cactus_context::cactus_context_mtmd *mtmd_wrapper, const std::string &prompt, const std::vector<std::string> &media_paths) {
    mtmd_tokenize_result result;
    mtmd::bitmaps bitmaps;

    for (const auto& media_path : media_paths) {
        LOG_VERBOSE("Loading media: %s", media_path.substr(0, 50).c_str());

        if (media_path.compare(0, 11, "data:image/") == 0 || media_path.compare(0, 11, "data:audio/") == 0) {
            LOG_VERBOSE("Detected base64 encoded media");

            size_t comma_pos = media_path.find(',');
            if (comma_pos == std::string::npos) {
                throw std::runtime_error("Invalid base64 media format, missing comma separator");
            }

            std::string header = media_path.substr(0, comma_pos);
            std::string base64_data = media_path.substr(comma_pos + 1);

            if (header.find("base64") == std::string::npos) {
                bitmaps.entries.clear();
                throw std::runtime_error("Media must be base64 encoded");
            }

            std::vector<uint8_t> media_data = base64_decode(base64_data);
            LOG_VERBOSE("Base64 decoded, size: %zu bytes", media_data.size());

            mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_buf(media_data.data(), media_data.size()));
            if (!bmp.ptr) {
                bitmaps.entries.clear();
                throw std::runtime_error("Failed to load base64 media");
            }

            std::string hash = fnv_hash(bmp.data(), bmp.n_bytes());
            bmp.set_id(hash.c_str());
            LOG_VERBOSE("Media hash: %s", hash.c_str());
            bitmaps.entries.push_back(std::move(bmp));
            result.bitmap_hashes.push_back(hash.c_str());
        } else if (media_path.compare(0, 7, "http://") == 0 || media_path.compare(0, 8, "https://") == 0) {
            LOG_ERROR("HTTP/HTTPS URLs are not supported yet: %s", media_path.c_str());
            throw std::runtime_error("HTTP/HTTPS URLs are not supported yet");
        } else {
            LOG_VERBOSE("Loading media from file");

            std::ifstream file(media_path, std::ios::binary);
            if (!file.is_open()) {
                bitmaps.entries.clear();
                throw std::runtime_error("File does not exist or cannot be opened");
            }

            file.seekg(0, std::ios::end);
            size_t file_size = file.tellg();
            file.seekg(0, std::ios::beg);
            
            LOG_VERBOSE("File exists and size is %zu bytes", file_size);
            file.close();

            mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(media_path.c_str()));
            if (!bmp.ptr) {
                bitmaps.entries.clear();
                throw std::runtime_error("Failed to load media");
            }

            std::string hash = fnv_hash(bmp.data(), bmp.nx()*bmp.ny()*3);
            bmp.set_id(hash.c_str());
            LOG_VERBOSE("Media hash: %s", hash.c_str());
            bitmaps.entries.push_back(std::move(bmp));
            result.bitmap_hashes.push_back(hash.c_str());
        }
    }

    LOG_VERBOSE("Initializing input chunks");
    result.chunks = mtmd_input_chunks_init();
    if (result.chunks == nullptr) {
        bitmaps.entries.clear();
        throw std::runtime_error("Failed to initialize input chunks");
    }

    mtmd_input_text input_text;
    input_text.text = prompt.c_str();
    input_text.add_special = true;
    input_text.parse_special = true;

    LOG_VERBOSE("Tokenizing text and %zu media", bitmaps.entries.size());
    auto bitmaps_c_ptr = bitmaps.c_ptr();
    int32_t res = mtmd_tokenize(mtmd_wrapper->mtmd_ctx, result.chunks, &input_text, bitmaps_c_ptr.data(), bitmaps_c_ptr.size());
    if (res != 0) {
        mtmd_input_chunks_free(result.chunks);
        bitmaps.entries.clear();
        throw std::runtime_error("Failed to tokenize text and media");
    }

    size_t num_chunks = mtmd_input_chunks_size(result.chunks);
    LOG_VERBOSE("Tokenization successful: num_chunks=%zu", num_chunks);

    size_t total_token_count = 0;

    for (size_t i = 0; i < num_chunks; i++) {
        result.chunk_pos.push_back(total_token_count);

        const mtmd_input_chunk* chunk = mtmd_input_chunks_get(result.chunks, i);
        mtmd_input_chunk_type chunk_type = mtmd_input_chunk_get_type(chunk);

        if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t n_tokens;
            const llama_token* tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);
            LOG_VERBOSE("Chunk %zu: type=TEXT, n_tokens=%zu", i, n_tokens);

            result.tokens.insert(result.tokens.end(), tokens, tokens + n_tokens);
            total_token_count += n_tokens;
        } else if (chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE || chunk_type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
            result.chunk_pos_media.push_back(total_token_count);

            size_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk);
            size_t n_pos = mtmd_input_chunk_get_n_pos(chunk);
            LOG_VERBOSE("Chunk %zu: type=%s, n_tokens=%zu, n_pos=%zu",
                     i, chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE ? "IMAGE" : "AUDIO", n_tokens, n_pos);

            for (size_t j = 0; j < n_pos; j++) {
                result.tokens.push_back(LLAMA_TOKEN_NULL);
            }
            total_token_count += n_pos;
        }
    }

    bitmaps.entries.clear();
    return result;
}

bool cactus_context::initMultimodal(const std::string &mmproj_path, bool use_gpu) {
    LOG_VERBOSE("Initializing multimodal with mmproj path: %s", mmproj_path.c_str());

    if (model == nullptr) {
        LOG_ERROR("Model not loaded, cannot initialize multimodal");
        return false;
    }

    LOG_VERBOSE("Model info: n_ctx=%d, n_embd=%d", llama_n_ctx(ctx), llama_model_n_embd(model));

    mtmd_context_params mtmd_params = mtmd_context_params_default();
    mtmd_params.use_gpu = use_gpu;
    mtmd_params.print_timings = false;
    mtmd_params.n_threads = params.cpuparams.n_threads;
    mtmd_params.verbosity = (lm_ggml_log_level)LM_GGML_LOG_LEVEL_INFO;

    LOG_VERBOSE("Initializing mtmd context with threads=%d", mtmd_params.n_threads);

    auto mtmd_ctx = mtmd_init_from_file(mmproj_path.c_str(), model, mtmd_params);
    if (mtmd_ctx == nullptr) {
        LOG_ERROR("Failed to initialize multimodal context with mmproj: %s", mmproj_path.c_str());
        return false;
    }
    mtmd_wrapper = new cactus_context_mtmd();
    mtmd_wrapper->mtmd_ctx = mtmd_ctx;

    has_multimodal = true;

    bool uses_mrope = mtmd_decode_use_mrope(mtmd_ctx);
    bool uses_non_causal = mtmd_decode_use_non_causal(mtmd_ctx);
    LOG_VERBOSE("Model multimodal properties: uses_mrope=%d, uses_non_causal=%d", uses_mrope ? 1 : 0, uses_non_causal ? 1 : 0);

    LOG_INFO("Multimodal context initialized successfully with mmproj: %s", mmproj_path.c_str());
    return true;
}

bool cactus_context::isMultimodalEnabled() const {
    return has_multimodal && mtmd_wrapper != nullptr;
}

bool cactus_context::isMultimodalSupportVision() const {
    return isMultimodalEnabled() && mtmd_support_vision(mtmd_wrapper->mtmd_ctx);
}

bool cactus_context::isMultimodalSupportAudio() const {
    return isMultimodalEnabled() && mtmd_support_audio(mtmd_wrapper->mtmd_ctx);
}

void cactus_context::releaseMultimodal() {
    if (mtmd_wrapper && mtmd_wrapper->mtmd_ctx != nullptr) {
        mtmd_free(mtmd_wrapper->mtmd_ctx);
        mtmd_wrapper->mtmd_ctx = nullptr;
        delete mtmd_wrapper;
        mtmd_wrapper = nullptr;
        has_multimodal = false;
    }
}

void cactus_context::processMedia(const std::string &prompt, const std::vector<std::string> &media_paths) {
    if (!isMultimodalEnabled()) {
        throw std::runtime_error("Multimodal is not enabled but image paths are provided");
    }

    std::string full_prompt = prompt;
    auto default_media_marker = mtmd_default_marker();
    if (full_prompt.find(default_media_marker) == std::string::npos) {
        full_prompt += " ";
        full_prompt += default_media_marker;
    }

    LOG_VERBOSE("Processing message with content=%s", full_prompt.c_str());
    LOG_VERBOSE("Processing %zu media with prompt: %s", media_paths.size(), prompt.c_str());
    LOG_VERBOSE("Current context state: n_past=%d, n_ctx=%d", n_past, n_ctx);

    auto result = tokenizeWithMedia(mtmd_wrapper, full_prompt, media_paths);

    auto all_tokens = result.tokens;
    auto chunks = result.chunks;
    auto chunk_pos = result.chunk_pos;
    auto chunk_pos_media = result.chunk_pos_media;
    auto bitmap_hashes = result.bitmap_hashes;

    if (all_tokens.size() >= (size_t)n_ctx) {
        mtmd_input_chunks_free(chunks);
        context_full = true;
        throw std::runtime_error("Not enough context space");
    }

    n_past = common_part(embd, all_tokens);
    llama_pos new_n_past = n_past;

    auto adjusted_n_past = -1;
    for (size_t i = 0; i < chunk_pos.size(); i++) {
        if (n_past < chunk_pos[i]) {
            break;
        }
        bool is_end = i + 1 == chunk_pos.size();
        if (chunk_pos[i] < n_past && (!is_end && chunk_pos[i + 1] > n_past)) {
            adjusted_n_past = chunk_pos[i];
        }
    }
    if (adjusted_n_past != -1) {
        n_past = adjusted_n_past;
        new_n_past = n_past;
        LOG_VERBOSE("Adjusted n_past to %d", n_past);
    }

    if (mtmd_bitmap_past_hashes.size() > 0) {
        for (size_t i = 0; i < bitmap_hashes.size(); i++) {
            auto pos = chunk_pos_media[i];
            if (n_past < pos) {
                break;
            }
            if (i >= mtmd_bitmap_past_hashes.size()) {
                break;
            }
            if (bitmap_hashes[i] != mtmd_bitmap_past_hashes[i]) {
                LOG_VERBOSE("Bitmap hash mismatch at position %zu, %s != %s", i, bitmap_hashes[i].c_str(), mtmd_bitmap_past_hashes[i].c_str());
                n_past = chunk_pos_media[i];
                new_n_past = n_past;
                break;
            }
        }
    }

    llama_kv_self_seq_rm(ctx, 0, n_past, -1);

    LOG_VERBOSE("Evaluating chunks: n_past=%d, n_batch=%d", n_past, params.n_batch);

    size_t num_chunks = mtmd_input_chunks_size(chunks);

    for (size_t i = 0; i < chunk_pos.size(); i++) {
        LOG_VERBOSE("Evaluating chunk %zu: n_past=%d, chunk_pos=%zu", i, n_past, chunk_pos[i]);

        if (chunk_pos[i] >= n_past) {
            bool chunk_logits_last = (i == num_chunks - 1);
            auto chunk = mtmd_input_chunks_get(chunks, i);

            int32_t res = mtmd_helper_eval_chunk_single(
                mtmd_wrapper->mtmd_ctx,
                ctx,
                chunk,
                n_past,
                0,
                params.n_batch,
                chunk_logits_last,
                &new_n_past
            );
            if (res != 0) {
                mtmd_input_chunks_free(chunks);
                throw std::runtime_error("Failed to evaluate chunks");
            }
            n_past = new_n_past;
        }
    }

    if (n_past == all_tokens.size() && n_past > 0 && all_tokens[n_past - 1] != LLAMA_TOKEN_NULL) {
        n_past--;
    }

    embd = all_tokens;

    for (auto & token : all_tokens) {
        if (token == LLAMA_TOKEN_NULL) {
            continue;
        }
        common_sampler_accept(ctx_sampling, token, false);
    }
    
    mtmd_bitmap_past_hashes = bitmap_hashes;

    LOG_VERBOSE("Multimodal processing completed");
    mtmd_input_chunks_free(chunks);
}



} // namespace cactus 