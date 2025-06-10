#include "cactus.h"
#include "common.h"
#include "tools/mtmd/mtmd.h"

namespace cactus {

cactus_tokenize_result cactus_context::tokenize(const std::string &text, const std::vector<std::string> &media_paths) {
    if (media_paths.size() > 0) {
        if (!isMultimodalEnabled()) {
            throw std::runtime_error("Multimodal is not enabled but media paths are provided");
        }
        
        std::string full_text = text;
        auto default_media_marker = mtmd_default_marker();
        
        if (full_text.find(default_media_marker) == std::string::npos) {
            full_text += " ";
            full_text += default_media_marker;
        }
        
        std::vector<llama_token> text_tokens = common_tokenize(ctx, full_text, false);
        std::vector<std::string> bitmap_hashes;
        std::vector<size_t> chunk_pos = {0};
        std::vector<size_t> chunk_pos_media;
        
        size_t total_token_count = text_tokens.size();
        
        for (size_t i = 0; i < media_paths.size(); ++i) {
            chunk_pos_media.push_back(total_token_count);
            
            for (int j = 0; j < 256; ++j) {
                text_tokens.push_back(LLAMA_TOKEN_NULL);
            }
            total_token_count += 256;
            
            bitmap_hashes.push_back("placeholder_hash_" + std::to_string(i));
        }
        
        cactus_tokenize_result tokenize_result = {
            .tokens = text_tokens,
            .has_media = true,
            .bitmap_hashes = bitmap_hashes,
            .chunk_pos = chunk_pos,
            .chunk_pos_media = chunk_pos_media,
        };
        return tokenize_result;
    }
    
    std::vector<llama_token> text_tokens = common_tokenize(ctx, text, false);
    cactus_tokenize_result tokenize_result = {
        .tokens = text_tokens,
        .has_media = false,
        .bitmap_hashes = {},
        .chunk_pos = {},
        .chunk_pos_media = {},
    };
    return tokenize_result;
}

} // namespace cactus 