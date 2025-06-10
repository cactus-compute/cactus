#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <ctime>

#include "utils.h"
#include "../../cactus/cactus_ffi.h"

bool prompt_and_respond_ffi(cactus_context_handle_t handle, const std::string& prompt, const std::vector<std::string>& media_paths = {}, int max_tokens = 50) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "PROMPT: " << prompt << std::endl;
    if (!media_paths.empty()) {
        std::cout << "MEDIA: " << media_paths.size() << " file(s)" << std::endl;
    }
    std::cout << std::string(80, '-') << std::endl;

    // Prepare completion parameters
    cactus_completion_params_c_t comp_params = {};
    comp_params.n_predict = max_tokens;
    comp_params.temperature = 0.7;
    comp_params.top_k = 40;
    comp_params.top_p = 0.9;
    comp_params.penalty_repeat = 1.1;
    comp_params.seed = -1;
    comp_params.ignore_eos = false;
    comp_params.token_callback = nullptr;

    // Format the prompt using chat template 
    std::string messages = R"([{"role": "user", "content": ")" + prompt + R"("}])";
    
    char* formatted_prompt = cactus_get_formatted_chat_c(handle, messages.c_str(), "");
    
    if (!formatted_prompt) {
        std::cerr << "Failed to format chat template, using raw prompt" << std::endl;
        formatted_prompt = const_cast<char*>(prompt.c_str());
    }

    comp_params.prompt = formatted_prompt;

    // Set up stop sequences
    const char* stop_sequences[] = {"<|im_end|>", "</s>", "<|end|>"};
    comp_params.stop_sequences = stop_sequences;
    comp_params.stop_sequence_count = 3;

    // Use appropriate completion function based on whether media is provided
    cactus_completion_result_c_t result = {};
    int completion_result = 0;
    
    if (!media_paths.empty()) {
        // Use multimodal completion for image processing
        std::vector<const char*> media_c_strs;
        for (const auto& path : media_paths) {
            media_c_strs.push_back(path.c_str());
        }
        completion_result = cactus_multimodal_completion_c(handle, &comp_params, media_c_strs.data(), media_c_strs.size(), &result);
    } else {
        // Use standard text completion
        completion_result = cactus_completion_c(handle, &comp_params, &result);
    }
    
    if (completion_result != 0) {
        std::cerr << "Completion failed with code: " << completion_result << std::endl;
        if (formatted_prompt != prompt.c_str()) {
            cactus_free_string_c(formatted_prompt);
        }
        return false;
    }

    std::cout << result.text << std::endl;
    
    // Free the result
    cactus_free_completion_result_members_c(&result);

    // Clean up
    if (formatted_prompt != prompt.c_str()) {
        cactus_free_string_c(formatted_prompt);
    }

    return true;
}

int main(int argc, char **argv) {
    const std::string model_filename = "SmolVLM-256M.gguf";
    const std::string mmproj_filename = "mmproj-SmolVLM-256M.gguf";
    const std::string image_path = "../files/image.jpg";

    const std::string model_url = "https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/SmolVLM-256M-Instruct-Q8_0.gguf";
    const std::string mmproj_url = "https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf";

    if (!downloadFile(model_url, model_filename, "VLM model")) {
        return 1;
    }

    if (!downloadFile(mmproj_url, mmproj_filename, "Multimodal projector")) {
        return 1;
    }

    if (!fileExists(image_path)) {
        std::cerr << "Image file not found: " << image_path << std::endl;
        return 1;
    }

    std::cout << "\n=== Cactus FFI VLM Example ===" << std::endl;

    try {
        // Initialize context using FFI
        cactus_init_params_c_t init_params = {};
        init_params.model_path = model_filename.c_str();
        init_params.chat_template = nullptr;
        init_params.n_ctx = 2048;
        init_params.n_batch = 32;
        init_params.n_ubatch = 32;
        init_params.n_gpu_layers = 99;
        init_params.n_threads = 4;
        init_params.use_mmap = true;
        init_params.use_mlock = false;
        init_params.embedding = false;
        init_params.pooling_type = 0; // LLAMA_POOLING_TYPE_UNSPECIFIED
        init_params.embd_normalize = 2;
        init_params.flash_attn = false;
        init_params.cache_type_k = nullptr;
        init_params.cache_type_v = nullptr;
        init_params.progress_callback = nullptr;

        std::cout << "Loading model..." << std::endl;
        cactus_context_handle_t handle = cactus_init_context_c(&init_params);
        if (!handle) {
            std::cerr << "Failed to load model" << std::endl;
            return 1;
        }

        std::cout << "Initializing multimodal..." << std::endl;
        if (cactus_init_multimodal_c(handle, mmproj_filename.c_str(), true) != 0) {
            std::cerr << "Failed to initialize multimodal" << std::endl;
            cactus_free_context_c(handle);
            return 1;
        }

        bool vision_support = cactus_supports_vision_c(handle);
        std::cout << "Vision support: " << (vision_support ? "Yes" : "No") << std::endl;

        if (!vision_support) {
            std::cerr << "Vision support not available" << std::endl;
            cactus_free_context_c(handle);
            return 1;
        }

        // Test sequence: language, image, language, language, image, image
        std::cout << "\nStarting multi-turn conversation test..." << std::endl;
        
        if (!prompt_and_respond_ffi(handle, "Hello! Can you tell me what you are?", {}, 100)) {
            cactus_free_context_c(handle);
            return 1;
        }
        
        if (!prompt_and_respond_ffi(handle, "Describe what you see in this image.", {image_path}, 150)) {
            cactus_free_context_c(handle);
            return 1;
        }
        
        if (!prompt_and_respond_ffi(handle, "What are the main colors you observed?", {}, 100)) {
            cactus_free_context_c(handle);
            return 1;
        }
        
        if (!prompt_and_respond_ffi(handle, "Can you write a short poem about vision?", {}, 150)) {
            cactus_free_context_c(handle);
            return 1;
        }
        
        if (!prompt_and_respond_ffi(handle, "What emotions or mood does this image convey?", {image_path}, 150)) {
            cactus_free_context_c(handle);
            return 1;
        }
        
        if (!prompt_and_respond_ffi(handle, "If you had to give this image a title, what would it be?", {image_path}, 100)) {
            cactus_free_context_c(handle);
            return 1;
        }
        
        std::cout << "\nMulti-turn conversation test completed!" << std::endl;

        // Clean up
        cactus_release_multimodal_c(handle);
        cactus_free_context_c(handle);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
