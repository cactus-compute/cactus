#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <chrono>

#include "utils.h"
#include "../../cpp/cactus_ffi.h"
#include "../../cpp/json.hpp"
using json = nlohmann::ordered_json;

std::chrono::high_resolution_clock::time_point first_token_time;
bool first_token_received = false;

bool token_callback(const char* token_text) {
    if (!first_token_received) {
        first_token_time = std::chrono::high_resolution_clock::now();
        first_token_received = true;
    }
    return true;
}

std::string build_messages_json(const std::vector<std::pair<std::string, std::string>>& history) {
    json messages = json::array();
    for (const auto& turn : history) {
        messages.push_back({
            {"role", turn.first},
            {"content", turn.second}
        });
    }
    return messages.dump();
}

void run_conversation_demo(cactus_context_handle_t handle) {
    std::cout << "\n=== FFI Multi-Turn Conversation Demo ===" << std::endl;
    std::cout << "NOTE: This demo mirrors the formatting logic from main_llm.cpp" << std::endl;

    std::vector<std::string> user_prompts = {
        "Hi there! Can you tell me about the history of personal computing?",
        "That's fascinating. What were some of the key breakthroughs in the 1980s?",
        "Excellent summary. Finally, what do you see as the next major evolution in computing?"
    };

    bool first_turn = true;

    for (const auto &prompt : user_prompts) {
        std::cout << "\nUser: " << prompt << std::endl;

        // Build JSON with only the new user message
        json msgs = json::array();
        msgs.push_back({{"role", "user"}, {"content", prompt}});

        char *formatted = cactus_get_formatted_chat_c(handle, msgs.dump().c_str(), "");
        if (!formatted) {
            std::cerr << "Failed to format chat." << std::endl;
            continue;
        }

        std::string prompt_to_send = formatted;
        cactus_free_string_c(formatted);

        if (!first_turn) {
            // On subsequent turns, we only need to send the user part, not the full history.
            // The core C++ context is stateful and remembers the prior conversation.
            size_t assistant_pos = prompt_to_send.find("<|im_start|>assistant");
            if (assistant_pos != std::string::npos) {
                prompt_to_send = prompt_to_send.substr(0, assistant_pos) + "<|im_start|>assistant\n";
            }
        }
        first_turn = false;


        // Setup completion params
        first_token_received = false;
        cactus_completion_params_c_t comp_params = {};
        comp_params.prompt = prompt_to_send.c_str();
        comp_params.n_predict = 250;
        comp_params.temperature = 0.7;
        comp_params.seed = -1;
        const char *stop_sequences[] = {"<|im_end|>", "</s>"};
        comp_params.stop_sequences = stop_sequences;
        comp_params.stop_sequence_count = 2;
        comp_params.token_callback = &token_callback;

        cactus_completion_result_c_t result = {};
        auto start = std::chrono::high_resolution_clock::now();
        int status = cactus_completion_c(handle, &comp_params, &result);
        auto end = std::chrono::high_resolution_clock::now();
        auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if (status != 0) {
            std::cerr << "Completion error: " << status << std::endl;
            continue;
        }

        std::string assistant_reply = result.text ? result.text : "";
        std::cout << "Assistant: " << assistant_reply << std::endl;

        // Note: We no longer need to manually track the full prompt string.
        // The state is managed internally by the cactus_context.

        long long ttft_ms = first_token_received ? std::chrono::duration_cast<std::chrono::milliseconds>(first_token_time - start).count() : 0;
        std::cout << "[PERFORMANCE] TTFT: " << ttft_ms << "ms, Total: " << total_ms << "ms, Tokens: " << result.tokens_predicted;
        if (result.tokens_predicted > 0 && total_ms > 0) {
            float tps = (float)result.tokens_predicted * 1000.0f / total_ms;
            std::cout << ", Speed: " << std::fixed << std::setprecision(1) << tps << " tok/s";
        }
        std::cout << std::endl;

        cactus_free_completion_result_members_c(&result);
        std::cout << std::string(60, '-') << std::endl;
    }
}


int main(int argc, char **argv) {
    const std::string model_url = "https://huggingface.co/QuantFactory/SmolLM-360M-Instruct-GGUF/resolve/main/SmolLM-360M-Instruct.Q6_K.gguf";
    const std::string model_filename = "SmolLM-360M-Instruct.Q6_K.gguf";
    
    if (!downloadFile(model_url, model_filename, "LLM model")) {
        return 1;
    }
    
    std::cout << "\n=== Cactus Conversation FFI Example ===" << std::endl;
    
    try {
        cactus_init_params_c_t init_params = {};
        init_params.model_path = model_filename.c_str();
        init_params.chat_template = nullptr;
        init_params.n_ctx = 2048;
        init_params.n_batch = 512;
        init_params.n_ubatch = 512;
        init_params.n_gpu_layers = 99;
        init_params.n_threads = 4;
        init_params.use_mmap = true;
        init_params.use_mlock = false;
        init_params.embedding = false;
        init_params.pooling_type = 0;
        init_params.embd_normalize = 2;
        init_params.flash_attn = false;
        init_params.cache_type_k = nullptr;
        init_params.cache_type_v = nullptr;
        init_params.progress_callback = nullptr;

        std::cout << "Loading model: " << model_filename << std::endl;
        cactus_context_handle_t handle = cactus_init_context_c(&init_params);
        if (!handle) {
            std::cerr << "Failed to load model" << std::endl;
            return 1;
        }

        std::cout << "Model loaded successfully!" << std::endl;
        
        char* model_desc = cactus_get_model_desc_c(handle);
        std::cout << "Model: " << (model_desc ? model_desc : "Unknown") << std::endl;
        cactus_free_string_c(model_desc);
        
        run_conversation_demo(handle);

        cactus_free_context_c(handle);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 