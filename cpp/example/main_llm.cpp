#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <thread>
#include <chrono>
#include <iomanip>

#include "utils.h"
#include "../../cpp/cactus.h"

// Include nlohmann json for proper JSON handling
#include "../../cpp/json.hpp"
using json = nlohmann::ordered_json;

struct GenerationResult {
    std::string text;
    std::chrono::milliseconds time_to_first_token;
    std::chrono::milliseconds total_time;
    int tokens_generated;
};

GenerationResult generateText(cactus::cactus_context& context, const std::string& prompt, int max_tokens = 100) {
    context.generated_text.clear();
    auto start_time = std::chrono::high_resolution_clock::now();
    
    context.params.prompt = prompt;
    context.params.n_predict = max_tokens;
    
    if (!context.initSampling()) {
        std::cerr << "Failed to initialize sampling" << std::endl;
        return {"", std::chrono::milliseconds(0), std::chrono::milliseconds(0), 0};
    }
    
    context.beginCompletion();
    context.loadPrompt();
    
    bool first_token = true;
    std::chrono::high_resolution_clock::time_point first_token_time;
    int tokens_generated = 0;
    
    while (context.has_next_token && !context.is_interrupted) {
        auto token_output = context.doCompletion();
        if (token_output.tok == -1) break;
        
        if (first_token) {
            first_token_time = std::chrono::high_resolution_clock::now();
            first_token = false;
        }
        tokens_generated++;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();

    context.endCompletion();
    
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    auto ttft = first_token ? std::chrono::milliseconds(0) : 
                std::chrono::duration_cast<std::chrono::milliseconds>(first_token_time - start_time);
    
    return {context.generated_text, ttft, total_time, tokens_generated};
}

void runConversationDemo(cactus::cactus_context& context) {
    std::cout << "\n=== Automated Multi-Turn Conversation Demo ===" << std::endl;

    std::vector<std::string> messages = {
        "Hi there! Can you tell me about the history of personal computing?",
        "That's fascinating. What were some of the key breakthroughs in the 1980s?",
        "Excellent summary. Finally, what do you see as the next major evolution in computing?"
    };

    if (!context.initSampling()) {
        std::cerr << "Failed to initialize sampling" << std::endl;
        return;
    }

    for (const auto& message : messages) {
        std::cout << "\nUser: " << message << std::endl;

        context.generated_text.clear();
        
        json msgs = json::array();
        msgs.push_back({{"role", "user"}, {"content", message}});

        std::string prompt;
        if (context.embd.empty()) {
            prompt = context.getFormattedChat(msgs.dump(), "");
        } else {
            std::string user_part = context.getFormattedChat(msgs.dump(), "");
            size_t assistant_start = user_part.find("<|im_start|>assistant");
            if (assistant_start != std::string::npos) {
                prompt = user_part.substr(0, assistant_start) + "<|im_start|>assistant\n";
            } else {
                prompt = user_part;
            }
        }
        
        context.params.prompt = prompt;
        context.params.n_predict = 250;

        context.beginCompletion();
        context.loadPrompt();

        auto start_time = std::chrono::high_resolution_clock::now();
        bool first_token = true;
        std::chrono::high_resolution_clock::time_point first_token_time;
        int tokens_generated = 0;

        while (context.has_next_token && !context.is_interrupted) {
            auto token_output = context.doCompletion();
            if (token_output.tok == -1) break;

            if (first_token) {
                first_token_time = std::chrono::high_resolution_clock::now();
                first_token = false;
            }
            tokens_generated++;
        }

        context.endCompletion();
        auto end_time = std::chrono::high_resolution_clock::now();

        auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        auto ttft = first_token ? std::chrono::milliseconds(0) : 
                    std::chrono::duration_cast<std::chrono::milliseconds>(first_token_time - start_time);

        std::cout << "Bot: " << context.generated_text << std::endl;
        std::cout << "(TTFT: " << ttft.count() << "ms, "
                  << "Total: " << total_time.count() << "ms, "
                  << "Tokens: " << tokens_generated;
        
        if (tokens_generated > 0 && total_time.count() > 0) {
            float tokens_per_second = (float)tokens_generated * 1000.0f / total_time.count();
            std::cout << ", Speed: " << std::fixed << std::setprecision(1) << tokens_per_second << " tok/s";
        }
        std::cout << ")" << std::endl;
        std::cout << std::string(60, '-') << std::endl;
    }
}

int main(int argc, char **argv) {
    const std::string model_url = "https://huggingface.co/QuantFactory/SmolLM-360M-Instruct-GGUF/resolve/main/SmolLM-360M-Instruct.Q6_K.gguf";
    const std::string model_filename = "SmolLM-360M-Instruct.Q6_K.gguf";
    
    if (!downloadFile(model_url, model_filename, "LLM model")) {
        return 1;
    }
    
    std::cout << "\n=== Cactus LLM Example ===" << std::endl;
    
    try {
        cactus::cactus_context context;
        
        common_params params;
        params.model.path = model_filename;
        params.n_ctx = 4096;
        params.n_batch = 512;
        params.n_gpu_layers = 99;
        params.cpuparams.n_threads = std::thread::hardware_concurrency();
        
        params.n_cache_reuse = 256;
        params.n_keep = 32;
        
        params.sampling.temp = 0.7f;
        params.sampling.top_k = 40;
        params.sampling.top_p = 0.9f;
        params.sampling.penalty_repeat = 1.1f;
        
        params.antiprompt.push_back("<|im_end|>");
        
        std::cout << "Loading model: " << model_filename << std::endl;
        if (!context.loadModel(params)) {
            std::cerr << "Failed to load model" << std::endl;
            return 1;
        }
        
        std::cout << "Model loaded successfully!" << std::endl;
        
        runConversationDemo(context);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}