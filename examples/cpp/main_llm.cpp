#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <thread>
#include <chrono>

#include "utils.h"
#include "../../cactus/cactus.h"

std::string generateText(cactus::cactus_context& context, const std::string& prompt, int max_tokens = 100) {
    context.params.prompt = prompt;
    context.params.n_predict = max_tokens;
    
    if (!context.initSampling()) {
        std::cerr << "Failed to initialize sampling" << std::endl;
        return "";
    }
    
    context.rewind();
    context.beginCompletion();
    context.loadPrompt();
    
    while (context.has_next_token && !context.is_interrupted) {
        auto token_output = context.doCompletion();
        if (token_output.tok == -1) break;
    }
    
    return context.generated_text;
}

std::string chatWithModel(cactus::cactus_context& context, const std::vector<std::string>& conversation_history, const std::string& user_input) {
    // Build conversation context
    std::string messages = "[";
    for (size_t i = 0; i < conversation_history.size(); i += 2) {
        if (i < conversation_history.size()) {
            messages += R"({"role": "user", "content": ")" + conversation_history[i] + R"("})";
            if (i + 1 < conversation_history.size()) {
                messages += R"(,{"role": "assistant", "content": ")" + conversation_history[i + 1] + R"("})";
            }
            if (i + 2 < conversation_history.size()) messages += ",";
        }
    }
    
    // Add current user input
    if (!conversation_history.empty()) messages += ",";
    messages += R"({"role": "user", "content": ")" + user_input + R"("})";
    messages += "]";
    
    // Format with chat template
    std::string formatted_prompt;
    try {
        formatted_prompt = context.getFormattedChat(messages, "");
    } catch (const std::exception& e) {
        std::cerr << "Warning: Chat template formatting failed (" << e.what() << "), using raw prompt" << std::endl;
        formatted_prompt = user_input;
    }
    
    return generateText(context, formatted_prompt, 200);
}

void demonstrateBasicGeneration(cactus::cactus_context& context) {
    std::cout << "\n=== Basic Text Generation Demo ===" << std::endl;
    
    std::vector<std::string> prompts = {
        "The future of artificial intelligence is",
        "Write a short story about a robot who discovers emotions:",
        "Explain quantum computing in simple terms:",
        "What are the benefits of renewable energy?"
    };
    
    for (const auto& prompt : prompts) {
        std::cout << "\nPrompt: " << prompt << std::endl;
        std::cout << "Response: " << generateText(context, prompt, 150) << std::endl;
        std::cout << std::string(80, '-') << std::endl;
    }
}

void demonstrateChatMode(cactus::cactus_context& context) {
    std::cout << "\n=== Interactive Chat Demo ===" << std::endl;
    std::cout << "Type 'quit' to exit, 'clear' to reset conversation" << std::endl;
    
    std::vector<std::string> conversation_history;
    std::string input;
    
    while (true) {
        std::cout << "\nYou: ";
        std::getline(std::cin, input);
        
        if (input == "quit" || input == "exit") {
            break;
        }
        
        if (input == "clear") {
            conversation_history.clear();
            std::cout << "Conversation cleared." << std::endl;
            continue;
        }
        
        if (input.empty()) continue;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        std::string response = chatWithModel(context, conversation_history, input);
        auto end_time = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "Bot: " << response << std::endl;
        std::cout << "(Generated in " << duration.count() << "ms)" << std::endl;
        
        // Update conversation history
        conversation_history.push_back(input);
        conversation_history.push_back(response);
        
        // Keep conversation history manageable
        if (conversation_history.size() > 10) {
            conversation_history.erase(conversation_history.begin(), conversation_history.begin() + 2);
        }
    }
}

void demonstrateSamplingVariations(cactus::cactus_context& context) {
    std::cout << "\n=== Sampling Variations Demo ===" << std::endl;
    
    const std::string prompt = "Write a creative opening line for a science fiction novel:";
    
    struct SamplingConfig {
        std::string name;
        float temperature;
        int top_k;
        float top_p;
        float repeat_penalty;
    };
    
    std::vector<SamplingConfig> configs = {
        {"Conservative", 0.3f, 20, 0.8f, 1.05f},
        {"Balanced", 0.7f, 40, 0.9f, 1.1f},
        {"Creative", 1.0f, 60, 0.95f, 1.15f},
        {"Wild", 1.3f, 80, 0.98f, 1.2f}
    };
    
    for (const auto& config : configs) {
        std::cout << "\n" << config.name << " sampling (temp=" << config.temperature 
                  << ", top_k=" << config.top_k << ", top_p=" << config.top_p << "):" << std::endl;
        
        // Configure sampling parameters
        context.params.sampling.temp = config.temperature;
        context.params.sampling.top_k = config.top_k;
        context.params.sampling.top_p = config.top_p;
        context.params.sampling.penalty_repeat = config.repeat_penalty;
        
        std::string response = generateText(context, prompt, 80);
        std::cout << "Response: " << response << std::endl;
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
        
        // Configure model parameters
        common_params params;
        params.model.path = model_filename;
        params.n_ctx = 4096;
        params.n_batch = 512;
        params.n_gpu_layers = 99; // Use GPU acceleration
        params.cpuparams.n_threads = std::thread::hardware_concurrency();
        
        // Default sampling parameters
        params.sampling.temp = 0.7f;
        params.sampling.top_k = 40;
        params.sampling.top_p = 0.9f;
        params.sampling.penalty_repeat = 1.1f;
        
        // Configure stop tokens
        params.antiprompt.push_back("<|im_end|>");
        
        std::cout << "Loading model: " << model_filename << std::endl;
        if (!context.loadModel(params)) {
            std::cerr << "Failed to load model" << std::endl;
            return 1;
        }
        
        std::cout << "Model loaded successfully!" << std::endl;
        std::cout << "Model: " << context.llama_init.model.get() << std::endl;
        std::cout << "Context: " << context.llama_init.context.get() << std::endl;
        
        // Run different demonstrations
        if (argc > 1 && std::string(argv[1]) == "chat") {
            demonstrateChatMode(context);
        } else if (argc > 1 && std::string(argv[1]) == "sampling") {
            demonstrateSamplingVariations(context);
        } else if (argc > 1 && std::string(argv[1]) == "basic") {
            demonstrateBasicGeneration(context);
        } else {
            std::cout << "\nAvailable demos:" << std::endl;
            std::cout << "  ./cactus_llm basic    - Basic text generation examples" << std::endl;
            std::cout << "  ./cactus_llm chat     - Interactive chat mode" << std::endl;
            std::cout << "  ./cactus_llm sampling - Different sampling strategies" << std::endl;
            std::cout << "\nRunning basic demo by default...\n" << std::endl;
            
            demonstrateBasicGeneration(context);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}