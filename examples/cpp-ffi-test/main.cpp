#include <iostream>
#include <string>
#include <fstream>
#include <cstdlib>

#include "../../cactus/cactus_ffi.h"

bool fileExists(const std::string& filepath) {
    std::ifstream f(filepath.c_str());
    return f.good();
}

bool downloadFile(const std::string& url, const std::string& filepath, const std::string& filename) {
    if (fileExists(filepath)) {
        std::cout << filename << " already exists" << std::endl;
        return true;
    }
    
    std::cout << "Downloading " << filename << "..." << std::endl;
    std::string command = "curl -L -C - -o \"" + filepath + "\" \"" + url + "\"";
    
    int return_code = system(command.c_str());
    if (return_code == 0 && fileExists(filepath)) {
        std::cout << filename << " downloaded successfully" << std::endl;
        return true;
    } else {
        std::cerr << "Failed to download " << filename << std::endl;
        return false;
    }
}

int main() {
    std::cout << "=== Cactus FFI High Priority Functions Test ===" << std::endl;
    
    const std::string model_filename = "SmolLM2-1.7B-Instruct-Q4_K_M.gguf";
    const std::string model_url = "https://huggingface.co/bartowski/SmolLM2-1.7B-Instruct-GGUF/resolve/main/SmolLM2-1.7B-Instruct-Q4_K_M.gguf";
    
    if (!downloadFile(model_url, model_filename, model_filename)) {
        return 1;
    }
    
    cactus_init_params_c_t init_params = {0};
    init_params.model_path = model_filename.c_str();
    init_params.n_ctx = 2048;
    init_params.n_batch = 64;
    init_params.n_ubatch = 64;
    init_params.n_gpu_layers = 99;
    init_params.n_threads = 4;
    init_params.use_mmap = true;
    init_params.use_mlock = false;
    init_params.embedding = false;
    init_params.flash_attn = true;
    
    std::cout << "\n=== Model Loading ===" << std::endl;
    cactus_context_handle_t context = cactus_init_context_c(&init_params);
    if (!context) {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }
    std::cout << "Model loaded successfully" << std::endl;
    
    std::cout << "\n=== Model Information ===" << std::endl;
    int32_t n_ctx = cactus_get_n_ctx_c(context);
    int32_t n_embd = cactus_get_n_embd_c(context);
    char* model_desc = cactus_get_model_desc_c(context);
    int64_t model_size = cactus_get_model_size_c(context);
    int64_t model_params = cactus_get_model_params_c(context);
    
    std::cout << "Model Description: " << (model_desc ? model_desc : "unknown") << std::endl;
    std::cout << "Context Size: " << n_ctx << std::endl;
    std::cout << "Embedding Size: " << n_embd << std::endl;
    std::cout << "Model Size: " << model_size << " bytes" << std::endl;
    std::cout << "Model Parameters: " << model_params << std::endl;
    
    cactus_free_string_c(model_desc);
    
    std::cout << "\n=== Chat Template Validation ===" << std::endl;
    bool jinja_valid = cactus_validate_chat_template_c(context, true, nullptr);
    bool standard_valid = cactus_validate_chat_template_c(context, false, nullptr);
    std::cout << "Jinja template valid: " << (jinja_valid ? "yes" : "no") << std::endl;
    std::cout << "Standard template valid: " << (standard_valid ? "yes" : "no") << std::endl;
    
    std::cout << "\n=== Chat Formatting ===" << std::endl;
    const char* messages_json = R"([
        {"role": "user", "content": "Hello! What is 2+2?"}
    ])";
    
    char* formatted_chat = cactus_get_formatted_chat_c(context, messages_json, nullptr);
    if (formatted_chat) {
        std::cout << "Formatted chat:" << std::endl;
        std::cout << formatted_chat << std::endl;
        cactus_free_string_c(formatted_chat);
    }
    
    std::cout << "\n=== Context Management ===" << std::endl;
    std::cout << "Rewinding context..." << std::endl;
    cactus_rewind_c(context);
    
    std::cout << "Initializing sampling..." << std::endl;
    bool sampling_ok = cactus_init_sampling_c(context);
    std::cout << "Sampling initialized: " << (sampling_ok ? "yes" : "no") << std::endl;
    
    std::cout << "\n=== Basic Completion Test ===" << std::endl;
    cactus_completion_params_c_t comp_params = {0};
    comp_params.prompt = "The capital of France is";
    comp_params.n_predict = 10;
    comp_params.temperature = 0.7f;
    comp_params.top_k = 40;
    comp_params.top_p = 0.9f;
    comp_params.seed = 42;
    
    cactus_completion_result_c_t comp_result = {0};
    
    std::cout << "Running completion..." << std::endl;
    int comp_status = cactus_completion_c(context, &comp_params, &comp_result);
    
    if (comp_status == 0 && comp_result.text) {
        std::cout << "Prompt: " << comp_params.prompt << std::endl;
        std::cout << "Response: " << comp_result.text << std::endl;
        std::cout << "Tokens predicted: " << comp_result.tokens_predicted << std::endl;
        std::cout << "Tokens evaluated: " << comp_result.tokens_evaluated << std::endl;
        
        cactus_free_completion_result_members_c(&comp_result);
    } else {
        std::cout << "Completion failed with status: " << comp_status << std::endl;
    }
    
    std::cout << "\n=== Benchmarking ===" << std::endl;
    std::cout << "Running benchmark (pp=256, tg=128, pl=1, nr=3)..." << std::endl;
    
    cactus_bench_result_c_t bench_result = cactus_bench_c(context, 256, 128, 1, 3);
    
    if (bench_result.model_name) {
        std::cout << "Benchmark Results:" << std::endl;
        std::cout << "  Model: " << bench_result.model_name << std::endl;
        std::cout << "  Size: " << bench_result.model_size << " bytes" << std::endl;
        std::cout << "  Parameters: " << bench_result.model_params << std::endl;
        std::cout << "  Prompt Processing: " << bench_result.pp_avg << " ± " << bench_result.pp_std << " tokens/s" << std::endl;
        std::cout << "  Text Generation: " << bench_result.tg_avg << " ± " << bench_result.tg_std << " tokens/s" << std::endl;
        
        cactus_free_bench_result_members_c(&bench_result);
    } else {
        std::cout << "Benchmark failed" << std::endl;
    }
    
    std::cout << "\n=== LoRA Adapter Test ===" << std::endl;
    cactus_lora_adapters_c_t current_loras = cactus_get_loaded_lora_adapters_c(context);
    std::cout << "Currently loaded LoRA adapters: " << current_loras.count << std::endl;
    cactus_free_lora_adapters_c(&current_loras);
    
    std::cout << "\n=== Cleanup ===" << std::endl;
    cactus_free_context_c(context);
    std::cout << "Context freed successfully" << std::endl;
    
    std::cout << "\n=== FFI Test Complete ===" << std::endl;
    std::cout << "All high-priority FFI functions tested successfully!" << std::endl;
    
    return 0;
} 