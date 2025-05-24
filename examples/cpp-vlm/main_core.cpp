#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <cassert>
#include <cstring>

#include "../../cactus/cactus.h"

bool fileExists(const std::string& filepath) {
    std::ifstream f(filepath.c_str());
    return f.good();
}

bool downloadFile(const std::string& url, const std::string& filepath, const std::string& filename) {
    if (fileExists(filepath)) {
        std::cout << filename << " already exists at " << filepath << std::endl;
        return true;
    }

    std::cout << "Downloading " << filename << " from " << url << " to " << filepath << "..." << std::endl;
    std::string command = "curl -L -o \"" + filepath + "\" \"" + url + "\"";
    
    int return_code = system(command.c_str());

    if (return_code == 0 && fileExists(filepath)) {
        std::cout << filename << " downloaded successfully." << std::endl;
        return true;
    } else {
        std::cerr << "Failed to download " << filename << "." << std::endl;
        std::cerr << "Please ensure curl is installed and the URL is correct." << std::endl;
        std::cerr << "You can try downloading it manually using the command:" << std::endl;
        std::cerr << command << std::endl;
        
        if (fileExists(filepath)) {
            std::remove(filepath.c_str());
        }
        return false;
    }
}

int main(int argc, char **argv) {
    const std::string model_url = "https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF/resolve/main/SmolVLM-500M-Instruct-Q8_0.gguf";
    const std::string model_filename = "SmolVLM-500M-Instruct-Q8_0.gguf";
    
    const std::string mmproj_url = "https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF/resolve/main/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf";
    const std::string mmproj_filename = "mmproj-SmolVLM-500M-Instruct-Q8_0.gguf";

    if (!downloadFile(model_url, model_filename, "VLM model")) {
        return 1;
    }

    if (!downloadFile(mmproj_url, mmproj_filename, "Multimodal projector")) {
        return 1;
    }
    
    common_params params;
    params.model.path = model_filename;
    params.mmproj.path = mmproj_filename;
    params.image.push_back("../image.jpg"); 
    params.prompt = "USER: <__image__>\nDescribe this image in detail.\nASSISTANT:";
    
    params.n_predict = 100;           // Maximum number of tokens to generate in the output (response length)
    params.n_ctx = 2048;              // Context window size: how many tokens (prompt + response) the model can consider at once
    params.n_batch = 512;             // Batch size: number of tokens to process in parallel (affects speed and memory use)
    params.cpuparams.n_threads = 4;   // Number of CPU threads to use for computation (parallelism; set to number of CPU cores for speed)
    params.use_mmap = true;           // Use memory-mapped file I/O for loading the model (can reduce RAM usage for large models)
    params.warmup = false;            // Whether to run a model "warm-up" before actual inference (often set true for benchmarking, false for normal runs)

    cactus::cactus_context ctx;
    assert(ctx.loadModel(params) && "Model loading failed");
    assert(ctx.initSampling() && "Sampling initialization failed");

    ctx.loadPrompt();
    ctx.beginCompletion();

    std::string response;
    const llama_vocab * vocab = llama_model_get_vocab(ctx.model);
    while (ctx.has_next_token) {
        auto tok = ctx.nextToken();
        if (tok.tok < 0) break;
        if (tok.tok == llama_vocab_eos(vocab)) break;

        char buffer[64];
        int length = llama_token_to_piece(vocab, tok.tok, buffer, sizeof(buffer), false, false);
        if (length > 0) {
            response += std::string(buffer, length);
        }
    }

    assert(!response.empty() && "Response should not be empty");
    std::cout << "Response: " << response << std::endl;
    std::cout << "Basic completion test passed" << std::endl;

    return 0;
}
