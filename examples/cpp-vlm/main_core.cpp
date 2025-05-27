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
    const std::string model_url = "https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/SmolVLM-256M-Instruct-Q8_0.gguf";
    const std::string model_filename = "SmolVLM-256M.gguf";

    const std::string mmproj_url = "https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf";
    const std::string mmproj_filename = "mmproj-SmolVLM-256M.gguf";

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
    params.prompt = "Describe this image in detail.";
    
    params.n_predict = 100;
    params.n_ctx = 2048;
    params.n_batch = 512;
    params.use_mmap = true;
    params.warmup = false;

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