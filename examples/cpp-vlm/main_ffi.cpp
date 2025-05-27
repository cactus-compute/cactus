#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include "../../cactus/cactus_ffi.h"

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
    const std::string model_url_str = "https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/SmolVLM-256M-Instruct-Q8_0.gguf";
    const std::string model_filename_str = "SmolVLM-256M.gguf";

    const std::string mmproj_url_str = "https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf";
    const std::string mmproj_filename_str = "mmproj-SmolVLM-256M.gguf";

    if (!downloadFile(model_url_str, model_filename_str, "VLM model")) {
        return 1;
    }

    if (!downloadFile(mmproj_url_str, mmproj_filename_str, "Multimodal projector")) {
        return 1;
    }

    cactus_init_params_c_t params = cactus_default_init_params_c();
    params.model_path = model_filename_str.c_str();
    params.mmproj_path = mmproj_filename_str.c_str();
    params.n_ctx = 2048;
    params.n_batch = 512;
    params.n_gpu_layers = 10;
    params.use_mmap = true;
    params.use_mlock = true;
    params.warmup = false; 
    params.mmproj_use_gpu = true;

    cactus_context_handle_t ctx_handle = cactus_init_context_c(&params);
    assert(ctx_handle != nullptr && "Failed to initialize cactus context");

    cactus_completion_params_c_t comp_params = {};
    std::string prompt_str = "Describe this image in detail.";
    std::string image_path_str = "../image.jpg";
    comp_params.prompt = prompt_str.c_str();
    comp_params.image_path = image_path_str.c_str();
    comp_params.n_predict = 100;
    
    comp_params.token_callback = [](const char* token_text) {
        std::cout << token_text << std::flush;
        return true; 
    };

    cactus_completion_result_c_t result = {};
    std::cout << "Starting completion (streaming enabled)..." << std::endl;
    std::cout << "Response: "; 
    int completion_status = cactus_completion_c(ctx_handle, &comp_params, &result);
    std::cout << std::endl; 

    if (completion_status == 0) {
        std::cout << "Tokens predicted: " << result.tokens_predicted << std::endl;
        std::cout << "Tokens evaluated (prompt): " << result.tokens_evaluated << std::endl;
        if (result.stopped_eos) std::cout << "Stopped due to EOS." << std::endl;
        if (result.stopped_word) std::cout << "Stopped due to stop word: " << (result.stopping_word ? result.stopping_word : "N/A") << std::endl;
        if (result.stopped_limit) std::cout << "Stopped due to prediction limit." << std::endl;
        if (result.truncated) std::cout << "Prompt was truncated." << std::endl;

        cactus_free_completion_result_members_c(&result);
    } else {
        std::cerr << std::endl << "Completion failed with status: " << completion_status << std::endl;
    }

    cactus_free_context_c(ctx_handle);
    std::cout << "Cactus context freed." << std::endl;

    std::cout << "FFI VLM completion test finished." << std::endl;
    return 0;
}