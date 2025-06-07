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
    const std::string model_url = "https://huggingface.co/lm-kit/qwen-3-0.6b-instruct-gguf/resolve/main/Qwen3-0.6B-Q6_K.gguf";
    const std::string model_filename = "Qwen3-0.6B-Q6_K.gguf";
    
    if (!downloadFile(model_url, model_filename, "LLM")) {
        return 1;
    }
    
    return 0;
}