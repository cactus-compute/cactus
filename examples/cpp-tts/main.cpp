#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <thread>

#include "../../cactus/cactus.h"

bool fileExists(const std::string& filepath) {
    std::ifstream f(filepath.c_str()); 
    return f.good(); 
}

bool downloadFile(const std::string& url, const std::string& filepath, const std::string& filename_desc) {
    if (url.empty() || filepath.empty()) {
        if (filepath.empty()) {
             std::cout << "No filepath specified for " << filename_desc << ", skipping download." << std::endl;
        } else {
            std::cout << "No URL specified for " << filename_desc << " at " << filepath << ", skipping download." << std::endl;
        }
        return fileExists(filepath);
    }

    if (fileExists(filepath)) {
        std::cout << filename_desc << " already exists at " << filepath << std::endl;
        return true;
    }

    std::cout << "Downloading " << filename_desc << " from " << url << " to " << filepath << "..." << std::endl;
    std::string command = "curl -L -o \"" + filepath + "\" \"" + url + "\"";
    
    int return_code = system(command.c_str());

    if (return_code == 0 && fileExists(filepath)) {
        std::cout << filename_desc << " downloaded successfully." << std::endl;
        return true;
    } else {
        std::cerr << "Failed to download " << filename_desc << "." << std::endl;
        return false;
    }
}

void writeWavFile(const std::string& filename, const std::vector<float>& audio_data, int sample_rate = 24000) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }

    int num_samples = audio_data.size();
    int byte_rate = sample_rate * 2; // 16-bit mono
    int data_size = num_samples * 2;
    int file_size = 36 + data_size;

    // WAV header
    file.write("RIFF", 4);
    file.write(reinterpret_cast<const char*>(&file_size), 4);
    file.write("WAVE", 4);
    file.write("fmt ", 4);
    
    int fmt_size = 16;
    short audio_format = 1; // PCM
    short num_channels = 1; // Mono
    short bits_per_sample = 16;
    short block_align = num_channels * bits_per_sample / 8;
    
    file.write(reinterpret_cast<const char*>(&fmt_size), 4);
    file.write(reinterpret_cast<const char*>(&audio_format), 2);
    file.write(reinterpret_cast<const char*>(&num_channels), 2);
    file.write(reinterpret_cast<const char*>(&sample_rate), 4);
    file.write(reinterpret_cast<const char*>(&byte_rate), 4);
    file.write(reinterpret_cast<const char*>(&block_align), 2);
    file.write(reinterpret_cast<const char*>(&bits_per_sample), 2);
    
    file.write("data", 4);
    file.write(reinterpret_cast<const char*>(&data_size), 4);
    
    // Convert float audio data to 16-bit PCM
    for (float sample : audio_data) {
        short pcm_sample = static_cast<short>(std::max(-32768.0f, std::min(32767.0f, sample * 32767.0f)));
        file.write(reinterpret_cast<const char*>(&pcm_sample), 2);
    }
    
    file.close();
    std::cout << "Audio saved to " << filename << std::endl;
}

int main(int argc, char **argv) {
    const std::string model_url = "https://huggingface.co/OuteAI/OuteTTS-0.2-500M-GGUF/resolve/main/OuteTTS-0.2-500M-Q6_K.gguf";
    const std::string model_filename = "OuteTTS-0.2-500M-Q6_K.gguf";
    const std::string vocoder_model_url = "https://huggingface.co/ggml-org/WavTokenizer/resolve/main/WavTokenizer-Large-75-F16.gguf";
    const std::string vocoder_model_filename = "WavTokenizer-Large-75-F16.gguf";
    
    std::string text_to_speak = "This is a test run of the text to speech system for Cactus, I hope you enjoy it as much as I do, thank you";
    
    // Get text from command line if provided
    if (argc > 1) {
        text_to_speak = "";
        for (int i = 1; i < argc; i++) {
            if (i > 1) text_to_speak += " ";
            text_to_speak += argv[i];
        }
    }

    if (!downloadFile(model_url, model_filename, "TTS Model")) {
        return 1;
    }
    if (!downloadFile(vocoder_model_url, vocoder_model_filename, "Vocoder Model")) {
        return 1;
    }

    try {
        // Load TTS model
        common_params params;
        params.model.path = model_filename;
        params.n_ctx = 2048;
        params.n_batch = 512;
        params.n_gpu_layers = 99; // Enable GPU acceleration
        params.cpuparams.n_threads = std::thread::hardware_concurrency();
        
        // TTS-specific settings
        params.n_predict = 500;
        params.sampling.temp = 0.7f;
        params.sampling.top_k = 40;
        params.sampling.top_p = 0.9f;

        cactus::cactus_context context;
        
        std::cout << "Loading TTS model: " << model_filename << std::endl;
        if (!context.loadModel(params)) {
            std::cerr << "Failed to load TTS model." << std::endl;
            return 1;
        }

        std::cout << "Loading vocoder model: " << vocoder_model_filename << std::endl;
        if (!context.initVocoder(vocoder_model_filename)) {
            std::cerr << "Failed to load vocoder model." << std::endl;
            return 1;
        }

        if (!context.initSampling()) {
            std::cerr << "Failed to initialize sampling context." << std::endl;
            return 1;
        }

        std::cout << "Generating TTS prompt..." << std::endl;
        std::string formatted_prompt = context.getFormattedAudioCompletion("", text_to_speak);
        context.params.prompt = formatted_prompt;
        
        std::cout << "Getting guide tokens..." << std::endl;
        std::vector<llama_token> guide_tokens = context.getAudioCompletionGuideTokens(text_to_speak);
        context.setGuideTokens(guide_tokens);
        
        std::cout << "Starting TTS generation..." << std::endl;
        context.beginCompletion();
        context.loadPrompt();

        std::vector<llama_token> audio_tokens;
        int max_tokens = 500;
        int generated_tokens = 0;
        
        while (context.has_next_token && !context.is_interrupted && generated_tokens < max_tokens) {
            const auto token_output = context.doCompletion();
            generated_tokens++;
            
            // Check if token is in audio range (151672-155772)
            if (token_output.tok >= 151672 && token_output.tok <= 155772) {
                audio_tokens.push_back(token_output.tok);
            }
            
            // Check for end token
            if (token_output.tok == 151668) { // <|audio_end|>
                std::cout << "Found audio end token" << std::endl;
                break;
            }
        }

        std::cout << "Generated " << audio_tokens.size() << " audio tokens" << std::endl;
        
        if (audio_tokens.empty()) {
            std::cerr << "No audio tokens generated!" << std::endl;
            return 1;
        }

        std::cout << "Decoding audio tokens..." << std::endl;
        std::vector<float> audio_data = context.decodeAudioTokens(audio_tokens);
        
        if (audio_data.empty()) {
            std::cerr << "Failed to decode audio tokens!" << std::endl;
            return 1;
        }

        std::cout << "Generated " << audio_data.size() << " audio samples" << std::endl;
        
        std::string output_filename = "output.wav";
        writeWavFile(output_filename, audio_data);
        
        std::cout << "TTS generation complete! Audio saved to " << output_filename << std::endl;
        std::cout << "You can play it with: aplay " << output_filename << " (Linux) or open " << output_filename << " (macOS)" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}