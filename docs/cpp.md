# Cactus C++

A GGML-based C++ library for running Large Language Models (LLMs) and Vision Language Models (VLMs) locally on small devices, with full support for chat completions, multimodal inputs, embeddings, text-to-speech and advanced features.

## Installation

### Build from Source

```bash
git clone https://github.com/your-org/cactus.git
cd cactus
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### CMake Integration

Add to your `CMakeLists.txt`:

```cmake
# Add Cactus as subdirectory
add_subdirectory(cactus)

# Link to your target
target_link_libraries(your_target cactus)
target_include_directories(your_target PRIVATE cactus)
```

**Requirements:**
- C++17 or higher
- CMake 3.14+

## Quick Start

### Basic Text Completion

```cpp
#include "cactus/cactus.h"
#include <iostream>

int main() {
    cactus::cactus_context context;
    
    // Configure parameters
    common_params params;
    params.model.path = "model.gguf";
    params.n_ctx = 2048;
    params.n_threads = 4;
    params.n_gpu_layers = 99; // Use GPU acceleration
    
    // Load model
    if (!context.loadModel(params)) {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }
    
    // Set prompt
    context.params.prompt = "Hello, how are you?";
    context.params.n_predict = 100;
    
    // Initialize sampling
    if (!context.initSampling()) {
        std::cerr << "Failed to initialize sampling" << std::endl;
        return 1;
    }
    
    // Generate response
    context.beginCompletion();
    context.loadPrompt();
    
    while (context.has_next_token && !context.is_interrupted) {
        auto token_output = context.doCompletion();
        if (token_output.tok == -1) break;
    }
    
    std::cout << "Response: " << context.generated_text << std::endl;
    return 0;
}
```

### Complete Chat Application

```cpp
#include "cactus/cactus.h"
#include <iostream>
#include <string>
#include <vector>
#include <fstream>

bool downloadModel(const std::string& url, const std::string& filename) {
    std::ifstream f(filename);
    if (f.good()) {
        std::cout << filename << " already exists" << std::endl;
        return true;
    }
    
    std::cout << "Downloading " << filename << "..." << std::endl;
    std::string command = "curl -L -o \"" + filename + "\" \"" + url + "\"";
    return system(command.c_str()) == 0;
}

class ChatBot {
private:
    cactus::cactus_context context;
    std::vector<std::string> conversation_history;
    
public:
    bool initialize(const std::string& model_path) {
        common_params params;
        params.model.path = model_path;
        params.n_ctx = 4096;
        params.n_batch = 512;
        params.n_gpu_layers = 99;
        params.cpuparams.n_threads = std::thread::hardware_concurrency();
        
        // Generation settings
        params.sampling.temp = 0.7f;
        params.sampling.top_k = 40;
        params.sampling.top_p = 0.9f;
        params.sampling.repeat_penalty = 1.1f;
        
        return context.loadModel(params);
    }
    
    std::string chat(const std::string& user_input) {
        // Build conversation context
        std::string messages = "[";
        for (const auto& msg : conversation_history) {
            messages += msg + ",";
        }
        
        // Add user message
        std::string user_msg = R"({"role": "user", "content": ")" + user_input + R"("})";
        messages += user_msg + "]";
        
        // Format with chat template
        std::string formatted_prompt = context.getFormattedChat(messages, "");
        
        context.params.prompt = formatted_prompt;
        context.params.n_predict = 256;
        
        if (!context.initSampling()) {
            return "Error: Failed to initialize sampling";
        }
        
        context.rewind();
        context.beginCompletion();
        context.loadPrompt();
        
        while (context.has_next_token && !context.is_interrupted) {
            auto token_output = context.doCompletion();
            if (token_output.tok == -1) break;
        }
        
        std::string response = context.generated_text;
        
        // Update conversation history
        conversation_history.push_back(user_msg);
        std::string assistant_msg = R"({"role": "assistant", "content": ")" + response + R"("})";
        conversation_history.push_back(assistant_msg);
        
        return response;
    }
    
    void clearHistory() {
        conversation_history.clear();
    }
};

int main() {
    const std::string model_url = "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf";
    const std::string model_filename = "phi-3-mini.gguf";
    
    if (!downloadModel(model_url, model_filename)) {
        std::cerr << "Failed to download model" << std::endl;
        return 1;
    }
    
    ChatBot bot;
    if (!bot.initialize(model_filename)) {
        std::cerr << "Failed to initialize chat bot" << std::endl;
        return 1;
    }
    
    std::cout << "ChatBot initialized! Type 'quit' to exit.\n" << std::endl;
    
    std::string input;
    while (true) {
        std::cout << "You: ";
        std::getline(std::cin, input);
        
        if (input == "quit" || input == "exit") break;
        if (input == "clear") {
            bot.clearHistory();
            std::cout << "Conversation cleared." << std::endl;
            continue;
        }
        
        std::string response = bot.chat(input);
        std::cout << "Bot: " << response << std::endl << std::endl;
    }
    
    return 0;
}
```

## Core Concepts

### Context Management

The `cactus_context` is the main interface for model operations:

```cpp
cactus::cactus_context context;

// Configure model parameters
common_params params;
params.model.path = "model.gguf";        // Model file path
params.n_ctx = 4096;                     // Context window size
params.n_batch = 512;                    // Batch size for processing
params.n_gpu_layers = 99;                // GPU layers (0 = CPU only)
params.cpuparams.n_threads = 8;          // CPU threads

// Sampling parameters
params.sampling.temp = 0.7f;             // Temperature
params.sampling.top_k = 40;              // Top-k sampling
params.sampling.top_p = 0.9f;            // Nucleus sampling
params.sampling.repeat_penalty = 1.1f;   // Repetition penalty

// Load model
bool success = context.loadModel(params);
```

### Generation Workflow

The standard generation process follows this pattern:

```cpp
// 1. Set prompt and generation parameters
context.params.prompt = "Your prompt here";
context.params.n_predict = 256;

// 2. Initialize sampling context
context.initSampling();

// 3. Reset context state
context.rewind();

// 4. Begin generation
context.beginCompletion();

// 5. Load prompt into context
context.loadPrompt();

// 6. Generate tokens
while (context.has_next_token && !context.is_interrupted) {
    auto token_output = context.doCompletion();
    if (token_output.tok == -1) break;
}

// 7. Get generated text
std::string result = context.generated_text;
```

## Text Completion

### Basic Generation

```cpp
std::string generateText(cactus::cactus_context& context, const std::string& prompt) {
    context.params.prompt = prompt;
    context.params.n_predict = 200;
    context.params.sampling.temp = 0.8f;
    
    if (!context.initSampling()) {
        return "Error: Failed to initialize sampling";
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
```

### Streaming Generation

```cpp
void generateStreaming(cactus::cactus_context& context, const std::string& prompt) {
    context.params.prompt = prompt;
    context.params.n_predict = 500;
    
    context.initSampling();
    context.rewind();
    context.beginCompletion();
    context.loadPrompt();
    
    std::cout << "Streaming response: ";
    std::string current_text;
    
    while (context.has_next_token && !context.is_interrupted) {
        auto token_output = context.doCompletion();
        if (token_output.tok == -1) break;
        
        // Print only new tokens
        if (context.generated_text.length() > current_text.length()) {
            std::string new_token = context.generated_text.substr(current_text.length());
            std::cout << new_token << std::flush;
            current_text = context.generated_text;
        }
    }
    std::cout << std::endl;
}
```

### Advanced Parameter Control

```cpp
void configureAdvancedSampling(common_params& params) {
    // Temperature control
    params.sampling.temp = 0.7f;          // Randomness (0.0 - 2.0)
    params.sampling.top_k = 40;           // Top-k sampling
    params.sampling.top_p = 0.9f;         // Nucleus sampling
    params.sampling.min_p = 0.05f;        // Minimum probability
    
    // Repetition control
    params.sampling.repeat_penalty = 1.1f; // Repetition penalty
    params.sampling.frequency_penalty = 0.0f;
    params.sampling.presence_penalty = 0.0f;
    
    // Mirostat sampling
    params.sampling.mirostat = 0;         // 0=disabled, 1=v1, 2=v2
    params.sampling.mirostat_tau = 5.0f;  // Target entropy
    params.sampling.mirostat_eta = 0.1f;  // Learning rate
    
    // Generation limits
    params.n_predict = 256;               // Max tokens to generate
    params.sampling.seed = -1;            // Random seed (-1 = random)
}
```

## Multimodal (Vision)

### Setup Vision Model

```cpp
#include "cactus/cactus.h"

bool setupVisionModel(cactus::cactus_context& context, 
                     const std::string& model_path,
                     const std::string& mmproj_path) {
    // Load base model
    common_params params;
    params.model.path = model_path;
    params.n_ctx = 4096;
    params.n_gpu_layers = 99;
    
    if (!context.loadModel(params)) {
        std::cerr << "Failed to load VLM model" << std::endl;
        return false;
    }
    
    // Initialize multimodal capabilities
    if (!context.initMultimodal(mmproj_path, true)) {
        std::cerr << "Failed to initialize multimodal" << std::endl;
        return false;
    }
    
    std::cout << "Vision support: " << (context.isMultimodalSupportVision() ? "Yes" : "No") << std::endl;
    return true;
}
```

### Image Analysis

```cpp
std::string analyzeImage(cactus::cactus_context& context, 
                        const std::string& image_path, 
                        const std::string& question) {
    // Format multimodal prompt
    std::string messages = R"([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": ")" + question + R"("}]}])";
    std::string formatted_prompt = context.getFormattedChat(messages, "");
    
    context.params.prompt = formatted_prompt;
    context.params.n_predict = 200;
    context.params.sampling.temp = 0.3f; // Lower temperature for factual responses
    
    if (!context.initSampling()) {
        return "Error: Failed to initialize sampling";
    }
    
    context.rewind();
    context.beginCompletion();
    
    // Load prompt with image
    std::vector<std::string> media_paths = {image_path};
    context.loadPrompt(media_paths);
    
    while (context.has_next_token && !context.is_interrupted) {
        auto token_output = context.doCompletion();
        if (token_output.tok == -1) break;
    }
    
    return context.generated_text;
}
```

### Multi-Image Comparison

```cpp
std::string compareImages(cactus::cactus_context& context,
                         const std::vector<std::string>& image_paths,
                         const std::string& question) {
    std::string messages = R"([{"role": "user", "content": [)";
    
    // Add image references
    for (size_t i = 0; i < image_paths.size(); ++i) {
        messages += R"({"type": "image"})";
        if (i < image_paths.size() - 1) messages += ",";
    }
    
    messages += R"(, {"type": "text", "text": ")" + question + R"("}]}])";
    
    std::string formatted_prompt = context.getFormattedChat(messages, "");
    context.params.prompt = formatted_prompt;
    context.params.n_predict = 300;
    
    context.initSampling();
    context.rewind();
    context.beginCompletion();
    context.loadPrompt(image_paths);
    
    while (context.has_next_token && !context.is_interrupted) {
        auto token_output = context.doCompletion();
        if (token_output.tok == -1) break;
    }
    
    return context.generated_text;
}
```

### Complete Vision Example

```cpp
#include "cactus/cactus.h"
#include <iostream>
#include <fstream>

class VisionAnalyzer {
private:
    cactus::cactus_context context;
    
public:
    bool initialize(const std::string& model_path, const std::string& mmproj_path) {
        common_params params;
        params.model.path = model_path;
        params.n_ctx = 4096;
        params.n_batch = 32;
        params.n_gpu_layers = 99;
        params.cpuparams.n_threads = 4;
        
        if (!context.loadModel(params)) return false;
        return context.initMultimodal(mmproj_path, true);
    }
    
    std::string describe(const std::string& image_path) {
        return analyzeImage(context, image_path, "Describe this image in detail.");
    }
    
    std::string answer(const std::string& image_path, const std::string& question) {
        return analyzeImage(context, image_path, question);
    }
    
    std::string compare(const std::vector<std::string>& images) {
        return compareImages(context, images, "Compare these images and explain the differences.");
    }
};

int main() {
    VisionAnalyzer analyzer;
    
    if (!analyzer.initialize("vlm-model.gguf", "mmproj.gguf")) {
        std::cerr << "Failed to initialize vision analyzer" << std::endl;
        return 1;
    }
    
    // Analyze single image
    std::string description = analyzer.describe("image.jpg");
    std::cout << "Image description: " << description << std::endl;
    
    // Answer specific question
    std::string answer = analyzer.answer("image.jpg", "What colors are prominent in this image?");
    std::cout << "Color analysis: " << answer << std::endl;
    
    // Compare multiple images
    std::vector<std::string> images = {"image1.jpg", "image2.jpg"};
    std::string comparison = analyzer.compare(images);
    std::cout << "Comparison: " << comparison << std::endl;
    
    return 0;
}
```

## Text-to-Speech (TTS)

### Setup TTS Model

```cpp
bool setupTTS(cactus::cactus_context& context,
              const std::string& model_path,
              const std::string& vocoder_path) {
    // Load TTS model
    common_params params;
    params.model.path = model_path;
    params.n_ctx = 2048;
    params.n_gpu_layers = 99;
    
    if (!context.loadModel(params)) {
        std::cerr << "Failed to load TTS model" << std::endl;
        return false;
    }
    
    // Initialize vocoder
    if (!context.initVocoder(vocoder_path)) {
        std::cerr << "Failed to initialize vocoder" << std::endl;
        return false;
    }
    
    std::cout << "TTS Type: " << static_cast<int>(context.getTTSType()) << std::endl;
    return true;
}
```

### Basic Speech Generation

```cpp
void writeWavFile(const std::string& filename, const std::vector<float>& audio_data, int sample_rate = 24000) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) return;

    int num_samples = audio_data.size();
    int byte_rate = sample_rate * 2;
    int data_size = num_samples * 2;
    int file_size = 36 + data_size;

    // WAV header
    file.write("RIFF", 4);
    file.write(reinterpret_cast<const char*>(&file_size), 4);
    file.write("WAVE", 4);
    file.write("fmt ", 4);
    
    int fmt_size = 16;
    short audio_format = 1, num_channels = 1, bits_per_sample = 16;
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
    
    // Convert float to 16-bit PCM
    for (float sample : audio_data) {
        short pcm_sample = static_cast<short>(std::max(-32768.0f, std::min(32767.0f, sample * 32767.0f)));
        file.write(reinterpret_cast<const char*>(&pcm_sample), 2);
    }
}

std::vector<float> generateSpeech(cactus::cactus_context& context, const std::string& text) {
    // Format TTS prompt
    std::string formatted_prompt = context.getFormattedAudioCompletion("", text);
    context.params.prompt = formatted_prompt;
    context.params.n_predict = 500;
    context.params.sampling.temp = 0.7f;
    
    // Get guide tokens
    std::vector<llama_token> guide_tokens = context.getAudioCompletionGuideTokens(text);
    context.setGuideTokens(guide_tokens);
    
    if (!context.initSampling()) {
        std::cerr << "Failed to initialize sampling" << std::endl;
        return {};
    }
    
    context.rewind();
    context.beginCompletion();
    context.loadPrompt();
    
    std::vector<llama_token> audio_tokens;
    int max_tokens = 500;
    int generated = 0;
    
    while (context.has_next_token && !context.is_interrupted && generated < max_tokens) {
        auto token_output = context.doCompletion();
        generated++;
        
        // Check if token is in audio range
        if (token_output.tok >= 151672 && token_output.tok <= 155772) {
            audio_tokens.push_back(token_output.tok);
        }
        
        // Check for end token
        if (token_output.tok == 151668) break;
    }
    
    if (audio_tokens.empty()) {
        std::cerr << "No audio tokens generated" << std::endl;
        return {};
    }
    
    return context.decodeAudioTokens(audio_tokens);
}
```

### Complete TTS Example

```cpp
class TextToSpeech {
private:
    cactus::cactus_context context;
    
public:
    bool initialize(const std::string& model_path, const std::string& vocoder_path) {
        return setupTTS(context, model_path, vocoder_path);
    }
    
    bool speakToFile(const std::string& text, const std::string& output_file) {
        std::vector<float> audio_data = generateSpeech(context, text);
        if (audio_data.empty()) return false;
        
        writeWavFile(output_file, audio_data);
        std::cout << "Audio saved to " << output_file << std::endl;
        return true;
    }
    
    std::vector<float> speak(const std::string& text) {
        return generateSpeech(context, text);
    }
};

int main() {
    TextToSpeech tts;
    
    if (!tts.initialize("tts-model.gguf", "vocoder.gguf")) {
        std::cerr << "Failed to initialize TTS" << std::endl;
        return 1;
    }
    
    std::string text = "Hello, this is a test of the text-to-speech system.";
    
    if (tts.speakToFile(text, "output.wav")) {
        std::cout << "Speech generated successfully!" << std::endl;
        std::cout << "Play with: aplay output.wav (Linux) or open output.wav (macOS)" << std::endl;
    }
    
    return 0;
}
```

## Embeddings

### Setup Embedding Model

```cpp
bool setupEmbeddings(cactus::cactus_context& context, const std::string& model_path) {
    common_params params;
    params.model.path = model_path;
    params.n_ctx = 512;
    params.n_batch = 32;
    params.n_gpu_layers = 99;
    params.embedding = true;                    // Enable embedding mode
    params.embd_normalize = 2;                  // L2 normalization
    params.pooling_type = LLAMA_POOLING_TYPE_MEAN; // Mean pooling
    
    return context.loadModel(params);
}
```

### Generate Embeddings

```cpp
std::vector<float> getEmbedding(cactus::cactus_context& context, const std::string& text) {
    context.rewind();
    context.params.prompt = text;
    context.params.n_predict = 0; // No text generation, just embeddings
    
    if (!context.initSampling()) {
        std::cerr << "Failed to initialize sampling" << std::endl;
        return {};
    }
    
    context.beginCompletion();
    context.loadPrompt();
    context.doCompletion(); // Process the text
    
    common_params embd_params;
    embd_params.embd_normalize = context.params.embd_normalize;
    
    return context.getEmbedding(embd_params);
}

float cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return 0.0f;
    
    float dot_product = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    
    for (size_t i = 0; i < a.size(); ++i) {
        dot_product += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    norm_a = std::sqrt(norm_a);
    norm_b = std::sqrt(norm_b);
    
    return (norm_a == 0.0f || norm_b == 0.0f) ? 0.0f : dot_product / (norm_a * norm_b);
}
```

### Similarity Search

```cpp
#include <algorithm>
#include <iomanip>

class EmbeddingSearch {
private:
    cactus::cactus_context context;
    std::vector<std::string> texts;
    std::vector<std::vector<float>> embeddings;
    
public:
    bool initialize(const std::string& model_path) {
        return setupEmbeddings(context, model_path);
    }
    
    void addText(const std::string& text) {
        auto embedding = getEmbedding(context, text);
        if (!embedding.empty()) {
            texts.push_back(text);
            embeddings.push_back(embedding);
        }
    }
    
    std::vector<std::pair<std::string, float>> search(const std::string& query, int top_k = 5) {
        auto query_embedding = getEmbedding(context, query);
        if (query_embedding.empty()) return {};
        
        std::vector<std::pair<std::string, float>> results;
        
        for (size_t i = 0; i < texts.size(); ++i) {
            float similarity = cosineSimilarity(query_embedding, embeddings[i]);
            results.emplace_back(texts[i], similarity);
        }
        
        // Sort by similarity (highest first)
        std::sort(results.begin(), results.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // Return top_k results
        if (results.size() > static_cast<size_t>(top_k)) {
            results.resize(top_k);
        }
        
        return results;
    }
    
    void printSimilarityMatrix() {
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "\nSimilarity Matrix:\n";
        
        for (size_t i = 0; i < embeddings.size(); ++i) {
            for (size_t j = 0; j < embeddings.size(); ++j) {
                float sim = cosineSimilarity(embeddings[i], embeddings[j]);
                std::cout << std::setw(6) << sim << " ";
            }
            std::cout << " | " << texts[i] << std::endl;
        }
    }
};

int main() {
    EmbeddingSearch search;
    
    if (!search.initialize("embedding-model.gguf")) {
        std::cerr << "Failed to initialize embedding model" << std::endl;
        return 1;
    }
    
    // Add documents
    search.addText("The cat sits on the mat.");
    search.addText("A feline rests on the carpet.");
    search.addText("The dog runs in the park.");
    search.addText("I love programming in Python.");
    search.addText("Machine learning is fascinating.");
    
    // Search for similar texts
    auto results = search.search("Animals and pets", 3);
    
    std::cout << "Search results for 'Animals and pets':" << std::endl;
    for (const auto& result : results) {
        std::cout << std::fixed << std::setprecision(4);
        std::cout << result.second << " - " << result.first << std::endl;
    }
    
    search.printSimilarityMatrix();
    
    return 0;
}
```

## Advanced Features

### Session Management

```cpp
bool saveSession(cactus::cactus_context& context, const std::string& session_path) {
    // Sessions are managed internally by the context
    // Save current state for later restoration
    std::ofstream file(session_path + ".info");
    if (!file.is_open()) return false;
    
    file << "tokens_processed: " << context.n_past << std::endl;
    file << "context_size: " << context.n_ctx << std::endl;
    file << "generated_tokens: " << context.num_tokens_predicted << std::endl;
    
    return true;
}

bool loadSession(cactus::cactus_context& context, const std::string& session_path) {
    std::ifstream file(session_path + ".info");
    if (!file.is_open()) return false;
    
    std::string line;
    while (std::getline(file, line)) {
        std::cout << "Session info: " << line << std::endl;
    }
    
    return true;
}
```

### LoRA Adapters

```cpp
bool applyLoRAAdapters(cactus::cactus_context& context, 
                      const std::vector<std::string>& lora_paths) {
    std::vector<common_adapter_lora_info> lora_adapters;
    
    for (const auto& path : lora_paths) {
        common_adapter_lora_info lora_info;
        lora_info.path = path;
        lora_info.scaled = 1.0f; // Full strength
        lora_adapters.push_back(lora_info);
    }
    
    int result = context.applyLoraAdapters(lora_adapters);
    std::cout << "Applied " << result << " LoRA adapters" << std::endl;
    
    return result > 0;
}

void removeLoRAAdapters(cactus::cactus_context& context) {
    context.removeLoraAdapters();
    std::cout << "LoRA adapters removed" << std::endl;
}

void listLoRAAdapters(cactus::cactus_context& context) {
    auto adapters = context.getLoadedLoraAdapters();
    std::cout << "Loaded LoRA adapters:" << std::endl;
    
    for (const auto& adapter : adapters) {
        std::cout << "- " << adapter.path << " (scale: " << adapter.scaled << ")" << std::endl;
    }
}
```

### Performance Monitoring

```cpp
struct PerformanceMetrics {
    size_t prompt_tokens = 0;
    size_t generated_tokens = 0;
    double generation_time = 0.0;
    double tokens_per_second = 0.0;
};

PerformanceMetrics measurePerformance(cactus::cactus_context& context, 
                                     const std::string& prompt) {
    PerformanceMetrics metrics;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    context.params.prompt = prompt;
    context.params.n_predict = 100;
    
    context.initSampling();
    context.rewind();
    context.beginCompletion();
    context.loadPrompt();
    
    metrics.prompt_tokens = context.num_prompt_tokens;
    
    while (context.has_next_token && !context.is_interrupted) {
        auto token_output = context.doCompletion();
        if (token_output.tok == -1) break;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    metrics.generated_tokens = context.num_tokens_predicted;
    metrics.generation_time = duration.count() / 1000.0;
    metrics.tokens_per_second = metrics.generated_tokens / metrics.generation_time;
    
    return metrics;
}

void printMetrics(const PerformanceMetrics& metrics) {
    std::cout << "Performance Metrics:" << std::endl;
    std::cout << "- Prompt tokens: " << metrics.prompt_tokens << std::endl;
    std::cout << "- Generated tokens: " << metrics.generated_tokens << std::endl;
    std::cout << "- Generation time: " << std::fixed << std::setprecision(2) 
              << metrics.generation_time << "s" << std::endl;
    std::cout << "- Speed: " << std::fixed << std::setprecision(2) 
              << metrics.tokens_per_second << " tokens/sec" << std::endl;
}
```

### Custom Sampling

```cpp
class CustomSampler {
private:
    float temperature = 0.7f;
    int top_k = 40;
    float top_p = 0.9f;
    float repeat_penalty = 1.1f;
    
public:
    void setTemperature(float temp) { temperature = temp; }
    void setTopK(int k) { top_k = k; }
    void setTopP(float p) { top_p = p; }
    void setRepeatPenalty(float penalty) { repeat_penalty = penalty; }
    
    void applySampling(cactus::cactus_context& context) {
        context.params.sampling.temp = temperature;
        context.params.sampling.top_k = top_k;
        context.params.sampling.top_p = top_p;
        context.params.sampling.repeat_penalty = repeat_penalty;
    }
    
    // Dynamic temperature adjustment based on context
    void adjustTemperatureByLength(cactus::cactus_context& context) {
        size_t generated = context.num_tokens_predicted;
        
        if (generated < 50) {
            temperature = 0.5f; // More focused for short responses
        } else if (generated < 150) {
            temperature = 0.7f; // Balanced
        } else {
            temperature = 0.9f; // More creative for longer responses
        }
        
        context.params.sampling.temp = temperature;
    }
};
```

## API Reference

### Core Classes

**`cactus::cactus_context`** - Main interface for model operations
- `bool loadModel(common_params& params)` - Load and initialize model
- `bool initSampling()` - Initialize sampling context
- `void rewind()` - Reset context state
- `void beginCompletion()` - Start generation process
- `void loadPrompt()` - Load prompt into context
- `completion_token_output doCompletion()` - Generate next token
- `void endCompletion()` - End generation process

### Multimodal Methods

- `bool initMultimodal(const std::string& mmproj_path, bool use_gpu)` - Initialize vision
- `bool isMultimodalEnabled()` - Check multimodal status
- `bool isMultimodalSupportVision()` - Check vision support
- `void loadPrompt(const std::vector<std::string>& media_paths)` - Load with media
- `void releaseMultimodal()` - Clean up multimodal resources

### Text-to-Speech Methods

- `bool initVocoder(const std::string& vocoder_path)` - Initialize TTS
- `bool isVocoderEnabled()` - Check TTS status
- `tts_type getTTSType()` - Get TTS model type
- `std::string getFormattedAudioCompletion(const std::string& speaker, const std::string& text)` - Format TTS prompt
- `std::vector<llama_token> getAudioCompletionGuideTokens(const std::string& text)` - Get guide tokens
- `std::vector<float> decodeAudioTokens(const std::vector<llama_token>& tokens)` - Decode to audio
- `void releaseVocoder()` - Clean up TTS resources

### Utility Methods

- `std::vector<float> getEmbedding(common_params& params)` - Generate embeddings
- `cactus_tokenize_result tokenize(const std::string& text, const std::vector<std::string>& media)` - Tokenize text
- `std::string getFormattedChat(const std::string& messages, const std::string& template)` - Format chat
- `int applyLoraAdapters(std::vector<common_adapter_lora_info> lora)` - Apply LoRA
- `void removeLoraAdapters()` - Remove LoRA adapters
- `std::string bench(int pp, int tg, int pl, int nr)` - Performance benchmark

## Best Practices

### Memory Management

```cpp
class ModelManager {
private:
    std::unique_ptr<cactus::cactus_context> context;
    
public:
    ModelManager() : context(std::make_unique<cactus::cactus_context>()) {}
    
    ~ModelManager() {
        cleanup();
    }
    
    void cleanup() {
        if (context) {
            context->releaseMultimodal();
            context->releaseVocoder();
            context.reset();
        }
    }
    
    cactus::cactus_context* getContext() { return context.get(); }
};
```

### Error Handling

```cpp
enum class ModelError {
    LOAD_FAILED,
    SAMPLING_FAILED,
    GENERATION_FAILED,
    MULTIMODAL_FAILED,
    TTS_FAILED
};

class SafeModelWrapper {
private:
    cactus::cactus_context context;
    
public:
    std::variant<std::string, ModelError> generateSafely(const std::string& prompt) {
        try {
            context.params.prompt = prompt;
            
            if (!context.initSampling()) {
                return ModelError::SAMPLING_FAILED;
            }
            
            context.rewind();
            context.beginCompletion();
            context.loadPrompt();
            
            while (context.has_next_token && !context.is_interrupted) {
                auto token_output = context.doCompletion();
                if (token_output.tok == -1) break;
            }
            
            return context.generated_text;
            
        } catch (const std::exception& e) {
            std::cerr << "Generation error: " << e.what() << std::endl;
            return ModelError::GENERATION_FAILED;
        }
    }
};
```

### Thread Safety

```cpp
#include <mutex>
#include <queue>
#include <condition_variable>

class ThreadSafeModelPool {
private:
    std::queue<std::unique_ptr<cactus::cactus_context>> available_contexts;
    std::mutex context_mutex;
    std::condition_variable context_cv;
    
public:
    void addContext(std::unique_ptr<cactus::cactus_context> context) {
        std::lock_guard<std::mutex> lock(context_mutex);
        available_contexts.push(std::move(context));
        context_cv.notify_one();
    }
    
    std::unique_ptr<cactus::cactus_context> getContext() {
        std::unique_lock<std::mutex> lock(context_mutex);
        context_cv.wait(lock, [this] { return !available_contexts.empty(); });
        
        auto context = std::move(available_contexts.front());
        available_contexts.pop();
        return context;
    }
    
    void returnContext(std::unique_ptr<cactus::cactus_context> context) {
        addContext(std::move(context));
    }
};
```

## Troubleshooting

### Common Issues

**Model Loading Fails**
```cpp
bool validateModel(const std::string& model_path) {
    std::ifstream file(model_path);
    if (!file.good()) {
        std::cerr << "Model file not found: " << model_path << std::endl;
        return false;
    }
    
    // Check file size
    file.seekg(0, std::ios::end);
    auto size = file.tellg();
    if (size < 1024 * 1024) { // Less than 1MB
        std::cerr << "Model file appears corrupted (too small)" << std::endl;
        return false;
    }
    
    return true;
}
```

**Memory Issues**
```cpp
void configureForLowMemory(common_params& params) {
    params.n_ctx = 1024;        // Reduce context size
    params.n_batch = 128;       // Smaller batch size
    params.n_gpu_layers = 0;    // Use CPU only
    params.use_mmap = true;     // Enable memory mapping
    params.use_mlock = false;   // Don't lock memory
}
```

**GPU Issues**
```cpp
bool testGPU(cactus::cactus_context& context) {
    // Try loading with GPU
    common_params params;
    params.model.path = "test-model.gguf";
    params.n_gpu_layers = 1;
    
    if (!context.loadModel(params)) {
        std::cout << "GPU acceleration not available, falling back to CPU" << std::endl;
        params.n_gpu_layers = 0;
        return context.loadModel(params);
    }
    
    return true;
}
```

### Performance Tips

1. **Use appropriate context sizes** - Larger contexts use exponentially more memory
2. **Optimize batch sizes** - Balance between throughput and memory usage  
3. **Enable GPU acceleration** - Use `n_gpu_layers = 99` when available
4. **Use memory mapping** - Enable `use_mmap = true` for large models
5. **Thread optimization** - Set `n_threads` to your CPU core count
6. **Model quantization** - Use Q4_0 or Q8_0 models for better performance
7. **Profile your application** - Use the built-in benchmarking functions

This documentation covers the essential usage patterns for Cactus C++. For more examples, check the [example applications](../examples/cpp/) in the repository.
